"""BigQuery table/field checker.

  describe  -> columns, types, rows, size, partitioning, freshness (metadata only, free)
  infer     -> bootstrap a spec from a table as it is today
  diff      -> schema difference between two tables (e.g. 2026 vs historical)
  check     -> validate a table against a spec: columns, types, duplicates on a key, null rates,
               value ranges, row count and staleness. Data checks are ONE aggregate query that
               goes through bq()'s cost guard.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

_TYPE_ALIASES = {"INT64": "INTEGER", "INT": "INTEGER", "SMALLINT": "INTEGER", "BIGINT": "INTEGER",
                 "FLOAT64": "FLOAT", "BOOL": "BOOLEAN", "DECIMAL": "NUMERIC", "BIGDECIMAL": "BIGNUMERIC"}
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TABLE = re.compile(r"^([A-Za-z0-9_-]+[.:])?[A-Za-z0-9_]+\.[A-Za-z0-9_]+$")


class SpecError(Exception):
    pass


def norm_type(t: str | None) -> str | None:
    if t is None:
        return None
    t = t.upper().strip()
    return _TYPE_ALIASES.get(t, t)


def _ident(name: str) -> str:
    if not _IDENT.match(name):
        raise SpecError(f"invalid column name {name!r}")  # names go into SQL, so never accept anything odd
    return name


def _table(ref: str) -> str:
    if not _TABLE.match(ref):
        raise SpecError(f"invalid table reference {ref!r}; use dataset.table or project.dataset.table")
    return ref


@dataclass
class ColumnSpec:
    name: str
    type: str | None = None
    nullable: bool | None = None  # False = must have zero nulls
    max_null_rate: float | None = None
    min: float | None = None
    max: float | None = None


@dataclass
class TableSpec:
    table: str
    columns: list[ColumnSpec] = field(default_factory=list)
    key: list[str] = field(default_factory=list)
    date_column: str | None = None
    max_staleness_days: float | None = None
    freshness_months: list[int] | None = None  # months the freshness rule applies in (the season); outside them it is only reported
    min_rows: int | None = None
    allow_extra: bool = True


@dataclass
class Finding:
    level: str  # error | warn | info
    check: str
    message: str


@dataclass
class Report:
    table: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(f.level == "error" for f in self.findings)

    def render(self) -> str:
        lines = [f"{self.table}: {'OK' if self.ok else 'FAILED'}"]
        lines += [f"  [{f.level:5}] {f.check}: {f.message}" for f in self.findings]
        return "\n".join(lines)


def load_spec(path: str | Path) -> TableSpec:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if "table" not in raw:
        raise SpecError("spec needs a 'table'")
    cols = []
    for c in raw.get("columns", []):
        c = {"name": c} if isinstance(c, str) else dict(c)
        try:
            cols.append(ColumnSpec(**c))
        except TypeError as e:
            raise SpecError(f"bad column entry {c}: {e}") from e
    try:
        spec = TableSpec(**{**raw, "columns": cols})
    except TypeError as e:
        raise SpecError(f"bad spec: {e}") from e
    validate_spec(spec)
    return spec


def validate_spec(spec: TableSpec) -> None:
    _table(spec.table)
    for c in spec.columns:
        _ident(c.name)
        if c.max_null_rate is not None and not 0 <= c.max_null_rate <= 1:
            raise SpecError(f"{c.name}: max_null_rate must be 0..1")
    for k in spec.key:
        _ident(k)
    if spec.date_column:
        _ident(spec.date_column)
    if spec.freshness_months is not None and (not spec.freshness_months or not all(isinstance(m, int) and 1 <= m <= 12 for m in spec.freshness_months)):
        raise SpecError("freshness_months must be a non-empty list of months 1..12")


def dump_spec(spec: TableSpec) -> str:
    d = {"table": spec.table, "key": spec.key, "date_column": spec.date_column,
         "max_staleness_days": spec.max_staleness_days, "freshness_months": spec.freshness_months, "min_rows": spec.min_rows, "allow_extra": spec.allow_extra,
         "columns": [{k: v for k, v in c.__dict__.items() if v is not None} for c in spec.columns]}
    return yaml.safe_dump({k: v for k, v in d.items() if v not in (None, [])}, sort_keys=False)


# --- metadata (free) -----------------------------------------------------------------------------

def _client(project: str, client=None):
    if client is not None:
        return client
    from google.cloud import bigquery

    return bigquery.Client(project=project)


def get_schema(table: str, *, project: str = "hankstank", client=None) -> dict:
    client = _client(project, client)
    t = client.get_table(_table(table) if table.count(".") == 2 else f"{project}.{_table(table)}")
    return {
        "columns": [{"name": f.name, "type": norm_type(f.field_type), "mode": f.mode} for f in t.schema],
        "num_rows": t.num_rows, "num_bytes": t.num_bytes, "modified": t.modified,
        "partitioning": getattr(getattr(t, "time_partitioning", None), "field", None),
        "clustering": list(t.clustering_fields or []),
    }


def infer_spec(table: str, *, project: str = "hankstank", client=None) -> TableSpec:
    info = get_schema(table, project=project, client=client)
    return TableSpec(table=table, columns=[ColumnSpec(c["name"], c["type"]) for c in info["columns"]])


def diff_schemas(a: dict, b: dict) -> dict:
    ca = {c["name"]: c["type"] for c in a["columns"]}
    cb = {c["name"]: c["type"] for c in b["columns"]}
    return {
        "only_in_a": sorted(set(ca) - set(cb)),
        "only_in_b": sorted(set(cb) - set(ca)),
        "type_changed": {n: (ca[n], cb[n]) for n in sorted(set(ca) & set(cb)) if ca[n] != cb[n]},
    }


# --- checks ----------------------------------------------------------------------------------------

def check_schema(info: dict, spec: TableSpec) -> list[Finding]:
    actual = {c["name"]: c["type"] for c in info["columns"]}
    out = []
    for c in spec.columns:
        if c.name not in actual:
            out.append(Finding("error", "column", f"missing column {c.name}"))
        elif c.type and norm_type(c.type) != actual[c.name]:
            out.append(Finding("error", "type", f"{c.name}: expected {norm_type(c.type)}, found {actual[c.name]}"))
    if not spec.allow_extra:
        extra = sorted(set(actual) - {c.name for c in spec.columns})
        if extra:
            out.append(Finding("warn", "extra", f"{len(extra)} unexpected column(s): {', '.join(extra[:10])}"))
    return out


def profile_sql(spec: TableSpec, project: str) -> str:
    """One aggregate query. Only columns the spec has data rules for are touched."""
    full = spec.table if spec.table.count(".") == 2 else f"{project}.{spec.table}"
    parts = ["COUNT(*) AS n"]
    for c in spec.columns:
        if c.nullable is False or c.max_null_rate is not None:
            parts.append(f"COUNTIF(`{c.name}` IS NULL) AS null__{c.name}")
        if c.min is not None:
            parts.append(f"MIN(`{c.name}`) AS min__{c.name}")
        if c.max is not None:
            parts.append(f"MAX(`{c.name}`) AS max__{c.name}")
    if spec.key:
        keys = ", ".join(f"`{k}`" for k in spec.key)
        parts.append(f"COUNT(*) - COUNT(DISTINCT TO_JSON_STRING(STRUCT({keys}))) AS dup_rows")
    if spec.date_column:
        parts.append(f"MAX(`{spec.date_column}`) AS max_date")
        parts.append(f"MIN(`{spec.date_column}`) AS min_date")
    return f"SELECT {', '.join(parts)} FROM `{full}`"


def needs_data_query(spec: TableSpec) -> bool:
    return bool(spec.key or spec.date_column or spec.min_rows or spec.max_staleness_days
                or any(c.nullable is False or c.max_null_rate is not None or c.min is not None or c.max is not None
                       for c in spec.columns))


def check_data(row: dict, spec: TableSpec, now: datetime | None = None) -> list[Finding]:
    now = now or datetime.now(timezone.utc)
    out = []
    n = int(row["n"])
    if spec.min_rows is not None and n < spec.min_rows:
        out.append(Finding("error", "rows", f"{n} rows, expected at least {spec.min_rows}"))
    if spec.key and int(row.get("dup_rows", 0)) > 0:
        out.append(Finding("error", "duplicates", f"{int(row['dup_rows'])} duplicate row(s) on key ({', '.join(spec.key)})"))
    for c in spec.columns:
        nulls = row.get(f"null__{c.name}")
        if nulls is not None and n:
            rate = int(nulls) / n
            if c.nullable is False and nulls:
                out.append(Finding("error", "nulls", f"{c.name}: {int(nulls)} null(s) but nullable: false"))
            elif c.max_null_rate is not None and rate > c.max_null_rate:
                out.append(Finding("error", "null_rate", f"{c.name}: {rate:.1%} null, limit {c.max_null_rate:.1%}"))
        lo, hi = row.get(f"min__{c.name}"), row.get(f"max__{c.name}")
        if c.min is not None and lo is not None and lo < c.min:
            out.append(Finding("error", "range", f"{c.name}: min {lo} below {c.min}"))
        if c.max is not None and hi is not None and hi > c.max:
            out.append(Finding("error", "range", f"{c.name}: max {hi} above {c.max}"))
    if spec.date_column and row.get("max_date") is not None:
        import pandas as pd

        newest = pd.Timestamp(row["max_date"])
        newest = newest.tz_localize("UTC") if newest.tzinfo is None else newest
        age = (pd.Timestamp(now) - newest).total_seconds() / 86400
        out.append(Finding("info", "range", f"{spec.date_column}: {row.get('min_date')} .. {row['max_date']}"))
        if spec.max_staleness_days is not None and age > spec.max_staleness_days:
            month = pd.Timestamp(now).month
            if spec.freshness_months and month not in spec.freshness_months:  # off-season: no new rows is expected, not an incident
                out.append(Finding("info", "freshness", f"newest {spec.date_column} is {age:.1f} days old; not checked in month {month} "
                                   f"(season months {', '.join(map(str, spec.freshness_months))})"))
            else:
                out.append(Finding("error", "freshness", f"newest {spec.date_column} is {age:.1f} days old, limit {spec.max_staleness_days}"))
    return out


def run_check(spec: TableSpec, *, project: str = "hankstank", client=None, bq_kwargs: dict | None = None) -> Report:
    client = _client(project, client)
    report = Report(spec.table)
    info = get_schema(spec.table, project=project, client=client)
    report.findings += check_schema(info, spec)
    missing = {f.message.split()[-1] for f in report.findings if f.check == "column"}
    if missing:  # do not build SQL against columns that are not there
        report.findings.append(Finding("warn", "data", "data checks skipped: fix missing columns first"))
        return report
    if needs_data_query(spec):
        from .bq import bq

        df = bq(profile_sql(spec, project), project=project, client=client, refresh=True, **(bq_kwargs or {}))
        report.findings += check_data(df.iloc[0].to_dict(), spec)
    return report
