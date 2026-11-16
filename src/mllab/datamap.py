"""The data map: every table the pipeline reads and writes, per sport and stage, and whether it is healthy.

A table is declared once in `catalog/<sport>.yaml`: which stage it belongs to (raw -> features -> predictions -> sims -> rankings -> site), which job writes it, how
often it should change and what that depends on. This module checks the declaration against BigQuery's own metadata, which is free to read (`__TABLES__`: row count,
size, last-modified time), and against what the pipeline's jobs report. It answers three questions the job list cannot:

    is the data fresh for what this table is for?          (age against its own cadence, only while the sport is in season)
    did the job that should have written it actually do so?  (a live job finished OK but its table did not change)
    is anything the pipeline expects missing?               (declared, absent in BigQuery)

Nothing here writes. It reads metadata, so it costs nothing and cannot break anything; row-level checks are the table specs (`mllab bq check`)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

STAGES = ["source", "raw", "features", "predictions", "shadows", "sims", "rankings", "site"]
KINDS = {"daily", "weekly", "per_game", "static"}
HOW = {"append", "upsert", "replace_partition", "truncate", "ddl", "external", None}


class CatalogError(Exception):
    pass


@dataclass(frozen=True)
class Cadence:
    kind: str                        # daily | weekly | per_game | static
    max_age_hours: float | None = None   # older than this while in season = late; twice this = stale
    months: tuple[int, ...] | None = None  # the months it is expected to change in; outside them it is idle, never stale
    note: str = ""


@dataclass(frozen=True)
class TableDef:
    sport: str
    id: str                          # dataset.table
    stage: str
    label: str
    written_by: tuple[str, ...] = ()
    read_by: tuple[str, ...] = ()    # models / site routes / jobs that depend on it
    key: tuple[str, ...] = ()
    date_column: str | None = None
    cadence: Cadence = field(default_factory=lambda: Cadence("static"))
    how: str | None = None           # how a write lands, which tells you what is safe to re-run
    optional: bool = False           # may not exist yet (a shadow that has not run)
    writes_every_run: bool = True    # False where a run legitimately writes nothing (no games that day)
    notes: str = ""


# ---------------------------------------------------------------------------------------------------------------------------------- loading
def _cadence(raw: dict | None, where: str) -> Cadence:
    raw = raw or {"kind": "static"}
    if raw.get("kind") not in KINDS:
        raise CatalogError(f"{where}: cadence.kind must be one of {sorted(KINDS)}")
    months = tuple(int(m) for m in raw["months"]) if raw.get("months") else None
    if months and not all(1 <= m <= 12 for m in months):
        raise CatalogError(f"{where}: months must be 1..12")
    if raw["kind"] != "static" and not raw.get("max_age_hours"):
        raise CatalogError(f"{where}: a {raw['kind']} table needs max_age_hours")
    return Cadence(raw["kind"], float(raw["max_age_hours"]) if raw.get("max_age_hours") else None, months, str(raw.get("note", "")))


def parse(raw: dict, source: str = "catalog") -> list[TableDef]:
    sport = raw.get("sport")
    if sport not in ("mlb", "nfl", "cfb"):
        raise CatalogError(f"{source}: sport must be mlb, nfl or cfb")
    out, seen = [], set()
    for i, t in enumerate(raw.get("tables") or []):
        where = f"{source}: table #{i + 1} ({t.get('id', '?')})"
        tid = t.get("id", "")
        if tid.count(".") != 1 or not all(tid.split(".")):
            raise CatalogError(f"{where}: id must be dataset.table")
        if tid in seen:
            raise CatalogError(f"{where}: declared twice")
        seen.add(tid)
        if t.get("stage") not in STAGES:
            raise CatalogError(f"{where}: stage must be one of {STAGES}")
        if t.get("how") not in HOW:
            raise CatalogError(f"{where}: how must be one of {sorted(h for h in HOW if h)}")
        out.append(TableDef(sport=sport, id=tid, stage=t["stage"], label=t.get("label", tid.split(".")[1]), written_by=tuple(t.get("written_by") or ()),
                            read_by=tuple(t.get("read_by") or ()), key=tuple(t.get("key") or ()), date_column=t.get("date_column"),
                            cadence=_cadence(t.get("cadence"), where), how=t.get("how"), optional=bool(t.get("optional", False)),
                            writes_every_run=bool(t.get("writes_every_run", True)), notes=str(t.get("notes", ""))))
    return out


def load(directory: str | Path) -> list[TableDef]:
    d = Path(directory)
    files = sorted(d.glob("*.yaml"))
    if not files:
        raise CatalogError(f"no catalog files in {d}")
    out: list[TableDef] = []
    for f in files:
        out += parse(yaml.safe_load(f.read_text()) or {}, f.name)
    ids = [t.id for t in out]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise CatalogError(f"tables declared in more than one sport: {sorted(dup)}")
    return out


def find_catalog() -> Path:
    import os
    for p in (os.environ.get("MLLAB_CATALOG"), "/opt/mllab/src/catalog", Path(__file__).resolve().parents[2] / "catalog"):
        if p and Path(p).is_dir():
            return Path(p)
    raise CatalogError("no catalog directory (set MLLAB_CATALOG)")


# ------------------------------------------------------------------------------------------------------------------------------ evaluation
SEVERITY = {"ok": 0, "idle": 0, "unknown": 1, "late": 2, "unchanged": 2, "missing": 3, "stale": 3}


def _age_hours(then: datetime, now: datetime) -> float:
    return max(0.0, (now - then).total_seconds() / 3600.0)


def table_state(t: TableDef, fact: dict | None, now: datetime, run_effects: list[dict] | None = None, unreadable: str | None = None) -> dict:
    """{status, age_hours, reasons[]} for one table. `fact` is {rows, bytes, modified} from BigQuery, or None if the table does not exist. `unreadable` is why its whole
    dataset could not be read: then nothing is known, which is not the same as the table being missing."""
    reasons: list[str] = []
    if unreadable:
        return {"status": "unknown", "age_hours": None, "reasons": [f"cannot read dataset {t.id.split('.')[0]}: {unreadable}"]}
    if fact is None:
        status = "ok" if t.optional else "missing"
        reasons.append("not created yet (optional: its first run creates it)" if t.optional else "declared in the catalog, but BigQuery has no such table")
        return {"status": "idle" if t.optional else status, "age_hours": None, "reasons": reasons}
    age = _age_hours(fact["modified"], now)
    status = "ok"
    active = t.cadence.kind == "static" or t.cadence.months is None or now.month in t.cadence.months
    if t.cadence.kind != "static":
        limit = t.cadence.max_age_hours
        if not active:
            status = "idle"
            reasons.append(f"out of season: expected to change in months {', '.join(map(str, t.cadence.months))}")
        elif age > 2 * limit:
            status = "stale"
            reasons.append(f"last changed {age:.0f} h ago; expected every {limit:.0f} h or less")
        elif age > limit:
            status = "late"
            reasons.append(f"last changed {age:.0f} h ago; expected every {limit:.0f} h or less")
    for e in run_effects or []:
        if e["unchanged"] and t.writes_every_run and active:  # out of season a run that writes nothing is expected (MLB games in November)
            reasons.append(f"{e['job']} finished OK at {e['finished'][:16]}Z but this table has not changed since {fact['modified'].strftime('%m-%d %H:%M')}Z")
            # >= so it wins over "late": a run that finished OK but wrote nothing is the more useful (and the reported) fact
            if SEVERITY["unchanged"] >= SEVERITY[status]:
                status = "unchanged"
    return {"status": status, "age_hours": round(age, 1), "reasons": reasons}


def run_effects(t: TableDef, runs: dict[str, dict], modes: dict[str, str], fact: dict | None) -> list[dict]:
    """For each LIVE job that writes `t`: did its latest successful run leave the table changed? Only a live job can be judged: a dry run writes nothing by design,
    and a job still run by Cloud Functions is not in `runs` at all."""
    out = []
    for job in t.written_by:
        r = runs.get(job)
        if not r or modes.get(job) != "live" or r.get("mode") != "live" or not r.get("ok") or fact is None or not r.get("started"):
            continue
        started = datetime.fromisoformat(r["started"])
        out.append({"job": job, "finished": r.get("finished") or r["started"], "unchanged": fact["modified"] < started})
    return out


def build(catalog: list[TableDef], facts: dict[str, dict], now: datetime, runs: dict[str, dict] | None = None, modes: dict[str, str] | None = None,
          errors: dict[str, str] | None = None) -> dict:
    """The whole map, grouped by sport and stage, with an overall status per sport. `errors` maps a dataset that could not be read to why."""
    runs, modes, errors = runs or {}, modes or {}, errors or {}
    sports: dict[str, dict] = {}
    for t in catalog:
        fact = facts.get(t.id)
        eff = run_effects(t, runs, modes, fact)
        st = table_state(t, fact, now, eff, errors.get(t.id.split(".")[0]))
        row = {"id": t.id, "label": t.label, "stage": t.stage, "written_by": list(t.written_by), "read_by": list(t.read_by), "key": list(t.key), "date_column": t.date_column,
               "cadence": {"kind": t.cadence.kind, "max_age_hours": t.cadence.max_age_hours, "note": t.cadence.note}, "how": t.how, "optional": t.optional, "notes": t.notes,
               "rows": fact["rows"] if fact else None, "bytes": fact["bytes"] if fact else None, "modified": fact["modified"].isoformat() if fact else None, **st}
        s = sports.setdefault(t.sport, {"sport": t.sport, "tables": [], "status": "ok", "counts": {}})
        s["tables"].append(row)
        s["counts"][st["status"]] = s["counts"].get(st["status"], 0) + 1
        if SEVERITY[st["status"]] > SEVERITY[s["status"]]:
            s["status"] = st["status"]
    for s in sports.values():
        s["tables"].sort(key=lambda r: (STAGES.index(r["stage"]), r["id"]))
    return {"generated_at": now.isoformat(timespec="seconds"), "sports": sports, "stages": STAGES, "unreadable": errors}


# ------------------------------------------------------------------------------------------------------------------------------ collection
def collect_facts(runner, datasets: set[str], project: str = "hankstank") -> tuple[dict[str, dict], dict[str, str]]:
    """({dataset.table: {rows, bytes, modified}}, {dataset: why it could not be read}) from each dataset's `__TABLES__` (metadata: free, one tiny query per dataset).
    A dataset that cannot be read is reported with its reason and the others are still read: its tables are 'unknown', never guessed to be missing."""
    facts: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for ds in sorted(datasets):
        try:
            rows = runner.query(f"SELECT table_id, row_count, size_bytes, last_modified_time FROM `{project}.{ds}.__TABLES__`")
        except Exception as e:  # noqa: BLE001
            errors[ds] = " ".join(str(e).split())[:160]
            continue
        for r in rows:
            ms = float(r["last_modified_time"])
            facts[f"{ds}.{r['table_id']}"] = {"rows": int(r["row_count"] or 0), "bytes": int(r["size_bytes"] or 0),
                                              "modified": datetime.fromtimestamp(ms / 1000.0 if ms > 1e11 else ms, timezone.utc)}
    return facts, errors


class Cache:
    """Metadata is cheap but not free of latency; the page refreshes every 15 s, so keep the last answer for a minute."""

    def __init__(self, ttl: float = 60.0, clock=time.monotonic):
        self.ttl, self.clock, self._at, self._val = ttl, clock, None, None

    def get(self, fn):
        if self._at is not None and self.clock() - self._at < self.ttl:
            return self._val
        self._val, self._at = fn(), self.clock()
        return self._val


# ------------------------------------------------------------------------------------------------------------------------------------ cli
def render(m: dict, only_problems: bool = False) -> str:
    lines = []
    for sp in m["sports"].values():
        lines.append(f"\n{sp['sport'].upper()}  overall: {sp['status']}   {sp['counts']}")
        for r in sp["tables"]:
            if only_problems and r["status"] in ("ok", "idle"):
                continue
            age = "-" if r["age_hours"] is None else (f"{r['age_hours']:.0f}h" if r["age_hours"] < 72 else f"{r['age_hours'] / 24:.0f}d")
            lines.append(f"  {r['status']:9s} {r['id']:48s} {str(r['rows'] if r['rows'] is not None else '-'):>9s} rows  last change {age:>5s} ago  {'; '.join(r['reasons'])[:110]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """python -m mllab.datamap [--problems] [--json]   Reads table metadata only (free)."""
    import argparse
    import json

    from .health_collect import BqCliRunner

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--problems", action="store_true", help="only tables that are not fresh or idle")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    cat = load(find_catalog())
    facts, errors = collect_facts(BqCliRunner(), {t.id.split(".")[0] for t in cat})
    m = build(cat, facts, datetime.now(timezone.utc), errors=errors)
    print(json.dumps(m, indent=1) if a.json else render(m, a.problems))
    return 1 if any(s["status"] in ("stale", "missing") for s in m["sports"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
