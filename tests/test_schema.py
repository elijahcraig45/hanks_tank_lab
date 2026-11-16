from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("google.cloud.bigquery")

from mllab import schema as S


def field(name, typ, mode="NULLABLE"):
    return SimpleNamespace(name=name, field_type=typ, mode=mode)


TABLE = SimpleNamespace(
    schema=[field("game_pk", "INT64"), field("game_date", "DATE"), field("home_win", "BOOL"), field("elo", "FLOAT64")],
    num_rows=10, num_bytes=100, modified=datetime(2026, 9, 1, tzinfo=timezone.utc),
    time_partitioning=None, clustering_fields=None)


class Client:
    """Fake BigQuery client: metadata from TABLE, aggregate query answered from `profile`."""

    def __init__(self, profile=None, table=TABLE):
        self.profile, self.table, self.sql = profile, table, []

    def get_table(self, name):
        return SimpleNamespace(**{**self.table.__dict__, "num_bytes": 10})

    def query(self, sql, job_config=None):
        if job_config.dry_run:
            return SimpleNamespace(total_bytes_processed=10)
        self.sql.append(sql)
        df = pd.DataFrame([self.profile])
        return SimpleNamespace(result=lambda: SimpleNamespace(total_rows=1, schema=[0] * len(df.columns)),
                               destination="t", to_dataframe=lambda **kw: df)


def spec(**kw):
    return S.TableSpec(table="mlb_2026_season.games", columns=[S.ColumnSpec("game_pk", "INTEGER")], **kw)


def test_types_are_normalised():
    assert S.norm_type("int64") == "INTEGER" and S.norm_type("BOOL") == "BOOLEAN" and S.norm_type("FLOAT64") == "FLOAT"


def test_get_schema_and_infer_roundtrip(tmp_path):
    c = Client()
    info = S.get_schema("mlb_2026_season.games", client=c)
    assert [x["type"] for x in info["columns"]] == ["INTEGER", "DATE", "BOOLEAN", "FLOAT"]
    inferred = S.infer_spec("mlb_2026_season.games", client=c)
    p = tmp_path / "s.yaml"; p.write_text(S.dump_spec(inferred))
    assert S.load_spec(p).columns[3].type == "FLOAT"  # infer -> file -> load
    assert S.check_schema(info, inferred) == []


def test_missing_column_and_wrong_type():
    info = S.get_schema("d.t", client=Client())
    sp = S.TableSpec(table="d.t", columns=[S.ColumnSpec("nope", "STRING"), S.ColumnSpec("elo", "INTEGER")])
    f = {(x.check, x.level) for x in S.check_schema(info, sp)}
    assert f == {("column", "error"), ("type", "error")}


def test_allow_extra_false_warns_only():
    info = S.get_schema("d.t", client=Client())
    out = S.check_schema(info, S.TableSpec(table="d.t", columns=[S.ColumnSpec("game_pk")], allow_extra=False))
    assert [f.level for f in out] == ["warn"]


def test_diff_schemas():
    a = {"columns": [{"name": "x", "type": "INTEGER"}, {"name": "y", "type": "STRING"}]}
    b = {"columns": [{"name": "x", "type": "FLOAT"}, {"name": "z", "type": "STRING"}]}
    assert S.diff_schemas(a, b) == {"only_in_a": ["y"], "only_in_b": ["z"], "type_changed": {"x": ("INTEGER", "FLOAT")}}


NOW = datetime(2026, 9, 29, tzinfo=timezone.utc)


def test_data_checks_catch_each_problem():
    sp = S.TableSpec(table="d.t", key=["game_pk"], date_column="game_date", max_staleness_days=2, min_rows=100,
                     columns=[S.ColumnSpec("elo", nullable=False), S.ColumnSpec("home_win", max_null_rate=0.01),
                              S.ColumnSpec("game_pk", min=1)])
    row = {"n": 50, "dup_rows": 3, "null__elo": 2, "null__home_win": 5, "min__game_pk": 0,
           "max_date": "2026-09-20", "min_date": "2026-03-27"}
    checks = sorted(f.check for f in S.check_data(row, sp, NOW) if f.level == "error")
    assert checks == ["duplicates", "freshness", "null_rate", "nulls", "range", "rows"]


def test_freshness_is_only_an_error_in_its_season_months():
    sp = S.TableSpec(table="d.t", date_column="game_date", max_staleness_days=4, freshness_months=[3, 4, 5, 6, 7, 8, 9, 10])
    row = {"n": 50, "max_date": "2026-10-28", "min_date": "2026-03-27"}
    november = datetime(2026, 11, 3, tzinfo=timezone.utc)
    found = [f for f in S.check_data(row, sp, november) if f.check == "freshness"]
    assert [f.level for f in found] == ["info"] and "not checked in month 11" in found[0].message
    october = datetime(2026, 10, 9, tzinfo=timezone.utc)
    assert [f.level for f in S.check_data({**row, "max_date": "2026-10-01"}, sp, october) if f.check == "freshness"] == ["error"]


def test_freshness_months_are_validated():
    with pytest.raises(S.SpecError):
        S.validate_spec(S.TableSpec(table="d.t", freshness_months=[0, 13]))
    with pytest.raises(S.SpecError):
        S.validate_spec(S.TableSpec(table="d.t", freshness_months=[]))


def test_clean_data_passes():
    sp = S.TableSpec(table="d.t", key=["game_pk"], date_column="game_date", max_staleness_days=2,
                     columns=[S.ColumnSpec("elo", nullable=False)])
    row = {"n": 50, "dup_rows": 0, "null__elo": 0, "max_date": "2026-09-28", "min_date": "2026-03-27"}
    assert not [f for f in S.check_data(row, sp, NOW) if f.level == "error"]


def test_profile_sql_touches_only_ruled_columns():
    sp = S.TableSpec(table="mlb_2026_season.games", key=["game_pk", "game_date"], date_column="game_date",
                     columns=[S.ColumnSpec("elo", max_null_rate=0.1), S.ColumnSpec("untouched", "STRING")])
    sql = S.profile_sql(sp, "hankstank")
    assert "`hankstank.mlb_2026_season.games`" in sql and "null__elo" in sql and "untouched" not in sql
    assert "COUNT(DISTINCT TO_JSON_STRING(STRUCT(`game_pk`, `game_date`)))" in sql


def test_schema_only_spec_runs_no_data_query(tmp_path):
    c = Client()
    r = S.run_check(spec(), client=c, bq_kwargs={"cache_dir": tmp_path})
    assert r.ok and c.sql == []  # metadata is free; no query was run


def test_run_check_end_to_end_reports_failure(tmp_path):
    c = Client(profile={"n": 5, "dup_rows": 2})
    r = S.run_check(spec(key=["game_pk"]), client=c, bq_kwargs={"cache_dir": tmp_path})
    assert not r.ok and "duplicate" in r.render()


def test_missing_columns_skip_data_query(tmp_path):
    c = Client()
    sp = S.TableSpec(table="d.t", key=["game_pk"], columns=[S.ColumnSpec("ghost")])
    r = S.run_check(sp, client=c, bq_kwargs={"cache_dir": tmp_path})
    assert not r.ok and c.sql == [] and any("skipped" in f.message for f in r.findings)


@pytest.mark.parametrize("bad", ["x; DROP TABLE t", "a b", "`x`", "1abc", "x--", ""])
def test_hostile_column_names_rejected(bad):
    with pytest.raises(S.SpecError):
        S.validate_spec(S.TableSpec(table="d.t", columns=[S.ColumnSpec(bad)]))
    with pytest.raises(S.SpecError):
        S.validate_spec(S.TableSpec(table="d.t", key=[bad]))


@pytest.mark.parametrize("bad", ["justatable", "a.b.c.d", "d.t; select 1", "d.`t`", "../x"])
def test_hostile_table_refs_rejected(bad):
    with pytest.raises(S.SpecError):
        S.validate_spec(S.TableSpec(table=bad))


def test_load_spec_rejects_unknown_keys(tmp_path):
    p = tmp_path / "s.yaml"; p.write_text("table: d.t\nbogus: 1\n")
    with pytest.raises(S.SpecError):
        S.load_spec(p)
