"""The data map: declarations are validated, freshness is judged against each table's own cadence and only in season, and 'the job ran but the table did not
change' is only ever said about a job that is actually live here."""
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from mllab import datamap as D

NOW = datetime(2026, 10, 5, 14, 0, tzinfo=timezone.utc)


def t(**kw):
    base = dict(sport="mlb", id="ds.games", stage="raw", label="games", written_by=("daily",), cadence=D.Cadence("daily", 30.0, (3, 4, 5, 6, 7, 8, 9, 10, 11)))
    base.update(kw)
    return D.TableDef(**base)


def fact(hours_ago, rows=100):
    return {"rows": rows, "bytes": 1000, "modified": NOW - timedelta(hours=hours_ago)}


# ----------------------------------------------------------------------------------------------------------------------------- declarations
GOOD = {"sport": "mlb", "tables": [{"id": "ds.games", "stage": "raw", "cadence": {"kind": "daily", "max_age_hours": 30, "months": [3, 4]}, "written_by": ["daily"], "how": "upsert"}]}


def test_a_valid_declaration_parses():
    (x,) = D.parse(GOOD)
    assert x.id == "ds.games" and x.cadence.max_age_hours == 30.0 and x.cadence.months == (3, 4) and x.written_by == ("daily",) and x.label == "games"


@pytest.mark.parametrize("mutate,why", [
    (lambda g: g.update(sport="nba"), "sport"),
    (lambda g: g["tables"][0].update(id="nodot"), "dataset.table"),
    (lambda g: g["tables"][0].update(stage="warehouse"), "stage"),
    (lambda g: g["tables"][0].update(how="yolo"), "how"),
    (lambda g: g["tables"][0].update(cadence={"kind": "hourly", "max_age_hours": 1}), "cadence.kind"),
    (lambda g: g["tables"][0].update(cadence={"kind": "daily"}), "max_age_hours"),
    (lambda g: g["tables"][0].update(cadence={"kind": "daily", "max_age_hours": 5, "months": [13]}), "months"),
    (lambda g: g["tables"].append(dict(g["tables"][0])), "twice"),
])
def test_bad_declarations_are_refused_with_the_reason(mutate, why):
    import copy
    g = copy.deepcopy(GOOD)
    mutate(g)
    with pytest.raises(D.CatalogError, match=why):
        D.parse(g)


def test_a_table_may_not_belong_to_two_sports(tmp_path):
    for sport in ("mlb", "nfl"):
        (tmp_path / f"{sport}.yaml").write_text(yaml.safe_dump({"sport": sport, "tables": [{"id": "ds.shared", "stage": "raw"}]}))
    with pytest.raises(D.CatalogError, match="more than one sport"):
        D.load(tmp_path)
    with pytest.raises(D.CatalogError, match="no catalog files"):
        D.load(tmp_path / "empty-nonexistent")


def test_the_shipped_catalog_is_valid_and_every_table_is_in_a_known_stage():
    from pathlib import Path
    cat = D.load(Path(__file__).resolve().parents[1] / "catalog")
    assert cat and {x.sport for x in cat} == {"mlb", "nfl", "cfb"}
    assert all(x.stage in D.STAGES for x in cat)


# ----------------------------------------------------------------------------------------------------------------------------- freshness
@pytest.mark.parametrize("hours,status", [(2, "ok"), (30, "ok"), (31, "late"), (60, "late"), (61, "stale"), (500, "stale")])
def test_age_is_judged_against_the_tables_own_cadence(hours, status):
    assert D.table_state(t(), fact(hours), NOW)["status"] == status


def test_out_of_season_a_table_is_idle_not_stale():
    off = datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)
    f = {"rows": 1, "bytes": 1, "modified": off - timedelta(days=90)}
    s = D.table_state(t(), f, off)
    assert s["status"] == "idle" and "out of season" in s["reasons"][0]


def test_a_static_table_is_never_late():
    assert D.table_state(t(cadence=D.Cadence("static")), fact(100000), NOW)["status"] == "ok"


def test_a_missing_table_is_a_problem_unless_it_is_optional():
    assert D.table_state(t(), None, NOW)["status"] == "missing"
    s = D.table_state(t(optional=True), None, NOW)
    assert s["status"] == "idle" and "first run creates it" in s["reasons"][0]


# --------------------------------------------------------------------------------------------------------------------- did the job change it?
def run(ok=True, hours_ago=3, mode="live"):
    started = NOW - timedelta(hours=hours_ago)
    return {"job": "daily", "mode": mode, "ok": ok, "started": started.isoformat(), "finished": (started + timedelta(minutes=2)).isoformat()}


def test_a_live_job_that_finished_ok_but_left_its_table_untouched_is_flagged():
    f = fact(10)                                          # table last changed 10 h ago; the job started 3 h ago
    eff = D.run_effects(t(), {"daily": run(hours_ago=3)}, {"daily": "live"}, f)
    assert eff and eff[0]["unchanged"] is True
    s = D.table_state(t(), f, NOW, eff)
    assert s["status"] == "unchanged" and "finished OK" in " ".join(s["reasons"])


def test_a_table_that_is_late_and_untouched_by_a_live_run_reads_unchanged_not_late():
    f = fact(40)                                          # past its 30 h limit (late) AND the job that ran 3 h ago did not write it
    eff = D.run_effects(t(), {"daily": run(hours_ago=3)}, {"daily": "live"}, f)
    assert D.table_state(t(), f, NOW, eff)["status"] == "unchanged"   # "late" is not reported by the heartbeat; this must be


def test_out_of_season_a_run_that_writes_nothing_is_not_flagged():
    off = t(cadence=D.Cadence("daily", 30.0, ((NOW.month % 12) + 1,)))   # a season that excludes this month
    f = fact(500)
    eff = D.run_effects(off, {"daily": run(hours_ago=3)}, {"daily": "live"}, f)
    assert eff and eff[0]["unchanged"] is True
    assert D.table_state(off, f, NOW, eff)["status"] == "idle"


def test_a_table_the_job_did_change_is_fine():
    eff = D.run_effects(t(), {"daily": run(hours_ago=3)}, {"daily": "live"}, fact(2))
    assert eff[0]["unchanged"] is False and D.table_state(t(), fact(2), NOW, eff)["status"] == "ok"


@pytest.mark.parametrize("r,mode", [(run(mode="dry"), "dry"), (run(), "dry"), (run(), "off"), (run(ok=False), "live"), (None, "live")])
def test_only_a_live_successful_run_can_be_judged(r, mode):
    runs = {"daily": r} if r else {}
    assert D.run_effects(t(), runs, {"daily": mode}, fact(10)) == []     # a dry run writes nothing by design; a Cloud-Functions job is not reported here


def test_a_job_that_legitimately_writes_nothing_some_days_is_not_flagged():
    x = t(writes_every_run=False)
    eff = D.run_effects(x, {"daily": run()}, {"daily": "live"}, fact(10))
    assert D.table_state(x, fact(10), NOW, eff)["status"] == "ok"


# ----------------------------------------------------------------------------------------------------------------------------- the whole map
def test_build_groups_by_sport_and_stage_and_reports_the_worst_status():
    cat = [t(id="ds.games"), t(id="ds.feat", stage="features", label="features"), t(id="ds.late", stage="predictions", label="late"),
           D.TableDef(sport="nfl", id="n.pred", stage="predictions", label="p", cadence=D.Cadence("weekly", 200.0, (9, 10)))]
    facts = {"ds.games": fact(2), "ds.feat": fact(2), "ds.late": fact(40), "n.pred": fact(100)}
    m = D.build(cat, facts, NOW)
    mlb = m["sports"]["mlb"]
    assert mlb["status"] == "late" and mlb["counts"] == {"ok": 2, "late": 1}
    assert [r["stage"] for r in mlb["tables"]] == ["raw", "features", "predictions"]          # pipeline order, not alphabetical
    assert m["sports"]["nfl"]["status"] == "ok" and m["sports"]["nfl"]["tables"][0]["rows"] == 100


def test_collect_facts_reads_what_it_can_and_names_what_it_cannot_without_guessing():
    class R:
        def __init__(self): self.sql = []
        def query(self, sql, params=None):
            self.sql.append(sql)
            if "locked_ds" in sql:
                raise RuntimeError("403 Access   Denied: User does not have permission")
            return [{"table_id": "games", "row_count": "12", "size_bytes": "340", "last_modified_time": "1790971374000"}]
    r = R()
    f, errors = D.collect_facts(r, {"mlb_2026_season", "nfl_season", "locked_ds"})
    assert set(f) == {"mlb_2026_season.games", "nfl_season.games"} and f["nfl_season.games"]["rows"] == 12
    assert f["nfl_season.games"]["modified"].year == 2026 and all("__TABLES__" in q for q in r.sql)
    assert list(errors) == ["locked_ds"] and "Access Denied" in errors["locked_ds"]            # whitespace tidied, reason kept; the other datasets were still read


def test_a_table_in_an_unreadable_dataset_is_unknown_never_missing():
    s = D.table_state(t(id="locked_ds.games"), None, NOW, unreadable="403 Access Denied")
    assert s["status"] == "unknown" and "cannot read dataset locked_ds" in s["reasons"][0]
    m = D.build([t(id="locked_ds.games"), t(id="mlb_ds.games")], {"mlb_ds.games": fact(2)}, NOW, errors={"locked_ds": "403"})
    tabs = {r["id"]: r for r in m["sports"]["mlb"]["tables"]}
    assert tabs["locked_ds.games"]["status"] == "unknown" and tabs["mlb_ds.games"]["status"] == "ok" and m["unreadable"] == {"locked_ds": "403"}
    assert m["sports"]["mlb"]["status"] == "unknown"                    # worse than ok: it is not known to be fine


def test_the_cache_serves_the_last_answer_until_it_expires():
    now = [0.0]
    c = D.Cache(ttl=60, clock=lambda: now[0])
    calls = []
    fn = lambda: calls.append(1) or len(calls)  # noqa: E731
    assert (c.get(fn), c.get(fn)) == (1, 1)
    now[0] = 61
    assert c.get(fn) == 2
