import json
import random
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from mllab import control as C
from mllab.control import Control, ControlError, Event, MemoryStore, UNSET

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
SQL_DIR = Path(__file__).resolve().parents[1] / "sql" / "control"
SHA = "a" * 64


def ev(field, value, *, sport="mlb", target="v10", ts=T0, rec=None, eid=None, note=""):
    return Event(eid or str(uuid.uuid4()), ts, rec or ts, sport, target, field, value, "test", note)


def ctl(events=None, models=("v10", "v8", "logit3"), now=T0 + timedelta(days=1)):
    clock = {"t": now}
    c = Control(MemoryStore(events), "henry", now=lambda: clock["t"], known_models=lambda s: list(models), audit_path=None)
    c._clock = clock
    return c


@pytest.fixture(autouse=True)
def _audit(tmp_path, monkeypatch):
    monkeypatch.setenv("MLLAB_AUDIT", str(tmp_path / "audit.log"))


# --- validation ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("field, value, want", [
    ("site_visible", " TRUE ", "true"), ("site_visible", "false", "false"), ("run_state", "paused", "paused"), ("lifecycle", "shadow", "shadow"),
    ("artifact_uri", "gs://hanks_tank_data/models/vertex/x/model.pkl", "gs://hanks_tank_data/models/vertex/x/model.pkl"),
    ("artifact_sha256", SHA.upper(), SHA), ("display_label", "  Production XGBoost ", "Production XGBoost"),
    ("public_note", "Refit weekly from the latest games.", "Refit weekly from the latest games."),
    ("sort_order", "5", "5"), ("tier_high", "0.64", "0.64"), ("tier_medium", "0.6", "0.6"), (
        "site_visible", UNSET, UNSET)])
def test_valid_values_are_canonicalised(field, value, want):
    assert C.validate("mlb", "v10", field, value) == want


@pytest.mark.parametrize("field, value", [
    ("site_visible", "yes"), ("run_state", "stopped"), ("lifecycle", "primary"), ("artifact_uri", "http://evil/x"), ("artifact_uri", "gs://b"),
    ("artifact_sha256", "abc"), ("artifact_sha256", "z" * 64), ("display_label", ""), ("display_label", "x" * 61),
    ("display_label", "<b>hi</b>"), ("public_note", "click <a href=x>here</a>"), ("public_note", "bad\x00char"),
    ("sort_order", "1.5"), ("sort_order", "99999"), ("tier_high", "0.3"), ("tier_high", "1.0"), ("tier_high", "abc")])
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ControlError):
        C.validate("mlb", "v10", field, value)


@pytest.mark.parametrize("sport, target, field", [
    ("hockey", "v10", "site_visible"), ("mlb", "*", "site_visible"), ("mlb", "v10", "banner"), ("mlb", "v 10", "site_visible"),
    ("mlb", "../x", "lifecycle"), ("mlb", "v10", "no_such_field")])
def test_sport_target_and_field_are_checked(sport, target, field):
    with pytest.raises(ControlError):
        C.validate(sport, target, field, "true")


def test_banner_is_sport_wide_only_and_plain_text():
    assert C.validate("nfl", "*", "banner", "Scores delayed today") == "Scores delayed today"
    assert C.validate("nfl", "*", "banner_level", "warn") == "warn"
    with pytest.raises(ControlError, match="markup"):
        C.validate("nfl", "*", "banner", "<script>alert(1)</script>")


# --- state semantics ---------------------------------------------------------------------------------------------------

def test_latest_effective_event_wins_and_defaults_apply_when_absent():
    evs = [ev("site_visible", "false", ts=T0), ev("site_visible", "true", ts=T0 + timedelta(hours=1))]
    st = C.current_state(evs, T0 + timedelta(days=1))[("mlb", "v10")]
    assert st["site_visible"] == "true"
    assert C.effective({}, "site_visible") == "true" and C.effective({}, "run_state") == "active" and C.effective({}, "lifecycle") is None


def test_future_dated_events_are_scheduled_not_applied():
    evs = [ev("site_visible", "false", ts=T0 + timedelta(days=3))]
    assert C.current_state(evs, T0 + timedelta(days=1)) == {}
    assert C.current_state(evs, T0 + timedelta(days=4))[("mlb", "v10")]["site_visible"] == "false"


def test_unset_returns_a_field_to_its_default():
    evs = [ev("display_label", "Old", ts=T0), ev("display_label", UNSET, ts=T0 + timedelta(hours=1))]
    assert "display_label" not in C.current_state(evs, T0 + timedelta(days=1)).get(("mlb", "v10"), {})


def test_ties_break_by_recorded_at_then_event_id():
    a = ev("lifecycle", "shadow", ts=T0, rec=T0, eid="aaa"); b = ev("lifecycle", "live", ts=T0, rec=T0 + timedelta(seconds=1), eid="000")
    assert C.current_state([a, b], T0 + timedelta(days=1))[("mlb", "v10")]["lifecycle"] == "live"  # later recorded_at wins
    c = ev("lifecycle", "archived", ts=T0, rec=T0, eid="zzz")
    assert C.current_state([a, c], T0 + timedelta(days=1))[("mlb", "v10")]["lifecycle"] == "archived"  # same time: higher event_id wins


# --- the real SQL view must agree with the Python logic ----------------------------------------------------------------

def _run_view_in_duckdb(events, now):
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE events (event_id VARCHAR, event_ts TIMESTAMP, recorded_at TIMESTAMP, sport VARCHAR, target VARCHAR, field VARCHAR, "
                "value VARCHAR, actor VARCHAR, note VARCHAR)")
    con.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)",
                    [(e.event_id, e.event_ts.replace(tzinfo=None), e.recorded_at.replace(tzinfo=None), e.sport, e.target, e.field, e.value, e.actor, e.note)
                     for e in events])
    sql = (SQL_DIR / "02_current_view.sql").read_text()
    sql = sql.replace("`hankstank.control.model_control_events`", "events").replace("`hankstank.control.model_control_current`", "current_view")
    sql = sql.replace("CURRENT_TIMESTAMP()", f"TIMESTAMP '{now.replace(tzinfo=None).isoformat(sep=' ')}'")
    con.execute(sql)
    cols = [c[1] for c in con.execute("PRAGMA table_info('current_view')").fetchall()]
    return {(r[0], r[1]): {k: v for k, v in zip(cols[2:], r[2:]) if v is not None} for r in con.execute("SELECT * FROM current_view").fetchall()}


@pytest.mark.parametrize("seed", range(12))
def test_sql_view_matches_python_semantics_on_random_event_streams(seed):
    rng = random.Random(seed)
    fields = list(C.MODEL_FIELDS)
    values = {"site_visible": ["true", "false"], "run_state": ["active", "paused"], "lifecycle": ["live", "shadow", "archived"], "artifact_uri": ["gs://b/a", "gs://b/c"],
              "artifact_sha256": [SHA, "b" * 64], "display_label": ["A", "B"], "public_note": ["n1", "n2"], "sort_order": ["1", "2"],
              "tier_high": ["0.64", "0.7"], "tier_medium": ["0.57", "0.6"]}
    events = []
    for _ in range(80):
        f = rng.choice(fields + ["banner"])
        sport, target = rng.choice(["mlb", "nfl"]), ("*" if f == "banner" else rng.choice(["v10", "v8", "ridge"]))
        v = UNSET if rng.random() < 0.15 else (rng.choice(["a", "b"]) if f == "banner" else rng.choice(values[f]))
        ts = T0 + timedelta(hours=rng.choice([-5, -5, -2, 0, 0, 3, 40]))  # deliberate ties and some future events
        events.append(ev(f, v, sport=sport, target=target, ts=ts, rec=T0 + timedelta(seconds=rng.randrange(0, 5)), eid=f"{rng.randrange(10**6):06d}-{_}"))
    now = T0 + timedelta(hours=10)
    want = {k: {f: v for f, v in row.items() if f != "updated_at"} for k, row in C.current_state(events, now).items()}
    got = _run_view_in_duckdb(events, now)
    assert all(row.get("role") == row.get("lifecycle") for row in got.values())  # the transition alias always mirrors lifecycle
    got = {k: {f: v for f, v in row.items() if f not in ("updated_at", "role")} for k, row in got.items()}
    assert got == want


def test_old_field_name_role_is_refused_with_a_pointer_to_lifecycle():
    with pytest.raises(ControlError, match="renamed 'lifecycle'"):
        C.validate("mlb", "v10", "role", "live")
    with pytest.raises(ControlError, match="renamed 'lifecycle'"):
        ctl().set("mlb", "v10", "role", "live")


# --- operations --------------------------------------------------------------------------------------------------------

def test_live_affecting_changes_need_the_typed_phrase():
    c = ctl()
    with pytest.raises(ControlError, match="confirm with 'mlb/v8'"):
        c.set("mlb", "v8", "site_visible", "false")
    with pytest.raises(ControlError, match="confirm"):
        c.set("mlb", "v8", "run_state", "paused", confirm="yes")
    c.set("mlb", "v8", "site_visible", "false", confirm="mlb/v8")
    assert c.state("mlb")[("mlb", "v8")]["site_visible"] == "false"
    c.set("mlb", "v8", "display_label", "Old V8")  # cosmetic: no phrase needed
    c.set("mlb", "v8", "site_visible", "true")     # showing a model again is not risky


def test_the_last_visible_model_cannot_be_hidden_by_accident():
    c = ctl(models=("v10", "v8"))
    c.set("mlb", "v8", "site_visible", "false", confirm="mlb/v8")
    with pytest.raises(ControlError, match="only visible mlb model"):
        c.set("mlb", "v10", "site_visible", "false", confirm="mlb/v10")
    c.set("mlb", "v10", "site_visible", "false", confirm="mlb/v10", allow_empty=True)  # deliberate


def test_dry_run_writes_nothing_and_actor_and_audit_are_recorded(tmp_path):
    c = ctl(); c.audit_path = tmp_path / "a.log"
    c.set("mlb", "v8", "display_label", "Dry", dry_run=True)
    assert c.store.all_events() == []
    e = c.set("mlb", "v8", "display_label", "Real", note="rename")
    assert e.actor == "henry" and c.store.all_events()[0].note == "rename"
    assert "control_set" in (tmp_path / "a.log").read_text()


def test_pin_needs_the_phrase_and_writes_the_artifact_before_going_live():
    c = ctl()
    with pytest.raises(ControlError, match="confirm"):
        c.pin("mlb", "v11", "gs://b/m.pkl", SHA)
    with pytest.raises(ControlError, match="64 hex"):
        c.pin("mlb", "v11", "gs://b/m.pkl", "short", confirm="mlb/v11")
    evs = c.pin("mlb", "v11", "gs://b/m.pkl", SHA, confirm="mlb/v11")
    assert [e.field for e in evs] == ["artifact_uri", "artifact_sha256", "lifecycle"]  # never live without its artifact
    assert c.state("mlb")[("mlb", "v11")]["lifecycle"] == "live"


def test_revert_restores_the_previous_value_or_unsets():
    c = ctl(); clock = c._clock
    c.set("mlb", "v8", "display_label", "One"); clock["t"] += timedelta(minutes=1)
    c.set("mlb", "v8", "display_label", "Two"); clock["t"] += timedelta(minutes=1)
    c.revert("mlb", "v8", "display_label")
    assert c.state("mlb")[("mlb", "v8")]["display_label"] == "One"      # back to what was in force before "Two"
    c2 = ctl(); c2.set("mlb", "v8", "display_label", "Only"); c2._clock["t"] += timedelta(minutes=1)
    c2.revert("mlb", "v8", "display_label")
    assert "display_label" not in c2.state("mlb").get(("mlb", "v8"), {})  # nothing came before it: back to the default
    with pytest.raises(ControlError, match="no history"):
        ctl().revert("mlb", "v8", "lifecycle")


def test_scheduled_change_takes_effect_only_when_its_time_comes():
    c = ctl(); clock = c._clock
    at = clock["t"] + timedelta(days=2)
    c.set("mlb", "v8", "display_label", "Later", at=at)
    assert "display_label" not in c.state("mlb").get(("mlb", "v8"), {})
    clock["t"] = at + timedelta(minutes=1)
    assert c.state("mlb")[("mlb", "v8")]["display_label"] == "Later"
    with pytest.raises(ControlError, match="timezone"):
        c.set("mlb", "v8", "display_label", "x", at=datetime(2026, 12, 1, 9, 0))


def test_freeze_hides_everything_with_a_banner_and_unfreeze_restores_exactly():
    c = ctl(models=("v10", "v8")); clock = c._clock
    c.set("mlb", "v8", "site_visible", "false", confirm="mlb/v8", allow_empty=False); clock["t"] += timedelta(minutes=1)
    with pytest.raises(ControlError, match="confirm"):
        c.freeze("mlb", "Maintenance")
    c.freeze("mlb", "Predictions paused for maintenance", confirm="mlb/*"); clock["t"] += timedelta(minutes=1)
    st = c.state("mlb")
    assert st[("mlb", "v10")]["site_visible"] == "false" and st[("mlb", "*")]["banner"].startswith("Predictions paused")
    clock["t"] += timedelta(minutes=1)
    c.unfreeze("mlb")
    st = c.state("mlb")
    assert C.effective(st.get(("mlb", "v10"), {}), "site_visible") == "true" and "banner" not in st.get(("mlb", "*"), {})  # no row = defaults
    assert st[("mlb", "v8")]["site_visible"] == "false"  # v8 was hidden BEFORE the freeze and stays hidden


def test_unfreeze_does_not_clobber_a_change_made_after_the_freeze():
    c = ctl(models=("v10", "v8")); clock = c._clock
    c.freeze("mlb", "Maintenance", confirm="mlb/*"); clock["t"] += timedelta(minutes=1)
    c.set("mlb", "v10", "site_visible", "true"); clock["t"] += timedelta(minutes=1)  # someone deliberately re-showed v10
    c.unfreeze("mlb")
    assert c.state("mlb")[("mlb", "v10")]["site_visible"] == "true"
    with pytest.raises(ControlError, match="no freeze"):
        ctl().unfreeze("mlb")


def test_freeze_needs_a_model_list():
    c = Control(MemoryStore(), "h", known_models=None)
    with pytest.raises(ControlError, match="known models is unavailable"):
        c.freeze("mlb", "x", confirm="mlb/*")


# --- the BigQuery CLI store ---------------------------------------------------------------------------------------------

class FakeRun:
    def __init__(self, stdout="", code=0, stderr=""):
        self.calls, self.stdout, self.code, self.stderr = [], stdout, code, stderr

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        return SimpleNamespace(returncode=self.code, stdout=self.stdout, stderr=self.stderr)


def test_bq_store_writes_with_parameters_never_string_built_sql():
    run = FakeRun(); s = C.BqCliStore(runner=run)
    hostile = "x'); DROP TABLE t; --"
    s.append(Event("id1", T0, T0, "mlb", "v10", "public_note", hostile, "henry", "note"))
    cmd, kw = run.calls[0]
    sql = cmd[-1]
    assert "@value" in sql and "DROP" not in sql and "INSERT INTO `hankstank.control.model_control_events`" in sql
    assert f"--parameter=value:STRING:{hostile}" in cmd  # the hostile text travels only as a parameter
    assert kw["env"]["CLOUDSDK_CORE_ACCOUNT"] == "you@example.com"


def test_bq_store_reads_events_back_and_reports_failures():
    rows = [{"event_id": "e1", "event_ts": "2026-10-01 12:00:00+00", "recorded_at": "2026-10-01 12:00:01+00", "sport": "mlb", "target": "v10",
             "field": "lifecycle", "value": "live", "actor": "h", "note": ""}]
    got = C.BqCliStore(runner=FakeRun(stdout=json.dumps(rows))).all_events()
    assert got[0].field == "lifecycle" and got[0].event_ts.tzinfo is not None
    with pytest.raises(ControlError, match="bq failed"):
        C.BqCliStore(runner=FakeRun(code=1, stderr="Access Denied")).all_events()


def test_bq_store_refuses_the_work_account():
    with pytest.raises(ControlError, match="work account"):
        C.BqCliStore(account="someone@blocked.example")


# --- production guard + live registry ------------------------------------------------------------------------------------

def ctl_prod(events=None):
    return Control(MemoryStore(events), "henry", now=lambda: T0 + timedelta(days=1), known_models=lambda s: ["v10", "v8", "logit3"],
                   production_keys=lambda s: {"v10"})


def test_hiding_the_production_model_needs_a_banner_that_explains_why():
    c = ctl_prod()
    with pytest.raises(ControlError, match="needs a banner"):
        c.set("mlb", "v10", "site_visible", "false", confirm="mlb/v10")
    c.set("mlb", "logit3", "site_visible", "false", confirm="mlb/logit3")   # not production: no banner needed
    c.set("mlb", "*", "banner", "Model under review", confirm="mlb/*")
    c.set("mlb", "v10", "site_visible", "false", confirm="mlb/v10")          # banner now in force
    assert C.effective(c.state("mlb")[("mlb", "v10")], "site_visible") == "false"


def test_hide_with_a_banner_sets_the_banner_first():
    c = ctl_prod()
    evs = c.hide("mlb", "v10", confirm="mlb/v10", banner="Predictions are being rebuilt")
    assert [e.field for e in evs] == ["banner", "banner_level", "site_visible"]  # never a silent gap
    st = c.state("mlb")
    assert st[("mlb", "*")]["banner"] == "Predictions are being rebuilt" and st[("mlb", "v10")]["site_visible"] == "false"
    with pytest.raises(ControlError, match="confirm"):
        ctl_prod().hide("mlb", "v10", banner="x")


def test_site_registry_reads_keys_roles_and_production():
    from mllab import site
    payload = {"success": True, "data": {"models": [{"key": "v10", "label": "Production XGBoost", "role": "production", "available": True},
                                                      {"key": "logit3", "role": "shadow"}, {"key": "sim_blend", "role": "shadow", "available": False}, {"noKey": 1}]}}
    fetch = lambda url: payload
    assert site.known_models("mlb", fetch=fetch) == ["v10", "logit3", "sim_blend"]
    assert site.production_keys("mlb", fetch=fetch) == {"v10"}
    assert site.registry("mlb", fetch=fetch)[2]["available"] is False
    with pytest.raises(site.SiteError, match="unknown sport"):
        site.registry("hockey", fetch=fetch)
    with pytest.raises(site.SiteError, match="unexpected shape"):
        site.registry("mlb", fetch=lambda u: {"data": {}})
