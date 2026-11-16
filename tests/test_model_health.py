import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from mllab import health as H
from mllab.health import Finding, ModelFacts, Snapshot, SportFacts, Thresholds

ET = ZoneInfo("America/New_York")
T = Thresholds()


def at(hour, minute=0, day=15):
    return datetime(2026, 9, day, hour, minute, tzinfo=ET)


def model(name="v10", **kw):
    return ModelFacts(sport=kw.pop("sport", "mlb"), model=name, **kw)


def sport(models, **kw):
    return SportFacts(sport=kw.pop("sport", "mlb"), models=models, **kw)


def codes(findings):
    return [f.code for f in findings]


# --- coverage --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("hour, games, want", [(9, 0, []), (12, 4, ["coverage_gap"]), (12, 10, []), (16, 12, ["coverage_gap"]), (16, 15, [])])
def test_coverage_thresholds_by_time_of_day(hour, games, want):
    s = sport([model(rows_today=games, games_today=games)], scheduled_today=15)
    assert codes(H.rule_coverage(s, at(hour), T)) == want


def test_coverage_severity_and_remedy():
    s = sport([model(games_today=8)], scheduled_today=15)
    f = H.rule_coverage(s, at(16), T)[0]
    assert f.severity == "critical" and f.remedy.action == "trigger_function" and f.remedy.params == {"sport": "mlb", "mode": "predict_today"}
    assert H.rule_coverage(sport([model(games_today=6)], scheduled_today=15), at(12), T)[0].severity == "warn"


def test_no_coverage_alarm_off_season_no_games_or_when_the_live_model_is_paused_or_hidden():
    base = dict(games_today=0)
    assert H.rule_coverage(sport([model(**base)], scheduled_today=15, in_season=False), at(16), T) == []
    assert H.rule_coverage(sport([model(**base)], scheduled_today=0), at(16), T) == []
    assert H.rule_coverage(sport([model(control={"run_state": "paused"}, **base)], scheduled_today=15), at(16), T) == []  # paused on purpose
    assert H.rule_coverage(sport([model(control={"site_visible": "false"}, **base)], scheduled_today=15), at(16), T) == []
    assert H.rule_coverage(sport([model(is_shadow=True, **base)], scheduled_today=15), at(16), T) == []  # shadows never make the site incomplete


# --- features, output, shadows ------------------------------------------------------------------------------------------

def test_stale_features_only_matter_on_a_game_day():
    old = at(16) - timedelta(hours=9)
    assert codes(H.rule_stale_features(sport([], scheduled_today=10, features_computed_at=old), at(16), T)) == ["stale_features"]
    assert H.rule_stale_features(sport([], scheduled_today=0, features_computed_at=old), at(16), T) == []
    assert H.rule_stale_features(sport([], scheduled_today=10, features_computed_at=at(16) - timedelta(hours=2)), at(16), T) == []
    assert H.rule_stale_features(sport([], scheduled_today=10, features_computed_at=None), at(16), T) == []


def test_bad_output_hides_the_model_only_if_a_healthy_fallback_exists():
    bad, good = model("v10", bad_rows=3, rows_today=5), model("v8", rows_today=5)
    f = H.rule_bad_output(sport([bad, good]), bad, T)[0]
    assert f.severity == "critical" and f.remedy.action == "control_hide" and f.remedy.params == {"sport": "mlb", "model": "v10"}
    alone = H.rule_bad_output(sport([bad]), bad, T)[0]
    assert alone.remedy.action == "alert" and "stays visible" in alone.message  # hiding the only model would empty the site
    also_bad = model("v8", rows_today=5, bad_rows=1)
    assert H.rule_bad_output(sport([bad, also_bad]), bad, T)[0].remedy.action == "alert"


def test_collapsed_probabilities_need_enough_rows_and_shadows_and_hidden_models_are_ignored():
    collapsed = dict(n_recent=40, prob_std=0.001, rows_today=5)
    assert codes(H.rule_bad_output(sport([model(**collapsed), model("v8", rows_today=5)]), model(**collapsed), T)) == ["bad_output"]
    few = model(n_recent=5, prob_std=0.0)
    assert H.rule_bad_output(sport([few]), few, T) == []
    assert H.rule_bad_output(sport([]), model(is_shadow=True, bad_rows=9), T) == []
    assert H.rule_bad_output(sport([]), model(control={"site_visible": "false"}, bad_rows=9), T) == []


@pytest.mark.parametrize("hour, rows, control, want", [(13, 0, {}, ["no_output_today"]), (11, 0, {}, []), (13, 4, {}, []), (13, 0, {"run_state": "paused"}, []),
                                                       (13, 0, {"site_visible": "false"}, [])])
def test_no_output_today(hour, rows, control, want):
    m = model(rows_today=rows, control=control)
    assert codes(H.rule_no_output(sport([m], scheduled_today=10), m, at(hour))) == want


def test_shadow_staleness():
    now = at(12)
    stale = model("logit3", is_shadow=True, last_prediction_at=now - timedelta(days=5))
    assert codes(H.rule_shadow_stale(sport([stale]), stale, now, T)) == ["shadow_stale"]
    fresh = model("logit3", is_shadow=True, last_prediction_at=now - timedelta(hours=5))
    assert H.rule_shadow_stale(sport([fresh]), fresh, now, T) == []
    never = model("sim", is_shadow=True)
    assert codes(H.rule_shadow_stale(sport([never]), never, now, T)) == ["shadow_never_ran"]
    paused = model("logit3", is_shadow=True, control={"run_state": "paused"}, last_prediction_at=now - timedelta(days=30))
    assert H.rule_shadow_stale(sport([paused]), paused, now, T) == []  # paused on purpose is not stale
    assert H.rule_shadow_stale(sport([stale], in_season=False), stale, now, T) == []
    assert H.rule_shadow_stale(sport([model("v10", last_prediction_at=now - timedelta(days=30))]), model("v10"), now, T) == []  # not a shadow


def test_probability_shift_and_accuracy_drift_only_warn_never_act():
    m = model(n_recent=60, prob_mean=0.68)
    f = H.rule_calibration_shift(sport([m]), m, T)[0]
    assert f.severity == "warn" and f.remedy.action == "alert"
    assert H.rule_calibration_shift(sport([]), model(n_recent=10, prob_mean=0.9), T) == []
    drift = {"season": {"n": 2000, "log_loss": 0.688, "home_prior_log_loss": 0.693}, "recent": {"n": 300, "log_loss": 0.715, "home_prior_log_loss": 0.694}}
    f = H.rule_accuracy_drift(sport([], accuracy=drift), T)[0]
    assert f.code == "accuracy_drift" and f.remedy.action == "alert"
    still_beats_prior = {**drift, "recent": {"n": 300, "log_loss": 0.715, "home_prior_log_loss": 0.74}}
    assert H.rule_accuracy_drift(sport([], accuracy=still_beats_prior), T) == []      # worse than its own season, but still better than always-home
    tiny = {**drift, "recent": {"n": 40, "log_loss": 0.9, "home_prior_log_loss": 0.69}}
    assert H.rule_accuracy_drift(sport([], accuracy=tiny), T) == []                    # 40 games say nothing


def test_scheduler_rules():
    s = sport([], jobs=[{"name": "a", "state": "ENABLED", "body_ok": False}, {"name": "b", "state": "ENABLED", "body_ok": True, "mode": "mystery"},
                        {"name": "c", "state": "ENABLED", "body_ok": True, "mode": "daily", "last_attempt_ok": False},
                        {"name": "d", "state": "PAUSED", "body_ok": True, "mode": "predict", "last_attempt_ok": False}])
    got = H.rule_scheduler(s, T)
    assert codes(got) == ["scheduler_bad_body", "scheduler_unknown_mode", "scheduler_job_failing"]  # a paused job that failed earlier is not a problem
    assert got[0].severity == "critical" and "silently runs" in got[0].message


def test_site_must_never_be_empty_without_saying_why():
    hidden = [model("v10", control={"site_visible": "false"}), model("v8", control={"site_visible": "false"})]
    assert codes(H.rule_control_consistency(sport(hidden))) == ["site_empty"]
    assert H.rule_control_consistency(sport(hidden, banner="Back soon")) == []
    assert H.rule_control_consistency(sport([model("v10"), model("v8", control={"site_visible": "false"})])) == []
    assert codes(H.rule_control_consistency(sport([model(control={"lifecycle": "live", "artifact_uri": "gs://b/x"})]))) == ["pin_without_sha"]


def test_lab_rules():
    assert codes(H.rule_lab({"disk_used_pct": 91.0}, T)) == ["lab_disk"] and H.rule_lab({"disk_used_pct": 91.0}, T)[0].remedy.action == "prune_envs"
    assert codes(H.rule_lab({"services_ok": False}, T)) == ["lab_services"] and H.rule_lab({"disk_used_pct": 40.0, "services_ok": True}, T) == []


# --- whole-snapshot behaviour ------------------------------------------------------------------------------------------

def test_evaluate_orders_by_severity_and_reports_unavailable_sources_as_info_not_health():
    bad = model("v10", bad_rows=2, rows_today=5, games_today=5)
    snap = Snapshot(at(16), {"mlb": sport([bad, model("v8", rows_today=4, games_today=4)], scheduled_today=15)}, {"disk_used_pct": 90.0},
                    {"bigquery": "ok", "nfl_bigquery": "unavailable: 403 no access to nfl_season"})
    got = H.evaluate(snap)
    sev = [f.severity for f in got]
    assert sev == sorted(sev, key=lambda s: -H.SEVERITIES.index(s))
    assert "source_unavailable" in codes(got) and got[-1].severity == "info"


def test_model_status_precedence():
    bad = model("v10", bad_rows=2, rows_today=5)
    s = sport([bad, model("v8", rows_today=3), model("old", control={"site_visible": "false"}), model("stopped", control={"run_state": "paused"})])
    snap = Snapshot(at(16), {"mlb": s})
    f = H.evaluate(snap)
    st = {m.model: H.model_status(s, m, f) for m in s.models}
    assert st == {"v10": "down", "v8": "ok", "old": "hidden", "stopped": "paused"}


def test_unavailable_source_does_not_turn_a_model_red():
    m = model("v10", rows_today=5, games_today=5)
    snap = Snapshot(at(12), {"mlb": sport([m], scheduled_today=5)}, sources={"scheduler": "unavailable: needs cloudscheduler.viewer"})
    assert H.model_status(snap.sports["mlb"], m, H.evaluate(snap)) == "ok"


def test_snapshot_serialises_to_json():
    snap = Snapshot(at(16), {"mlb": sport([model("v10", last_prediction_at=at(15), bad_rows=1, rows_today=2)], scheduled_today=15)}, {}, {"bigquery": "ok"})
    d = H.to_json(snap, H.evaluate(snap))
    assert json.loads(json.dumps(d)) == d and d["sports"]["mlb"]["models"][0]["status"] in ("down", "warn", "ok") and d["findings"][0]["severity"]


def test_weekly_sports_tolerate_a_week_between_shadow_writes():
    now = at(12)
    four_days = model("ridge", is_shadow=True, last_prediction_at=now - timedelta(days=4))
    nine_days = model("ridge", is_shadow=True, last_prediction_at=now - timedelta(days=9))
    assert H.rule_shadow_stale(sport([four_days], sport="nfl"), four_days, now, T) == []      # normal for a weekly sport
    assert codes(H.rule_shadow_stale(sport([nine_days], sport="nfl"), nine_days, now, T)) == ["shadow_stale"]
    assert codes(H.rule_shadow_stale(sport([four_days], sport="mlb"), four_days, now, T)) == ["shadow_stale"]  # still stale for daily MLB
