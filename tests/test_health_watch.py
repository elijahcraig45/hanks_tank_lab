"""What the heartbeat reports about the pipeline and the data: a Live job that failed, a broker that stopped publishing, tables that are stale, missing or not updated."""
import calendar
import importlib.machinery
import importlib.util
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "host" / "mllab-health"


def load():
    loader = importlib.machinery.SourceFileLoader("mllab_health_under_test", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


H = load()
NOW = calendar.timegm(time.strptime("2026-10-05 15:00:00", "%Y-%m-%d %H:%M:%S"))      # 15:00 UTC, exactly


def state(taken="2026-10-05T14:59:30+00:00", **jobs):
    return {"taken_at": taken, "jobs": [{"name": n, "mode": m, "last": l} for n, (m, l) in jobs.items()]}


def test_a_healthy_pipeline_reports_nothing():
    st = state(a=("live", {"mode": "live", "ok": True}), b=("dry", {"mode": "dry", "ok": True}), c=("off", None))
    assert H.pipeline_problems(st, NOW) == []


def test_a_live_job_whose_last_live_run_failed_is_reported_with_its_error():
    out = H.pipeline_problems(state(a=("live", {"mode": "live", "ok": False, "error": "KeyError: week"})), NOW)
    assert out == ["a last live run FAILED (KeyError: week)"]


def test_a_failed_dry_run_or_a_job_that_is_not_live_is_not_an_alert():
    st = state(a=("live", {"mode": "dry", "ok": False}), b=("dry", {"mode": "dry", "ok": False}), c=("off", {"mode": "live", "ok": False}))
    assert H.pipeline_problems(st, NOW) == []                # a rehearsal failing is what rehearsals are for


def test_a_broker_that_stopped_publishing_is_reported():
    out = H.pipeline_problems(state(taken="2026-10-05T14:40:00+00:00"), NOW)
    assert len(out) == 1 and "stopped publishing" in out[0] and "20 min" in out[0]
    assert H.pipeline_problems(state(taken="2026-10-05T14:56:00+00:00"), NOW) == []     # 4 min old is within the 2-minute timer plus slack


def test_an_unreadable_snapshot_time_is_reported_rather_than_trusted():
    assert H.pipeline_problems({"jobs": []}, NOW) == ["pipeline snapshot has no readable time"]


def table(label, status):
    return {"label": label, "status": status}


def test_only_stale_missing_and_not_updated_tables_are_reported_late_and_idle_are_not():
    data = {"available": True, "sports": {"nfl": {"tables": [table("Stat leaders", "stale"), table("Games", "late"), table("Teams", "ok")]},
                                          "cfb": {"tables": [table("Drives", "idle"), table("Season sim", "missing"), table("Players", "unchanged")]}}}
    out = H.data_problems(data)
    assert out == ["data: nfl stat leaders stale; cfb season sim missing; cfb players not updated"]


def test_a_long_list_is_cut_to_three_names_and_a_count():
    data = {"available": True, "sports": {"mlb": {"tables": [table(f"T{i}", "stale") for i in range(6)]}}}
    assert H.data_problems(data) == ["data: mlb t0 stale; mlb t1 stale; mlb t2 stale and 3 more"]


def test_a_data_map_that_cannot_be_read_is_a_problem_in_itself():
    assert H.data_problems({"available": False, "error": "403 Access Denied"}) == ["data map unavailable: 403 Access Denied"]
    assert H.data_problems({"available": True, "sports": {}}) == []


def test_a_long_message_keeps_every_problem_visible():
    weekly = "weekly run failed: data check failed: mlb_2026_season.games (freshness: newest game_date is 5.5 days old, limit 4); " \
             "data check failed: mlb_2026_season.game_predictions (freshness: newest game_date is 5.5 days old, limit 4)"
    msg = H.fit([weekly, "data: cfb stat leaders not updated; cfb player season stats not updated"])
    assert len(msg) <= H.MSG_MAX and "cfb stat leaders" in msg and msg.startswith("weekly run failed")


def test_a_short_message_is_untouched_and_too_many_problems_are_counted():
    assert H.fit(["a", "b"]) == "a; b"
    msg = H.fit([f"problem number {i} " + "x" * 60 for i in range(12)])
    assert len(msg) <= H.MSG_MAX and msg.endswith("more)") and "problem number 0" in msg


def test_a_clean_data_map_reports_nothing():
    assert H.data_problems({"available": True, "sports": {"mlb": {"tables": [table("Games", "ok"), table("Standings", "idle"), table("Rosters", "late")]}}}) == []


# ---------------------------------------------------------------------------------------------------------------- wired into the probe
import json  # noqa: E402


def probe(monkeypatch, tmp_path, state_json=None, app=None):
    monkeypatch.setenv("MLLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "nokey"))
    mod = load()
    monkeypatch.setattr(mod, "status", lambda url, headers=None: 401 if url.endswith("/auth") else 200)
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0})())
    if state_json is not None:
        (tmp_path / "pipeline-status").mkdir(exist_ok=True)
        (tmp_path / "pipeline-status" / "state.json").write_text(json.dumps(state_json))

    def get(path, timeout=30):
        if isinstance(app, Exception):
            raise app
        return app
    monkeypatch.setattr(mod, "app_get", get)
    return mod


def test_the_probe_reports_a_failed_live_run_and_stale_data_together(monkeypatch, tmp_path):
    st = {"taken_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), "jobs": [{"name": "nfl-weekly-predict", "mode": "live", "last": {"mode": "live", "ok": False, "error": "boom"}}]}
    app = {"available": True, "sports": {"nfl": {"tables": [table("Stat leaders", "stale")]}}}
    probs = probe(monkeypatch, tmp_path, st, app).problems()
    assert any("nfl-weekly-predict last live run FAILED" in p for p in probs) and any("data: nfl stat leaders stale" in p for p in probs)


def test_the_probe_reports_when_it_cannot_look_instead_of_skipping_the_check(monkeypatch, tmp_path):
    probs = probe(monkeypatch, tmp_path, state_json=None, app=OSError("refused")).problems()
    assert any("pipeline status unreadable" in p for p in probs) and any("data check failed (OSError)" in p for p in probs)


def test_a_healthy_pipeline_and_data_leave_the_probe_quiet(monkeypatch, tmp_path):
    st = {"taken_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), "jobs": []}
    assert probe(monkeypatch, tmp_path, st, {"available": True, "sports": {}}).problems() == []


def test_a_failed_per_game_task_is_an_alert_only_while_pregame_is_live():
    live = {"mode": "live", "stores": {"live": {"counts": {"done": 3, "failed": 2}}}}
    st = dict(state(), pregame=live)
    assert H.pipeline_problems(st, NOW) == ["per-game MLB: 2 task(s) failed"]
    assert H.pipeline_problems(dict(state(), pregame={**live, "mode": "dry"}), NOW) == []            # a rehearsal failing is what rehearsals are for
    assert H.pipeline_problems(dict(state(), pregame={"mode": "live", "stores": {"live": {"counts": {"done": 9}}}}), NOW) == []
    assert H.pipeline_problems(dict(state(), pregame={"mode": "live", "stores": {}}), NOW) == []     # nothing planned yet is not a failure


def test_data_check_retries_once_then_tolerates_a_lone_timeout(tmp_path):
    f, calls = tmp_path / "n", []

    def slow_then_ok(t):
        calls.append(t)
        if len(calls) == 1:
            raise TimeoutError()
        return {"available": True, "sports": {}}
    assert H.data_check(slow_then_ok, str(f)) == [] and calls == [30, 60]


def test_data_check_reports_only_after_consecutive_failures(tmp_path):
    f = str(tmp_path / "n")

    def down(t):
        raise TimeoutError()
    assert H.data_check(down, f) == [] and H.data_check(down, f) == []
    assert H.data_check(down, f) == ["data check failed (TimeoutError) 3 runs in a row"]


def test_data_check_resets_on_success_and_never_hides_real_problems(tmp_path):
    f = str(tmp_path / "n")
    open(f, "w").write("2")
    stale = {"available": True, "sports": {"mlb": {"tables": [{"label": "Games", "status": "stale"}]}}}
    assert H.data_check(lambda t: stale, f) == ["data: mlb games stale"] and open(f).read() == "0"


def test_data_check_unwritable_state_reports_at_once(tmp_path):
    def down(t):
        raise TimeoutError()
    assert H.data_check(down, str(tmp_path / "nodir" / "n")) == ["data check failed (TimeoutError) 3 runs in a row"]
