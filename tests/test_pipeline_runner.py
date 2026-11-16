"""The runner: modes, locking, the real subprocess shim, failure paths, and what a failed secret read does."""
import json
import threading
import time
from pathlib import Path

import pytest

from mllab.pipeline import jobs as J
from mllab.pipeline import runner as R


# ----------------------------------------------------------------------------------------------------------------- a real fake bundle
ENTRY_SRC = '''
import json, os, sys, time

def run(request):
    p = request.get_json(silent=True) or {}
    print(json.dumps({"severity": "NOTICE", "message": "log line on stdout"}))   # the real pipeline logs to stdout
    kind = p.get("kind", "mlb")
    if kind == "mlb":                      # (json string, status, headers)
        return (json.dumps({"status": "ok", "date": "2026-10-01", "steps": [{"step": "a", "status": "ok", "seconds": 1.5, "secret": "x" * 50}]}), 200, {})
    if kind == "football":                 # (dict, status)
        return ({"mode": "m", "steps": {"s1": {}, "s2": 3}}, 200)
    if kind == "football_step_error":      # 200, but one step caught an exception (cfb shadow_drive_sim, 2026-11-10)
        return ({"status": "ok", "mode": "m", "steps": {"s1": {}, "shadow": {"error": "negative axis 1 index: -1"}}}, 200)
    if kind == "mlb_step_failed":
        return (json.dumps({"status": "ok", "steps": [{"step": "a", "status": "ok"}, {"step": "b", "status": "failed"}]}), 200, {})
    if kind == "error":
        return ({"status": "error", "error": "nope"}, 500)
    if kind == "soft":                     # 200 whose body says error
        return (json.dumps({"status": "error", "error": "step failed"}), 200, {})
    if kind == "echo":
        env = {k: os.environ.get(k) for k in ("K_SERVICE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NFL_DATASET", "CFBD_API_KEY", "PYTHONPATH", "AWS_SECRET_ACCESS_KEY")}
        return ({"status": "ok", "payload": p, "env": env}, 200)
    if kind == "raise":
        raise RuntimeError("boom")
    if kind == "sleep":
        time.sleep(30)
    return ({}, 200)
'''


@pytest.fixture
def release(tmp_path):
    d = tmp_path / "release" / "fakebundle"
    d.mkdir(parents=True)
    (d / "fake_entry.py").write_text(ENTRY_SRC)
    return tmp_path / "release"


BUNDLE = J.Bundle("fakebundle", "fake_entry:run", {"NFL_DATASET": "nfl_season"}, {"CFBD_API_KEY": "sm://projects/p/secrets/s/versions/latest"})


def go(release, payload, **kw):
    return R.execute(BUNDLE, payload, timeout=kw.pop("timeout", 60), release=release, resolve=kw.pop("resolve", lambda ref: "SECRET-VALUE"), **kw)


def test_mlb_style_result_is_parsed_despite_log_lines_on_stdout(release):
    r = go(release, {"kind": "mlb"})
    assert r["ok"] and r["http_status"] == 200 and r["error"] is None
    assert r["summary"] == {"status": "ok", "date": "2026-10-01", "steps": [{"step": "a", "status": "ok", "seconds": 1.5}]}
    assert "xxxxx" not in json.dumps(r["summary"])  # only step name/status/seconds are kept, never a body field
    assert "log line on stdout" in r["log_tail"] and "@@RESULT@@" not in r["log_tail"]


def test_football_style_dict_result_is_parsed(release):
    r = go(release, {"kind": "football"})
    assert r["ok"] and r["summary"]["steps"] == {"s1": "ok", "s2": "ok"}


def test_a_200_with_a_failed_step_is_a_failed_run(release):
    # The CFB drive-sim shadow raised on 2026-11-10; the pipeline caught it into its step, answered 200, and the run
    # was recorded ok, so nothing alerted. A step that reports an error now fails the run and names the step.
    r = go(release, {"kind": "football_step_error"})
    assert not r["ok"] and r["http_status"] == 200
    assert r["error"] == "step shadow: negative axis 1 index: -1"
    assert r["summary"]["steps"] == {"s1": "ok", "shadow": "negative axis 1 index: -1"}


def test_an_mlb_step_marked_failed_is_a_failed_run(release):
    r = go(release, {"kind": "mlb_step_failed"})
    assert not r["ok"] and r["error"] == "step b: failed"


def test_http_500_is_a_failed_run_with_the_error_text(release):
    r = go(release, {"kind": "error"})
    assert not r["ok"] and r["http_status"] == 500 and r["error"] == "nope"


def test_a_200_whose_body_says_error_is_still_a_failure(release):
    r = go(release, {"kind": "soft"})
    assert not r["ok"] and r["error"] == "step failed"


def test_an_exception_in_the_entrypoint_is_a_failed_run_not_a_crash_of_the_runner(release):
    r = go(release, {"kind": "raise"})
    assert not r["ok"] and r["http_status"] is None and "no result" in r["error"] and "boom" in r["log_tail"]


def test_timeout_kills_the_process_and_says_so(release):
    t = time.time()
    r = go(release, {"kind": "sleep"}, timeout=1)
    assert not r["ok"] and r["error"] == "timeout" and r["exit"] == 124 and time.time() - t < 10


def test_subprocess_environment_is_exactly_what_the_function_had_plus_caps(release, monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-leak")
    monkeypatch.setenv("K_SERVICE", "must-not-leak")
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    body = None
    # run the shim for real and read the env the entrypoint saw
    env = R.build_env(BUNDLE, release, resolve=lambda ref: "SECRET-VALUE")
    code, out, _ = R.subprocess_executor(env, json.dumps({"kind": "echo"}), 30)
    body = R.parse_result(out)["body"]
    seen = body["env"]
    assert seen["K_SERVICE"] is None and seen["AWS_SECRET_ACCESS_KEY"] is None  # neither parent variable crossed over
    assert seen["OMP_NUM_THREADS"] == "1" and seen["MKL_NUM_THREADS"] == "1"
    assert seen["NFL_DATASET"] == "nfl_season" and seen["CFBD_API_KEY"] == "SECRET-VALUE"
    assert seen["PYTHONPATH"] == str(release / "fakebundle")


def test_payload_reaches_the_entrypoint_unchanged(release):
    env = R.build_env(BUNDLE, release, resolve=lambda ref: "x")
    _, out, _ = R.subprocess_executor(env, json.dumps({"kind": "echo", "mode": "daily", "dry_run": True, "game_pks": [1, 2]}), 30)
    assert R.parse_result(out)["body"]["payload"] == {"kind": "echo", "mode": "daily", "dry_run": True, "game_pks": [1, 2]}


def test_preflight_imports_the_entrypoint_and_runs_nothing(release):
    r = go(release, {"kind": "raise"}, check=True)  # would raise if it were called
    assert r["ok"] and "preflight" in r["summary"]
    bad = R.execute(J.Bundle("fakebundle", "no_such_module:run"), {}, timeout=30, release=release, check=True)
    assert not bad["ok"] and "preflight exit" in bad["error"]


def test_a_secret_that_cannot_be_read_fails_the_run_and_never_calls_the_entrypoint(release):
    called = []
    def boom(ref):
        raise PermissionError("denied: sm://projects/p/secrets/s")
    r = R.execute(BUNDLE, {"kind": "mlb"}, timeout=5, release=release, resolve=boom, executor=lambda *a: called.append(a) or (0, "", ""))
    assert not r["ok"] and r["error"].startswith("environment: PermissionError") and not called


# ------------------------------------------------------------------------------------------------------------------------------ modes
def test_mode_files(tmp_path):
    assert R.read_mode("mlb-2026-daily", tmp_path) == "off"  # no file: off
    R.write_mode("mlb-2026-daily", "dry", tmp_path)
    assert R.read_mode("mlb-2026-daily", tmp_path) == "dry"
    R.write_mode("mlb-2026-daily", "live", tmp_path)
    assert R.read_mode("mlb-2026-daily", tmp_path) == "live"
    (tmp_path / "x").write_text("# comment only\n\n")
    assert R.read_mode("x", tmp_path) == "off"
    (tmp_path / "y").write_text("LIVE\n")
    assert R.read_mode("y", tmp_path) == "off"  # unrecognised never means live
    (tmp_path / "z").write_text("# first\nlive   # then a comment\n")
    assert R.read_mode("z", tmp_path) == "live"


def test_write_mode_validates(tmp_path):
    with pytest.raises(ValueError):
        R.write_mode("mlb-2026-daily", "yes", tmp_path)
    with pytest.raises(KeyError):
        R.write_mode("nope", "live", tmp_path)
    R.write_mode(J.PREGAME, "dry", tmp_path)
    assert R.read_mode(J.PREGAME, tmp_path) == "dry"


# ---------------------------------------------------------------------------------------------------------------------------- run_job
class Spy:
    def __init__(self, result=(0, R.RESULT_MARK + json.dumps({"status": 200, "body": {"status": "ok"}}), "")):
        self.calls, self.result = [], result

    def __call__(self, env, stdin, timeout, cwd=None):
        self.calls.append({"env": env, "payload": json.loads(stdin), "timeout": timeout})
        return self.result


@pytest.fixture
def dirs(tmp_path):
    return {"jobs_dir": tmp_path / "jobs.d", "state_dir": tmp_path / "state", "release": tmp_path / "rel"}


def run(name, mode, dirs, spy, push=None, **kw):
    if mode:
        R.write_mode(name, mode, dirs["jobs_dir"])
    pushed = []
    code = R.run_job(name, executor=spy, resolve=lambda ref: "S", push=push or (lambda ok, msg: pushed.append((ok, msg))), **dirs, **kw)
    return code, pushed


def test_off_runs_nothing_and_writes_nothing(dirs):
    spy = Spy()
    assert run("mlb-2026-daily", "off", dirs, spy)[0] == 0 and not spy.calls
    assert run("mlb-2026-daily", None, dirs, spy)[0] == 0 and not spy.calls  # a fresh install is off
    assert not (dirs["state_dir"] / "runs.jsonl").exists()


def test_dry_run_sends_dry_run_true_and_never_pushes_a_heartbeat(dirs):
    spy = Spy()
    code, pushed = run("mlb-2026-daily", "dry", dirs, spy)
    assert code == 0 and spy.calls[0]["payload"] == {"mode": "daily", "dry_run": True} and "PIPELINE_CHECK" not in spy.calls[0]["env"] and not pushed


def test_a_job_whose_entrypoint_refuses_dry_run_gets_a_preflight_not_a_run(dirs):
    spy = Spy(result=(0, '{"preflight": "ok"}', ""))
    code, _ = run("nfl-weekly-ingest", "dry", dirs, spy)  # nfl ingest answers dry_run with HTTP 400
    assert code == 0 and spy.calls[0]["env"]["PIPELINE_CHECK"] == "1"
    rec = json.loads((dirs["state_dir"] / "status" / "nfl-weekly-ingest.json").read_text())
    assert rec["mode"] == "dry(preflight)" and rec["ok"]


def test_live_sends_the_exact_scheduler_body_and_pushes_a_heartbeat(dirs):
    spy = Spy()
    code, pushed = run("cfb-weekly-predict", "live", dirs, spy)
    assert code == 0 and spy.calls[0]["payload"] == {"mode": "predict_next"}
    env = spy.calls[0]["env"]
    assert env["CFBD_API_KEY"] == "S" and env["CFB_DATASET"] == "cfb_season" and env["PIPELINE_ENTRY"] == "main:cfb_pipeline"
    assert pushed and pushed[0][0] is True and "cfb-weekly-predict ok" in pushed[0][1]


def test_live_overrides_a_dry_run_flag_in_a_body(dirs, monkeypatch):
    monkeypatch.setitem(J.JOBS, "t", J.Job("t", "mlb", {"mode": "daily", "dry_run": True}, "0 4 * * *"))
    spy = Spy()
    run("t", "live", dirs, spy)
    assert "dry_run" not in spy.calls[0]["payload"]


def test_a_failed_live_run_exits_1_records_the_error_and_pushes_down(dirs):
    spy = Spy(result=(0, R.RESULT_MARK + json.dumps({"status": 500, "body": {"status": "error", "error": "BigQuery 403"}}), ""))
    code, pushed = run("mlb-2026-daily", "live", dirs, spy)
    assert code == 1 and pushed[0][0] is False
    rec = json.loads((dirs["state_dir"] / "status" / "mlb-2026-daily.json").read_text())
    assert not rec["ok"] and rec["error"] == "BigQuery 403" and rec["http_status"] == 500


def test_runs_are_appended_and_the_status_file_is_the_latest(dirs):
    spy = Spy()
    run("mlb-2026-daily", "live", dirs, spy)
    run("mlb-2026-daily", None, dirs, spy)
    lines = (dirs["state_dir"] / "runs.jsonl").read_text().splitlines()
    assert len(lines) == 2 and all(json.loads(x)["job"] == "mlb-2026-daily" for x in lines)
    assert R.last_runs(dirs["state_dir"])["mlb-2026-daily"]["ok"]


def test_a_second_run_while_one_is_in_progress_is_skipped_not_stacked(dirs):
    spy = Spy()
    R.write_mode("mlb-2026-daily", "live", dirs["jobs_dir"])
    with R.lock("mlb-2026-daily", dirs["state_dir"]):
        code = R.run_job("mlb-2026-daily", executor=spy, resolve=lambda r: "S", push=lambda *a: None, **dirs)
    assert code == 0 and not spy.calls
    assert json.loads((dirs["state_dir"] / "status" / "mlb-2026-daily.json").read_text())["skipped"] == "already running"
    assert run("mlb-2026-daily", None, dirs, spy)[0] == 0 and spy.calls  # and once the lock is free it runs


def test_different_jobs_do_not_block_each_other(dirs):
    with R.lock("mlb-2026-daily", dirs["state_dir"]):
        with R.lock("nfl-weekly-predict", dirs["state_dir"]):
            pass


def test_lock_is_released_on_exit(dirs):
    with R.lock("a", dirs["state_dir"]):
        pass
    with R.lock("a", dirs["state_dir"]):
        pass


def test_unknown_job_is_an_error(dirs):
    assert R.run_job("nope", **dirs) == 2


def test_a_failed_secret_read_in_run_job_is_recorded_and_exits_1(dirs):
    R.write_mode("cfb-weekly-predict", "live", dirs["jobs_dir"])
    spy = Spy()
    def boom(ref):
        raise RuntimeError("secret manager unreachable")
    code = R.run_job("cfb-weekly-predict", executor=spy, resolve=boom, push=lambda *a: None, **dirs)
    assert code == 1 and not spy.calls
    assert "environment: RuntimeError" in json.loads((dirs["state_dir"] / "status" / "cfb-weekly-predict.json").read_text())["error"]


def test_old_run_logs_are_pruned(dirs, monkeypatch):
    monkeypatch.setattr(R, "LOG_KEEP", 2)
    for i in range(4):
        rec = {"job": "j", "started": f"2026-10-0{i + 1}T04:00:00+00:00", "ok": True}
        R.record("j", rec, dirs["state_dir"], log_text=f"log {i}")
    logs = sorted((dirs["state_dir"] / "logs" / "j").glob("*.log"))
    assert [p.read_text() for p in logs] == ["log 2", "log 3"]


def test_summarize_never_returns_a_body_field():
    assert R.summarize({"status": "ok", "mode": "daily", "token": "t0ps3cret", "steps": [{"step": "a", "status": "ok", "api_key": "k"}]}) == \
        {"status": "ok", "mode": "daily", "steps": [{"step": "a", "status": "ok"}]}
    assert R.summarize("not a dict") == {}


def test_kuma_push_is_best_effort_and_needs_a_url(monkeypatch):
    monkeypatch.delenv("PIPELINE_KUMA_PUSH_URL", raising=False)
    R.kuma_push(True, "x")  # no URL: nothing happens
    R.kuma_push(False, "x", url="http://127.0.0.1:9/never-listening")  # unreachable: swallowed
