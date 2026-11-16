import base64
import json
from types import SimpleNamespace

import pytest

from mllab import scheduler as S
from mllab.scheduler import Scheduler, SchedulerError


def b64(obj):
    raw = obj if isinstance(obj, (bytes, str)) else json.dumps(obj)
    return base64.b64encode(raw.encode() if isinstance(raw, str) else raw).decode()


JOBS = [
    {"name": "projects/hankstank/locations/us-central1/jobs/mlb-2026-daily", "state": "ENABLED", "schedule": "0 4 * 3-11 *", "timeZone": "America/New_York",
     "httpTarget": {"body": b64({"mode": "daily"})}, "lastAttemptTime": "2026-09-29T08:00:00Z", "scheduleTime": "2026-09-30T08:00:00Z"},
    {"name": "projects/hankstank/locations/us-central1/jobs/mlb-2026-weekly-predict", "state": "PAUSED", "schedule": "0 5 * 3-11 5", "timeZone": "America/New_York",
     "httpTarget": {"body": b64({"mode": "predict"})}, "lastAttemptTime": "2026-09-25T09:00:00Z", "status": {"code": 13}},
    {"name": "projects/hankstank/locations/us-central1/jobs/broken", "state": "ENABLED", "schedule": "0 1 * * *", "httpTarget": {"body": b64('{mode:daily}')}},
]


@pytest.fixture(autouse=True)
def _audit(tmp_path, monkeypatch):
    monkeypatch.setenv("MLLAB_AUDIT", str(tmp_path / "audit.log"))


class Gcloud:
    def __init__(self, jobs=JOBS, code=0):
        self.calls, self.jobs, self.code = [], jobs, code

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        out = json.dumps(self.jobs) if "list" in cmd else ""
        return SimpleNamespace(returncode=self.code, stdout=out, stderr="denied" if self.code else "")


@pytest.mark.parametrize("body, want", [
    (b64({"mode": "daily"}), ({"mode": "daily"}, True)), (None, ({}, True)), ("", ({}, True)),
    (b64("{mode:daily}"), (None, False)),            # the unquoted-body trap: NOT valid JSON
    (b64("[1,2]"), (None, False)), ("%%%not base64", (None, False)), (b64(b"\xff\xfe"), (None, False))])
def test_decode_body(body, want):
    assert S.decode_body(body) == want


def test_parse_jobs_surfaces_state_mode_and_the_invalid_body_trap():
    jobs = {j["name"]: j for j in S.parse_jobs(JOBS)}
    assert jobs["mlb-2026-daily"]["mode"] == "daily" and jobs["mlb-2026-daily"]["body_ok"] and jobs["mlb-2026-daily"]["last_attempt_ok"] is True
    assert jobs["mlb-2026-weekly-predict"]["state"] == "PAUSED" and jobs["mlb-2026-weekly-predict"]["last_attempt_ok"] is False
    assert jobs["broken"]["body_ok"] is False and jobs["broken"]["mode"] is None and jobs["broken"]["last_attempt_ok"] is None
    assert list(jobs) == sorted(jobs)
    assert "INVALID JSON" in S.render(list(jobs.values()))


def test_pause_needs_a_known_job_and_the_typed_name():
    g = Gcloud(); s = Scheduler(runner=g)
    with pytest.raises(SchedulerError, match="confirm by typing"):
        s.pause("mlb-2026-daily")
    with pytest.raises(SchedulerError, match="no such job"):
        s.pause("nope", confirm="nope")
    r = s.pause("mlb-2026-daily", confirm="mlb-2026-daily")
    assert r["changed"] and g.calls[-1][0][:4] == ["gcloud", "scheduler", "jobs", "pause"] and "mlb-2026-daily" in g.calls[-1][0]
    assert g.calls[-1][1]["env"]["CLOUDSDK_CORE_ACCOUNT"] == "you@example.com"


def test_already_in_the_requested_state_is_a_noop():
    g = Gcloud(); s = Scheduler(runner=g)
    assert not s.pause("mlb-2026-weekly-predict", confirm="mlb-2026-weekly-predict")["changed"]
    assert not any("pause" in c[0] for c in g.calls)


def test_hostile_job_names_never_reach_a_command():
    g = Gcloud(); s = Scheduler(runner=g)
    for bad in ("x; rm -rf /", "a b", "../x", "$(id)", "", "-rf"):
        with pytest.raises(SchedulerError, match="invalid job name"):
            s.pause(bad, confirm=bad)
    assert len(g.calls) == 0  # not even the list call: rejected before anything ran


def test_resume_and_run_refuse_a_job_whose_body_would_silently_run_daily(tmp_path):
    g = Gcloud(); s = Scheduler(runner=g)
    with pytest.raises(SchedulerError, match="not valid JSON"):
        s.resume("broken", confirm="broken")
    with pytest.raises(SchedulerError, match="not valid JSON"):
        s.run_now("broken", confirm="broken")
    assert not any(c[0][4] in ("resume", "run") for c in g.calls if len(c[0]) > 4)


def test_run_now_and_resume_flow_and_audit(tmp_path):
    g = Gcloud(); s = Scheduler(runner=g); ap = tmp_path / "a.log"
    assert s.resume("mlb-2026-weekly-predict", confirm="mlb-2026-weekly-predict", audit_path=ap)["changed"]
    assert s.run_now("mlb-2026-daily", confirm="mlb-2026-daily", audit_path=ap)["detail"].startswith("run mlb-2026-daily (mode daily)")
    text = ap.read_text()
    assert "scheduler_resume" in text and "scheduler_run" in text
    with pytest.raises(SchedulerError, match="confirm by typing"):
        s.run_now("mlb-2026-daily")


def test_dry_run_changes_nothing():
    g = Gcloud(); s = Scheduler(runner=g)
    r = s.pause("mlb-2026-daily", confirm="mlb-2026-daily", dry_run=True)
    assert r["dry_run"] and not r["changed"] and all("pause" not in c[0] for c in g.calls)


def test_gcloud_failure_and_work_account():
    with pytest.raises(SchedulerError, match="gcloud failed"):
        Scheduler(runner=Gcloud(code=1)).list()
    with pytest.raises(SchedulerError, match="work account"):
        Scheduler(account="someone@blocked.example")


def test_a_failed_audit_write_does_not_make_a_done_change_look_failed(tmp_path, capsys):
    blocked = tmp_path / "file-not-dir"
    blocked.write_text("x")                       # a file where the log's directory should be: mkdir raises, as on a laptop
    g = Gcloud(); s = Scheduler(runner=g)
    r = s.pause("mlb-2026-daily", confirm="mlb-2026-daily", audit_path=blocked / "deploys.log")
    assert r["changed"] is True and g.calls[-1][0][3] == "pause"          # the pause happened and is reported as done
    err = capsys.readouterr().err
    assert "WAS DONE" in err and "audit log could not be written" in err and "MLLAB_AUDIT" in err
