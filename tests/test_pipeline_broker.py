"""The broker is the only code that acts on the app's behalf as a privileged user, so its rules are pinned here: what a request may name, what it must
confirm, and that a hostile request file is refused rather than followed."""
import getpass
import json
import os

import pytest

from mllab.pipeline import broker as B, runner
from mllab.pipeline.jobs import JOBS, PREGAME

ME = getpass.getuser()


def req(**kw):
    base = {"id": "abcd1234-0000", "action": "set_mode", "job": "nfl-weekly-predict", "mode": "dry", "actor": "me@example.com"}
    base.update(kw)
    if base["action"] == "run":
        base.pop("mode", None)  # a run names no mode; the strict key check is what rejects a stray one
    return json.dumps(base).encode()


@pytest.fixture
def host(tmp_path, monkeypatch):
    for n in ("jobs", "spool", "results", "state", "status"):
        (tmp_path / n).mkdir()
    monkeypatch.setattr(B, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(B, "AUDIT", tmp_path / "audit.log")
    monkeypatch.setattr(B.shutil, "chown", lambda *a, **k: None)  # root:pipeline does not exist on a laptop
    return tmp_path


def drop(host, rid="abcd1234-0000", raw=None, name=None):
    p = host / "spool" / (name or f"pipe-{rid}.json")
    p.write_bytes(raw if raw is not None else req(id=rid))
    return p


class Runs:
    def __init__(self, rc=0):
        self.calls, self.rc = [], rc

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        return type("R", (), {"returncode": self.rc, "stdout": "", "stderr": "boom"})()


def apply(host, run=None, owner=ME):
    return B.cmd_apply(spool=host / "spool", owner=owner, jobs_dir=host / "jobs", run=run or Runs(), log=lambda *_: None)


def result(host, rid="abcd1234-0000"):
    return json.loads((host / "results" / f"{rid}.json").read_text())


# ------------------------------------------------------------------------------------------------------------------------- validation
def test_a_valid_mode_change_is_accepted():
    assert B.validate(req(), current_mode=lambda j: "off")["mode"] == "dry"


@pytest.mark.parametrize("raw,why", [
    (b"not json", "JSON"),
    (req(action="rm -rf"), "action"),
    (req(job="../../etc/passwd"), "unknown job"),
    (req(job="nfl-weekly-predict; reboot"), "unknown job"),
    (req(mode="LIVE"), "mode"),
    (req(mode="live", confirm="nope"), "typed back"),
    (req(extra="x"), "exactly"),
    (b"x" * 5000, "large"),
    (req(job="mlb-2026-weekly-train-v10", mode="dry"), "paused"),
])
def test_bad_requests_are_refused(raw, why):
    with pytest.raises(B.Rejected, match=why):
        B.validate(raw, current_mode=lambda j: "off")


def test_going_live_needs_the_job_name_typed_back():
    with pytest.raises(B.Rejected):
        B.validate(req(mode="live"), current_mode=lambda j: "off")
    assert B.validate(req(mode="live", confirm="nfl-weekly-predict"), current_mode=lambda j: "off")["mode"] == "live"


def test_staying_live_does_not_ask_again():
    assert B.validate(req(mode="live"), current_mode=lambda j: "live")["mode"] == "live"


def test_pregame_is_a_known_target_for_modes_but_not_for_run():
    assert B.validate(req(job=PREGAME, mode="dry"), current_mode=lambda j: "off")["job"] == PREGAME
    with pytest.raises(B.Rejected, match="unknown job"):
        B.validate(req(action="run", job=PREGAME, dry=True), current_mode=lambda j: "off")


def test_a_real_run_needs_confirmation_and_a_dry_run_does_not():
    assert B.validate(req(action="run", job="nfl-weekly-predict", dry=True), current_mode=lambda j: "off")["dry"] is True
    with pytest.raises(B.Rejected, match="typed back"):
        B.validate(req(action="run", job="nfl-weekly-predict", dry=False), current_mode=lambda j: "off")
    assert B.validate(req(action="run", job="nfl-weekly-predict", dry=False, confirm="nfl-weekly-predict"), current_mode=lambda j: "off")
    with pytest.raises(B.Rejected, match="true or false"):
        B.validate(req(action="run", job="nfl-weekly-predict", dry="false", confirm="nfl-weekly-predict"), current_mode=lambda j: "off")


def test_run_argv_names_only_checked_values_and_never_a_shell():
    r = B.validate(req(action="run", job="nfl-weekly-predict", dry=True), current_mode=lambda j: "off")
    argv = B.run_argv(r)
    assert argv[0] == "systemd-run" and "--uid=pipeline" in argv and "MemoryMax=3G" in argv
    assert argv[argv.index("--") + 1:] == [B.PY, "-m", "mllab.pipeline.ctl", "run", "nfl-weekly-predict", "--dry", "--yes"]
    assert not any(a in ("sh", "bash", "-c") for a in argv)


def test_real_run_drops_the_dry_flag():
    r = B.validate(req(action="run", job="nfl-weekly-predict", dry=False, confirm="nfl-weekly-predict"), current_mode=lambda j: "off")
    assert "--dry" not in B.run_argv(r)


# ----------------------------------------------------------------------------------------------------------------------------- apply
def test_apply_changes_a_mode_records_a_result_and_an_audit_line(host):
    drop(host)
    assert apply(host) == 0
    assert runner.read_mode("nfl-weekly-predict", host / "jobs") == "dry"
    assert result(host)["status"] == "done"
    assert not list((host / "spool").glob("pipe-*"))  # consumed
    line = json.loads((host / "audit.log").read_text().splitlines()[-1])
    assert line["action"] == "pipeline_set_mode" and line["by"] == "me@example.com"
    assert line["before"] == "off" and line["mode"] == "dry"   # structured: the app's rollback reads these


def test_apply_starts_a_run_through_systemd_run(host):
    runs = Runs()
    drop(host, raw=req(action="run", job="cfb-weekly-predict", dry=True))
    apply(host, runs)
    assert len(runs.calls) == 1 and "cfb-weekly-predict" in runs.calls[0]
    assert result(host)["status"] == "done"


def test_a_failed_systemd_run_is_reported_not_swallowed(host):
    drop(host, raw=req(action="run", job="cfb-weekly-predict", dry=True))
    apply(host, Runs(rc=1))
    assert result(host)["status"] == "rejected" and "boom" in result(host)["detail"]


def test_a_rejected_request_changes_nothing_and_says_why(host):
    drop(host, raw=req(mode="live"))
    apply(host)
    assert runner.read_mode("nfl-weekly-predict", host / "jobs") == "off"
    r = result(host)
    assert r["status"] == "rejected" and "typed back" in r["detail"]
    assert json.loads((host / "audit.log").read_text().splitlines()[-1])["action"] == "pipeline_rejected"


def test_a_symlink_is_never_followed(host):
    target = host / "elsewhere.json"
    target.write_bytes(req(id="abcd1234-0001"))
    (host / "spool" / "pipe-abcd1234-0001.json").symlink_to(target)
    apply(host)
    assert runner.read_mode("nfl-weekly-predict", host / "jobs") == "off"
    assert result(host, "abcd1234-0001")["status"] == "rejected"
    assert target.exists()  # only the link was removed


def test_a_request_from_another_user_is_refused(host):
    drop(host)
    apply(host, owner="root" if ME != "root" else "nobody")
    assert runner.read_mode("nfl-weekly-predict", host / "jobs") == "off"


def test_the_id_inside_must_match_the_file_name(host):
    drop(host, rid="abcd1234-0002", raw=req(id="zzzz9999-0000"))
    apply(host)
    assert result(host, "abcd1234-0002")["status"] == "rejected"
    assert runner.read_mode("nfl-weekly-predict", host / "jobs") == "off"


def test_stray_files_in_the_spool_are_ignored(host):
    (host / "spool" / "other.start").write_text("1 2 3")
    drop(host, name="pipe-..%2f.json", raw=b"{}")
    apply(host)
    assert (host / "spool" / "other.start").exists()  # not ours


def test_apply_with_no_such_app_user_refuses_everything(host):
    drop(host)
    assert apply(host, owner="no-such-user-xyz") == 1
    assert runner.read_mode("nfl-weekly-predict", host / "jobs") == "off"


# -------------------------------------------------------------------------------------------------------------------------- snapshot
def test_snapshot_lists_every_job_with_its_mode_and_last_run(host):
    runner.write_mode("nfl-weekly-predict", "dry", host / "jobs")
    runner.record("nfl-weekly-predict", {"started": "2026-10-03T06:00:00+00:00", "finished": "2026-10-03T06:01:00+00:00", "ok": True, "seconds": 60},
                  host / "state", log_text="hello log")
    state, logs = B.build_snapshot(jobs_dir=host / "jobs", state_dir=host / "state", release=host / "none",
                                   show=lambda unit: {"NextElapseUSecRealtime": "Sat soon"})
    by = {j["name"]: j for j in state["jobs"]}
    assert set(by) == set(JOBS) and by["nfl-weekly-predict"]["mode"] == "dry" and by["nfl-weekly-predict"]["last"]["ok"] is True
    assert by["nfl-weekly-predict"]["next_run"] == "Sat soon" and by["mlb-2026-weekly-train-v10"]["next_run"] is None  # a paused job has no timer
    assert by["mlb-2026-daily"]["mode"] == "off"  # a job with no mode file is off
    assert state["pregame"]["mode"] == "off" and state["release"] is None
    assert "hello log" in logs["nfl-weekly-predict"]


def test_snapshot_files_are_private_and_complete(host):
    assert B.cmd_snapshot(jobs_dir=host / "jobs", state_dir=host / "state", release=host / "none", show=lambda u: {}, status_dir=host / "status") == 0
    assert json.loads((host / "status" / "state.json").read_text())["jobs"]
    assert oct(os.stat(host / "status" / "state.json").st_mode & 0o777) == "0o640"
    assert not list((host / "status").rglob(".*tmp"))
