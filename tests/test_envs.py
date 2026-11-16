import os
import sys
import time
from pathlib import Path

import pytest

from mllab import envs as E
from mllab.jobs import DirectLauncher, JobSpec, LocalRunner

DAY = 86400


@pytest.fixture
def runner(tmp_path):
    return LocalRunner(tmp_path / "lab", launcher=DirectLauncher(), python_bin=sys.executable)


def make_env(runner, name, *, idle_days=None, size=1000):
    d = runner.envs / name
    (d / "bin").mkdir(parents=True)
    (d / "bin" / "python").write_bytes(b"x" * size)
    if idle_days is not None:
        stamp = d / ".last_used"; stamp.write_text("")
        t = time.time() - idle_days * DAY
        os.utime(stamp, (t, t))
    return d


def test_list_reports_size_last_use_and_untracked(runner):
    make_env(runner, "aaaa", idle_days=3, size=2048); make_env(runner, "bbbb")
    rows = {r["name"]: r for r in E.list_envs(runner)}
    assert rows["aaaa"]["tracked"] and 2.9 < rows["aaaa"]["idle_days"] < 3.1 and rows["aaaa"]["size_bytes"] == 2048
    assert not rows["bbbb"]["tracked"] and rows["bbbb"]["idle_days"] is None
    assert "untracked" in E.render(list(rows.values())) and E.render([]) == "no cached environments"


def test_prune_dry_run_deletes_nothing_then_real_prune_removes_only_old_tracked_envs(runner):
    make_env(runner, "old", idle_days=40); make_env(runner, "fresh", idle_days=2); make_env(runner, "untracked")
    assert [e["name"] for e in E.prune(runner, 30)] == ["old"] and (runner.envs / "old").exists()  # dry run by default
    assert [e["name"] for e in E.prune(runner, 30, dry_run=False)] == ["old"]
    assert not (runner.envs / "old").exists() and (runner.envs / "fresh").exists() and (runner.envs / "untracked").exists()


def test_untracked_envs_are_only_deleted_when_explicitly_asked(runner):
    make_env(runner, "untracked")
    assert E.prune(runner, 30, dry_run=False) == [] and (runner.envs / "untracked").exists()
    assert [e["name"] for e in E.prune(runner, 30, dry_run=False, include_untracked=True)] == ["untracked"]


def test_an_env_referenced_by_a_running_job_is_never_pruned(runner, tmp_path):
    make_env(runner, "busyenv1234", idle_days=90)
    code = tmp_path / "code"; code.mkdir(); (code / "s.py").write_text("import time; time.sleep(20)")
    jid = runner.submit(JobSpec(name="j", script="s.py", code_dir=str(code), timeout_s=60))
    (runner.jobs / jid / "run.sh").write_text((runner.jobs / jid / "run.sh").read_text() + "\n# uses busyenv1234\n")
    try:
        assert E.prune(runner, 30, dry_run=False) == [] and (runner.envs / "busyenv1234").exists()
    finally:
        runner.launcher.stop(jid); runner.launcher.wait(jid)


def test_prune_refuses_a_recent_cutoff(runner):
    with pytest.raises(E.EnvError, match="at least 1"):
        E.prune(runner, 0)


def test_a_symlink_cannot_make_prune_delete_outside_the_envs_dir(runner, tmp_path):
    outside = tmp_path / "precious"; outside.mkdir(); (outside / "data").write_text("keep")
    runner.envs.mkdir(parents=True, exist_ok=True)
    link = runner.envs / "sneaky"; link.symlink_to(outside, target_is_directory=True)
    stamp = outside / ".last_used"; stamp.write_text(""); os.utime(stamp, (time.time() - 90 * DAY,) * 2)
    with pytest.raises(E.EnvError, match="refusing to delete"):
        E.prune(runner, 30, dry_run=False)
    assert (outside / "data").read_text() == "keep"


def test_jobs_stamp_the_env_when_they_use_it(runner, tmp_path):
    lock = tmp_path / "code"; lock.mkdir()
    (lock / "s.py").write_text("print('ok')"); (lock / "requirements.lock").write_text("\n")
    jid = runner.submit(JobSpec(name="j", script="s.py", code_dir=str(lock), timeout_s=60, lock="requirements.lock"))
    text = (runner.jobs / jid / "run.sh").read_text()
    assert 'touch "$ENV/.last_used"' in text and text.index("uv venv") < text.index("touch")  # stamped after the env exists
    runner.launcher.stop(jid)
