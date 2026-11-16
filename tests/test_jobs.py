import getpass
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from mllab.jobs import DirectLauncher, HostCaps, JobError, JobSpec, LocalRunner, SpoolLauncher

JOBCTL = Path(__file__).resolve().parents[1] / "host" / "mllab-jobctl"


@pytest.fixture
def code(tmp_path):
    d = tmp_path / "code"
    d.mkdir()
    (d / "train.py").write_text(textwrap.dedent("""
        import sys, pathlib
        pathlib.Path("outputs").mkdir(exist_ok=True)
        pathlib.Path("outputs/result.txt").write_text("args=" + ",".join(sys.argv[1:]))
        print("hello from job")
        sys.exit(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0)
    """))
    return d


def runner(tmp_path, **kw):
    return LocalRunner(tmp_path / "lab", launcher=DirectLauncher(), python_bin=sys.executable, **kw)


def spec(code, **kw):
    return JobSpec(name="t", script="train.py", code_dir=str(code), timeout_s=60, **kw)


def test_job_runs_streams_logs_and_returns_outputs(tmp_path, code):
    r = runner(tmp_path)
    jid = r.submit(spec(code, args=["abc"]))
    r.launcher.wait(jid)
    assert r.status(jid)["state"] == "finished"
    text, off = r.logs(jid)
    assert "hello from job" in text
    assert r.logs(jid, off) == ("", off)  # offset-based tailing returns only new bytes
    assert [p.name for p in r.outputs(jid)] == ["result.txt"]
    assert (r.outputs(jid)[0]).read_text() == "args=abc"


def test_failure_records_exit_code(tmp_path, code):
    r = runner(tmp_path)
    jid = r.submit(spec(code, args=["3"]))
    r.launcher.wait(jid)
    st = r.status(jid)
    assert st["state"] == "failed" and st["exit_code"] == 3


def test_concurrency_limit(tmp_path, code):
    (code / "slow.py").write_text("import time; time.sleep(5)")
    r = runner(tmp_path)
    jid = r.submit(JobSpec(name="s", script="slow.py", code_dir=str(code), timeout_s=60))
    with pytest.raises(JobError, match="max_concurrent"):
        r.submit(spec(code))
    r.cancel(jid); r.launcher.stop(jid)


def test_cancel_is_recorded_as_terminated_not_lost(tmp_path, code):
    (code / "slow.py").write_text("import time; print('up', flush=True); time.sleep(30)")
    r = runner(tmp_path)
    jid = r.submit(JobSpec(name="s", script="slow.py", code_dir=str(code), timeout_s=60))
    for _ in range(50):  # wait until it is really running
        if "up" in r.logs(jid)[0]:
            break
        __import__("time").sleep(0.1)
    r.launcher.stop(jid)
    r.launcher.wait(jid)
    st = r.status(jid)
    assert st["state"] == "terminated" and st["exit_code"] == 143


def test_env_dir_is_reused_per_lock_hash_not_per_job(tmp_path, code):
    (code / "requirements.lock").write_text("numpy==1.0\n")
    r = runner(tmp_path)
    jid = r.submit(spec(code, lock="requirements.lock"))
    sh = (r.jobs / jid / "run.sh").read_text()
    assert str(r.envs) in sh and "uv venv" in sh and jid not in sh.split("ENV=")[1].split("\n")[0]


@pytest.mark.parametrize("kw, msg", [
    (dict(script="../evil.py"), "relative path"),
    (dict(script="/etc/passwd"), "relative path"),
    (dict(script="-rf"), "relative path"),
    (dict(timeout_s=0), "timeout_s"),
    (dict(timeout_s=10**7), "timeout_s"),
    (dict(mem_mb=10**6), "mem_mb"),
    (dict(cpu_pct=900), "cpu_pct"),
    (dict(name="bad name!"), "invalid job name"),
    (dict(outputs="../up"), "outputs"),
])
def test_spec_validation(tmp_path, code, kw, msg):
    base = dict(name="t", script="train.py", code_dir=str(code), timeout_s=60)
    with pytest.raises(JobError, match=msg):
        LocalRunner(tmp_path, launcher=DirectLauncher()).submit(JobSpec(**{**base, **kw}))


def test_yaml_requires_timeout(tmp_path, code):
    p = tmp_path / "job.yaml"
    p.write_text("name: t\nscript: train.py\ncode_dir: code\n")
    with pytest.raises(JobError, match="timeout_s"):
        JobSpec.from_yaml(p)
    p.write_text("name: t\nscript: train.py\ncode_dir: code\ntimeout_s: 30\n")
    assert JobSpec.from_yaml(p).code_dir == str(code)


def test_inputs_must_live_under_data_root(tmp_path, code):
    r = runner(tmp_path)
    (r.data_root).mkdir(parents=True)
    (r.data_root / "ok.csv").write_text("a\n1\n")
    outside = tmp_path / "secret.txt"; outside.write_text("x")
    with pytest.raises(JobError, match="under"):
        r.submit(spec(code, inputs=["../secret.txt"]))
    jid = r.submit(spec(code, inputs=["ok.csv"]))
    assert (r.jobs / jid / "inputs" / "ok.csv").is_symlink()


def test_bad_job_id_rejected(tmp_path):
    with pytest.raises(JobError, match="invalid job id"):
        runner(tmp_path).status("../../etc")


def test_spool_launcher_writes_atomic_request(tmp_path, code):
    r = LocalRunner(tmp_path / "lab", launcher=SpoolLauncher(tmp_path / "lab"), python_bin=sys.executable)
    jid = r.submit(spec(code, mem_mb=512, cpu_pct=150))
    req = tmp_path / "lab" / "spool" / f"{jid}.start"
    assert req.read_text().split() == ["512", "150", "60"]
    assert not list((tmp_path / "lab" / "spool").glob(".*tmp"))
    r.cancel(jid)
    assert (tmp_path / "lab" / "spool" / f"{jid}.stop").exists()


# --- root-side validator, exercised in DRY_RUN mode ---------------------------------------------

def jobctl(root, files):
    spool = root / "spool"; spool.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (spool / name).write_text(body)
    env = {**os.environ, "MLLAB_ROOT": str(root), "MLLAB_USER": getpass.getuser(), "DRY_RUN": "1"}
    out = subprocess.run(["bash", str(JOBCTL)], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout, spool


def make_job(root, jid):
    d = root / "jobs" / jid; d.mkdir(parents=True)
    (d / "run.sh").write_text("#!/bin/bash\n")


GOOD = "20260929101010-abc123"


def test_jobctl_starts_valid_job_with_limits_and_never_as_root(tmp_path):
    make_job(tmp_path, GOOD)
    out, spool = jobctl(tmp_path, {f"{GOOD}.start": "1024 100 600\n"})
    assert "DRY systemd-run" in out and f"--unit=mllab-job-{GOOD}" in out
    assert "--slice=mllab.slice" in out and "MemoryMax=1024M" in out and "RuntimeMaxSec=600" in out
    assert f"--uid={getpass.getuser()}" in out and "NoNewPrivileges=yes" in out
    assert not list(spool.glob("*.start"))  # consumed


@pytest.mark.parametrize("body", [
    "99999 100 600", "1024 900 600", "1024 100 99999999", "1024 100 600 extra", "abc 1 2",
    "1024; touch /tmp/pwn 100 600", "$(id) 100 600", "10 100 600", "",
])
def test_jobctl_rejects_hostile_or_out_of_range_limits(tmp_path, body):
    make_job(tmp_path, GOOD)
    out, spool = jobctl(tmp_path, {f"{GOOD}.start": body + "\n"})
    assert "DRY systemd-run" not in out
    assert not list(spool.glob("*.start"))


@pytest.mark.parametrize("jid", ["short", "../../etc/passwd-x", "UPPERCASE-12345678", "a b c d e f g h"])
def test_jobctl_rejects_bad_ids(tmp_path, jid):
    out, _ = jobctl(tmp_path, {f"{jid}.start": "1024 100 600\n"}) if "/" not in jid else ("", None)
    assert "DRY systemd-run" not in out


def test_jobctl_rejects_missing_job_dir_and_symlinked_dir(tmp_path):
    out, _ = jobctl(tmp_path, {f"{GOOD}.start": "1024 100 600\n"})
    assert "DRY systemd-run" not in out
    real = tmp_path / "elsewhere"; real.mkdir(); (real / "run.sh").write_text("x")
    (tmp_path / "jobs").mkdir(exist_ok=True)
    os.symlink(real, tmp_path / "jobs" / GOOD)
    out, _ = jobctl(tmp_path, {f"{GOOD}.start": "1024 100 600\n"})
    assert "DRY systemd-run" not in out


def test_jobctl_stop(tmp_path):
    make_job(tmp_path, GOOD)
    out, _ = jobctl(tmp_path, {f"{GOOD}.stop": "stop\n"})
    assert f"DRY systemctl stop mllab-job-{GOOD}.service" in out


def test_code_dir_containing_the_lab_is_refused(tmp_path):
    """Regression: copytree of a dir that contains jobs/ recursed into itself until the path name overflowed."""
    (tmp_path / "s.py").write_text("print(1)")
    r = LocalRunner(tmp_path / "lab", launcher=DirectLauncher())
    with pytest.raises(JobError, match="contains the lab's own directory"):
        r.submit(JobSpec(name="x", script="s.py", code_dir=str(tmp_path), timeout_s=30))
    assert not (tmp_path / "lab" / "jobs").exists() or not list((tmp_path / "lab" / "jobs").iterdir())
