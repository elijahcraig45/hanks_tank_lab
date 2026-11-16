"""Local job runner.

A job = a copy of a code dir + a script, run detached with hard memory/cpu/time limits.
Layout: <root>/jobs/<id>/{spec.json,code/,run.sh,log.txt,status.json,outputs/}

Launching goes through a Launcher so the same runner works two ways:
  - SpoolLauncher (homelab): drops a request file that a root-owned systemd path unit validates and
    starts as a transient unit inside mllab.slice. Needed because the services run with NoNewPrivileges,
    so sudo is unavailable from Jupyter.
  - DirectLauncher (tests / dev): plain subprocess.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

ID_RE = re.compile(r"^[a-z0-9-]{8,64}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class JobError(Exception):
    pass


@dataclass
class HostCaps:
    max_mem_mb: int = 4096
    max_cpu_pct: int = 200
    max_timeout_s: int = 6 * 3600
    max_concurrent: int = 1


@dataclass
class JobSpec:
    name: str
    script: str
    code_dir: str
    timeout_s: int  # required: no unbounded jobs
    args: list[str] = field(default_factory=list)
    lock: str | None = None  # requirements lock file; env is built once per lock hash and reused
    mem_mb: int = 2048
    cpu_pct: int = 100
    inputs: list[str] = field(default_factory=list)  # files under data_root, symlinked read-only into inputs/
    outputs: str = "outputs"  # dir under the job's code dir that is kept
    notebook: bool = False  # run via papermill instead of python

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "JobSpec":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if "timeout_s" not in raw:
            raise JobError("spec must set timeout_s")
        base = Path(path).resolve().parent
        code = Path(raw.get("code_dir", "."))
        raw["code_dir"] = str(code if code.is_absolute() else (base / code).resolve())
        try:
            return cls(**raw)
        except TypeError as e:
            raise JobError(f"bad spec: {e}") from e

    def validate(self, caps: HostCaps, code_root_ok=None) -> None:
        if not NAME_RE.match(self.name):
            raise JobError(f"invalid job name {self.name!r}")
        s = Path(self.script)
        if s.is_absolute() or ".." in s.parts or self.script.startswith("-"):
            raise JobError(f"script must be a relative path inside code_dir: {self.script!r}")
        if not Path(self.code_dir).is_dir():
            raise JobError(f"code_dir not found: {self.code_dir}")
        if not (Path(self.code_dir) / self.script).is_file():
            raise JobError(f"script not found in code_dir: {self.script}")
        if not 1 <= self.timeout_s <= caps.max_timeout_s:
            raise JobError(f"timeout_s must be 1..{caps.max_timeout_s}")
        if not 64 <= self.mem_mb <= caps.max_mem_mb:
            raise JobError(f"mem_mb must be 64..{caps.max_mem_mb}")
        if not 10 <= self.cpu_pct <= caps.max_cpu_pct:
            raise JobError(f"cpu_pct must be 10..{caps.max_cpu_pct}")
        Path(self.outputs)  # type check only
        if Path(self.outputs).is_absolute() or ".." in Path(self.outputs).parts:
            raise JobError("outputs must be a relative dir")


# --- launchers ---------------------------------------------------------------------------------

class DirectLauncher:
    """Dev/test launcher: run run.sh as a plain child process (no limits enforced)."""

    def __init__(self):
        self._procs: dict[str, subprocess.Popen] = {}

    def start(self, job_id: str, job_dir: Path, spec: JobSpec) -> None:
        self._procs[job_id] = subprocess.Popen(["bash", str(job_dir / "run.sh")], start_new_session=True,
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def is_active(self, job_id: str) -> bool:
        p = self._procs.get(job_id)
        return p is not None and p.poll() is None

    def stop(self, job_id: str) -> None:
        p = self._procs.get(job_id)
        if p and p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)  # like systemd: signal the whole group, not just bash

    def wait(self, job_id: str, timeout: float = 30) -> None:
        self._procs[job_id].wait(timeout)


class SpoolLauncher:
    """Production launcher: request files in <root>/spool, consumed by mllab-spool.service (root)."""

    def __init__(self, root: Path):
        self.spool = root / "spool"

    def _write(self, name: str, body: str) -> None:
        self.spool.mkdir(parents=True, exist_ok=True)
        tmp = self.spool / f".{name}.tmp"
        tmp.write_text(body)
        os.replace(tmp, self.spool / name)  # atomic, so the path unit never sees a half-written file

    def start(self, job_id: str, job_dir: Path, spec: JobSpec) -> None:
        self._write(f"{job_id}.start", f"{spec.mem_mb} {spec.cpu_pct} {spec.timeout_s}\n")

    def stop(self, job_id: str) -> None:
        self._write(f"{job_id}.stop", "stop\n")

    def is_active(self, job_id: str) -> bool:
        return subprocess.run(["systemctl", "is-active", "--quiet", f"mllab-job-{job_id}.service"]).returncode == 0


# --- runner ------------------------------------------------------------------------------------

def _run_sh(job_dir: Path, spec: JobSpec, envs_root: Path, python_bin: str) -> str:
    q = shlex.quote
    if spec.lock:
        lock_hash = hashlib.sha256((Path(spec.code_dir) / spec.lock).read_bytes()).hexdigest()[:16]
        env_dir = envs_root / lock_hash
        setup = (
            f'ENV={q(str(env_dir))}\n'
            f'if [ ! -x "$ENV/bin/python" ]; then\n'
            f'  uv venv --python 3.12 "$ENV" && uv pip install --python "$ENV/bin/python" -r {q(spec.lock)} || exit 97\n'
            f'fi\ntouch "$ENV/.last_used"\nPY="$ENV/bin/python"\n'
        )
    else:
        setup = f"PY={q(python_bin)}\n"
    if spec.notebook:
        cmd = f'"$PY" -m papermill {q(spec.script)} {q("outputs/" + Path(spec.script).name)} ' + " ".join(map(q, spec.args))
    else:
        cmd = f'"$PY" -u {q(spec.script)} ' + " ".join(map(q, spec.args))
    return f"""#!/usr/bin/env bash
set -uo pipefail
trap 'true' TERM   # a handler, not "ignore": systemd's stop/timeout SIGTERM reaches us and the child, and we still record it
JOB={q(str(job_dir))}
cd "$JOB/code"
mkdir -p {q(spec.outputs)}
echo '{{"state":"running","started_at":"'"$(date -u +%FT%TZ)"'"}}' > "$JOB/started.json"
(
{setup}exec {cmd}
) >> "$JOB/log.txt" 2>&1
RC=$?
STATE=failed; [ $RC -eq 0 ] && STATE=finished; [ $RC -eq 143 ] && STATE=terminated
printf '{{"state":"%s","exit_code":%s,"finished_at":"%s"}}\\n' "$STATE" "$RC" "$(date -u +%FT%TZ)" > "$JOB/status.json.tmp"
mv "$JOB/status.json.tmp" "$JOB/status.json"
exit $RC
"""


class LocalRunner:
    def __init__(self, root: str | os.PathLike, launcher=None, caps: HostCaps | None = None,
                 data_root: str | os.PathLike | None = None, python_bin: str = "python3"):
        self.root = Path(root)
        self.jobs = self.root / "jobs"
        self.envs = self.root / "envs"
        self.data_root = Path(data_root) if data_root else self.root / "data"
        self.launcher = launcher or SpoolLauncher(self.root)
        self.caps = caps or HostCaps()
        self.python_bin = python_bin

    def _dir(self, job_id: str) -> Path:
        if not ID_RE.match(job_id):
            raise JobError(f"invalid job id {job_id!r}")
        return self.jobs / job_id

    def active_ids(self) -> list[str]:
        if not self.jobs.is_dir():
            return []
        return [p.name for p in self.jobs.iterdir()
                if ID_RE.match(p.name) and self.status(p.name)["state"] in ("queued", "running")]

    def submit(self, spec: JobSpec) -> str:
        spec.validate(self.caps)
        code = Path(spec.code_dir).resolve()
        if code == self.root.resolve() or code in self.root.resolve().parents:
            # copytree would recurse into the very directory it is filling (jobs/<id>/code/lab/jobs/...) until the disk is full
            raise JobError(f"code_dir {code} contains the lab's own directory {self.root}; use a dedicated code directory")
        spec.validate(self.caps)
        if len(self.active_ids()) >= self.caps.max_concurrent:
            raise JobError(f"host is at max_concurrent={self.caps.max_concurrent}; wait or cancel a job")
        job_id = f"{time.strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(3)}"
        jd = self._dir(job_id)
        try:
            self._materialize(spec, job_id, jd)
        except BaseException:
            shutil.rmtree(jd, ignore_errors=True)  # never leave a half-built job that counts as active
            raise
        self.launcher.start(job_id, jd, spec)
        return job_id

    def _materialize(self, spec: JobSpec, job_id: str, jd: Path) -> None:
        inputs = []
        for rel in spec.inputs:  # validate everything before touching disk
            src = (self.data_root / rel).resolve()
            if self.data_root.resolve() not in src.parents or not src.is_file():
                raise JobError(f"input must be an existing file under {self.data_root}: {rel!r}")
            inputs.append(src)
        jd.mkdir(parents=True)
        shutil.copytree(spec.code_dir, jd / "code", symlinks=False,
                        ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", "*.pyc", ".ipynb_checkpoints"))
        (jd / "inputs").mkdir()
        for src in inputs:
            os.symlink(src, jd / "inputs" / src.name)
        (jd / "spec.json").write_text(json.dumps({**asdict(spec), "id": job_id,
                                                  "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2))
        (jd / "log.txt").touch()
        (jd / "run.sh").write_text(_run_sh(jd, spec, self.envs, self.python_bin))
        (jd / "run.sh").chmod(0o750)

    def status(self, job_id: str) -> dict:
        jd = self._dir(job_id)
        if not jd.is_dir():
            raise JobError(f"unknown job {job_id}")
        done = jd / "status.json"
        if done.exists():
            return json.loads(done.read_text())
        if self.launcher.is_active(job_id):
            return {"state": "running"} if (jd / "started.json").exists() else {"state": "queued"}
        if (jd / "started.json").exists():
            return {"state": "lost", "detail": "killed without a status: almost certainly an OOM kill (a timeout or cancel is recorded as terminated)"}
        try:
            submitted = json.loads((jd / "spec.json").read_text()).get("submitted_at")
        except (OSError, ValueError):
            return {"state": "lost", "detail": "incomplete job directory"}
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(submitted)).total_seconds() if submitted else 0
        # the spool unit picks requests up within seconds; a long wait means it never started
        return {"state": "queued"} if age < 120 else {"state": "lost", "detail": "never started (spool unit not running?)"}

    def logs(self, job_id: str, offset: int = 0) -> tuple[str, int]:
        with open(self._dir(job_id) / "log.txt", "rb") as f:
            f.seek(offset)
            data = f.read()
        return data.decode(errors="replace"), offset + len(data)

    def cancel(self, job_id: str) -> None:
        self._dir(job_id)
        self.launcher.stop(job_id)

    def outputs(self, job_id: str) -> list[Path]:
        spec = json.loads((self._dir(job_id) / "spec.json").read_text())
        out = self._dir(job_id) / "code" / spec["outputs"]
        return sorted(p for p in out.rglob("*") if p.is_file()) if out.is_dir() else []
