"""The broker between the mllab web app and the pipeline.

The web app runs as the unprivileged `mllab` user and must not be able to read or write the pipeline's own directories (/etc/pipeline, /var/lib/pipeline
belong to the `pipeline` group): it is the one piece that listens on a socket. So it never touches them. Two small jobs here do, and each is the
narrowest thing that works:

    snapshot   (as `pipeline`, every 30 s)  writes what the app may SHOW: modes, last runs, per-game counts, release, latest log of each job
    apply      (as root, when the app drops a request)  validates one request and does it: set a job's mode, or run a job once

The request files are UNTRUSTED input from the mllab user, as the job spool's are (host/mllab-jobctl): regular files only, owned by mllab, small, strict
JSON, a fixed set of keys, job names from the job table (never a free-form name), modes from MODES. Nothing in a request is executed or interpolated
into a command line except values that passed those checks. Going live, or running for real, also needs the job's name typed back. The web app asks for
that too, and checks that Cloud Scheduler is not still running the same job; this side cannot see GCP, so it enforces only what it can see.

    python -m mllab.pipeline.broker snapshot
    python -m mllab.pipeline.broker apply"""
from __future__ import annotations

import json
import os
import pwd
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import pregame, runner
from .jobs import JOBS, MODES, PREGAME, cron_to_oncalendar

LIB = "/opt/pipeline/lib"
PY = "/opt/pipeline/venv/bin/python"
MLLAB_ROOT = Path(os.environ.get("MLLAB_ROOT", "/mnt/appdata/mllab"))
STATUS_DIR = Path(os.environ.get("BROKER_STATUS_DIR", MLLAB_ROOT / "pipeline-status"))
SPOOL = Path(os.environ.get("BROKER_SPOOL", MLLAB_ROOT / "spool"))
RESULTS = Path(os.environ.get("BROKER_RESULTS", MLLAB_ROOT / "pipeline-results"))
AUDIT = Path(os.environ.get("MLLAB_AUDIT", MLLAB_ROOT / "audit" / "deploys.log"))
RELEASE = Path(os.environ.get("PIPELINE_RELEASE", "/opt/pipeline/current"))

REQUEST_RE = re.compile(r"^pipe-([a-z0-9-]{8,64})\.json$")
MAX_REQUEST_BYTES = 2048
LOG_BYTES = 48 * 1024
RESULTS_KEEP = 200
ACTIONS = {"set_mode": {"id", "action", "job", "mode", "confirm", "actor"}, "run": {"id", "action", "job", "dry", "confirm", "actor"}}
# what a run-now unit gets: the same limits and environment as pipeline-job@.service (units.py), so a run from the app is a run the timer would do
ENV = {"PYTHONPATH": LIB, "PIPELINE_RELEASE": "/opt/pipeline/current", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic(path: Path, text: str, mode: int = 0o640) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------------------------------------- snapshot
def _systemctl_show(unit: str) -> dict:
    try:
        out = subprocess.run(["systemctl", "show", unit, "-p", "NextElapseUSecRealtime", "-p", "LastTriggerUSec"], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def _latest_log(job: str, state_dir: Path) -> str:
    d = state_dir / "logs" / job
    logs = sorted(d.glob("*.log")) if d.is_dir() else []
    if not logs:
        return ""
    data = logs[-1].read_bytes()[-LOG_BYTES:]
    return f"# {logs[-1].name}\n" + data.decode(errors="replace")


def build_snapshot(*, jobs_dir: Path | None = None, state_dir: Path | None = None, release: Path | None = None, show=_systemctl_show) -> tuple[dict, dict]:
    """(state, logs by job). Read-only over the pipeline's own directories."""
    sd = state_dir or runner.STATE_DIR
    last = runner.last_runs(sd)
    jobs = []
    for j in JOBS.values():
        timer = {} if j.paused else show(f"pipeline-{j.name}.timer")
        jobs.append({"name": j.name, "bundle": j.bundle, "mode": runner.read_mode(j.name, jobs_dir), "cron": j.cron, "schedule": cron_to_oncalendar(j.cron),
                     "paused": j.paused, "dry_ok": j.dry_ok, "body_mode": j.body.get("mode"), "last": last.get(j.name),
                     "next_run": timer.get("NextElapseUSecRealtime") or None})
    pg = {"name": PREGAME, "mode": runner.read_mode(PREGAME, jobs_dir), "stores": {}, "last": last.get("pregame-ticker")}
    for m in ("dry", "live"):
        p = sd / f"pregame-{m}.sqlite"
        if p.exists():
            try:
                s = pregame.Store(p)
                pg["stores"][m] = {"counts": s.counts(), "next_due": str(s.next_due())}
            except Exception as e:  # noqa: BLE001  a locked or half-written store must not stop the rest of the snapshot
                pg["stores"][m] = {"error": f"{type(e).__name__}: {e}"[:160]}
    try:
        rel = json.loads(((release or RELEASE) / "RELEASE.json").read_text())
    except (OSError, ValueError):
        rel = None
    state = {"taken_at": _now(), "jobs": jobs, "pregame": pg, "release": rel, "modes": list(MODES)}
    logs = {j.name: _latest_log(j.name, sd) for j in JOBS.values() if not j.paused}
    logs["pregame-ticker"] = _latest_log("pregame-ticker", sd)
    return state, logs


def cmd_snapshot(*, status_dir: Path | None = None, **kw) -> int:
    state, logs = build_snapshot(**kw)
    out = status_dir or STATUS_DIR
    (out / "logs").mkdir(parents=True, exist_ok=True)
    for name, text in logs.items():
        _atomic(out / "logs" / f"{name}.txt", text)
    _atomic(out / "state.json", json.dumps(state, indent=1, default=str) + "\n")  # last: its presence and age say the logs beside it are current
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ requests
class Rejected(Exception):
    pass


def validate(raw: bytes, *, current_mode=lambda job: runner.read_mode(job)) -> dict:
    """The parsed request, or Rejected. Pure: no side effects, so the rules are testable without a host."""
    if len(raw) > MAX_REQUEST_BYTES:
        raise Rejected("request too large")
    try:
        req = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise Rejected("not valid JSON") from None
    if not isinstance(req, dict) or req.get("action") not in ACTIONS:
        raise Rejected("unknown action")
    keys = ACTIONS[req["action"]]
    if set(req) - keys or (keys - {"confirm"}) - set(req):
        raise Rejected(f"fields must be exactly: {sorted(keys)}")
    job = req["job"]
    if not isinstance(job, str) or not isinstance(req.get("actor"), str) or not isinstance(req.get("id"), str):
        raise Rejected("job, actor and id must be strings")
    confirm = req.get("confirm")
    if req["action"] == "set_mode":
        if job != PREGAME and job not in JOBS:
            raise Rejected(f"unknown job {job!r}")
        if req["mode"] not in MODES:
            raise Rejected(f"mode must be one of {MODES}")
        if job != PREGAME and JOBS[job].paused and req["mode"] != "off":
            raise Rejected(f"{job} is paused in Cloud Scheduler on purpose and is not installed here")
        if req["mode"] == "live" and current_mode(job) != "live" and confirm != job:
            raise Rejected(f"going live needs the job name typed back: {job!r}")
    else:
        if job not in JOBS:
            raise Rejected(f"unknown job {job!r}")
        if JOBS[job].paused:
            raise Rejected(f"{job} is paused in Cloud Scheduler on purpose and is not installed here")
        if not isinstance(req["dry"], bool):
            raise Rejected("dry must be true or false")
        if not req["dry"] and confirm != job:
            raise Rejected(f"a real run needs the job name typed back: {job!r}")
    return req


def run_argv(req: dict) -> list[str]:
    """systemd-run for one run-now: a transient unit as `pipeline`, with the limits of pipeline-job@.service. Only validated values reach it."""
    unit = f"pipeline-adhoc-{req['job']}-{req['id'][:8]}"
    argv = ["systemd-run", f"--unit={unit}", "--uid=pipeline", "--gid=pipeline", "--slice=pipeline.slice", "--collect", "--no-block",
            "-p", "EnvironmentFile=/etc/pipeline/env", "-p", "WorkingDirectory=/var/lib/pipeline/work", "-p", "MemoryMax=3G", "-p", "RuntimeMaxSec=10800",
            "-p", "NoNewPrivileges=yes", "-p", "ProtectSystem=strict", "-p", "ProtectHome=yes", "-p", "PrivateTmp=yes", "-p", "ReadWritePaths=/var/lib/pipeline"]
    for k, v in ENV.items():
        argv += ["-p", f"Environment={k}={v}"]
    return argv + ["--", PY, "-m", "mllab.pipeline.ctl", "run", req["job"], *(["--dry"] if req["dry"] else []), "--yes"]


def execute(req: dict, *, jobs_dir: Path | None = None, run=subprocess.run) -> str:
    if req["action"] == "set_mode":
        before = runner.read_mode(req["job"], jobs_dir)
        p = runner.write_mode(req["job"], req["mode"], jobs_dir)
        try:
            shutil.chown(p, "root", "pipeline")
            os.chmod(p, 0o664)
        except (OSError, LookupError):
            pass
        return f"{req['job']}: {before} -> {req['mode']} (takes effect at its next run)"
    argv = run_argv(req)
    r = run(argv, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise Rejected(f"systemd-run failed: {(r.stderr or r.stdout).strip()[:200]}")
    return f"started {argv[1].split('=', 1)[1]} ({'dry' if req['dry'] else 'REAL'} run); its log appears under the job once it finishes"


def _result(rid: str, status: str, detail: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    _atomic(RESULTS / f"{rid}.json", json.dumps({"id": rid, "status": status, "detail": detail, "at": _now()}) + "\n")
    for old in sorted(RESULTS.glob("*.json"), key=lambda p: p.stat().st_mtime)[:-RESULTS_KEEP]:
        old.unlink(missing_ok=True)


def _audit(action: str, **fields) -> None:
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": _now(), "actor": "pipeline-broker", "action": action, **fields}, sort_keys=True) + "\n"
    fd = os.open(AUDIT, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def cmd_apply(*, spool: Path | None = None, owner: str = "mllab", jobs_dir: Path | None = None, run=subprocess.run, log=print) -> int:
    sp = spool or SPOOL
    try:
        uid = pwd.getpwnam(owner).pw_uid
    except KeyError:
        log(f"no user {owner!r}: refusing every request")
        return 1
    for f in sorted(sp.glob("pipe-*.json")):
        m = REQUEST_RE.match(f.name)
        rid = m.group(1) if m else None
        try:
            st = os.lstat(f)
            if not m or not os.path.isfile(f) or os.path.islink(f) or st.st_uid != uid:
                raise Rejected("not a regular request file owned by the app user")
            with open(f, "rb") as fh:
                raw = fh.read(MAX_REQUEST_BYTES + 1)
            req = validate(raw, current_mode=lambda job: runner.read_mode(job, jobs_dir))
            if req["id"] != rid:
                raise Rejected("request id does not match its file name")
            before = runner.read_mode(req["job"], jobs_dir) if req["action"] == "set_mode" else None
            detail = execute(req, jobs_dir=jobs_dir, run=run)
            # structured fields as well as the sentence: the app's rollback reads `before` and `mode` to know what to go back to
            extra = {"before": before, "mode": req["mode"]} if req["action"] == "set_mode" else {"dry": req["dry"]}
            _audit(f"pipeline_{req['action']}", request=rid, job=req["job"], by=req["actor"], detail=detail, **extra)
            _result(rid, "done", detail)
            log(f"{rid}: {detail}")
        except Rejected as e:
            _audit("pipeline_rejected", request=rid or f.name, reason=str(e))
            if rid:
                _result(rid, "rejected", str(e))
            log(f"{f.name}: rejected: {e}")
        except OSError as e:
            log(f"{f.name}: {e}")
        finally:
            f.unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if a == ["snapshot"]:
        return cmd_snapshot()
    if a == ["apply"]:
        return cmd_apply()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
