"""Run one pipeline job: the same entrypoint, request body and configuration the Cloud Function had, in a subprocess against its staged bundle.

    python -m mllab.pipeline.runner run <job>        run (or skip, or preflight) according to the job's mode
    python -m mllab.pipeline.runner list             the job table
    python -m mllab.pipeline.runner status           last run of each job

Each job has a mode file, /etc/pipeline/jobs.d/<job>, whose first word decides what a timer firing does:
    off   nothing. This is the default, so installing the units can never run anything.
    dry   the entrypoint with dry_run=true. For the football jobs that refuse dry_run (ingest, score, cfbd) it is a PREFLIGHT instead:
          import the bundle, resolve its secrets, run nothing.
    live  the real run.

One run of a job at a time (flock). A subprocess per run: the NFL and CFB bundles each have their own config.py and cannot share an
interpreter, a crash cannot take the runner with it, and the timeout can kill the whole process group."""
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .jobs import BUNDLES, JOBS, MODES, PREGAME, THREAD_ENV, Bundle, Job

STATE_DIR = Path(os.environ.get("PIPELINE_STATE", "/var/lib/pipeline/state"))
JOBS_DIR = Path(os.environ.get("PIPELINE_JOBS_DIR", "/etc/pipeline/jobs.d"))
RELEASE = Path(os.environ.get("PIPELINE_RELEASE", "/opt/pipeline/current"))
LOG_KEEP = 30  # run logs kept per job

# What runs inside the subprocess. Entrypoints are @functions_framework.http functions, which return the original function, so a request
# object with get_json() is enough (the repo's own tests do the same). The MLB entrypoint returns (json_string, status, headers); the
# football ones return (dict, status).
SHIM = r"""
import importlib, json, os, sys
mod, fn = os.environ["PIPELINE_ENTRY"].split(":")
entry = getattr(importlib.import_module(mod), fn)
if os.environ.get("PIPELINE_CHECK") == "1":
    print(json.dumps({"preflight": "ok", "entry": os.environ["PIPELINE_ENTRY"], "callable": callable(entry)}))
    sys.exit(0 if callable(entry) else 3)
payload = json.loads(sys.stdin.read())
class Req:
    method = "POST"
    headers = {}
    args = {}
    def __init__(self, p): self.p = p
    def get_json(self, silent=True): return self.p
out = entry(Req(payload))
body, status = out[0], out[1] if len(out) > 1 else 200
if isinstance(body, (bytes, str)):
    body = json.loads(body)
print("\n@@RESULT@@" + json.dumps({"status": int(status), "body": body}, default=str))
"""
RESULT_MARK = "@@RESULT@@"

Executor = Callable[..., "tuple[int, str, str]"]


# ----------------------------------------------------------------------------------------------------------------------------- modes
def read_mode(name: str, jobs_dir: Path | None = None) -> str:
    """The mode file's first word; a missing, empty or unrecognised file is `off`, never `live`."""
    try:
        for line in (jobs_dir or JOBS_DIR).joinpath(name).read_text().splitlines():
            word = line.split("#")[0].strip().split()
            if word:
                return word[0] if word[0] in MODES else "off"
    except OSError:
        pass
    return "off"


def write_mode(name: str, mode: str, jobs_dir: Path | None = None) -> Path:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if name != PREGAME and name not in JOBS:
        raise KeyError(f"unknown job {name!r}")
    d = jobs_dir or JOBS_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(f"{mode}  # set {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
    os.replace(tmp, p)
    return p


# --------------------------------------------------------------------------------------------------------------------------- secrets
def resolve_secret(ref: str) -> str:
    """sm://projects/P/secrets/S/versions/V -> its value. Imported lazily: only the CFB job needs it, and tests inject a fake."""
    from google.cloud import secretmanager  # type: ignore

    name = ref[len("sm://"):]
    return secretmanager.SecretManagerServiceClient().access_secret_version(name=name).payload.data.decode("utf-8").strip()


def build_env(bundle: Bundle, release: Path, resolve: Callable[[str], str] = resolve_secret, base: dict | None = None) -> dict:
    """The subprocess environment: only what the function carried in production, plus the thread caps, plus credentials the unit supplies.
    K_SERVICE is deliberately never set: pa_sim's memory check treats it as 'inside a managed runtime, limit unknown'."""
    src = base if base is not None else os.environ
    env = {k: src[k] for k in ("PATH", "HOME", "LANG", "GOOGLE_APPLICATION_CREDENTIALS", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "TMPDIR") if k in src}
    env.update(bundle.env)
    env.update(THREAD_ENV)
    env["PYTHONPATH"] = str(release / bundle.name)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PIPELINE_ENTRY"] = bundle.entry
    for key, ref in bundle.secrets.items():
        env[key] = resolve(ref)
    return env


# ---------------------------------------------------------------------------------------------------------------------------- execute
def subprocess_executor(env: dict, stdin: str, timeout: int, cwd: Path | None = None) -> tuple[int, str, str]:
    """Run the shim in its own process group so a timeout can kill everything it started. Returns (exit code, stdout, stderr)."""
    p = subprocess.Popen([sys.executable, "-c", SHIM], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, start_new_session=True, cwd=str(cwd) if cwd else None)
    try:
        out, err = p.communicate(stdin, timeout=timeout)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        return 124, out or "", (err or "") + f"\n[runner] killed after {timeout}s"


def parse_result(stdout: str) -> dict | None:
    if RESULT_MARK not in stdout:
        return None
    try:
        return json.loads(stdout.split(RESULT_MARK, 1)[1].strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def summarize(body) -> dict:
    """A short, safe summary of a result body for the status file: its status and each step's name/status/seconds. Never the full body."""
    if not isinstance(body, dict):
        return {}
    out = {k: body[k] for k in ("status", "mode", "date", "error") if k in body}
    steps = body.get("steps")
    if isinstance(steps, list):
        out["steps"] = [{k: s.get(k) for k in ("step", "status", "seconds") if isinstance(s, dict) and k in s} for s in steps][:20]
    elif isinstance(steps, dict):
        out["steps"] = {k: (v.get("error") if isinstance(v, dict) and "error" in v else "ok") for k, v in list(steps.items())[:20]}
    return out


FAILED_STEP_STATUSES = ("error", "failed")


def step_errors(body) -> list[str]:
    """Steps that failed inside an otherwise successful (200) result, as "name: message".

    The pipelines catch an exception in an optional step (a shadow model, a snapshot) into that step so the headline
    still ships, and answer 200. That is right for the site, but the run must not be recorded ok: the CFB drive-sim
    shadow crashed on 2026-11-10 and nothing alerted. Football steps carry {"error": ...}; MLB list steps a status."""
    if not isinstance(body, dict):
        return []
    steps = body.get("steps")
    if isinstance(steps, dict):
        return [f"{k}: {str(v['error'])[:160]}" for k, v in steps.items() if isinstance(v, dict) and v.get("error")]
    if isinstance(steps, list):
        return [f"{s.get('step')}: {s.get('status')}" for s in steps
                if isinstance(s, dict) and str(s.get("status", "")).lower() in FAILED_STEP_STATUSES]
    return []


def execute(bundle: Bundle, payload: dict, *, timeout: int, release: Path = None, executor: Executor = subprocess_executor,
            resolve: Callable[[str], str] = resolve_secret, check: bool = False, workdir: Path | None = None) -> dict:
    """One entrypoint call. Returns {ok, exit, http_status, seconds, summary, error, log_tail}. Never raises for a failed run."""
    release = release or RELEASE
    t0 = time.time()
    try:
        env = build_env(bundle, release, resolve)
    except Exception as e:  # a secret that cannot be read is a failed run, and the message must not echo the reference's value
        return {"ok": False, "exit": 2, "http_status": None, "seconds": 0.0, "summary": {}, "error": f"environment: {type(e).__name__}: {str(e)[:200]}",
                "log_tail": ""}
    if check:
        env["PIPELINE_CHECK"] = "1"
    code, out, err = executor(env, json.dumps(payload), timeout, workdir)
    # The pipeline's JSON log lines go to stdout, errors and tracebacks to stderr: keep the tail of both, minus the result line.
    log_tail = "\n".join((out.split(RESULT_MARK, 1)[0].strip().splitlines() + err.strip().splitlines())[-60:])[-6000:]
    res = {"exit": code, "seconds": round(time.time() - t0, 1), "log_tail": log_tail}
    if check:
        res.update(ok=code == 0, http_status=None, summary={"preflight": out.strip()[-200:]}, error=None if code == 0 else f"preflight exit {code}")
        return res
    parsed = parse_result(out)
    if parsed is None:
        return {**res, "ok": False, "http_status": None, "summary": {}, "error": "timeout" if code == 124 else f"no result (exit {code})"}
    body = parsed.get("body")
    status = parsed.get("status")
    failed_steps = step_errors(body)
    ok = status == 200 and not (isinstance(body, dict) and body.get("status") == "error") and not failed_steps
    step_msg = None
    if failed_steps:
        step_msg = "step " + failed_steps[0] + (f" (+{len(failed_steps) - 1} more)" if len(failed_steps) > 1 else "")
    return {**res, "ok": ok, "http_status": status, "summary": summarize(body),
            "error": None if ok else (body.get("error") if isinstance(body, dict) else None) or step_msg or f"HTTP {status}"}


# ----------------------------------------------------------------------------------------------------------------------------- state
def _now() -> datetime:
    return datetime.now(timezone.utc)


def record(job: str, rec: dict, state_dir: Path | None = None, log_text: str = "") -> None:
    """Append to runs.jsonl, replace status/<job>.json, and keep a log file per run."""
    d = state_dir or STATE_DIR
    (d / "status").mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, default=str)
    with open(d / "runs.jsonl", "a") as f:
        f.write(line + "\n")
    tmp = d / "status" / f"{job}.json.tmp"
    tmp.write_text(line + "\n")
    os.replace(tmp, d / "status" / f"{job}.json")
    if log_text:
        ld = d / "logs" / job
        ld.mkdir(parents=True, exist_ok=True)
        (ld / (rec["started"].replace(":", "").replace("-", "")[:15] + ".log")).write_text(log_text)
        for old in sorted(ld.glob("*.log"))[:-LOG_KEEP]:
            old.unlink(missing_ok=True)


def kuma_push(ok: bool, msg: str, url: str | None = None) -> None:
    """Heartbeat to an Uptime Kuma push monitor, if PIPELINE_KUMA_PUSH_URL is set. Best effort: monitoring must never fail a run."""
    url = url or os.environ.get("PIPELINE_KUMA_PUSH_URL")
    if not url:
        return
    try:
        q = urllib.parse.urlencode({"status": "up" if ok else "down", "msg": msg[:120]})
        urllib.request.urlopen(f"{url}{'&' if '?' in url else '?'}{q}", timeout=8).read()  # noqa: S310 - operator-configured URL
    except Exception:
        pass


class Busy(Exception):
    pass


class lock:
    """One holder per name. flock is released when the process dies, so a crash cannot leave a stale lock."""

    def __init__(self, name: str, state_dir: Path | None = None):
        d = (state_dir or STATE_DIR) / "locks"
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"{name}.lock"

    def __enter__(self):
        self.f = open(self.path, "w")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.f.close()
            raise Busy(self.path.name)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


# -------------------------------------------------------------------------------------------------------------------------------- run
def run_job(name: str, *, release: Path | None = None, jobs_dir: Path | None = None, state_dir: Path | None = None,
            executor: Executor = subprocess_executor, resolve: Callable[[str], str] = resolve_secret,
            push: Callable[[bool, str], None] = kuma_push, mode: str | None = None) -> int:
    """What a timer firing does. Exit 0 for off/busy/ok, 1 for a failed run so systemd marks the unit failed."""
    job = JOBS.get(name)
    if job is None:
        print(f"unknown job {name!r}", file=sys.stderr)
        return 2
    mode = mode or read_mode(name, jobs_dir)
    started = _now().isoformat(timespec="seconds")
    if mode == "off":
        print(f"{name}: mode is off; nothing to do")
        return 0
    try:
        with lock(name, state_dir):
            bundle = BUNDLES[job.bundle]
            preflight = mode == "dry" and not job.dry_ok
            payload = dict(job.body)
            if mode == "dry" and job.dry_ok:
                payload["dry_run"] = True
            if mode == "live":
                payload.pop("dry_run", None)
            res = execute(bundle, payload, timeout=job.timeout_s, release=release, executor=executor, resolve=resolve, check=preflight)
            rec = {"job": name, "mode": mode + ("(preflight)" if preflight else ""), "started": started, "finished": _now().isoformat(timespec="seconds"),
                   **{k: res[k] for k in ("ok", "exit", "http_status", "seconds", "summary", "error")}}
            record(name, rec, state_dir, log_text=res.get("log_tail", ""))
            print(json.dumps({k: rec[k] for k in ("job", "mode", "ok", "seconds", "error")}))
            if mode == "live":
                push(rec["ok"], f"{name} {'ok' if rec['ok'] else 'FAILED'} {rec['seconds']}s")
            return 0 if rec["ok"] else 1
    except Busy:
        record(name, {"job": name, "mode": mode, "started": started, "finished": started, "ok": True, "skipped": "already running"}, state_dir)
        print(f"{name}: already running; skipped")
        return 0


def last_runs(state_dir: Path | None = None) -> dict[str, dict]:
    out = {}
    for p in sorted((state_dir or STATE_DIR).joinpath("status").glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text())
        except (OSError, ValueError):
            pass
    return out


def main(argv: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    cmd = a[0] if a else "list"
    if cmd == "run" and len(a) >= 2:
        return run_job(a[1])
    if cmd == "list":
        for j in JOBS.values():
            print(f"{j.name:28s} {j.bundle:4s} {read_mode(j.name):5s} {'paused-upstream' if j.paused else ''}")
        return 0
    if cmd == "status":
        for n, r in last_runs().items():
            print(f"{n:28s} {r.get('finished', '?'):26s} {'ok' if r.get('ok') else 'FAIL':5s} {r.get('seconds', '')}s {r.get('error') or ''}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
