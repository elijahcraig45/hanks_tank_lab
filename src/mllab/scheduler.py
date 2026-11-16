"""Cloud Scheduler controls: list jobs (with their decoded request body), pause, resume, run now.

Uses the caller's own gcloud identity through the `gcloud` CLI, so no service-account grants are involved. Job names must exist in the live
list (no free-form names reach the command), changes need the job name typed back as confirmation, and every change is audited.
The listing also surfaces a real trap in this project: a job whose body is not valid JSON silently runs the default `daily` mode.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys

from . import audit
from .auth import is_blocked_account

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class SchedulerError(Exception):
    pass


def decode_body(b64: str | None) -> tuple[dict | None, bool]:
    """(parsed body or None, is_valid_json). An empty body is valid-and-empty; garbage is invalid."""
    if not b64:
        return {}, True
    try:
        raw = base64.b64decode(b64)
    except Exception:  # noqa: BLE001
        return None, False
    try:
        d = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None, False
    return (d, True) if isinstance(d, dict) else (None, False)


def parse_jobs(raw: list[dict]) -> list[dict]:
    out = []
    for j in raw:
        body, ok = decode_body((j.get("httpTarget") or {}).get("body"))
        status = j.get("status") or {}
        attempted = bool(j.get("lastAttemptTime"))
        out.append({"name": j["name"].rsplit("/", 1)[-1], "state": j.get("state"), "schedule": j.get("schedule"), "tz": j.get("timeZone"),
                    "body_ok": ok, "mode": (body or {}).get("mode") if body is not None else None, "body": body,
                    "last_attempt_at": j.get("lastAttemptTime"), "last_attempt_ok": (not status.get("code")) if attempted else None,
                    "next_run": j.get("scheduleTime")})
    return sorted(out, key=lambda x: x["name"])


def _audit(action: str, path, **fields) -> None:
    """Record a change that has ALREADY happened. If the log cannot be written (on a laptop the default path is the homelab's), say so plainly and carry on:
    a traceback here would make a successful pause look like a failed one, and the next move would be to retry it."""
    try:
        audit.record(action, path=path, **fields)
    except OSError as e:
        print(f"warning: {action} {fields.get('job')} WAS DONE, but the audit log could not be written ({e}). Set MLLAB_AUDIT to a writable file.", file=sys.stderr)


class Scheduler:
    def __init__(self, project: str = "hankstank", location: str = "us-central1", account: str | None = "you@example.com",
                 runner=subprocess.run, gcloud: str = "gcloud"):
        if is_blocked_account(account):
            raise SchedulerError("refusing to use the work account")
        self.project, self.location, self.account, self.runner, self.gcloud = project, location, account, runner, gcloud

    def _run(self, args: list[str]) -> str:
        env = {**os.environ, **({"CLOUDSDK_CORE_ACCOUNT": self.account} if self.account else {})}
        r = self.runner([self.gcloud, "scheduler", "jobs", *args, "--project", self.project, "--location", self.location], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise SchedulerError(f"gcloud failed: {(r.stderr or r.stdout).strip()[:300]}")
        return r.stdout

    def list(self) -> list[dict]:
        return parse_jobs(json.loads(self._run(["list", "--format=json"]) or "[]"))

    def _known(self, name: str) -> dict:
        if not NAME_RE.match(name):
            raise SchedulerError(f"invalid job name {name!r}")
        for j in self.list():
            if j["name"] == name:
                return j
        raise SchedulerError(f"no such job {name!r} (it must appear in `control scheduler list`)")

    def _change(self, verb: str, name: str, confirm: str | None, dry_run: bool, audit_path=None) -> dict:
        job = self._known(name)
        if confirm != name:
            raise SchedulerError(f"this changes production; confirm by typing the job name: {name!r}")
        if verb == "pause" and job["state"] == "PAUSED" or verb == "resume" and job["state"] == "ENABLED":
            return {"changed": False, "detail": f"{name} is already {job['state']}"}
        if not dry_run:
            self._run([verb, name])
            _audit(f"scheduler_{verb}", audit_path, job=name, was=job["state"])
        return {"changed": not dry_run, "dry_run": dry_run, "detail": f"{verb} {name}"}

    def pause(self, name: str, *, confirm: str | None = None, dry_run: bool = False, audit_path=None) -> dict:
        return self._change("pause", name, confirm, dry_run, audit_path)

    def resume(self, name: str, *, confirm: str | None = None, dry_run: bool = False, audit_path=None) -> dict:
        job = self._known(name)
        if job["body_ok"] is False:
            raise SchedulerError(f"refusing to resume {name}: its body is not valid JSON, so it would silently run the default 'daily' mode")
        return self._change("resume", name, confirm, dry_run, audit_path)

    def run_now(self, name: str, *, confirm: str | None = None, dry_run: bool = False, audit_path=None) -> dict:
        job = self._known(name)
        if confirm != name:
            raise SchedulerError(f"this triggers production work now; confirm by typing the job name: {name!r}")
        if job["body_ok"] is False:
            raise SchedulerError(f"refusing to run {name}: its body is not valid JSON, so it would run the default 'daily' mode")
        if not dry_run:
            self._run(["run", name])
            _audit("scheduler_run", audit_path, job=name, mode=job["mode"])
        return {"changed": not dry_run, "dry_run": dry_run, "detail": f"run {name} (mode {job['mode']})"}


def render(jobs: list[dict]) -> str:
    L = ["| job | state | schedule | mode | body | last run |", "|---|---|---|---|---|---|"]
    for j in jobs:
        last = "-" if j["last_attempt_ok"] is None else ("ok" if j["last_attempt_ok"] else "FAILED")
        body = "ok" if j["body_ok"] else "INVALID JSON (runs 'daily'!)"
        L.append(f"| {j['name']} | {j['state']} | {j['schedule']} {j['tz'] or ''} | {j['mode'] or '-'} | {body} | {last} |")
    return "\n".join(L)
