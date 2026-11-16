"""The pipeline as the app shows and changes it.

Reads what the broker publishes (pipeline-status/state.json), joins it with Cloud Scheduler's state, and turns changes into request files the broker
validates and applies. Nothing here touches /etc/pipeline or /var/lib/pipeline: the app's user cannot.

The gate is the point of the screen. A job must not write production from two places, so before one goes live here (or runs for real) the app checks
that Cloud Scheduler is not still running the same job, and refuses if it is, or if it cannot tell. It cannot see the App Engine cron entry or the Cloud
Tasks queues that also feed per-game MLB work, so for those the person acknowledges each step explicitly."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..pipeline.jobs import JOBS, LOCATION, MODES, PREGAME, PROJECT
from . import history
from .views import Refused

PREGAME_SCHEDULER = "mlb-2026-pregame-schedule"
FRESH_S = 150  # the snapshot refreshes every 30 s; older than this and the broker is not running
GCP_TTL_S = 30
ID_RE = re.compile(r"^[a-z0-9-]{8,64}$")

PREGAME_ACKS = [
    {"key": "cron", "text": "I removed the /api/lineup/schedule-today entry from hanks_tank_backend/cron.yaml and deployed it (App Engine's cron also creates per-game tasks)."},
    {"key": "queues", "text": f"I paused the Cloud Tasks queues lineup-pregame and sim-blend ({LOCATION}) so nothing already queued runs on top of this."},
]
QUEUE_COMMANDS = [f"gcloud tasks queues pause lineup-pregame --location {LOCATION} --project {PROJECT}",
                  f"gcloud tasks queues pause sim-blend --location {LOCATION} --project {PROJECT}"]


def scheduler_name(job: str) -> str:
    return PREGAME_SCHEDULER if job == PREGAME else job


def gate(job: str, gcp: dict[str, dict] | None, gcp_error: str | None = None) -> dict:
    """What must be true in GCP before `job` writes from here. `gcp` maps Cloud Scheduler job name to its listing, or is None when it could not be read."""
    name = scheduler_name(job)
    reasons, commands = [], []
    if gcp is None:
        reasons.append(f"cannot read Cloud Scheduler to check that {name} is paused ({gcp_error or 'unavailable'}); refusing rather than guessing")
    else:
        sched = gcp.get(name)
        if sched is None:
            reasons.append(f"Cloud Scheduler has no job named {name}; check the name before going live")
        elif sched.get("state") != "PAUSED":
            reasons.append(f"Cloud Scheduler job {name} is {sched.get('state')}: it would write to the same tables")
            commands.append(f"mllab control scheduler pause {name} --confirm {name}")
    if job == PREGAME:
        commands += QUEUE_COMMANDS
    return {"scheduler_job": name, "allowed": not reasons, "reasons": reasons, "commands": commands, "acks": PREGAME_ACKS if job == PREGAME else []}


class PipelineView:
    def __init__(self, status_dir: Path, spool: Path, results: Path, scheduler=None, clock=time.time, audit_path: Path | None = None):
        self.status_dir, self.spool, self.results = Path(status_dir), Path(spool), Path(results)
        self.audit_path = audit_path
        self.scheduler, self.clock = scheduler, clock
        self._gcp: tuple[float, dict | None, str | None] | None = None

    # ------------------------------------------------------------------------------------------------------------------------- reading
    def gcp(self, *, fresh: bool = False) -> tuple[dict[str, dict] | None, str | None]:
        """(Cloud Scheduler jobs by name, error). Cached briefly so a page of status does not hammer the API; `fresh` for a decision."""
        now = self.clock()
        if not fresh and self._gcp and now - self._gcp[0] < GCP_TTL_S:
            return self._gcp[1], self._gcp[2]
        if self.scheduler is None:
            res = (None, "no Cloud Scheduler client configured")
        else:
            try:
                res = ({j["name"]: j for j in self.scheduler.list()}, None)
            except Exception as e:  # noqa: BLE001  any failure means "cannot verify", which the gate treats as a refusal
                res = (None, f"{type(e).__name__}: {e}"[:200])
        self._gcp = (now, *res)
        return res

    def _snapshot(self) -> tuple[dict | None, float | None]:
        try:
            raw = json.loads((self.status_dir / "state.json").read_text())
            taken = datetime.fromisoformat(raw["taken_at"]).timestamp()
            return raw, max(0.0, self.clock() - taken)
        except (OSError, ValueError, KeyError):
            return None, None

    def state(self, *, fresh: bool = False) -> dict:
        raw, age = self._snapshot()
        gcp, gcp_error = self.gcp(fresh=fresh)
        out = {"broker": {"available": raw is not None, "fresh": age is not None and age <= FRESH_S, "age_s": None if age is None else round(age),
                          "hint": None if raw is not None and age <= FRESH_S else
                          "The pipeline broker is not publishing (pipeline-broker-snapshot.timer on the homelab). Shown values may be stale or missing."},
               "gcp": {"available": gcp is not None, "error": gcp_error}, "modes": list(MODES), "jobs": [], "pregame": None, "release": None}
        if raw is None:
            return out
        out["release"] = raw.get("release")
        entries = self._audit()
        for j in raw["jobs"]:
            j["rollback"] = history.previous_mode(entries, j["name"], j["mode"])
            sched = (gcp or {}).get(j["name"])
            j["gcp"] = None if sched is None else {k: sched.get(k) for k in ("state", "schedule", "last_attempt_at", "last_attempt_ok", "next_run", "body_ok")}
            j["gate"] = gate(j["name"], gcp, gcp_error)
            j["conflict"] = bool(sched) and j["mode"] == "live" and sched.get("state") == "ENABLED"
            out["jobs"].append(j)
        pg = raw["pregame"]
        sched = (gcp or {}).get(PREGAME_SCHEDULER)
        pg["rollback"] = history.previous_mode(entries, PREGAME, pg["mode"])
        pg["gate"] = gate(PREGAME, gcp, gcp_error)
        pg["conflict"] = bool(sched) and pg["mode"] == "live" and sched.get("state") == "ENABLED"
        out["pregame"] = pg
        return out

    def _audit(self) -> list[dict]:
        from .. import audit
        try:
            return audit.read(self.audit_path)
        except (OSError, ValueError):
            return []

    def log(self, job: str) -> dict:
        if job not in JOBS and job not in (PREGAME, "pregame-ticker"):
            raise Refused([f"unknown job {job!r}"], 404)
        name = "pregame-ticker" if job == PREGAME else job
        try:
            return {"job": job, "text": (self.status_dir / "logs" / f"{name}.txt").read_text()}
        except OSError:
            return {"job": job, "text": ""}

    # ------------------------------------------------------------------------------------------------------------------------- changing
    def _mode_now(self, job: str) -> str | None:
        raw, _ = self._snapshot()
        if raw is None:
            return None
        row = raw["pregame"] if job == PREGAME else next((j for j in raw["jobs"] if j["name"] == job), None)
        return row["mode"] if row else None

    def _check_gate(self, job: str, confirm: str | None, acks: list[str]) -> None:
        if confirm != job:
            raise Refused([f"type the job name to confirm: {job}"], 400)
        gcp, err = self.gcp(fresh=True)
        g = gate(job, gcp, err)
        reasons = list(g["reasons"])
        missing = [a["key"] for a in g["acks"] if a["key"] not in acks]
        if missing:
            reasons.append("acknowledge each step: " + ", ".join(missing))
        if reasons:
            raise Refused(reasons)

    def _drop(self, body: dict) -> str:
        rid = str(uuid.uuid4())
        body = {"id": rid, **body}
        self.spool.mkdir(parents=True, exist_ok=True)
        tmp = self.spool / f".pipe-{rid}.tmp"  # a leading dot: the broker's path unit does not match it until the rename
        tmp.write_text(json.dumps(body))
        os.chmod(tmp, 0o640)
        os.replace(tmp, self.spool / f"pipe-{rid}.json")
        return rid

    def set_mode(self, job: str, mode: str, actor: str, confirm: str | None = None, acks: list[str] | None = None) -> str:
        if job != PREGAME and job not in JOBS:
            raise Refused([f"unknown job {job!r}"], 404)
        if mode not in MODES:
            raise Refused([f"mode must be one of {', '.join(MODES)}"], 400)
        if job != PREGAME and JOBS[job].paused and mode != "off":
            raise Refused([f"{job} is paused in Cloud Scheduler on purpose and is not installed here"])
        if mode == "live" and self._mode_now(job) != "live":
            self._check_gate(job, confirm, acks or [])
        return self._drop({"action": "set_mode", "job": job, "mode": mode, "confirm": confirm or "", "actor": actor})

    def rollback(self, job: str, actor: str, confirm: str | None = None, acks: list[str] | None = None) -> str:
        """Undo the job's latest mode change, as another mode change. Going back to Live goes through the gate like any other way to Live; going back to Dry or Off
        needs nothing. Changes only the homelab: Cloud Scheduler is not touched, so a job rolled back from Live is, for now, run by nobody in production."""
        if job != PREGAME and job not in JOBS:
            raise Refused([f"unknown job {job!r}"], 404)
        now = self._mode_now(job)
        if now is None:
            raise Refused(["no pipeline status yet: cannot tell what the job is doing now"])
        prev = history.previous_mode(self._audit(), job, now)
        if prev is None:
            raise Refused([f"no earlier mode on record for {job}: nothing to roll back to"])
        return self.set_mode(job, prev["to"], actor, confirm, acks)

    def run(self, job: str, dry: bool, actor: str, confirm: str | None = None, acks: list[str] | None = None) -> str:
        if job not in JOBS:
            raise Refused([f"unknown job {job!r}"], 404)
        if JOBS[job].paused:
            raise Refused([f"{job} is paused in Cloud Scheduler on purpose and is not installed here"])
        if not dry:
            self._check_gate(job, confirm, acks or [])
        return self._drop({"action": "run", "job": job, "dry": dry, "confirm": confirm or "", "actor": actor})

    def request(self, rid: str) -> dict:
        if not ID_RE.match(rid):
            raise Refused(["bad request id"], 400)
        try:
            return json.loads((self.results / f"{rid}.json").read_text())
        except (OSError, ValueError):
            pass
        if (self.spool / f"pipe-{rid}.json").exists():
            return {"id": rid, "status": "queued", "detail": "waiting for the broker (it picks requests up within a few seconds)"}
        return {"id": rid, "status": "unknown", "detail": "no such request: it may have been applied long ago, or the broker is not running"}
