"""Self-healing: turn health findings into bounded, reversible actions, only where you have opted in.

Safety rules (all enforced here, not by convention):
- Off by default: with no policy file, or `dry_run: true` (the default), nothing is changed; the log says what WOULD have happened.
- Per-rule opt-in: a finding's rule code must be enabled in the policy, with its own `max_per_day` and `cooldown_minutes`.
- Actions are a closed set: trigger a scheduler job that already exists, hide a model behind a banner, prune idle envs, or alert. Nothing is ever
  un-hidden, deleted from the registry, redeployed or written to a model table automatically.
- The engine never acts on data it could not read (source_unavailable findings carry no remedy).
- Every decision, including skips, is appended to a JSONL log that the dashboard shows.

Policy file (JSON):
  {"dry_run": false, "rules": {"coverage_gap": {"enabled": true, "max_per_day": 2, "cooldown_minutes": 90}}}
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from . import health as H

ACTIONS = ("trigger_function", "control_hide", "prune_envs", "alert")
DEFAULT_RULE = {"enabled": False, "max_per_day": 1, "cooldown_minutes": 120}
HIDE_BANNER = "{model} is temporarily unavailable while we fix a problem. Other models are unaffected."


class RemedyError(Exception):
    pass


@dataclass
class Policy:
    dry_run: bool = True
    rules: dict = field(default_factory=dict)

    def rule(self, code: str) -> dict:
        return {**DEFAULT_RULE, **(self.rules.get(code) or {})}


def load_policy(path: str | Path | None) -> Policy:
    """Missing or unreadable policy means dry-run with nothing enabled (fail safe)."""
    if not path or not Path(path).exists():
        return Policy()
    try:
        raw = json.loads(Path(path).read_text())
        rules = raw.get("rules", {})
        if not isinstance(rules, dict):
            raise ValueError("rules must be an object")
        return Policy(dry_run=raw.get("dry_run", True) is not False, rules=rules)
    except (ValueError, OSError):
        return Policy()


def log_path() -> Path:
    return Path(os.environ.get("MLLAB_REMEDY_LOG", "/mnt/appdata/mllab/audit/remedies.jsonl"))


def read_log(path: Path | None = None) -> list[dict]:
    p = path or log_path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _append(entry: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
    try:
        os.write(fd, (json.dumps(entry, sort_keys=True) + "\n").encode())
    finally:
        os.close(fd)


def _acted(log: list[dict], key: str, since: datetime) -> list[datetime]:
    times = [datetime.fromisoformat(e["at"]) for e in log if e.get("key") == key and e.get("executed")]
    return [t for t in times if t >= since]


def _blocked(policy: Policy, f: H.Finding, log: list[dict], now: datetime) -> str | None:
    r = policy.rule(f.code)
    if not r["enabled"]:
        return "rule not enabled in policy"
    if f.remedy is None or f.remedy.action not in ACTIONS:
        return "no permitted remedy"
    if f.remedy.action == "alert":
        return None
    day = _acted(log, f.key(), now - timedelta(days=1))
    if len(day) >= int(r["max_per_day"]):
        return f"daily limit reached ({r['max_per_day']})"
    recent = _acted(log, f.key(), now - timedelta(minutes=float(r["cooldown_minutes"])))
    if recent:
        return f"in cooldown ({r['cooldown_minutes']} min)"
    return None


@dataclass
class Executors:
    """Injected so tests never touch production. Each returns a short outcome string or raises."""
    run_job: Callable[[str, str], str]              # (sport, mode) -> detail
    hide: Callable[[str, str, str], str]            # (sport, model, banner) -> detail
    prune: Callable[[float], str]                   # (older_than_days) -> detail


def _execute(f: H.Finding, ex: Executors) -> str:
    r = f.remedy
    p = r.params
    if r.action == "trigger_function":
        return ex.run_job(p["sport"], p["mode"])
    if r.action == "control_hide":
        return ex.hide(p["sport"], p["model"], HIDE_BANNER.format(model=p["model"]))
    if r.action == "prune_envs":
        return ex.prune(float(p.get("older_than_days", 14)))
    return "alert only"


def run(findings: list[H.Finding], policy: Policy, ex: Executors, *, now: datetime | None = None, path: Path | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    path = path or log_path()
    log = read_log(path)
    out = []
    for f in findings:
        if f.remedy is None:
            continue
        why = _blocked(policy, f, log, now)
        entry = {"at": now.isoformat(timespec="seconds"), "key": f.key(), "rule": f.code, "action": f.remedy.action, "executed": False}
        if why:
            entry["outcome"] = f"skipped: {why}"
            if why != "rule not enabled in policy":  # not-enabled is the normal case: keep the log readable
                _append(entry, path)
                out.append(entry)
            continue
        if f.remedy.action == "alert":
            entry["outcome"] = "alert: " + f.message
        elif policy.dry_run:
            entry["outcome"] = f"dry-run: would {f.remedy.action} {json.dumps(f.remedy.params, sort_keys=True)}"
            entry["executed"] = True  # counts toward cooldown so a dry-run log is not spammed every tick
        else:
            try:
                entry["outcome"] = "done: " + _execute(f, ex)
            except Exception as e:  # noqa: BLE001
                entry["outcome"] = f"failed: {str(e)[:200]}"
            entry["executed"] = True  # a failed attempt also counts, so a broken action is not retried in a loop
        _append(entry, path)
        out.append(entry)
    return out


def make_executors(control, scheduler, envs_runner=None) -> Executors:
    """Real executors. Scheduler jobs are looked up by their decoded `mode` among existing jobs: never a free-form name or a new job."""
    from . import envs

    def run_job(sport: str, mode: str) -> str:
        jobs = [j for j in scheduler.list() if j["name"].startswith(f"{sport}-") and j["mode"] == mode and j["body_ok"]]
        if len(jobs) != 1:
            raise RemedyError(f"need exactly one {sport} job with mode {mode!r}, found {len(jobs)}")
        name = jobs[0]["name"]
        return scheduler.run_now(name, confirm=name)["detail"]

    def hide(sport: str, model: str, banner: str) -> str:
        from .control import confirm_phrase
        evs = control.hide(sport, model, confirm=confirm_phrase(sport, model), banner=banner, banner_level="warn", note="auto-heal: bad output")
        return f"hid {sport}/{model} with a banner ({len(evs)} events)"

    def prune(days: float) -> str:
        if envs_runner is None:
            raise RemedyError("no local runner configured")
        return f"pruned {len(envs.prune(envs_runner, days, dry_run=False))} idle environments"

    return Executors(run_job, hide, prune)
