"""Weekly job: run the data checks and any backtests listed in a config, record the outcome, and let the health probe report it.

config (YAML):
  bq_checks: [specs/bq/games.yaml, ...]        # `mllab bq check` specs; any error fails the run
  backtests: [specs/mlb_v10_param_sweep.yaml]  # submitted and awaited; a verdict CHANGE is recorded, not alarmed
  backtest_timeout_s: 3000

Result: <root>/audit/weekly.json (previous one kept as weekly.prev.json). mllab-health turns a failed or stale run into a Kuma "down".
A verdict change is information, not an incident, so it never makes the run fail.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


class WeeklyError(Exception):
    pass


def load_config(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    extra = set(raw) - {"bq_checks", "backtests", "backtest_timeout_s"}
    if extra:
        raise WeeklyError(f"unknown config key(s): {sorted(extra)}")
    for k in ("bq_checks", "backtests"):
        v = raw.get(k, [])
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise WeeklyError(f"{k} must be a list of spec file paths")
    if not (raw.get("bq_checks") or raw.get("backtests")):
        raise WeeklyError("nothing to run: list bq_checks and/or backtests")
    return raw


def result_path(root: Path) -> Path:
    return root / "audit" / "weekly.json"


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


def run_weekly(cfg: dict, *, root: Path, base: Path, check_fn, submit_fn, status_fn, verdict_fn, sleep=time.sleep, now=None) -> dict:
    """check_fn(path)->Report | submit_fn(path)->job_id | status_fn(job_id)->state | verdict_fn(job_id)->str. All injectable."""
    t0 = time.time()
    prev_file = result_path(root)
    prev = json.loads(prev_file.read_text()) if prev_file.exists() else {}
    prev_verdicts = {b["spec"]: b.get("verdict") for b in prev.get("backtests", [])}
    res = {"ran_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"), "checks": [], "backtests": [], "errors": []}

    for spec in cfg.get("bq_checks", []):
        try:
            rep = check_fn(base / spec)
            errs = [f"{f.check}: {f.message}" for f in rep.findings if f.level == "error"]
            res["checks"].append({"spec": spec, "table": rep.table, "ok": rep.ok, "errors": errs})
            if not rep.ok:
                res["errors"].append(f"data check failed: {rep.table} ({errs[0] if errs else 'see report'})")
        except Exception as e:  # noqa: BLE001 - one broken spec must not stop the others
            res["checks"].append({"spec": spec, "ok": False, "errors": [f"{type(e).__name__}: {str(e)[:160]}"]})
            res["errors"].append(f"data check could not run: {spec} ({type(e).__name__})")

    timeout = int(cfg.get("backtest_timeout_s", 3000))
    for spec in cfg.get("backtests", []):
        entry = {"spec": spec}
        try:
            job = submit_fn(base / spec)
            entry["job"] = job
            waited = 0
            while status_fn(job) in ("queued", "running") and waited < timeout:
                sleep(5); waited += 5
            entry["state"] = status_fn(job)
            if entry["state"] == "finished":
                entry["verdict"] = verdict_fn(job)
                old = prev_verdicts.get(spec)
                entry["changed_vs_last"] = bool(old) and old != entry["verdict"]
                if entry["changed_vs_last"]:
                    entry["previous_verdict"] = old
            else:
                res["errors"].append(f"backtest {spec} ended {entry['state']}")
        except Exception as e:  # noqa: BLE001
            entry["state"] = "error"; entry["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            res["errors"].append(f"backtest could not run: {spec} ({type(e).__name__})")
        res["backtests"].append(entry)

    res["ok"] = not res["errors"]
    res["duration_s"] = round(time.time() - t0, 1)
    if prev_file.exists():
        _atomic_write(prev_file.with_name("weekly.prev.json"), prev)
    _atomic_write(prev_file, res)
    return res


def render(res: dict) -> str:
    L = [f"weekly run {res['ran_at']}: {'OK' if res['ok'] else 'PROBLEMS'} ({res['duration_s']}s)"]
    for c in res["checks"]:
        L.append(f"  check {'ok  ' if c['ok'] else 'FAIL'} {c.get('table', c['spec'])}" + ("" if c["ok"] else f"  {c['errors'][0]}"))
    for b in res["backtests"]:
        L.append(f"  backtest {b.get('state', '?'):9} {b['spec']}  {b.get('verdict', b.get('error', ''))}"
                 + (f"  [CHANGED from: {b['previous_verdict']}]" if b.get("changed_vs_last") else ""))
    return "\n".join(L)
