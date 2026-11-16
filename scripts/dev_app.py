#!/usr/bin/env python3
"""Run the control app locally against fake data, to look at it and click through it. Nothing here touches GCP, the homelab or any real file.

    PYTHONPATH=src .venv/bin/python scripts/dev_app.py            # http://127.0.0.1:8890/app/   (lab password: devpass)
    PYTHONPATH=src .venv/bin/python scripts/dev_app.py --port 9000

The fake "Cloud Scheduler" has two jobs enabled so the gate has something to refuse, and one job that is Live and still enabled (the two-writers case).
It also plays the broker's part: requests the app drops in the spool are applied to the fake mode files, so a click goes all the way round."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

LOGIN, SECRET = "dev@example.com", "devpass"


def seed(root: Path) -> dict:
    from mllab.pipeline import broker, runner
    from mllab.pipeline.jobs import JOBS
    from mllab.registry import Registry
    for d in ("status", "spool", "results", "registry", "work", "state", "pjobs", "jobs", "health", "audit"):
        (root / d).mkdir(parents=True, exist_ok=True)
    modes = {"mlb-2026-daily": "dry", "mlb-2026-roster-refresh": "dry", "nfl-weekly-predict": "live", "nfl-weekly-ingest": "dry", "cfb-weekly-predict": "off", "cfb-weekly-drives": "live", "pregame": "dry"}
    for name, mode in modes.items():
        runner.write_mode(name, mode, root / "pjobs")
    from datetime import datetime, timedelta, timezone
    fin = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(timespec="seconds")
    for name, ok, err in (("mlb-2026-daily", True, None), ("nfl-weekly-ingest", True, None), ("nfl-weekly-predict", False, "KeyError: 'week' (no games scheduled)")):
        runner.record(name, {"started": fin, "finished": fin, "ok": ok, "seconds": 372, "error": err}, root / "state",
                      log_text=f"[{name}] starting\nloaded 14 games\n{'ERROR ' + err if err else 'done: wrote 14 rows (dry run: nothing written)'}\n")
    (root / "health" / "health.json").write_text(json.dumps({
        "taken_at": broker._now(), "sources": {"control": "ok", "bigquery:mlb": "ok", "scheduler": "ok", "logs": "unavailable: no permission"}, "lab": {"disk_used_pct": 23.1},
        "findings": [{"code": "shadow_stale", "severity": "warn", "message": "logit3 has not predicted in 31h", "sport": "mlb", "model": "logit3", "remedy": None},
                     {"code": "coverage", "severity": "critical", "message": "only 3 of 15 games have a v10 prediction", "sport": "mlb", "model": "v10", "remedy": None}],
        "sports": {"mlb": {"sport": "mlb", "models": [{"model": "v10", "status": "down", "last_prediction_at": "2026-10-02T23:16:44+00:00", "n_recent": 184, "prob_mean": 0.531, "is_shadow": False},
                                                     {"model": "logit3", "status": "warn", "last_prediction_at": "2026-10-01T15:00:00+00:00", "n_recent": 284, "prob_mean": 0.5316, "is_shadow": True}]}}}))
    art = root / "m.pkl"
    art.write_bytes(b"model")
    reg = Registry(root / "registry")
    reg.register("hankstank", "game_outcome", "v10", art, metrics={"accuracy": 0.5398, "log_loss": 0.6891})
    reg.register("hankstank", "game_outcome", "v11", art, parent="v10", metrics={"accuracy": 0.5421, "log_loss": 0.6870}, features=["a", "b", "c"])
    reg.promote("hankstank", "game_outcome", "v10", "prod")
    reg.promote("hankstank", "game_outcome", "v11", "prod")   # moved once, so there is a rollback to show
    for i, (name, state) in enumerate((("backtest-v10-v11", "finished"), ("train-v11-d2", "failed"))):
        jd = root / "jobs" / f"2026100{i + 1}1200-abc12{i}"
        jd.mkdir(parents=True, exist_ok=True)
        (jd / "spec.json").write_text(json.dumps({"name": name, "submitted_at": f"2026-10-0{i + 1}T12:00:00+00:00", "memory_mb": 2048, "timeout_s": 3600}))
        (jd / "status.json").write_text(json.dumps({"state": state}))
        (jd / "log.txt").write_text(f"{name}: started\nstep 1 ok\n")
    (root / "work" / "01_overview.ipynb").write_text("{}")
    (root / "audit" / "deploys.log").write_text("".join(json.dumps(e) + "\n" for e in (
        {"ts": "2026-10-02T14:21:00+00:00", "actor": "pipeline-broker", "action": "pipeline_set_mode", "job": "mlb-2026-daily", "by": LOGIN, "before": "off", "mode": "dry"},
        {"ts": "2026-10-02T16:00:00+00:00", "actor": "pipeline-broker", "action": "pipeline_set_mode", "job": "nfl-weekly-predict", "by": LOGIN, "before": "dry", "mode": "live"},
        {"ts": "2026-10-02T16:30:00+00:00", "actor": "pipeline-broker", "action": "pipeline_set_mode", "job": "cfb-weekly-drives", "by": LOGIN, "before": "dry", "mode": "live"},
        {"ts": "2026-10-02T15:00:00+00:00", "actor": "mllab", "action": "promote", "project": "hankstank", "model": "game_outcome", "channel": "prod", "version": "v10", "previous": None},
        {"ts": "2026-10-02T17:00:00+00:00", "actor": "mllab", "action": "promote", "project": "hankstank", "model": "game_outcome", "channel": "prod", "version": "v11", "previous": "v10", "by": LOGIN},
        {"ts": "2026-10-02T18:00:00+00:00", "actor": "henry", "action": "scheduler_pause", "job": "nfl-weekly-score", "was": "ENABLED"})))
    return modes


class FakeBq:
    """Table metadata for the Data page, generated from the real catalog with a mix of fresh, late, stale and missing tables."""

    def __init__(self, control_store=None):
        from mllab import datamap
        self.cat = datamap.load(datamap.find_catalog())
        self.control_store = control_store

    def query(self, sql, params=None):
        import re
        import time
        from datetime import datetime, timezone
        if "INFORMATION_SCHEMA" in sql:                      # the Site page asks whether the control view has the rankings columns
            return [{"column_name": c} for c in ("sport", "target", "rankings_show", "rankings_media", "rankings_primary", "banner")]
        if "model_control_current" in sql:                   # the control state the Site page shows, from the in-memory store the app also writes to
            from mllab import control as K
            st = K.current_state(self.control_store.all_events(), datetime.now(timezone.utc)) if self.control_store else {}
            return [{"sport": sp, "target": t, **{k: v for k, v in row.items() if k != "updated_at"}} for (sp, t), row in st.items()]
        ds = re.search(r"\.([a-z0-9_]+)\.__TABLES__", sql).group(1)
        now = time.time() * 1000
        out = []
        for i, t in enumerate(x for x in self.cat if x.id.startswith(ds + ".")):
            if t.id.endswith(("fpi_game_predictions", "season_sim_bracket")) and t.sport == "cfb":
                continue                                    # a declared table BigQuery does not have: shows as Missing
            age_h = {"nfl_season.stat_leaders": 485, "cfb_season.player_season_stats": 196}.get(t.id, 1 + (i * 7) % 20)
            out.append({"table_id": t.id.split(".")[1], "row_count": str(100 * (i + 3)), "size_bytes": str(50_000 * (i + 1)), "last_modified_time": str(int(now - age_h * 3600_000))})
        return out


class FakeScheduler:
    ENABLED = {"mlb-2026-daily", "nfl-weekly-predict", "nfl-weekly-ingest", "cfb-weekly-predict", "mlb-2026-pregame-schedule"}

    def list(self):
        from mllab.pipeline.jobs import JOBS
        names = set(JOBS) | {"mlb-2026-pregame-schedule"}
        return [{"name": n, "state": "ENABLED" if n in self.ENABLED else "PAUSED", "schedule": "0 6 * * 2", "tz": "America/New_York", "mode": "x", "body_ok": True,
                 "last_attempt_at": "2026-10-03T08:00:00Z", "last_attempt_ok": True, "next_run": None} for n in sorted(names)]


def fake_broker(root: Path, stop: threading.Event):
    """Plays the broker: refresh the snapshot, and apply requests to the fake mode files."""
    from mllab.pipeline import broker
    while not stop.is_set():
        broker.RESULTS, broker.AUDIT = root / "results", root / "audit" / "deploys.log"
        broker.shutil.chown = lambda *a, **k: None
        broker.cmd_apply(spool=root / "spool", owner=os.environ.get("USER", "root"), jobs_dir=root / "pjobs", run=lambda argv, **kw: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(), log=lambda *_: None)
        broker.cmd_snapshot(jobs_dir=root / "pjobs", state_dir=root / "state", release=root / "none", show=lambda u: {"NextElapseUSecRealtime": "Sat 2026-10-04 06:00:00 EDT"}, status_dir=root / "status")
        stop.wait(1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8890)
    a = ap.parse_args()
    root = Path(tempfile.mkdtemp(prefix="mllab-dev-"))
    os.environ.update(MLLAB_ALLOWED_LOGIN=LOGIN, MLLAB_SECRET=SECRET, MLLAB_ROOT=str(root), MLLAB_AUDIT=str(root / "audit" / "deploys.log"))
    seed(root)
    from fastapi import FastAPI
    from mllab.webapp import routes
    cfg = routes.Config(root=root, registry=root / "registry", status_dir=root / "status", spool=root / "spool", results=root / "results", audit_path=root / "audit" / "deploys.log")
    app = FastAPI()
    from mllab.control import MemoryStore
    store = MemoryStore()
    app.include_router(routes.build(cfg, scheduler=FakeScheduler(), bq=FakeBq(store), control_store=store))

    @app.middleware("http")
    async def tailscale(request, call_next):  # stands in for `tailscale serve`, which adds this header on the real host
        scope = request.scope
        scope["headers"] = [(k, v) for k, v in scope["headers"] if k != b"tailscale-user-login"] + [(b"tailscale-user-login", LOGIN.encode())]
        return await call_next(request)

    stop = threading.Event()
    threading.Thread(target=fake_broker, args=(root, stop), daemon=True).start()
    time.sleep(1.5)
    print(f"preview: http://127.0.0.1:{a.port}/app/    lab password: {SECRET}    (fake data in {root})")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
    stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
