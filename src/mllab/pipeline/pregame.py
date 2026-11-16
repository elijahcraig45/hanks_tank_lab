"""Per-game MLB scheduling on the homelab: replaces the backend's /api/lineup/schedule-today and the Cloud Tasks queues `lineup-pregame` and
`sim-blend`. The rules are the backend's (src/services/lineup-scheduler.service.ts), reproduced exactly:

  every game that is not Final and has not started gets
    a baseline pregame run now,
    pregame runs at first pitch minus 360, 180, 90 and 45 minutes (only those still in the future when it is planned),
    sim_blend runs at minus 80 and minus 35 (the second with lineup_fallback).
  A task is keyed by game, checkpoint and the first pitch it was computed from, so planning twice changes nothing and a game whose first
  pitch moves gets new tasks. Games whose first pitch has passed are skipped.

Two commands, both safe to run as often as you like:
    python -m mllab.pipeline.pregame plan [--date YYYY-MM-DD]   read the day's schedule and record the tasks (timer: every 30 min)
    python -m mllab.pipeline.pregame tick                       run what is due, 3 at a time, with retries (timer: every minute)
    python -m mllab.pipeline.pregame status

State is a small sqlite file per mode, so a reboot loses nothing and dry-run tasks can never consume live ones. Mode is /etc/pipeline/jobs.d/pregame:
off does nothing, dry sends dry_run=true (writes nothing), live is real. The queue allowed 5 concurrent dispatches against a function with 3
instances, which dropped tasks; the ticker runs 3 at a time and nothing is dropped, only delayed."""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from . import runner
from .jobs import BUNDLES, PREGAME

ET = ZoneInfo("America/New_York")
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
OFFSETS = (360, 180, 90, 45)
SIM = ((80, False), (35, True))
MAX_ATTEMPTS = 3  # the queue's maxAttempts
BACKOFF_S = (30, 60, 120)  # the queue's minBackoff 30 s, doubling
MAX_PARALLEL = 3  # the function's maxInstances
TASK_TIMEOUT_S = 1800
TICK_MAX_S = 3300

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
  key TEXT PRIMARY KEY, game_pk INTEGER NOT NULL, game_date TEXT NOT NULL, kind TEXT NOT NULL, first_pitch TEXT NOT NULL,
  due TEXT NOT NULL, body TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
  next_try TEXT, last_error TEXT, created TEXT NOT NULL, updated TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS tasks_due ON tasks(state, due);
"""


def _utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc)


def stamp(dt: datetime) -> str:
    """Sortable UTC text. Comparing these strings compares the times."""
    return _utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def js_iso(dt: datetime) -> str:
    """JavaScript's Date.toISOString(), which the backend put into its dedupe keys."""
    return _utc(dt).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def parse(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


# ------------------------------------------------------------------------------------------------------------------------------- plan
@dataclass(frozen=True)
class Game:
    pk: int
    date: str
    first_pitch: datetime
    state: str = ""


def _http_json(url: str, timeout: float = 20.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "mllab-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https MLB URL
        return json.loads(r.read().decode("utf-8"))


def fetch_games(day: str, fetch: Callable[[str], dict] = _http_json) -> list[Game]:
    """Today's non-final games from the MLB schedule, as the backend reads them (it skips only Final)."""
    out = []
    for d in (fetch(SCHEDULE_URL.format(date=day)) or {}).get("dates", []):
        for g in d.get("games", []):
            state = (g.get("status") or {}).get("abstractGameState", "")
            if state == "Final":
                continue
            raw = g.get("gameDate") or ""
            try:
                fp = parse(raw if raw.endswith("Z") else raw + "Z")
            except ValueError:
                continue  # the backend logs 'Invalid game_time_utc' and skips
            out.append(Game(int(g["gamePk"]), day, fp, state))
    return out


def plan_tasks(games: list[Game], now: datetime, sim: bool = True) -> list[dict]:
    tasks = []
    for g in games:
        if g.first_pitch <= now:
            continue  # first pitch has already passed
        fp = js_iso(g.first_pitch)
        base = {"game_pk": g.pk, "game_date": g.date, "first_pitch": stamp(g.first_pitch)}
        pregame = {"mode": "pregame_v10", "game_pks": [g.pk], "date": g.date, "run_logit3": True}
        tasks.append({**base, "key": f"{g.date}-{g.pk}-baseline", "kind": "baseline", "due": stamp(now), "body": pregame})
        for m in OFFSETS:
            due = g.first_pitch - timedelta(minutes=m)
            if due > now:
                tasks.append({**base, "key": f"{g.date}-{g.pk}-t{m}-{fp}", "kind": "checkpoint", "due": stamp(due), "body": pregame})
        if sim:
            for m, fallback in SIM:
                due = g.first_pitch - timedelta(minutes=m)
                if due > now:
                    body = {"mode": "sim_blend", "game_pks": [g.pk], "date": g.date, **({"lineup_fallback": True} if fallback else {})}
                    tasks.append({**base, "key": f"{g.date}-{g.pk}-sim{m}-{fp}", "kind": "sim", "due": stamp(due), "body": body})
    return tasks


# ------------------------------------------------------------------------------------------------------------------------------ state
class Store:
    def __init__(self, path: str | Path, readonly: bool = False):
        if readonly:  # `pipeline-ctl status` as a login that cannot write the state dir: look, never create or alter
            self.db = self._open_readonly(Path(path))
            return
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    @staticmethod
    def _open_readonly(path: Path) -> sqlite3.Connection:
        """Open without writing. A WAL database needs its -shm file even to be read, and a login that cannot write the
        state dir cannot create it ("attempt to write a readonly database"), so fall back to a private copy."""
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        try:
            db.execute("SELECT 1 FROM tasks LIMIT 1").fetchall()
            return db
        except sqlite3.OperationalError:
            db.close()
        tmp = Path(tempfile.mkdtemp(prefix="pregame-ro-"))
        for suffix in ("", "-wal"):  # the WAL holds the newest rows
            if Path(f"{path}{suffix}").exists():
                shutil.copy2(f"{path}{suffix}", tmp / f"{path.name}{suffix}")
        db = sqlite3.connect(str(tmp / path.name))
        db.row_factory = sqlite3.Row
        return db

    def plan(self, tasks: list[dict], now: datetime) -> dict:
        n = stamp(now)
        new = same = superseded = 0
        for t in tasks:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO tasks (key, game_pk, game_date, kind, first_pitch, due, body, created, updated) VALUES (?,?,?,?,?,?,?,?,?)",
                (t["key"], t["game_pk"], t["game_date"], t["kind"], t["first_pitch"], t["due"], json.dumps(t["body"]), n, n))
            new += cur.rowcount
            same += 1 - cur.rowcount
        # A game whose first pitch moved: its not-yet-run checkpoints for the old time are dead.
        for (pk, day), fp in {(t["game_pk"], t["game_date"]): t["first_pitch"] for t in tasks}.items():
            superseded += self.db.execute(
                "UPDATE tasks SET state='superseded', updated=? WHERE game_pk=? AND game_date=? AND kind IN ('checkpoint','sim') "
                "AND first_pitch != ? AND state='pending'", (n, pk, day, fp)).rowcount
        self.db.commit()
        return {"new": new, "existing": same, "superseded": superseded}

    def due(self, now: datetime, limit: int) -> list[sqlite3.Row]:
        n = stamp(now)
        return self.db.execute("SELECT * FROM tasks WHERE state='pending' AND due<=? AND (next_try IS NULL OR next_try<=?) ORDER BY due, key LIMIT ?",
                               (n, n, limit)).fetchall()

    def claim(self, key: str, now: datetime) -> None:
        self.db.execute("UPDATE tasks SET state='running', attempts=attempts+1, updated=? WHERE key=?", (stamp(now), key))
        self.db.commit()

    def skip(self, key: str, reason: str, now: datetime) -> None:
        self.db.execute("UPDATE tasks SET state='skipped', last_error=?, updated=? WHERE key=?", (reason, stamp(now), key))
        self.db.commit()

    def finish(self, key: str, ok: bool, error: str | None, now: datetime) -> str:
        attempts = self.db.execute("SELECT attempts FROM tasks WHERE key=?", (key,)).fetchone()["attempts"]
        if ok:
            state, nxt = "done", None
        elif attempts >= MAX_ATTEMPTS:
            state, nxt = "failed", None
        else:
            state, nxt = "pending", stamp(now + timedelta(seconds=BACKOFF_S[min(attempts, len(BACKOFF_S)) - 1]))
        self.db.execute("UPDATE tasks SET state=?, next_try=?, last_error=?, updated=? WHERE key=?", (state, nxt, (error or "")[:300] or None, stamp(now), key))
        self.db.commit()
        return state

    def reap_running(self, now: datetime) -> int:
        """Tasks left 'running' by a crashed ticker. The ticker holds a lock, so at startup nothing is genuinely running."""
        n = self.db.execute("UPDATE tasks SET state='pending', updated=? WHERE state='running'", (stamp(now),)).rowcount
        self.db.commit()
        return n

    def counts(self) -> dict:
        return {r["state"]: r["n"] for r in self.db.execute("SELECT state, COUNT(*) n FROM tasks GROUP BY state")}

    def next_due(self) -> str | None:
        r = self.db.execute("SELECT MIN(due) d FROM tasks WHERE state='pending'").fetchone()
        return r["d"]


# ------------------------------------------------------------------------------------------------------------------------------ tick
def tick(store: Store, run_body: Callable[[dict], dict], *, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
         max_parallel: int = MAX_PARALLEL, poll_s: float = 5.0, max_seconds: float = TICK_MAX_S, clock: Callable[[], float] = time.monotonic,
         heartbeat: Callable[[], None] = lambda: None) -> dict:
    """Run everything that is due, `max_parallel` at a time, until nothing is due and nothing is running."""
    store.reap_running(now())
    ran = {"done": 0, "retry": 0, "failed": 0, "skipped": 0}
    deadline = clock() + max_seconds
    running: dict = {}
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        while True:
            heartbeat()  # a long tick must not look like a dead ticker to the monitor
            skipped_now = 0
            if clock() < deadline:
                for t in store.due(now(), max_parallel - len(running)):
                    if parse(t["first_pitch"]) <= now():
                        store.skip(t["key"], "game started", now())
                        ran["skipped"] += 1
                        skipped_now += 1
                        continue
                    store.claim(t["key"], now())
                    running[ex.submit(run_body, json.loads(t["body"]))] = t["key"]
            if not running:
                if skipped_now:
                    continue  # skipped tasks freed their slots: look for more that are due
                break
            done, _ = wait(list(running), timeout=poll_s, return_when=FIRST_COMPLETED)
            for f in done:
                key = running.pop(f)
                try:
                    res = f.result()
                except Exception as e:  # a bug in run_body must not strand the task as 'running'
                    res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                state = store.finish(key, bool(res.get("ok")), res.get("error"), now())
                ran["done" if state == "done" else "failed" if state == "failed" else "retry"] += 1
    return ran


# ------------------------------------------------------------------------------------------------------------------------------- CLI
def _store(mode: str, state_dir: Path | None) -> Store:
    return Store((state_dir or runner.STATE_DIR) / f"pregame-{mode}.sqlite")


def _write_status(name: str, data: dict, state_dir: Path | None) -> None:
    d = (state_dir or runner.STATE_DIR) / "status"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps({**data, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")


def cmd_plan(day: str | None = None, *, jobs_dir=None, state_dir=None, fetch=_http_json, now=None) -> int:
    mode = runner.read_mode(PREGAME, jobs_dir)
    if mode == "off":
        print("pregame: mode is off; nothing to plan")
        return 0
    now = now or datetime.now(timezone.utc)
    day = day or now.astimezone(ET).date().isoformat()
    try:
        games = fetch_games(day, fetch)
    except Exception as e:  # keep whatever is already planned; the next run tries again
        _write_status("pregame-planner", {"ok": False, "error": f"{type(e).__name__}: {e}"[:200], "mode": mode, "date": day}, state_dir)
        print(f"pregame plan: could not read the schedule: {e}", file=sys.stderr)
        return 1
    res = _store(mode, state_dir).plan(plan_tasks(games, now), now)
    _write_status("pregame-planner", {"ok": True, "mode": mode, "date": day, "games": len(games), **res}, state_dir)
    print(json.dumps({"date": day, "games": len(games), **res}))
    return 0


def cmd_tick(*, jobs_dir=None, state_dir=None, release=None, executor=None, resolve=None, push=None, now=None) -> int:
    mode = runner.read_mode(PREGAME, jobs_dir)
    if mode == "off":
        print("pregame: mode is off; nothing to run")
        return 0
    kw = {k: v for k, v in (("executor", executor), ("resolve", resolve)) if v is not None}

    def run_body(body: dict) -> dict:
        payload = {**body, "dry_run": True} if mode == "dry" else body
        return runner.execute(BUNDLES["mlb"], payload, timeout=TASK_TIMEOUT_S, release=release, **kw)

    try:
        with runner.lock("pregame-ticker", state_dir):
            store = _store(mode, state_dir)
            sent = [float("-inf")]

            def heartbeat():
                if mode == "live" and time.monotonic() - sent[0] >= 60:
                    sent[0] = time.monotonic()
                    (push or runner.kuma_push)(True, "pregame ticker running")

            ran = tick(store, run_body, heartbeat=heartbeat, **({"now": now} if now else {}))
            counts = store.counts()
            _write_status("pregame-ticker", {"ok": True, "mode": mode, "ran": ran, "counts": counts, "next_due": store.next_due()}, state_dir)
            print(json.dumps({"mode": mode, "ran": ran, "counts": counts}))
            if mode == "live":
                (push or runner.kuma_push)(counts.get("failed", 0) == 0, f"pregame tick: {ran}")
            return 0
    except runner.Busy:
        print("pregame: a ticker is already running")
        return 0


def cmd_status(*, state_dir=None, jobs_dir=None) -> int:
    mode = runner.read_mode(PREGAME, jobs_dir)
    print(f"mode: {mode}")
    for m in ("dry", "live"):
        p = (state_dir or runner.STATE_DIR) / f"pregame-{m}.sqlite"
        if p.exists():
            s = Store(p, readonly=True)
            print(f"  {m}: {s.counts()}  next due {s.next_due()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    cmd = a[0] if a else ""
    if cmd == "plan":
        return cmd_plan(a[a.index("--date") + 1] if "--date" in a else None)
    if cmd == "tick":
        return cmd_tick()
    if cmd == "status":
        return cmd_status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
