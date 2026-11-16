"""Per-game scheduling: the backend's rules, reproduced; idempotent planning; a ticker that retries, never stacks and never runs a started game."""
import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from mllab.pipeline import jobs as J
from mllab.pipeline import pregame as P
from mllab.pipeline import runner as R

UTC = timezone.utc
NOW = datetime(2026, 10, 2, 14, 0, 0, tzinfo=UTC)  # 10:00 ET
DAY = "2026-10-02"


def game(pk, hour, minute=0, state="Preview"):
    return P.Game(pk, DAY, datetime(2026, 10, 2, hour, minute, tzinfo=UTC), state)


G1 = game(101, 23, 5)   # 9 h away: every checkpoint is in the future
G2 = game(102, 17, 10)  # 3 h 10 m away: T-360 has passed
G3 = game(103, 13, 0)   # already started


def by_key(tasks):
    return {t["key"]: t for t in tasks}


# ------------------------------------------------------------------------------------------------------------------------------- plan
def test_a_distant_game_gets_a_baseline_four_checkpoints_and_two_sims():
    tasks = P.plan_tasks([G1], NOW)
    assert len(tasks) == 7
    assert sorted(t["kind"] for t in tasks) == ["baseline", "checkpoint", "checkpoint", "checkpoint", "checkpoint", "sim", "sim"]


def test_only_checkpoints_still_in_the_future_are_planned():
    keys = by_key(P.plan_tasks([G2], NOW))
    assert not any("-t360-" in k for k in keys)  # 11:10Z is before 14:00Z
    assert all(any(f"-t{m}-" in k for k in keys) for m in (180, 90, 45))
    assert len(keys) == 6  # baseline + 3 + 2 sims


def test_a_game_that_has_started_gets_nothing():
    assert P.plan_tasks([G3], NOW) == []
    assert P.plan_tasks([game(104, 14, 0)], NOW) == []  # first pitch exactly now counts as passed


def test_task_keys_are_the_backends_dedupe_keys():
    keys = by_key(P.plan_tasks([G1], NOW))
    fp = "2026-10-02T23:05:00.000Z"  # JavaScript's toISOString(), which is what the backend hashed
    assert f"{DAY}-101-baseline" in keys
    for m in (360, 180, 90, 45):
        assert f"{DAY}-101-t{m}-{fp}" in keys
    for m in (80, 35):
        assert f"{DAY}-101-sim{m}-{fp}" in keys


def test_due_times_are_first_pitch_minus_the_offset_and_the_baseline_is_due_now():
    t = by_key(P.plan_tasks([G1], NOW))
    assert t[f"{DAY}-101-baseline"]["due"] == "2026-10-02T14:00:00Z"
    assert t[f"{DAY}-101-t360-2026-10-02T23:05:00.000Z"]["due"] == "2026-10-02T17:05:00Z"
    assert t[f"{DAY}-101-t45-2026-10-02T23:05:00.000Z"]["due"] == "2026-10-02T22:20:00Z"
    assert t[f"{DAY}-101-sim80-2026-10-02T23:05:00.000Z"]["due"] == "2026-10-02T21:45:00Z"
    assert t[f"{DAY}-101-sim35-2026-10-02T23:05:00.000Z"]["due"] == "2026-10-02T22:30:00Z"


def test_bodies_are_exactly_what_the_backend_sent_to_each_function():
    t = by_key(P.plan_tasks([G1], NOW))
    pre = {"mode": "pregame_v10", "game_pks": [101], "date": DAY, "run_logit3": True}
    assert t[f"{DAY}-101-baseline"]["body"] == pre and t[f"{DAY}-101-t90-2026-10-02T23:05:00.000Z"]["body"] == pre
    assert t[f"{DAY}-101-sim80-2026-10-02T23:05:00.000Z"]["body"] == {"mode": "sim_blend", "game_pks": [101], "date": DAY}  # no lineup_fallback
    assert t[f"{DAY}-101-sim35-2026-10-02T23:05:00.000Z"]["body"] == {"mode": "sim_blend", "game_pks": [101], "date": DAY, "lineup_fallback": True}


def test_sims_can_be_switched_off():
    assert not [t for t in P.plan_tasks([G1], NOW, sim=False) if t["kind"] == "sim"]


def test_fetch_games_skips_final_and_unparseable_games():
    sched = {"dates": [{"games": [
        {"gamePk": 1, "gameDate": "2026-10-02T23:05:00Z", "status": {"abstractGameState": "Preview"}},
        {"gamePk": 2, "gameDate": "2026-10-02T20:05:00Z", "status": {"abstractGameState": "Final"}},
        {"gamePk": 3, "gameDate": "garbage", "status": {"abstractGameState": "Preview"}},
        {"gamePk": 4, "gameDate": "2026-10-02T18:05:00", "status": {"abstractGameState": "Live"}},  # no Z: the backend appended one
    ]}]}
    seen = []
    games = P.fetch_games(DAY, lambda url: seen.append(url) or sched)
    assert [g.pk for g in games] == [1, 4] and "date=2026-10-02" in seen[0] and "sportId=1" in seen[0]
    assert games[1].first_pitch == datetime(2026, 10, 2, 18, 5, tzinfo=UTC)


# ------------------------------------------------------------------------------------------------------------------------------ store
def test_planning_twice_changes_nothing():
    s = P.Store(":memory:")
    tasks = P.plan_tasks([G1, G2], NOW)
    assert s.plan(tasks, NOW) == {"new": 13, "existing": 0, "superseded": 0}
    assert s.plan(tasks, NOW + timedelta(minutes=30)) == {"new": 0, "existing": 13, "superseded": 0}
    assert s.counts() == {"pending": 13}


def test_a_game_whose_first_pitch_moves_gets_new_tasks_and_the_old_ones_are_superseded():
    s = P.Store(":memory:")
    s.plan(P.plan_tasks([G1], NOW), NOW)
    moved = P.Game(101, DAY, datetime(2026, 10, 3, 0, 40, tzinfo=UTC), "Delayed")  # pushed back 1 h 35 m
    res = s.plan(P.plan_tasks([moved], NOW + timedelta(minutes=30)), NOW + timedelta(minutes=30))
    assert res["superseded"] == 6 and res["new"] == 6  # four checkpoints and two sims; the baseline key does not change
    assert s.counts() == {"pending": 7, "superseded": 6}


def test_superseding_never_touches_a_task_that_already_ran():
    s = P.Store(":memory:")
    s.plan(P.plan_tasks([G1], NOW), NOW)
    s.claim(f"{DAY}-101-baseline", NOW)
    s.finish(f"{DAY}-101-baseline", True, None, NOW)
    moved = P.Game(101, DAY, datetime(2026, 10, 3, 1, 0, tzinfo=UTC))
    s.plan(P.plan_tasks([moved], NOW), NOW)
    assert s.counts()["done"] == 1


def test_due_respects_order_next_try_and_state():
    s = P.Store(":memory:")
    s.plan(P.plan_tasks([G1], NOW), NOW)
    assert [r["kind"] for r in s.due(NOW, 10)] == ["baseline"]
    s.claim(f"{DAY}-101-baseline", NOW)
    assert s.due(NOW, 10) == []  # running tasks are not due
    s.finish(f"{DAY}-101-baseline", False, "boom", NOW)
    assert s.due(NOW, 10) == []  # waiting out its backoff
    assert [r["kind"] for r in s.due(NOW + timedelta(seconds=31), 10)] == ["baseline"]


# ------------------------------------------------------------------------------------------------------------------------------- tick
class Clock:
    def __init__(self, t=NOW):
        self.t = t

    def __call__(self):
        return self.t


def tick(store, run_body, clock=None, **kw):
    return P.tick(store, run_body, now=clock or Clock(), poll_s=0.01, **kw)


def due_store(*games, now=NOW):
    s = P.Store(":memory:")
    s.plan(P.plan_tasks(list(games), now), now)
    return s


def test_tick_runs_what_is_due_and_nothing_else():
    s = due_store(G1)
    seen = []
    ran = tick(s, lambda body: seen.append(body) or {"ok": True})
    assert ran["done"] == 1 and seen == [{"mode": "pregame_v10", "game_pks": [101], "date": DAY, "run_logit3": True}]
    assert s.counts() == {"done": 1, "pending": 6}


def test_tick_runs_at_most_three_at_once_and_does_run_three():
    s = P.Store(":memory:")
    games = [game(200 + i, 23, 5) for i in range(6)]
    s.plan(P.plan_tasks(games, NOW), NOW)  # six baselines are due
    live, peak, lock = [0], [0], threading.Lock()

    def body(_):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.15)
        with lock:
            live[0] -= 1
        return {"ok": True}

    ran = tick(s, body)
    assert ran["done"] == 6 and peak[0] == 3  # the function allowed three instances: not five, not one


def test_a_failure_waits_30_then_60_seconds_and_the_third_failure_is_final():
    s = due_store(G1)
    key = f"{DAY}-101-baseline"
    clock = Clock()
    calls = []
    def body(_):
        calls.append(clock.t)
        return {"ok": False, "error": "HTTP 500"}
    assert tick(s, body, clock)["retry"] == 1 and len(calls) == 1
    assert tick(s, body, clock)["retry"] == 0 and len(calls) == 1  # still backing off: nothing runs
    clock.t += timedelta(seconds=31)
    assert tick(s, body, clock)["retry"] == 1 and len(calls) == 2
    clock.t += timedelta(seconds=30)
    tick(s, body, clock)
    assert len(calls) == 2  # the second backoff is 60 s, so 30 s is not enough
    clock.t += timedelta(seconds=31)
    assert tick(s, body, clock)["failed"] == 1 and len(calls) == 3
    row = s.db.execute("SELECT state, attempts, last_error FROM tasks WHERE key=?", (key,)).fetchone()
    assert (row["state"], row["attempts"], row["last_error"]) == ("failed", 3, "HTTP 500")
    clock.t += timedelta(hours=1)
    tick(s, body, clock)
    assert len(calls) == 3  # three attempts and no more, as the queue allowed


def test_a_retry_that_succeeds_ends_done():
    s = due_store(G1)
    clock, outcomes = Clock(), iter([False, True])
    body = lambda _: {"ok": next(outcomes), "error": "x"}
    tick(s, body, clock)
    clock.t += timedelta(seconds=31)
    assert tick(s, body, clock)["done"] == 1
    assert s.counts()["done"] == 1


def test_a_task_whose_game_has_started_is_skipped_not_run():
    s = due_store(G1)
    seen = []
    late = Clock(datetime(2026, 10, 2, 23, 10, tzinfo=UTC))  # 5 minutes after first pitch
    ran = tick(s, lambda b: seen.append(b) or {"ok": True}, late)
    assert not seen and ran == {"done": 0, "retry": 0, "failed": 0, "skipped": 7}  # all seven are overdue, and the game is under way
    assert s.counts() == {"skipped": 7}


def test_skipping_frees_slots_so_later_due_tasks_still_run_in_the_same_tick():
    s = P.Store(":memory:")
    started = [game(300 + i, 14, 30) for i in range(4)]  # planned at 14:00, start at 14:30
    ok_games = [game(310, 23, 5)]
    s.plan(P.plan_tasks(started + ok_games, NOW), NOW)
    clock = Clock(datetime(2026, 10, 2, 14, 31, tzinfo=UTC))  # the four have started; the baselines of all are overdue
    seen = []
    tick(s, lambda b: seen.append(b["game_pks"][0]) or {"ok": True}, clock, max_parallel=2)
    assert 310 in seen and not [x for x in seen if x in (300, 301, 302, 303)]


def test_a_crashed_ticker_does_not_strand_tasks_as_running():
    s = due_store(G1)
    s.claim(f"{DAY}-101-baseline", NOW)  # the process died here
    seen = []
    tick(s, lambda b: seen.append(b) or {"ok": True})
    assert len(seen) == 1 and s.counts()["done"] == 1


def test_an_exception_in_the_run_function_is_a_retry_not_a_stranded_task():
    s = due_store(G1)
    def body(_):
        raise RuntimeError("kaboom")
    assert tick(s, body)["retry"] == 1
    assert s.counts()["pending"] == 7 and "running" not in s.counts()


# ----------------------------------------------------------------------------------------------------------------------------- the CLI
def schedule_for(*games):
    return {"dates": [{"games": [{"gamePk": g.pk, "gameDate": g.first_pitch.strftime("%Y-%m-%dT%H:%M:%SZ"), "status": {"abstractGameState": g.state or "Preview"}}
                                 for g in games]}]}


@pytest.fixture
def dirs(tmp_path):
    return {"jobs_dir": tmp_path / "jobs.d", "state_dir": tmp_path / "state"}


def test_off_plans_and_runs_nothing(dirs):
    called = []
    assert P.cmd_plan(DAY, fetch=lambda u: called.append(u) or schedule_for(G1), now=NOW, **dirs) == 0 and not called
    assert P.cmd_tick(executor=lambda *a: called.append(a), **dirs) == 0 and not called
    assert not (dirs["state_dir"]).exists()


def test_plan_records_tasks_in_the_modes_own_database(dirs):
    R.write_mode(J.PREGAME, "dry", dirs["jobs_dir"])
    assert P.cmd_plan(DAY, fetch=lambda u: schedule_for(G1, G2), now=NOW, **dirs) == 0
    assert P.Store(dirs["state_dir"] / "pregame-dry.sqlite").counts() == {"pending": 13}
    assert not (dirs["state_dir"] / "pregame-live.sqlite").exists()  # a dry plan can never be consumed by a live ticker
    status = json.loads((dirs["state_dir"] / "status" / "pregame-planner.json").read_text())
    assert status["ok"] and status["games"] == 2 and status["new"] == 13


def test_plan_when_the_schedule_is_unreachable_fails_loudly_and_keeps_what_it_has(dirs):
    R.write_mode(J.PREGAME, "live", dirs["jobs_dir"])
    P.cmd_plan(DAY, fetch=lambda u: schedule_for(G1), now=NOW, **dirs)
    def down(url):
        raise OSError("network is down")
    assert P.cmd_plan(DAY, fetch=down, now=NOW, **dirs) == 1
    assert P.Store(dirs["state_dir"] / "pregame-live.sqlite").counts() == {"pending": 7}
    assert json.loads((dirs["state_dir"] / "status" / "pregame-planner.json").read_text())["ok"] is False


class Spy:
    def __init__(self):
        self.payloads = []

    def __call__(self, env, stdin, timeout, cwd=None):
        self.payloads.append(json.loads(stdin))
        return 0, R.RESULT_MARK + json.dumps({"status": 200, "body": {"status": "ok"}}), ""


def test_dry_tick_sends_dry_run_true_and_does_not_heartbeat(dirs):
    R.write_mode(J.PREGAME, "dry", dirs["jobs_dir"])
    P.cmd_plan(DAY, fetch=lambda u: schedule_for(G1), now=NOW, **dirs)
    spy, pushed = Spy(), []
    assert P.cmd_tick(executor=spy, resolve=lambda r: "x", push=lambda *a: pushed.append(a), now=Clock(), release=dirs["state_dir"], **dirs) == 0
    assert [p["dry_run"] for p in spy.payloads] == [True] and spy.payloads[0]["mode"] == "pregame_v10" and not pushed


def test_live_tick_sends_the_real_body_and_heartbeats(dirs):
    R.write_mode(J.PREGAME, "live", dirs["jobs_dir"])
    P.cmd_plan(DAY, fetch=lambda u: schedule_for(G1), now=NOW, **dirs)
    spy, pushed = Spy(), []
    P.cmd_tick(executor=spy, resolve=lambda r: "x", push=lambda ok, msg: pushed.append((ok, msg)), now=Clock(), release=dirs["state_dir"], **dirs)
    assert spy.payloads == [{"mode": "pregame_v10", "game_pks": [101], "date": DAY, "run_logit3": True}]
    assert pushed and pushed[0][0] is True


def test_heartbeat_reports_down_once_a_task_has_permanently_failed(dirs):
    R.write_mode(J.PREGAME, "live", dirs["jobs_dir"])
    P.cmd_plan(DAY, fetch=lambda u: schedule_for(G1), now=NOW, **dirs)
    store = P.Store(dirs["state_dir"] / "pregame-live.sqlite")
    store.db.execute("UPDATE tasks SET state='failed' WHERE kind='baseline'")
    store.db.commit()
    pushed = []
    P.cmd_tick(executor=Spy(), resolve=lambda r: "x", push=lambda ok, msg: pushed.append(ok), now=Clock(), release=dirs["state_dir"], **dirs)
    assert pushed[-1] is False  # the last word to the monitor is "down" while a task has permanently failed


def test_a_second_ticker_does_not_start_while_one_is_running(dirs):
    R.write_mode(J.PREGAME, "live", dirs["jobs_dir"])
    P.cmd_plan(DAY, fetch=lambda u: schedule_for(G1), now=NOW, **dirs)
    spy = Spy()
    with R.lock("pregame-ticker", dirs["state_dir"]):
        assert P.cmd_tick(executor=spy, resolve=lambda r: "x", now=Clock(), release=dirs["state_dir"], **dirs) == 0
    assert not spy.payloads


def test_the_planner_uses_the_eastern_date(dirs):
    R.write_mode(J.PREGAME, "dry", dirs["jobs_dir"])
    seen = []
    late = datetime(2026, 10, 3, 1, 30, tzinfo=UTC)  # 21:30 ET on Oct 2, already Oct 3 in UTC
    P.cmd_plan(None, fetch=lambda u: seen.append(u) or {"dates": []}, now=late, **dirs)
    assert "date=2026-10-02" in seen[0]


def test_the_ticker_sends_heartbeats_while_a_long_task_runs_but_not_in_dry_mode(dirs):
    s = due_store(G1)
    beats = []
    P.tick(s, lambda b: time.sleep(0.12) or {"ok": True}, now=Clock(), poll_s=0.01, heartbeat=lambda: beats.append(1))
    assert len(beats) > 3  # once per loop pass, not once per tick

    sent = []
    R.write_mode(J.PREGAME, "dry", dirs["jobs_dir"])
    P.cmd_plan(DAY, fetch=lambda u: schedule_for(G1), now=NOW, **dirs)
    P.cmd_tick(executor=Spy(), resolve=lambda r: "x", push=lambda *a: sent.append(a), now=Clock(), release=dirs["state_dir"], **dirs)
    assert not sent


def test_live_heartbeats_are_rate_limited_to_about_one_a_minute(dirs):
    R.write_mode(J.PREGAME, "live", dirs["jobs_dir"])
    P.cmd_plan(DAY, fetch=lambda u: schedule_for(G1, G2), now=NOW, **dirs)
    sent = []
    P.cmd_tick(executor=Spy(), resolve=lambda r: "x", push=lambda ok, msg: sent.append(msg), now=Clock(), release=dirs["state_dir"], **dirs)
    assert sum("running" in m for m in sent) == 1  # many loop passes, one heartbeat inside a minute

def test_status_opens_the_state_read_only_and_never_writes(tmp_path):
    import os
    from mllab.pipeline import pregame
    p = tmp_path / "pregame-live.sqlite"
    pregame.Store(p).db.commit()
    os.chmod(p, 0o444)
    s = pregame.Store(p, readonly=True)
    assert s.counts() is not None
    import sqlite3
    import pytest
    with pytest.raises(sqlite3.OperationalError):
        s.db.execute("DELETE FROM tasks")


def test_status_reads_a_wal_database_in_a_directory_it_cannot_write(tmp_path):
    import os
    from mllab.pipeline import pregame
    d = tmp_path / "state"
    d.mkdir()
    p = d / "pregame-live.sqlite"
    st = pregame.Store(p)
    st.db.execute("INSERT INTO tasks (key, game_pk, game_date, kind, first_pitch, due, body, created, updated) "
                  "VALUES ('k', 1, '2026-10-14', 'pre', 'x', 'x', '{}', 'x', 'x')")
    st.db.commit()
    st.db.close()
    for extra in ("-wal", "-shm"):  # a checkpointed database leaves no -shm behind for a reader to open
        if (d / f"pregame-live.sqlite{extra}").exists():
            os.remove(d / f"pregame-live.sqlite{extra}")
    os.chmod(p, 0o444)
    os.chmod(d, 0o555)
    before = (p.stat().st_size, p.stat().st_mtime_ns, sorted(os.listdir(d)))
    try:
        s = pregame.Store(p, readonly=True)
        assert sum(s.counts().values()) == 1
        s.db.close()
        assert (p.stat().st_size, p.stat().st_mtime_ns, sorted(os.listdir(d))) == before  # the live state was not touched
    finally:
        os.chmod(d, 0o755)
