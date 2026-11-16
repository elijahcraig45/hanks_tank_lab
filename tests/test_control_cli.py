import json
from datetime import datetime, timedelta, timezone

import pytest

from mllab import cli
from mllab.control import Control, MemoryStore

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
SHA = "c" * 64


@pytest.fixture
def env(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MLLAB_AUDIT", str(tmp_path / "audit.log"))
    store = MemoryStore()
    clock = {"t": NOW}
    made = lambda: Control(store, "henry", now=lambda: clock["t"], known_models=lambda s: ["v10", "logit3", "sim_blend"] if s == "mlb" else ["xgb", "ridge"],  # noqa: E731
                           production_keys=lambda s: {"v10"} if s == "mlb" else {"xgb"})
    monkeypatch.setattr(cli, "_make_control", made)
    return SimpleEnv(store, clock, capsys)


class SimpleEnv:
    def __init__(self, store, clock, capsys):
        self.store, self.clock, self.capsys = store, clock, capsys

    def run(self, *argv):
        code = cli.main(["control", *argv])
        out = self.capsys.readouterr()
        self.clock["t"] += timedelta(seconds=90)
        return code, out.out, out.err


def test_show_is_honest_when_nothing_is_set(env):
    code, out, _ = env.run("show")
    assert code == 0 and "behave as they always have" in out


def test_hide_needs_the_typed_phrase_then_works_and_shows_in_state(env):
    code, _, err = env.run("hide", "mlb", "logit3")
    assert code == 3 and "confirm with 'mlb/logit3'" in err  # confirmation failures use exit code 3, like deploys
    assert env.run("hide", "mlb", "logit3", "--confirm", "mlb/logit3")[0] == 0
    code, out, _ = env.run("show", "--sport", "mlb")
    assert "logit3" in out and "false" in out
    code, out, _ = env.run("show", "--json")
    assert json.loads(out)["mlb/logit3"]["site_visible"] == "false"


def test_a_typo_cannot_create_a_row_for_a_model_that_does_not_exist(env):
    code, _, err = env.run("hide", "mlb", "v11", "--confirm", "mlb/v11")
    assert code == 1 and "not a mlb model" in err and "known: v10, logit3, sim_blend" in err
    assert env.store.all_events() == []


def test_hiding_production_needs_a_banner_and_hide_with_banner_does_both(env):
    code, _, err = env.run("hide", "mlb", "v10", "--confirm", "mlb/v10")
    assert code == 1 and "needs a banner" in err
    code, out, _ = env.run("hide", "mlb", "v10", "--confirm", "mlb/v10", "--banner", "Rebuilding predictions", "--banner-level", "warn")
    assert code == 0 and out.index("banner") < out.index("site_visible")  # banner first: no silent gap
    assert env.run("unhide", "mlb", "v10")[0] == 0


def test_pause_resume_label_describe_order_tiers(env):
    assert env.run("pause", "mlb", "sim_blend")[0] == 3
    assert env.run("pause", "mlb", "sim_blend", "--confirm", "mlb/sim_blend")[0] == 0
    assert env.run("label", "mlb", "sim_blend", "Simulation blend")[0] == 0
    assert env.run("describe", "mlb", "sim_blend", "Monte Carlo of plate appearances.")[0] == 0
    assert env.run("order", "mlb", "sim_blend", "5")[0] == 0
    assert env.run("tiers", "mlb", "v10", "0.66", "0.58")[0] == 0
    st = env.run("show", "--json")[1]
    row = json.loads(st)
    assert row["mlb/sim_blend"]["run_state"] == "paused" and row["mlb/sim_blend"]["display_label"] == "Simulation blend" and row["mlb/v10"]["tier_high"] == "0.66"
    assert env.run("resume", "mlb", "sim_blend")[0] == 0
    assert json.loads(env.run("show", "--json")[1])["mlb/sim_blend"]["run_state"] == "active"


def test_markup_in_a_label_is_refused_before_anything_is_written(env):
    code, _, err = env.run("label", "mlb", "logit3", "<b>hi</b>")
    assert code == 1 and "markup" in err and env.store.all_events() == []


def test_pin_unpin_and_the_sha_requirement(env):
    assert env.run("pin", "mlb", "v10", "gs://hanks_tank_data/models/vertex/x/model.pkl", SHA)[0] == 3
    assert env.run("pin", "mlb", "v10", "gs://hanks_tank_data/models/vertex/x/model.pkl", "short", "--confirm", "mlb/v10")[0] == 1
    assert env.run("pin", "mlb", "v10", "gs://hanks_tank_data/models/vertex/x/model.pkl", SHA, "--confirm", "mlb/v10")[0] == 0
    assert json.loads(env.run("show", "--json")[1])["mlb/v10"]["lifecycle"] == "live"
    assert env.run("unpin", "mlb", "v10")[0] == 3
    assert env.run("unpin", "mlb", "v10", "--confirm", "mlb/v10")[0] == 0
    assert "lifecycle" not in json.loads(env.run("show", "--json")[1]).get("mlb/v10", {})


def test_banner_set_and_clear(env):
    assert env.run("banner", "nfl", "Scores delayed today")[0] == 3
    assert env.run("banner", "nfl", "Scores delayed today", "--level", "warn", "--confirm", "nfl/*")[0] == 0
    row = json.loads(env.run("show", "--json")[1])["nfl/*"]
    assert row["banner"] == "Scores delayed today" and row["banner_level"] == "warn"
    assert env.run("banner-clear", "nfl")[0] == 0
    assert "nfl/*" not in json.loads(env.run("show", "--json")[1])
    assert env.run("banner", "nfl", "<script>", "--confirm", "nfl/*")[0] == 1


def test_scheduled_change_history_and_revert(env):
    later = (NOW + timedelta(days=2)).isoformat()
    code, out, _ = env.run("set", "mlb", "logit3", "display_label", "Later name", "--at", later)
    assert code == 0 and "effective" in out
    assert "display_label" not in json.loads(env.run("show", "--json")[1]).get("mlb/logit3", {})  # not yet
    assert env.run("set", "mlb", "logit3", "display_label", "Naive time", "--at", "2026-12-01T09:00:00")[0] == 1  # no timezone
    env.run("label", "mlb", "logit3", "One"); env.run("label", "mlb", "logit3", "Two")
    assert env.run("revert", "mlb", "logit3", "display_label")[0] == 0
    assert json.loads(env.run("show", "--json")[1])["mlb/logit3"]["display_label"] == "One"
    code, out, _ = env.run("history", "mlb", "logit3", "--field", "display_label")
    assert code == 0 and out.count("display_label") >= 3 and "henry" in out


def test_freeze_and_unfreeze(env):
    assert env.run("freeze", "nfl", "Maintenance")[0] == 3
    assert env.run("freeze", "nfl", "Maintenance", "--confirm", "nfl/*")[0] == 0
    rows = json.loads(env.run("show", "--json")[1])
    assert rows["nfl/xgb"]["site_visible"] == "false" and rows["nfl/*"]["banner"] == "Maintenance"
    assert env.run("unfreeze", "nfl")[0] == 0
    assert "banner" not in json.loads(env.run("show", "--json")[1]).get("nfl/*", {})


def test_scheduler_subcommands_delegate_with_the_confirmation(monkeypatch, capsys):
    calls = []

    class FakeSched:
        def list(self):
            return [{"name": "mlb-2026-daily", "state": "ENABLED", "schedule": "0 4 * * *", "tz": "ET", "mode": "daily", "body_ok": True, "last_attempt_ok": True}]

        def pause(self, job, **kw):
            calls.append(("pause", job, kw)); return {"detail": f"pause {job}"}

        def run_now(self, job, **kw):
            calls.append(("run", job, kw)); return {"detail": f"run {job}", "dry_run": kw["dry_run"]}

    monkeypatch.setattr("mllab.scheduler.Scheduler", FakeSched)
    assert cli.main(["control", "scheduler", "list"]) == 0 and "mlb-2026-daily" in capsys.readouterr().out
    assert cli.main(["control", "scheduler", "pause", "mlb-2026-daily", "--confirm", "mlb-2026-daily"]) == 0
    assert cli.main(["control", "scheduler", "run", "mlb-2026-daily", "--dry-run"]) == 0
    assert calls[0] == ("pause", "mlb-2026-daily", {"confirm": "mlb-2026-daily", "dry_run": False}) and calls[1][2]["dry_run"] is True


def test_scheduler_errors_exit_nonzero(monkeypatch, capsys):
    from mllab.scheduler import SchedulerError

    class Boom:
        def pause(self, *a, **k):
            raise SchedulerError("no such job 'x'")

    monkeypatch.setattr("mllab.scheduler.Scheduler", Boom)
    assert cli.main(["control", "scheduler", "pause", "x"]) == 1 and "no such job" in capsys.readouterr().err
