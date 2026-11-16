"""pipeline-ctl: going live must make the operator face what has to be switched off first."""
import json

import pytest

from mllab.pipeline import ctl as C
from mllab.pipeline import jobs as J
from mllab.pipeline import pregame as P
from mllab.pipeline import runner as R


@pytest.fixture
def d(tmp_path):
    return {"jobs_dir": tmp_path / "jobs.d", "state_dir": tmp_path / "state"}


class Say(list):
    def __call__(self, *a):
        self.append(" ".join(str(x) for x in a))

    @property
    def text(self):
        return "\n".join(self)


def test_off_and_dry_need_no_confirmation(d):
    out = Say()
    assert C.cmd_set("mlb-2026-daily", "dry", jobs_dir=d["jobs_dir"], input_fn=lambda p: pytest.fail("must not ask"), out=out) == 0
    assert R.read_mode("mlb-2026-daily", d["jobs_dir"]) == "dry"
    assert C.cmd_set("mlb-2026-daily", "off", jobs_dir=d["jobs_dir"], input_fn=lambda p: pytest.fail("must not ask"), out=out) == 0


def test_going_live_names_the_scheduler_job_to_pause_and_asks(d):
    out = Say()
    asked = []
    assert C.cmd_set("nfl-weekly-ingest", "live", jobs_dir=d["jobs_dir"], input_fn=lambda p: asked.append(p) or "y", out=out) == 0
    assert asked and "mllab control scheduler pause nfl-weekly-ingest --confirm nfl-weekly-ingest" in out.text
    assert R.read_mode("nfl-weekly-ingest", d["jobs_dir"]) == "live"


@pytest.mark.parametrize("answer", ["", "n", "no", "maybe", "yess"])
def test_anything_but_yes_leaves_the_mode_unchanged(d, answer):
    out = Say()
    assert C.cmd_set("nfl-weekly-ingest", "live", jobs_dir=d["jobs_dir"], input_fn=lambda p: answer, out=out) == 1
    assert R.read_mode("nfl-weekly-ingest", d["jobs_dir"]) == "off" and "not changed" in out.text


def test_no_terminal_means_no(d):
    def eof(p):
        raise EOFError
    assert C.cmd_set("nfl-weekly-ingest", "live", jobs_dir=d["jobs_dir"], input_fn=eof, out=Say()) == 1
    assert R.read_mode("nfl-weekly-ingest", d["jobs_dir"]) == "off"


def test_yes_flag_skips_the_question_but_still_prints_the_warning(d):
    out = Say()
    assert C.cmd_set("nfl-weekly-ingest", "live", yes=True, jobs_dir=d["jobs_dir"], input_fn=lambda p: pytest.fail("must not ask"), out=out) == 0
    assert "mllab control scheduler pause nfl-weekly-ingest" in out.text


def test_the_pregame_warning_names_both_triggers_and_both_queues(d):
    out = Say()
    C.cmd_set(J.PREGAME, "live", yes=True, jobs_dir=d["jobs_dir"], out=out)
    t = out.text
    assert "mllab control scheduler pause mlb-2026-pregame-schedule --confirm mlb-2026-pregame-schedule" in t and "cron.yaml" in t and "schedule-today" in t
    assert "gcloud tasks queues pause lineup-pregame" in t and "gcloud tasks queues pause sim-blend" in t


def test_setting_live_when_already_live_does_not_ask_again(d):
    C.cmd_set("nfl-weekly-ingest", "live", yes=True, jobs_dir=d["jobs_dir"], out=Say())
    assert C.cmd_set("nfl-weekly-ingest", "live", jobs_dir=d["jobs_dir"], input_fn=lambda p: pytest.fail("must not ask"), out=Say()) == 0


def test_paused_upstream_jobs_cannot_be_turned_on_here(d):
    out = Say()
    assert C.cmd_set("mlb-2026-weekly-train-v10", "live", yes=True, jobs_dir=d["jobs_dir"], out=out) == 2
    assert R.read_mode("mlb-2026-weekly-train-v10", d["jobs_dir"]) == "off" and "paused" in out.text


def test_unknown_job_or_mode_is_refused(d):
    assert C.cmd_set("nope", "live", jobs_dir=d["jobs_dir"], out=Say()) == 2
    assert C.cmd_set("mlb-2026-daily", "maybe", jobs_dir=d["jobs_dir"], out=Say()) == 2


def test_list_shows_every_job_its_mode_its_schedule_and_pregame(d):
    R.write_mode("mlb-2026-daily", "dry", d["jobs_dir"])
    R.record("mlb-2026-daily", {"job": "mlb-2026-daily", "started": "2026-10-02T08:00:00+00:00", "finished": "2026-10-02T08:01:30+00:00", "ok": True, "seconds": 90.0}, d["state_dir"])
    out = Say()
    C.cmd_list(out=out, **d)
    t = out.text
    for name in J.JOBS:
        assert name in t
    assert "pregame" in t and "paused in GCP too" in t
    line = [l for l in out if l.startswith("mlb-2026-daily ")][0]
    assert " dry " in line and "04:00:00" in line and "ok 90.0s" in line
    assert [l for l in out if l.startswith("nfl-weekly-ingest")][0].split()[1] == "off"


def test_run_dry_never_asks_and_forces_dry(d):
    seen = []
    def spy(env, stdin, timeout, cwd=None):
        seen.append(json.loads(stdin))
        return 0, R.RESULT_MARK + json.dumps({"status": 200, "body": {"status": "ok"}}), ""
    assert C.cmd_run("mlb-2026-daily", dry=True, input_fn=lambda p: pytest.fail("must not ask"), out=Say(), executor=spy, resolve=lambda r: "x", **d) == 0
    assert seen == [{"mode": "daily", "dry_run": True}]


def test_run_for_real_asks_first_and_does_not_change_the_mode_file(d):
    seen = []
    def spy(env, stdin, timeout, cwd=None):
        seen.append(json.loads(stdin))
        return 0, R.RESULT_MARK + json.dumps({"status": 200, "body": {"status": "ok"}}), ""
    out = Say()
    assert C.cmd_run("mlb-2026-daily", input_fn=lambda p: "n", out=out, executor=spy, resolve=lambda r: "x", **d) == 1 and not seen
    assert C.cmd_run("mlb-2026-daily", input_fn=lambda p: "y", out=out, executor=spy, resolve=lambda r: "x", push=lambda *a: None, **d) == 0
    assert seen == [{"mode": "daily"}] and R.read_mode("mlb-2026-daily", d["jobs_dir"]) == "off"
