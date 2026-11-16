"""pipeline-ctl: see and change what the homelab pipeline does.

    pipeline-ctl list                    every job: schedule, mode, last run
    pipeline-ctl status                  the same plus per-game scheduling
    pipeline-ctl set <job> off|dry|live  change a job's mode (takes effect at its next run)
    pipeline-ctl run <job> [--dry]       run one job now, in the foreground

`set ... live` is the cutover. It prints what must already be switched off in GCP and asks before it writes anything: two writers on the
same tables is worse than a missed run. `--yes` skips the question for scripted use; nothing else does. Modes live in /etc/pipeline/jobs.d
(group-writable for the `pipeline` group, so no sudo is needed), and `off` is what every job starts as."""
from __future__ import annotations

import json
import sys

from . import pregame, runner
from .jobs import JOBS, LOCATION, MODES, PREGAME, PROJECT, cron_to_oncalendar

# What must be off in GCP before a job goes live here. The Scheduler job name is the job's own name.
PREGAME_WARNING = """  Per-game scheduling is created in TWO places today, and both must be switched off first:
    1. mllab control scheduler pause mlb-2026-pregame-schedule --confirm mlb-2026-pregame-schedule      (audited; run it from your laptop)
    2. the App Engine cron entry /api/lineup/schedule-today (hanks_tank_backend/cron.yaml): remove it and `gcloud app deploy cron.yaml`
  and anything already queued must be drained, or it will run on top of this:
    3. gcloud tasks queues pause lineup-pregame --location {loc} --project {proj}
       gcloud tasks queues pause sim-blend       --location {loc} --project {proj}"""


def warning(name: str) -> str:
    if name == PREGAME:
        return PREGAME_WARNING.format(loc=LOCATION, proj=PROJECT)
    return (f"  The Cloud Scheduler job of the same name must be paused first (audited; run it from your laptop):\n"
            f"    mllab control scheduler pause {name} --confirm {name}")


def confirm(prompt: str, input_fn=input) -> bool:
    try:
        return input_fn(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def cmd_set(name: str, mode: str, *, yes: bool = False, jobs_dir=None, input_fn=input, out=print) -> int:
    if name != PREGAME and name not in JOBS:
        out(f"unknown job {name!r}; see `pipeline-ctl list`")
        return 2
    if mode not in MODES:
        out(f"mode must be one of: {', '.join(MODES)}")
        return 2
    if name != PREGAME and JOBS[name].paused and mode != "off":
        out(f"{name} is paused in Cloud Scheduler on purpose (see docs/HOMELAB_PIPELINE.md) and is not installed here.")
        return 2
    before = runner.read_mode(name, jobs_dir)
    if mode == "live" and before != "live":
        out(f"Switching {name} to LIVE: it will write to production BigQuery from this machine.")
        out(warning(name))
        if not yes and not confirm("Has that been done, and do you want to go live?", input_fn):
            out("not changed")
            return 1
    runner.write_mode(name, mode, jobs_dir)
    out(f"{name}: {before} -> {mode}")
    return 0


def _last(name: str, runs: dict) -> str:
    r = runs.get(name)
    if not r:
        return "never"
    if r.get("skipped"):
        return f"{r['finished'][5:16]} skipped ({r['skipped']})"
    return f"{r['finished'][5:16]} {'ok' if r['ok'] else 'FAILED'} {r.get('seconds', '')}s{'  ' + str(r['error'])[:50] if r.get('error') else ''}"


def cmd_list(*, jobs_dir=None, state_dir=None, out=print) -> int:
    runs = runner.last_runs(state_dir)
    out(f"{'job':27s} {'mode':5s} {'schedule (America/New_York)':50s} last run (UTC)")
    for j in JOBS.values():
        mode = "paused" if j.paused else runner.read_mode(j.name, jobs_dir)
        out(f"{j.name:27s} {mode:5s} {cron_to_oncalendar(j.cron).rsplit(' ', 1)[0]:50s} {_last(j.name, runs) if not j.paused else 'paused in GCP too'}")
    out(f"{PREGAME:27s} {runner.read_mode(PREGAME, jobs_dir):5s} {'plan every 30 min, tick every minute (months 3-11)':50s} {_last('pregame-ticker', runs)}")
    return 0


def cmd_status(*, jobs_dir=None, state_dir=None, out=print) -> int:
    cmd_list(jobs_dir=jobs_dir, state_dir=state_dir, out=out)
    out("")
    pregame.cmd_status(state_dir=state_dir, jobs_dir=jobs_dir)
    return 0


def cmd_run(name: str, *, dry: bool = False, yes: bool = False, jobs_dir=None, state_dir=None, input_fn=input, out=print, **kw) -> int:
    """Run now, whatever the mode file says. --dry is always allowed; a real run needs the same confirmation as going live."""
    if name not in JOBS:
        out(f"unknown job {name!r}")
        return 2
    if not dry:
        out(f"Running {name} FOR REAL once, now (the mode file is not changed).")
        out(warning(name))
        if not yes and not confirm("Proceed?", input_fn):
            out("not run")
            return 1
    return runner.run_job(name, jobs_dir=jobs_dir, state_dir=state_dir, mode="dry" if dry else "live", **kw)


def main(argv: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    yes = "--yes" in a
    a = [x for x in a if x != "--yes"]
    cmd = a[0] if a else "list"
    if cmd == "list":
        return cmd_list()
    if cmd == "status":
        return cmd_status()
    if cmd == "set" and len(a) == 3:
        return cmd_set(a[1], a[2], yes=yes)
    if cmd == "run" and len(a) >= 2:
        return cmd_run(a[1], dry="--dry" in a, yes=yes)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
