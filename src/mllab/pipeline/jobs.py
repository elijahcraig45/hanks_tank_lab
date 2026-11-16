"""The job table: what runs, in which staged bundle, with which request body, on which schedule.

One source of truth for the runner, the systemd unit generator and the docs. Every body and schedule is copied from what Cloud Scheduler
sends today (checked against `gcloud scheduler jobs list` on 2026-10-02), and job names are Cloud Scheduler's, so a cutover is one job
for one job."""
from __future__ import annotations

from dataclasses import dataclass, field

TZ = "America/New_York"
PROJECT = "hankstank"
LOCATION = "us-central1"

# numpy/BLAS defaults to one thread per core. On the homelab that made one CPU-bound test take 138 s instead of 7 s (13 min of CPU time).
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


@dataclass(frozen=True)
class Bundle:
    """One deployed function's source tree and the configuration it carries in production."""
    name: str
    entry: str  # module:function, resolved inside the staged bundle
    env: dict = field(default_factory=dict)
    secrets: dict = field(default_factory=dict)  # ENV_NAME -> sm://projects/P/secrets/S/versions/V, resolved when a run starts


_FOOTBALL_SHADOWS = {"FPI_SNAPSHOT": "1"}
BUNDLES: dict[str, Bundle] = {
    # mlb-2026-daily-pipeline and mlb-2026-sim-blend are the same source and entrypoint, deployed twice with different sizes.
    "mlb": Bundle("mlb", "cloud_function_main:daily_pipeline", {"GCP_PROJECT": PROJECT}),
    "nfl": Bundle("nfl", "main:nfl_pipeline", {
        "GCP_PROJECT": PROJECT, "NFL_DATASET": "nfl_season", "NFL_HIST_DATASET": "nfl_historical",
        "NFL_DRIVE_SIM_SHADOW": "1", "NFL_RIDGE_SHADOW": "1", "NFL_XGB_REG_SHADOW": "1", **_FOOTBALL_SHADOWS}),
    "cfb": Bundle("cfb", "main:cfb_pipeline", {
        "GCP_PROJECT": PROJECT, "CFB_DATASET": "cfb_season", "CFB_HIST_DATASET": "cfb_historical",
        "CFB_DRIVE_SIM_SHADOW": "1", "CFB_RIDGE_SHADOW": "1", **_FOOTBALL_SHADOWS},
        {"CFBD_API_KEY": f"sm://projects/{PROJECT}/secrets/cfbd-api-key/versions/latest"}),
}


@dataclass(frozen=True)
class Job:
    name: str  # the Cloud Scheduler job it replaces
    bundle: str
    body: dict
    cron: str | None  # 5-field cron in TZ, exactly as Cloud Scheduler has it
    # False where the entrypoint refuses dry_run (NFL and CFB ingest/score/cfbd return 400): "dry" then means a preflight, not a run.
    dry_ok: bool = True
    timeout_s: int = 3600  # the function allowed 540 s; the homelab has no such limit, but nothing should run for hours
    paused: bool = False  # paused in Cloud Scheduler today: known, documented, never installed


def _j(name, bundle, mode, cron, dry_ok=True, paused=False, **extra):
    return Job(name, bundle, {"mode": mode, **extra}, cron, dry_ok, paused=paused)


JOBS: dict[str, Job] = {j.name: j for j in [
    # MLB. `pregame-schedule` is not here: per-game scheduling is pregame.py.
    _j("mlb-2026-daily", "mlb", "daily", "0 4 * 3-11 *"),
    _j("mlb-2026-roster-refresh", "mlb", "rosters", "0 3 * 3-11 1"),
    _j("mlb-2026-power-rankings", "mlb", "power_rankings", "30 6 * 3-11 1"),
    _j("mlb-2026-weekly-train-v10", "mlb", "train_weekly", "0 2 * 3-11 0", paused=True, model_version="v10"),
    _j("mlb-2026-weekly-predict", "mlb", "predict", "0 5 * 3-11 5", paused=True),
    # NFL
    _j("nfl-weekly-ingest", "nfl", "ingest", "0 6 * 9-12,1,2 2", dry_ok=False),
    _j("nfl-weekly-predict", "nfl", "predict_week", "0 6 * 9-12,1,2 3"),
    _j("nfl-weekly-score", "nfl", "score", "0 7 * 9-12,1,2 2", dry_ok=False),
    _j("nfl-weekly-season-sim", "nfl", "season_sim", "30 7 * 9-12,1 2"),
    # CFB
    _j("cfb-weekly-ingest", "cfb", "ingest", "0 6 * 8-12,1 0", dry_ok=False),
    _j("cfb-weekly-cfbd", "cfb", "cfbd", "0 7 * 8-12,1 0", dry_ok=False),
    _j("cfb-weekly-drives", "cfb", "drives", "30 7 * 8-12,1 0"),
    _j("cfb-weekly-season-sim", "cfb", "season_sim", "0 8 * 8-12,1 0"),
    _j("cfb-weekly-predict", "cfb", "predict_next", "0 6 * 8-12,1 2"),
]}

# Per-game scheduling (pregame.py) is a pseudo-job with a mode file of its own.
PREGAME = "pregame"
MODES = ("off", "dry", "live")


def installed_jobs() -> list[Job]:
    return [j for j in JOBS.values() if not j.paused]


# ------------------------------------------------------------------------------------------------------------------ cron -> systemd
_DOW = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


def _numbers(spec: str, lo: int, hi: int) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        if part == "*":
            out.update(range(lo, hi + 1))
        elif "-" in part:
            a, b = (int(x) for x in part.split("-"))
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    bad = [n for n in out if not lo <= n <= hi]
    if bad:
        raise ValueError(f"{spec!r}: {bad} outside {lo}..{hi}")
    return sorted(out)


def _runs(nums: list[int]) -> list[tuple[int, int]]:
    runs: list[list[int]] = []
    for n in nums:
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return [(a, b) for a, b in runs]


def cron_to_oncalendar(cron: str, tz: str = TZ) -> str:
    """'30 6 * 3-11 1' -> 'Mon *-03..11-* 06:30:00 America/New_York'. Handles what the job table uses and refuses the rest, because a
    schedule that is silently wrong would not show until the 6 AM it should have run."""
    f = cron.split()
    if len(f) != 5:
        raise ValueError(f"expected 5 cron fields: {cron!r}")
    minute, hour, dom, month, dow = f
    if dom != "*":
        raise ValueError("day-of-month is not supported")
    if not (minute.isdigit() and hour.isdigit()):
        raise ValueError("minute and hour must each be a single number")
    months = _numbers(month, 1, 12)
    month_part = "*" if months == list(range(1, 13)) else ",".join(f"{a:02d}" if a == b else f"{a:02d}..{b:02d}" for a, b in _runs(months))
    dows = _numbers(dow, 0, 6)
    dow_part = "" if dows == list(range(7)) else ",".join(_DOW[d] for d in dows) + " "
    return f"{dow_part}*-{month_part}-* {int(hour):02d}:{int(minute):02d}:00 {tz}"
