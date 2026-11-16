"""Model health: a snapshot of facts about what the pipeline and the site are doing, and pure rules that turn facts into findings.

Facts come from collectors (BigQuery, the scheduler, logs). A collector that cannot run (missing permission, table absent) reports
`unavailable: <why>` and the rest of the dashboard still renders, so a gap in access never looks like a healthy system and never breaks the
page. The rules here are pure functions of a Snapshot: no network, no clock except the `now` they are given, easy to test exhaustively.

Severity: info (worth knowing), warn (look at it today), critical (the site is wrong or empty now).
A rule may attach a Remedy: a bounded, reversible action that mllab.remedy may take (in dry-run until you switch it on).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
SEVERITIES = ("info", "warn", "critical")


@dataclass
class Remedy:
    action: str            # trigger_function | control_hide | prune_envs | alert
    params: dict = field(default_factory=dict)
    reversible: bool = True


@dataclass
class Finding:
    code: str
    severity: str
    message: str
    sport: str = "*"
    model: str = "*"
    remedy: Remedy | None = None

    def key(self) -> str:
        return f"{self.code}:{self.sport}:{self.model}"


@dataclass
class ModelFacts:
    sport: str
    model: str
    rows_today: int = 0
    games_today: int = 0
    last_prediction_at: datetime | None = None
    n_recent: int = 0                      # rows in the recent window used for distribution checks
    prob_mean: float | None = None
    prob_std: float | None = None
    bad_rows: int = 0                      # NaN, null or outside [0, 1]
    high_conf_share: float | None = None
    is_shadow: bool = False                # writes to its own table, never the site
    control: dict = field(default_factory=dict)   # the model's row from model_control_current


@dataclass
class SportFacts:
    sport: str
    in_season: bool = True
    scheduled_today: int = 0
    features_computed_at: datetime | None = None   # newest feature row
    models: list[ModelFacts] = field(default_factory=list)
    jobs: list[dict] = field(default_factory=list)   # {name, state, body_ok, mode, last_attempt_ok, last_attempt_at}
    banner: str | None = None
    accuracy: dict = field(default_factory=dict)     # {window: {n, log_loss, home_prior_log_loss}}
    pin_mismatches: list[dict] = field(default_factory=list)   # {model, uri, at}: recent "sha256 mismatch" errors from the pipeline logs


@dataclass
class Snapshot:
    taken_at: datetime
    sports: dict[str, SportFacts] = field(default_factory=dict)
    lab: dict = field(default_factory=dict)          # {disk_used_pct, services_ok, ...}
    sources: dict = field(default_factory=dict)      # collector name -> "ok" | "unavailable: why"


@dataclass
class Thresholds:
    coverage_warn_after_et: int = 11       # hour of day (ET) after which missing predictions matter
    coverage_critical_after_et: int = 15
    coverage_warn: float = 0.5
    coverage_critical: float = 0.9
    features_stale_hours: float = 6.0
    collapsed_std: float = 0.005           # a model whose probabilities barely vary is broken
    min_rows_for_distribution: int = 12
    mean_shift: float = 0.12               # |mean probability - 0.5| this large is suspicious
    shadow_stale_hours: float = 72.0
    shadow_stale_hours_weekly: float = 200.0   # football predicts once a week (about 8 days between writes is normal)
    weekly_sports: tuple = ("nfl", "cfb")
    accuracy_worse_by: float = 0.02        # rolling log loss worse than the season by this much
    disk_warn_pct: float = 85.0
    pin_mismatch_hours: float = 24.0       # how long a logged pin mismatch stays a finding while the pin is unchanged
    known_job_modes: tuple = ("daily", "schedule_pregame_tasks", "rosters", "power_rankings", "ingest", "score", "predict_week", "predict_next",
                              "cfbd", "drives", "season_sim", "train_weekly", "predict", "predict_today", "pregame_v10", "pa_sim", "logit3",
                              "sim_blend", "validate", "stats", "fpi_snapshot", "rankings", "backfill", "features", "v10_features", "v8_features")


# --- rules -----------------------------------------------------------------------------------------------

def _hours(a: datetime, b: datetime) -> float:
    return (a - b).total_seconds() / 3600


def _visible(m: ModelFacts) -> bool:
    return m.control.get("site_visible", "true") == "true"


def _paused(m: ModelFacts) -> bool:
    return m.control.get("run_state", "active") == "paused"


def rule_coverage(s: SportFacts, now: datetime, t: Thresholds) -> list[Finding]:
    """Games are scheduled today but the live model has not produced predictions for them."""
    if not s.in_season or s.scheduled_today <= 0:
        return []
    live = [m for m in s.models if not m.is_shadow and _visible(m) and not _paused(m)]
    if not live:
        return []
    covered = max((m.games_today for m in live), default=0)
    ratio = covered / s.scheduled_today
    hour = now.astimezone(ET).hour
    if hour >= t.coverage_critical_after_et and ratio < t.coverage_critical:
        sev = "critical"
    elif hour >= t.coverage_warn_after_et and ratio < t.coverage_warn:
        sev = "warn"
    else:
        return []
    return [Finding("coverage_gap", sev, f"{covered} of {s.scheduled_today} {s.sport.upper()} games today have a prediction at {now.astimezone(ET):%H:%M} ET",
                    s.sport, remedy=Remedy("trigger_function", {"sport": s.sport, "mode": "predict_today"}))]


def rule_stale_features(s: SportFacts, now: datetime, t: Thresholds) -> list[Finding]:
    if not s.in_season or s.scheduled_today <= 0 or s.features_computed_at is None:
        return []
    age = _hours(now, s.features_computed_at)
    if age <= t.features_stale_hours:
        return []
    return [Finding("stale_features", "warn", f"newest {s.sport.upper()} feature row is {age:.1f} h old on a game day", s.sport,
                    remedy=Remedy("trigger_function", {"sport": s.sport, "mode": "v10_features"}))]


def rule_bad_output(s: SportFacts, m: ModelFacts, t: Thresholds) -> list[Finding]:
    if m.is_shadow or not _visible(m):
        return []
    problems = []
    if m.bad_rows > 0:
        problems.append(f"{m.bad_rows} predictions are null, NaN or outside 0-1")
    if m.n_recent >= t.min_rows_for_distribution and m.prob_std is not None and m.prob_std < t.collapsed_std:
        problems.append(f"probabilities have collapsed (std {m.prob_std:.4f})")
    if not problems:
        return []
    fallback = [o for o in s.models if o is not m and not o.is_shadow and _visible(o) and o.rows_today > 0 and o.bad_rows == 0]
    remedy = Remedy("control_hide", {"sport": s.sport, "model": m.model}) if fallback else Remedy("alert")
    tail = " (another visible model can take over)" if fallback else " (no healthy fallback, so it stays visible: alerting instead)"
    return [Finding("bad_output", "critical", f"{m.model}: " + "; ".join(problems) + tail, s.sport, m.model, remedy)]


def rule_no_output(s: SportFacts, m: ModelFacts, now: datetime) -> list[Finding]:
    if not s.in_season or s.scheduled_today <= 0 or m.is_shadow or _paused(m) or not _visible(m):
        return []
    if m.rows_today == 0 and now.astimezone(ET).hour >= 12:
        return [Finding("no_output_today", "critical", f"{m.model} has written no predictions today ({s.scheduled_today} games scheduled)", s.sport, m.model,
                        Remedy("trigger_function", {"sport": s.sport, "mode": "predict_today"}))]
    return []


def rule_shadow_stale(s: SportFacts, m: ModelFacts, now: datetime, t: Thresholds) -> list[Finding]:
    if not (s.in_season and m.is_shadow) or _paused(m):
        return []
    if m.last_prediction_at is None:
        return [Finding("shadow_never_ran", "info", f"shadow {m.model} has no rows yet", s.sport, m.model)]
    age = _hours(now, m.last_prediction_at)
    limit = t.shadow_stale_hours_weekly if s.sport in t.weekly_sports else t.shadow_stale_hours
    if age > limit:
        return [Finding("shadow_stale", "warn", f"shadow {m.model} last wrote {age / 24:.1f} days ago", s.sport, m.model, Remedy("alert"))]
    return []


def rule_calibration_shift(s: SportFacts, m: ModelFacts, t: Thresholds) -> list[Finding]:
    if m.is_shadow or m.n_recent < 30 or m.prob_mean is None:
        return []
    if abs(m.prob_mean - 0.5) > t.mean_shift:
        return [Finding("probability_shift", "warn", f"{m.model}: mean predicted home-win probability is {m.prob_mean:.2f} over {m.n_recent} recent rows", s.sport,
                        m.model, Remedy("alert"))]
    return []


def rule_accuracy_drift(s: SportFacts, t: Thresholds) -> list[Finding]:
    a = s.accuracy
    if not a or "season" not in a or "recent" not in a:
        return []
    season, recent = a["season"], a["recent"]
    if recent.get("n", 0) < 100 or season.get("n", 0) < 300:
        return []  # too few games for the difference to mean anything
    if recent["log_loss"] - season["log_loss"] > t.accuracy_worse_by and recent["log_loss"] > recent["home_prior_log_loss"]:
        return [Finding("accuracy_drift", "warn", f"recent log loss {recent['log_loss']:.4f} is worse than the season's {season['log_loss']:.4f} and worse than "
                        f"always-home ({recent['home_prior_log_loss']:.4f}) over {recent['n']} games", s.sport, remedy=Remedy("alert"))]
    return []


def rule_scheduler(s: SportFacts, t: Thresholds) -> list[Finding]:
    out = []
    for j in s.jobs:
        if j.get("body_ok") is False:
            out.append(Finding("scheduler_bad_body", "critical", f"job {j['name']} has a body that is not valid JSON: it silently runs the default 'daily' mode instead",
                               s.sport, remedy=Remedy("alert")))
        elif j.get("mode") and j["mode"] not in t.known_job_modes:
            out.append(Finding("scheduler_unknown_mode", "warn", f"job {j['name']} runs unknown mode {j['mode']!r}", s.sport, remedy=Remedy("alert")))
        if j.get("state") == "ENABLED" and j.get("last_attempt_ok") is False:
            out.append(Finding("scheduler_job_failing", "warn", f"job {j['name']} failed its last run", s.sport, remedy=Remedy("alert")))
    return out


def rule_control_consistency(s: SportFacts) -> list[Finding]:
    out = []
    real = [m for m in s.models if not m.is_shadow]
    if real and not any(_visible(m) for m in real) and not s.banner:
        out.append(Finding("site_empty", "critical", f"every {s.sport.upper()} model is hidden and there is no banner: the site shows nothing and does not say why", s.sport))
    for m in s.models:
        if m.control.get("lifecycle") == "live" and not m.control.get("artifact_sha256") and m.control.get("artifact_uri"):
            out.append(Finding("pin_without_sha", "warn", f"{m.model} is pinned to an artifact without its sha256", s.sport, m.model))
    return out


def rule_pin_mismatch(s: SportFacts, now: datetime, t: Thresholds) -> list[Finding]:
    """The pipeline refused a pinned artifact whose sha256 did not match. It stops the run and writes nothing, so this is critical.
    Only while the pin still names the artifact in the log line: correcting or removing the pin clears the finding. No remedy, ever."""
    by_model = {m.model: m for m in s.models}
    out, seen = [], set()
    for x in s.pin_mismatches:
        m = by_model.get(x.get("model"))
        if m is None or m.control.get("artifact_uri") != x.get("uri") or (m.model, x["uri"]) in seen:
            continue
        at = x.get("at")
        if isinstance(at, datetime) and _hours(now, at) > t.pin_mismatch_hours:
            continue
        seen.add((m.model, x["uri"]))
        out.append(Finding("pin_mismatch", "critical", f"{m.model}: pinned artifact {x['uri']} failed its sha256 check; the pipeline refused to run it and wrote nothing "
                           "(re-pin with the right sha256, or unpin)", s.sport, m.model))
    return out


def rule_lab(lab: dict, t: Thresholds) -> list[Finding]:
    out = []
    if lab.get("disk_used_pct") is not None and lab["disk_used_pct"] > t.disk_warn_pct:
        out.append(Finding("lab_disk", "warn", f"lab disk is {lab['disk_used_pct']:.0f}% full", remedy=Remedy("prune_envs", {"older_than_days": 14})))
    if lab.get("services_ok") is False:
        out.append(Finding("lab_services", "critical", "a lab service is down"))
    return out


def evaluate(snap: Snapshot, t: Thresholds | None = None) -> list[Finding]:
    t = t or Thresholds()
    out: list[Finding] = []
    for name, why in snap.sources.items():
        if why != "ok":
            out.append(Finding("source_unavailable", "info", f"{name}: {why}"))
    for s in snap.sports.values():
        out += rule_coverage(s, snap.taken_at, t) + rule_stale_features(s, snap.taken_at, t) + rule_accuracy_drift(s, t)
        out += rule_scheduler(s, t) + rule_control_consistency(s) + rule_pin_mismatch(s, snap.taken_at, t)
        for m in s.models:
            out += rule_bad_output(s, m, t) + rule_no_output(s, m, snap.taken_at) + rule_shadow_stale(s, m, snap.taken_at, t) + rule_calibration_shift(s, m, t)
    out += rule_lab(snap.lab, t)
    rank = {s: i for i, s in enumerate(SEVERITIES)}
    return sorted(out, key=lambda f: (-rank[f.severity], f.sport, f.model, f.code))


def model_status(s: SportFacts, m: ModelFacts, findings: list[Finding]) -> str:
    if _paused(m):
        return "paused"
    if not _visible(m):
        return "hidden"
    mine = [f for f in findings if f.sport == s.sport and f.model in (m.model, "*") and f.code != "source_unavailable"]
    if any(f.severity == "critical" for f in mine):
        return "down"
    if any(f.severity == "warn" for f in mine):
        return "warn"
    return "ok"


def to_json(snap: Snapshot, findings: list[Finding]) -> dict:
    def conv(o):
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(type(o))
    import json
    d = {"taken_at": snap.taken_at, "sources": snap.sources, "lab": snap.lab,
         "sports": {k: {**{f: v for f, v in asdict(s).items() if f != "models"},
                        "models": [{**asdict(m), "status": model_status(s, m, findings)} for m in s.models]} for k, s in snap.sports.items()},
         "findings": [{**asdict(f), "remedy": asdict(f.remedy) if f.remedy else None} for f in findings]}
    return json.loads(json.dumps(d, default=conv))
