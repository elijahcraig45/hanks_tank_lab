"""Collectors: gather the facts the health rules evaluate. Each source is independent and degrades to `unavailable: <why>`; a missing permission
never breaks the page and never masquerades as health. Queries are tiny and parameterised; table and column names come from fixed maps, never
from input."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Protocol
from zoneinfo import ZoneInfo

from . import health as H
from .auth import is_blocked_account

ET = ZoneInfo("America/New_York")
DATASETS = {"mlb": "mlb_2026_season", "nfl": "nfl_season", "cfb": "cfb_season"}
# model key -> (table, key column, is_shadow). Keys are the site registry keys; derived models (elo, market) have no table.
TABLES = {
    "mlb": {"v10": ("game_predictions", "game_pk", False), "logit3": ("game_predictions_logit3", "game_pk", True),
            "sim_blend": ("game_predictions_sim_blend", "game_pk", True)},
    # Football: the margin ridge is the headline (game_predictions) since 2026 (NFL week 9, CFB week 11); the former headline XGBoost has its own table.
    "nfl": {"ridge": ("game_predictions", "game_id", False), "xgb": ("game_predictions_xgb_shadow", "game_id", True),
            "drive_sim": ("game_predictions_drive_sim", "game_id", True), "fpi": ("fpi_game_predictions", "game_id", True)},
    "cfb": {"ridge": ("game_predictions", "game_id", False), "xgb": ("game_predictions_xgb_shadow", "game_id", True),
            "drive_sim": ("game_predictions_drive_sim", "game_id", True), "fpi": ("fpi_game_predictions", "game_id", True)},
}
SEASON_START = {"mlb": "2026-03-27"}
IN_SEASON_MONTHS = {"mlb": range(3, 12), "nfl": (9, 10, 11, 12, 1, 2), "cfb": (8, 9, 10, 11, 12, 1)}


class Unavailable(Exception):
    """A source could not be read; str(e) is shown on the dashboard."""


class Runner(Protocol):
    def query(self, sql: str, params: dict | None = None) -> list[dict]: ...


class BqCliRunner:
    """Uses the `bq` CLI with the caller's own gcloud identity (works on a laptop; nothing to grant)."""

    def __init__(self, project: str = "hankstank", account: str | None = "you@example.com", runner=subprocess.run, bq: str = "bq"):
        if is_blocked_account(account):
            raise Unavailable("refusing to use the work account")
        self.project, self.account, self.runner, self.bq = project, account, runner, bq

    def query(self, sql: str, params: dict | None = None) -> list[dict]:
        args = [f"--parameter={k}:{t}:{v}" for k, (t, v) in (params or {}).items()]
        env = {**os.environ, **({"CLOUDSDK_CORE_ACCOUNT": self.account} if self.account else {})}
        r = self.runner([self.bq, "--project_id", self.project, "query", "--nouse_legacy_sql", "--quiet", "--format=json", "--max_rows=100000", *args, sql],
                        capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise Unavailable(_short(r.stderr or r.stdout))
        return json.loads(r.stdout or "[]")


class BqClientRunner:
    """Uses the Python client (the lab's service account on the server). Read access is whatever that account was granted."""

    def __init__(self, project: str = "hankstank", client=None):
        self.project, self._client = project, client

    def query(self, sql: str, params: dict | None = None) -> list[dict]:
        try:
            from google.cloud import bigquery
            c = self._client or bigquery.Client(project=self.project)
            cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter(k, t, _cast(t, v)) for k, (t, v) in (params or {}).items()])
            return [dict(r) for r in c.query(sql, job_config=cfg).result(timeout=60)]
        except Exception as e:  # noqa: BLE001
            raise Unavailable(_short(str(e))) from e


def _cast(t: str, v):
    if t == "DATE":
        return date.fromisoformat(v) if isinstance(v, str) else v
    if t == "TIMESTAMP":
        return _ts(v)
    if t == "INT64":
        return int(v)
    return v


def _short(s: str) -> str:
    s = " ".join(str(s).split())
    return s[:200] + ("..." if len(s) > 200 else "")


def _ts(v) -> datetime | None:
    if v in (None, ""):
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromtimestamp(float(v), timezone.utc)  # bq CLI JSON gives epoch seconds for TIMESTAMP
    except ValueError:
        d = datetime.fromisoformat(str(v).replace(" UTC", "+00:00").replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _num(v, cast=float):
    return None if v in (None, "") else cast(v)


# --- per-source collectors ----------------------------------------------------------------------------------

def model_stats_sql(sport: str, key: str) -> str:
    table, keycol, _ = TABLES[sport][key]
    ds = DATASETS[sport]
    if sport == "mlb":  # MLB tables carry game_date, so "today" means games scheduled today
        today = f"COUNTIF(game_date = @today) AS rows_today, COUNT(DISTINCT IF(game_date = @today, {keycol}, NULL)) AS games_today"
        window = "game_date >= @since_date"
    else:
        today = "0 AS rows_today, 0 AS games_today"
        window = "predicted_at >= @since_ts"
    return (f"SELECT {today}, MAX(predicted_at) AS last_at, COUNT(*) AS n_recent, AVG(home_win_probability) AS mean_p, "
            f"STDDEV(home_win_probability) AS std_p, "
            f"COUNTIF(home_win_probability IS NULL OR IS_NAN(home_win_probability) OR home_win_probability < 0 OR home_win_probability > 1) AS bad_rows, "
            f"SAFE_DIVIDE(COUNTIF(GREATEST(home_win_probability, 1 - home_win_probability) >= 0.64), COUNT(*)) AS high_share "
            f"FROM `hankstank.{ds}.{table}` WHERE {window}")


def collect_model(sport: str, key: str, runner: Runner, now: datetime) -> H.ModelFacts:
    table, _, shadow = TABLES[sport][key]
    today = now.astimezone(ET).date()
    params = {"today": ("DATE", today.isoformat()), "since_date": ("DATE", (today - timedelta(days=14)).isoformat()),
              "since_ts": ("TIMESTAMP", (now - timedelta(days=14)).strftime("%Y-%m-%d %H:%M:%S+00"))}
    used = {k: v for k, v in params.items() if f"@{k}" in model_stats_sql(sport, key)}
    rows = runner.query(model_stats_sql(sport, key), used)
    r = rows[0] if rows else {}
    return H.ModelFacts(sport=sport, model=key, is_shadow=shadow, rows_today=int(r.get("rows_today") or 0), games_today=int(r.get("games_today") or 0),
                        last_prediction_at=_ts(r.get("last_at")), n_recent=int(r.get("n_recent") or 0), prob_mean=_num(r.get("mean_p")),
                        prob_std=_num(r.get("std_p")), bad_rows=int(r.get("bad_rows") or 0), high_conf_share=_num(r.get("high_share")))


def collect_schedule(runner: Runner, now: datetime) -> tuple[int, bool]:
    """(games scheduled today, is the season running). No games today AND none nearby means off-season, so no coverage alarms."""
    today = now.astimezone(ET).date()
    p = {"today": ("DATE", today.isoformat()), "lo": ("DATE", (today - timedelta(days=10)).isoformat()), "hi": ("DATE", (today + timedelta(days=1)).isoformat())}
    real = "game_type IN ('R', 'F', 'D', 'L', 'W')"
    r = runner.query(f"SELECT COUNT(DISTINCT IF(game_date = @today AND status NOT IN ('Postponed', 'Cancelled', 'Suspended'), game_pk, NULL)) AS today_games, "
                     f"COUNT(DISTINCT game_pk) AS nearby FROM `hankstank.mlb_2026_season.games` WHERE {real} AND game_date BETWEEN @lo AND @hi", p)[0]
    return int(r["today_games"] or 0), int(r["nearby"] or 0) > 0


def collect_feature_freshness(runner: Runner, now: datetime) -> datetime | None:
    since = (now.astimezone(ET).date() - timedelta(days=7)).isoformat()
    r = runner.query("SELECT MAX(computed_at) AS newest FROM `hankstank.mlb_2026_season.game_v10_features` WHERE game_date >= @since", {"since": ("DATE", since)})
    return _ts(r[0]["newest"]) if r else None


def _logloss(pairs: list[tuple[float, int]]) -> float:
    eps = 1e-15
    return -sum(y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps)) for p, y in pairs) / len(pairs)


def collect_accuracy_mlb(runner: Runner, now: datetime, recent_days: int = 30) -> dict:
    """Pregame-only accuracy: predictions made before first pitch, scored against final games (the same rules the backtests use)."""
    rows = runner.query(
        "WITH pred AS (SELECT game_pk, home_win_probability AS p FROM `hankstank.mlb_2026_season.game_predictions` WHERE predicted_at < game_time_utc "
        "QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY predicted_at DESC) = 1), "
        "final AS (SELECT game_pk, game_date, CAST(home_score > away_score AS INT64) AS y FROM `hankstank.mlb_2026_season.games` "
        "WHERE status IN ('Final', 'Game Over') AND home_score IS NOT NULL AND away_score IS NOT NULL AND home_score != away_score "
        "QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY synced_at DESC NULLS LAST) = 1) "
        "SELECT CAST(final.game_date AS STRING) AS d, pred.p AS p, final.y AS y FROM pred JOIN final USING (game_pk) WHERE final.game_date >= @start",
        {"start": ("DATE", SEASON_START["mlb"])})
    data = [(date.fromisoformat(str(r["d"])[:10]), float(r["p"]), int(r["y"])) for r in rows if r.get("p") is not None]
    if not data:
        return {}
    cutoff = now.astimezone(ET).date() - timedelta(days=recent_days)
    season_rate = sum(y for _, _, y in data) / len(data)  # always-home uses the season's home rate for BOTH windows: a fixed, honest yardstick

    def window(items):
        prior = [(min(max(season_rate, 1e-6), 1 - 1e-6), y) for _, _, y in items]
        return {"n": len(items), "log_loss": _logloss([(p, y) for _, p, y in items]), "home_prior_log_loss": _logloss(prior),
                "accuracy": sum((p > 0.5) == (y == 1) for _, p, y in items) / len(items)}

    out = {"season": window(data)}
    recent = [x for x in data if x[0] >= cutoff]
    if recent:
        out["recent"] = window(recent)
    return out


def collect_jobs(scheduler, sport: str) -> list[dict]:
    return [{"name": j["name"], "state": j["state"], "body_ok": j["body_ok"], "mode": j["mode"], "last_attempt_ok": j["last_attempt_ok"],
             "last_attempt_at": j["last_attempt_at"], "schedule": j["schedule"]} for j in scheduler.list() if j["name"].startswith(f"{sport}-")]


PIN_RE = __import__("re").compile(r"pinned model (gs://\S+) sha256 mismatch")
PIN_MODELS = {"mlb": "v10"}   # only the MLB production key can be pinned today


class GcloudLogReader:
    """Reads recent Cloud Logging text lines through the gcloud CLI with the caller's own identity. Read-only."""

    def __init__(self, project: str = "hankstank", account: str | None = "you@example.com", runner=subprocess.run, gcloud: str = "gcloud"):
        if is_blocked_account(account):
            raise Unavailable("refusing to use the work account")
        self.project, self.account, self.runner, self.gcloud = project, account, runner, gcloud

    def read(self, filter_: str, hours: int = 24, limit: int = 20) -> list[dict]:
        env = {**os.environ, **({"CLOUDSDK_CORE_ACCOUNT": self.account} if self.account else {})}
        r = self.runner([self.gcloud, "logging", "read", filter_, "--project", self.project, "--freshness", f"{hours}h", "--limit", str(limit), "--format=json"],
                        capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise Unavailable(_short(r.stderr or r.stdout))
        return json.loads(r.stdout or "[]")


# --- server-side collectors: REST with the service account's own credentials (the server has no bq or gcloud CLI) -----------

class ApiSession:
    """Read-only REST calls with application-default credentials. `session` is injectable for tests."""

    def __init__(self, session=None):
        self._session = session

    def _s(self):
        if self._session is None:
            try:
                import google.auth
                from google.auth.transport.requests import AuthorizedSession
                creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
                self._session = AuthorizedSession(creds)
            except Exception as e:  # noqa: BLE001
                raise Unavailable(_short(f"no credentials: {e}")) from e
        return self._session

    def call(self, method: str, url: str, **kw) -> dict:
        try:
            r = self._s().request(method, url, timeout=30, **kw)
        except Unavailable:
            raise
        except Exception as e:  # noqa: BLE001
            raise Unavailable(_short(str(e))) from e
        if r.status_code >= 400:
            raise Unavailable(_short(f"HTTP {r.status_code}: {r.text}"))
        return r.json()


class ApiScheduler:
    """Lists Cloud Scheduler jobs (the same decoded shape as mllab.scheduler.Scheduler.list). Listing only: changes stay with the CLI."""

    def __init__(self, project: str = "hankstank", location: str = "us-central1", api: ApiSession | None = None):
        self.project, self.location, self.api = project, location, api or ApiSession()

    def list(self) -> list[dict]:
        from .scheduler import parse_jobs
        d = self.api.call("GET", f"https://cloudscheduler.googleapis.com/v1/projects/{self.project}/locations/{self.location}/jobs", params={"pageSize": 100})
        return parse_jobs(d.get("jobs", []))


class ApiLogReader:
    """Reads recent Cloud Logging entries over REST. Same interface as GcloudLogReader."""

    def __init__(self, project: str = "hankstank", api: ApiSession | None = None):
        self.project, self.api = project, api or ApiSession()

    def read(self, filter_: str, hours: int = 24, limit: int = 20) -> list[dict]:
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        body = {"resourceNames": [f"projects/{self.project}"], "filter": f'{filter_} AND timestamp>="{since}"', "orderBy": "timestamp desc", "pageSize": limit}
        return self.api.call("POST", "https://logging.googleapis.com/v2/entries:list", json=body).get("entries", [])


def control_from_view(runner: Runner) -> dict:
    """The control state as {(sport, target): {field: value}} read from the current-state view with the runner's credentials."""
    rows = runner.query("SELECT * FROM `hankstank.control.model_control_current`")
    return {(r["sport"], r["target"]): {k: v for k, v in r.items() if k not in ("sport", "target") and v is not None} for r in rows}


def collect_pin_mismatches(reader, sport: str) -> list[dict]:
    """Pin mismatch lines from the pipeline's logs, newest first: [{model, uri, at}]. Raises Unavailable if the logs cannot be read."""
    model = PIN_MODELS.get(sport)
    if model is None:
        return []
    flt = 'resource.type=("cloud_run_revision" OR "cloud_function") AND severity>=ERROR AND "sha256 mismatch"'
    out = []
    for e in reader.read(flt):
        text = e.get("textPayload") or (e.get("jsonPayload") or {}).get("message") or ""
        m = PIN_RE.search(text)
        if m:
            out.append({"model": model, "uri": m.group(1), "at": _ts(e.get("timestamp"))})
    return out


def collect_lab(path: str = "/") -> dict:
    lab: dict = {}
    try:
        u = shutil.disk_usage(path)
        lab["disk_used_pct"] = round(100 * u.used / u.total, 1)
    except OSError:
        pass
    if shutil.which("systemctl") and os.path.isdir("/run/systemd/system"):
        units = ["mllab-api", "mllab-jupyter", "mllab-caddy", "mllab-spool.path"]
        lab["services_ok"] = all(subprocess.run(["systemctl", "is-active", "--quiet", u]).returncode == 0 for u in units)
    return lab


# --- assembling the snapshot ------------------------------------------------------------------------------------

def in_season_by_month(sport: str, now: datetime) -> bool:
    return now.astimezone(ET).month in IN_SEASON_MONTHS[sport]


def build_snapshot(now: datetime, runner: Runner, *, sports: tuple[str, ...] = ("mlb",), registry_fn: Callable[[str], list[dict]] | None = None,
                   control_fn: Callable[[], dict] | None = None, scheduler=None, log_reader=None, lab_fn: Callable[[], dict] = collect_lab) -> H.Snapshot:
    snap = H.Snapshot(taken_at=now)
    control: dict = {}
    if control_fn is not None:
        try:
            control = control_fn()
            snap.sources["control"] = "ok"
        except Exception as e:  # noqa: BLE001
            snap.sources["control"] = f"unavailable: {_short(e)}"
    for sport in sports:
        sf = H.SportFacts(sport=sport, in_season=in_season_by_month(sport, now))
        roles: dict = {}
        if registry_fn is not None:
            try:
                roles = {m["key"]: m for m in registry_fn(sport)}
                snap.sources[f"registry:{sport}"] = "ok"
            except Exception as e:  # noqa: BLE001
                snap.sources[f"registry:{sport}"] = f"unavailable: {_short(e)}"
        for key in TABLES[sport]:
            if roles and key not in roles:
                continue  # not on the site (planned, or removed): nothing to monitor
            try:
                m = collect_model(sport, key, runner, now)
                if roles.get(key, {}).get("role"):
                    m.is_shadow = roles[key]["role"] != "production"
                m.control = dict(control.get((sport, key), {}))
                sf.models.append(m)
                snap.sources.setdefault(f"bigquery:{sport}", "ok")
            except Exception as e:  # noqa: BLE001
                snap.sources[f"bigquery:{sport}"] = f"unavailable: {_short(e)}"
                break  # one failing table usually means the whole dataset: do not spam a line per model
        sf.banner = (control.get((sport, "*"), {}) or {}).get("banner")
        if sport == "mlb" and snap.sources.get("bigquery:mlb") == "ok":
            try:
                sf.scheduled_today, season_running = collect_schedule(runner, now)
                sf.in_season = sf.in_season and season_running
                sf.features_computed_at = collect_feature_freshness(runner, now)
                sf.accuracy = collect_accuracy_mlb(runner, now)
            except Exception as e:  # noqa: BLE001
                snap.sources["bigquery:mlb:schedule"] = f"unavailable: {_short(e)}"
        if log_reader is not None:
            try:
                sf.pin_mismatches = collect_pin_mismatches(log_reader, sport)
                snap.sources.setdefault("logs", "ok")
            except Exception as e:  # noqa: BLE001
                snap.sources["logs"] = f"unavailable: {_short(e)}"
        if scheduler is not None:
            try:
                sf.jobs = collect_jobs(scheduler, sport)
                snap.sources.setdefault("scheduler", "ok")
            except Exception as e:  # noqa: BLE001
                snap.sources["scheduler"] = f"unavailable: {_short(e)}"
        snap.sports[sport] = sf
    try:
        snap.lab = lab_fn()
    except Exception as e:  # noqa: BLE001
        snap.sources["lab"] = f"unavailable: {_short(e)}"
    return snap
