"""Model control plane: change what Hank's Tank shows and runs WITHOUT editing code or redeploying.

Every change is an event appended to `control.model_control_events` (BigQuery). The site and the pipeline read the current state from the
`model_control_current` view and FAIL OPEN: if the control data is missing or unreadable they behave exactly as they did before this existed.
Nothing is ever updated or deleted, so every change is auditable and reversible (`revert` appends the previous value), and a future-dated
event is a scheduled change.

Safety rails built in: every value is validated and rendered as inert text (no markup), changes that affect the live site need a typed
confirmation, the last visible model of a sport cannot be hidden by accident, a pin requires the artifact's sha256, and writes are
parameterised SQL (never string-built).
"""
from __future__ import annotations

import json
import re
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol

from . import audit
from .auth import is_blocked_account

SPORTS = ("mlb", "nfl", "cfb")
UNSET = "__unset__"
TARGET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
BANNER_LEVELS = ("info", "warn", "error")
MODEL_FIELDS = ("site_visible", "run_state", "lifecycle", "artifact_uri", "artifact_sha256", "display_label", "public_note", "sort_order",
                "tier_high", "tier_medium")
SPORT_FIELDS = ("banner", "banner_level", "rankings_show", "rankings_media", "rankings_primary")  # only valid on target '*'
ALL_FIELDS = MODEL_FIELDS + SPORT_FIELDS
DEFAULTS = {"site_visible": "true", "run_state": "active"}  # what "no row" means: exactly today's behaviour

# Which ranking columns the site's power-rankings table may show. The ML job writes one `<key>_rank` column per custom ranking; the site draws
# the ones chosen here, in this order, next to the media polls it is allowed to show. Unset means today's behaviour (every column that has data).
# The keys are a contract with hanks_tank_ml (columns), hanks_tank_backend (rankings.config) and hanks_tank (RankingsBoard): change all four.
MAX_CUSTOM_RANKINGS = 3
RANKINGS: dict[str, dict[str, list[dict]]] = {
    "cfb": {
        "custom": [
            {"key": "results", "label": "Results", "about": "Who has beaten whom this season and nothing else: the order with the fewest contradicted games. A record, not a forecast."},
            {"key": "season", "label": "This season", "about": "The rating fitted on this season's games only, schedule-adjusted, with no carry-over from last season."},
            {"key": "forecast", "label": "Forecast", "about": "The earlier headline weighting, where last season counts for more. Best at predicting the next games."},
            {"key": "resume", "label": "Resume", "about": "The season resume: results against schedule strength, graded against a typical top-25 team, with head-to-head deciding teams whose records cannot be told apart. (When the table is already ordered by the resume this column is dropped.)"},
        ],
        "media": [
            {"key": "ap", "label": "AP poll", "about": "The Associated Press poll."},
            {"key": "coaches", "label": "Coaches poll", "about": "The AFCA Coaches poll (the FCS coaches poll on the FCS board)."},
        ],
        # Which order the table is numbered and sorted by (`rankings_primary`). The other order stays visible as a column.
        "primary": [
            {"key": "resume", "label": "Season resume", "about": "Who has earned it: results against schedule strength, with the next-week rating beside it. The default."},
            {"key": "rating", "label": "Next-week rating", "about": "Who would win next week: the forecast rating, as the site showed before the resume board."},
        ],
    },
    "nfl": {"custom": [], "media": [], "primary": []},
    "mlb": {"custom": [], "media": [], "primary": []},
}
PRIMARY_DEFAULT = {"cfb": "resume"}  # what the site draws when `rankings_primary` is unset (the backend's rankings.config primaryDefault)


class ControlError(Exception):
    pass


class ConfirmationRequired(ControlError):
    """The change affects the live site or production and needs the typed confirmation phrase."""


@dataclass
class Event:
    event_id: str
    event_ts: datetime
    recorded_at: datetime
    sport: str
    target: str
    field: str
    value: str
    actor: str
    note: str = ""


def _text(value: str, name: str, limit: int) -> str:
    v = value.strip()
    if not v or len(v) > limit:
        raise ControlError(f"{name} must be 1..{limit} characters")
    if re.search(r"[<>\x00-\x08\x0b\x0c\x0e-\x1f]", v):
        raise ControlError(f"{name} may not contain markup or control characters (it is shown on the public site as plain text)")
    return v


def _rankings_list(sport: str, field: str, v: str) -> str:
    """Canonical 'a,b' (order kept) or 'none'. Every key must exist for this sport; custom rankings are capped at MAX_CUSTOM_RANKINGS."""
    kind = "custom" if field == "rankings_show" else "media"
    known = [r["key"] for r in RANKINGS[sport][kind]]
    keys = [k.strip().lower() for k in v.split(",") if k.strip()]
    if keys == ["none"]:
        return "none"
    if not keys:
        raise ControlError(f"{field} needs a list of {kind} rankings, or 'none'")
    if not known:
        raise ControlError(f"{sport} has no {kind} rankings to choose from")
    bad = [k for k in keys if k not in known]
    if bad:
        raise ControlError(f"unknown {kind} ranking(s) for {sport}: {', '.join(bad)}; valid: {', '.join(known)}")
    if len(set(keys)) != len(keys):
        raise ControlError(f"{field} lists the same ranking twice")
    if kind == "custom" and len(keys) > MAX_CUSTOM_RANKINGS:
        raise ControlError(f"at most {MAX_CUSTOM_RANKINGS} custom rankings can be shown beside the media polls")
    return ",".join(keys)


def _rankings_primary(sport: str, v: str) -> str:
    known = [r["key"] for r in RANKINGS[sport]["primary"]]
    if not known:
        raise ControlError(f"{sport} has no season resume to order by")
    k = v.strip().lower()
    if k not in known:
        raise ControlError(f"rankings_primary must be one of {', '.join(known)}")
    return k


def validate(sport: str, target: str, field: str, value: str) -> str:
    """Return the canonical value or raise ControlError."""
    if sport not in SPORTS:
        raise ControlError(f"sport must be one of {SPORTS}")
    if field == "role":  # renamed: "role" now means only the site registry's card type
        raise ControlError("the field 'role' was renamed 'lifecycle' (live, shadow or archived); use `lifecycle`")
    if field not in ALL_FIELDS:
        raise ControlError(f"unknown field {field!r}; valid: {', '.join(ALL_FIELDS)}")
    if field in SPORT_FIELDS:
        if target != "*":
            raise ControlError(f"{field} is sport-wide: use target '*'")
    else:
        if target == "*" or not TARGET_RE.match(target):
            raise ControlError(f"target must be a model key like 'v10' (letters, digits, . _ -), got {target!r}")
    if value == UNSET:
        return UNSET
    v = value.strip()
    if field == "site_visible":
        if v.lower() not in ("true", "false"):
            raise ControlError("site_visible must be true or false")
        return v.lower()
    if field == "run_state":
        if v not in ("active", "paused"):
            raise ControlError("run_state must be active or paused")
        return v
    if field == "lifecycle":
        if v not in ("live", "shadow", "archived"):
            raise ControlError("lifecycle must be live, shadow or archived")
        return v
    if field == "artifact_uri":
        if not re.match(r"^gs://[a-z0-9._-]+/[^\s]+$", v):
            raise ControlError("artifact_uri must look like gs://bucket/path/model.pkl")
        return v
    if field == "artifact_sha256":
        if not HEX64.match(v.lower()):
            raise ControlError("artifact_sha256 must be 64 hex characters")
        return v.lower()
    if field == "display_label":
        return _text(v, "display_label", 60)
    if field == "public_note":
        return _text(v, "public_note", 300)
    if field == "banner":
        return _text(v, "banner", 240)
    if field in ("rankings_show", "rankings_media"):
        return _rankings_list(sport, field, v)
    if field == "rankings_primary":
        return _rankings_primary(sport, v)
    if field == "banner_level":
        if v not in BANNER_LEVELS:
            raise ControlError(f"banner_level must be one of {BANNER_LEVELS}")
        return v
    if field == "sort_order":
        try:
            n = int(v)
        except ValueError as e:
            raise ControlError("sort_order must be an integer") from e
        if not -1000 <= n <= 1000:
            raise ControlError("sort_order must be between -1000 and 1000")
        return str(n)
    if field in ("tier_high", "tier_medium"):
        try:
            x = float(v)
        except ValueError as e:
            raise ControlError(f"{field} must be a number") from e
        if not 0.5 < x < 1.0:
            raise ControlError(f"{field} must be between 0.5 and 1.0 (it is a probability of the favoured side)")
        return repr(round(x, 4))
    raise ControlError(f"unhandled field {field}")  # pragma: no cover


def current_state(events: list[Event], now: datetime) -> dict[tuple[str, str], dict]:
    """Python twin of the model_control_current view (a test runs the real SQL and compares the two)."""
    latest: dict[tuple[str, str, str], Event] = {}
    for e in events:
        if e.event_ts > now:
            continue
        k = (e.sport, e.target, e.field)
        cur = latest.get(k)
        if cur is None or (e.event_ts, e.recorded_at, e.event_id) > (cur.event_ts, cur.recorded_at, cur.event_id):
            latest[k] = e
    out: dict[tuple[str, str], dict] = {}
    for (sport, target, field), e in latest.items():
        if e.value == UNSET:
            continue
        row = out.setdefault((sport, target), {})
        row[field] = e.value
        row["updated_at"] = max(row.get("updated_at", e.event_ts), e.event_ts)
    return out


def effective(state: dict, field: str) -> str | None:
    return state.get(field, DEFAULTS.get(field))


# --- stores ---------------------------------------------------------------------------------------

class Store(Protocol):
    def append(self, ev: Event) -> None: ...
    def all_events(self) -> list[Event]: ...


class MemoryStore:
    def __init__(self, events: list[Event] | None = None):
        self.events = list(events or [])

    def append(self, ev: Event) -> None:
        self.events.append(ev)

    def all_events(self) -> list[Event]:
        return list(self.events)


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00:00")


class BqCliStore:
    """Talks to BigQuery through the `bq` CLI with the caller's own gcloud identity, so no service-account permissions are involved."""

    def __init__(self, project: str = "hankstank", dataset: str = "control", account: str | None = "you@example.com",
                 runner=subprocess.run, bq: str = "bq"):
        if is_blocked_account(account):
            raise ControlError("refusing to use the work account")
        self.project, self.dataset, self.account, self.runner, self.bq = project, dataset, account, runner, bq

    def _env(self) -> dict:
        import os
        return {**os.environ, **({"CLOUDSDK_CORE_ACCOUNT": self.account} if self.account else {})}

    def _run(self, args: list[str], sql: str) -> str:
        r = self.runner([self.bq, "--project_id", self.project, "query", "--nouse_legacy_sql", "--quiet", *args, sql],
                        capture_output=True, text=True, env=self._env())
        if r.returncode != 0:
            raise ControlError(f"bq failed: {(r.stderr or r.stdout).strip()[:300]}")
        return r.stdout

    def append(self, ev: Event) -> None:
        sql = (f"INSERT INTO `{self.project}.{self.dataset}.model_control_events` "
               "(event_id, event_ts, recorded_at, sport, target, field, value, actor, note) "
               "VALUES (@event_id, @event_ts, @recorded_at, @sport, @target, @field, @value, @actor, NULLIF(@note, ''))")
        params = [f"--parameter={n}:{t}:{v}" for n, t, v in (
            ("event_id", "STRING", ev.event_id), ("event_ts", "TIMESTAMP", _iso(ev.event_ts)), ("recorded_at", "TIMESTAMP", _iso(ev.recorded_at)),
            ("sport", "STRING", ev.sport), ("target", "STRING", ev.target), ("field", "STRING", ev.field), ("value", "STRING", ev.value),
            ("actor", "STRING", ev.actor), ("note", "STRING", ev.note))]
        self._run(params, sql)

    def all_events(self) -> list[Event]:
        out = self._run(["--format=json", "--max_rows=100000"],
                        f"SELECT event_id, CAST(event_ts AS STRING) event_ts, CAST(recorded_at AS STRING) recorded_at, sport, target, field, value, actor, "
                        f"IFNULL(note, '') note FROM `{self.project}.{self.dataset}.model_control_events`")
        rows = json.loads(out or "[]")
        return [Event(r["event_id"], _parse_ts(r["event_ts"]), _parse_ts(r["recorded_at"]), r["sport"], r["target"], r["field"], r["value"] or "",
                      r["actor"] or "", r["note"] or "") for r in rows]


class BqClientStore:
    """The same store over the BigQuery API with the process's own credentials (the lab's service account on the homelab, where there is no gcloud).

    Reading needs the view/table read grant the health job already has. APPENDING needs dataEditor on the `control` dataset, a grant that is the
    owner's to make; without it `append` raises ControlError with the BigQuery message, and the caller shows the equivalent command instead."""

    def __init__(self, project: str = "hankstank", dataset: str = "control", client=None):
        self.project, self.dataset, self._client = project, dataset, client

    def _bq(self):
        if self._client is None:
            from google.cloud import bigquery
            self._client = bigquery.Client(project=self.project)
        return self._client

    def _query(self, sql: str, params: list[tuple[str, str, str]] | None = None):
        from google.cloud import bigquery
        cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter(n, t, v) for n, t, v in (params or [])])
        try:
            return list(self._bq().query(sql, job_config=cfg).result())
        except Exception as e:  # noqa: BLE001 - permission, quota, network: all the same to the caller
            raise ControlError(f"bq failed: {str(e).strip()[:300]}") from e

    def append(self, ev: Event) -> None:
        sql = (f"INSERT INTO `{self.project}.{self.dataset}.model_control_events` "
               "(event_id, event_ts, recorded_at, sport, target, field, value, actor, note) "
               "VALUES (@event_id, TIMESTAMP(@event_ts), TIMESTAMP(@recorded_at), @sport, @target, @field, @value, @actor, NULLIF(@note, ''))")
        self._query(sql, [("event_id", "STRING", ev.event_id), ("event_ts", "STRING", _iso(ev.event_ts)), ("recorded_at", "STRING", _iso(ev.recorded_at)),
                          ("sport", "STRING", ev.sport), ("target", "STRING", ev.target), ("field", "STRING", ev.field), ("value", "STRING", ev.value),
                          ("actor", "STRING", ev.actor), ("note", "STRING", ev.note)])

    def all_events(self) -> list[Event]:
        rows = self._query(f"SELECT event_id, CAST(event_ts AS STRING) event_ts, CAST(recorded_at AS STRING) recorded_at, sport, target, field, value, actor, "
                           f"IFNULL(note, '') note FROM `{self.project}.{self.dataset}.model_control_events`")
        return [Event(r["event_id"], _parse_ts(r["event_ts"]), _parse_ts(r["recorded_at"]), r["sport"], r["target"], r["field"], r["value"] or "",
                      r["actor"] or "", r["note"] or "") for r in rows]


def _parse_ts(s: str) -> datetime:
    s = s.replace(" UTC", "+00:00").replace(" ", "T", 1) if "T" not in s else s.replace(" UTC", "+00:00")
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


# --- operations -------------------------------------------------------------------------------------

def confirm_phrase(sport: str, target: str) -> str:
    return f"{sport}/{target}"


def needs_confirmation(field: str, value: str) -> bool:
    """Changes that alter what the live site shows or what runs in production need the typed phrase."""
    if field in ("rankings_show", "rankings_media", "rankings_primary"):
        return True  # either way it changes the table visitors see
    if value == UNSET:
        return field in ("artifact_uri", "artifact_sha256", "lifecycle")
    return ((field == "site_visible" and value == "false") or (field == "run_state" and value == "paused")
            or field in ("artifact_uri", "artifact_sha256") or (field == "lifecycle" and value == "live") or field == "banner")


class Control:
    def __init__(self, store: Store, actor: str, *, now: Callable[[], datetime] | None = None,
                 known_models: Callable[[str], list[str]] | None = None, audit_path=None,
                 production_keys: Callable[[str], set[str]] | None = None):
        self.store, self.actor = store, actor
        self.production_keys = production_keys
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.known_models = known_models
        self.audit_path = audit_path

    # reading
    def state(self, sport: str | None = None) -> dict[tuple[str, str], dict]:
        st = current_state(self.store.all_events(), self._now())
        return {k: v for k, v in st.items() if sport in (None, k[0])}

    def history(self, sport: str, target: str, field: str | None = None, limit: int = 20) -> list[Event]:
        evs = [e for e in self.store.all_events() if e.sport == sport and e.target == target and (field is None or e.field == field)]
        return sorted(evs, key=lambda e: (e.event_ts, e.recorded_at, e.event_id), reverse=True)[:limit]

    def visible_models(self, sport: str) -> list[str] | None:
        """Known models that are currently visible, or None if the model list is unavailable."""
        if self.known_models is None:
            return None
        st = self.state(sport)
        return [m for m in self.known_models(sport) if effective(st.get((sport, m), {}), "site_visible") == "true"]

    # writing
    def set(self, sport: str, target: str, field: str, value: str, *, note: str = "", at: datetime | None = None,
            confirm: str | None = None, allow_empty: bool = False, dry_run: bool = False) -> Event:
        v = validate(sport, target, field, value)
        if needs_confirmation(field, v) and confirm != confirm_phrase(sport, target):
            raise ConfirmationRequired(f"this changes the live site or production; confirm with {confirm_phrase(sport, target)!r}")
        if field == "site_visible" and v == "false" and not allow_empty:
            vis = self.visible_models(sport)
            if vis is not None and set(vis) <= {target}:
                raise ControlError(f"{target} is the only visible {sport} model; hiding it would empty the site. "
                                   "Show another model first, or pass allow_empty to do it on purpose.")
        if field == "site_visible" and v == "false" and self.production_keys is not None and not allow_empty:
            if target in self.production_keys(sport) and not self.state(sport).get((sport, "*"), {}).get("banner"):
                raise ControlError(f"{target} is the production model and the spine of the {sport} pages: hiding it needs a banner that explains why "
                                   "(use `hide --banner TEXT`, or set a banner first)")
        if field == "lifecycle" and v == "live":
            st = self.state(sport).get((sport, target), {})
            if not (st.get("artifact_uri") and st.get("artifact_sha256")) and (st.get("artifact_uri") or st.get("artifact_sha256")):
                raise ControlError("a pinned artifact needs both artifact_uri and artifact_sha256")
        now = self._now()
        if at is not None and at.tzinfo is None:
            raise ControlError("a scheduled time needs a timezone (e.g. 2026-10-05T09:00:00-04:00)")
        ev = Event(str(uuid.uuid4()), at or now, now, sport, target, field, v, self.actor, note.strip()[:300])
        if not dry_run:
            self.store.append(ev)
            audit.record("control_set", path=self.audit_path, sport=sport, target=target, field=field, value=v, at=str(ev.event_ts), note=ev.note)
        return ev

    def hide(self, sport: str, target: str, *, confirm: str | None = None, banner: str | None = None, banner_level: str = "warn",
             note: str = "", allow_empty: bool = False, dry_run: bool = False) -> list[Event]:
        """Hide a model from the site. `banner` (shown to visitors) is set FIRST, so the site never shows a silent gap."""
        evs = []
        if banner:
            if confirm != confirm_phrase(sport, target):
                raise ConfirmationRequired(f"this changes the live site; confirm with {confirm_phrase(sport, target)!r}")
            evs.append(self.set(sport, "*", "banner", banner, note=note, confirm=confirm_phrase(sport, "*"), dry_run=dry_run))
            evs.append(self.set(sport, "*", "banner_level", banner_level, note=note, dry_run=dry_run))
        evs.append(self.set(sport, target, "site_visible", "false", note=note, confirm=confirm, allow_empty=allow_empty, dry_run=dry_run))
        return evs

    def pin(self, sport: str, target: str, uri: str, sha256: str, *, confirm: str | None = None, note: str = "", dry_run: bool = False) -> list[Event]:
        """Make `target` the live model, running exactly this artifact. The sha256 is mandatory: the pipeline verifies it before loading."""
        validate(sport, target, "artifact_uri", uri)
        validate(sport, target, "artifact_sha256", sha256)
        if confirm != confirm_phrase(sport, target):
            raise ConfirmationRequired(f"pinning changes what runs in production; confirm with {confirm_phrase(sport, target)!r}")
        evs = [self.set(sport, target, "artifact_uri", uri, note=note, confirm=confirm, dry_run=dry_run),
               self.set(sport, target, "artifact_sha256", sha256, note=note, confirm=confirm, dry_run=dry_run),
               self.set(sport, target, "lifecycle", "live", note=note, confirm=confirm, dry_run=dry_run)]  # lifecycle last: never live without its artifact
        return evs

    def revert(self, sport: str, target: str, field: str, *, confirm: str | None = None, dry_run: bool = False) -> Event:
        """Undo the latest change to a field by appending the value that was in force before it."""
        now = self._now()
        effective_events = sorted((e for e in self.history(sport, target, field, limit=1000) if e.event_ts <= now),
                                  key=lambda e: (e.event_ts, e.recorded_at, e.event_id))
        if len(effective_events) < 1:
            raise ControlError(f"no history for {sport}/{target}.{field}")
        prev = effective_events[-2].value if len(effective_events) >= 2 else UNSET
        return self.set(sport, target, field, prev, note=f"revert of {effective_events[-1].event_id[:8]}", confirm=confirm, allow_empty=True,
                        dry_run=dry_run)

    def freeze(self, sport: str, message: str, *, confirm: str | None = None) -> list[Event]:
        """Kill switch: hide every known model of a sport and show a banner. `unfreeze` restores exactly what this hid."""
        if confirm != confirm_phrase(sport, "*"):
            raise ConfirmationRequired(f"freezing hides every {sport} model from the site; confirm with {confirm_phrase(sport, '*')!r}")
        if self.known_models is None:
            raise ControlError("cannot freeze: the list of known models is unavailable")
        tag = f"freeze:{uuid.uuid4().hex[:8]}"
        evs = []
        for m in self.visible_models(sport) or []:
            evs.append(self.set(sport, m, "site_visible", "false", note=tag, confirm=confirm_phrase(sport, m), allow_empty=True))
        evs.append(self.set(sport, "*", "banner", message, note=tag, confirm=confirm_phrase(sport, "*")))
        evs.append(self.set(sport, "*", "banner_level", "warn", note=tag))
        return evs

    def unfreeze(self, sport: str) -> list[Event]:
        """Undo the latest freeze. A field someone changed AFTER the freeze is left alone: only fields still set by the freeze are reverted."""
        evs_all = self.store.all_events()
        tags = {e.note for e in evs_all if e.sport == sport and e.note.startswith("freeze:")}
        if not tags:
            raise ControlError(f"no freeze found for {sport}")
        tag = sorted(tags, key=lambda t: max(e.recorded_at for e in evs_all if e.note == t))[-1]
        out = []
        for e in [x for x in evs_all if x.note == tag and x.sport == sport]:
            newest = self.history(sport, e.target, e.field, limit=1)[0]
            if newest.event_id == e.event_id:  # still the freeze's own value, so it is ours to undo
                out.append(self.revert(sport, e.target, e.field, confirm=confirm_phrase(sport, e.target)))
        return out
