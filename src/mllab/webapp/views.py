"""The read side of the app (models, jobs, health, GCP, audit, notebooks) and the one registry write it offers. Plain functions over the existing modules
(registry, jobs, runs, audit, scheduler): the app adds no storage of its own."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .. import audit, control as K, runs
from . import history
from ..jobs import ID_RE, JobError, LocalRunner
from ..pipeline.jobs import JOBS, PREGAME
from ..registry import Registry, RegistryError

PROD_LIKE = ("prod", "production")


class Refused(Exception):
    def __init__(self, reasons: list[str], status: int = 409):
        super().__init__("; ".join(reasons))
        self.reasons, self.status = reasons, status


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------------------------------------ models
def models(reg: Registry, audit_path: Path | None = None) -> dict:
    out = []
    entries = _audit_entries(audit_path)
    root = reg.root
    if not root.is_dir():
        return {"projects": out}
    for pdir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        mods = []
        for mdir in sorted(p for p in pdir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            try:
                versions = reg.versions(pdir.name, mdir.name)
                if not versions:
                    continue
                channels = reg.channels(pdir.name, mdir.name)
                by_channel: dict[str, list[str]] = {}
                for ch, v in channels.items():
                    by_channel.setdefault(v, []).append(ch)
                vs = []
                for v in versions:
                    m = reg.meta(pdir.name, mdir.name, v)
                    vs.append({"version": v, "created_at": m.created_at, "parent": m.parent, "sha256": m.sha256[:12], "git_sha": (m.git_sha or "")[:10],
                               "git_dirty": m.git_dirty, "features": len(m.features), "metrics": m.metrics, "source": m.source,
                               "channels": sorted(by_channel.get(v, []))})
                rollback = {ch: r for ch, v in channels.items() if (r := history.previous_version(entries, pdir.name, mdir.name, ch, v)) and r["to"] in versions}
                mods.append({"model": mdir.name, "channels": channels, "rollback": rollback, "versions": list(reversed(vs)),
                             "deployments": reg.deployments(pdir.name, mdir.name)[-5:]})
            except (RegistryError, OSError, ValueError, KeyError) as e:  # one unreadable model must not hide the rest
                mods.append({"model": mdir.name, "error": f"{type(e).__name__}: {e}"[:160], "channels": {}, "rollback": {}, "versions": [], "deployments": []})
        if mods:
            out.append({"project": pdir.name, "models": mods})
    return {"projects": out}


def promote(reg: Registry, project: str, model: str, version: str, channel: str, actor: str, confirm: str | None, audit_path: Path | None = None) -> dict:
    """Point a channel at a version. A pointer only: serving a model is a separate, confirmed deploy. The prod channel still needs the typed phrase,
    because other tools read it as 'what is live'."""
    phrase = f"{model}@{channel}"
    if channel in PROD_LIKE and confirm != phrase:
        raise Refused([f"this moves the {channel} pointer; type {phrase} to confirm"], 400)
    try:
        previous = reg.promote(project, model, version, channel)
    except (RegistryError, OSError) as e:
        raise Refused([str(e)], 400) from e
    audit.record("promote", path=audit_path, project=project, model=model, version=version, channel=channel, previous=previous, by=actor)
    return {"previous": previous, "now": version}


def rollback_channel(reg: Registry, project: str, model: str, channel: str, actor: str, confirm: str | None, audit_path: Path | None = None) -> dict:
    """Point a channel back at the version it had before its latest move. Only where the audit log shows that move, and only if that version still exists."""
    now = reg.channels(project, model).get(channel)
    prev = history.previous_version(_audit_entries(audit_path), project, model, channel, now)
    if prev is None:
        raise Refused([f"no earlier version on record for {model}@{channel}: nothing to roll back to"])
    if prev["to"] not in reg.versions(project, model):
        raise Refused([f"{prev['to']} is no longer in the registry"])
    res = promote(reg, project, model, prev["to"], channel, actor, confirm, audit_path)
    return {**res, "rolled_back": True}


def _audit_entries(path: Path | None) -> list[dict]:
    try:
        return audit.read(path)
    except (OSError, ValueError):
        return []


# -------------------------------------------------------------------------------------------------------------------------------- jobs
def jobs(runner: LocalRunner, limit: int = 40) -> dict:
    rows = []
    if runner.jobs.is_dir():
        for jd in runner.jobs.iterdir():
            if not (ID_RE.match(jd.name) and jd.is_dir()):
                continue
            spec = _read_json(jd / "spec.json") or {}
            try:
                st = runner.status(jd.name)
            except (JobError, OSError, ValueError):
                st = {"state": "?"}
            rows.append({"id": jd.name, "name": spec.get("name", "?"), "submitted_at": spec.get("submitted_at", ""), "state": st.get("state", "?"),
                         "detail": st.get("detail"), "caps": {k: spec.get(k) for k in ("memory_mb", "cpu_pct", "timeout_s") if k in spec}})
    rows.sort(key=lambda r: r["submitted_at"], reverse=True)
    return {"jobs": rows[:limit], "experiments": runs.list_runs(runner, limit=limit)}


def job_log(runner: LocalRunner, job_id: str, offset: int = 0, cap: int = 64 * 1024) -> dict:
    try:
        text, nxt = runner.logs(job_id, max(0, offset))
    except (JobError, OSError) as e:
        raise Refused([str(e)], 404) from e
    if len(text) > cap:
        text, nxt = text[-cap:], nxt
    return {"id": job_id, "text": text, "offset": nxt}


def cancel(runner: LocalRunner, job_id: str, actor: str, audit_path: Path | None = None) -> dict:
    try:
        st = runner.status(job_id)
        if st.get("state") not in ("queued", "running"):
            raise Refused([f"job is {st.get('state')}, nothing to cancel"])
        runner.cancel(job_id)
    except JobError as e:
        raise Refused([str(e)], 404) from e
    audit.record("job_cancel", path=audit_path, job=job_id, by=actor)
    return {"id": job_id, "cancelling": True}


# ------------------------------------------------------------------------------------------------------------------------ health, gcp, audit
def health(root: Path) -> dict:
    return _read_json(root / "health" / "health.json") or {"findings": [], "sports": {}, "sources": {}, "missing": True}


def gcp(jobs_by_name: dict[str, dict] | None, error: str | None, local_modes: dict[str, str]) -> dict:
    """Cloud Scheduler as it is, next to where each job runs now. A job live here AND enabled there is the one thing this page exists to catch."""
    rows = []
    for name, j in sorted((jobs_by_name or {}).items()):
        here = local_modes.get(name)
        enabled = j.get("state") == "ENABLED"
        row = {k: j.get(k) for k in ("name", "state", "schedule", "tz", "mode", "body_ok", "last_attempt_at", "last_attempt_ok", "next_run")}
        row.update(here=here, conflict=enabled and here == "live",
                   commands=[f"mllab control scheduler {'pause' if enabled else 'resume'} {name} --confirm {name}"])
        rows.append(row)
    return {"available": jobs_by_name is not None, "error": error, "jobs": rows}


def audit_tail(path: Path | None = None, limit: int = 100) -> dict:
    try:
        rows = audit.read(path)
    except (OSError, ValueError):
        rows = []
    return {"entries": list(reversed(rows[-limit:]))}


def notebooks(work: Path) -> dict:
    out = []
    if work.is_dir():
        for p in sorted(work.rglob("*.ipynb")):
            if ".ipynb_checkpoints" in p.parts:
                continue
            rel = p.relative_to(work).as_posix()
            out.append({"path": rel, "modified": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"), "link": f"/lab/tree/{rel}"})
    return {"notebooks": out, "lab": "/lab"}


# ------------------------------------------------------------------------------------------------------------------------------ the site
def _split(v: str | None) -> list[str] | None:
    """A stored 'a,b' / 'none' / unset as the list the page works with: None = default, [] = none."""
    if v is None:
        return None
    return [] if v == "none" else [x for x in v.split(",") if x]


def site_rankings(read_state, view_columns=None) -> dict:
    """Which ranking columns the live site draws, per sport, beside what could be drawn.

    `read_state()` returns the control view as {(sport, target): {field: value}} (it may raise: the page then says so and shows the catalog only).
    `view_columns()` returns the view's column names, so the page can warn when the view has not been given the two new columns yet (writes would
    be recorded and then have no effect on the site)."""
    catalog = {sp: {"custom": c["custom"], "media": c["media"], "primary": c["primary"], "primary_default": K.PRIMARY_DEFAULT.get(sp),
                    "max_custom": K.MAX_CUSTOM_RANKINGS} for sp, c in K.RANKINGS.items() if c["custom"] or c["media"] or c["primary"]}
    out = {"catalog": catalog, "current": {}, "available": True, "error": None, "view_ready": None}
    try:
        state = read_state()
        for sp in catalog:
            row = state.get((sp, "*"), {})
            out["current"][sp] = {"show": _split(row.get("rankings_show")), "media": _split(row.get("rankings_media")),
                                  "primary": row.get("rankings_primary") or None}
    except Exception as e:  # noqa: BLE001 - BigQuery unreachable or not permitted: show the catalog, say why
        out.update(available=False, error=f"{type(e).__name__}: {e}"[:300])
    if view_columns is not None:
        try:
            cols = set(view_columns())
            out["view_ready"] = {"rankings_show", "rankings_media", "rankings_primary"} <= cols
        except Exception:  # noqa: BLE001 - a hint, never fatal
            out["view_ready"] = None
    return out


def _cli_commands(sport: str, show, media, primary=...) -> list[str]:
    cmd = f"mllab control rankings {sport} --confirm {sport}/*"
    parts = []
    if show is not ...:
        parts.append("--show-default" if show is None else f"--show {','.join(show) or 'none'}")
    if media is not ...:
        parts.append("--media-default" if media is None else f"--media {','.join(media) or 'none'}")
    if primary is not ...:
        parts.append("--primary-default" if primary is None else f"--primary {primary}")
    return [f"{cmd} {' '.join(parts)}".strip()]


def set_site_rankings(store, sport: str, show, media, login: str, confirm: str | None, audit_path: Path | None = None, primary=...) -> dict:
    """Choose which ranking columns hankstank.com draws for a sport. `show` / `media`: Ellipsis = leave as is, None = back to the default (every
    column that has data), a list = exactly these in this order ([] = none). Typed confirmation, validated against the catalog, audited, undoable
    (the next change is just another event, and `mllab control revert` works on it). `primary`: Ellipsis = leave as is, None = the sport's default
    order, 'resume' / 'rating' = order the table by that."""
    r = K.RANKINGS.get(sport)
    if not r or not (r["custom"] or r["media"] or r["primary"]):
        raise Refused([f"{sport} has no ranking columns to choose"], 400)
    if show is ... and media is ... and primary is ...:
        raise Refused(["nothing to change"], 400)
    c = K.Control(store, f"app:{login}", audit_path=audit_path)
    changes = []
    for field, val in (("rankings_media", media), ("rankings_show", show)):
        if val is ...:
            continue
        changes.append((field, K.UNSET if val is None else ("none" if not val else ",".join(val))))
    if primary is not ...:
        changes.append(("rankings_primary", K.UNSET if primary is None else str(primary)))
    try:
        for field, value in changes:  # validate all before writing any, so a typo in the second cannot leave the first applied
            K.validate(sport, "*", field, value)
        for field, value in changes:
            c.set(sport, "*", field, value, note="via mllab app", confirm=confirm)
    except K.ConfirmationRequired as e:
        raise Refused([str(e)], 409) from e
    except K.ControlError as e:
        msg = str(e)
        hint = ("The lab's service account may not write the control dataset yet; run this from your laptop instead, or grant it dataEditor on `hankstank.control` "
                "(scripts/grant_control_write.sh, which you run).") if "bq failed" in msg else ""
        raise Refused([msg, *([hint] if hint else []), "Equivalent command: " + _cli_commands(sport, show, media, primary)[0]], 409) from e
    return {"sport": sport, "show": None if show is ... else show, "media": None if media is ... else media,
            "primary": None if primary is ... else primary,
            "commands": _cli_commands(sport, show, media, primary), "note": "The site picks this up within about a minute."}
