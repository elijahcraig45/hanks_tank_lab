"""What the audit log says about the past, for rollback: a job's earlier mode, a channel's earlier version.

Nothing here keeps its own state: the audit log is the record, written by the broker for mode changes and by the CLI and the app for channel moves. A rollback
is only offered where the log shows a change to undo, and it is just another change (so it is audited, and goes through the same gates)."""
from __future__ import annotations

import re

MODE_DETAIL = re.compile(r"^(?P<job>[\w-]+): (?P<before>off|dry|live) -> (?P<mode>off|dry|live)\b")


def mode_changes(entries: list[dict], job: str) -> list[dict]:
    """Every recorded change of `job`'s mode, oldest first: {ts, before, mode, by}. Structured entries are read as written; the first ones, which only carried a
    sentence ("job: dry -> live (...)"), are read from it."""
    out = []
    for e in entries:
        if e.get("action") != "pipeline_set_mode" or e.get("job") != job:
            continue
        before, mode = e.get("before"), e.get("mode")
        if not (before and mode):
            m = MODE_DETAIL.match(str(e.get("detail", "")))
            if not m or m.group("job") != job:
                continue
            before, mode = m.group("before"), m.group("mode")
        if before != mode:
            out.append({"ts": e.get("ts"), "before": before, "mode": mode, "by": e.get("by")})
    return out


def previous_mode(entries: list[dict], job: str, current: str) -> dict | None:
    """The mode to roll `job` back to: the one before the most recent change that led to `current`. None if the log shows no such change (the job has always
    been in this mode, or was changed by hand with pipeline-ctl, which is not in this log)."""
    for ch in reversed(mode_changes(entries, job)):
        if ch["mode"] == current:
            return {"to": ch["before"], "from": current, "at": ch["ts"], "by": ch["by"]}
    return None


def channel_moves(entries: list[dict], project: str, model: str, channel: str) -> list[dict]:
    """Every recorded move of a channel, oldest first: {ts, version, previous, by}. The CLI writes `previous`; older app entries wrote `was`."""
    out = []
    for e in entries:
        if e.get("action") == "promote" and (e.get("project"), e.get("model"), e.get("channel")) == (project, model, channel):
            out.append({"ts": e.get("ts"), "version": e.get("version"), "previous": e.get("previous", e.get("was")), "by": e.get("by") or e.get("actor")})
    return out


def previous_version(entries: list[dict], project: str, model: str, channel: str, current: str | None) -> dict | None:
    """The version to roll a channel back to: where it pointed before the most recent move to `current`. None for a channel's first-ever move."""
    for mv in reversed(channel_moves(entries, project, model, channel)):
        if mv["version"] == current and mv["previous"] and mv["previous"] != current:
            return {"to": mv["previous"], "from": current, "at": mv["ts"], "by": mv["by"]}
    return None
