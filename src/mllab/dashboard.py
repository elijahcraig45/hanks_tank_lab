"""Private health dashboard: one self-contained static HTML page plus health.json, rebuilt by `mllab health --build`.

Design rules followed: status colours are reserved and always paired with an icon and a word; text uses text tokens, never a status colour;
light and dark surfaces are declared as tokens under the media query and the data-theme attribute; thin hairline marks; a table is the primary view.
Every string that reaches the page is HTML-escaped. There is no script, no external asset and no network call."""
from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path

from . import health as H

STATUS = {  # status -> (icon, word). The colour lives in CSS via data-s so it stays reserved for status.
    "ok": ("✓", "Healthy"), "warn": ("▲", "Warning"), "down": ("✖", "Down"),
    "paused": ("⏸", "Paused"), "hidden": ("◌", "Hidden"),
}
SEV = {"critical": ("✖", "Critical", "down"), "warn": ("▲", "Warning", "warn"), "info": ("ℹ", "Info", "info")}

CSS = """
:root{--bg:#f7f7f5;--surface:#fff;--ink:#1a1a19;--ink2:#52514e;--muted:#767570;--line:#e3e2de;--bar:#3f6fa8;
--good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b;--info:#52514e}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#121211;--surface:#1b1b1a;--ink:#f0efec;--ink2:#b9b8b3;--muted:#8d8c87;--line:#2f2f2d;--bar:#7aa6dc}}
:root[data-theme=dark]{--bg:#121211;--surface:#1b1b1a;--ink:#f0efec;--ink2:#b9b8b3;--muted:#8d8c87;--line:#2f2f2d;--bar:#7aa6dc}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 48px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 8px}
.sub{color:var(--muted);font-size:13px}.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;overflow-x:auto}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:top;font-size:14px}
th{color:var(--ink2);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}tr:last-child td{border-bottom:0}
.num{text-align:right;font-variant-numeric:tabular-nums}
.st{display:inline-flex;gap:6px;align-items:center;white-space:nowrap}.st i{font-style:normal;font-size:13px}
[data-s=ok] i{color:var(--good)}[data-s=warn] i{color:var(--warning)}[data-s=down] i{color:var(--critical)}[data-s=info] i,[data-s=paused] i,[data-s=hidden] i{color:var(--info)}
.tiles{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}.tile{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px 16px;min-width:140px}
.tile b{display:block;font-size:26px;font-variant-numeric:tabular-nums}.tile span{color:var(--ink2);font-size:13px}
.bar{height:6px;width:96px;background:var(--line);border-radius:3px;overflow:hidden;display:inline-block;vertical-align:middle}.bar>span{display:block;height:100%;background:var(--bar)}
code{font-size:12px;background:var(--bg);padding:1px 5px;border-radius:4px}.empty{padding:16px;color:var(--ink2)}
"""


def _st(kind: str, icon: str, word: str) -> str:
    return f'<span class="st" data-s="{escape(kind)}"><i aria-hidden="true">{icon}</i>{escape(word)}</span>'


def _ago(then: str | None, now: datetime) -> str:
    if not then:
        return "never"
    h = (now - datetime.fromisoformat(then)).total_seconds() / 3600
    return f"{h * 60:.0f} min ago" if h < 1 else f"{h:.1f} h ago" if h < 48 else f"{h / 24:.1f} d ago"


def _cov(m: dict, scheduled: int) -> str:
    if m["is_shadow"] or scheduled <= 0:
        return f'<span class="num">{m["rows_today"]}</span>'
    pct = min(100, round(100 * m["games_today"] / scheduled))
    return f'<span class="bar" role="img" aria-label="{pct}% of games covered"><span style="width:{pct}%"></span></span> {m["games_today"]}/{scheduled}'


def _tiles(d: dict) -> str:
    c = {k: 0 for k in ("critical", "warn", "info")}
    for f in d["findings"]:
        c[f["severity"]] += 1
    items = [("critical", "Critical"), ("warn", "Warnings"), ("info", "Notes")]
    return '<div class="tiles">' + "".join(f'<div class="tile"><b>{c[k]}</b><span>{lab}</span></div>' for k, lab in items) + "</div>"


def _findings(d: dict) -> str:
    if not d["findings"]:
        return '<div class="card"><div class="empty">' + _st("ok", *STATUS["ok"]) + " Nothing needs attention.</div></div>"
    rows = []
    for f in d["findings"]:
        icon, word, kind = SEV[f["severity"]]
        where = " / ".join(x for x in (f["sport"], f["model"]) if x != "*") or "lab"
        fix = f'<code>{escape(f["remedy"]["action"])}</code>' if f["remedy"] else ""
        rows.append(f"<tr><td>{_st(kind, icon, word)}</td><td>{escape(where)}</td><td>{escape(f['message'])}</td><td>{fix}</td></tr>")
    return '<div class="card"><table><tr><th>Severity</th><th>Where</th><th>What</th><th>Auto-fix</th></tr>' + "".join(rows) + "</table></div>"


def _models(sport: str, s: dict, now: datetime) -> str:
    banner = f'<p class="sub">Banner: {escape(s["banner"])}</p>' if s.get("banner") else ""
    rows = []
    for m in s["models"]:
        icon, word = STATUS[m["status"]]
        role = "shadow" if m["is_shadow"] else "live"
        std = "-" if m["prob_std"] is None else f'{m["prob_std"]:.3f}'
        rows.append(f'<tr><td>{escape(m["model"])}</td><td>{role}</td><td>{_st(m["status"], icon, word)}</td><td>{_cov(m, s["scheduled_today"])}</td>'
                    f'<td>{escape(_ago(m["last_prediction_at"], now))}</td><td class="num">{std}</td><td class="num">{m["bad_rows"]}</td></tr>')
    body = ('<table><tr><th>Model</th><th>Role</th><th>Status</th><th>Today</th><th>Last write</th><th class="num">Spread</th><th class="num">Bad rows</th></tr>'
            + "".join(rows) + "</table>") if rows else '<div class="empty">No model data (source unavailable).</div>'
    season = "in season" if s["in_season"] else "off-season"
    return f'<h2>{escape(sport.upper())} <span class="sub">{season}, {s["scheduled_today"]} games today</span></h2>{banner}<div class="card">{body}</div>'


def _jobs(d: dict) -> str:
    rows = [f'<tr><td>{escape(j["name"])}</td><td>{escape(str(j["state"]))}</td><td>{escape(str(j["mode"] or "-"))}</td>'
            f'<td>{"ok" if j["body_ok"] else "INVALID JSON"}</td><td>{"-" if j["last_attempt_ok"] is None else "ok" if j["last_attempt_ok"] else "FAILED"}</td></tr>'
            for s in d["sports"].values() for j in s["jobs"]]
    if not rows:
        return ""
    return '<h2>Scheduler jobs</h2><div class="card"><table><tr><th>Job</th><th>State</th><th>Mode</th><th>Body</th><th>Last run</th></tr>' + "".join(rows) + "</table></div>"


def _sources(d: dict) -> str:
    rows = [f'<tr><td>{escape(k)}</td><td>{escape(v)}</td></tr>' for k, v in sorted(d["sources"].items())]
    lab = ", ".join(f"{escape(k)}: {escape(str(v))}" for k, v in d["lab"].items()) or "no lab data"
    return f'<h2>Data sources</h2><div class="card"><table>{"".join(rows) or "<tr><td>none</td></tr>"}</table></div><p class="sub">Lab: {lab}</p>'


def render(d: dict, remedies: list[dict] | None = None) -> str:
    """`d` is health.to_json output. Returns a complete HTML document."""
    now = datetime.fromisoformat(d["taken_at"])
    worst = "critical" if any(f["severity"] == "critical" for f in d["findings"]) else "warn" if any(f["severity"] == "warn" for f in d["findings"]) else "ok"
    head = {"critical": _st("down", *STATUS["down"]), "warn": _st("warn", *STATUS["warn"]), "ok": _st("ok", *STATUS["ok"])}[worst]
    log = ""
    if remedies:
        rows = "".join(f'<tr><td>{escape(r.get("at", ""))}</td><td>{escape(r.get("rule", ""))}</td><td>{escape(r.get("action", ""))}</td><td>{escape(r.get("outcome", ""))}</td></tr>'
                       for r in remedies[-15:][::-1])
        log = f'<h2>Recent self-healing</h2><div class="card"><table><tr><th>When</th><th>Rule</th><th>Action</th><th>Outcome</th></tr>{rows}</table></div>'
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<meta name="robots" content="noindex"><title>Model Health</title><style>{CSS}</style></head><body><main>'
            f'<h1>Model health {head}</h1><p class="sub">Updated {escape(now.strftime("%Y-%m-%d %H:%M UTC"))}. Private to the tailnet.</p>'
            f'{_tiles(d)}<h2>Needs attention</h2>{_findings(d)}'
            + "".join(_models(k, s, now) for k, s in d["sports"].items()) + _jobs(d) + log + _sources(d) + "</main></body></html>")


def build(snap: H.Snapshot, out_dir: Path, *, thresholds: H.Thresholds | None = None, remedies: list[dict] | None = None) -> dict:
    """Evaluate the snapshot, write health.json and index.html atomically, append to history.jsonl. Returns the JSON document."""
    findings = H.evaluate(snap, thresholds)
    d = H.to_json(snap, findings)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in (("health.json", json.dumps(d, indent=1)), ("index.html", render(d, remedies))):
        tmp = out_dir / (name + ".tmp")
        tmp.write_text(text)
        tmp.replace(out_dir / name)
    counts = {s: sum(f["severity"] == s for f in d["findings"]) for s in H.SEVERITIES}
    with (out_dir / "history.jsonl").open("a") as fh:
        fh.write(json.dumps({"at": d["taken_at"], **counts}) + "\n")
    return d
