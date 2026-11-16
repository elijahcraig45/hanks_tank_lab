"""Experiment leaderboard and model cards, built from what jobs already wrote to disk (nothing new is recorded).

Runs use different games and date ranges, so numbers are NEVER ranked across runs. Each run gets its own verdict:
does its best contender (by log loss) beat the always-home prior with a paired CI that excludes zero, or is it noise?
"""
from __future__ import annotations

import json
from pathlib import Path

from .jobs import ID_RE, LocalRunner
from .registry import Registry


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _paired(res: dict, a: str, b: str) -> dict | None:
    """Paired log-loss difference a - b with its CI, whichever way round the worker stored the pair."""
    p = res.get("bootstrap", {}).get("paired", {})
    if f"{a} - {b}" in p:
        d = p[f"{a} - {b}"]
        return {"diff": d["log_loss_diff"], "ci": d["log_loss_diff_ci"]}
    if f"{b} - {a}" in p:
        d = p[f"{b} - {a}"]
        return {"diff": -d["log_loss_diff"], "ci": [-d["log_loss_diff_ci"][1], -d["log_loss_diff_ci"][0]]}
    return None


def verdict(res: dict) -> dict:
    metrics = res["metrics"]
    best = min(metrics, key=lambda k: metrics[k]["log_loss"])
    out = {"best": best, "log_loss": metrics[best]["log_loss"], "accuracy": metrics[best]["accuracy"]}
    if best == "home_prior" or "home_prior" not in metrics:
        out["verdict"] = "the always-home prior is the best contender" if best == "home_prior" else "no home prior to compare against"
        return out
    d = _paired(res, best, "home_prior")
    out["vs_home_prior"] = d
    if d and d["ci"][1] < 0:
        out["verdict"] = "beats the home prior (log-loss CI excludes 0)"
    else:
        out["verdict"] = "NOT distinguishable from the home prior (CI spans 0)"
    # the home prior is a low bar; also say how the winner fares against each probability-column baseline (Elo, prod, ...)
    extras = {}
    for k, v in res.get("info", {}).items():
        if v.get("kind") == "column" and k != best:
            c = _paired(res, best, k)
            if c:
                extras[k] = c
    out["vs_columns"] = extras
    if extras:
        parts = [f"{k}: {'beats' if c['ci'][1] < 0 else 'WORSE than' if c['ci'][0] > 0 else 'not distinguishable from'}" for k, c in extras.items()]
        out["verdict"] += "; vs " + ", ".join(parts)
    return out


def list_runs(runner: LocalRunner, *, limit: int = 30) -> list[dict]:
    rows = []
    if not runner.jobs.is_dir():
        return rows
    for jd in runner.jobs.iterdir():
        if not (ID_RE.match(jd.name) and jd.is_dir()):
            continue
        spec = _read(jd / "spec.json") or {}
        out = jd / "code" / spec.get("outputs", "outputs")
        row = {"job": jd.name, "name": spec.get("name", "?"), "submitted_at": spec.get("submitted_at", ""),
               "state": (_read(jd / "status.json") or {}).get("state", "?")}
        res, met = _read(out / "results.json"), _read(out / "metrics.json")
        if res:
            row.update(kind="backtest", games=res["games_compared"], range=res["date_range"], contenders=len(res["metrics"]), **verdict(res))
        elif met:
            h = met.get("holdout") or {}
            row.update(kind="train", games=met.get("holdout_games_compared"), range=met.get("holdout_range"),
                       contenders=len(h), train_rows=met["train_rows"], verdict="holdout only: run a backtest for confidence intervals")
        else:
            continue  # plain job, not an experiment
        rows.append(row)
    rows.sort(key=lambda r: r["submitted_at"], reverse=True)
    return rows[:limit]


def render_runs(rows: list[dict]) -> str:
    if not rows:
        return "no backtest or training runs yet"
    L = ["Runs use different games; numbers are only comparable within a run.", "",
         "| job | kind | name | games | dates | best (log loss) | verdict |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        dates = f"{r['range'][0]}..{r['range'][1]}" if r.get("range") else "-"
        best = f"{r['best']} ({r['log_loss']:.4f})" if r["kind"] == "backtest" else f"{r['contenders']} scored"
        L.append(f"| {r['job']} | {r['kind']} | {r['name']} | {r['games']} | {dates} | {best} | {r['verdict']} |")
    return "\n".join(L)


# --- model cards -----------------------------------------------------------------------------------

def backtests_including(runner: LocalRunner, sha256: str) -> list[dict]:
    """Backtest runs whose spec used a model with this exact artifact hash."""
    hits = []
    for r in list_runs(runner, limit=10_000):
        if r["kind"] != "backtest":
            continue
        spec = _read(runner.jobs / r["job"] / "code" / "spec.json") or {}
        names = [m["name"] for m in spec.get("models", []) if m.get("sha256") == sha256]
        if names:
            res = _read(runner.jobs / r["job"] / "code" / "outputs" / "results.json")
            for n in names:
                hits.append({**r, "as": n, "metrics": res["metrics"].get(n), "ci": res["bootstrap"]["accuracy_ci"].get(n),
                             "info": res.get("info", {}).get(n, {})})
    return hits


def card(reg: Registry, runner: LocalRunner | None, project: str, model: str, version: str) -> str:
    m = reg.meta(project, model, version)
    chain, cur = [version], m.parent
    while cur and len(chain) < 20:
        chain.append(cur)
        cur = reg.meta(project, model, cur).parent
    live = [c for c, v in reg.channels(project, model).items() if v == version]
    every = reg.deployments(project, model)
    deploys = [d for d in every if d.get("version") == version or d.get("undid_version") == version]
    hits = backtests_including(runner, m.sha256) if runner is not None else []
    inspected = next((h["info"]["n_features"] for h in hits if h["info"].get("n_features")), None)
    L = [f"# {project}/{model}/{version}", "",
         f"- created {m.created_at} · sha256 `{m.sha256[:16]}…` · source `{m.source or 'local file'}`",
         f"- lineage: {' <- '.join(chain)}",
         f"- code: git `{(m.git_sha or 'n/a')[:10]}`{' (dirty)' if m.git_dirty else ''} · lock `{(m.lock_hash or 'none')[:12]}` · data `{(m.data_hash or 'n/a')[:12]}`",
         f"- channels pointing here: {', '.join(live) or 'none'}",
         "- features: " + (f"{len(m.features)} ({', '.join(m.features[:8])}{', …' if len(m.features) > 8 else ''})" if m.features
                          else (f"{inspected} (found by inspecting the artifact in a backtest; not recorded at registration)" if inspected
                                else "not recorded"))]
    if deploys:
        L.append("- deployment history (a swap to a scratch object is a rehearsal, not a live deploy):")
        for d in deploys:
            what = d.get("object_name") or d.get("service") or "?"
            if d.get("target") == "cloudfn-model-swap":
                undone = any(x.get("target") == "cloudfn-rollback" and x.get("object_name") == d.get("object_name") and x["ts"] > d["ts"] for x in every)
                L.append(f"  - {d['ts'][:16]} swap -> `{what}`{' (since rolled back)' if undone else ''}")
            elif d.get("target") == "cloudfn-rollback":
                L.append(f"  - {d['ts'][:16]} rollback of `{what}` (undid {d.get('undid_version')})")
            else:
                L.append(f"  - {d.get('ts', '')[:16]} {d.get('target')} `{what}`")
    else:
        L.append("- deployment history: none")
    hold = (m.metrics or {}).get("holdout")
    if hold:
        L += ["", f"## Training holdout ({m.metrics.get('holdout_games')} games after {m.metrics.get('train_end')}, never seen in training)", "",
              "| contender | accuracy | log loss | auc |", "|---|---|---|---|"]
        L += [f"| {k} | {v['accuracy']:.1%} | {v['log_loss']:.4f} | {v['auc']:.3f} |" for k, v in sorted(hold.items(), key=lambda kv: kv[1]["log_loss"])]
    if runner is not None:
        if hits:
            L += ["", "## Backtests that included this exact artifact", "", "| job | games | dates | accuracy (95% CI) | log loss | that run's verdict |", "|---|---|---|---|---|---|"]
            for h in hits:
                lo, hi = h["ci"]
                L.append(f"| {h['job']} | {h['games']} | {h['range'][0]}..{h['range'][1]} | {h['metrics']['accuracy']:.1%} ({lo:.1%}-{hi:.1%}) | "
                         f"{h['metrics']['log_loss']:.4f} | {h['verdict']} |")
        else:
            L += ["", "No backtest has included this artifact yet."]
    L += ["", "_Generated from the registry and job outputs. Numbers from different runs use different games and are not directly comparable._"]
    return "\n".join(L) + "\n"


def save_card(reg: Registry, runner, project: str, model: str, version: str) -> Path:
    text = card(reg, runner, project, model, version)
    out = reg.version_dir(project, model, version).parent / "cards"
    out.mkdir(exist_ok=True)
    p = out / f"{version}.md"
    p.write_text(text)
    return p
