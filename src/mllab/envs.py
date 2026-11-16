"""Cached per-lock environments (<root>/envs/<lock hash>): list them and prune the ones that are provably idle.

Each distinct requirements lock builds its own venv the first time a job needs it (5 GB across the ones so far). A job stamps
<env>/.last_used every time it reuses one. Pruning only removes environments that (a) have that stamp, (b) were last used more than N days
ago and (c) are not referenced by any queued or running job. Environments without a stamp (built before tracking existed) are reported
as untracked and are never deleted unless you pass include_untracked. A pruned environment is simply rebuilt from its lock on next use.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from .jobs import LocalRunner


class EnvError(Exception):
    pass


def _size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def _in_use(runner: LocalRunner, name: str) -> bool:
    for jid in runner.active_ids():
        try:
            if name in (runner.jobs / jid / "run.sh").read_text():
                return True
        except OSError:
            return True  # cannot tell, so assume it is in use
    return False


def list_envs(runner: LocalRunner, *, now: float | None = None) -> list[dict]:
    now = now or time.time()
    if not runner.envs.is_dir():
        return []
    rows = []
    for d in sorted(p for p in runner.envs.iterdir() if p.is_dir()):
        stamp = d / ".last_used"
        rows.append({"name": d.name, "size_bytes": _size(d), "tracked": stamp.exists(),
                     "idle_days": round((now - stamp.stat().st_mtime) / 86400, 1) if stamp.exists() else None,
                     "in_use": _in_use(runner, d.name)})
    return rows


def prune(runner: LocalRunner, older_than_days: float, *, dry_run: bool = True, include_untracked: bool = False,
          now: float | None = None) -> list[dict]:
    if older_than_days < 1:
        raise EnvError("older_than_days must be at least 1: pruning something used today is never what you want")
    victims = []
    for e in list_envs(runner, now=now):
        if e["in_use"]:
            continue
        if e["tracked"] and e["idle_days"] > older_than_days:
            victims.append(e)
        elif not e["tracked"] and include_untracked:
            victims.append(e)
    if not dry_run:
        for e in victims:
            target = (runner.envs / e["name"]).resolve()
            if target.parent != runner.envs.resolve():  # never follow a symlink or odd name out of the envs directory
                raise EnvError(f"refusing to delete {target}")
            shutil.rmtree(target)
    return victims


def render(rows: list[dict]) -> str:
    if not rows:
        return "no cached environments"
    L = ["| environment | size | last used | in use |", "|---|---|---|---|"]
    for e in rows:
        used = "untracked (built before tracking)" if not e["tracked"] else f"{e['idle_days']} days ago"
        L.append(f"| {e['name']} | {e['size_bytes'] / 1024**3:.2f} GiB | {used} | {'yes' if e['in_use'] else 'no'} |")
    L.append(f"\ntotal {sum(e['size_bytes'] for e in rows) / 1024**3:.2f} GiB")
    return "\n".join(L)
