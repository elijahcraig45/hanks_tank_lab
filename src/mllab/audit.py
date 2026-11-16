"""Append-only deploy audit log (JSONL). One line per action; never rewritten."""
from __future__ import annotations

import getpass
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def log_path() -> Path:
    return Path(os.environ.get("MLLAB_AUDIT", "/mnt/appdata/mllab/audit/deploys.log"))


def record(action: str, *, path: Path | None = None, **fields) -> dict:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "actor": getpass.getuser(),
        "action": action,
        **fields,
    }
    p = path or log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # O_APPEND makes each single-line write atomic enough for one writer at a time.
    fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
    try:
        os.write(fd, (json.dumps(entry, sort_keys=True) + "\n").encode())
    finally:
        os.close(fd)
    return entry


def read(path: Path | None = None) -> list[dict]:
    p = path or log_path()
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
