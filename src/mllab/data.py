"""The lab's data directory: list files and promote a finished job's output into it (so backtests/training can read it)."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from .jobs import LocalRunner

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}\.(parquet|csv)$")


class DataError(Exception):
    pass


def list_files(data_root: Path) -> list[tuple[str, int]]:
    if not data_root.is_dir():
        return []
    return sorted((p.name, p.stat().st_size) for p in data_root.iterdir() if p.is_file() and not p.name.startswith("."))


def promote(runner: LocalRunner, job_id: str, file: str, name: str, *, replace: bool = False) -> Path:
    if not NAME.match(name):
        raise DataError(f"name must look like something.parquet or something.csv, got {name!r}")
    st = runner.status(job_id)
    if st["state"] != "finished":
        raise DataError(f"job {job_id} is {st['state']}; only a finished job's output can be promoted")
    out = (runner.jobs / job_id / "code" / "outputs").resolve()
    src = (out / file).resolve()
    if out not in src.parents or not src.is_file():  # no ../ and no symlink escapes: only real files inside outputs/
        raise DataError(f"{file!r} is not a file inside the job's outputs directory")
    dest = runner.data_root / name
    if dest.exists() and not replace:
        raise DataError(f"{name} already exists; pass --replace to overwrite it")
    runner.data_root.mkdir(parents=True, exist_ok=True)
    tmp = runner.data_root / f".{name}.tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)  # atomic: a backtest never sees a half-copied file
    return dest
