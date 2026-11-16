#!/usr/bin/env python3
"""Run the repo's own train_nfl_models.py unmodified, with the EPA cache seeded from the BigQuery dump.

Purpose: a cross-check. If the lab's season-by-season walk-forward (specs/nfl_walkforward.yaml) reproduces the numbers this prints,
the lab's engine is doing what the repo's trainer does. Run from the job's code dir (layout mirrors the repo: src/nfl/...).
"""
import runpy
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "src/nfl")
from config import RAW_CACHE  # noqa: E402  (resolves under this job's own directory because of the src/nfl layout)

RAW_CACHE.mkdir(parents=True, exist_ok=True)
shutil.copy("../inputs/team_week_epa.parquet", RAW_CACHE / "team_week_epa.parquet")
print(f"EPA cache seeded at {RAW_CACHE}", flush=True)
sys.argv = ["train_nfl_models.py", "--walk-forward", "--variant", "pure", "--variant", "market", "--model", "both"]
runpy.run_path("src/nfl/train_nfl_models.py", run_name="__main__")
