#!/usr/bin/env python3
"""Export the NFL feature frame that hanks_tank_ml's own trainer builds, as a parquet the lab can backtest and train on.

Runs as a job in the environment built from src/nfl/requirements.txt, with the repo's src/nfl/*.py beside it.
Games come from nflverse (via the repo's data.py); EPA comes from a parquet dump of nfl_historical.team_week_epa, which is
what production reads (predict_nfl.load_epa). The repo's rule is respected: the 2025 season is a TOUCH-ONCE holdout, so it is
excluded unless you pass --include-holdout on purpose.

  python export_nfl.py --src . --epa ../inputs/team_week_epa.parquet --out outputs
  python export_nfl.py --src . --no-epa --out outputs          # the repo's own EPA ablation
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.environ.get("NFL_SRC", "."), help="dir holding config.py, data.py, features.py")
    ap.add_argument("--epa", help="parquet dump of nfl_historical.team_week_epa")
    ap.add_argument("--no-epa", action="store_true")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--through-season", type=int)
    ap.add_argument("--include-holdout", action="store_true", help="include the touch-once holdout season. Only do this once, on purpose.")
    a = ap.parse_args()
    if bool(a.epa) == a.no_epa:
        ap.error("give exactly one of --epa FILE or --no-epa")
    sys.path.insert(0, a.src)
    from config import HOLDOUT_SEASON  # noqa: E402
    from data import completed_games  # noqa: E402
    from features import build_features, feature_columns  # noqa: E402

    last = a.through_season if a.through_season is not None else HOLDOUT_SEASON - 1
    if last >= HOLDOUT_SEASON and not a.include_holdout:
        ap.error(f"season {HOLDOUT_SEASON} is the touch-once holdout; refusing --through-season {last} without --include-holdout")

    games = completed_games()
    epa = None
    if a.epa:
        epa = pd.read_parquet(a.epa)
        games = games[games["season"].isin(sorted(epa["season"].unique()))]  # same restriction as the repo's trainer
    feats = build_features(games, epa=epa)
    feats = feats[feats["season"] <= last].reset_index(drop=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out / "nfl_features.parquet")
    cols = {"pure": feature_columns(feats, include_market=False), "market": feature_columns(feats, include_market=True),
            "rows": int(len(feats)), "seasons": [int(feats.season.min()), int(feats.season.max())], "with_epa": epa is not None,
            "holdout_included": bool(last >= HOLDOUT_SEASON), "all_columns": list(feats.columns)}
    (out / "columns.json").write_text(json.dumps(cols, indent=2))
    print(f"{len(feats)} games, seasons {cols['seasons'][0]}-{cols['seasons'][1]}, {len(cols['pure'])} pure features, EPA={epa is not None}, "
          f"holdout included={cols['holdout_included']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
