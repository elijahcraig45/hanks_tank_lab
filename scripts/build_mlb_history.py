#!/usr/bin/env python3
"""Build the MLB warm-start history for backtest retrains: V8-era features 2015-2025 plus the lineup-matchup
features, joined on game_pk. Keeps the label as home_won (use `history: {label: home_won}` in a backtest spec).

  python scripts/build_mlb_history.py --dir ~/Documents/personalDev/mlb/hanks_tank_ml/data/training --out mlb_history_2015_2025.parquet
"""
import argparse
from pathlib import Path

import db_dtypes  # noqa: F401  (registers 'dbdate' so BigQuery-exported parquet loads)
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir).expanduser()
    hist = pd.concat([pd.read_parquet(d / "train_v8_2015_2024.parquet"), pd.read_parquet(d / "val_v8_2025.parquet")], ignore_index=True)
    mf = pd.read_parquet(d / "matchup_features_historical.parquet").drop_duplicates("game_pk")
    extra = ["game_pk"] + [c for c in mf.columns if c not in hist.columns]
    out = hist.drop_duplicates("game_pk").merge(mf[extra], on="game_pk", how="left")
    out["game_date"] = pd.to_datetime(out["game_date"])
    out = out.dropna(subset=["home_won"]).sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    out.to_parquet(a.out)
    print(f"{len(out)} games, {out.game_date.min().date()} to {out.game_date.max().date()}, {out.shape[1]} columns "
          f"({len(extra) - 1} from the matchup file) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
