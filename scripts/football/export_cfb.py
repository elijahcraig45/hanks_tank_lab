#!/usr/bin/env python3
"""Export the CFB feature frame that hanks_tank_ml's own pipeline builds (pipeline.build), plus the repo's own per-division baselines.

Runs as a job (layout mirrors the repo: src/nfl and src/cfb side by side, because the CFB pipeline reuses the NFL feature builder and
its config roots its cache two levels up). Games are fetched from ESPN through the repo's espn_data.fetch_history.
Season 2025 is the repo's touch-once holdout: excluded unless --include-holdout is passed on purpose. Features are causal (each game
sees only earlier games), so dropping later seasons after the build cannot leak anything into earlier rows.
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="src", help="dir holding nfl/ and cfb/")
    ap.add_argument("--first-season", type=int)
    ap.add_argument("--through-season", type=int)
    ap.add_argument("--include-holdout", action="store_true")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    sys.path.insert(0, f"{a.src}/nfl")
    sys.path.insert(0, f"{a.src}/cfb")
    import cfb_config  # noqa: E402
    import espn_data  # noqa: E402
    import pipeline  # noqa: E402

    first = a.first_season or cfb_config.FIRST_SEASON
    last = a.through_season if a.through_season is not None else cfb_config.HOLDOUT_SEASON - 1
    if last >= cfb_config.HOLDOUT_SEASON and not a.include_holdout:
        ap.error(f"season {cfb_config.HOLDOUT_SEASON} is the touch-once holdout; refusing --through-season {last} without --include-holdout")

    games = espn_data.fetch_history(first_season=first, last_season=last)
    feats = pipeline.build(games)
    feats = feats[feats["season"] <= last].reset_index(drop=True)
    cols = pipeline.cfb_feature_columns(feats)
    baselines = {}
    for div in sorted(feats["division"].dropna().unique()):
        baselines[div] = {int(s): pipeline.baselines(feats, div, int(s)) for s in sorted(feats.loc[feats["division"] == div, "season"].unique())}

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out / "cfb_features.parquet")
    (out / "columns.json").write_text(json.dumps({
        "features": cols, "rows": int(len(feats)), "seasons": [int(feats.season.min()), int(feats.season.max())],
        "divisions": {str(k): int(v) for k, v in feats["division"].value_counts().items()},
        "holdout_included": bool(last >= cfb_config.HOLDOUT_SEASON), "all_columns": list(feats.columns)}, indent=2))
    (out / "baselines.json").write_text(json.dumps(baselines, indent=2, default=float))
    print(f"{len(feats)} games, seasons {feats.season.min()}-{feats.season.max()}, divisions {dict(feats['division'].value_counts())}, "
          f"{len(cols)} features, holdout included={last >= cfb_config.HOLDOUT_SEASON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
