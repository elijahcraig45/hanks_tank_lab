"""Standalone training worker. Runs as a job in the pinned environment (needs backtest_worker.py beside it).

Trains one model on history + data up to `train_end`, scores the untouched games after it (an honest holdout,
next to any probability columns and the home-field prior), and writes:
  outputs/model.pkl      V10-style payload {model, features, fill_values, version, model_name, trained_at, metrics, ...}
  outputs/metrics.json   train/holdout sizes, feature list, holdout metrics per contender
Register the result with:  mllab register-job JOB PROJECT MODEL VERSION --parent VERSION
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_worker as BW


def run(spec: dict, out_dir: Path) -> dict:
    df = pd.read_parquet(spec["data"]) if spec["data"].endswith(".parquet") else pd.read_csv(spec["data"])
    label, date, gid = spec.get("label", "home_win"), spec.get("date", "game_date"), spec.get("id", "game_pk")
    df = df.dropna(subset=[label]).copy()
    df[date] = pd.to_datetime(df[date])
    flt = spec.get("filter")
    if flt:  # e.g. one division, applied after features were built on the shared frame
        if flt["column"] not in df.columns:
            raise RuntimeError(f"filter column {flt['column']!r} not in data")
        df = df[df[flt["column"]].isin(flt["in"] if "in" in flt else [flt["equals"]])]
        if df.empty:
            raise RuntimeError(f"filter {flt} matched no rows")
    if spec.get("start"):
        df = df[df[date] >= pd.Timestamp(spec["start"])]
    df = df.sort_values([date, gid]).reset_index(drop=True)
    if isinstance(spec.get("features"), dict):  # {all_numeric: true, exclude: [...], exclude_pattern: regex}, same as backtests
        feats = BW.resolve_features({"name": spec["name"], "features": spec["features"]}, df, {label, date, gid}, {})
    elif spec.get("features"):
        feats = list(spec["features"])
    else:  # take the feature list from a registry model (its pickle is hash-verified by the loader)
        _, info = BW.load_predictor(spec["features_from"])
        feats = list(info["features"])
    BW.check_leak(df, feats, label)

    cut = pd.Timestamp(spec["train_end"]) if spec.get("train_end") else df[date].max()
    train, hold = df[df[date] <= cut], df[df[date] > cut]
    Xh, yh, hist_missing = df.iloc[:0][feats].astype(float), np.array([], dtype=int), []
    if spec.get("history"):
        (Xh, yh), hist_missing = BW.load_history(spec, feats, df[date].min(), label, date)
    Xtr = pd.concat([Xh, train[feats].astype(float)])
    ytr = np.concatenate([yh, train[label].to_numpy().astype(int)])
    if len(ytr) < 200 or len(set(ytr)) < 2:
        raise RuntimeError(f"only {len(ytr)} training rows: not enough to fit")
    impute = spec.get("impute", "median")
    if impute not in ("median", "zero"):
        raise RuntimeError(f"impute must be median or zero, got {impute!r}")
    med = Xtr.median() if impute == "median" else pd.Series(0.0, index=Xtr.columns)  # zero: the football trainers' way
    model = BW._make_model(spec.get("family", "xgboost"), spec.get("params", {}))
    model.fit(Xtr.fillna(med).fillna(0.0), ytr)

    metrics = {"train_rows": int(len(ytr)), "history_rows": int(len(yh)), "history_missing_features": hist_missing,
               "train_end": str(cut.date()), "holdout_rows": int(len(hold)), "features": feats,
               "family": spec.get("family", "xgboost"), "params": spec.get("params", {}), "impute": impute,
               "filter": flt, "holdout": {}}
    if len(hold) >= 30:
        yhold = hold[label].to_numpy().astype(int)
        p = model.predict_proba(hold[feats].astype(float).fillna(med).fillna(0.0))[:, 1]
        contenders = {spec["name"]: p, "home_prior": np.full(len(hold), float(ytr.mean()))}
        for c in spec.get("columns", []):
            contenders[c["name"]] = hold[c["column"]].to_numpy(dtype=float)
        ok = np.all([np.isfinite(v) for v in contenders.values()], axis=0)
        tiers = spec.get("tiers", {"high": 0.64, "medium": 0.57})
        metrics["holdout"] = {k: BW.summarize(v[ok], yhold[ok], tiers, spec.get("thresholds", [0.5, 0.55, 0.6, 0.64]))
                              for k, v in contenders.items()}
        metrics["holdout_games_compared"] = int(ok.sum())
        metrics["holdout_range"] = [str(hold[date].min().date()), str(hold[date].max().date())]
    payload = {"model": model, "features": feats, "fill_values": {k: float(v) for k, v in med.items() if np.isfinite(v)},
               "version": spec["name"], "model_name": f"mllab_{spec.get('family', 'xgboost')}",
               "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "metrics": {"train_rows": metrics["train_rows"], "train_end": metrics["train_end"]},
               "params": spec.get("params", {}),
               # the artifact says which feature path it needs, so production never has to guess from a label (only when declared)
               **({"feature_set": spec["feature_set"]} if spec.get("feature_set") else {})}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.pkl").write_bytes(pickle.dumps(payload))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return metrics


def render(m: dict, name: str) -> str:
    L = [f"# Trained {name}: {m['train_rows']} rows (of which {m['history_rows']} history), train_end {m['train_end']}"]
    if not m["holdout"]:
        return "\n".join(L + ["No holdout (train_end at or after the last game)."]) + "\n"
    L += [f"Holdout: {m['holdout_games_compared']} games, {m['holdout_range'][0]} to {m['holdout_range'][1]} (never seen in training)", "",
          "| contender | accuracy | log loss | brier | auc |", "|---|---|---|---|---|"]
    for k, v in sorted(m["holdout"].items(), key=lambda kv: kv[1]["log_loss"]):
        L.append(f"| {k} | {v['accuracy']:.1%} | {v['log_loss']:.4f} | {v['brier']:.4f} | {v['auc']:.3f} |")
    L.append("\nOne holdout is a noisy read; run a backtest for confidence intervals before trusting any difference.")
    return "\n".join(L) + "\n"


def main(argv: list[str]) -> int:
    spec = json.loads(Path(argv[1]).read_text())
    out = Path(spec.get("out_dir", "outputs"))
    m = run(spec, out)
    (out / "report.md").write_text(render(m, spec["name"]))
    print((out / "report.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
