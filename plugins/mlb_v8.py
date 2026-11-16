"""MLB V8-family ensemble adapter for mllab backtests.

Reproduces predict_today_games.V8EnsemblePredictor (weighted blend of CatBoost x2, LightGBM and MLP sub-models,
each with its own feature set and optional scaler, then an optional 2026 calibrator) plus the NaN handling
predict_game applies first (NaN -> the column's stored fill value, default 0.0).

Use in a backtest spec (path is relative to the spec file):
  models:
    - {name: v8, predictor: ../plugins/mlb_v8.py:load, version: {project: hankstank, model: game_outcome, version: v8}}

It REFUSES to run when the data lacks more than max_missing_frac of the bundle's features. Production would fill
missing columns with 0.0; in a backtest that silently produces garbage, so this fails loudly instead.
"""
import pickle

import numpy as np
import pandas as pd


def load(path, max_missing_frac=0.05):
    with open(path, "rb") as f:
        bundle = pickle.load(f)  # noqa: S301 - registry-verified by the worker before this is called
    models, weights = bundle["models"], bundle["weights"]
    scalers, fsets = bundle.get("scalers", {}), bundle["feature_sets"]
    fill, calibrator = bundle.get("fill_values", {}), bundle.get("calibrator")
    cat_cols = fsets.get("cat_tuned", [])
    cat_names = [cat_cols[i] for i in bundle.get("cat_feature_indices", []) if i < len(cat_cols)]
    all_feats = sorted({c for cols in fsets.values() for c in cols})
    info = {"detail": "V8 ensemble", "weights": weights, "sub_models": sorted(models), "n_features": len(all_feats),
            "calibrated": calibrator is not None, "calibrator_type": bundle.get("calibrator_type")}

    def predict(df):
        missing = [c for c in all_feats if c not in df.columns]
        info["missing_frac"] = len(missing) / max(len(all_feats), 1)
        info["missing_columns"] = missing[:25]
        if info["missing_frac"] > max_missing_frac:
            raise RuntimeError(f"V8 plugin: data lacks {len(missing)}/{len(all_feats)} of the bundle's features "
                               f"({info['missing_frac']:.0%}, limit {max_missing_frac:.0%}); first missing: {missing[:8]}. "
                               "Join the V8 feature table into the data query.")
        X = df.copy()
        for col in X.columns:  # predict_game: impute NaN from the stored fill values
            if X[col].isna().any():
                X[col] = X[col].fillna(fill.get(col, 0.0))
        total = sum(weights.values()) or 1.0
        blended = np.zeros(len(X))
        for name, model in models.items():
            w = weights.get(name, 0.0) / total
            if w == 0:
                continue
            sub = pd.DataFrame(index=X.index)
            for col in fsets[name]:
                sub[col] = X[col] if col in X.columns else fill.get(col, 0.0)
            sub = sub.fillna(0.0)
            if type(model).__module__.startswith("catboost"):
                for col in [c for c in cat_names if c in sub.columns]:
                    sub[col] = sub[col].astype(int)
                p = model.predict_proba(sub, ntree_start=0)[:, 1]
            else:
                scaler = scalers.get(name)
                p = model.predict_proba(scaler.transform(sub.values) if scaler else sub.values)[:, 1]
            blended += w * p
        if calibrator is not None:
            blended = calibrator.predict(blended)
        return blended

    predict.info = info
    return predict
