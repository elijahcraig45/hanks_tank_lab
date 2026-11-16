"""Standalone backtest worker. Runs as a job inside the MODEL's pinned environment (numpy, pandas, sklearn,
xgboost; no mllab import) and writes outputs/{results.json,predictions.parquet,report.md}.

  python backtest_worker.py spec.json

Contenders, all scored on the SAME games so numbers are comparable:
  models    registry pickles (hash verified before unpickling; dict payloads, bare estimators, or a plugin)
  retrain   models refit walk-forward with given parameters (each param set is its own contender)
  columns   probability columns already in the data (e.g. prod's served probability, Elo)
  baseline  home-field prior (expanding home win rate)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:  # registers the 'dbdate' dtype so parquet exported from BigQuery loads
    import db_dtypes  # noqa: F401
except ImportError:
    pass

LEAKY = {"home_score", "away_score", "home_won", "home_win", "winner", "prod_prob", "predicted_at", "final_score"}
EPS = 1e-15


# --- loading ---------------------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _plugin(ref: str):
    path, _, fn = ref.partition(":")
    spec = importlib.util.spec_from_file_location("mllab_plugin", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn or "load")


def _frame(df: pd.DataFrame, features: list[str], fill: dict) -> tuple[pd.DataFrame, list[str]]:
    missing = [c for c in features if c not in df.columns]
    X = df.reindex(columns=features).astype(float)
    if fill:
        X = X.fillna(pd.Series({k: v for k, v in fill.items() if k in X.columns}, dtype=float))
    return X.fillna(0.0), missing


def load_predictor(m: dict):
    """Return (predict(df)->P(home win), info). The pickle is only opened after its hash matches."""
    path = m["pickle"]
    if m.get("sha256") and sha256_file(path) != m["sha256"]:
        raise RuntimeError(f"{m['name']}: sha256 mismatch, refusing to unpickle {path}")
    for d in m.get("code", []):
        sys.path.insert(0, d)
    if m.get("predictor"):
        fn = _plugin(m["predictor"])(path)
        # plugins may expose a mutable .info dict (coverage, weights...); it is shared, so later updates reach the results
        return fn, {"kind": "plugin", "plugin": m["predictor"], "details": getattr(fn, "info", {})}
    with open(path, "rb") as f:
        obj = pickle.load(f)  # noqa: S301 - registry-verified artifact
    info = {"kind": type(obj).__name__}
    if isinstance(obj, dict):
        info["keys"] = sorted(obj)[:20]
        if "model" in obj and "features" in obj:
            feats, fill, scaler, model = list(obj["features"]), obj.get("fill_values") or {}, obj.get("scaler"), obj["model"]
            info.update(kind="dict/model", model=type(model).__name__, n_features=len(feats), scaler=scaler is not None, features=feats)

            def predict(df):
                X, missing = _frame(df, feats, fill)
                info["missing_columns"] = missing
                Z = scaler.transform(X) if scaler is not None else X
                return model.predict_proba(Z)[:, 1]
            return predict, info
        if "models" in obj and "weights" in obj:
            raise RuntimeError(f"{m['name']}: this is a V8-style ensemble; its blend needs a predictor plugin "
                               "(models: [{predictor: 'path.py:load'}]). Keys: " + ", ".join(info["keys"]))
        raise RuntimeError(f"{m['name']}: unrecognised dict payload, keys {info['keys']}")
    feats = getattr(obj, "feature_names_in_", None)
    if feats is None and hasattr(obj, "get_booster"):
        feats = obj.get_booster().feature_names
    if feats is None or not hasattr(obj, "predict_proba"):
        raise RuntimeError(f"{m['name']}: cannot infer features for {type(obj).__name__}; provide a plugin")
    feats = list(feats)
    info.update(n_features=len(feats), features=feats)

    def predict_bare(df):
        X, missing = _frame(df, feats, {})
        info["missing_columns"] = missing
        return obj.predict_proba(X)[:, 1]
    return predict_bare, info


# --- retraining (walk-forward) -----------------------------------------------------------------------

def _make_model(family: str, params: dict):
    if family == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**{"eval_metric": "logloss", "random_state": 42, **params})
    if family == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**{"random_state": 42, "verbose": -1, **params})
    if family == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(**{"max_iter": 1000, **params}))
    raise ValueError(f"unknown family {family!r} (xgboost, lightgbm, logreg)")


def check_leak(df: pd.DataFrame, feats: list[str], label: str) -> None:
    bad = [f for f in feats if f in LEAKY or f == label]
    if bad:
        raise RuntimeError(f"features contain outcome/leaky columns: {bad}")
    y = df[label].astype(float)
    for f in feats:
        if f in df.columns and df[f].nunique() > 1:
            c = abs(np.corrcoef(df[f].fillna(df[f].median()).astype(float), y)[0, 1])
            if c > 0.98:
                raise RuntimeError(f"feature {f} correlates {c:.3f} with the label: probable leak")


def resolve_features(item: dict, df: pd.DataFrame, exclude_always: set, feature_lists: dict) -> list[str]:
    """features: [names] | {all_numeric: true, exclude: [names], exclude_pattern: regex} | features_from: <model name>."""
    f = item.get("features")
    if isinstance(f, dict):
        ex = set(f.get("exclude", [])) | exclude_always
        pat = re.compile(f["exclude_pattern"]) if f.get("exclude_pattern") else None  # e.g. "epa|success_rate" drops feature families
        cols = [c for c in df.columns if c not in ex and not (pat and pat.search(c)) and pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            raise RuntimeError(f"retrain {item['name']}: all_numeric selected no columns")
        return cols
    if f:
        return list(f)
    feats = feature_lists.get(item.get("features_from"))
    if not feats:
        raise RuntimeError(f"retrain {item['name']}: give features or features_from")
    return feats


def _blocks(df: pd.DataFrame, date: str, min_train: int, step: int, group: str | None, min_groups: int, has_hist: bool):
    """Yield prediction index arrays. Row mode: consecutive blocks of `step` rows. Group mode: one whole group (e.g. a season)
    at a time, like a season-by-season walk-forward. Either way training is limited to games dated before the block starts."""
    if group:
        col = df[group].to_numpy()
        order = list(pd.unique(col))
        for g in order[min_groups:]:
            yield np.flatnonzero(col == g)
        return
    i = 0 if has_hist and min_train == 0 else min_train
    while i < len(df):
        j = min(i + step, len(df))
        yield np.arange(i, j)
        i = j


def walk_forward(df: pd.DataFrame, feats: list[str], label: str, date: str, family: str, params: dict,
                 min_train: int, step: int, hist: tuple[pd.DataFrame, np.ndarray] | None = None,
                 group: str | None = None, min_groups: int = 0, impute: str = "median", seeds: list[int] | None = None) -> np.ndarray:
    """Expanding window. A block trains ONLY on games dated strictly before the block's first date.

    hist = (features, labels) of older games (all dated before the first evaluated game) used as a warm start,
    so parameter comparisons are not limited to the few games seen so far this season.
    group = column to walk forward by whole groups (e.g. season) instead of `step` rows.
    impute = "median" (training medians, the MLB trainer's way) or "zero" (fillna(0.0), the NFL trainer's way).
    seeds = average the predicted probabilities of one model per seed (xgboost/lightgbm). A single seed can move a season by 1-2 points
    through row/column subsampling alone; averaging measures the recipe, not one random draw. logreg is deterministic and ignores it.
    """
    if impute not in ("median", "zero"):
        raise RuntimeError(f"impute must be median or zero, got {impute!r}")
    check_leak(df, feats, label)
    out = np.full(len(df), np.nan)
    dates = df[date].to_numpy()
    y = df[label].to_numpy().astype(int)
    X = df[feats].astype(float)
    Xh, yh = hist if hist is not None else (X.iloc[:0], y[:0])
    for pred in _blocks(df, date, min_train, step, group, min_groups, hist is not None):
        if len(pred) == 0:
            continue
        seen = np.flatnonzero(dates < dates[pred].min())
        Xtr = pd.concat([Xh, X.iloc[seen]]) if len(Xh) else X.iloc[seen]
        ytr = np.concatenate([yh, y[seen]])
        if len(ytr) >= max(min_train, 1) and len(set(ytr)) == 2:
            med = Xtr.median() if impute == "median" else pd.Series(0.0, index=Xtr.columns)
            Xa, Xp = Xtr.fillna(med).fillna(0.0), X.iloc[pred].fillna(med).fillna(0.0)
            if seeds and family in ("xgboost", "lightgbm"):
                out[pred] = np.mean([_make_model(family, {**params, "random_state": int(sd)}).fit(Xa, ytr).predict_proba(Xp)[:, 1] for sd in seeds], axis=0)
            else:
                out[pred] = _make_model(family, params).fit(Xa, ytr).predict_proba(Xp)[:, 1]
    return out


def _sample(rng, spec):
    """One value from a search-space entry: a list (choice) or {low, high, [log], [type: int]}."""
    if isinstance(spec, list):
        return spec[int(rng.integers(len(spec)))]
    lo, hi = float(spec["low"]), float(spec["high"])
    v = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if spec.get("log") else float(rng.uniform(lo, hi))
    return int(round(v)) if spec.get("type") == "int" else v


def _inner_loss(df, feats, label, date, group, family, params, folds_from, impute, seeds=None) -> float:
    """Mean log loss of a walk-forward over the LAST groups of `df` (all of it is tuning data, never evaluation data)."""
    p = walk_forward(df, feats, label, date, family, params, 0, 0, group=group, min_groups=folds_from, impute=impute, seeds=seeds)
    ok = np.isfinite(p)
    return float(logloss_vec(p[ok], df[label].to_numpy().astype(int)[ok]).mean())


def tune_params(df: pd.DataFrame, feats: list[str], label: str, date: str, group: str, family: str, base_params: dict,
                tune: dict, eval_start_groups: int, impute: str) -> dict:
    """Random search over `tune["space"]` using ONLY the first `eval_start_groups` groups, i.e. strictly before anything that will be
    scored. Candidates are ranked by an inner walk-forward across the last `inner_folds` of those groups. Nothing here ever sees an
    evaluation game, so the outer score of the frozen winner carries no selection bias."""
    order = list(pd.unique(df[group]))
    if eval_start_groups < 3:
        raise RuntimeError("tuning needs at least 3 groups before the evaluation window (min_groups >= 3)")
    inner_folds = min(int(tune.get("inner_folds", 3)), eval_start_groups - 1)
    window = df[df[group].isin(order[:eval_start_groups])].reset_index(drop=True)
    folds_from = eval_start_groups - inner_folds
    rng = np.random.default_rng(int(tune.get("seed", 0)))
    trials = []
    for _ in range(int(tune.get("trials", 20))):
        cand = {k: _sample(rng, v) for k, v in tune["space"].items()}
        trials.append({"params": cand, "inner_log_loss": _inner_loss(window, feats, label, date, group, family, {**base_params, **cand}, folds_from, impute)})
    trials.sort(key=lambda t: t["inner_log_loss"])
    default = _inner_loss(window, feats, label, date, group, family, base_params, folds_from, impute)
    return {"best_params": trials[0]["params"], "best_inner_log_loss": trials[0]["inner_log_loss"], "default_inner_log_loss": default,
            "n_trials": len(trials), "top_trials": trials[:5], "tuning_window": [str(window[date].min().date()), str(window[date].max().date())],
            "inner_folds": inner_folds,
            "note": "The winner's inner score is the best of N and therefore optimistic. Trust only the outer walk-forward numbers, which never touched tuning data."}


def load_history(spec: dict, feats: list[str], first_date, label: str, date: str):
    """Warm-start rows: strictly older than the first evaluated game, labelled, columns aligned to feats."""
    h = spec["history"]
    hd = pd.read_parquet(h["file"]) if h["file"].endswith(".parquet") else pd.read_csv(h["file"])
    hl, hdt = h.get("label", label), h.get("date", date)
    hd[hdt] = pd.to_datetime(hd[hdt])
    hd = hd[hd[hdt] < first_date].dropna(subset=[hl])
    missing = [c for c in feats if c not in hd.columns]
    Xh = hd.reindex(columns=feats).astype(float)
    return (Xh, hd[hl].to_numpy().astype(int)), missing


def home_prior(y: np.ndarray) -> np.ndarray:
    """Expanding home win rate before each game (0.5 to start): the always-home benchmark."""
    prior = np.empty(len(y))
    wins = 0
    for i, v in enumerate(y):
        prior[i] = (wins + 1) / (i + 2)  # Laplace smoothed
        wins += int(v)
    return prior


# --- metrics ---------------------------------------------------------------------------------------

def logloss_vec(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def auc(p: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")


def calibration(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                         "mean_pred": float(p[m].mean()), "actual": float(y[m].mean())})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"ece": float(ece), "bins": rows}


def summarize(p: np.ndarray, y: np.ndarray, tiers: dict, thresholds: list[float]) -> dict:
    correct = ((p > 0.5) == (y == 1)).astype(float)
    conf = np.maximum(p, 1 - p)
    out = {"n": int(len(y)), "accuracy": float(correct.mean()), "log_loss": float(logloss_vec(p, y).mean()),
           "brier": float(((p - y) ** 2).mean()), "auc": auc(p, y), "calibration": calibration(p, y), "thresholds": [], "tiers": {}}
    for t in thresholds:
        m = conf >= t
        out["thresholds"].append({"min_confidence": t, "games": int(m.sum()), "coverage": float(m.mean()),
                                  "accuracy": float(correct[m].mean()) if m.any() else None})
    names = sorted(tiers.items(), key=lambda kv: -kv[1])
    for k, (name, lo) in enumerate(names):
        hi = names[k - 1][1] if k else 1.01
        m = (conf >= lo) & (conf < hi)
        out["tiers"][name] = {"min": lo, "games": int(m.sum()), "accuracy": float(correct[m].mean()) if m.any() else None}
    return out


def bootstrap(P: dict[str, np.ndarray], y: np.ndarray, B: int, seed: int) -> dict:
    """Paired percentile bootstrap: every contender is scored on the same resampled games."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.integers(0, n, size=(B, n))
    acc = {k: ((p > 0.5) == (y == 1)).astype(float)[idx].mean(1) for k, p in P.items()}
    ll = {k: logloss_vec(p, y)[idx].mean(1) for k, p in P.items()}
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    out = {"accuracy_ci": {k: ci(v) for k, v in acc.items()}, "log_loss_ci": {k: ci(v) for k, v in ll.items()}, "paired": {}}
    names = list(P)
    for a in names:
        for b in names:
            if a < b:
                da, dl = acc[a] - acc[b], ll[a] - ll[b]
                out["paired"][f"{a} - {b}"] = {
                    "accuracy_diff": float(da.mean()), "accuracy_diff_ci": ci(da), "p_a_not_better_acc": float((da <= 0).mean()),
                    "log_loss_diff": float(dl.mean()), "log_loss_diff_ci": ci(dl), "p_a_not_better_ll": float((dl >= 0).mean())}
    return out


def parity(p: np.ndarray, ref: np.ndarray) -> dict:
    d = np.abs(p - ref)
    return {"n": int(len(p)), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "corr": float(np.corrcoef(p, ref)[0, 1]) if len(p) > 1 else float("nan"),
            "same_pick_pct": float(((p > 0.5) == (ref > 0.5)).mean() * 100),
            "within_1e-4_pct": float((d < 1e-4).mean() * 100)}


# --- run -------------------------------------------------------------------------------------------

def run(spec: dict, out_dir: Path) -> dict:
    df = pd.read_parquet(spec["data"]) if spec["data"].endswith(".parquet") else pd.read_csv(spec["data"])
    label, date, gid = spec.get("label", "home_win"), spec.get("date", "game_date"), spec.get("id", "game_pk")
    df = df.dropna(subset=[label]).copy()
    df[date] = pd.to_datetime(df[date])
    flt = spec.get("filter")
    if flt:  # keep only some rows (e.g. one division) AFTER features were built, so shared state like an Elo pool is untouched
        col = flt["column"]
        if col not in df.columns:
            raise RuntimeError(f"filter column {col!r} not in data")
        df = df[df[col].isin(flt["in"] if "in" in flt else [flt["equals"]])]
        if df.empty:
            raise RuntimeError(f"filter {flt} matched no rows")
    if spec.get("start"):
        df = df[df[date] >= pd.Timestamp(spec["start"])]
    if spec.get("end"):
        df = df[df[date] <= pd.Timestamp(spec["end"])]
    df = df.sort_values([date, gid]).reset_index(drop=True)
    y = df[label].to_numpy().astype(int)
    if len(df) < 30:
        raise RuntimeError(f"only {len(df)} games in range; nothing meaningful to backtest")

    contenders, info, feature_lists = {}, {}, {}
    for m in spec.get("models", []):
        predict, info[m["name"]] = load_predictor(m)
        feature_lists[m["name"]] = info[m["name"]].pop("features", None)
        contenders[m["name"]] = np.asarray(predict(df), dtype=float)
    for r in spec.get("retrain", []):
        feats = resolve_features(r, df, {label, date, gid}, feature_lists)
        wf = r.get("walk_forward", {})  # group mode: min_groups is the only threshold (min_train defaults to 0)
        hist, hist_missing = (None, [])
        if spec.get("history") and r.get("use_history", True):
            hist, hist_missing = load_history(spec, feats, df[date].min(), label, date)
        params = dict(r.get("params", {}))
        tuned = None
        if r.get("tune"):
            if not wf.get("group"):
                raise RuntimeError(f"retrain {r['name']}: tune needs walk_forward.group (it tunes on whole groups before min_groups)")
            tuned = tune_params(df, feats, label, date, wf["group"], r.get("family", "xgboost"), params, r["tune"], int(wf.get("min_groups", 0)),
                                r.get("impute", "median"))
            params = {**params, **tuned["best_params"]}
        contenders[r["name"]] = walk_forward(df, feats, label, date, r.get("family", "xgboost"), params,
                                             int(wf.get("min_train", 0 if (hist or wf.get("group")) else 500)), int(wf.get("step", 100)), hist,
                                             group=wf.get("group"), min_groups=int(wf.get("min_groups", 0)),
                                             impute=r.get("impute", "median"), seeds=r.get("seeds"))
        info[r["name"]] = {"kind": f"retrain/{r.get('family', 'xgboost')}", "params": params, "n_features": len(feats),
                           "impute": r.get("impute", "median"), "walk_forward": wf, "seeds": r.get("seeds"), **({"tuned": tuned} if tuned else {}),
                           **({"history_rows": len(hist[1]), "history_missing_features": hist_missing} if hist else {})}
    for c in spec.get("columns", []):
        if c["column"] not in df.columns:
            raise RuntimeError(f"column {c['column']} not in data")
        contenders[c["name"]] = df[c["column"]].to_numpy(dtype=float)
        info[c["name"]] = {"kind": "column", "column": c["column"]}
    if spec.get("baseline", True):
        contenders["home_prior"] = home_prior(y)
        info["home_prior"] = {"kind": "baseline"}

    ok = np.ones(len(df), dtype=bool)  # common games: every contender has a finite prediction
    for p in contenders.values():
        ok &= np.isfinite(p)
    if ok.sum() < 30:
        raise RuntimeError(f"only {int(ok.sum())} games are scored by every contender; lower min_train or shorten the list")
    yc = y[ok]
    P = {k: v[ok] for k, v in contenders.items()}
    tiers = spec.get("tiers", {"high": 0.64, "medium": 0.57})
    ths = spec.get("thresholds", [0.5, 0.55, 0.6, 0.64])
    res = {"games_total": int(len(df)), "games_compared": int(ok.sum()),
           "date_range": [str(df[date][ok].min().date()), str(df[date][ok].max().date())],
           "home_win_rate": float(yc.mean()), "info": info,
           "metrics": {k: summarize(p, yc, tiers, ths) for k, p in P.items()},
           "bootstrap": bootstrap(P, yc, int(spec.get("bootstrap", 2000)), int(spec.get("seed", 42)))}
    ref = next((c for c in spec.get("columns", []) if c.get("reference")), None)
    if ref:
        res["parity"] = {m["name"]: parity(P[m["name"]], P[ref["name"]]) for m in spec.get("models", []) if m.get("parity")}
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = pd.DataFrame({gid: df[gid][ok].to_numpy(), date: df[date][ok].to_numpy(), label: yc, **P})
    preds.to_parquet(out_dir / "predictions.parquet")
    (out_dir / "results.json").write_text(json.dumps(res, indent=2, default=float))
    (out_dir / "report.md").write_text(render(res))
    return res


def render(res: dict) -> str:
    ms = res["metrics"]
    bo = res["bootstrap"]
    order = sorted(ms, key=lambda k: ms[k]["log_loss"])
    L = [f"# Backtest: {res['games_compared']} games, {res['date_range'][0]} to {res['date_range'][1]}",
         f"Home win rate {res['home_win_rate']:.1%}. Every contender scored on the same games. Sorted by log loss (lower is better).", "",
         "| contender | accuracy | 95% CI | log loss | brier | auc | ECE |", "|---|---|---|---|---|---|---|"]
    for k in order:
        m = ms[k]; lo, hi = bo["accuracy_ci"][k]
        L.append(f"| {k} | {m['accuracy']:.1%} | {lo:.1%}-{hi:.1%} | {m['log_loss']:.4f} | {m['brier']:.4f} | {m['auc']:.3f} | {m['calibration']['ece']:.3f} |")
    L += ["", "## Accuracy at minimum confidence (games / accuracy)", "", "| contender | " + " | ".join(f">={t['min_confidence']:.2f}" for t in ms[order[0]]["thresholds"]) + " |",
          "|---|" + "---|" * len(ms[order[0]]["thresholds"])]
    for k in order:
        L.append(f"| {k} | " + " | ".join(f"{t['games']} / {t['accuracy']:.1%}" if t["accuracy"] is not None else f"{t['games']} / -" for t in ms[k]["thresholds"]) + " |")
    if bo["paired"]:
        L += ["", "## Paired differences (same games; A - B)", "", "| A - B | accuracy diff | 95% CI | log loss diff | 95% CI |", "|---|---|---|---|---|"]
        for k, d in bo["paired"].items():
            L.append(f"| {k} | {d['accuracy_diff']:+.2%} | {d['accuracy_diff_ci'][0]:+.2%}..{d['accuracy_diff_ci'][1]:+.2%} | "
                     f"{d['log_loss_diff']:+.4f} | {d['log_loss_diff_ci'][0]:+.4f}..{d['log_loss_diff_ci'][1]:+.4f} |")
        L.append("\nA difference whose CI spans 0 is not distinguishable from noise at this sample size.")
    if res.get("parity"):
        L += ["", "## Parity with the served probability", "", "| model | games | max abs diff | mean abs diff | same pick | within 1e-4 |", "|---|---|---|---|---|---|"]
        for k, p in res["parity"].items():
            L.append(f"| {k} | {p['n']} | {p['max_abs_diff']:.2e} | {p['mean_abs_diff']:.2e} | {p['same_pick_pct']:.1f}% | {p['within_1e-4_pct']:.1f}% |")
    return "\n".join(L) + "\n"


def main(argv: list[str]) -> int:
    spec = json.loads(Path(argv[1]).read_text())
    res = run(spec, Path(spec.get("out_dir", "outputs")))
    print((Path(spec.get("out_dir", "outputs")) / "report.md").read_text())
    print(f"artifact info: {json.dumps(res['info'], default=str)[:1500]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
