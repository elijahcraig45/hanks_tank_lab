import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")
pytest.importorskip("pyarrow")

from xgboost import XGBClassifier

from mllab import backtest_worker as W

WORKER = Path(W.__file__)


def make_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(5)})
    p = 1 / (1 + np.exp(-(1.2 * df.f0 - 0.8 * df.f1)))
    df["home_win"] = (rng.random(n) < p).astype(int)
    df["game_pk"] = np.arange(n)
    df["game_date"] = pd.Timestamp("2026-04-01") + pd.to_timedelta(np.arange(n) // 5, unit="D")
    return df


@pytest.fixture
def env(tmp_path):
    df = make_data()
    df.to_parquet(tmp_path / "d.parquet")
    feats = [f"f{i}" for i in range(5)]
    model = XGBClassifier(n_estimators=40, max_depth=3, learning_rate=0.1, eval_metric="logloss").fit(df[feats][:300], df.home_win[:300])
    pkl = tmp_path / "model.pkl"
    pkl.write_bytes(pickle.dumps({"model": model, "features": feats, "fill_values": {c: 0.0 for c in feats}, "version": "t"}))
    df["prod_prob"] = model.predict_proba(df[feats])[:, 1]  # "served" probability = exactly this model
    df.to_parquet(tmp_path / "d.parquet")
    return tmp_path, pkl, feats, df


def spec(tmp, pkl, **kw):
    base = {"data": str(tmp / "d.parquet"), "out_dir": str(tmp / "out"), "bootstrap": 300, "start": "2026-05-01",
            "models": [{"name": "m", "pickle": str(pkl), "sha256": W.sha256_file(str(pkl)), "parity": True}]}
    base.update(kw)
    return base


def test_model_beats_home_prior_and_report_written(env):
    tmp, pkl, feats, df = env
    res = W.run(spec(tmp, pkl), tmp / "out")
    assert res["metrics"]["m"]["accuracy"] > res["metrics"]["home_prior"]["accuracy"]
    assert res["metrics"]["m"]["log_loss"] < res["metrics"]["home_prior"]["log_loss"]
    for f in ("results.json", "report.md", "predictions.parquet"):
        assert (tmp / "out" / f).exists()
    assert "| m |" in (tmp / "out" / "report.md").read_text()


def test_sha_mismatch_refuses_to_unpickle(env):
    tmp, pkl, *_ = env
    s = spec(tmp, pkl)
    s["models"][0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        W.run(s, tmp / "out")


def test_parity_with_served_column_is_exact(env):
    tmp, pkl, *_ = env
    s = spec(tmp, pkl, columns=[{"name": "prod", "column": "prod_prob", "reference": True}])
    p = W.run(s, tmp / "out")["parity"]["m"]
    assert p["max_abs_diff"] < 1e-6 and p["same_pick_pct"] == 100.0


def test_all_contenders_share_the_same_games(env):
    tmp, pkl, feats, df = env
    s = spec(tmp, pkl, retrain=[{"name": "rt", "features": feats, "params": {"n_estimators": 20}, "walk_forward": {"min_train": 150, "step": 50}}])
    res = W.run(s, tmp / "out")
    n = {v["n"] for v in res["metrics"].values()}
    assert len(n) == 1 and res["games_compared"] < res["games_total"]  # retrain needs history, so all are cut to the same set


def test_walk_forward_cannot_see_the_future(env):
    """Flip every label AFTER a cutoff date; predictions at or before the cutoff block must not move."""
    tmp, pkl, feats, df = env
    d = df.copy()
    d["game_date"] = pd.to_datetime(d["game_date"])
    base = W.walk_forward(d, feats, "home_win", "game_date", "xgboost", {"n_estimators": 20}, 150, 50)
    cutoff = d.game_date.iloc[300]
    poisoned = d.copy()
    poisoned.loc[poisoned.game_date > cutoff, "home_win"] = 1 - poisoned.home_win
    alt = W.walk_forward(poisoned, feats, "home_win", "game_date", "xgboost", {"n_estimators": 20}, 150, 50)
    seen = np.flatnonzero((d.game_date <= cutoff).to_numpy() & np.isfinite(base))
    assert len(seen) > 100
    assert np.allclose(base[seen], alt[seen])  # identical: later labels never influenced earlier predictions
    assert not np.allclose(base[300:], alt[300:], equal_nan=True)  # and the poison DID matter later, so the test can detect leaks


def test_retrain_refuses_leaky_features(env):
    tmp, pkl, feats, df = env
    s = spec(tmp, pkl, retrain=[{"name": "bad", "features": feats + ["home_score"]}])
    d = pd.read_parquet(tmp / "d.parquet"); d["home_score"] = d.home_win * 3; d.to_parquet(tmp / "d.parquet")
    with pytest.raises(RuntimeError, match="leaky"):
        W.run(s, tmp / "out")
    d["near_copy"] = d.home_win + np.random.default_rng(1).normal(0, 0.01, len(d)); d.to_parquet(tmp / "d.parquet")
    with pytest.raises(RuntimeError, match="probable leak"):
        W.run(spec(tmp, pkl, retrain=[{"name": "bad2", "features": feats + ["near_copy"]}]), tmp / "out")


def test_features_from_model_and_param_sweep(env):
    tmp, pkl, feats, df = env
    wf = {"min_train": 150, "step": 100}
    s = spec(tmp, pkl, retrain=[
        {"name": "d2", "features_from": "m", "params": {"max_depth": 2, "n_estimators": 20}, "walk_forward": wf},
        {"name": "d6", "features_from": "m", "params": {"max_depth": 6, "n_estimators": 20}, "walk_forward": wf},
        {"name": "lr", "family": "logreg", "features_from": "m", "walk_forward": wf}])
    res = W.run(s, tmp / "out")
    assert {"m", "d2", "d6", "lr", "home_prior"} <= set(res["metrics"])
    assert "d2 - d6" in res["bootstrap"]["paired"] or "d6 - d2" in res["bootstrap"]["paired"]


def test_v8_style_ensemble_gets_a_clear_error(env):
    tmp, _, feats, _ = env
    p = tmp / "v8.pkl"; p.write_bytes(pickle.dumps({"models": {}, "weights": {}, "feature_sets": {}}))
    with pytest.raises(RuntimeError, match="predictor plugin"):
        W.load_predictor({"name": "v8", "pickle": str(p)})


def test_plugin_predictor(env):
    tmp, pkl, feats, df = env
    (tmp / "plug.py").write_text("import numpy as np\ndef load(path):\n    return lambda df: 1/(1+np.exp(-df['f0']))\n")
    s = spec(tmp, pkl, models=[{"name": "plug", "pickle": str(pkl), "predictor": f"{tmp}/plug.py:load"}])
    assert "plug" in W.run(s, tmp / "out")["metrics"]


def test_identical_contenders_have_zero_paired_difference(env):
    tmp, pkl, *_ = env
    s = spec(tmp, pkl, columns=[{"name": "prod", "column": "prod_prob"}])
    d = W.run(s, tmp / "out")["bootstrap"]["paired"]["m - prod"]
    assert abs(d["accuracy_diff"]) < 1e-9 and d["accuracy_diff_ci"] == [0.0, 0.0]


def test_too_few_games_is_an_error(env):
    tmp, pkl, *_ = env
    with pytest.raises(RuntimeError, match="nothing meaningful"):
        W.run(spec(tmp, pkl, start="2026-08-30", end="2026-08-31"), tmp / "out")


def test_runs_as_a_standalone_script_without_mllab(env):
    tmp, pkl, *_ = env
    (tmp / "spec.json").write_text(json.dumps(spec(tmp, pkl)))
    r = subprocess.run([sys.executable, str(WORKER), str(tmp / "spec.json")], capture_output=True, text=True, cwd=tmp)
    assert r.returncode == 0, r.stderr
    assert "# Backtest" in r.stdout


def make_history(tmp, feats, n=400, seed=9, label="home_won", overlap_rows=0):
    rng = np.random.default_rng(seed)
    h = pd.DataFrame({f: rng.normal(size=n) for f in feats})
    h[label] = (rng.random(n) < 1 / (1 + np.exp(-(1.2 * h.f0 - 0.8 * h.f1)))).astype(int)
    h["game_date"] = pd.Timestamp("2025-04-01") + pd.to_timedelta(np.arange(n) // 5, unit="D")
    if overlap_rows:  # rows dated INSIDE the evaluation window: must be ignored
        h.loc[h.index[-overlap_rows:], "game_date"] = pd.Timestamp("2026-06-01")
    h.to_parquet(tmp / "hist.parquet")
    return h


def test_history_warm_start_removes_the_min_train_cut(env):
    tmp, pkl, feats, df = env
    make_history(tmp, feats)
    base = spec(tmp, pkl, retrain=[{"name": "rt", "features": feats, "params": {"n_estimators": 15}, "walk_forward": {"min_train": 150, "step": 100}}])
    warm = spec(tmp, pkl, history={"file": str(tmp / "hist.parquet"), "label": "home_won"},
                retrain=[{"name": "rt", "features": feats, "params": {"n_estimators": 15}, "walk_forward": {"min_train": 0, "step": 100}}])
    cold_n, warm_n = W.run(base, tmp / "o1")["games_compared"], W.run(warm, tmp / "o2")
    assert warm_n["games_compared"] > cold_n  # retrain now scores from game one, so every contender covers more games
    assert warm_n["info"]["rt"]["history_rows"] == 400 and warm_n["info"]["rt"]["history_missing_features"] == []


def test_history_rows_inside_the_eval_window_are_ignored(env):
    """History dated on/after the first evaluated game must not train the model (it would be look-ahead)."""
    tmp, pkl, feats, df = env
    d = df.copy(); d["game_date"] = pd.to_datetime(d.game_date)
    first = d.game_date.min()
    h = make_history(tmp, feats, overlap_rows=60)
    hd = pd.read_parquet(tmp / "hist.parquet")
    (Xh, yh), _ = W.load_history({"history": {"file": str(tmp / "hist.parquet"), "label": "home_won"}}, feats, first, "home_win", "game_date")
    assert len(yh) == len(h) - 60  # the 60 rows dated 2026-06-01 were dropped
    args = dict(df=d, feats=feats, label="home_win", date="game_date", family="xgboost", params={"n_estimators": 10}, min_train=0, step=100)
    a = W.walk_forward(hist=(Xh, yh), **args)
    hd.loc[hd.game_date >= pd.Timestamp("2026-06-01"), "home_won"] = 1 - hd.home_won  # poison the dropped rows
    hd.to_parquet(tmp / "hist.parquet")
    (Xh2, yh2), _ = W.load_history({"history": {"file": str(tmp / "hist.parquet"), "label": "home_won"}}, feats, first, "home_win", "game_date")
    assert np.allclose(a, W.walk_forward(hist=(Xh2, yh2), **args), equal_nan=True)


def test_history_missing_features_are_reported_not_hidden(env):
    tmp, pkl, feats, df = env
    make_history(tmp, feats[:-1])  # history lacks f4
    s = spec(tmp, pkl, history={"file": str(tmp / "hist.parquet"), "label": "home_won"},
             retrain=[{"name": "rt", "features": feats, "params": {"n_estimators": 10}, "walk_forward": {"min_train": 0, "step": 200}}])
    assert W.run(s, tmp / "out")["info"]["rt"]["history_missing_features"] == ["f4"]


# --- season-aligned walk-forward, all-numeric features, impute ------------------------------------------------------

def seasons_frame(n_seasons=8, per=120, seed=2):
    rng = np.random.default_rng(seed)
    n = n_seasons * per
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(4)})
    df["season"] = np.repeat(np.arange(2010, 2010 + n_seasons), per)
    df["game_date"] = pd.to_datetime(df.season.astype(str) + "-09-01") + pd.to_timedelta(np.tile(np.arange(per), n_seasons), unit="D")
    df["home_won"] = (rng.random(n) < 1 / (1 + np.exp(-(1.1 * df.f0 - 0.7 * df.f1)))).astype(int)
    df["game_id"] = [f"g{i}" for i in range(n)]
    df["home_team"] = "X"  # a string column: never a feature
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def test_group_walk_forward_scores_whole_seasons_after_min_groups():
    df = seasons_frame()
    out = W.walk_forward(df, ["f0", "f1", "f2"], "home_won", "game_date", "xgboost", {"n_estimators": 15}, 0, 100, group="season", min_groups=3)
    scored = df.season[np.isfinite(out)].unique()
    assert list(scored) == [2013, 2014, 2015, 2016, 2017]  # seasons 2010-12 are training-only
    for s in scored:  # each scored season is scored entirely, never partially
        assert np.isfinite(out[df.season == s]).all()
    assert not np.isfinite(out[df.season < 2013]).any()


def test_group_walk_forward_cannot_see_the_future():
    df = seasons_frame()
    args = dict(feats=["f0", "f1", "f2"], label="home_won", date="game_date", family="xgboost", params={"n_estimators": 15},
                min_train=0, step=100, group="season", min_groups=3)
    base = W.walk_forward(df, **args)
    poisoned = df.copy()
    poisoned.loc[poisoned.season >= 2015, "home_won"] = 1 - poisoned.home_won  # flip every label from 2015 on
    alt = W.walk_forward(poisoned, **args)
    early = (df.season <= 2015).to_numpy() & np.isfinite(base)
    assert early.sum() > 200
    assert np.allclose(base[early], alt[early])  # 2013-2015 unchanged: 2015 is predicted before its own labels are ever used
    assert not np.allclose(base[df.season >= 2016], alt[df.season >= 2016], equal_nan=True)  # and the poison did matter later


def test_impute_zero_differs_from_median_and_bad_value_is_rejected():
    df = seasons_frame()
    rng = np.random.default_rng(1)
    df.loc[rng.random(len(df)) < 0.3, "f0"] = np.nan
    df["f0"] = df["f0"] + 5  # shift so median and zero are far apart
    kw = dict(df=df, feats=["f0", "f1"], label="home_won", date="game_date", family="xgboost", params={"n_estimators": 20},
              min_train=0, step=100, group="season", min_groups=3)
    med, zero = W.walk_forward(impute="median", **kw), W.walk_forward(impute="zero", **kw)
    ok = np.isfinite(med) & np.isfinite(zero)
    assert ok.sum() > 100 and not np.allclose(med[ok], zero[ok])
    with pytest.raises(RuntimeError, match="impute must be"):
        W.walk_forward(impute="mean", **kw)


def test_all_numeric_features_exclude_identity_labels_and_strings():
    df = seasons_frame()
    item = {"name": "x", "features": {"all_numeric": True, "exclude": ["season"]}}
    feats = W.resolve_features(item, df, {"home_won", "game_date", "game_id"}, {})
    assert feats == ["f0", "f1", "f2", "f3"]  # no label, no ids, no string column, no excluded season
    with pytest.raises(RuntimeError, match="selected no columns"):
        W.resolve_features({"name": "y", "features": {"all_numeric": True, "exclude": ["f0", "f1", "f2", "f3", "season"]}}, df, {"home_won", "game_date", "game_id"}, {})


def test_group_mode_end_to_end_with_all_numeric(env, tmp_path):
    df = seasons_frame(); df.to_parquet(tmp_path / "s.parquet")
    spec = {"data": str(tmp_path / "s.parquet"), "label": "home_won", "date": "game_date", "id": "game_id", "out_dir": str(tmp_path / "o"),
            "bootstrap": 100, "columns": [{"name": "f0col", "column": "f0"}], "baseline": True,
            "retrain": [{"name": "x", "features": {"all_numeric": True, "exclude": ["season"]}, "family": "logreg", "impute": "zero",
                         "walk_forward": {"group": "season", "min_groups": 3}}]}
    spec["columns"] = []
    res = W.run(spec, tmp_path / "o")
    assert res["games_compared"] == 5 * 120 and res["info"]["x"]["n_features"] == 4 and res["info"]["x"]["impute"] == "zero"


def test_seeds_average_the_per_seed_probabilities_exactly():
    df = seasons_frame()
    kw = dict(df=df, feats=["f0", "f1", "f2"], label="home_won", date="game_date", family="xgboost", params={"n_estimators": 15, "subsample": 0.7},
              min_train=0, step=100, group="season", min_groups=3)
    singles = [W.walk_forward(seeds=[sd], **kw) for sd in (1, 2, 3)]
    avg = W.walk_forward(seeds=[1, 2, 3], **kw)
    ok = np.isfinite(avg)
    assert ok.sum() > 100 and np.allclose(avg[ok], np.mean(singles, axis=0)[ok])  # the ensemble IS the mean of its members
    assert not np.allclose(singles[0][ok], singles[1][ok])  # and the seeds genuinely differ, so averaging does something


def test_averaging_seeds_reduces_run_to_run_variation():
    df = seasons_frame()
    kw = dict(df=df, feats=["f0", "f1", "f2"], label="home_won", date="game_date", family="xgboost",
              params={"n_estimators": 20, "subsample": 0.6, "colsample_bytree": 0.6}, min_train=0, step=100, group="season", min_groups=3)
    one = [W.walk_forward(seeds=[s], **kw) for s in (10, 11, 12, 13)]
    five = [W.walk_forward(seeds=list(range(s, s + 5)), **kw) for s in (100, 200, 300, 400)]
    ok = np.isfinite(one[0])
    spread = lambda runs: float(np.mean(np.std(np.array(runs)[:, ok], axis=0)))  # noqa: E731
    assert spread(five) < spread(one)


def test_seeds_are_ignored_by_logreg_and_recorded_in_the_run_info(tmp_path):
    df = seasons_frame(); df.to_parquet(tmp_path / "s.parquet")
    spec = {"data": str(tmp_path / "s.parquet"), "label": "home_won", "date": "game_date", "id": "game_id", "out_dir": str(tmp_path / "o"), "bootstrap": 50,
            "retrain": [{"name": "x", "features": ["f0", "f1"], "family": "xgboost", "params": {"n_estimators": 10}, "seeds": [1, 2],
                         "walk_forward": {"group": "season", "min_groups": 3}}]}
    assert W.run(spec, tmp_path / "o")["info"]["x"]["seeds"] == [1, 2]


def test_exclude_pattern_drops_whole_feature_families():
    df = seasons_frame()
    df["home_off_epa_3g"] = 1.0; df["net_epa_8g"] = 2.0; df["home_win_pct"] = 3.0
    item = {"name": "x", "features": {"all_numeric": True, "exclude": ["season"], "exclude_pattern": "epa"}}
    feats = W.resolve_features(item, df, {"home_won", "game_date", "game_id"}, {})
    assert "home_win_pct" in feats and not [c for c in feats if "epa" in c]


def test_filter_keeps_one_division_but_features_come_from_the_shared_frame(tmp_path):
    df = seasons_frame()
    df["division"] = np.where(np.arange(len(df)) % 3 == 0, "fcs", "fbs")
    df["shared_pool_feature"] = df.f0.expanding().mean()  # stands in for an Elo pool built over BOTH divisions before filtering
    df.to_parquet(tmp_path / "d.parquet")
    spec = {"data": str(tmp_path / "d.parquet"), "label": "home_won", "date": "game_date", "id": "game_id", "out_dir": str(tmp_path / "o"), "bootstrap": 50,
            "filter": {"column": "division", "equals": "fbs"},
            "retrain": [{"name": "x", "features": ["f0", "shared_pool_feature"], "family": "logreg", "walk_forward": {"group": "season", "min_groups": 3}}]}
    res = W.run(spec, tmp_path / "o")
    preds = pd.read_parquet(tmp_path / "o" / "predictions.parquet")
    fbs_ids = set(df.loc[df.division == "fbs", "game_id"])
    assert set(preds.game_id) <= fbs_ids and len(preds) > 100 and res["games_compared"] == len(preds)
    spec["filter"] = {"column": "division", "in": ["fbs", "fcs"]}
    assert W.run(spec, tmp_path / "o2")["games_compared"] > res["games_compared"]  # `in` keeps both


def test_filter_errors_are_clear(tmp_path):
    df = seasons_frame(); df["division"] = "fbs"; df.to_parquet(tmp_path / "d.parquet")
    base = {"data": str(tmp_path / "d.parquet"), "label": "home_won", "date": "game_date", "id": "game_id", "out_dir": str(tmp_path / "o"), "bootstrap": 20,
            "retrain": [{"name": "x", "features": ["f0"], "walk_forward": {"group": "season", "min_groups": 3}}]}
    with pytest.raises(RuntimeError, match="not in data"):
        W.run({**base, "filter": {"column": "nope", "equals": 1}}, tmp_path / "o")
    with pytest.raises(RuntimeError, match="matched no rows"):
        W.run({**base, "filter": {"column": "division", "equals": "fcs"}}, tmp_path / "o")


# --- nested hyperparameter search ---------------------------------------------------------------------------------

SPACE = {"max_depth": {"low": 1, "high": 4, "type": "int"}, "learning_rate": {"low": 0.01, "high": 0.3, "log": True},
         "subsample": {"low": 0.5, "high": 1.0}, "n_estimators": [10, 20, 40]}
TUNE_KW = dict(feats=["f0", "f1", "f2"], label="home_won", date="game_date", group="season", family="xgboost",
               base_params={"n_estimators": 10}, eval_start_groups=5, impute="median")


def test_sampling_stays_in_bounds_types_and_is_seed_deterministic():
    rng = np.random.default_rng(3)
    draws = [{k: W._sample(rng, v) for k, v in SPACE.items()} for _ in range(300)]
    assert all(1 <= d["max_depth"] <= 4 and isinstance(d["max_depth"], int) for d in draws)
    assert all(0.01 <= d["learning_rate"] <= 0.3 and 0.5 <= d["subsample"] <= 1.0 for d in draws)
    assert {d["n_estimators"] for d in draws} == {10, 20, 40}
    lr = np.array([d["learning_rate"] for d in draws])
    assert np.sum(lr < 0.055) > 100  # log scale puts about half the mass below the geometric midpoint, unlike a linear draw
    rng2 = np.random.default_rng(3)
    assert draws[:3] == [{k: W._sample(rng2, v) for k, v in SPACE.items()} for _ in range(3)]


def test_tuning_ranks_trials_and_reports_the_default_for_comparison():
    df = seasons_frame(n_seasons=8)
    t = W.tune_params(df, tune={"space": SPACE, "trials": 6, "seed": 1, "inner_folds": 2}, **TUNE_KW)
    losses = [x["inner_log_loss"] for x in t["top_trials"]]
    assert losses == sorted(losses) and t["best_inner_log_loss"] == losses[0] and t["n_trials"] == 6 and len(t["top_trials"]) == 5
    assert set(t["best_params"]) == set(SPACE) and "optimistic" in t["note"] and t["default_inner_log_loss"] > 0
    assert t == W.tune_params(df, tune={"space": SPACE, "trials": 6, "seed": 1, "inner_folds": 2}, **TUNE_KW)  # same seed, same search


def test_tuning_never_sees_evaluation_data():
    """Flip every label from the first evaluated season onward: the chosen parameters and inner scores must not move at all."""
    df = seasons_frame(n_seasons=8)
    poisoned = df.copy()
    poisoned.loc[poisoned.season >= 2015, "home_won"] = 1 - poisoned.home_won  # eval starts at group index 5 = season 2015
    tune = {"space": SPACE, "trials": 5, "seed": 2, "inner_folds": 2}
    a, b = W.tune_params(df, tune=tune, **TUNE_KW), W.tune_params(poisoned, tune=tune, **TUNE_KW)
    assert a["best_params"] == b["best_params"] and a["best_inner_log_loss"] == b["best_inner_log_loss"]
    assert a["default_inner_log_loss"] == b["default_inner_log_loss"]
    assert a["tuning_window"][1] < "2015-01-01"  # the window ends before the first evaluated season begins


def test_tuning_needs_history_before_the_evaluation_window():
    df = seasons_frame(n_seasons=8)
    with pytest.raises(RuntimeError, match="at least 3 groups"):
        W.tune_params(df, tune={"space": SPACE, "trials": 2}, **{**TUNE_KW, "eval_start_groups": 2})


def test_tuned_contender_end_to_end_only_scores_after_the_tuning_window(tmp_path):
    df = seasons_frame(n_seasons=9); df.to_parquet(tmp_path / "s.parquet")
    spec = {"data": str(tmp_path / "s.parquet"), "label": "home_won", "date": "game_date", "id": "game_id", "out_dir": str(tmp_path / "o"), "bootstrap": 50,
            "retrain": [
                {"name": "default", "features": ["f0", "f1", "f2"], "params": {"n_estimators": 10}, "walk_forward": {"group": "season", "min_groups": 5}},
                {"name": "tuned", "features": ["f0", "f1", "f2"], "params": {"n_estimators": 10}, "walk_forward": {"group": "season", "min_groups": 5},
                 "tune": {"space": SPACE, "trials": 4, "seed": 3, "inner_folds": 2}}]}
    res = W.run(spec, tmp_path / "o")
    info = res["info"]["tuned"]
    assert res["games_compared"] == 4 * 120  # seasons 2015-2018 only; 2010-2014 were tuning/training-only
    assert info["tuned"]["tuning_window"][1] < res["date_range"][0]  # tuning ended before the first scored game
    assert all(info["params"][k] == v for k, v in info["tuned"]["best_params"].items())  # the frozen winner is what was scored
    assert "tuned" not in res["info"]["default"]


def test_tune_requires_group_mode(tmp_path):
    df = seasons_frame(); df.to_parquet(tmp_path / "s.parquet")
    spec = {"data": str(tmp_path / "s.parquet"), "label": "home_won", "date": "game_date", "id": "game_id", "out_dir": str(tmp_path / "o"), "bootstrap": 20,
            "retrain": [{"name": "t", "features": ["f0"], "walk_forward": {"min_train": 100, "step": 100}, "tune": {"space": SPACE}}]}
    with pytest.raises(RuntimeError, match="tune needs walk_forward.group"):
        W.run(spec, tmp_path / "o")
