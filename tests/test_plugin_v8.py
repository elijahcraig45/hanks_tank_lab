import ast
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("catboost")
pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = Path.home() / "Documents/personalDev/mlb/hanks_tank_ml/src/predict_today_games.py"


def load_plugin():
    spec = importlib.util.spec_from_file_location("mlb_v8_plugin", ROOT / "plugins/mlb_v8.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def real_class():
    """The production V8EnsemblePredictor, lifted out of the repo file without importing its heavy module."""
    if not REPO_SRC.exists():
        pytest.skip("hanks_tank_ml source not available")
    node = next(n for n in ast.parse(REPO_SRC.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "V8EnsemblePredictor")
    ns = {"np": np, "pd": pd}
    exec(compile(ast.Module([node], []), "v8", "exec"), ns)  # noqa: S102
    return ns["V8EnsemblePredictor"]


def make(n=300, calibrated=False, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(8)})
    df["t_home"], df["t_away"] = rng.integers(100, 130, n), rng.integers(100, 130, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(df.f0 - 0.5 * df.f1)))).astype(int)
    fsets = {"cat_tuned": ["f0", "f1", "f2", "t_home", "t_away"], "cat_wide": ["f0", "f1", "f2", "f3", "f4", "f5", "t_home", "t_away"],
             "lgb": ["f2", "f3", "f4", "f5", "f6", "f7"], "mlp": ["f0", "f1", "f2", "f3"]}
    cats = ["t_home", "t_away"]
    models = {
        "cat_tuned": CatBoostClassifier(iterations=25, verbose=0, cat_features=cats, random_seed=1, allow_writing_files=False).fit(df[fsets["cat_tuned"]], y),
        "cat_wide": CatBoostClassifier(iterations=25, verbose=0, cat_features=cats, random_seed=2, allow_writing_files=False).fit(df[fsets["cat_wide"]], y),
        "lgb": LGBMClassifier(n_estimators=25, verbose=-1, random_state=0).fit(df[fsets["lgb"]], y)}
    sc = StandardScaler().fit(df[fsets["mlp"]].values)
    models["mlp"] = MLPClassifier((8,), max_iter=300, random_state=0).fit(sc.transform(df[fsets["mlp"]].values), y)
    bundle = {"models": models, "scalers": {"mlp": sc}, "weights": {"cat_tuned": 0.3, "cat_wide": 0.2, "lgb": 0.3, "mlp": 0.2},
              "feature_sets": fsets, "fill_values": {c: float(df[c].median()) for c in df.columns if c.startswith("f")},
              "cat_feature_indices": [3, 4]}
    if calibrated:
        raw = real_class()(bundle).predict_proba(df)[:, 1]
        bundle["calibrator"] = IsotonicRegression(out_of_bounds="clip").fit(raw, y)
        bundle["calibrator_type"] = "isotonic"
    return bundle, df


def reference(bundle, df):
    """What production does: predict_game's NaN fill, then the real class (calibrator attached like the loader does)."""
    X = df.copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(bundle.get("fill_values", {}).get(col, 0.0))
    pred = real_class()(bundle)
    if "calibrator" in bundle:
        pred.calibrator = bundle["calibrator"]
    return pred.predict_proba(X)[:, 1]


@pytest.mark.parametrize("calibrated", [False, True])
def test_plugin_matches_the_production_class_including_nans(tmp_path, calibrated):
    bundle, df = make(calibrated=calibrated)
    rng = np.random.default_rng(5)
    holes = df.copy()
    for c in ["f0", "f3", "f6"]:
        holes.loc[rng.random(len(df)) < 0.15, c] = np.nan  # production imputes these; so must the plugin
    p = tmp_path / "v8.pkl"; p.write_bytes(pickle.dumps(bundle))
    predict = load_plugin().load(str(p))
    assert np.allclose(predict(holes), reference(bundle, holes), atol=1e-12)
    assert predict.info["calibrated"] is calibrated and predict.info["missing_frac"] == 0


def test_plugin_refuses_when_features_are_missing(tmp_path):
    bundle, df = make()
    p = tmp_path / "v8.pkl"; p.write_bytes(pickle.dumps(bundle))
    predict = load_plugin().load(str(p))
    with pytest.raises(RuntimeError, match="lacks"):
        predict(df.drop(columns=["f2", "f3", "f4"]))  # 3 of 10 features gone: refuse, do not fill zeros


def test_plugin_tolerates_and_reports_a_small_gap(tmp_path):
    bundle, df = make()
    p = tmp_path / "v8.pkl"; p.write_bytes(pickle.dumps(bundle))
    predict = load_plugin().load(str(p), max_missing_frac=0.2)
    out = predict(df.drop(columns=["f7"]))
    assert len(out) == len(df) and predict.info["missing_columns"] == ["f7"]


def test_worker_reports_plugin_details_and_orchestrator_stages_the_plugin(tmp_path):
    from mllab import backtest as B
    from mllab import backtest_worker as W
    bundle, df = make()
    p = tmp_path / "v8.pkl"; p.write_bytes(pickle.dumps(bundle))
    fn, info = W.load_predictor({"name": "v8", "pickle": str(p), "predictor": str(ROOT / "plugins/mlb_v8.py") + ":load"})
    fn(df)
    assert info["details"]["sub_models"] == ["cat_tuned", "cat_wide", "lgb", "mlp"]  # shared dict, updated by the call
    work = tmp_path / "w"; work.mkdir()
    ref = B._stage_plugin("../plugins/mlb_v8.py:load", ROOT / "specs", work, "v8")
    assert ref == "plugins/v8.py:load" and (work / "plugins/v8.py").is_file()
    with pytest.raises(B.BacktestError, match="not found"):
        B._stage_plugin("../plugins/nope.py:load", ROOT / "specs", work, "x")
