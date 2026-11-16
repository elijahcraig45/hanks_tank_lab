import json
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("pyarrow")
from xgboost import XGBClassifier

from mllab import backtest_worker as BW
from mllab import train as T
from mllab.backtest import BacktestError
from mllab.jobs import DirectLauncher, JobError, LocalRunner
from mllab.registry import Registry

FEATS = [f"f{i}" for i in range(4)]


def frame(n=500, seed=4):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATS})
    df["home_win"] = (rng.random(n) < 1 / (1 + np.exp(-(1.3 * df.f0 - df.f1)))).astype(int)
    df["game_pk"] = np.arange(n)
    df["game_date"] = pd.Timestamp("2026-04-01") + pd.to_timedelta(np.arange(n) // 5, unit="D")
    return df


@pytest.fixture
def lab(tmp_path):
    df = frame()
    base = XGBClassifier(n_estimators=20, max_depth=3, eval_metric="logloss").fit(df[FEATS][:200], df.home_win[:200])
    df["prod_prob"] = base.predict_proba(df[FEATS])[:, 1]
    runner = LocalRunner(tmp_path / "lab", launcher=DirectLauncher(), python_bin=sys.executable)
    runner.data_root.mkdir(parents=True)
    df.to_parquet(runner.data_root / "feat.parquet")
    h = frame(300, seed=8); h["home_won"] = h.home_win; h["game_date"] = pd.Timestamp("2025-04-01") + pd.to_timedelta(np.arange(300) // 5, unit="D")
    h.to_parquet(runner.data_root / "hist.parquet")
    pkl = tmp_path / "base.pkl"
    pkl.write_bytes(pickle.dumps({"model": base, "features": FEATS, "fill_values": {}}))
    reg = Registry(tmp_path / "reg")
    reg.register("p", "game", "v1", pkl, features=FEATS)
    return runner, reg


def spec(**kw):
    s = {"name": "cand", "data": {"file": "feat.parquet"}, "env": "host", "features_from": {"project": "p", "model": "game", "version": "v1"},
         "params": {"n_estimators": 25, "max_depth": 2}, "train_end": "2026-06-10", "columns": [{"name": "prod", "column": "prod_prob"}]}
    s.update(kw)
    return s


def run_job(runner, reg, s):
    jid = T.submit(s, reg, runner)
    runner.launcher.wait(jid, 180)
    assert runner.status(jid)["state"] == "finished", runner.logs(jid)[0][-1500:]
    return jid


def test_train_scores_an_untouched_holdout_and_writes_a_loadable_model(lab):
    runner, reg = lab
    jid = run_job(runner, reg, spec())
    out = runner.jobs / jid / "code" / "outputs"
    m = json.loads((out / "metrics.json").read_text())
    df = frame()
    n_train = int((df.game_date <= "2026-06-10").sum())
    assert m["train_rows"] == n_train and m["holdout_rows"] == len(df) - n_train  # the cut is respected exactly
    assert {"cand", "prod", "home_prior"} <= set(m["holdout"]) and m["features"] == FEATS
    fn, info = BW.load_predictor({"name": "c", "pickle": str(out / "model.pkl")})  # same verified loader the backtests use
    p = fn(df)
    assert p.shape == (len(df),) and ((p > 0) & (p < 1)).all() and info["n_features"] == 4


def test_history_adds_training_rows(lab):
    runner, reg = lab
    plain = json.loads((runner.jobs / run_job(runner, reg, spec()) / "code/outputs/metrics.json").read_text())
    warm = json.loads((runner.jobs / run_job(runner, reg, spec(history={"file": "hist.parquet", "label": "home_won"})) / "code/outputs/metrics.json").read_text())
    assert warm["train_rows"] == plain["train_rows"] + 300 and warm["history_rows"] == 300


def test_register_job_records_lineage_features_and_data_hash(lab):
    runner, reg = lab
    jid = run_job(runner, reg, spec())
    meta = T.register_from_job(reg, runner, jid, "p", "game", "v2", parent="v1")
    assert meta.parent == "v1" and meta.features == FEATS and meta.source == f"job:{jid}" and len(meta.data_hash) == 64
    assert meta.metrics["holdout"]["cand"]["n"] == meta.metrics["holdout_games"]
    assert reg.verify("p", "game", "v2").exists()
    with pytest.raises(Exception, match="already exists"):
        T.register_from_job(reg, runner, jid, "p", "game", "v2")  # immutable: cannot overwrite a version


def test_same_inputs_give_same_data_hash(lab):
    runner, reg = lab
    a = T.register_from_job(reg, runner, run_job(runner, reg, spec()), "p", "game", "a")
    b = T.register_from_job(reg, runner, run_job(runner, reg, spec(params={"max_depth": 3})), "p", "game", "b")
    assert a.data_hash == b.data_hash and a.sha256 != b.sha256


def test_only_finished_training_jobs_can_be_registered(lab, tmp_path):
    runner, reg = lab
    code = tmp_path / "plain"; code.mkdir()
    (code / "s.py").write_text("print('not a training job')")
    from mllab.jobs import JobSpec
    jid = runner.submit(JobSpec(name="x", script="s.py", code_dir=str(code), timeout_s=30))
    runner.launcher.wait(jid)
    with pytest.raises(JobError, match="no model.pkl"):
        T.register_from_job(reg, runner, jid, "p", "game", "v9")


@pytest.mark.parametrize("patch, msg", [
    ({"name": "bad name"}, "needs a name"),
    ({"features": ["a"]}, "exactly one of features or features_from"),
    ({"features_from": None}, "exactly one of features or features_from"),
    ({"features_from": {"project": "p"}}, "features_from needs"),
    ({"family": "deeplearning"}, "family must be"),
    ({"data": {}}, "exactly one of sql_file or file"),
    ({"bogus": 1}, "unknown spec key"),
    ({"history": {"file": "../x"}}, "history needs"),
])
def test_spec_validation(patch, msg):
    with pytest.raises(BacktestError, match=msg):
        T.validate(spec(**patch))


def test_lock_env_needs_a_lockfile(lab):
    runner, reg = lab
    with pytest.raises(BacktestError, match="requirements.lock"):
        T.prepare(spec(env="lock"), reg, "feat.parquet")


def test_leaky_features_are_refused(lab):
    runner, reg = lab
    d = pd.read_parquet(runner.data_root / "feat.parquet"); d["home_score"] = d.home_win * 4; d.to_parquet(runner.data_root / "feat.parquet")
    s = spec(features=FEATS + ["home_score"]); s.pop("features_from")
    jid = T.submit(s, reg, runner)
    runner.launcher.wait(jid, 120)
    assert runner.status(jid)["state"] == "failed" and "leaky" in runner.logs(jid)[0]


def football_frame(tmp_path, runner):
    df = frame(600)
    df["season"] = 2015 + np.arange(600) // 100
    df["division"] = np.where(np.arange(600) % 2 == 0, "fbs", "fcs")
    df["home_team"] = "X"
    df["home_won"] = df["home_win"]; df = df.drop(columns=["home_win"])
    df["game_id"] = [f"g{i}" for i in range(600)]
    df.to_parquet(runner.data_root / "fb.parquet")
    return df


def fb_spec(**kw):
    s = {"name": "snap", "data": {"file": "fb.parquet"}, "env": "host", "label": "home_won", "id": "game_id",
         "features": {"all_numeric": True, "exclude": ["season", "game_pk", "prod_prob"]}, "family": "logreg", "impute": "zero",
         "params": {"C": 0.1}, "train_end": "2026-08-01"}
    s.update(kw)
    return s


def test_football_style_snapshot_with_all_numeric_features_zero_impute_and_division_filter(lab):
    runner, reg = lab
    football_frame(runner.data_root.parent, runner)
    jid = run_job(runner, reg, fb_spec(filter={"column": "division", "equals": "fbs"}))
    m = json.loads((runner.jobs / jid / "code/outputs/metrics.json").read_text())
    assert m["features"] == ["f0", "f1", "f2", "f3"] and m["impute"] == "zero" and m["filter"] == {"column": "division", "equals": "fbs"}
    assert m["train_rows"] + m["holdout_rows"] == 300  # only the fbs half of the 600 games
    payload = pickle.loads((runner.jobs / jid / "code/outputs/model.pkl").read_bytes())
    assert set(payload["fill_values"].values()) == {0.0}  # serving imputes exactly like training: zeros
    meta = T.register_from_job(reg, runner, jid, "p", "nfl", "v1")
    assert meta.features == ["f0", "f1", "f2", "f3"] and meta.metrics["family"] == "logreg"


def test_zero_imputation_differs_from_median_in_the_saved_fill_values(lab):
    runner, reg = lab
    football_frame(runner.data_root.parent, runner)
    med = pickle.loads((runner.jobs / run_job(runner, reg, fb_spec(impute="median")) / "code/outputs/model.pkl").read_bytes())
    assert any(v != 0.0 for v in med["fill_values"].values())


def test_train_spec_validation_for_football_options():
    T.validate(fb_spec())
    for patch, msg in [({"features": {"exclude": ["a"]}}, "all_numeric"), ({"features": []}, "non-empty list"), ({"impute": "mean"}, "impute must be"),
                       ({"filter": {"column": "d"}}, "filter must be"), ({"filter": {"column": "d", "equals": 1, "in": [1]}}, "filter must be")]:
        with pytest.raises(BacktestError, match=msg):
            T.validate(fb_spec(**patch))


def test_env_lock_gives_an_explicit_feature_spec_a_pinned_environment(lab, tmp_path):
    runner, reg = lab
    football_frame(tmp_path, runner)
    (tmp_path / "specs").mkdir(); (tmp_path / "specs" / "r.lock").write_text("numpy==2.4.0\n")
    prep = T.prepare(fb_spec(env="lock", env_lock="r.lock"), reg, "fb.parquet", base=tmp_path / "specs")
    assert prep.job_spec.lock == "requirements.lock" and (prep.workdir / "requirements.lock").read_text() == "numpy==2.4.0\n"
    with pytest.raises(BacktestError, match="env_lock not found"):
        T.prepare(fb_spec(env="lock", env_lock="nope.lock"), reg, "fb.parquet", base=tmp_path / "specs")


def test_feature_set_is_written_into_the_artifact_only_when_declared(lab):
    runner, reg = lab
    with_fs = pickle.loads((runner.jobs / run_job(runner, reg, spec(feature_set="v10")) / "code/outputs/model.pkl").read_bytes())
    without = pickle.loads((runner.jobs / run_job(runner, reg, spec()) / "code/outputs/model.pkl").read_bytes())
    assert with_fs["feature_set"] == "v10" and "feature_set" not in without  # undeclared artifacts stay exactly as before


def test_feature_set_validation():
    T.validate(spec(feature_set="v8"))
    with pytest.raises(BacktestError, match="feature_set must be"):
        T.validate(spec(feature_set="v11"))
