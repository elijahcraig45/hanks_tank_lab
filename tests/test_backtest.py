import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("pyarrow")
from xgboost import XGBClassifier

from mllab import backtest as B
from mllab.jobs import DirectLauncher, LocalRunner
from mllab.registry import Registry


def data(n=500):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(4)})
    df["home_win"] = (rng.random(n) < 1 / (1 + np.exp(-(1.3 * df.f0 - df.f1)))).astype(int)
    df["game_pk"] = np.arange(n)
    df["game_date"] = pd.Timestamp("2026-04-01") + pd.to_timedelta(np.arange(n) // 5, unit="D")
    return df


@pytest.fixture
def lab(tmp_path):
    df = data()
    feats = [f"f{i}" for i in range(4)]
    m = XGBClassifier(n_estimators=30, max_depth=3, eval_metric="logloss").fit(df[feats][:250], df.home_win[:250])
    df["prod_prob"] = m.predict_proba(df[feats])[:, 1]
    runner = LocalRunner(tmp_path / "lab", launcher=DirectLauncher(), python_bin=sys.executable)
    runner.data_root.mkdir(parents=True)
    df.to_parquet(runner.data_root / "feat.parquet")
    pkl = tmp_path / "m.pkl"
    pkl.write_bytes(pickle.dumps({"model": m, "features": feats, "fill_values": {}}))
    reg = Registry(tmp_path / "reg")
    reg.register("p", "game", "v1", pkl)
    return runner, reg, tmp_path


def spec(**kw):
    s = {"name": "t1", "data": {"file": "feat.parquet"}, "env": "host", "bootstrap": 200, "start": "2026-05-01",
         "models": [{"name": "v1", "version": {"project": "p", "model": "game", "version": "v1"}, "parity": True}],
         "columns": [{"name": "prod", "column": "prod_prob", "reference": True}]}
    s.update(kw)
    return s


def test_end_to_end_through_registry_and_job_runner(lab):
    runner, reg, tmp = lab
    jid = B.submit(spec(retrain=[{"name": "rt", "features_from": "v1", "params": {"n_estimators": 15},
                                  "walk_forward": {"min_train": 120, "step": 60}}]), reg, runner)
    runner.launcher.wait(jid, 180)
    assert runner.status(jid)["state"] == "finished", runner.logs(jid)[0][-1500:]
    res = B.results(runner, jid)
    assert {"v1", "prod", "rt", "home_prior"} <= set(res["metrics"])
    assert res["parity"]["v1"]["max_abs_diff"] < 1e-6  # registry copy reproduces the served probabilities
    assert "# Backtest" in B.report_path(runner, jid).read_text()


def test_tampered_registry_copy_never_reaches_a_job(lab):
    runner, reg, _ = lab
    vdir = reg.version_dir("p", "game", "v1"); vdir.chmod(0o755)
    (vdir / "model.pkl").chmod(0o644); (vdir / "model.pkl").write_bytes(b"evil")
    with pytest.raises(B.BacktestError, match="hash mismatch"):
        B.submit(spec(), reg, runner)
    assert runner.active_ids() == []


def test_lock_env_needs_a_lockfile(lab):
    runner, reg, _ = lab
    with pytest.raises(B.BacktestError, match="requirements.lock"):
        B.submit(spec(env="lock"), reg, runner)


def test_mixed_locks_are_refused(lab, tmp_path):
    runner, reg, tmp = lab
    for v, txt in (("a", "numpy==1\n"), ("b", "numpy==2\n")):
        lock = tmp / f"{v}.lock"; lock.write_text(txt)
        art = tmp / f"{v}.pkl"; art.write_bytes(b"x")
        reg.register("p", "game", v, art, lockfile=lock)
    two = spec(env="lock", models=[{"name": "a", "version": {"project": "p", "model": "game", "version": "a"}},
                                   {"name": "b", "version": {"project": "p", "model": "game", "version": "b"}}], columns=[])
    with pytest.raises(B.BacktestError, match="different requirements locks"):
        B.prepare(two, reg, "feat.parquet")


@pytest.mark.parametrize("patch, msg", [
    ({"name": "bad name"}, "needs a name"),
    ({"data": {}}, "exactly one"),
    ({"data": {"file": "a", "sql_file": "b"}}, "exactly one"),
    ({"bogus": 1}, "unknown spec key"),
    ({"models": [], "columns": [], "retrain": []}, "nothing to compare"),
    ({"columns": [{"name": "v1", "column": "x"}]}, "unique"),
    ({"columns": [{"name": "home_prior", "column": "x"}]}, "unique"),
    ({"columns": [{"name": "prod", "column": "x"}]}, "parity needs exactly one"),
    ({"retrain": [{"name": "r", "features_from": "nope"}]}, "features_from"),
    ({"retrain": [{"name": "r"}]}, "features or features_from"),
])
def test_spec_validation(patch, msg):
    with pytest.raises(B.BacktestError, match=msg):
        B.validate(spec(**patch))


def test_data_file_must_exist_and_not_traverse(lab):
    runner, *_ = lab
    with pytest.raises(B.BacktestError):
        B.prepare_data(spec(data={"file": "../etc/passwd"}), runner.data_root)
    with pytest.raises(B.BacktestError):
        B.prepare_data(spec(data={"file": "missing.parquet"}), runner.data_root)


def test_sql_data_goes_through_bq_guards(lab, monkeypatch, tmp_path):
    runner, *_ = lab
    seen = {}
    monkeypatch.setattr("mllab.bq.bq", lambda sql, **kw: (seen.update(sql=sql, kw=kw), data(50))[1])
    (tmp_path / "q.sql").write_text("SELECT 1")
    out = B.prepare_data(spec(data={"sql_file": "q.sql"}), runner.data_root, bq_kwargs={"max_bytes": 5}, base=tmp_path)
    assert out == "t1.parquet" and (runner.data_root / out).exists()
    assert seen["sql"] == "SELECT 1" and seen["kw"] == {"max_bytes": 5}


def test_yaml_dates_survive_into_the_json_worker_spec(lab, tmp_path):
    """Regression: PyYAML parses 2026-05-01 as a date object, which json.dumps rejects."""
    runner, reg, _ = lab
    p = tmp_path / "s.yaml"
    p.write_text("name: t1\ndata: {file: feat.parquet}\nenv: host\nstart: 2026-05-01\nend: 2026-07-01\n"
                 "columns: [{name: prod, column: prod_prob}]\n")
    raw = B.load(p)
    assert raw["start"] == "2026-05-01" and raw["end"] == "2026-07-01"
    prep = B.prepare(raw, reg, "feat.parquet")
    assert '"start": "2026-05-01"' in (prep.workdir / "spec.json").read_text()


def test_history_becomes_a_job_input_and_rejects_paths(lab):
    runner, reg, tmp = lab
    data(60).assign(home_won=1).to_parquet(runner.data_root / "hist.parquet")
    s = spec(history={"file": "hist.parquet", "label": "home_won"})
    prep = B.prepare(s, reg, "feat.parquet")
    assert prep.job_spec.inputs == ["feat.parquet", "hist.parquet"]
    assert prep.worker_spec["history"]["file"] == "../inputs/hist.parquet"
    for bad in ({"file": "../x.parquet"}, {"file": ""}, "hist.parquet"):
        with pytest.raises(B.BacktestError, match="history needs"):
            B.validate(spec(history=bad))


@pytest.mark.parametrize("item, msg", [
    ({"name": "r", "features": {"exclude": ["a"]}}, "all_numeric"),
    ({"name": "r", "features": ["a"], "impute": "mean"}, "impute must be"),
])
def test_retrain_feature_and_impute_validation(item, msg):
    with pytest.raises(B.BacktestError, match=msg):
        B.validate(spec(retrain=[item]))
    B.validate(spec(retrain=[{"name": "ok", "features": {"all_numeric": True, "exclude": ["a"]}, "impute": "zero"}]))


def test_env_lock_gives_a_retrain_only_spec_a_pinned_environment(lab, tmp_path):
    runner, reg, _ = lab
    (tmp_path / "specs").mkdir(); (tmp_path / "specs" / "req.lock").write_text("numpy==2.4.0\n")
    s = spec(models=[], columns=[{"name": "prod", "column": "prod_prob"}], env="lock", env_lock="req.lock",
             retrain=[{"name": "rt", "features": ["f0"]}])
    prep = B.prepare(s, reg, "feat.parquet", base=tmp_path / "specs")
    assert (prep.workdir / "requirements.lock").read_text() == "numpy==2.4.0\n" and prep.job_spec.lock == "requirements.lock"
    with pytest.raises(B.BacktestError, match="env_lock not found"):
        B.prepare({**s, "env_lock": "missing.lock"}, reg, "feat.parquet", base=tmp_path / "specs")
    with pytest.raises(B.BacktestError, match="env_lock: FILE"):
        B.prepare({k: v for k, v in s.items() if k != "env_lock"}, reg, "feat.parquet", base=tmp_path / "specs")


@pytest.mark.parametrize("seeds, family, ok", [([1, 2, 3], "xgboost", True), ([], "xgboost", False), ("1,2", "xgboost", False),
                                              ([1.5], "xgboost", False), (list(range(21)), "xgboost", False), ([1, 2], "logreg", False)])
def test_seeds_validation(seeds, family, ok):
    item = {"name": "r", "features": ["a"], "family": family, "seeds": seeds}
    if ok:
        B.validate(spec(retrain=[item]))
    else:
        with pytest.raises(B.BacktestError, match="seeds|deterministic"):
            B.validate(spec(retrain=[item]))


def test_bad_exclude_pattern_is_rejected_up_front():
    with pytest.raises(B.BacktestError, match="bad exclude_pattern"):
        B.validate(spec(retrain=[{"name": "r", "features": {"all_numeric": True, "exclude_pattern": "([unclosed"}}]))
    B.validate(spec(retrain=[{"name": "r", "features": {"all_numeric": True, "exclude_pattern": "epa|success"}}]))


@pytest.mark.parametrize("flt, ok", [({"column": "division", "equals": "fbs"}, True), ({"column": "division", "in": ["fbs", "fcs"]}, True),
                                    ({"column": "division"}, False), ({"equals": "x"}, False), ({"column": "d", "equals": 1, "in": [1]}, False),
                                    ({"column": "d", "in": "fbs"}, False), ("division=fbs", False)])
def test_filter_validation(flt, ok):
    if ok:
        B.validate(spec(filter=flt))
    else:
        with pytest.raises(B.BacktestError, match="filter must be"):
            B.validate(spec(filter=flt))


@pytest.mark.parametrize("tune, msg", [
    ({"space": {}}, "non-empty space"), ("x", "non-empty space"), ({"space": {"a": {"low": 1}}}, "list of choices or"),
    ({"space": {"a": {"low": 5, "high": 1}}}, "list of choices or"), ({"space": {"a": {"low": 0, "high": 1, "log": True}}}, "log scale needs low > 0"),
    ({"space": {"a": [1, 2]}, "trials": 0}, "trials must be"), ({"space": {"a": [1, 2]}, "trials": 999}, "trials must be"),
])
def test_tune_validation(tune, msg):
    item = {"name": "r", "features": ["a"], "walk_forward": {"group": "season", "min_groups": 5}, "tune": tune}
    with pytest.raises(B.BacktestError, match=msg):
        B.validate(spec(retrain=[item]))


def test_tune_needs_group_mode_and_logreg_only_tunes_c():
    good = {"space": {"max_depth": {"low": 1, "high": 4, "type": "int"}}, "trials": 5}
    B.validate(spec(retrain=[{"name": "r", "features": ["a"], "walk_forward": {"group": "season", "min_groups": 5}, "tune": good}]))
    with pytest.raises(B.BacktestError, match="walk_forward.group"):
        B.validate(spec(retrain=[{"name": "r", "features": ["a"], "walk_forward": {"min_train": 5}, "tune": good}]))
    with pytest.raises(B.BacktestError, match="logreg can only tune C"):
        B.validate(spec(retrain=[{"name": "r", "features": ["a"], "family": "logreg", "walk_forward": {"group": "season"}, "tune": good}]))
