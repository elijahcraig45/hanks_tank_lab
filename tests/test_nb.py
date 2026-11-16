"""The notebook helpers: they must read what the app reads, copy specs so they run from anywhere, and never raise on an empty lab."""
import json

import pytest
import yaml

from mllab import nb
from mllab.pipeline import broker, runner as prunner
from mllab.registry import Registry


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("MLLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("MLLAB_REGISTRY", raising=False)
    monkeypatch.delenv("BROKER_STATUS_DIR", raising=False)
    monkeypatch.setenv("MLLAB_SPECS", str(tmp_path / "shipped"))
    for d in ("registry", "jobs", "work", "pipeline-status", "shipped", "pj", "ps"):
        (tmp_path / d).mkdir()
    return tmp_path


def publish(lab, **modes):
    for name, mode in modes.items():
        prunner.write_mode(name, mode, lab / "pj")
    broker.cmd_snapshot(jobs_dir=lab / "pj", state_dir=lab / "ps", release=lab / "none", show=lambda u: {}, status_dir=lab / "pipeline-status")


def test_an_empty_lab_gives_empty_frames_not_errors(lab, capsys):
    assert len(nb.models()) == 0 and len(nb.experiments()) == 0 and len(nb.jobs()) == 0
    assert len(nb.pipeline()) == 0 and "not publishing" in capsys.readouterr().out
    nb.status()  # must not raise


def test_pipeline_reads_the_broker_snapshot(lab):
    publish(lab, **{"nfl-weekly-predict": "dry", "mlb-2026-daily": "live"})
    df = nb.pipeline().set_index("job")
    assert df.loc["nfl-weekly-predict", "mode"] == "dry" and df.loc["mlb-2026-daily", "mode"] == "live"
    assert df.loc["mlb-2026-weekly-train-v10", "mode"] == "paused" and "pregame" in df.index


def test_models_lists_versions_with_their_channels(lab):
    art = lab / "m.pkl"
    art.write_bytes(b"m")
    reg = Registry(lab / "registry")
    reg.register("hankstank", "game_outcome", "v10", art, metrics={"acc": 0.5})
    reg.promote("hankstank", "game_outcome", "v10", "prod")
    df = nb.models()
    assert list(df["version"]) == ["v10"] and df.iloc[0]["channels"] == "prod" and df.iloc[0]["metrics"] == {"acc": 0.5}


def test_status_summarises_without_changing_anything(lab, capsys):
    publish(lab, **{"nfl-weekly-predict": "dry"})
    nb.status()
    out = capsys.readouterr().out
    assert "PIPELINE (live / dry / off): 0 / 1 / " in out and "/app/" in out


def test_specs_are_classified_and_described(lab):
    (lab / "shipped" / "bt.yaml").write_text("# compare two models\nname: bt\nmodels: []\nretrain: []\n")
    (lab / "shipped" / "tr.yaml").write_text("name: tr\nfamily: xgboost\n")
    (lab / "shipped" / "chk.yaml").write_text("table: a.b\n")
    df = nb.specs().set_index("spec")
    assert df.loc["bt", "kind"] == "backtest" and df.loc["bt", "about"] == "compare two models"
    assert df.loc["tr", "kind"] == "train" and df.loc["chk", "kind"] == "check"


def test_new_spec_copies_and_makes_the_sql_path_absolute(lab):
    sub = lab / "shipped" / "specs"
    sub.mkdir()
    (lab / "sql").mkdir()
    (lab / "sql" / "q.sql").write_text("select 1")
    (sub / "base.yaml").write_text("name: base\ndata: {sql_file: ../../sql/q.sql}\nfamily: xgboost\n")
    import os
    os.environ["MLLAB_SPECS"] = str(sub)
    dest = nb.new_spec("base", "mine")
    got = yaml.safe_load(dest.read_text())
    assert got["name"] == "mine" and got["data"]["sql_file"] == str((lab / "sql" / "q.sql").resolve())
    assert dest == lab / "work" / "specs" / "mine.yaml"
    with pytest.raises(FileExistsError):
        nb.new_spec("base", "mine")  # never overwrites your edits
    with pytest.raises(FileNotFoundError):
        nb.new_spec("nope", "x")


def test_the_cli_runs_in_process_and_a_usage_error_does_not_kill_the_kernel(lab):
    rc, text = nb.capture("runs")
    assert rc == 0
    rc, text = nb.capture("definitely-not-a-command")
    assert rc != 0 and "invalid choice" in text


def test_wait_follows_a_job_to_its_end_and_reports_each_change(lab):
    jd = lab / "jobs" / "abcd1234-0001"
    jd.mkdir()
    (jd / "spec.json").write_text(json.dumps({"name": "x", "submitted_at": "2026-01-01T00:00:00+00:00"}))
    (jd / "started.json").write_text(json.dumps({"state": "running"}))
    seen, sleeps = [], []

    def sleep(_):
        sleeps.append(1)
        (jd / "status.json").write_text(json.dumps({"state": "finished"}))

    class Alive:  # the launcher says the unit is still running until status.json appears
        def is_active(self, _): return not (jd / "status.json").exists()
    import mllab.nb as m
    orig = m.runner
    m.runner = lambda: type(orig())(lab, launcher=Alive())
    try:
        assert nb.wait("abcd1234-0001", every=0, show=seen.append, sleep=sleep) == "finished"
    finally:
        m.runner = orig
    assert seen == ["abcd1234-0001: running", "abcd1234-0001: finished"]


def test_wait_gives_up_politely_and_leaves_the_job_running(lab):
    jd = lab / "jobs" / "abcd1234-0002"
    jd.mkdir()
    (jd / "spec.json").write_text(json.dumps({"name": "x", "submitted_at": "2026-01-01T00:00:00+00:00"}))
    (jd / "started.json").write_text("{}")
    class Alive:
        def is_active(self, _): return True
    import mllab.nb as m
    orig = m.runner
    m.runner = lambda: type(orig())(lab, launcher=Alive())
    ticks = iter(range(0, 1000, 100))
    try:
        seen = []
        st = nb.wait("abcd1234-0002", every=0, timeout=250, show=seen.append, sleep=lambda _: None, clock=lambda: next(ticks))
    finally:
        m.runner = orig
    assert st == "running" and "keeps running" in seen[-1]


def test_submit_returns_the_job_id_and_refuses_clearly(lab, monkeypatch):
    calls = []
    monkeypatch.setattr(nb, "capture", lambda *a: (calls.append(a), (0, "job-abc123-xyz\nextra\n"))[1])
    assert nb.submit("backtest", "s.yaml", max_gb=2) == "job-abc123-xyz"
    assert calls[0] == ("backtest", "run", "s.yaml", "--max-gb", "2")
    monkeypatch.setattr(nb, "capture", lambda *a: (1, "error: spec has no name"))
    with pytest.raises(RuntimeError, match="spec has no name"):
        nb.submit("train", "s.yaml")
    with pytest.raises(ValueError):
        nb.submit("deploy", "s.yaml")


def test_report_prints_the_report_or_explains_its_absence(lab, capsys):
    jd = lab / "jobs" / "abcd1234-0003"
    (jd / "code" / "outputs").mkdir(parents=True)
    (jd / "spec.json").write_text(json.dumps({"name": "x", "submitted_at": "2026-01-01T00:00:00+00:00", "outputs": "outputs"}))
    (jd / "status.json").write_text(json.dumps({"state": "finished"}))
    (jd / "log.txt").write_text("all done\n")
    (jd / "code" / "outputs" / "report.md").write_text("# the report")
    nb.report("abcd1234-0003")
    assert "# the report" in capsys.readouterr().out
    (jd / "code" / "outputs" / "report.md").unlink()
    nb.report("abcd1234-0003")
    out = capsys.readouterr().out
    assert "No report" in out and "all done" in out


# ------------------------------------------------------------------------------------------------------------------------------- tuning
def real_specs(monkeypatch):
    from pathlib import Path
    monkeypatch.setenv("MLLAB_SPECS", str(Path(__file__).resolve().parents[1] / "specs"))


def test_a_copied_football_spec_keeps_its_environment_lock(lab, monkeypatch):
    real_specs(monkeypatch)
    raw = yaml.safe_load(nb.new_spec("nfl_walkforward", "mine").read_text())
    from pathlib import Path
    assert Path(raw["env_lock"]).is_absolute() and Path(raw["env_lock"]).is_file()  # a relative ../scripts path would break from work/specs


def test_tune_spec_adds_a_tuned_twin_and_keeps_the_original(lab, monkeypatch):
    real_specs(monkeypatch)
    path = nb.tune_spec("nfl_walkforward", "t1", "xgb_pure", trials=10, seeds=[1, 2])
    raw = yaml.safe_load(path.read_text())
    by = {c["name"]: c for c in raw["retrain"]}
    assert "xgb_pure" in by and "xgb_pure_tuned" in by and "tune" not in by["xgb_pure"]
    t = by["xgb_pure_tuned"]["tune"]
    assert t["trials"] == 10 and "learning_rate" in t["space"] and by["xgb_pure_tuned"]["seeds"] == [1, 2]
    assert by["xgb_pure_tuned"]["params"] == by["xgb_pure"]["params"]  # the search starts from, and is compared with, the same settings


def test_tune_spec_refuses_what_the_engine_cannot_do_and_leaves_nothing_behind(lab, monkeypatch):
    real_specs(monkeypatch)
    from mllab.backtest import BacktestError
    with pytest.raises(BacktestError, match="walk_forward.group"):
        nb.tune_spec("mlb_v10_param_sweep", "t2", "xgb_d2")  # MLB walks forward by rows, not seasons
    with pytest.raises(KeyError, match="not a retrain contender"):
        nb.tune_spec("nfl_walkforward", "t3", "nope")
    with pytest.raises(BacktestError, match="trials"):
        nb.tune_spec("nfl_walkforward", "t4", "xgb_pure", trials=500)
    assert not list((lab / "work" / "specs").glob("*.yaml"))  # a failed attempt must not block the next try with the same name
    nb.tune_spec("nfl_walkforward", "t2", "xgb_pure")  # the name is free again


def test_grid_spec_expands_the_grid_with_short_unique_valid_names(lab, monkeypatch):
    real_specs(monkeypatch)
    path = nb.grid_spec("mlb_v10_param_sweep", "g1", "xgb_d2", {"max_depth": [2, 3], "learning_rate": [0.03, 0.05]})
    raw = yaml.safe_load(path.read_text())
    names = [c["name"] for c in raw["retrain"]]
    assert {"xgb_d2_md2_lr0p03", "xgb_d2_md3_lr0p05"} <= set(names) and len(names) == len(set(names))
    assert all(len(n) <= 41 for n in names) and "xgb_d2" in names
    got = {c["name"]: c["params"] for c in raw["retrain"]}
    assert got["xgb_d2_md3_lr0p05"]["max_depth"] == 3 and got["xgb_d2_md3_lr0p05"]["learning_rate"] == 0.05
    assert got["xgb_d2_md3_lr0p05"]["n_estimators"] == got["xgb_d2"]["n_estimators"]  # only the grid's parameters change


def test_grid_spec_refuses_a_grid_that_is_mostly_luck_and_bad_input(lab, monkeypatch):
    real_specs(monkeypatch)
    with pytest.raises(ValueError, match="max_contenders"):
        nb.grid_spec("mlb_v10_param_sweep", "g2", "xgb_d2", {"max_depth": [2, 3, 4, 5], "learning_rate": [0.01, 0.02, 0.03, 0.05]})
    with pytest.raises(ValueError, match="at least one"):
        nb.grid_spec("mlb_v10_param_sweep", "g3", "xgb_d2", {})
    with pytest.raises(KeyError):
        nb.grid_spec("mlb_v10_param_sweep", "g4", "nope", {"max_depth": [2]})


def test_contenders_lists_what_a_spec_compares(lab, monkeypatch):
    real_specs(monkeypatch)
    df = nb.contenders(nb._find_spec("nfl_tune")).set_index("name")
    assert df.loc["xgb_tuned", "tuned"] and not df.loc["xgb_default", "tuned"] and df.loc["xgb_default", "params"]["max_depth"] == 3
    assert df.loc["elo", "kind"] == "baseline column"


RESULT = {"games_total": 10, "games_compared": 10, "date_range": ["2017-09-07", "2025-02-09"], "home_win_rate": 0.55,
          "metrics": {"xgb_default": {"n": 10, "accuracy": 0.64, "log_loss": 0.640, "brier": 0.22, "auc": 0.69},
                      "xgb_tuned": {"n": 10, "accuracy": 0.65, "log_loss": 0.636, "brier": 0.22, "auc": 0.69},
                      "elo": {"n": 10, "accuracy": 0.62, "log_loss": 0.650, "brier": 0.23, "auc": 0.66},
                      "home_prior": {"n": 10, "accuracy": 0.55, "log_loss": 0.690, "brier": 0.25, "auc": 0.5}},
          "info": {"xgb_tuned": {"kind": "retrain", "tuned": {"best_params": {"max_depth": 3}, "best_inner_log_loss": 0.634, "default_inner_log_loss": 0.653, "n_trials": 40,
                                                              "top_trials": [{"params": {"max_depth": 3}, "inner_log_loss": 0.634}, {"params": {"max_depth": 4}, "inner_log_loss": 0.636}],
                                                              "tuning_window": ["2006-09-07", "2017-02-05"]}}},
          "bootstrap": {"paired": {"xgb_tuned - elo": {"log_loss_diff": -0.014, "log_loss_diff_ci": [-0.03, -0.002]},
                                   "xgb_default - elo": {"log_loss_diff": -0.010, "log_loss_diff_ci": [-0.02, 0.001]},
                                   "xgb_default - xgb_tuned": {"log_loss_diff": 0.004, "log_loss_diff_ci": [-0.0003, 0.0127]},
                                   "elo - home_prior": {"log_loss_diff": -0.04, "log_loss_diff_ci": [-0.06, -0.02]}}}}


def finished_job(lab, res=RESULT, jid="abcd1234-0009"):
    jd = lab / "jobs" / jid
    (jd / "code" / "outputs").mkdir(parents=True)
    (jd / "spec.json").write_text(json.dumps({"name": "x", "submitted_at": "2026-01-01T00:00:00+00:00", "outputs": "outputs"}))
    (jd / "status.json").write_text(json.dumps({"state": "finished"}))
    (jd / "code" / "outputs" / "results.json").write_text(json.dumps(res))
    return jid


def test_results_gives_each_contender_an_honest_verdict_against_elo(lab):
    df = nb.results(finished_job(lab)).set_index("contender")
    assert list(nb.results(finished_job(lab, jid="abcd1234-0010"))["contender"])[0] == "xgb_tuned"  # best log loss first
    assert df.loc["xgb_tuned", "verdict"] == "better"           # CI below zero
    assert df.loc["xgb_default", "verdict"] == "not distinguishable"   # CI spans zero: the usual honest answer
    assert df.loc["elo", "verdict"] == "(reference)"
    assert df.attrs["games"] == 10


def test_results_reads_a_pair_stored_either_way_round_and_picks_vs(lab):
    df = nb.results(finished_job(lab), vs="xgb_default").set_index("contender")
    assert df.loc["xgb_tuned", "diff"] == pytest.approx(-0.004)       # stored as "default - tuned": flipped, with its interval
    assert (df.loc["xgb_tuned", "ci_low"], df.loc["xgb_tuned", "ci_high"]) == pytest.approx((-0.0127, 0.0003))
    with pytest.raises(KeyError, match="not in this backtest"):
        nb.results(finished_job(lab, jid="abcd1234-0011"), vs="nope")


def test_tuned_and_trials_expose_the_search(lab):
    jid = finished_job(lab)
    t = nb.tuned(jid)
    assert list(t["contender"]) == ["xgb_tuned"] and t.iloc[0]["inner_gain"] == pytest.approx(0.019) and t.iloc[0]["best_params"] == {"max_depth": 3}
    assert list(nb.trials(jid, "xgb_tuned")["max_depth"]) == [3, 4]
    with pytest.raises(KeyError, match="not tuned"):
        nb.trials(jid, "elo")


def test_results_of_an_unfinished_job_says_what_to_do(lab):
    jd = lab / "jobs" / "abcd1234-0012"
    jd.mkdir()
    (jd / "spec.json").write_text(json.dumps({"name": "x", "submitted_at": "2026-01-01T00:00:00+00:00"}))
    (jd / "started.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="no results.json"):
        nb.results("abcd1234-0012")  # whatever state it is in, including one that cannot even be looked up here (no systemctl on a laptop)
