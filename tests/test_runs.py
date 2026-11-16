import json
import sys

import pytest

from mllab import runs as R
from mllab.jobs import DirectLauncher, LocalRunner
from mllab.registry import Registry, RegistryError


def paired(a, b, ll, ci):
    return {f"{a} - {b}": {"accuracy_diff": 0.0, "accuracy_diff_ci": [0, 0], "log_loss_diff": ll, "log_loss_diff_ci": ci}}


def make_backtest(runner, jid, name, *, best_ll, home_ll=0.6935, ci=(-0.01, 0.002), sha="abc", submitted="2026-09-29T10:00:00+00:00"):
    d = runner.jobs / jid; (d / "code" / "outputs").mkdir(parents=True)
    (d / "spec.json").write_text(json.dumps({"name": name, "submitted_at": submitted, "outputs": "outputs"}))
    (d / "status.json").write_text('{"state": "finished"}')
    (d / "code" / "spec.json").write_text(json.dumps({"models": [{"name": "v10", "sha256": sha}]}))
    m = lambda ll, acc: {"accuracy": acc, "log_loss": ll, "n": 100}
    res = {"games_compared": 1957, "date_range": ["2026-05-01", "2026-09-27"],
           "metrics": {"v10": m(best_ll, 0.54), "home_prior": m(home_ll, 0.524)},
           "bootstrap": {"paired": paired("home_prior", "v10", home_ll - best_ll, [-ci[1], -ci[0]]), "accuracy_ci": {"v10": [0.518, 0.562]}}}
    (d / "code" / "outputs" / "results.json").write_text(json.dumps(res))


@pytest.fixture
def lab(tmp_path):
    return LocalRunner(tmp_path / "lab", launcher=DirectLauncher(), python_bin=sys.executable), Registry(tmp_path / "reg"), tmp_path


def test_verdict_says_noise_when_ci_spans_zero(lab):
    runner, *_ = lab
    make_backtest(runner, "20260929100000-aaaaaa", "sweep", best_ll=0.6888, ci=(-0.0125, 0.0031))
    r = R.list_runs(runner)[0]
    assert r["best"] == "v10" and "NOT distinguishable" in r["verdict"]
    assert r["vs_home_prior"]["ci"] == pytest.approx([-0.0125, 0.0031])  # oriented as best - home_prior


def test_verdict_credits_a_real_win_only_when_ci_excludes_zero(lab):
    runner, *_ = lab
    make_backtest(runner, "20260929100000-aaaaaa", "win", best_ll=0.660, ci=(-0.05, -0.02))
    assert "beats the home prior" in R.list_runs(runner)[0]["verdict"]


def test_home_prior_winning_is_reported_plainly(lab):
    runner, *_ = lab
    make_backtest(runner, "20260929100000-aaaaaa", "bad", best_ll=0.71, home_ll=0.6935)
    assert "always-home prior is the best" in R.list_runs(runner)[0]["verdict"]


def test_runs_are_never_ranked_across_runs_and_sorted_newest_first(lab):
    runner, *_ = lab
    make_backtest(runner, "20260929100000-aaaaaa", "old", best_ll=0.60, submitted="2026-09-29T09:00:00+00:00")
    make_backtest(runner, "20260929110000-bbbbbb", "new", best_ll=0.69, submitted="2026-09-29T11:00:00+00:00")
    rows = R.list_runs(runner)
    assert [r["name"] for r in rows] == ["new", "old"]  # by time, not by log loss
    assert "only comparable within a run" in R.render_runs(rows)


def test_plain_jobs_and_corrupt_files_are_skipped_not_fatal(lab):
    runner, *_ = lab
    (runner.jobs / "20260929120000-cccccc").mkdir(parents=True)  # a job with nothing in it
    d = runner.jobs / "20260929130000-dddddd"; (d / "code" / "outputs").mkdir(parents=True)
    (d / "spec.json").write_text("{not json"); (d / "code" / "outputs" / "results.json").write_text("garbage")
    make_backtest(runner, "20260929100000-aaaaaa", "ok", best_ll=0.69)
    assert [r["name"] for r in R.list_runs(runner)] == ["ok"]
    assert R.render_runs([]) == "no backtest or training runs yet"


def test_training_run_is_labelled_holdout_only(lab):
    runner, *_ = lab
    d = runner.jobs / "20260929100000-eeeeee"; (d / "code" / "outputs").mkdir(parents=True)
    (d / "spec.json").write_text(json.dumps({"name": "cand", "submitted_at": "2026-09-29T10:00:00+00:00", "outputs": "outputs"}))
    (d / "code" / "outputs" / "metrics.json").write_text(json.dumps({"train_rows": 500, "holdout_games_compared": 360,
                                                                       "holdout_range": ["2026-09-01", "2026-09-27"], "holdout": {"cand": {}, "prod": {}}}))
    r = R.list_runs(runner)[0]
    assert r["kind"] == "train" and "run a backtest for confidence intervals" in r["verdict"]


def register(reg, tmp, version, parent=None, data=b"x", **kw):
    a = tmp / f"{version}.pkl"; a.write_bytes(data)
    return reg.register("p", "game", version, a, parent=parent, features=["a", "b"], **kw)


def test_card_shows_lineage_channels_and_matching_backtests(lab):
    runner, reg, tmp = lab
    m1 = register(reg, tmp, "v1", data=b"one"); m2 = register(reg, tmp, "v2", parent="v1", data=b"two")
    reg.promote("p", "game", "v2")
    make_backtest(runner, "20260929100000-aaaaaa", "sweep", best_ll=0.6888, sha=m2.sha256)
    make_backtest(runner, "20260929110000-bbbbbb", "other", best_ll=0.69, sha="not-this-model")
    text = R.card(reg, runner, "p", "game", "v2")
    assert "v2 <- v1" in text and "channels pointing here: prod" in text
    assert "20260929100000-aaaaaa" in text and "20260929110000-bbbbbb" not in text  # matched by artifact hash, not by name
    assert "not directly comparable" in text
    assert "No backtest has included this artifact yet" in R.card(reg, runner, "p", "game", "v1")


def test_saved_card_does_not_become_a_version(lab):
    """Regression: cards/ lives inside the model dir and versions() used to treat every subfolder as a version."""
    runner, reg, tmp = lab
    register(reg, tmp, "v1")
    path = R.save_card(reg, runner, "p", "game", "v1")
    assert path.name == "v1.md" and path.parent.name == "cards"
    assert reg.versions("p", "game") == ["v1"]
    with pytest.raises(RegistryError, match="reserved"):
        register(reg, tmp, "cards")


def test_card_lists_deployments_honestly_and_recovers_feature_count(lab):
    runner, reg, tmp = lab
    m = register(reg, tmp, "v1", data=b"one")
    reg.record_deploy("p", "game", {"target": "cloudfn-model-swap", "version": "v1", "object_name": "models/_selftest/model.pkl", "ts": "2026-09-29T15:53:00+00:00"})
    reg.record_deploy("p", "game", {"target": "cloudfn-rollback", "object_name": "models/_selftest/model.pkl", "undid_version": "v1", "ts": "2026-09-29T15:55:00+00:00"})
    make_backtest(runner, "20260929100000-aaaaaa", "sweep", best_ll=0.69, sha=m.sha256)
    res = json.loads((runner.jobs / "20260929100000-aaaaaa/code/outputs/results.json").read_text())
    res["info"] = {"v10": {"n_features": 73}}
    (runner.jobs / "20260929100000-aaaaaa/code/outputs/results.json").write_text(json.dumps(res))
    reg2 = Registry(tmp / "reg2"); a = tmp / "nofeat.pkl"; a.write_bytes(b"one"); reg2.register("p", "game", "v1", a)  # same bytes, no features recorded
    text = R.card(reg2, runner, "p", "game", "v1")
    assert "- features: 73 (found by inspecting the artifact" in text  # exact: once printed "073"
    text = R.card(reg, runner, "p", "game", "v1")
    assert "since rolled back" in text and "a rehearsal, not a live deploy" in text and "rollback of `models/_selftest/model.pkl`" in text


def test_verdict_also_compares_against_column_baselines_like_elo(lab):
    runner, *_ = lab
    make_backtest(runner, "20260929100000-aaaaaa", "nfl", best_ll=0.6431, home_ll=0.6891, ci=(-0.06, -0.03))
    p = runner.jobs / "20260929100000-aaaaaa/code/outputs/results.json"
    res = json.loads(p.read_text())
    res["metrics"]["elo"] = {"accuracy": 0.637, "log_loss": 0.6462, "n": 100}
    res["info"] = {"elo": {"kind": "column"}, "v10": {"kind": "retrain/xgboost"}}
    res["bootstrap"]["paired"].update(paired("elo", "v10", 0.0031, [-0.0063, 0.0126]))  # v10 - elo = -0.0031, CI spans 0
    p.write_text(json.dumps(res))
    v = R.list_runs(runner)[0]["verdict"]
    assert v.startswith("beats the home prior") and "vs elo: not distinguishable from" in v  # beats the low bar, not the real one


def test_a_clearly_worse_baseline_comparison_is_called_out(lab):
    runner, *_ = lab
    make_backtest(runner, "20260929100000-aaaaaa", "x", best_ll=0.69)
    p = runner.jobs / "20260929100000-aaaaaa/code/outputs/results.json"
    res = json.loads(p.read_text())
    res["metrics"]["elo"] = {"accuracy": 0.5, "log_loss": 0.72, "n": 100}
    res["info"] = {"elo": {"kind": "column"}}
    res["bootstrap"]["paired"].update(paired("v10", "elo", -0.03, [-0.04, -0.02]))  # v10 better than elo, CI excludes 0
    p.write_text(json.dumps(res))
    assert "vs elo: beats" in R.list_runs(runner)[0]["verdict"]
