import json
from types import SimpleNamespace

import pytest

from mllab import schema as S
from mllab import weekly as W


def report(table, ok=True, msg="boom"):
    return S.Report(table, [] if ok else [S.Finding("error", "nulls", msg)])


def run(tmp_path, cfg, *, check=None, submit=None, status=None, verdict=None):
    return W.run_weekly(cfg, root=tmp_path, base=tmp_path, sleep=lambda s: None,
                        check_fn=check or (lambda p: report(p.stem)), submit_fn=submit or (lambda p: "job-" + p.stem),
                        status_fn=status or (lambda j: "finished"), verdict_fn=verdict or (lambda j: "NOT distinguishable from the home prior"))


def test_all_green_run_is_recorded(tmp_path):
    res = run(tmp_path, {"bq_checks": ["a.yaml", "b.yaml"], "backtests": ["bt.yaml"]})
    assert res["ok"] and res["errors"] == [] and len(res["checks"]) == 2 and res["backtests"][0]["state"] == "finished"
    assert json.loads(W.result_path(tmp_path).read_text())["ok"] is True


def test_a_failed_data_check_fails_the_run_with_the_reason(tmp_path):
    res = run(tmp_path, {"bq_checks": ["good.yaml", "bad.yaml"]}, check=lambda p: report(p.stem, ok=p.stem == "good", msg="game_time_utc: 30% null"))
    assert not res["ok"] and "data check failed: bad" in res["errors"][0] and "30% null" in res["errors"][0]


def test_one_broken_spec_does_not_stop_the_others(tmp_path):
    def check(p):
        if p.stem == "broken":
            raise FileNotFoundError("no such spec")
        return report(p.stem)
    res = run(tmp_path, {"bq_checks": ["broken.yaml", "fine.yaml"]}, check=check)
    assert [c["ok"] for c in res["checks"]] == [False, True] and not res["ok"] and "could not run: broken.yaml" in res["errors"][0]


def test_verdict_change_is_recorded_but_is_not_an_incident(tmp_path):
    run(tmp_path, {"backtests": ["bt.yaml"]}, verdict=lambda j: "NOT distinguishable from the home prior")
    res = run(tmp_path, {"backtests": ["bt.yaml"]}, verdict=lambda j: "beats the home prior")
    b = res["backtests"][0]
    assert b["changed_vs_last"] and b["previous_verdict"].startswith("NOT distinguishable") and res["ok"]  # information, not a failure
    assert "[CHANGED from" in W.render(res)


def test_previous_result_is_kept_and_first_run_has_no_change_flag(tmp_path):
    first = run(tmp_path, {"backtests": ["bt.yaml"]})
    assert first["backtests"][0]["changed_vs_last"] is False and not (tmp_path / "audit" / "weekly.prev.json").exists()
    run(tmp_path, {"backtests": ["bt.yaml"]})
    assert (tmp_path / "audit" / "weekly.prev.json").exists()


def test_a_backtest_that_fails_or_times_out_is_an_error(tmp_path):
    assert "ended failed" in run(tmp_path, {"backtests": ["bt.yaml"]}, status=lambda j: "failed")["errors"][0]
    res = run(tmp_path, {"backtests": ["bt.yaml"], "backtest_timeout_s": 10}, status=lambda j: "running")  # never finishes
    assert not res["ok"] and "ended running" in res["errors"][0]


def test_submit_exception_is_captured(tmp_path):
    def boom(p):
        raise RuntimeError("spool unavailable")
    res = run(tmp_path, {"backtests": ["bt.yaml"]}, submit=boom)
    assert res["backtests"][0]["state"] == "error" and not res["ok"]


@pytest.mark.parametrize("cfg, msg", [({}, "nothing to run"), ({"bq_checks": "x"}, "list of spec"), ({"bq_checks": [1]}, "list of spec"),
                                     ({"bq_checks": ["a"], "bogus": 1}, "unknown config key")])
def test_config_validation(tmp_path, cfg, msg):
    import yaml
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(W.WeeklyError, match=msg):
        W.load_config(p)


def test_shipped_weekly_config_is_valid():
    from pathlib import Path
    cfg = W.load_config(Path(__file__).resolve().parents[1] / "specs" / "weekly.yaml")
    root = Path(__file__).resolve().parents[1]
    assert all((root / s).exists() for s in cfg["bq_checks"] + cfg["backtests"])  # every referenced spec exists
