"""The job table and the cron -> systemd conversion. A schedule that is silently wrong would not show until the 6 AM it should have run."""
import pytest

from mllab.pipeline import jobs as J


@pytest.mark.parametrize("cron, want", [
    ("0 4 * 3-11 *", "*-03..11-* 04:00:00 America/New_York"),
    ("30 6 * 3-11 1", "Mon *-03..11-* 06:30:00 America/New_York"),
    ("0 6 * 9-12,1,2 2", "Tue *-01..02,09..12-* 06:00:00 America/New_York"),
    ("30 7 * 9-12,1 2", "Tue *-01,09..12-* 07:30:00 America/New_York"),
    ("0 8 * 8-12,1 0", "Sun *-01,08..12-* 08:00:00 America/New_York"),
    ("0 6 * * 5", "Fri *-*-* 06:00:00 America/New_York"),
    ("5 9 * * *", "*-*-* 09:05:00 America/New_York"),
    ("0 6 * 1-12 0,6", "Sun,Sat *-*-* 06:00:00 America/New_York"),
])
def test_cron_to_oncalendar(cron, want):
    assert J.cron_to_oncalendar(cron) == want


@pytest.mark.parametrize("bad", ["0 4 * 3-11", "*/5 * * * *", "0 4 15 * *", "0 4,16 * * *", "0 4 * 13 *", "0 4 * * 7", "x y * * *"])
def test_unsupported_or_malformed_cron_is_refused_not_guessed(bad):
    with pytest.raises(ValueError):
        J.cron_to_oncalendar(bad)


def test_every_job_converts_and_names_a_real_bundle():
    for j in J.JOBS.values():
        assert j.bundle in J.BUNDLES
        assert j.cron and J.cron_to_oncalendar(j.cron).endswith(J.TZ)
        assert "mode" in j.body


def test_job_names_are_cloud_schedulers_so_cutover_is_one_for_one():
    assert set(J.JOBS) >= {"mlb-2026-daily", "nfl-weekly-ingest", "cfb-weekly-predict", "cfb-weekly-season-sim"}
    assert "mlb-2026-pregame-schedule" not in J.JOBS  # per-game scheduling is pregame.py, not a timer job


def test_paused_jobs_are_known_but_never_installed():
    paused = {j.name for j in J.JOBS.values() if j.paused}
    assert paused == {"mlb-2026-weekly-train-v10", "mlb-2026-weekly-predict"}
    assert not paused & {j.name for j in J.installed_jobs()}
    assert len(J.installed_jobs()) == 12


def test_dry_ok_matches_which_football_modes_actually_refuse_dry_run():
    # nfl: dry_run only for predict_week and season_sim. cfb: predict_week, predict_next, drives, season_sim. Others return 400.
    no_dry = {j.name for j in J.JOBS.values() if not j.dry_ok}
    assert no_dry == {"nfl-weekly-ingest", "nfl-weekly-score", "cfb-weekly-ingest", "cfb-weekly-cfbd"}
    for j in J.JOBS.values():
        if j.dry_ok:
            assert j.body["mode"] in ("daily", "rosters", "power_rankings", "predict_week", "season_sim", "drives", "predict_next", "train_weekly", "predict")


def test_bodies_are_exactly_what_scheduler_sends():
    assert J.JOBS["mlb-2026-daily"].body == {"mode": "daily"}
    assert J.JOBS["nfl-weekly-predict"].body == {"mode": "predict_week"}
    assert J.JOBS["cfb-weekly-predict"].body == {"mode": "predict_next"}
    assert J.JOBS["mlb-2026-weekly-train-v10"].body == {"mode": "train_weekly", "model_version": "v10"}


def test_bundles_carry_the_configuration_the_functions_have_in_production():
    assert J.BUNDLES["mlb"].entry == "cloud_function_main:daily_pipeline"
    assert J.BUNDLES["nfl"].env["NFL_DATASET"] == "nfl_season" and J.BUNDLES["nfl"].env["NFL_RIDGE_SHADOW"] == "1"
    # the regularized-XGBoost shadow is on for NFL only: it is an NFL model, and its table is created by its first run
    assert J.BUNDLES["nfl"].env["NFL_XGB_REG_SHADOW"] == "1" and "NFL_XGB_REG_SHADOW" not in J.BUNDLES["cfb"].env and "NFL_XGB_REG_SHADOW" not in J.BUNDLES["mlb"].env
    assert J.BUNDLES["cfb"].env["CFB_HIST_DATASET"] == "cfb_historical"
    assert J.BUNDLES["cfb"].secrets == {"CFBD_API_KEY": "sm://projects/hankstank/secrets/cfbd-api-key/versions/latest"}
    assert not J.BUNDLES["mlb"].secrets and not J.BUNDLES["nfl"].secrets


def test_no_bundle_sets_k_service():
    # pa_sim.memory_ok() reads K_SERVICE as "inside a managed runtime with an unknown limit" and then refuses to run
    for b in J.BUNDLES.values():
        assert "K_SERVICE" not in b.env


def test_thread_caps_cover_every_blas_backend():
    assert set(J.THREAD_ENV) == {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"}
    assert set(J.THREAD_ENV.values()) == {"1"}
