"""The generated systemd units: current with the job table, and carrying the properties that matter."""
import re
from pathlib import Path

import pytest

from mllab.pipeline import jobs as J
from mllab.pipeline import units as U

COMMITTED = Path(__file__).resolve().parents[1] / "host" / "pipeline" / "units"


def test_committed_units_are_current():
    assert U.stale(COMMITTED) == [], "run: python -m mllab.pipeline.units write host/pipeline/units"


def test_every_installed_job_has_exactly_one_timer_and_paused_jobs_have_none():
    files = U.generate()
    for j in J.installed_jobs():
        assert f"pipeline-{j.name}.timer" in files
    for j in J.JOBS.values():
        if j.paused:
            assert f"pipeline-{j.name}.timer" not in files


def test_each_timer_fires_the_calendar_derived_from_its_cron_and_starts_the_template():
    files = U.generate()
    for j in J.installed_jobs():
        t = files[f"pipeline-{j.name}.timer"]
        assert f"OnCalendar={J.cron_to_oncalendar(j.cron)}\n" in t
        assert f"Unit=pipeline-job@{j.name}.service" in t and "Persistent=true" in t and "WantedBy=timers.target" in t


def test_every_timer_points_at_a_unit_that_exists():
    files = U.generate()
    for name, text in files.items():
        if name.endswith(".timer"):
            unit = re.search(r"^Unit=(.+)$", text, re.M).group(1)
            assert unit in files or "@" in unit and unit.split("@")[0] + "@.service" in files, (name, unit)


@pytest.mark.parametrize("name", ["pipeline-job@.service", "pipeline-pregame-plan.service", "pipeline-pregame-tick.service"])
def test_services_are_unprivileged_hardened_and_capped(name):
    s = U.generate()[name]
    for line in ("User=pipeline", "Group=pipeline", "Slice=pipeline.slice", "NoNewPrivileges=yes", "ProtectSystem=strict", "PrivateTmp=yes", "ProtectHome=yes",
                 "ReadWritePaths=/var/lib/pipeline", "EnvironmentFile=/etc/pipeline/env", "Type=oneshot"):
        assert line in s, (name, line)
    assert re.search(r"^MemoryMax=\d", s, re.M)
    assert "OMP_NUM_THREADS=1" in s and "OPENBLAS_NUM_THREADS=1" in s and "MKL_NUM_THREADS=1" in s
    assert "K_SERVICE" not in s


@pytest.mark.parametrize("name", ["pipeline-job@.service", "pipeline-pregame-plan.service", "pipeline-pregame-tick.service"])
def test_oneshot_services_use_timeoutstartsec_because_runtimemaxsec_is_ignored_for_them(name):
    s = U.generate()[name]
    assert re.search(r"^TimeoutStartSec=", s, re.M) and "RuntimeMaxSec" not in s


def test_nothing_runs_without_an_installed_release():
    for name in ("pipeline-job@.service", "pipeline-pregame-plan.service", "pipeline-pregame-tick.service"):
        assert "ConditionPathExists=/opt/pipeline/current/RELEASE.json" in U.generate()[name]


def test_the_runner_comes_from_the_pipelines_own_lib_not_mllabs_install():
    s = U.generate()["pipeline-job@.service"]
    assert "PYTHONPATH=/opt/pipeline/lib" in s and "/opt/mllab" not in s
    assert "ExecStart=/opt/pipeline/venv/bin/python -m mllab.pipeline.runner run %i" in s


def test_the_ticker_gets_the_big_share_of_memory_because_sim_blend_peaks_at_gigabytes():
    gb = lambda name: int(re.search(r"^MemoryMax=(\d+)G", U.generate()[name], re.M).group(1))
    assert gb("pipeline-pregame-tick.service") >= 4 > gb("pipeline-job@.service") - 1
    assert "MemoryMax=6G" in U.generate()["pipeline.slice"]


def test_the_ticker_timer_is_every_minute_in_season_and_not_persistent():
    t = U.generate()["pipeline-pregame-tick.timer"]
    assert "OnCalendar=*-03..11-* *:*:00 America/New_York" in t and "Persistent=false" in t


def test_write_removes_stale_files(tmp_path):
    (tmp_path / "pipeline-old.timer").write_text("x")
    U.write(tmp_path)
    assert not (tmp_path / "pipeline-old.timer").exists() and U.stale(tmp_path) == []


# ---------------------------------------------------------------------------------------------------------------------------------- broker
def test_the_broker_units_agree_with_the_broker_code():
    from mllab.pipeline import broker
    files = U.generate()
    # the path unit watches exactly the files the broker accepts, in the directory it reads
    assert f"PathExistsGlob={broker.SPOOL}/pipe-*.json" in files["pipeline-broker-apply.path"]
    assert broker.REQUEST_RE.match("pipe-abcd1234-0000.json") and not broker.REQUEST_RE.match(".pipe-abcd1234-0000.tmp")
    # the snapshot may write only where the app reads it
    assert f"ReadWritePaths=/var/lib/pipeline {broker.STATUS_DIR}" in files["pipeline-broker-snapshot.service"]


def test_only_the_apply_unit_is_root_and_it_is_hardened():
    files = U.generate()
    assert "User=root" in files["pipeline-broker-apply.service"] and "User=pipeline" in files["pipeline-broker-snapshot.service"]
    apply = files["pipeline-broker-apply.service"]
    for line in ("NoNewPrivileges=yes", "ProtectSystem=strict", "ProtectHome=yes", "PrivateTmp=yes"):
        assert line in apply
    # it may write the mode files and the app's own tree, nothing else
    assert "ReadWritePaths=/etc/pipeline/jobs.d /mnt/appdata/mllab" in apply
    assert not any(n.endswith(".service") and "User=root" in t for n, t in files.items() if n != "pipeline-broker-apply.service")


def test_the_status_snapshot_refreshes_often_enough_to_be_called_current():
    t = U.generate()["pipeline-broker-snapshot.timer"]
    assert "OnUnitActiveSec=30s" in t and "Persistent" not in t
