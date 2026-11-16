"""Generate the systemd units from the job table, so the schedules cannot drift from the runner's idea of them.

    python -m mllab.pipeline.units write host/pipeline/units     (re)generate the committed files
    python -m mllab.pipeline.units check host/pipeline/units     exit 1 if the committed files are stale

Every timer is installed and enabled, and every job starts `off`: the units only decide *when* the runner is asked, and the mode file decides
whether it does anything. Cutting a job over is therefore `pipeline-ctl set <job> live`, never a systemd edit."""
from __future__ import annotations

import sys
from pathlib import Path

from .jobs import JOBS, TZ, installed_jobs, cron_to_oncalendar

LIB = "/opt/pipeline/lib"  # the runner's own copy of mllab.pipeline, so the pipeline does not depend on mllab's install or its permissions
PY = "/opt/pipeline/venv/bin/python"
# One environment for every unit. The thread caps are not tuning: with numpy's defaults the pipeline ran 19x slower on this CPU.
COMMON_ENV = (
    f"Environment=PYTHONPATH={LIB}\n"
    "Environment=PIPELINE_RELEASE=/opt/pipeline/current\n"
    "Environment=PYTHONDONTWRITEBYTECODE=1\n"
    "Environment=OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n"
)
HARDENING = (
    "NoNewPrivileges=yes\nProtectSystem=strict\nProtectHome=yes\nPrivateTmp=yes\nProtectKernelTunables=yes\nProtectControlGroups=yes\n"
    "RestrictSUIDSGID=yes\nReadWritePaths=/var/lib/pipeline\n"
)


STATUS_DIR = "/mnt/appdata/mllab/pipeline-status"  # where the broker publishes what the web app may show (pipeline:mllab 2750)
SPOOL_GLOB = "/mnt/appdata/mllab/spool/pipe-*.json"  # where the web app drops requests (written by mllab; consumed and deleted by the broker)


def _service(description: str, exec_start: str, *, memory: str, runtime_max: str, ident: str, extra_rw: str = "") -> str:
    # Type=oneshot ignores RuntimeMaxSec (systemd-analyze verify says so); TimeoutStartSec is what bounds a run that never exits.
    return f"""[Unit]
Description={description}
After=network-online.target
Wants=network-online.target
ConditionPathExists=/opt/pipeline/current/RELEASE.json

[Service]
Type=oneshot
User=pipeline
Group=pipeline
Slice=pipeline.slice
EnvironmentFile=/etc/pipeline/env
{COMMON_ENV}WorkingDirectory=/var/lib/pipeline/work
ExecStart={PY} -m {exec_start}
MemoryMax={memory}
TimeoutStartSec={runtime_max}
{HARDENING.rstrip()}{(" " + extra_rw) if extra_rw else ""}
SyslogIdentifier={ident}
"""


def _timer(description: str, *calendars: str, unit: str, persistent: bool = True) -> str:
    cal = "".join(f"OnCalendar={c}\n" for c in calendars)
    return f"""[Unit]
Description={description}

[Timer]
{cal}Persistent={'true' if persistent else 'false'}
AccuracySec=1s
Unit={unit}

[Install]
WantedBy=timers.target
"""


def generate() -> dict[str, str]:
    files: dict[str, str] = {}
    files["pipeline.slice"] = """[Unit]
Description=Pipeline jobs (one resource cap for all of them)

[Slice]
MemoryMax=6G
CPUQuota=400%
TasksMax=2048
"""
    # one template for the cron-shaped jobs: %i is the Cloud Scheduler job name
    files["pipeline-job@.service"] = _service("Pipeline job %i", "mllab.pipeline.runner run %i", memory="3G", runtime_max="3h", ident="pipeline-%i")
    for j in installed_jobs():
        spec = cron_to_oncalendar(j.cron)
        files[f"pipeline-{j.name}.timer"] = _timer(f"Pipeline timer: {j.name} (cron '{j.cron}' {TZ})", spec, unit=f"pipeline-job@{j.name}.service")

    # per-game scheduling (MLB). The ticker runs up to three tasks at once, sim_blend peaks at 1.3-2.5 GB, so it gets the larger share.
    files["pipeline-pregame-plan.service"] = _service("Pipeline: plan today's per-game tasks", "mllab.pipeline.pregame plan", memory="512M", runtime_max="10min",
                                                      ident="pipeline-pregame-plan")
    files["pipeline-pregame-plan.timer"] = _timer("Pipeline: plan per-game tasks every 30 min, 8:00-23:30 ET, March-November",
                                                  f"*-03..11-* 08..23:00,30:00 {TZ}", unit="pipeline-pregame-plan.service")
    files["pipeline-pregame-tick.service"] = _service("Pipeline: run due per-game tasks", "mllab.pipeline.pregame tick", memory="5G", runtime_max="1h",
                                                      ident="pipeline-pregame-tick")
    files["pipeline-pregame-tick.timer"] = _timer("Pipeline: look for due per-game tasks every minute, March-November",
                                                  f"*-03..11-* *:*:00 {TZ}", unit="pipeline-pregame-tick.service", persistent=False)

    # the broker between the web app and the pipeline (broker.py). `snapshot` publishes the status the app shows; it runs as `pipeline` because the
    # pipeline's directories are not readable by the app's user, which is the point. `apply` runs as root, only when the app drops a request, and
    # only does what broker.validate allows.
    files["pipeline-broker-snapshot.service"] = _service("Pipeline broker: publish status for the mllab app", "mllab.pipeline.broker snapshot", memory="256M",
                                                         runtime_max="2min", ident="pipeline-broker-snapshot", extra_rw=STATUS_DIR)
    files["pipeline-broker-snapshot.timer"] = """[Unit]
Description=Pipeline broker: refresh the status snapshot every 30 s

[Timer]
OnBootSec=20s
OnUnitActiveSec=30s
AccuracySec=1s
Unit=pipeline-broker-snapshot.service

[Install]
WantedBy=timers.target
"""
    files["pipeline-broker-apply.service"] = f"""[Unit]
Description=Pipeline broker: apply the mllab app's requests (root; validates untrusted request files)

[Service]
Type=oneshot
User=root
{COMMON_ENV}ExecStart={PY} -m mllab.pipeline.broker apply
TimeoutStartSec=2min
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ReadWritePaths=/etc/pipeline/jobs.d /mnt/appdata/mllab
SyslogIdentifier=pipeline-broker-apply
"""
    files["pipeline-broker-apply.path"] = f"""[Unit]
Description=Pipeline broker: watch for requests from the mllab app

[Path]
PathExistsGlob={SPOOL_GLOB}
Unit=pipeline-broker-apply.service

[Install]
WantedBy=multi-user.target
"""
    return files


def write(dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    want = generate()
    for stale in dest.glob("pipeline*"):
        if stale.name not in want:
            stale.unlink()
    for name, text in want.items():
        (dest / name).write_text(text)
    return sorted(want)


def stale(dest: Path) -> list[str]:
    want = generate()
    have = {p.name: p.read_text() for p in dest.glob("pipeline*")} if dest.is_dir() else {}
    return sorted([n for n in want if have.get(n) != want[n]] + [n for n in have if n not in want])


def main(argv: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) == 2 and a[0] == "write":
        print("\n".join(write(Path(a[1]))))
        return 0
    if len(a) == 2 and a[0] == "check":
        bad = stale(Path(a[1]))
        print("units are current" if not bad else "stale or missing: " + ", ".join(bad))
        return 1 if bad else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
