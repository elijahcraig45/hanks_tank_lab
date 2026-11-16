#!/usr/bin/env bash
# Install the prediction-pipeline runtime on the homelab. Idempotent. Run as root from this directory (rsync the mllab repo over first).
#   sudo ./install.sh --dry-run             show what would change, touch nothing
#   sudo ./install.sh                       install / update (latest `main` of hanks_tank_ml)
#   sudo ./install.sh --ref v1.2            stage a specific branch, tag or commit
#   sudo ./install.sh --release             only stage a new release of the pipeline code (keeps the venv, units and modes)
#   sudo ./install.sh --uninstall           remove units, /opt/pipeline and pipeline-ctl (keeps /var/lib/pipeline and /etc/pipeline)
#
# Installing changes NOTHING about what runs: every job starts in mode `off`, so the timers fire and do nothing. Going live is
# `pipeline-ctl set <job> live`, one job at a time, per docs/HOMELAB_PIPELINE.md. This script never touches Cloud Scheduler, Cloud Tasks,
# the backend or any BigQuery table.
set -euo pipefail

DRY=0; UNINSTALL=0; RELEASE_ONLY=0; REF=main
REPO_URL=https://github.com/elijahcraig45/hanks_tank_ml.git
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --uninstall) UNINSTALL=1 ;;
    --release) RELEASE_ONLY=1 ;;
    --ref) REF=${2:?--ref needs a value}; shift ;;
    --repo) REPO_URL=${2:?--repo needs a value}; shift ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

HERE=$(cd "$(dirname "$0")" && pwd)
MLLAB=$(cd "$HERE/../.." && pwd)            # the mllab repo this script came from
OPT=/opt/pipeline; VAR=/var/lib/pipeline; ETC=/etc/pipeline
APP_DATA=/mnt/appdata/mllab   # the mllab app's data tree: the control app and the pipeline broker meet here (and only here)
JOBS_DIR=$ETC/jobs.d
KEEP_RELEASES=3
# the cron-shaped jobs; the list is read from the job table so it cannot drift
mapfile -t JOB_NAMES < <(cd "$MLLAB/src" && python3 -c "from mllab.pipeline.jobs import installed_jobs; print('\n'.join(j.name for j in installed_jobs()))")

run() { if [ "$DRY" = 1 ]; then echo "[dry-run] $*"; else "$@"; fi; }
say() { echo "==> $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || die "run as root (sudo)"
[ "${#JOB_NAMES[@]}" -gt 0 ] || die "could not read the job table from $MLLAB/src"

if [ "$UNINSTALL" = 1 ]; then
  say "uninstalling (state in $VAR and modes/key in $ETC are kept)"
  for t in /etc/systemd/system/pipeline-*.timer /etc/systemd/system/pipeline-*.path; do [ -e "$t" ] && run systemctl disable --now "$(basename "$t")" || true; done
  run rm -f /etc/systemd/system/pipeline*.service /etc/systemd/system/pipeline*.timer /etc/systemd/system/pipeline*.path /etc/systemd/system/pipeline.slice /usr/local/bin/pipeline-ctl
  run rm -rf "$OPT"
  run systemctl daemon-reload
  exit 0
fi

say "preflight"
command -v uv >/dev/null  || die "uv not found on root's PATH (the mllab install uses it too)"
command -v git >/dev/null || die "git not found: apt install git"
command -v rsync >/dev/null || die "rsync not found: apt install rsync"
getent group pipeline >/dev/null || die "group 'pipeline' is missing. Create it and the key first: pipeline-runner-setup.sh key"
[ -r "$ETC/sa.json" ] || echo "WARNING: $ETC/sa.json is missing: nothing can reach BigQuery until the key is installed" >&2

if [ "$RELEASE_ONLY" = 0 ]; then
  say "user and directories"
  id pipeline >/dev/null 2>&1 || run useradd --system --gid pipeline --home-dir "$VAR" --shell /usr/sbin/nologin pipeline
  run install -d -o root -g root -m 0755 "$OPT" "$OPT/releases" "$OPT/src" "$OPT/lib" "$OPT/lib/mllab"
  run install -d -o pipeline -g pipeline -m 0750 "$VAR" "$VAR/work" "$VAR/state"
  run install -d -o root -g pipeline -m 0750 "$ETC"
  run install -d -o root -g pipeline -m 2775 "$JOBS_DIR"   # group-writable: members of `pipeline` change modes without sudo
  # The control app (user `mllab`) never touches /etc/pipeline or /var/lib/pipeline. The broker (pipeline-broker-*) publishes status into
  # pipeline-status (written by `pipeline`, readable by group mllab) and answers requests into pipeline-results (written by root, readable by mllab).
  if getent passwd mllab >/dev/null && [ -d "$APP_DATA" ]; then
    run install -d -o pipeline -g mllab -m 2750 "$APP_DATA/pipeline-status"
    run install -d -o root -g mllab -m 2750 "$APP_DATA/pipeline-results"
  else
    echo "WARNING: no 'mllab' user or $APP_DATA yet: install mllab first (host/install.sh), then re-run this: the control app's broker directories were not created" >&2
  fi
  if [ ! -e "$ETC/env" ]; then
    say "writing $ETC/env (non-secret settings; the Kuma URL, if you add one, is a token and belongs here too)"
    if [ "$DRY" = 1 ]; then echo "[dry-run] write $ETC/env"; else
      cat > "$ETC/env" <<'EOF'
# Settings for every pipeline unit. root:pipeline 0640. Not secrets, except the optional Kuma push URL (it carries a token).
GOOGLE_APPLICATION_CREDENTIALS=/etc/pipeline/sa.json
#PIPELINE_KUMA_PUSH_URL=http://localhost:3001/api/push/<push-token>?status=up&msg=OK
EOF
      chown root:pipeline "$ETC/env"; chmod 0640 "$ETC/env"
    fi
  fi
  say "modes: a job with no mode file starts OFF; existing modes are kept"
  for name in "${JOB_NAMES[@]}" pregame; do
    [ -e "$JOBS_DIR/$name" ] || { if [ "$DRY" = 1 ]; then echo "[dry-run] $JOBS_DIR/$name = off"; else printf 'off  # installed %s\n' "$(date -u +%FT%TZ)" > "$JOBS_DIR/$name"; chown root:pipeline "$JOBS_DIR/$name"; chmod 0664 "$JOBS_DIR/$name"; fi; }
  done

  say "python 3.12 environment (the Cloud Functions run 3.12)"
  # uv downloads the interpreter into the installing user's home by default, which for root is /root: the venv then works for root and
  # fails for the `pipeline` user ("Permission denied"). So the interpreter lives under /opt/pipeline/python, and "is the venv usable?" is
  # asked as the pipeline user: a plain `[ -x ]` is true for root even when the venv is broken for everyone else.
  if id pipeline >/dev/null 2>&1 && [ -x "$OPT/venv/bin/python" ] && sudo -u pipeline "$OPT/venv/bin/python" -c 'import sys' >/dev/null 2>&1; then
    echo "venv already usable by the pipeline user"
  else
    run rm -rf "$OPT/venv"
    run env UV_PYTHON_INSTALL_DIR="$OPT/python" uv venv --python 3.12 "$OPT/venv"
    run chmod -R a+rX "$OPT/python" 2>/dev/null || true
  fi
fi

say "the runner library (its own copy: the pipeline does not depend on mllab's install or permissions). Every run refreshes it, --release included: the stager lives here, and a release is staged by it"
run install -d -o root -g root -m 0755 "$OPT" "$OPT/lib" "$OPT/lib/mllab"
run rsync -a --delete --chown=root:root --exclude __pycache__ "$MLLAB/src/mllab/pipeline/" "$OPT/lib/mllab/pipeline/"  # root owns code that runs as another user
run install -m 0644 /dev/null "$OPT/lib/mllab/__init__.py"

say "stage a release from $REPO_URL @ $REF"
if [ -d "$OPT/src/hanks_tank_ml/.git" ]; then run git -C "$OPT/src/hanks_tank_ml" fetch --all --prune --tags --quiet
else run git clone --quiet "$REPO_URL" "$OPT/src/hanks_tank_ml"; fi
if [ "$DRY" = 1 ]; then SHA=dryrun; else
  git -C "$OPT/src/hanks_tank_ml" checkout --quiet --detach "$(git -C "$OPT/src/hanks_tank_ml" rev-parse --verify --quiet "origin/$REF" || echo "$REF")"
  SHA=$(git -C "$OPT/src/hanks_tank_ml" rev-parse --short=12 HEAD)
fi
NEW="$OPT/releases/${SHA}-$(date -u +%Y%m%dT%H%M%SZ)"
run env PYTHONPATH="$OPT/lib" "$OPT/venv/bin/python" -m mllab.pipeline.stage --repo "$OPT/src/hanks_tank_ml" --dest "$NEW" --ref "$SHA"
run uv pip install --quiet --python "$OPT/venv/bin/python" -r "$NEW/requirements.lock.txt"

say "preflight the new release: import all three entrypoints, each in its own process, as the pipeline user"
if [ "$DRY" = 0 ]; then
  sudo -u pipeline env PYTHONPATH="$OPT/lib" PIPELINE_RELEASE="$NEW" "$OPT/venv/bin/python" - <<'PY' || die "the new release failed its preflight; /opt/pipeline/current was NOT changed (check that the pipeline user can run $OPT/venv/bin/python: readlink -f it and namei -l the path)"
import os, sys
from pathlib import Path
from mllab.pipeline import jobs, runner
bad = 0
for name, bundle in jobs.BUNDLES.items():
    r = runner.execute(bundle, {}, timeout=300, release=Path(os.environ["PIPELINE_RELEASE"]), resolve=lambda ref: "preflight-placeholder", check=True)
    print("  %-4s %s" % (name, "ok" if r["ok"] else "FAILED: " + (r["error"] or "") + "\n" + r["log_tail"][-600:]))
    bad += not r["ok"]
sys.exit(1 if bad else 0)
PY
fi

say "switch /opt/pipeline/current (atomic) and prune old releases"
run ln -sfn "$NEW" "$OPT/current.new"
run mv -T "$OPT/current.new" "$OPT/current"
if [ "$DRY" = 0 ]; then ls -1dt "$OPT"/releases/*/ | tail -n +$((KEEP_RELEASES + 1)) | xargs -r rm -rf; fi

if [ "$RELEASE_ONLY" = 0 ]; then
  say "systemd units"
  for f in "$HERE"/units/pipeline*; do run install -m 0644 "$f" "/etc/systemd/system/$(basename "$f")"; done
  run systemctl daemon-reload
  for t in "$HERE"/units/pipeline-*.timer "$HERE"/units/pipeline-*.path; do run systemctl enable --now "$(basename "$t")"; done  # the .path unit is what applies the app's requests

  say "pipeline-ctl"
  if [ "$DRY" = 1 ]; then echo "[dry-run] write /usr/local/bin/pipeline-ctl"; else
    cat > /usr/local/bin/pipeline-ctl <<EOF
#!/usr/bin/env bash
# pipeline-ctl: see and change what the homelab pipeline does. Installed by mllab/host/pipeline/install.sh.
set -a; . $ETC/env; set +a
export PYTHONPATH=$OPT/lib
if [ "\${1:-}" = run ]; then exec sudo -u pipeline --preserve-env=GOOGLE_APPLICATION_CREDENTIALS,PYTHONPATH $OPT/venv/bin/python -m mllab.pipeline.ctl "\$@"; fi
exec $OPT/venv/bin/python -m mllab.pipeline.ctl "\$@"
EOF
    chmod 0755 /usr/local/bin/pipeline-ctl
  fi
fi

say "done: release ${SHA} is current. Job modes are unchanged (a job installed for the first time starts OFF)."
cat <<EOF

Next:
  pipeline-ctl list                          every job, its schedule and mode (all off)
  pipeline-ctl set <job> dry                 dry-run (or preflight) on its schedule: runs, writes nothing
  pipeline-ctl run <job> --dry               run one now, safely

CFB's cfbd job needs the CFBD key from Secret Manager:
  gcloud secrets add-iam-policy-binding cfbd-api-key --project=hankstank \\
    --member=serviceAccount:pipeline-runner@hankstank.iam.gserviceaccount.com --role=roles/secretmanager.secretAccessor
EOF
