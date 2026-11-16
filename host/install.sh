#!/usr/bin/env bash
# Install mllab on homelab. Idempotent. Run as root from the repo's host/ dir (rsync the repo over first).
#   sudo ./install.sh --dry-run      show what would change, touch nothing
#   sudo ./install.sh                install / update
#   sudo ./install.sh --uninstall    remove units and /opt/mllab (keeps /mnt/appdata/mllab data and /etc/mllab)
#
# This script does NOT touch `tailscale serve`, ufw, or docker. It prints the serve command for you to run.
set -euo pipefail

DRY=0; UNINSTALL=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --uninstall) UNINSTALL=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")
DATA=/mnt/appdata/mllab
UNITS=(mllab.slice mllab-api.service mllab-jupyter.service mllab-caddy.service mllab-backup.service mllab-backup.timer mllab-spool.path mllab-spool.service mllab-health.service mllab-health.timer mllab-weekly.service mllab-weekly.timer mllab-modelhealth.service mllab-modelhealth.timer)

run() { if [ "$DRY" = 1 ]; then echo "[dry-run] $*"; else "$@"; fi; }
say() { echo "==> $*"; }

[ "$(id -u)" = 0 ] || { echo "run as root (sudo)" >&2; exit 1; }

if [ "$UNINSTALL" = 1 ]; then
  say "uninstalling (data in $DATA and /etc/mllab are kept)"
  for u in "${UNITS[@]}"; do
    run systemctl disable --now "$u" 2>/dev/null || true
    run rm -f "/etc/systemd/system/$u"
  done
  run rm -rf /opt/mllab
  run systemctl daemon-reload
  echo "also remove the tailnet mapping yourself:  tailscale serve --https=8899 off"
  exit 0
fi

say "preflight"
command -v uv >/dev/null || { echo "uv not found. Install it first (https://docs.astral.sh/uv/) so it is on root's PATH." >&2; exit 1; }
[ -x /usr/local/bin/caddy ] || { echo "/usr/local/bin/caddy missing. Download the static binary from caddyserver.com (do NOT apt install: its default service would fight the existing Caddy on :80)." >&2; exit 1; }
for tool in rsync tmux; do command -v "$tool" >/dev/null || { echo "missing: apt install $tool" >&2; exit 1; }; done
# Only check for port conflicts on a first install; on reruns our own units legitimately hold these ports.
if ! systemctl is-active --quiet mllab-api.service mllab-jupyter.service mllab-caddy.service 2>/dev/null; then
  busy=$(ss -ltn 2>/dev/null | grep -E ':(8897|8898|8899)[[:space:]]' || true)
  if [ -n "$busy" ]; then
    echo "something already listens on 8897-8899:" >&2
    echo "$busy" >&2
    exit 1
  fi
fi

say "user and directories"
id mllab >/dev/null 2>&1 || run useradd --system --create-home --home-dir /var/lib/mllab --shell /usr/sbin/nologin mllab
for d in registry audit db work cache data jobs envs spool health; do run install -d -o mllab -g mllab -m 0750 "$DATA/$d"; done
run install -d -o root -g mllab -m 0750 /etc/mllab /etc/mllab/keys

say "env file (secrets generated once, never overwritten)"
if [ ! -f /etc/mllab/env ]; then
  if [ "$DRY" = 1 ]; then
    echo "[dry-run] would create /etc/mllab/env (0640 root:mllab) with random JUPYTER_TOKEN and MLLAB_SECRET"
  else
    umask 077
    cat > /etc/mllab/env <<EOF
JUPYTER_TOKEN=$(head -c 32 /dev/urandom | base64 | tr -d '=+/' | head -c 40)
MLLAB_SECRET=$(head -c 32 /dev/urandom | base64 | tr -d '=+/' | head -c 40)
# REQUIRED: your tailnet login, exactly as Tailscale reports it. Auth fails closed while this is empty.
MLLAB_ALLOWED_LOGIN=
MLLAB_REGISTRY=$DATA/registry
MLLAB_AUDIT=$DATA/audit/deploys.log
MLLAB_CACHE=$DATA/cache
# Service-account key for the GCP project (never a personal or work login):
# GOOGLE_APPLICATION_CREDENTIALS=/etc/mllab/keys/hankstank.json
CLOUDSDK_CONFIG=/var/lib/mllab/gcloud
EOF
    chown root:mllab /etc/mllab/env; chmod 0640 /etc/mllab/env
  fi
fi

say "env defaults for jobs (appended only if missing)"
if [ -f /etc/mllab/env ] && [ "$DRY" != 1 ]; then
  grep -q '^MLLAB_ROOT=' /etc/mllab/env || echo "MLLAB_ROOT=$DATA" >> /etc/mllab/env
  grep -q '^MLLAB_JOB_PYTHON=' /etc/mllab/env || echo "MLLAB_JOB_PYTHON=/opt/mllab/venv/bin/python" >> /etc/mllab/env
else
  echo "[dry-run or no env file yet] would ensure MLLAB_ROOT and MLLAB_JOB_PYTHON in /etc/mllab/env"
fi

# A running service keeps the code it started with, and `enable --now` does not restart it: so note what we are about to replace, and restart
# the API and Caddy afterwards only if it changed. Jupyter is never restarted here: that would end the notebooks that are open.
tree_hash() { (cd "$1" 2>/dev/null && find . -type f -not -path '*/__pycache__/*' | sort | xargs -r sha256sum | sha256sum) || echo none; }
CODE_BEFORE=$(tree_hash /opt/mllab/src/src); CADDY_BEFORE=$(sha256sum /etc/mllab/Caddyfile 2>/dev/null || echo none)

say "application venv (/opt/mllab/venv, Python 3.12)"
run install -d -o mllab -g mllab /opt/mllab
run rsync -a --delete --exclude .venv --exclude .git --exclude __pycache__ "$REPO/" /opt/mllab/src/
run chown -R mllab:mllab /opt/mllab
# cd away first: the invoking user's home is unreadable by mllab and uv looks for uv.toml in the cwd.
cd /opt/mllab 2>/dev/null || cd /
[ -x /opt/mllab/venv/bin/python ] || run runuser -u mllab -- uv venv --python 3.12 /opt/mllab/venv
run runuser -u mllab -- uv pip install --python /opt/mllab/venv/bin/python "/opt/mllab/src[lab]"

say "starter notebooks (copied into the work folder only if absent: a notebook you have edited is never overwritten)"
for nbk in "$REPO"/notebooks/*.ipynb; do
  [ -e "$nbk" ] || continue
  dest="$DATA/work/$(basename "$nbk")"
  [ -e "$dest" ] || run install -m 0640 -o mllab -g mllab "$nbk" "$dest"
done

say "systemd units and Caddyfile"
run install -m 0644 "$HERE/Caddyfile.mllab" /etc/mllab/Caddyfile
for u in "${UNITS[@]}"; do run install -m 0644 "$HERE/$u" "/etc/systemd/system/$u"; done
run install -m 0755 "$HERE/backup.sh" /usr/local/bin/mllab-backup
run install -m 0755 "$HERE/mllab-health" /usr/local/bin/mllab-health
run install -m 0750 -o root -g root "$HERE/mllab-jobctl" /usr/local/sbin/mllab-jobctl
run systemctl daemon-reload

if grep -q '^MLLAB_ALLOWED_LOGIN=$' /etc/mllab/env 2>/dev/null; then
  say "NOT starting services yet: set MLLAB_ALLOWED_LOGIN in /etc/mllab/env first, then re-run."
else
  run systemctl enable --now mllab-api.service mllab-jupyter.service mllab-caddy.service mllab-backup.timer mllab-spool.path mllab-health.timer mllab-weekly.timer mllab-modelhealth.timer
  if [ "$DRY" = 0 ]; then
    [ "$CODE_BEFORE" = "$(tree_hash /opt/mllab/src/src)" ] || { say "mllab code changed: restarting mllab-api (Jupyter is left alone)"; systemctl restart mllab-api.service; }
    [ "$CADDY_BEFORE" = "$(sha256sum /etc/mllab/Caddyfile 2>/dev/null || echo none)" ] || { say "Caddyfile changed: restarting mllab-caddy"; systemctl restart mllab-caddy.service; }
  fi
fi

cat <<'EOF'

Next steps (run by hand):
  1. Back up the existing serve config FIRST:   tailscale serve status --json > ~/serve-backup.json
  2. Publish additively (do not rebuild):       tailscale serve --bg --https=8899 http://127.0.0.1:8899
  3. Confirm the 12 old mappings survived:      tailscale serve status
  4. Verify nothing listens publicly:           ss -ltnp | grep -E ':(8897|8898|8899)'   # all must be 127.0.0.1
  5. Jupyter token for first login:             sudo grep JUPYTER_TOKEN /etc/mllab/env
EOF
