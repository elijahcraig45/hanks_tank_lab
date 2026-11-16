#!/usr/bin/env bash
# Push the current repo to homelab and (re)install the app into /opt/mllab/venv. Fails LOUDLY: nothing here hides the installer's output
# or exit status, and it ends by proving the installed CLI actually runs. Does not restart services or touch systemd units
# (use host/install.sh for those).
#   scripts/deploy_app.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")
rsync -a --delete --exclude .venv --exclude .git --exclude __pycache__ --exclude .pytest_cache --exclude '/20[0-9][0-9]-*.txt' "$REPO/" homelab:~/mllab-src/
ssh homelab 'set -euo pipefail
  sudo rsync -a --delete --exclude .venv --exclude .git --exclude __pycache__ ~/mllab-src/ /opt/mllab/src/
  sudo chown -R mllab:mllab /opt/mllab/src
  # Anything that ran the venv python as ROOT leaves root-owned __pycache__ inside site-packages, and then the installer (running as
  # mllab) cannot remove the old package. Repair ownership first. Run the venv python as root only with -B.
  sudo find /opt/mllab/venv -user root -exec chown mllab:mllab {} +
  cd /opt/mllab
  sudo -u mllab uv pip install --python /opt/mllab/venv/bin/python "/opt/mllab/src[lab]" 2>&1 | tail -3
  test -x /opt/mllab/venv/bin/mllab || { echo "DEPLOY FAILED: /opt/mllab/venv/bin/mllab is missing after install" >&2; exit 1; }
  sudo -u mllab bash -c "set -a; . /etc/mllab/env; set +a; /opt/mllab/venv/bin/mllab --help >/dev/null" \
    || { echo "DEPLOY FAILED: the installed mllab CLI does not run" >&2; exit 1; }
  echo "deploy verified: the installed mllab CLI runs"'
