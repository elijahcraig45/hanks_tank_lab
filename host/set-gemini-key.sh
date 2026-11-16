#!/usr/bin/env bash
# Store GEMINI_API_KEY on homelab for `mllab ask`. RUN THIS YOURSELF: it handles a secret.
# The key is typed at a silent prompt, sent over ssh on stdin, and written to /etc/mllab/env (0640 root:mllab).
# It never appears in shell history, process arguments, the repo or any log.
#   ./set-gemini-key.sh            prompt, install, restart services, run one live check
#   ./set-gemini-key.sh --remove   delete the key from the server
set -euo pipefail
if [ "${1:-}" = "--remove" ]; then
  ssh homelab 'sudo sed -i "/^GEMINI_API_KEY=/d" /etc/mllab/env && sudo systemctl restart mllab-api mllab-jupyter' && echo "removed."; exit 0
fi
read -rsp "Gemini API key (input hidden): " KEY; echo
[[ "$KEY" =~ ^[A-Za-z0-9_-]{30,}$ ]] || { echo "that does not look like an API key (expected 30+ letters, digits, - or _)" >&2; exit 1; }
printf '%s\n' "$KEY" | ssh homelab 'IFS= read -r K
  sudo sed -i "/^GEMINI_API_KEY=/d" /etc/mllab/env
  printf "GEMINI_API_KEY=%s\n" "$K" | sudo tee -a /etc/mllab/env >/dev/null
  sudo systemctl restart mllab-api mllab-jupyter'
unset KEY
echo "== live check (one real call, counts toward the daily cap)"
ssh homelab 'sudo -u mllab bash -c "set -a; . /etc/mllab/env; set +a; /opt/mllab/venv/bin/mllab ask \"Reply with the single word: ok\""' | head -3
