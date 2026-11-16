#!/usr/bin/env bash
# Service-account maintenance for mllab. RUN THIS YOURSELF: it changes credentials/IAM on the production project.
#   ./gcp-maintenance.sh rotate-key [--dry-run]          new key -> homelab -> verified -> old keys deleted
#   ./gcp-maintenance.sh enable-storage-api [--dry-run]  grant roles/bigquery.readSessionUser + switch on the fast download path
#
# rotate-key order matters: the NEW key is installed and proven to work (a real BigQuery dry-run as mllab) BEFORE any
# old key is deleted, and the old keys are only listed for deletion if that check passed.
set -euo pipefail

export CLOUDSDK_CORE_ACCOUNT=you@example.com
P=hankstank; SA=mllab-lab@$P.iam.gserviceaccount.com; KEYFILE=/etc/mllab/keys/$P.json
CMD=${1:-}; shift || true
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
run() { if [ "$DRY" = 1 ]; then echo "[dry-run] $*"; else echo "+ $*"; "$@"; fi; }
_bd=${MLLAB_BLOCKED_ACCOUNT_DOMAINS:-}; for _d in ${_bd//,/ }; do case "$CLOUDSDK_CORE_ACCOUNT" in *"@$_d") echo "refusing work account" >&2; exit 1 ;; esac; done

case "$CMD" in
rotate-key)
  echo "== current user-managed keys"
  gcloud iam service-accounts keys list --iam-account=$SA --managed-by=user --format="value(name.basename())" | tee /tmp/mllab-old-keys
  [ -s /tmp/mllab-old-keys ] || { echo "no existing user-managed key; use gcp-setup.sh" >&2; exit 1; }
  if [ "$DRY" = 1 ]; then
    echo "[dry-run] create a new key, scp to homelab, install at $KEYFILE (0640 root:mllab), restart mllab-api mllab-jupyter"
    echo "[dry-run] verify: mllab bq dry-run as the mllab user with the new key; abort (keeping old keys) if it fails"
    echo "[dry-run] delete old key ids: $(tr '\n' ' ' < /tmp/mllab-old-keys)"; exit 0
  fi
  TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
  gcloud iam service-accounts keys create "$TMP/key.json" --iam-account=$SA --project=$P
  NEW=$(python3 -c "import json;print(json.load(open('$TMP/key.json'))['private_key_id'])")
  scp -q "$TMP/key.json" homelab:/tmp/mllab-key.json
  ssh homelab "set -e; sudo cp -a $KEYFILE $KEYFILE.prev; sudo install -o root -g mllab -m 0640 /tmp/mllab-key.json $KEYFILE; shred -u /tmp/mllab-key.json; sudo systemctl restart mllab-api mllab-jupyter"
  echo "== verifying the new key works (BigQuery dry-run as the mllab user)"
  if ssh homelab 'sudo -u mllab bash -c "set -a; . /etc/mllab/env; set +a; /opt/mllab/venv/bin/python -c \"
from google.cloud import bigquery
c = bigquery.Client(project=\\\"hankstank\\\")
j = c.query(\\\"SELECT COUNT(*) FROM \\\`hankstank.mlb_2026_season.games\\\`\\\", job_config=bigquery.QueryJobConfig(dry_run=True))
print(\\\"new key OK, estimated bytes\\\", j.total_bytes_processed)
\""' ; then
    for k in $(grep -v "^$NEW" /tmp/mllab-old-keys); do run gcloud iam service-accounts keys delete "$k" --iam-account=$SA --quiet; done
    ssh homelab "sudo shred -u $KEYFILE.prev"; echo "rotated. New key id $NEW; old keys deleted."
  else
    echo "NEW KEY FAILED. Restoring the previous key file; old keys were NOT deleted." >&2
    ssh homelab "sudo mv $KEYFILE.prev $KEYFILE; sudo systemctl restart mllab-api mllab-jupyter"
    gcloud iam service-accounts keys delete "$NEW" --iam-account=$SA --quiet; exit 1
  fi ;;
enable-storage-api)
  run gcloud projects add-iam-policy-binding $P --member=serviceAccount:$SA --role=roles/bigquery.readSessionUser --condition=None --quiet
  run ssh homelab "grep -q '^MLLAB_BQ_STORAGE_API=' /etc/mllab/env || echo MLLAB_BQ_STORAGE_API=1 | sudo tee -a /etc/mllab/env >/dev/null; sudo systemctl restart mllab-api mllab-jupyter"
  echo "done. Large result downloads now use the Storage API (still capped at 1 GiB by the result guard)." ;;
*) sed -n 2,8p "$0"; exit 2 ;;
esac
