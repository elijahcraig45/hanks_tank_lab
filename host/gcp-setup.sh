#!/usr/bin/env bash
# Create the mllab service account on `hankstank` with least-privilege roles, then install its key on homelab.
# RUN THIS YOURSELF after reading it: it grants IAM permissions on the production project.
#   ./gcp-setup.sh --dry-run    print every command, change nothing
#   ./gcp-setup.sh              grant roles, create key, copy it to homelab
#   ./gcp-setup.sh --revoke     remove the roles and delete the SA (which deletes every key)
#
# Always the personal account. Accounts in MLLAB_BLOCKED_ACCOUNT_DOMAINS (e.g. a work domain) are refused.
set -euo pipefail

export CLOUDSDK_CORE_ACCOUNT=you@example.com
P=hankstank
NAME=mllab-lab
SA=$NAME@$P.iam.gserviceaccount.com
BUCKET=hanks_tank_data
DATASETS=(mlb_2026_season mlb_historical_data)
# Runtime identities the deploy adapters must be able to "act as" (scoped, not project-wide):
ACT_AS=(hankstank@appspot.gserviceaccount.com 127033547664-compute@developer.gserviceaccount.com)
PROJECT_ROLES=(roles/bigquery.jobUser roles/cloudfunctions.developer roles/run.developer roles/cloudbuild.builds.editor)

DRY=0; REVOKE=0
for a in "$@"; do
  case "$a" in --dry-run) DRY=1 ;; --revoke) REVOKE=1 ;; *) echo "unknown arg $a" >&2; exit 2 ;; esac
done
run() { if [ "$DRY" = 1 ]; then echo "[dry-run] $*"; else echo "+ $*"; "$@"; fi; }
# Same, but drop the (noisy) policy dump the command prints on success.
quiet() { if [ "$DRY" = 1 ]; then echo "[dry-run] $*"; else echo "+ $*"; "$@" >/dev/null; fi; }

_bd=${MLLAB_BLOCKED_ACCOUNT_DOMAINS:-}; for _d in ${_bd//,/ }; do case "$CLOUDSDK_CORE_ACCOUNT" in *"@$_d") echo "refusing work account" >&2; exit 1 ;; esac; done

if [ "$REVOKE" = 1 ]; then
  for r in "${PROJECT_ROLES[@]}"; do
    run gcloud projects remove-iam-policy-binding $P --member=serviceAccount:$SA --role=$r --condition=None --quiet || true
  done
  run gcloud storage buckets remove-iam-policy-binding gs://$BUCKET --member=serviceAccount:$SA --role=roles/storage.objectAdmin --quiet || true
  run gcloud iam service-accounts delete $SA --project=$P --quiet
  echo "revoked. Also delete /etc/mllab/keys/$P.json on homelab."
  exit 0
fi

echo "== 1. service account"
if gcloud iam service-accounts describe $SA --project=$P >/dev/null 2>&1; then
  echo "exists: $SA"
else
  run gcloud iam service-accounts create $NAME --project=$P \
    --display-name="mllab (homelab model lab)" --description="Private model lab on homelab. Key at /etc/mllab/keys."
fi

echo "== 2. project roles"
for r in "${PROJECT_ROLES[@]}"; do
  quiet gcloud projects add-iam-policy-binding $P --member=serviceAccount:$SA --role=$r --condition=None --quiet
done

echo "== 3. actAs on the two runtime service accounts only"
for t in "${ACT_AS[@]}"; do
  quiet gcloud iam service-accounts add-iam-policy-binding "$t" --project=$P \
    --member=serviceAccount:$SA --role=roles/iam.serviceAccountUser --quiet
done

echo "== 4. model bucket (objectAdmin on gs://$BUCKET; the bucket also holds non-model data, so this is bucket-wide)"
quiet gcloud storage buckets add-iam-policy-binding gs://$BUCKET --member=serviceAccount:$SA --role=roles/storage.objectAdmin --quiet

echo "== 5. BigQuery read access, per dataset only (GRANT on SCHEMA; bq add-iam-policy-binding only supports tables)"
for ds in "${DATASETS[@]}"; do
  LOC=$(bq show --format=prettyjson "$P:$ds" | python3 -c "import json,sys; print(json.load(sys.stdin)['location'])")
  quiet bq --location="$LOC" query --nouse_legacy_sql --quiet \
    "GRANT \`roles/bigquery.dataViewer\` ON SCHEMA \`$P.$ds\` TO \"serviceAccount:$SA\""
done

echo "== 6. key -> homelab (never written into the repo)"
if [ "$DRY" = 1 ]; then
  echo "[dry-run] gcloud iam service-accounts keys create <tmp> --iam-account=$SA"
  echo "[dry-run] scp <tmp> homelab:/tmp/ && sudo install -o root -g mllab -m 0640 ... /etc/mllab/keys/$P.json"
  echo "[dry-run] set GOOGLE_APPLICATION_CREDENTIALS in /etc/mllab/env, restart mllab-api and mllab-jupyter"
  exit 0
fi
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
gcloud iam service-accounts keys create "$TMP/key.json" --iam-account=$SA --project=$P
scp -q "$TMP/key.json" homelab:/tmp/mllab-key.json
ssh homelab '
  set -e
  sudo install -o root -g mllab -m 0640 /tmp/mllab-key.json /etc/mllab/keys/'$P'.json
  shred -u /tmp/mllab-key.json
  sudo sed -i "/^GOOGLE_APPLICATION_CREDENTIALS=/d;/^# GOOGLE_APPLICATION_CREDENTIALS=/d" /etc/mllab/env
  echo "GOOGLE_APPLICATION_CREDENTIALS=/etc/mllab/keys/'$P'.json" | sudo tee -a /etc/mllab/env >/dev/null
  sudo systemctl restart mllab-api mllab-jupyter
'
echo "done. List keys:  gcloud iam service-accounts keys list --iam-account=$SA"
