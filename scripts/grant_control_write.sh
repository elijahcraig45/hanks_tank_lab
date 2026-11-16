#!/usr/bin/env bash
# Lets the lab's service account WRITE the `control` dataset, and nothing else, so the app's Site page can change which ranking columns
# hankstank.com shows. YOU run this (IAM changes are yours to make). Without it the page still works for reading, and a save is refused
# with the equivalent `mllab control rankings ...` command to run from your laptop.
#   scripts/grant_control_write.sh                      # grants mllab-lab@hankstank.iam.gserviceaccount.com
#   scripts/grant_control_write.sh OTHER_SA_EMAIL
#   add --dry-run to print the command without running it
# Uses your personal gcloud account; refuses the work account. Undo: REVOKE `roles/bigquery.dataEditor` ON SCHEMA `hankstank.control` FROM "serviceAccount:SA".
set -euo pipefail
DRY=0; SA=mllab-lab@hankstank.iam.gserviceaccount.com
for a in "$@"; do case $a in --dry-run) DRY=1;; -*) echo "unknown arg $a" >&2; exit 2;; *) SA=$a;; esac; done
PROJECT=hankstank
ACCT=$(gcloud config get-value account 2>/dev/null)
_bd=${MLLAB_BLOCKED_ACCOUNT_DOMAINS:-}; for _d in ${_bd//,/ }; do case "$ACCT" in *"@$_d") echo "refusing: active gcloud account is the work account ($ACCT)" >&2; exit 1 ;; esac; done
SQL="GRANT \`roles/bigquery.dataEditor\` ON SCHEMA \`$PROJECT.control\` TO \"serviceAccount:$SA\""
if [ "$DRY" = 1 ]; then echo "[dry-run] bq query --project_id=$PROJECT --nouse_legacy_sql --quiet '$SQL'"; exit 0; fi
bq query --project_id="$PROJECT" --nouse_legacy_sql --quiet "$SQL"
echo "done: $SA can write the control dataset"
