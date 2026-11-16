#!/usr/bin/env bash
# Grants the lab's service account what the health dashboard needs. YOU run this (IAM changes are yours to make).
#   scripts/grant_health_access.sh SA_EMAIL            # read tier: dashboard only
#   scripts/grant_health_access.sh SA_EMAIL --remedy   # adds what self-healing needs (scheduler run, control write)
#   add --dry-run to print the commands without running them
# Uses your personal gcloud account; refuses the work account.
set -euo pipefail
SA=${1:?usage: $0 SERVICE_ACCOUNT_EMAIL [--remedy] [--dry-run]}; shift || true
REMEDY=0; DRY=0
for a in "$@"; do case $a in --remedy) REMEDY=1;; --dry-run) DRY=1;; *) echo "unknown arg $a" >&2; exit 2;; esac; done
PROJECT=hankstank
ACCT=$(gcloud config get-value account 2>/dev/null)
_bd=${MLLAB_BLOCKED_ACCOUNT_DOMAINS:-}; for _d in ${_bd//,/ }; do case "$ACCT" in *"@$_d") echo "refusing: active gcloud account is the work account ($ACCT)" >&2; exit 1 ;; esac; done
run() { if [ "$DRY" = 1 ]; then echo "[dry-run] $*"; else "$@"; fi; }
M="serviceAccount:$SA"
run gcloud projects add-iam-policy-binding $PROJECT --member="$M" --role=roles/bigquery.jobUser --condition=None
run gcloud projects add-iam-policy-binding $PROJECT --member="$M" --role=roles/cloudscheduler.viewer --condition=None
run gcloud projects add-iam-policy-binding $PROJECT --member="$M" --role=roles/logging.viewer --condition=None
# Dataset-level grants use SQL GRANT: `bq add-iam-policy-binding` needs allowlisting on this project.
grant_ds() { # role dataset
  run bq query --project_id="$PROJECT" --nouse_legacy_sql --quiet "GRANT \`$1\` ON SCHEMA \`$PROJECT.$2\` TO \"serviceAccount:$SA\"" || echo "note: could not grant $1 on $2"
}
# nfl_historical, cfb_historical and pickem are read by the Data page (table metadata) and by the football models' history
for ds in mlb_2026_season nfl_season cfb_season control nfl_historical cfb_historical pickem; do grant_ds roles/bigquery.dataViewer "$ds"; done
if [ "$REMEDY" = 1 ]; then
  run gcloud projects add-iam-policy-binding $PROJECT --member="$M" --role=roles/cloudscheduler.jobRunner --condition=None
  grant_ds roles/bigquery.dataEditor control
fi
echo "done: $( [ "$REMEDY" = 1 ] && echo "read + remedy tier" || echo "read tier" ) for $SA"
