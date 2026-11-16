#!/usr/bin/env bash
# Stage 1 of the move from Cloud Functions to the homelab: the eight jobs that can run in dry mode and have been rehearsed there.
#
#   bash scripts/cutover_stage1.sh           print the plan and the current state; change nothing
#   bash scripts/cutover_stage1.sh --yes     do it
#
# For each job, in order: (1) pause its Cloud Scheduler twin with the audited `mllab control scheduler pause`; (2) check Cloud Scheduler says PAUSED;
# (3) set the job Live THROUGH THE APP, which re-reads Cloud Scheduler fresh, demands the job's name, hands the change to the root-side broker, and records it
# in the audit log. It stops at the first problem; if the app refuses, it resumes that job's Cloud Scheduler twin so nothing is left unowned.
#
# Not in this stage, on purpose: the four jobs that refuse dry runs (nfl-weekly-ingest, nfl-weekly-score, cfb-weekly-ingest, cfb-weekly-cfbd: their first live run is
# their first real test) and `pregame` (two triggers and two queues). See docs/HOMELAB_PIPELINE.md.
#
# Undo one job:   mllab control scheduler resume JOB --confirm JOB   then   pipeline-ctl set JOB dry      (on the homelab)
# The Cloud Functions stay deployed throughout; nothing here deletes anything.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PY:-.venv/bin/python}
CLI="$PY -m mllab.cli"
HOST=homelab; TAILNAME=homelab.<tailnet>.ts.net; TAILIP=<homelab-tailnet-ip>
JOBS="mlb-2026-roster-refresh mlb-2026-power-rankings nfl-weekly-season-sim cfb-weekly-drives cfb-weekly-season-sim cfb-weekly-predict nfl-weekly-predict mlb-2026-daily"

# The CLI's default audit path is the homelab's. On this machine, keep the Cloud Scheduler pauses in a log of their own.
export MLLAB_AUDIT=${MLLAB_AUDIT:-$HOME/.mllab/audit/deploys.log}
export CLOUDSDK_CORE_ACCOUNT=you@example.com   # the personal account owns the project; the default account on this machine is the work one

state() { $CLI control scheduler list 2>&1 | awk -F'|' -v j="$1" '{n=$2; gsub(/ /,"",n); if (n==j) {s=$3; gsub(/ /,"",s); print s}}'; }

echo "Stage 1: eight jobs, one at a time"
printf '  %-28s %s\n' "job" "Cloud Scheduler now"
for j in $JOBS; do printf '  %-28s %s\n' "$j" "$(state "$j")"; done
echo
if [ "${1:-}" != "--yes" ]; then
  echo "Nothing changed. Re-run with --yes to pause each Cloud Scheduler twin and set the job Live on the homelab."
  exit 0
fi

for j in $JOBS; do
  echo "=== $j"
  $CLI control scheduler pause "$j" --confirm "$j" 2>&1 | tail -3 | sed 's/^/  pause: /'
  [ "$(state "$j")" = "PAUSED" ] || { echo "  STOP: Cloud Scheduler does not say PAUSED for $j"; exit 1; }
  res=$(ssh "$HOST" "curl -sk -m 40 --resolve $TAILNAME:8899:$TAILIP -X POST https://$TAILNAME:8899/app-api/pipeline/mode \
        -H 'Content-Type: application/json' -H 'X-Mllab-App: 1' -H 'Origin: https://$TAILNAME:8899' -d '{\"job\":\"$j\",\"mode\":\"live\",\"confirm\":\"$j\"}'")
  echo "  app: $res" | cut -c1-240
  rid=$(printf '%s' "$res" | python3 -c "import sys,json; print(json.load(sys.stdin).get('request',''))" 2>/dev/null)
  if [ -z "$rid" ]; then
    echo "  STOP: the app refused. Resuming the Cloud Scheduler twin so $j is not left with no owner."
    $CLI control scheduler resume "$j" --confirm "$j" 2>&1 | head -2 | sed 's/^/  resume: /'
    exit 1
  fi
  r=""
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 2
    r=$(ssh "$HOST" "curl -sk -m 10 --resolve $TAILNAME:8899:$TAILIP https://$TAILNAME:8899/app-api/pipeline/request/$rid")
    printf '%s' "$r" | grep -q '"status": *"queued"' || break
  done
  echo "  broker: $r" | cut -c1-240
  printf '%s' "$r" | grep -q '"status": *"done"' || { echo "  STOP: the broker did not apply it. $j is paused in GCP and not yet Live here: resume it or retry."; exit 1; }
done
echo "=== STAGE 1 DONE. Check: the app's Pipeline page (eight jobs Live, no red banner) and the GCP page (their twins Paused)."
