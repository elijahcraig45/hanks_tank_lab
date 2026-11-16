#!/usr/bin/env bash
# Stage 2 of the move from Cloud Functions to the homelab: the four jobs that REFUSE a dry run (their first live run is their first real test), done one at a time with a
# real test run after each move.
#
#   bash scripts/cutover_stage2.sh           print the plan and the current state; change nothing
#   bash scripts/cutover_stage2.sh --yes     do it
#
# For each job, in order (the order matters: score reads what ingest wrote):
#   1. pause its Cloud Scheduler twin (audited CLI) and check Cloud Scheduler says PAUSED
#   2. set the job Live THROUGH THE APP (it re-checks the twin fresh, takes the typed name, hands the change to the root-side broker, audits it)
#   3. run it once for real, through the app (same gate), and wait for the run to finish
#   4. check the run was OK and that every table the job writes (the data map's catalog) is fresh and was actually changed by it
# If step 2, 3 or 4 fails the job is rolled back (mode dry, twin resumed) and the script stops, so no job is left without an owner. The Cloud Functions stay deployed.
#
# Not in this stage: per-game MLB (`pregame`), which needs the backend's cron entry removed and the Cloud Tasks queues drained.
# Undo one job:   mllab control scheduler resume JOB --confirm JOB   then   pipeline-ctl set JOB dry      (on the homelab)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PY:-.venv/bin/python}
CLI="$PY -m mllab.cli"
HOST=homelab; TAILNAME=homelab.<tailnet>.ts.net; TAILIP=<homelab-tailnet-ip>
JOBS=${JOBS:-"nfl-weekly-ingest nfl-weekly-score cfb-weekly-ingest cfb-weekly-cfbd"}
WAIT_RUN_S=${WAIT_RUN_S:-1500}

export MLLAB_AUDIT=${MLLAB_AUDIT:-$HOME/.mllab/audit/deploys.log}
export CLOUDSDK_CORE_ACCOUNT=you@example.com   # the personal account owns the project; the default account on this machine is the work one

state() { $CLI control scheduler list 2>&1 | awk -F'|' -v j="$1" '{n=$2; gsub(/ /,"",n); if (n==j) {s=$3; gsub(/ /,"",s); print s}}'; }
# GET / POST the app on the homelab (it answers the tailnet name; the tailscale identity header is added by tailscale serve)
app_get()  { ssh "$HOST" "curl -sk -m 120 --resolve $TAILNAME:8899:$TAILIP 'https://$TAILNAME:8899$1'"; }
app_post() { ssh "$HOST" "curl -sk -m 60 --resolve $TAILNAME:8899:$TAILIP -X POST 'https://$TAILNAME:8899$1' -H 'Content-Type: application/json' -H 'X-Mllab-App: 1' -H 'Origin: https://$TAILNAME:8899' -d '$2'"; }
jget() { python3 -c "import sys,json
try:
    d=json.load(sys.stdin)
except Exception:
    print(''); sys.exit()
$1"; }

wait_request() { # request id -> prints the final status line
  local rid=$1 r=""
  for _ in $(seq 1 30); do
    sleep 2; r=$(app_get "/app-api/pipeline/request/$rid")
    printf '%s' "$r" | grep -q '"status": *"queued"' || break
  done
  printf '%s' "$r" | jget "print(d.get('status',''), '|', d.get('detail',''))"
}

rollback() { # job reason
  echo "  ROLLING BACK $1 ($2)"
  local res; res=$(app_post /app-api/pipeline/mode "{\"job\":\"$1\",\"mode\":\"dry\"}")
  echo "  set dry: $(printf '%s' "$res" | cut -c1-120)"
  $CLI control scheduler resume "$1" --confirm "$1" 2>&1 | tail -2 | sed 's/^/  resume: /'
}

echo "Stage 2: four jobs that cannot be dry-run, each moved then run once for real"
printf '  %-22s %-14s %s\n' "job" "Cloud Scheduler" "mode on the homelab"
SNAP=$(app_get /app-api/pipeline)
for j in $JOBS; do printf '  %-22s %-14s %s\n' "$j" "$(state "$j")" "$(printf '%s' "$SNAP" | jget "print(next((x['mode'] for x in d['jobs'] if x['name']=='$j'),'?'))")"; done
echo
if [ "${1:-}" != "--yes" ]; then
  echo "Nothing changed. Re-run with --yes. Each job: pause twin -> Live -> one real run -> verify its tables -> (rollback on any failure)."
  exit 0
fi

for j in $JOBS; do
  echo "=== $j"
  $CLI control scheduler pause "$j" --confirm "$j" 2>&1 | tail -3 | sed 's/^/  pause: /'
  [ "$(state "$j")" = "PAUSED" ] || { echo "  STOP: Cloud Scheduler does not say PAUSED for $j"; exit 1; }

  res=$(app_post /app-api/pipeline/mode "{\"job\":\"$j\",\"mode\":\"live\",\"confirm\":\"$j\"}")
  rid=$(printf '%s' "$res" | jget "print(d.get('request',''))")
  if [ -z "$rid" ]; then echo "  the app refused going live: $(printf '%s' "$res" | cut -c1-200)"; rollback "$j" "app refused"; exit 1; fi
  echo "  live: $(wait_request "$rid")"

  before=$(app_get /app-api/pipeline | jget "print(next((((x.get('last') or {}).get('finished')) or '' for x in d['jobs'] if x['name']=='$j'),''))")
  t0=$(date -u +%Y-%m-%dT%H:%M:%S)
  res=$(app_post /app-api/pipeline/run "{\"job\":\"$j\",\"dry\":false,\"confirm\":\"$j\"}")
  rid=$(printf '%s' "$res" | jget "print(d.get('request',''))")
  if [ -z "$rid" ]; then echo "  the app refused the real run: $(printf '%s' "$res" | cut -c1-200)"; rollback "$j" "app refused the run"; exit 1; fi
  echo "  real run started: $(wait_request "$rid")  (waiting up to $((WAIT_RUN_S / 60)) min for it to finish)"

  ok=""; waited=0
  while [ $waited -lt "$WAIT_RUN_S" ]; do
    sleep 20; waited=$((waited + 20))
    row=$(app_get /app-api/pipeline | jget "
x=next((x for x in d['jobs'] if x['name']=='$j'),None); l=(x or {}).get('last') or {}
if l.get('finished') and l.get('finished')!='$before' and l.get('mode')=='live':
    print('yes' if l.get('ok') else 'no', l.get('seconds'), str(l.get('error') or '')[:120])")
    [ -n "$row" ] && { ok=${row%% *}; echo "  run finished: $row"; break; }
  done
  [ "$ok" = "yes" ] || { rollback "$j" "${ok:+the run failed}${ok:-no result within the wait}"; exit 1; }

  # the snapshot refreshes every 30 s and the data map is cached for a minute: ask for a fresh one after the run
  sleep 35
  bad=$(app_get "/app-api/data?fresh=true" | jget "
rows=[(t['id'],t['status'],t['reasons'][0] if t['reasons'] else '') for s in d.get('sports',{}).values() for t in s['tables'] if '$j' in t['written_by'] and t['status'] in ('stale','missing','unchanged','late')]
print(d.get('error','') if not d.get('available') else '; '.join(f'{a} is {b}: {c[:70]}' for a,b,c in rows))")
  if [ -n "$bad" ]; then echo "  VERIFY FAILED: $bad"; rollback "$j" "its tables are not fresh after a clean run"; exit 1; fi
  echo "  verified: every table $j writes is fresh and was changed by the run"
done
echo "=== STAGE 2 DONE. Check the app's Pipeline page (these four Live, no red banner), the Data page (all fresh) and the GCP page (their twins Paused)."
