#!/usr/bin/env bash
# Stage 3, the last one: per-game MLB (`pregame`) moves from Cloud Functions + Cloud Tasks to the homelab.
#
#   bash scripts/cutover_stage3.sh                           readiness check only; changes nothing
#   bash scripts/cutover_stage3.sh --yes                     do it
#   bash scripts/cutover_stage3.sh --verify                  after the first live checkpoint has run: did it work?
#   bash scripts/cutover_stage3.sh --rollback                hand it back to Cloud Functions
#
# Per-game tasks are created by the Cloud Scheduler job mlb-2026-pregame-schedule, which this script pauses (audited). (The backend's cron.yaml also lists a
# /api/lineup/schedule-today entry, but its crontab syntax is invalid for App Engine, so it was never deployed: the script CHECKS that no App Engine cron job exists in
# Cloud Scheduler (aet-*) and refuses if one does, instead of taking your word for it.)
# and what is already queued must not run on top: the Cloud Tasks queues lineup-pregame and sim-blend are paused (a pause holds tasks, it deletes nothing).
#
# Choose a quiet moment: it refuses unless the next rehearsed checkpoint is at least 30 minutes away, so no checkpoint is run by both systems or by neither.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PY:-.venv/bin/python}
CLI="$PY -m mllab.cli"
HOST=homelab; TAILNAME=homelab.<tailnet>.ts.net; TAILIP=<homelab-tailnet-ip>
PROJECT=hankstank; LOCATION=us-central1; QUEUES="lineup-pregame sim-blend"; TWIN=mlb-2026-pregame-schedule
STATE=/var/lib/pipeline/state
MIN_WINDOW_MIN=${MIN_WINDOW_MIN:-30}

export MLLAB_AUDIT=${MLLAB_AUDIT:-$HOME/.mllab/audit/deploys.log}
export CLOUDSDK_CORE_ACCOUNT=you@example.com

state() { $CLI control scheduler list 2>&1 | awk -F'|' -v j="$1" '{n=$2; gsub(/ /,"",n); if (n==j) {s=$3; gsub(/ /,"",s); print s}}'; }
qstate() { gcloud tasks queues describe "$1" --location "$LOCATION" --project "$PROJECT" --format='value(state)' 2>/dev/null; }
app_get()  { ssh "$HOST" "curl -sk -m 60 --resolve $TAILNAME:8899:$TAILIP 'https://$TAILNAME:8899$1'"; }
app_post() { ssh "$HOST" "curl -sk -m 60 --resolve $TAILNAME:8899:$TAILIP -X POST 'https://$TAILNAME:8899$1' -H 'Content-Type: application/json' -H 'X-Mllab-App: 1' -H 'Origin: https://$TAILNAME:8899' -d '$2'"; }
jget() { python3 -c "import sys,json
try:
    d=json.load(sys.stdin)
except Exception:
    print(''); sys.exit()
$1"; }
sql() { ssh "$HOST" "sudo -n sqlite3 -readonly $STATE/pregame-$1.sqlite \"$2\" 2>/dev/null"; }
wait_request() { local rid=$1 r=""; for _ in $(seq 1 30); do sleep 2; r=$(app_get "/app-api/pipeline/request/$rid"); printf '%s' "$r" | grep -q '"status": *"queued"' || break; done
  printf '%s' "$r" | jget "print(d.get('status',''), '|', d.get('detail',''))"; }
set_mode() { local res rid; res=$(app_post /app-api/pipeline/mode "$2"); rid=$(printf '%s' "$res" | jget "print(d.get('request',''))")
  [ -n "$rid" ] && echo "$1: $(wait_request "$rid")" || echo "$1: REFUSED $(printf '%s' "$res" | cut -c1-240)"; }

readiness() {
  local ok=1
  local mode; mode=$(app_get /app-api/pipeline | jget "print((d.get('pregame') or {}).get('mode',''))")
  echo "  pregame mode on the homelab ........ ${mode:-?}"
  local bad done_n; bad=$(sql dry "select count(*) from tasks where state in ('failed','skipped') and game_date >= date('now','-4 day')"); done_n=$(sql dry "select count(*) from tasks where state='done'")
  echo "  dry rehearsal ...................... ${done_n:-?} tasks done, ${bad:-?} failed or skipped in the last 4 days"
  [ "${bad:-1}" = "0" ] && [ "${done_n:-0}" -ge 10 ] || { echo "    NOT READY: the rehearsal must be clean (0 failures) with at least 10 tasks done"; ok=0; }
  local mins; mins=$(sql dry "select cast((julianday(min(due)) - julianday('now')) * 1440 as integer) from tasks where state='pending' and due > strftime('%Y-%m-%dT%H:%M:%SZ','now')")
  echo "  next rehearsed checkpoint in ....... ${mins:-none queued} minutes (needs >= $MIN_WINDOW_MIN)"
  if [ -n "${mins:-}" ] && [ "$mins" -lt "$MIN_WINDOW_MIN" ]; then echo "    NOT READY: a checkpoint is too close; wait for it to pass (or pick the gap before the next game)"; ok=0; fi
  echo "  Cloud Scheduler $TWIN ... $(state "$TWIN")"
  for q in $QUEUES; do echo "  Cloud Tasks queue $q ........ $(qstate "$q")"; done
  [ $ok = 1 ]
}

rollback() {
  echo "ROLLING BACK per-game MLB to Cloud Functions"
  set_mode "pregame -> dry" '{"job":"pregame","mode":"dry"}'
  for q in $QUEUES; do gcloud tasks queues resume "$q" --location "$LOCATION" --project "$PROJECT" 2>&1 | tail -1 | sed "s/^/  resume $q: /"; done
  $CLI control scheduler resume "$TWIN" --confirm "$TWIN" 2>&1 | tail -2 | sed 's/^/  resume twin: /'
}

case "${1:-}" in
  --rollback) rollback; exit 0 ;;
  --verify)
    echo "per-game MLB on the homelab:"
    sql live "select state, count(*) from tasks group by state" | sed 's/^/  live store: /'
    f=$(sql live "select count(*) from tasks where state in ('failed','skipped')"); d=$(sql live "select count(*) from tasks where state='done'")
    last=$(sql live "select kind || ' for game ' || game_pk || ' at ' || updated from tasks where state='done' order by updated desc limit 1")
    echo "  done: ${d:-0}   failed or skipped: ${f:-?}   latest: ${last:-none yet}"
    app_get "/app-api/data?fresh=true" | jget "
t=[x for s in d.get('sports',{}).values() for x in s['tables'] if x['id'] in ('mlb_2026_season.game_predictions','mlb_2026_season.game_v10_features','mlb_2026_season.matchup_features')]
for x in t: print('  ', x['status'], x['id'], 'last change', x['modified'])"
    [ "${f:-1}" = "0" ] && [ "${d:-0}" -ge 1 ] && echo "RESULT: OK, the homelab has run checkpoints with no failures." || { echo "RESULT: NOT YET (no completed checkpoint, or a failure). If it failed: bash scripts/cutover_stage3.sh --rollback"; exit 1; }
    exit 0 ;;
esac

echo "Stage 3: per-game MLB (pregame)"
if ! readiness; then [ "${1:-}" = "--yes" ] && { echo "Refusing to go ahead."; exit 1; }; fi
echo
if [ "${1:-}" != "--yes" ]; then
  echo "Nothing changed. To go ahead: bash scripts/cutover_stage3.sh --yes"
  exit 0
fi
aet=$(gcloud scheduler jobs list --location "$LOCATION" --project "$PROJECT" --format='value(name.basename())' 2>/dev/null | grep -ci '^aet' || true)
[ "${aet:-1}" = "0" ] || { echo "Refusing: ${aet:-?} App Engine cron job(s) (aet-*) exist in Cloud Scheduler, and one of them may create per-game tasks. List them: gcloud scheduler jobs list --location $LOCATION --project $PROJECT | grep aet"; exit 1; }
echo "  checked: no App Engine cron job exists, so the Scheduler job is the only producer of per-game tasks"

echo "=== 1. stop the cloud producers"
$CLI control scheduler pause "$TWIN" --confirm "$TWIN" 2>&1 | tail -2 | sed 's/^/  pause twin: /'
[ "$(state "$TWIN")" = "PAUSED" ] || { echo "  STOP: Cloud Scheduler does not say PAUSED"; exit 1; }
for q in $QUEUES; do
  gcloud tasks queues pause "$q" --location "$LOCATION" --project "$PROJECT" 2>&1 | tail -1 | sed "s/^/  pause $q: /"
  [ "$(qstate "$q")" = "PAUSED" ] || { echo "  STOP: queue $q is not PAUSED"; rollback; exit 1; }
done
echo "=== 2. set pregame Live on the homelab (the app re-checks Cloud Scheduler and wants both acknowledgements)"
out=$(set_mode "pregame -> live" '{"job":"pregame","mode":"live","confirm":"pregame","acks":["cron","queues"]}'); echo "  $out"
case "$out" in *done*) ;; *) rollback; exit 1 ;; esac
echo "=== 3. plan today's games now instead of waiting for the next half hour"
ssh "$HOST" "sudo -n systemctl start pipeline-pregame-plan.service" 2>&1 | tail -2
n=$(sql live "select count(*) from tasks where state='pending'")
echo "  live store: ${n:-0} task(s) pending"
[ "${n:-0}" -ge 1 ] || { echo "  STOP: nothing was planned"; rollback; exit 1; }
echo "=== DONE. The homelab now runs every per-game checkpoint. After the next checkpoint has run:  bash scripts/cutover_stage3.sh --verify"
echo "    Undo at any time:  bash scripts/cutover_stage3.sh --rollback"
