#!/usr/bin/env bash
# pipeline-runner-setup.sh — a least-privilege service account for running the pipeline on the homelab.
#
# Run it yourself, one step at a time (each step is safe to re-run):
#   bash pipeline-runner-setup.sh plan    print every command and change nothing
#   bash pipeline-runner-setup.sh sa      create the account and grant its roles, then verify
#   bash pipeline-runner-setup.sh verify  read-only: show everything the account can do (changes nothing)
#   bash pipeline-runner-setup.sh key     create a key and install it on the homelab (never touches this Mac's disk)
#   bash pipeline-runner-setup.sh test    read-only checks FROM the homelab with the new key
#   bash pipeline-runner-setup.sh test --write-probe   also create and delete one tiny probe table and one probe object
#   bash pipeline-runner-setup.sh revoke  delete every key and the account (full rollback)
#
# `key` and `test` use `ssh -t` because the homelab's sudo asks for your password. Run this from a real
# terminal, not from Claude Code's `!`, which has no tty.
#
# What the account gets, and why (from reading what the pipeline code does):
#   project  bigquery.jobUser           run queries
#   project  bigquery.readSessionUser   the BigQuery Storage API behind to_dataframe(); without it reads fail
#   dataset  bigquery.dataEditor        mlb_2026_season mlb_historical_data nfl_season nfl_historical
#                                       cfb_season cfb_historical pickem   (the tables the three jobs write)
#   dataset  bigquery.dataViewer        control (READ ONLY: the pipeline reads model_control_current to honour a pause, pin or tier
#                                       override; mllab is the only writer. The code fails open, so without this a pause set in the
#                                       control plane would silently not apply on the homelab)
#   bucket   storage.objectAdmin        gs://hanks_tank_data only (models, sp_quality parquet; needs overwrite)
#   secret   secretmanager.secretAccessor  cfbd-api-key only (the CFB `cfbd` job reads it; Cloud Functions injects it today)
# It does NOT get: write access to `control`, Vertex AI, Cloud Tasks, or anything project-wide.

set -uo pipefail
# A terminal whose working directory was deleted makes every child process print getcwd errors, and `bq show` then returns nothing
# (which this script used to report as "(none)"). Start somewhere that exists.
cd "$HOME" 2>/dev/null || cd / || true
PROJECT=hankstank
NAME=pipeline-runner
SA="$NAME@$PROJECT.iam.gserviceaccount.com"
BUCKET=hanks_tank_data
SECRET=cfbd-api-key
DATASETS=(mlb_2026_season mlb_historical_data nfl_season nfl_historical cfb_season cfb_historical pickem)
READ_DATASETS=(control)
HOST=homelab
KEYFILE=/etc/pipeline/sa.json
export CLOUDSDK_CORE_ACCOUNT=you@example.com   # the default gcloud account on this Mac is the work one

step=${1:-plan}; flag=${2:-}
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()  { if [ "$step" = plan ]; then printf '  (plan) would: %s\n' "$*"; else printf '  \033[32mOK\033[0m   %s\n' "$*"; fi; }
bad() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAILED=1; }
FAILED=0
run() { if [ "$step" = plan ]; then echo "  + $*"; else "$@"; fi; }

need_account() {
  [ "$(gcloud config get-value account 2>/dev/null)" = "$CLOUDSDK_CORE_ACCOUNT" ] || { echo "run: gcloud auth login (as $CLOUDSDK_CORE_ACCOUNT)"; exit 1; }
}

do_sa() {
  say "1. service account $SA"
  if gcloud iam service-accounts describe "$SA" --project=$PROJECT >/dev/null 2>&1; then ok "already exists"
  else run gcloud iam service-accounts create $NAME --project=$PROJECT --display-name="Pipeline runner (homelab)" \
         --description="Runs the MLB/NFL/CFB pipeline jobs from the homelab. Least privilege; created 2026-10-02." && ok "created"; fi

  say "2. project roles"
  for r in roles/bigquery.jobUser roles/bigquery.readSessionUser; do
    # A just-created account takes a few seconds to be visible to IAM; the first binding can fail with "does not exist".
    n=0; until run gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:$SA" --role="$r" --condition=None --quiet >/dev/null 2>&1; do
      n=$((n+1)); [ "$step" = plan ] && break
      if [ $n -ge 6 ]; then bad "$r (still failing after $n tries)"; continue 2; fi
      echo "  ... waiting for IAM to see the new account ($n/6)"; sleep 8
    done
    ok "$r"
  done

  say "3. dataset roles (dataEditor on exactly these)"
  for d in "${DATASETS[@]}"; do
    run bq query --use_legacy_sql=false --location=US --project_id=$PROJECT --quiet \
      "GRANT \`roles/bigquery.dataEditor\` ON SCHEMA \`$PROJECT.$d\` TO \"serviceAccount:$SA\"" >/dev/null && ok "$d"
  done

  say "3b. read-only dataset (the control plane's state; mllab is its only writer)"
  for d in "${READ_DATASETS[@]}"; do
    run bq query --use_legacy_sql=false --location=US --project_id=$PROJECT --quiet \
      "GRANT \`roles/bigquery.dataViewer\` ON SCHEMA \`$PROJECT.$d\` TO \"serviceAccount:$SA\"" >/dev/null && ok "$d (read-only)"
  done

  say "4. bucket role"
  run gcloud storage buckets add-iam-policy-binding gs://$BUCKET --member="serviceAccount:$SA" --role=roles/storage.objectAdmin >/dev/null && ok "gs://$BUCKET objectAdmin"

  say "5. secret (one secret, read-only)"
  run gcloud secrets add-iam-policy-binding $SECRET --project=$PROJECT --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor >/dev/null && ok "secret $SECRET secretAccessor"
}

verify_sa() {
  say "verify: everything this account can do"
  echo "  project roles:"
  gcloud projects get-iam-policy $PROJECT --flatten="bindings[].members" --filter="bindings.members:$SA" --format="value(bindings.role)" | sed 's/^/    /'
  echo "  datasets with an entry for it:"
  for d in "${DATASETS[@]}" "${READ_DATASETS[@]}"; do
    raw=$(bq show --format=prettyjson $PROJECT:$d 2>/dev/null)
    if [ -z "$raw" ]; then printf '    %-20s %s\n' "$d" "(COULD NOT READ the dataset: a gcloud/bq problem, not evidence about the grant)"; continue; fi
    r=$(printf '%s' "$raw" | python3 -c "
import sys,json
a=json.load(sys.stdin).get('access',[]); print(','.join(x['role'] for x in a if x.get('userByEmail')=='$SA'))" 2>/dev/null)
    printf '    %-20s %s\n' "$d" "${r:-(none)}"
  done
  echo "  secret $SECRET:"; gcloud secrets get-iam-policy $SECRET --project=$PROJECT --format=json 2>/dev/null | python3 -c "
import sys,json
for b in json.load(sys.stdin).get('bindings',[]):
    if 'serviceAccount:$SA' in b['members']: print('   ',b['role'])"
  echo "  bucket:"; gcloud storage buckets get-iam-policy gs://$BUCKET --format=json | python3 -c "
import sys,json
for b in json.load(sys.stdin).get('bindings',[]):
    if 'serviceAccount:$SA' in b['members']: print('   ',b['role'])"
  echo "  (control should show READER only: it reads the control state and never writes it)"
}

do_key() {
  say "key -> $HOST:$KEYFILE   (the key is streamed over ssh; it is never written on this Mac)"
  echo "  This asks for your sudo password on $HOST. Existing keys for the account:"
  gcloud iam service-accounts keys list --iam-account="$SA" --managed-by=user --format="table(name.basename(),validAfterTime.date())"
  read -r -p "  Create a NEW key and install it? [y/N] " a </dev/tty; [[ "$a" =~ ^[Yy]$ ]] || { echo "  skipped"; return; }
  ssh -t $HOST 'sudo groupadd -f pipeline && sudo install -d -m 0750 -o root -g pipeline /etc/pipeline' || { bad "could not prepare /etc/pipeline"; return; }
  # stage in the homelab user's home with mode 0600, then move into place as root:pipeline 0640 and shred the stage copy
  gcloud iam service-accounts keys create /dev/stdout --iam-account="$SA" 2>/dev/null \
    | ssh $HOST 'umask 077; cat > ~/.pipeline-sa.json' || { bad "key create/stream"; return; }
  ssh -t $HOST "sudo install -m 0640 -o root -g pipeline ~/.pipeline-sa.json $KEYFILE && shred -u ~/.pipeline-sa.json && sudo ls -l $KEYFILE" && ok "installed $KEYFILE (root:pipeline 0640)" || bad "install"
}

do_test() {
  say "test from $HOST with the new key (read-only unless --write-probe)"
  PROBE=0; [ "$flag" = "--write-probe" ] && PROBE=1
  # Use the installed pipeline venv (it has the BigQuery, Storage and Secret Manager clients). Read the key directly when this user is
  # already in the `pipeline` group; otherwise through sudo, which asks for the password.
  ssh -t $HOST "cd ~ && for c in /opt/pipeline/venv/bin/python; do [ -x \$c ] && \$c -c 'from google.cloud import secretmanager' 2>/dev/null && PYBIN=\$c && break; done; [ -n \"\${PYBIN:-}\" ] || { echo 'the pipeline venv is missing or lacks google-cloud-secret-manager: run host/pipeline/install.sh first'; exit 1; }; if [ -r $KEYFILE ]; then RUN=''; else RUN='sudo -u homelab -g pipeline'; fi; \$RUN env GOOGLE_APPLICATION_CREDENTIALS=$KEYFILE PROBE=$PROBE PROJECT=$PROJECT BUCKET=$BUCKET \$PYBIN - <<'PY'
import os, time
from google.cloud import bigquery, storage
P, B, probe = os.environ['PROJECT'], os.environ['BUCKET'], os.environ['PROBE'] == '1'
bq = bigquery.Client(project=P)
def check(name, fn):
    try: print('  OK  ', name, fn() or '')
    except Exception as e: print('  FAIL', name, str(e).splitlines()[0][:150])
check('identity', lambda: getattr(bq._credentials, 'service_account_email', '?'))
check('query (jobUser)', lambda: bq.query('SELECT 1').result() and '')
for d in ['mlb_2026_season','mlb_historical_data','nfl_season','nfl_historical','cfb_season','cfb_historical','pickem']:
    check('read ' + d, lambda d=d: f'{len(list(bq.list_tables(f\"{P}.{d}\", max_results=3)))}+ tables listed')
check('storage-api read (readSessionUser)', lambda: f'{len(bq.query(f\"SELECT * FROM \`{P}.mlb_2026_season.games\` LIMIT 5\").to_dataframe(create_bqstorage_client=True))} rows')
check('read control.model_control_current (read-only)', lambda: f'{len(list(bq.query(f\"SELECT * FROM \`{P}.control.model_control_current\` LIMIT 5\").result()))} current rows')
if probe:
    ct = f'{P}.control._sa_probe_{int(time.time())}'
    try:
        bq.create_table(bigquery.Table(ct, schema=[bigquery.SchemaField('x','INTEGER')])); bq.delete_table(ct); print('  FAIL control is WRITABLE (it must be read-only)')
    except Exception:
        print('  OK   control is not writable, as intended')
def secret_check():
    from google.cloud import secretmanager
    v = secretmanager.SecretManagerServiceClient().access_secret_version(name=f'projects/{P}/secrets/cfbd-api-key/versions/latest').payload.data
    return f'readable ({len(v)} bytes; the value is not printed)'
check('secret cfbd-api-key (secretAccessor)', secret_check)
st = storage.Client(project=P)
check('list bucket', lambda: f'{len(list(st.list_blobs(B, max_results=3)))}+ objects')
if probe:
    t = f'{P}.mlb_2026_season._sa_probe_{int(time.time())}'
    check('write probe table (dataEditor)', lambda: (bq.create_table(bigquery.Table(t, schema=[bigquery.SchemaField('x','INTEGER')])), bq.delete_table(t), 'created and deleted')[2])
    check('write probe object (objectAdmin)', lambda: (lambda b: (b.upload_from_string('probe'), b.upload_from_string('probe2'), b.delete(), 'created, overwritten, deleted')[3])(st.bucket(B).blob(f'probe/sa-test-{int(time.time())}.txt')))
PY"
}

do_revoke() {
  say "revoke: delete the account's keys and the account"
  read -r -p "  This deletes $SA and every key it has. Continue? [y/N] " a </dev/tty; [[ "$a" =~ ^[Yy]$ ]] || { echo "  skipped"; return; }
  gcloud iam service-accounts delete "$SA" --project=$PROJECT --quiet && ok "deleted (its keys die with it)"
  echo "  On the homelab also: ssh -t $HOST 'sudo shred -u $KEYFILE'"
}

case "$step" in
  plan)   do_sa; echo; echo "  then: key, test" ;;
  sa)     need_account; do_sa; verify_sa ;;
  verify) need_account; verify_sa ;;
  key)    need_account; do_key ;;
  test)   need_account; do_test ;;
  revoke) need_account; do_revoke ;;
  *)      sed -n '2,13p' "$0"; exit 2 ;;
esac
[ "$FAILED" = 0 ] || { echo; echo "some steps failed (see FAIL above)"; exit 1; }
