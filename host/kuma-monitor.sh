#!/usr/bin/env bash
# Add the "Model Lab (mllab)" push monitor to Uptime Kuma and start the health timer. Run on homelab as root.
# RUN THIS YOURSELF: it edits Kuma's database and restarts the uptime-kuma container (about 10 s gap in all monitors).
#   sudo ./kuma-monitor.sh --dry-run    show what would happen
#   sudo ./kuma-monitor.sh              back up kuma.db, insert the monitor, install the timer, restart Kuma, verify
#   sudo ./kuma-monitor.sh --remove     delete the monitor row and stop the timer (restore the backup for a full undo)
#
# The monitor is a clone of the existing "Host health (pihome)" push monitor (same defaults, interval 300 s).
set -euo pipefail

DB=${KUMA_DB:-/mnt/appdata/uptime-kuma/kuma.db}
TEMPLATE_ID=${KUMA_TEMPLATE_ID:-13}
NAME='Model Lab (mllab)'
DRY=0; REMOVE=0
for a in "$@"; do
  case "$a" in --dry-run) DRY=1 ;; --remove) REMOVE=1 ;; *) echo "unknown arg $a" >&2; exit 2 ;; esac
done
[ "$(id -u)" = 0 ] || { echo "run as root (sudo)" >&2; exit 1; }
run() { if [ "$DRY" = 1 ]; then echo "[dry-run] $*"; else "$@"; fi; }
HERE=$(cd "$(dirname "$0")" && pwd)

exists() { sqlite3 "$DB" "select count(*) from monitor where name='$NAME'"; }

if [ "$REMOVE" = 1 ]; then
  run sqlite3 "$DB" ".timeout 10000" "delete from monitor where name='$NAME'"
  run systemctl disable --now mllab-health.timer
  run sed -i '/^MLLAB_KUMA_PUSH_TOKEN=/d' /etc/mllab/env
  run docker restart uptime-kuma
  echo "removed."; exit 0
fi

[ "$(sqlite3 "$DB" "select type from monitor where id=$TEMPLATE_ID")" = push ] \
  || { echo "monitor $TEMPLATE_ID is not a push monitor; set KUMA_TEMPLATE_ID" >&2; exit 1; }
[ "$(exists)" = 0 ] || { echo "monitor '$NAME' already exists; use --remove first" >&2; exit 1; }
grep -q '^MLLAB_KUMA_PUSH_TOKEN=' /etc/mllab/env && { echo "token already in /etc/mllab/env" >&2; exit 1; }

BK="$DB.bak-mllab-$(date +%Y%m%d-%H%M%S)"
TOKEN=$(head -c 200 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 20)
echo "== backup -> $BK"
run sqlite3 "$DB" ".backup '$BK'"

echo "== insert monitor (clone of #$TEMPLATE_ID)"
SQL="
.timeout 10000
BEGIN;
CREATE TEMP TABLE t AS SELECT * FROM monitor WHERE id=$TEMPLATE_ID;
UPDATE t SET id=NULL, name='$NAME', push_token='$TOKEN', created_date=datetime('now'),
  description='mllab: units, /healthz, auth fails closed, job spool. Pushed every 2 min by mllab-health.timer';
INSERT INTO monitor SELECT * FROM t;
COMMIT;
"
if [ "$DRY" = 1 ]; then echo "[dry-run] sqlite3 $DB <<SQL"; echo "$SQL" | sed "s/$TOKEN/<token>/"; echo "SQL"; else printf '%s' "$SQL" | sqlite3 "$DB"; fi

echo "== token -> /etc/mllab/env, install timer"
if [ "$DRY" = 1 ]; then echo "[dry-run] append MLLAB_KUMA_PUSH_TOKEN to /etc/mllab/env"; else echo "MLLAB_KUMA_PUSH_TOKEN=$TOKEN" >> /etc/mllab/env; fi
run "$HERE/install.sh"

echo "== restart Kuma so it starts watching the new monitor"
run docker restart uptime-kuma

if [ "$DRY" = 1 ]; then exit 0; fi
echo "== verify"
sleep 15
systemctl start mllab-health.service || true
sleep 3
ID=$(sqlite3 "$DB" "select id from monitor where name='$NAME'")
sqlite3 -header -column "$DB" "select monitor_id, status, msg, time from heartbeat where monitor_id=$ID order by id desc limit 3"
echo "status 1 = up. If empty, wait 2 minutes: the timer pushes every 2 min."
