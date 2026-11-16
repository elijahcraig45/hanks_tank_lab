#!/usr/bin/env bash
# Snapshot the registry, audit log and job DB to /mnt/storage. Every run makes a NEW, uniquely named snapshot (a same-day rerun used to
# nest inside the previous one and silently leave the top-level snapshot stale). Unchanged files are hardlinked to the previous snapshot,
# so snapshots are cheap. A run builds in a hidden partial directory and renames it into place, so a crashed run never looks like a snapshot.
# Paths are overridable so this can be tested on a scratch directory.
set -euo pipefail
SRC=${MLLAB_ROOT:-/mnt/appdata/mllab}
DEST=${MLLAB_BACKUP_DEST:-/mnt/storage/backups/mllab}
KEEP=${MLLAB_BACKUP_KEEP:-14}
mkdir -p "$DEST"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$DEST/.lock"
  flock -n 9 || { echo "another backup is running; skipping"; exit 0; }
fi
STAMP=$(date +%Y%m%dT%H%M%S)
rm -rf "$DEST"/.*.partial
LATEST=$(ls -1d "$DEST"/20* 2>/dev/null | sort | tail -1 || true)
PART="$DEST/.$STAMP.partial"
mkdir -p "$PART"
DIRS=()
for d in registry audit db; do [ -d "$SRC/$d" ] && DIRS+=("$SRC/$d"); done
[ ${#DIRS[@]} -gt 0 ] || { echo "nothing to back up under $SRC" >&2; exit 1; }
rsync -a ${LATEST:+--link-dest="$LATEST"} "${DIRS[@]}" "$PART/"
mv "$PART" "$DEST/$STAMP"
echo "snapshot $DEST/$STAMP"
total=$(ls -1d "$DEST"/20* | wc -l)
if [ "$total" -gt "$KEEP" ]; then
  # Registry version dirs are read-only, which blocks deleting the files inside them for any non-root user. Unlinking a file only needs
  # write permission on its DIRECTORY, so relax directories only. NEVER chmod files here: they are hardlinked to the live registry.
  for old in $(ls -1d "$DEST"/20* | sort | head -n $((total - KEEP))); do
    find "$old" -type d -exec chmod u+w {} +
    rm -rf "$old"
  done
fi
