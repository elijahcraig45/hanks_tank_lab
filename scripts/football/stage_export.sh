#!/usr/bin/env bash
# Stage the NFL export job on homelab: the repo's src/nfl code + export_nfl.py + a job spec, and the EPA dump as a lab data file.
# Reads BigQuery with YOUR gcloud credentials (read-only, one 1 MiB table); nothing here needs new IAM for the lab.
#   scripts/football/stage_export.sh [path/to/hanks_tank_ml]
set -euo pipefail
REPO=${1:-$HOME/Documents/personalDev/mlb/hanks_tank_ml}
HERE=$(cd "$(dirname "$0")" && pwd)
export CLOUDSDK_CORE_ACCOUNT=you@example.com
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
# Mirror the repo layout (src/nfl/...): the repo's config.py roots its data cache two levels above itself, so a flat copy
# would write outside the job's directory.
mkdir -p "$T/stage/src/nfl"
cp "$REPO"/src/nfl/*.py "$T/stage/src/nfl/"
mkdir -p "$T/stage/src/cfb"
cp "$REPO"/src/cfb/*.py "$T/stage/src/cfb/"
{ cat "$REPO/src/nfl/requirements.txt"; echo "requests==2.32.5"; } > "$T/stage/requirements_cfb.txt"   # espn_data needs requests
cp "$HERE/export_cfb.py" "$T/stage/"
cp "$REPO"/src/nfl/requirements.txt "$T/stage/requirements.txt"
cp "$HERE/export_nfl.py" "$HERE/run_nfl_trainer.py" "$T/stage/"
cat > "$T/stage/trainer.yaml" <<'YAML'
name: nfl-trainer-crosscheck
script: run_nfl_trainer.py
code_dir: .
lock: requirements.txt
inputs: [team_week_epa.parquet]
timeout_s: 1800
mem_mb: 2048
cpu_pct: 200
YAML
cat > "$T/stage/job_cfb.yaml" <<'YAML'
name: cfb-export
script: export_cfb.py
args: ["--src", "src", "--out", "outputs"]
code_dir: .
lock: requirements_cfb.txt
timeout_s: 2400
mem_mb: 2048
cpu_pct: 100
YAML
cat > "$T/stage/job_noepa.yaml" <<'YAML'
name: nfl-export-noepa
script: export_nfl.py
args: ["--src", "src/nfl", "--no-epa", "--out", "outputs"]
code_dir: .
lock: requirements.txt
timeout_s: 1800
mem_mb: 2048
cpu_pct: 100
YAML
cat > "$T/stage/job.yaml" <<'YAML'
name: nfl-export
script: export_nfl.py
args: ["--src", "src/nfl", "--epa", "../inputs/team_week_epa.parquet", "--out", "outputs"]
code_dir: .
lock: requirements.txt
inputs: [team_week_epa.parquet]
timeout_s: 1800
mem_mb: 2048
cpu_pct: 100
YAML
bq query --nouse_legacy_sql --format=csv --max_rows=50000 'SELECT * FROM `hankstank.nfl_historical.team_week_epa` ORDER BY season, week, team' > "$T/epa.csv" 2>/dev/null
python3 -c "import pandas as pd, sys; d=pd.read_csv('$T/epa.csv'); d.to_parquet('$T/team_week_epa.parquet'); print(len(d), 'EPA rows, seasons', d.season.min(), '-', d.season.max())"
rsync -a --delete "$T/stage/" homelab:/tmp/nfl_export_stage/
scp -q "$T/team_week_epa.parquet" homelab:/tmp/team_week_epa.parquet
ssh homelab 'set -e
  sudo install -d -o mllab -g mllab -m 0750 /mnt/appdata/mllab/code /mnt/appdata/mllab/code/nfl_export
  sudo rsync -a --delete --chown=mllab:mllab /tmp/nfl_export_stage/ /mnt/appdata/mllab/code/nfl_export/
  sudo install -o mllab -g mllab -m 0640 /tmp/team_week_epa.parquet /mnt/appdata/mllab/data/team_week_epa.parquet
  rm -rf /tmp/nfl_export_stage /tmp/team_week_epa.parquet
  sudo ls /mnt/appdata/mllab/code/nfl_export | tr "\n" " "; echo'
echo "staged. Submit with:  mllab job submit /mnt/appdata/mllab/code/nfl_export/job.yaml   (trainer.yaml = cross-check, job_noepa.yaml = the repo's own EPA ablation, job_cfb.yaml = CFB export)"
