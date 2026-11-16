# mllab reference (what `mllab ask` grounds its answers on)

mllab is a private ML lab on a home server: a model registry, BigQuery guards and checks, a job runner, backtests, training
and GCP deploys. Only use commands and spec keys listed here. If something is not listed, say you are not sure; never invent flags.

## Commands
- `mllab register PROJECT MODEL VERSION ARTIFACT [--parent V] [--lock FILE] [--features a,b]`  new immutable version
- `mllab adopt PROJECT MODEL VERSION gs://bucket/obj [--lock FILE] [--code FILE]`  import an existing GCS model
- `mllab list PROJECT MODEL` versions, channels (prod/staging pointers), deployments. `mllab promote PROJECT MODEL VERSION [--channel C]`
- `mllab data list` / `mllab data promote JOB FILE NAME`  files that backtests and training read (promote copies a finished job's output)
- `mllab runs` per-run verdicts (never ranks across runs); `mllab card PROJECT MODEL VERSION [--save]` model card
- `mllab envs list` / `mllab envs prune --older-than DAYS [--yes] [--include-untracked]` cached per-lock environments (5+ GB); prune is a dry run unless --yes and never touches an environment a job is using
- `mllab ask` this assistant
- `mllab weekly [--config specs/weekly.yaml] [--dry-run]` weekly data checks + backtests; writes audit/weekly.json; the health probe turns a failure into a Kuma alert
- `mllab job submit|status|logs|cancel|outputs`  jobs run as capped systemd units (memory, cpu, timeout are mandatory)
- `mllab bq describe TABLE` / `infer TABLE` / `diff A B` / `check SPEC.yaml [--max-gb N]`  (exit code 4 = a check failed)
- `mllab backtest run SPEC.yaml [--wait]` then `mllab backtest report JOB`
- `mllab train run SPEC.yaml [--wait]` then `mllab register-job JOB PROJECT MODEL VERSION --parent V`
- `mllab deploy cloudfn-swap PROJECT MODEL VERSION --object PATH [--dry-run] [--confirm "deploy P/M/V"] [--channel C]`,
  `cloudfn-rollback`, `cloudrun [--full-lock]` (lean image by default: only the libraries the pickle references, xgboost as xgboost-cpu). Real changes need the typed confirmation phrase; a dry run never changes anything.
- Notebook: `%load_ext mllab.magics` then `%%bq [--max-gb N] [--max-result-gb N] [--force] [varname]` runs SQL with the cost guards.

## BigQuery guards
Every query is dry-run first. Scans over 10 GiB are refused (`--max-gb`/`force=True` to override). Results over 1 GiB are refused
BEFORE download (select fewer columns, aggregate, or LIMIT). Results are cached as parquet; repeats within 24h are free.

## Backtest spec (YAML)
Keys: name, data ({sql_file: path} or {file: name.parquet}), history ({file, label}), env_lock (a requirements file, needed for retrain-only specs),
filter ({column, equals} or {column, in: [..]}, applied after features are built), label, date, id, start, end, models, columns, retrain,
baseline, tiers, thresholds, bootstrap, seed, env, timeout_s, mem_mb.
- models: `[{name, version: {project, model, version}, parity: true, predictor: path.py:load}]`
- columns: `[{name, column, reference: true}]` probability columns already in the data (e.g. prod's served probability, Elo)
- retrain: `[{name, features_from: <model name> | features: [..] | {all_numeric: true, exclude: [..], exclude_pattern: regex},
  family: xgboost|lightgbm|logreg, params: {..}, impute: median|zero, seeds: [ints], tune: {...},
  walk_forward: {min_train, step} | {group: season, min_groups: N}}]`
  Retrains use only games dated before the block they predict; `history` warm-starts them. In group mode `min_groups` is the only start
  threshold. `seeds` averages one xgboost/lightgbm model per seed (a single seed moves a season by 1-2 points). `impute: zero` is the
  football trainers' way; the default is training medians.
- tune: `{space: {param: [choices] | {low, high, log, type: int}}, trials, inner_folds, seed}` random search over ONLY the seasons before
  the evaluation window, then the frozen winner is scored on later seasons. Needs walk_forward.group and min_groups >= 3. The winner's
  inner score is best-of-N and optimistic; only the outer numbers count.
Contenders are scored on the same games with paired-bootstrap 95% CIs. A difference whose CI spans 0 is noise. About 2,000 games only
resolves differences of roughly 2-3 accuracy points. Picking the best of many retrains on the same games inflates it, and so does looking
at the same evaluation seasons many times.

## Train spec (YAML)
Keys: name, data, history, label, date, id, start, train_end, features | features_from ({project, model, version}), family, params, impute,
filter, columns, env, env_lock. `features` may be a list or {all_numeric: true, exclude: [..], exclude_pattern: regex}. Games after
`train_end` are an untouched holdout scored next to `columns` and a home prior. `mllab register-job` then records the model with lineage,
features, the hash of the exact input data and the pinned lock.

## Table check spec (YAML)
`table` (dataset.table), key [cols], date_column, max_staleness_days, min_rows, allow_extra, columns: [{name, type, nullable,
max_null_rate, min, max}]. Schema checks are free metadata; data checks are one aggregate query.

## Data traps (MLB)
Several feature rows per game (use the latest one at or before the prediction). `games` holds a skeleton row and a final row per game
(dedupe on synced_at). Predictions made after first pitch are backfills (exclude them). See sql/mlb_v10_pregame.sql.

## Football (NFL and CFB)
No model files exist: the repo refits XGBoost/ridge every run, so the lab exports the repo's own feature frame and registers dated
snapshots instead of adopting pickles. Workflow: `scripts/football/stage_export.sh` (stages the repo's src/nfl and src/cfb code, and a dump
of nfl_historical.team_week_epa, using your own gcloud credentials), then `mllab job submit /mnt/appdata/mllab/code/nfl_export/job.yaml`
(NFL) or `job_cfb.yaml` (CFB), then `mllab data promote JOB nfl_features.parquet nfl_features.parquet` (or cfb_features.parquet), then
`mllab backtest run specs/nfl_walkforward.yaml` / `cfb_fbs_walkforward.yaml` / `cfb_fcs_walkforward.yaml`.
The repo treats season 2025 as a TOUCH-ONCE holdout in both sports: the exports exclude it (by season, not date) unless --include-holdout.
`specs/nfl_ablation.yaml` measures what EPA adds on ONE frame (the repo's own --no-epa flag is confounded: it also moves the Elo start from
2006 to 1999). `specs/nfl_tune.yaml` is the nested tuning run. `specs/train_*_2023.yaml` train and register football snapshots.
Validated against the repo: NFL logistic regression matches train_nfl_models.py to four decimals, XGBoost to 0.05% with the same row order
and seed; CFB baselines match pipeline.baselines exactly. Findings so far (2017-2024, NFL, 2,202 games): feature models are NOT
distinguishable from Elo-only at default settings; tuned XGBoost beats Elo; market columns help. CFB (FBS and FCS): feature models beat Elo.

## Where things live (homelab)
Registry /mnt/appdata/mllab/registry, jobs /mnt/appdata/mllab/jobs, data /mnt/appdata/mllab/data. Services: mllab-api, mllab-jupyter,
mllab-caddy, mllab-spool.path, mllab-health.timer (pushes to Uptime Kuma), mllab-weekly.timer, mllab-backup.timer. Run heavy work as a job or inside the mllab slice.
