# Moving the prediction pipeline to the homelab

Status: **Done, 2026-10-05.** All 13 pieces (12 scheduled jobs and the per-game MLB planner) run live on the homelab. The Cloud Functions, their paused Cloud Scheduler jobs and the paused Cloud Tasks queues remain as a manual fallback. The stages below are the record of how it was done. Release ids in this doc are the names of releases installed before the 2026-11-16 history rewrite of `hanks_tank_ml`; the same commits now have new ids there.
Owner: Henry. Scope: every scheduled job that today runs as a Cloud Function (MLB, NFL, CFB), including the per-game pregame tasks.
Stays on GCP: hosting (App Engine, `hankstank.com`), storage (BigQuery, `gs://hanks_tank_data`). The homelab reads and writes both; it does not host the site.

## Why, and what it will and will not buy

- **Control.** No 540 s / 1 GB / 3-instance limits, a place to train (weekly V10 training is paused because it fails in the function), logs on a disk you own, and `mllab` as the one surface that runs, pauses and promotes things.
- **Money: small.** About $6-7 a month (function compute ~$3, 23 GiB of function build images ~$2.30, Scheduler ~$1.20). BigQuery query bytes do not move: BigQuery bills the bytes scanned wherever the query runs. App Engine was the big cost and is handled separately.
- **Speed: no change.** Measured on the homelab with one BLAS thread, the same steps take about as long as in the function (matchup 15.8 s vs 15.7 s, v7 19.6 vs 20.2, v8 5.3 vs 6.4); daily `features` is 2.4x faster (60 s vs 146 s).
- **The real cost: you become the SRE.** A power or ISP outage, a dead disk or a bad reboot now stops predictions. See Risks.

## Verified (Stage 0)

| Check | Result |
|---|---|
| Pinned dependencies install on the homelab (Python 3.13, no sudo) | yes, 60 s, 2.0 GB venv |
| Test suite on the homelab, `OMP_NUM_THREADS=1` | 342 pass / 1 fail (`polars` missing, same on the Mac), 24 s |
| Default BLAS threads | **19x slower** (138 s vs 7.4 s for one test). Every unit sets the thread env vars to 1. |
| `pipeline-runner` service account | created; project: jobUser + readSessionUser; dataEditor on 7 datasets; objectAdmin on `hanks_tank_data` only; write probes passed. `control` was denied at that point; it is now **read-only** (see below) |
| `daily` dry run as that account | HTTP 200, 94 s, peak 203 MB |
| `pregame_v10` dry run, one game | HTTP 200, 67.5 s, peak 296 MB (`v10_features` and `predict_today` did no work: the game was already played) |
| Production V10 model (`model.pkl`, 73 features) loads and scores on the homelab | yes |
| MLB API, Savant, BigQuery, Cloud Storage reachable; Google RTT 10-14 ms | yes |
| Merged requirements (24 packages) install into a **Python 3.12** venv with `uv` | yes, 12.5 s; all three bundles pass preflight on 3.12 (the functions' runtime) |
| Staged bundles reproduce the deploy scripts (flattened NFL and CFB trees, their own `config.py`) | yes: verified against the real checkout, each bundle imports in its own process on the homelab |
| Every schedule is accepted by `systemd-analyze calendar` on the homelab, with the right next-fire times | yes (New York time, DST by timezone suffix) |
| Generated units pass `systemd-analyze verify` | yes (`RuntimeMaxSec` is ignored for oneshot units, so `TimeoutStartSec` is used) |
| Installer `--dry-run` on the homelab (root check stubbed) | preflight passes (uv, git, rsync, group, key); prints every action; writes nothing |

**Not verified:** the installer as root; a real (non-dry) write; `predict_today` on an upcoming game; `sim-blend` (peak 1.3-2.5 GB) and football jobs end to end; home outage behaviour; reboot behaviour.

## Decisions (defaults; each is reversible)

1. **Code stays in `hanks_tank_ml`.** The homelab runs a checkout pinned to a tag. `mllab/host/pipeline/` owns the host layer (units, installer, runner). No pipeline code changes in Stage 1.
2. **Reuse the three entrypoints** (`daily_pipeline`, `nfl_pipeline`, `cfb_pipeline`) from a thin runner, sending the exact bodies Cloud Scheduler sends today. Behaviour matches production by construction.
3. **systemd timers, `OnCalendar=... America/New_York`, `Persistent=true`** so a reboot catches up and DST is handled by the timezone suffix.
4. **A dedicated `pipeline` system user**, a hardened unit (`ProtectSystem=strict`, `NoNewPrivileges`, `PrivateTmp`), `MemoryMax`, `CPUQuota`, `TimeoutStartSec`, and one run of a given mode at a time (`flock`). Key at `/etc/pipeline/sa.json`, `root:pipeline 0640`.
5. **Per-game scheduling is a planner plus a one-minute ticker with sqlite state**, using today's offsets. Idempotent; skips games already started; survives reboots.
6. **Safety net: cutover one job at a time; the Cloud Function stays deployed but idle as a manual fallback.** No automatic fallback: two writers at once is worse than one gap. Freshness alerts come from `/health/scheduler` (already reports the age of each league's newest prediction) and an Uptime Kuma push monitor.
7. **`mllab` becomes the control surface in Stage 6**, not Stage 1. Moving the jobs first keeps the blast radius small.

## What exists today (the thing being replaced)

Cloud Scheduler, project `hankstank`, us-central1, all `America/New_York`:

| Job | Cron | State | Body |
|---|---|---|---|
| mlb-2026-daily | `0 4 * 3-11 *` | on | `daily` |
| mlb-2026-roster-refresh | `0 3 * 3-11 1` | on | `rosters` |
| mlb-2026-power-rankings | `30 6 * 3-11 1` | on | `power_rankings` |
| mlb-2026-pregame-schedule | `0 10 * 3-11 *` | on | `schedule_pregame_tasks` |
| mlb-2026-weekly-train-v10 | `0 2 * 3-11 0` | **paused** (failed every Sunday) | `train_weekly`, v10 |
| mlb-2026-weekly-predict | `0 5 * 3-11 5` | **paused** (wrote a table nothing reads) | `predict` |
| nfl-weekly-ingest | `0 6 * 9-12,1,2 2` | on | `ingest` |
| nfl-weekly-predict | `0 6 * 9-12,1,2 3` | on | `predict_week` |
| nfl-weekly-score | `0 7 * 9-12,1,2 2` | on | `score` |
| nfl-weekly-season-sim | `30 7 * 9-12,1 2` | on | `season_sim` |
| cfb-weekly-ingest | `0 6 * 8-12,1 0` | on | `ingest` |
| cfb-weekly-cfbd | `0 7 * 8-12,1 0` | on | `cfbd` |
| cfb-weekly-drives | `30 7 * 8-12,1 0` | on | `drives` |
| cfb-weekly-season-sim | `0 8 * 8-12,1 0` | on | `season_sim` |
| cfb-weekly-predict | `0 6 * 8-12,1 2` | on | `predict_next` |

Per-game (MLB): the backend creates Cloud Tasks from `/api/lineup/schedule-today`: `pregame_v10` at **T-360, -180, -90, -45 min** (queue `lineup-pregame`, 5 concurrent, 3 attempts) and the PA-sim `sim_blend` at **T-80 and T-35** (queue `sim-blend`). `schedule-today` is called by **two** things at 10:00 ET: the `mlb-2026-pregame-schedule` job (via the function) and the backend's own App Engine `cron.yaml` (every day, year-round). **Both must be turned off at the pregame cutover**; the second needs a backend deploy.

## Stages

| # | Stage | Touches production? | Rollback |
|---|---|---|---|
| 0 | Service account + spike | no | `pipeline-runner-setup.sh revoke` |
| 1 | **Build** (done): runner (lock, logs, Kuma push), planner + ticker, `pipeline-ctl`, stager, unit generator, installer, 121 tests | no | delete the branch |
| 2 | **Install** on the homelab (done 2026-10-02); every mode on a timer in `dry` | no (dry-run only) | `install.sh --uninstall` |
| 3 | **Cut over batch jobs**, one at a time: football weekly (10 jobs), power rankings, roster refresh, then `daily`. Per job: pause the Scheduler job, enable the timer, watch the next run | yes | resume the Scheduler job, disable the timer |
| 4 | **Cut over pregame**: pause `mlb-2026-pregame-schedule`, drop the `cron.yaml` entry (backend deploy), purge/pause the `lineup-pregame` and `sim-blend` queues so nothing in flight double-runs, enable planner + ticker | yes | re-add the cron entry, resume the job and queues, disable the planner |
| 5 | **Retire**: after four clean weeks, delete the functions, jobs, queues and function images (Artifact Registry); keep the deploy scripts | yes | redeploy from the repo |
| 6 | **`mllab` control**: the registry becomes the source of truth for the production model (`promote` replaces the function swap), `mllab pipeline status/run/pause`, weekly V10 training re-enabled on the homelab | no | n/a |

Cutover order is chosen so the first things moved are the ones nobody notices for a week (football runs once a week) and the last is the one the site shows every day.

## Risks

| Risk | Mitigation |
|---|---|
| Power / ISP / hardware loss stops predictions | Kuma push monitor + `/health/scheduler` freshness; a documented manual fallback (re-enable the Scheduler job, resume the queues). A UPS and a second Pi as watchdog are worth considering but are not part of this plan. |
| Reboot or a missed timer | `Persistent=true`; the ticker catches up inside a window and skips games already started |
| Two writers during cutover | Pause first, then enable; never both. Writes are DELETE-before-INSERT or partition overwrite, so a double run is mostly idempotent but wastes work and can race. |
| BLAS thread oversubscription (19x slowdown) | Thread env vars set in every unit; asserted by a startup check |
| Key theft from a home server | Dedicated user, root:pipeline 0640, scope limited to 7 datasets and one bucket; `control` and project-wide roles excluded; rotate on a schedule |
| Python 3.13 here vs 3.12 in the function | Suite passes and the V10 model loads on 3.13; pin the same package versions |
| Silent breakage on the homelab | Every run writes a status row; `mllab health` reads it; stale = alert |
| Backend still creates per-game tasks after cutover | Stage 4 removes both triggers and drains both queues; verified by checking queue depth and function request logs |
| Homelab is shared with Immich, Jupyter, Samba, Docker | `MemoryMax`/`CPUQuota` on the pipeline slice; measured peak is 296 MB |

## Out of scope for this plan

App Engine hosting, the backend's other cron jobs (`daily-check`, `weekly-full`, `monthly-audit`, `sync/missing`), the news fetch, and BigQuery spend (the scouting-report fix and the response caching are separate).

## Resolved 2026-10-02: `main` now equals production

The model control plane (backend #19, frontend #14, ML #18) had been merged into the stacked branch `deploy/2026-09-28b`, never into `main`; production ran it (live backend `control-3`, frontend `20260929t214008` and function revision `00040` were deployed from `model-control`). It was reconciled the same day with three `model-control` -> `main` PRs (backend #21, frontend #16, ML #21), merged with merge commits after a conflict-free trial merge and passing suites (backend 369, frontend 278, ML 428 with the one known `polars` failure).

A deploy from `main` now changes only: the pipeline-health check and the cache fix (backend), `robots.txt` (frontend), and the scoped scouting reports (ML). **PRs go to `main`; never base one on a deploy branch.**

The homelab was re-staged from the reconciled `main` the same day (release `dad6ca7bd8fc`), after the two follow-ups below. **Done: the installed bundles carry the control plane.**

**Re-staging has a prerequisite.** The control-plane code reads `control.model_control_current` (`model_control.py`) and **fails open**: if it cannot read the view it logs one warning and runs with defaults (not paused, no pin, no tier override). The `pipeline-runner` account was created with `control` denied, which was right for the old `main`. With the new one it would make the control plane silently inert on the homelab: a pause or pin set in the lab would apply to the Cloud Function and be ignored here. So before the re-stage, run `pipeline-runner-setup.sh sa` (it now grants `bigquery.dataViewer` on `control`, **read-only**; `mllab` stays the only writer) and `test` (it checks that `control` is readable, and with `--write-probe` that it is not writable). Today the control table has 0 rows, so nothing differs yet; that is why it is easy to miss.

## What Stage 1 built

All of it is in `mllab`: `src/mllab/pipeline/` (standard library only), `host/pipeline/` (units and installer), `tests/test_pipeline_*.py` (121 tests).

| Piece | What it does |
|---|---|
| `jobs.py` | The job table: 14 jobs (12 installed, 2 paused upstream), each with the exact body and schedule Cloud Scheduler has, its bundle, and whether its entrypoint accepts `dry_run`. Also the cron-to-systemd converter. |
| `stage.py` | Builds the `mlb`, `nfl` and `cfb` bundles from a `hanks_tank_ml` checkout exactly as `deploy_nfl.sh` and `deploy_cfb.sh` lay them out, and merges the three requirement sets (24 packages; two different exact pins for one package is an error). |
| `runner.py` | Runs one job in a subprocess against its bundle with the function's own environment, thread caps and secrets (`sm://` references), one run at a time, with a timeout that kills the whole process group. Records every run. |
| `pregame.py` | `plan` reads the MLB schedule and records the backend's tasks (baseline, T-360/-180/-90/-45, sim T-80/-35); `tick` runs what is due, 3 at a time, retries 3 times, never runs a started game. State is sqlite, one file per mode. |
| `ctl.py` | `pipeline-ctl list / status / set / run`. Going live prints what must already be off in GCP and asks first. |
| `units.py`, `host/pipeline/units/` | The systemd units, generated from the job table and checked in. A test fails if they go stale. |
| `host/pipeline/install.sh` | Root installer: user, directories, 3.12 venv, release staging with a preflight of all three entrypoints before `current` moves, units, `pipeline-ctl`. Every job starts `off`. |

### The mode file is the whole safety model

`/etc/pipeline/jobs.d/<job>` holds one word:

| Mode | What a timer firing does |
|---|---|
| `off` (default) | nothing |
| `dry` | the entrypoint with `dry_run=true`; for jobs whose entrypoint refuses it (NFL/CFB `ingest`, `score`, `cfbd`) a **preflight**: import the bundle, resolve its secrets, run nothing |
| `live` | the real run, then a heartbeat to Uptime Kuma |

A missing, empty or unrecognised file is `off`, never `live`. Tests pin that, and the lock, the thread caps, the environment and `dry_run` handling, and each was mutation-tested.

## Runbook

### Stage 2: install (you, as root on the homelab)

```
# from your laptop
rsync -a --exclude .venv --exclude .git ~/Documents/personalDev/mllab/ homelab:~/mllab/
ssh -t homelab 'cd ~/mllab/host/pipeline && sudo ./install.sh --dry-run'     # read it
ssh -t homelab 'cd ~/mllab/host/pipeline && sudo ./install.sh'
```

Then, as a member of the `pipeline` group (no sudo):

```
pipeline-ctl list                      # 12 jobs + pregame, all off
pipeline-ctl run mlb-2026-daily --dry  # a real dry run; writes nothing
pipeline-ctl set mlb-2026-daily dry    # and let the timer do it tomorrow
```

Updating the pipeline code later is `sudo ./install.sh --release [--ref <tag>]`: it stages a new release, preflights all three bundles and only then switches `current`.

The service account and its key come from `host/pipeline/pipeline-runner-setup.sh` (`plan`, `sa`, `key`, `test [--write-probe]`, `revoke`). `sa` also grants read on the one secret the CFB `cfbd` job needs (`cfbd-api-key`); `test` proves the grant without printing the value. Rebuilding the homelab starts there.

### Stage 3: cut over one batch job

For each job, in this order: the ten football jobs, `mlb-2026-power-rankings`, `mlb-2026-roster-refresh`, `mlb-2026-daily`.

1. `pipeline-ctl set <job> dry`, wait for (or `run --dry`) one dry run; `pipeline-ctl status` shows it ok.
2. From the laptop: `mllab control scheduler pause <job> --confirm <job>` (audited).
3. `pipeline-ctl set <job> live` (it asks; the Scheduler job must already be paused).
4. After its next scheduled run: `pipeline-ctl status` ok, the expected rows landed in BigQuery, `mllab health` shows nothing new.

**Rollback:** `pipeline-ctl set <job> off`, then `mllab control scheduler resume <job> --confirm <job>`.

Note that `mllab health`'s data rules (`no_output_today`, `coverage_gap`) read BigQuery, so they stay a freshness check that does not care who ran the job; its scheduler rule does not flag paused jobs.

### Stage 4: cut over per-game scheduling (do this in the off-season or on a day without games first)

1. `pipeline-ctl set pregame dry`, let `plan` and `tick` run for a day or two on real games; compare the plan against the Cloud Tasks queue (`gcloud tasks list --queue=lineup-pregame --location=us-central1`): the same games, the same fire times.
2. Pause the Scheduler job: `mllab control scheduler pause mlb-2026-pregame-schedule --confirm mlb-2026-pregame-schedule`.
3. Remove the `/api/lineup/schedule-today` entry from `hanks_tank_backend/cron.yaml`, then `gcloud app deploy cron.yaml`.
4. Pause both queues: `gcloud tasks queues pause lineup-pregame sim-blend --location us-central1`.
5. `pipeline-ctl set pregame live`.

**Rollback:** `pipeline-ctl set pregame off`; restore the cron entry and redeploy it; resume the queues and the Scheduler job.

### Monitoring

- `pipeline-ctl status` is the quick view; `/var/lib/pipeline/state/runs.jsonl` is the history; each run's log is under `/var/lib/pipeline/state/logs/<job>/` (the last 30 per job); `journalctl -u 'pipeline-*'` has the unit logs.
- Create an Uptime Kuma **push** monitor (heartbeat interval 10 min is enough: the ticker beats about once a minute while live) and put its URL in `/etc/pipeline/env` as `PIPELINE_KUMA_PUSH_URL`. Live runs push up or down; the ticker pushes while it works.
- `/health/scheduler` on the site reports the age of each league's newest prediction, and `mllab health` checks that tables are filling: both are independent of the homelab.

## Known issues and open items

- `numpy==2.4.0`, pinned in the functions' requirements, is **yanked** on PyPI ("backward compatibility bug"). It still installs because it is pinned exactly; worth bumping in `hanks_tank_ml`.
- The football `ingest` and `score` modes cannot be dry-run (their entrypoints return 400), so their dry mode is a preflight only. Their first real run is therefore also the first real test.
- Not built yet: Stage 6 (`mllab` control integration), and a UPS or watchdog for the homelab.
- MLB `train_weekly` and `predict` stay paused: they were broken in GCP and are not part of this move.

## Stage 2 log (2026-10-02)

- Installed with `install.sh`; release `8c8a5154568e` staged from `main`; 14 timers enabled; all jobs and `pregame` set to `dry`.
- **The first install attempt failed its release preflight, as designed, and changed nothing.** `uv` had put Python 3.12 under `/root/.local/share/uv`, so the venv worked for root and failed for the `pipeline` user. Fixed: the interpreter now lives in `/opt/pipeline/python`, and the venv is recreated whenever the pipeline user cannot run it (a plain existence check is true for root even when it is broken for everyone else).
- Verified after install: code owned by root, state by `pipeline`, `/etc/pipeline` group-only, `jobs.d` group-writable with setgid; all 14 timers scheduled at the right New York times; the planner and ticker ran under systemd as `User=pipeline` in `pipeline.slice` with a 5 GiB cap, wrote their own state, `Result=success`.
- The interactive `homelab` login is in the `pipeline` group, which `pipeline-ctl set` needs (it writes `/etc/pipeline/jobs.d` without sudo). The cost: anything running as that login can read the service-account key.

### What the weekend dry runs exercise

| When (ET) | Jobs | Dry means |
|---|---|---|
| nightly 04:00 | `mlb-2026-daily` | the real entrypoint with `dry_run=true` (about 94 s) |
| Sun 06:00-08:00 | `cfb-weekly-ingest`, `-cfbd` | preflight only: import the bundle, resolve the CFBD secret |
| Sun 07:30-08:00 | `cfb-weekly-drives`, `-season-sim` | real computation, no writes (drives spends at most 3 CFBD calls) |
| Mon 03:00 / 06:30 | `mlb-2026-roster-refresh`, `mlb-2026-power-rankings` | dry run |
| Tue 06:00-07:30 | `nfl-weekly-ingest`, `-score` | preflight only |
| Tue 06:00 / 07:30 | `cfb-weekly-predict`, `nfl-weekly-season-sim` | dry run |
| Wed 06:00 | `nfl-weekly-predict` | dry run |

Read them with `pipeline-ctl status` after each: every job should be `ok`, and `/var/lib/pipeline/state/logs/<job>/` has the log of each run.

### Stage 2 follow-up: making the control plane actually live on the homelab (2026-10-02)

Re-staging from the reconciled `main` was not enough. Two things made the control plane inert, and each was silent because the control-plane client **fails open**:

1. **Permission.** `pipeline-runner` was denied the `control` dataset (right for the old `main`). It now has `bigquery.dataViewer` on `control`: read-only; `mllab` stays the only writer. `pipeline-runner-setup.sh test` checks it is readable and, with `--write-probe`, not writable.
2. **A file the stager did not copy.** The control-plane merge changed `deploy_nfl.sh` and `deploy_cfb.sh` to copy `model_control.py` into the football bundles; the stager mirrored the old scripts, so those bundles lacked it (`No module named 'model_control'`). Found by a dry run against the new release. `stage.py` now copies it, and **verifies itself against the deploy scripts**: it reads every `cp` in them and raises `StageDrift` if a bundle lacks any. The next change to those scripts fails the release instead of going quiet.

How to tell the control plane is live on a run: the run log has no `model control unavailable` warning. The check is `grep -c "model control unavailable" /var/lib/pipeline/state/logs/<job>/*.log` (0 is good). After the second follow-up, both football dry runs against the installed release logged 0.

`install.sh` now refreshes the runner library on every run, `--release` included (the stager lives in it).

## The control app

Modes, runs and the Live gate can be driven from the lab's control app (`/app/`) as well as `pipeline-ctl`: see `docs/CONTROL_APP.md`. It cannot change anything in GCP.
Going Live (and a real run) there refuses while the job's Cloud Scheduler twin is not **PAUSED**, and also if Cloud Scheduler cannot be read. A root-side broker
(`pipeline-broker-apply`) applies the request after re-validating it; the web service itself cannot touch `/etc/pipeline` or `/var/lib/pipeline`.
`pipeline-ctl` remains the way in when the app or the broker is down, and its confirmation text is unchanged.
