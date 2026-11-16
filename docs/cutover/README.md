# Moving per-game MLB (`pregame`) to the homelab

**Done 2026-10-05** (`scripts/cutover_stage3.sh --yes`). The Cloud Scheduler job `mlb-2026-pregame-schedule` and the Cloud Tasks queues `lineup-pregame` and `sim-blend` are paused, `pregame`
is Live on the homelab, and the first live baselines wrote fresh v10 predictions for both of that day's games.

## What the plan got wrong, and what is true

The plan assumed per-game tasks came from **two** places: the Scheduler job, and the backend's `cron.yaml` entry for `/api/lineup/schedule-today`. They do not. `cron.yaml` uses crontab
syntax, which App Engine rejects (`Cron jobs only support Groc format for schedule`), so that file was never deployed: no `aet-*` job exists in Cloud Scheduler, every task in the queues was
created at 10:00 ET by the Scheduler job, and the backend's only in-process scheduler (`node-cron`) fetches news. The Scheduler job is the only producer. The entry was removed from `cron.yaml`
anyway (hanks_tank_backend #22) so the file no longer promises something that does not run; `backend-cron-remove-pregame.patch` is kept only as a record of that change.

If you ever *do* deploy a working cron file, `gcloud app deploy cron.yaml` replaces App Engine's whole cron list with the file's. The script refuses to run while any `aet-*` job exists.

## Run it again (it is the same for a future season)

```
bash scripts/cutover_stage3.sh                 # readiness: the rehearsal is clean and the next checkpoint is >= 30 minutes away
bash scripts/cutover_stage3.sh --yes           # pause the Scheduler job and both queues, set pregame Live, plan at once
bash scripts/cutover_stage3.sh --verify        # after a checkpoint has run
bash scripts/cutover_stage3.sh --rollback      # pregame -> dry, queues and Scheduler job resumed
```

Tasks that were already queued in Cloud Tasks stay there, paused (nothing is deleted). They are stale after the day they were for; purge them when you are sure you are not going back:
`gcloud tasks queues purge lineup-pregame --location us-central1 --project hankstank` (and `sim-blend`).
