# mllab as the control center for the sports models

**The goal (2026-10-05):** one place to watch and run the whole path, from the data coming in, through cleanup and validation, features, models, predictions and
simulations, down to what visitors see on hankstank.com. Not a set of scripts you have to know about: a map you can read, and the actions that belong on it.

```
 sources ─▶ ingest ─▶ raw tables ─▶ validate / clean ─▶ features ─▶ models ─▶ predictions, sims, rankings ─▶ backend API ─▶ the site
   (1)       (2)         (3)              (4)              (5)        (6)               (7)                    (8)           (9)
```

## Where each stage stands

| # | Stage | Watch | Act | Where |
|---|---|---|---|---|
| 1 | External sources (MLB Stats API, Statcast, nflverse, CFBD, FPI) | partly: a failed fetch shows only in a job's log | | **gap** |
| 2 | Ingest jobs | yes: modes, schedule, last run, logs, did the run change its tables (Data page) | set mode, run once (dry/real), roll back | Pipeline, Data |
| 3 | Raw tables | yes: rows, size, last change, freshness against each table's own cadence | | Data |
| 4 | Validation and cleanup | partly: table specs for 3 MLB tables; the MLB validator's result sits in a job log | **gap**: dedupe, re-ingest a date or week, backfill | `mllab bq check`; hanks_tank_ml scripts |
| 5 | Features | yes: freshness (Data page); MLB feature freshness rule | | Data, Health |
| 6 | Models | yes: registry, backtests, tuning, shadows | register, point a channel, roll back, tune | Models, Jobs, notebooks |
| 7 | Predictions, sims, rankings | yes: coverage, freshness, distribution, accuracy drift, sim invariants | run a job | Health, Data |
| 8 | Backend API | **gap**: nothing checks what the API serves against BigQuery | | |
| 9 | The site | read: control state; **gap**: nothing checks what a visitor actually sees | CLI only: hide, pause, pin, banner, freeze, labels, order | `mllab control …` |

## Principles (they decide what gets built, and how)

* **The app reads everything and writes only what it owns.** It controls the homelab and the lab. GCP writes (Cloud Scheduler, the `control` dataset, deploys) stay audited
  commands until a deliberate grant says otherwise. A web page should not hold production-write by accident.
* **Every action is previewed, confirmed and audited.** A dry-run diff first; the typed name for anything that touches production; an audit-log line; and the
  action undoes cleanly where it can (rollback is a recorded change like any other).
* **Fail closed on the unknown.** If the app cannot read a source it says so; it never reports "fine" or "missing" by guessing.
* **Declare, then check.** The data map is a catalog file per sport (`catalog/*.yaml`): what each table is, which job writes it, how often it should change. Checks run against
  that declaration, so a new table or job is one entry, and a check that disagrees with the pipeline is a bug you can see.

## Phases

1. **See all of it** (first slice built): the Data page. Per sport, every table in pipeline order with rows, last change, freshness against its own cadence, the job
   that writes it, and "the job ran OK but the table did not change". Metadata only, so it costs nothing.
2. **Trust the data:** run the table specs (and new ones for the football tables) from the app on demand and on a schedule; show each check's result beside its table;
   source-health probes (can the pipeline reach the MLB API, nflverse, CFBD right now); row-count and duplicate guards.
3. **See the site:** a Site page that shows the control state (what is hidden, paused, pinned, bannered) and probes the live API the way a visitor sees it: do the endpoints
   answer, are the predictions fresh, does what the API serves match BigQuery for the same games.
4. **Run the site:** hide, unhide, pause, pin, banner and freeze from the app. This needs the lab's service account to be allowed to write the `control` dataset (and only that
   dataset): a grant you decide. Every change keeps the typed phrase, the history and revert that `mllab control` already has.
5. **Clean up and repair:** guarded actions with a dry-run diff, a typed confirmation and an audit entry: re-run an ingest for a date or week, dedupe the games table, backfill a
   season. Only for writes that are idempotent (the catalog records how each table is written).
6. **Tell me:** push alerts on a failed run, a stale table, an unchanged table or a mismatch between the API and BigQuery, plus a weekly digest.
7. **Finish the move:** the four ingest and score jobs and per-game MLB onto the homelab, then retire the Cloud Functions.

## Decisions that are yours

* Phase 4 needs the `control`-dataset write grant for the lab's service account. Until you grant it, the app shows the control state and the exact command.
* Phase 6 needs a channel (the Kuma instance already on the homelab can push to your phone, if it is set up for that).
