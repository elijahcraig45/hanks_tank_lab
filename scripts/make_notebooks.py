#!/usr/bin/env python3
"""Generate the lab's starter notebooks into notebooks/ (the committed .ipynb files are the output; tests/test_notebooks.py fails if they drift).

    PYTHONPATH=src python3 scripts/make_notebooks.py          write
    PYTHONPATH=src python3 scripts/make_notebooks.py --check  exit 1 if the committed notebooks are stale

Notebooks stay thin on purpose: the work happens in mllab.nb (shared with the control app) and in the specs, so they are short, and a fix lands once.
Seeded onto the homelab by host/install.sh into the work folder, never over a copy you have edited."""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells: list[dict]) -> str:
    nb = {"nbformat": 4, "nbformat_minor": 4, "cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}}
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


NOTEBOOKS: dict[str, list[dict]] = {}

NOTEBOOKS["00_start_here.ipynb"] = [
    md("""
# mllab: start here

This is the lab's workbench. There are three ways in, and they all work on the same data:

| | use it to |
|---|---|
| **Control app** (`/app/`) | watch and change what runs: pipeline job modes, models and their channels, lab jobs. Production decisions are made there. |
| **These notebooks** | explore the data, build and compare models, register them. |
| **The command line** (`mllab`, `pipeline-ctl`) | the same actions as scripts. The notebooks call it, so nothing here is a second implementation. |

A model's life: **explore** the data, **write a spec**, **backtest** it against the others, **train** a candidate, **register** it (an immutable, hash-verified version),
**point a channel** at it, **deploy** it (a confirmed step, from the CLI), and the pipeline serves it.
"""),
    code("""
from mllab import nb
nb.status()
"""),
    md("""
## The notebooks

| notebook | what it does |
|---|---|
| `01_explore_data` | look at the BigQuery tables the models read, and how well the live predictions have done. Every query is cost-guarded. |
| `02_backtest_models` | compare models fairly on the same games, with confidence intervals. |
| `03_train_and_register` | train a candidate with an honest holdout, then register it as a new version. |
| `04_pipeline_and_control` | what the homelab pipeline is doing, and the checklist for moving a job from Cloud Functions to here. |
| `05_tune_models` | tune NFL, CFB and MLB models, and read the result without fooling yourself. |
| `quickstart` | a short read-only tour of the guards, the registry and Ask Gemini. |
"""),
    code("nb.app()"),
    md("""
## What is safe here

* **Reading** is safe everywhere. BigQuery goes through a guard: a dry run first, a capped scan, a capped result, and a local cache.
* **Experiments** (backtests, training) run as lab jobs under hard memory, CPU and time limits. They cannot touch production.
* **Registering a model** only records it. A registered version does nothing until a channel points at it **and** it is deployed.
* **Nothing in these notebooks changes production.** Moving a pipeline job to Live, pausing a Cloud Scheduler job and deploying a model are done in the control
  app or the CLI, each with a typed confirmation and an audit-log entry.
"""),
]

NOTEBOOKS["01_explore_data.ipynb"] = [
    md("""
# Explore the data

`%%bq` runs SQL through the lab's guard: it dry-runs first and refuses a query that would scan more than the cap (raise it with `--max-gb`), caps the result
size, and caches the answer on disk (`--refresh` to re-run). The result lands in a DataFrame named by the last word on the line.
"""),
    code("%load_ext mllab.magics"),
    md("What is in the 2026 season dataset, and how big is it? (Table metadata, so this costs almost nothing.)"),
    code("""
%%bq --max-gb 0.1 tables
SELECT table_id, row_count, ROUND(size_bytes / 1e6, 1) AS mb
FROM `hankstank.mlb_2026_season.__TABLES__`
ORDER BY table_id
"""),
    code("tables"),
    md("""
## How have the live predictions done?

One row per completed game: the **last prediction made before first pitch**, and the result. This uses the same rules as the lab's own backtest data
(`sql/mlb_v10_pregame.sql`): predictions made after first pitch are contaminated, and a game can have a skeleton row plus a final row.
"""),
    code("""
%%bq --max-gb 1 scored
WITH pred AS (
  SELECT game_pk, home_win_probability AS p
  FROM `hankstank.mlb_2026_season.game_predictions`
  WHERE predicted_at < game_time_utc
  QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY predicted_at DESC) = 1
), final AS (
  SELECT game_pk, CAST(home_score > away_score AS INT64) AS home_win
  FROM `hankstank.mlb_2026_season.games`
  WHERE status IN ('Final', 'Game Over') AND home_score IS NOT NULL AND away_score IS NOT NULL AND home_score != away_score
  QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY synced_at DESC NULLS LAST) = 1
)
SELECT p, home_win FROM pred JOIN final USING (game_pk)
"""),
    code("""
import numpy as np
n = len(scored)
acc = float(((scored.p > 0.5) == (scored.home_win == 1)).mean())
ll = float(-np.mean(scored.home_win * np.log(scored.p.clip(1e-6, 1 - 1e-6)) + (1 - scored.home_win) * np.log((1 - scored.p).clip(1e-6, 1 - 1e-6))))
base = float(scored.home_win.mean())
print(f"{n} completed games with a pre-game prediction")
print(f"accuracy {acc:.3f}   log loss {ll:.4f}   (always-home: accuracy {max(base, 1 - base):.3f})")
"""),
    md("**Calibration**: when the model says 60%, does the home team win about 60% of the time?"),
    code("""
import pandas as pd
bins = pd.cut(scored.p, [0, .40, .45, .50, .55, .60, .65, 1.0])
cal = scored.groupby(bins, observed=True).agg(games=("p", "size"), predicted=("p", "mean"), actual=("home_win", "mean")).round(3)
cal
"""),
    md("""
A few hundred games per bin is the least you need before a gap means anything. With about 2,000 games in total, only differences of two or three accuracy points
are detectable at all, which is why the next notebook reports confidence intervals rather than a single number.

**Next:** `02_backtest_models`.
"""),
]

NOTEBOOKS["02_backtest_models.ipynb"] = [
    md("""
# Compare models fairly

A backtest scores several contenders **on the same games**, with paired-bootstrap confidence intervals for the *difference* between each one and the reference.
A retrain only ever learns from games dated before the block it predicts. Read the intervals: if one spans zero, the contenders are not distinguishable.

The specs that ship with the lab:
"""),
    code("""
from mllab import nb
nb.specs()
"""),
    md("""
Copy one to your own folder (`work/specs/`), where you can edit it. A copy keeps working from anywhere, because its SQL path is made absolute. `new_spec` never
overwrites a file, so pick a new name each time.
"""),
    code("""
path = nb.new_spec("mlb_v10_param_sweep", "my_sweep")
print(path)
print(path.read_text())
"""),
    md("""
**Edit the copy** (open `specs/my_sweep.yaml` from the file browser): change a `params:` block, or delete contenders you do not need. Each retrained contender adds
minutes, and the whole job has a time limit (`timeout_s`).

Then submit it. It runs as a lab job with hard memory, CPU and time caps; this cell returns the job id immediately.
"""),
    code("""
job = nb.submit("backtest", path)
job
"""),
    code("nb.wait(job)   # follows the job until it ends; stopping this cell does not stop the job"),
    code("nb.report(job)"),
    md("""
## All your experiments

Each run gets its own verdict. **Runs are never ranked against each other**: they use different games and different windows, so comparing their headline numbers
would mislead.
"""),
    code("nb.experiments()"),
    md("""
**Next:** if a contender is worth keeping, train it properly and register it in `03_train_and_register`.

For football the repo keeps 2025 as a touch-once holdout that the lab never scores. Do not point a spec at it.
"""),
]

NOTEBOOKS["03_train_and_register.ipynb"] = [
    md("""
# Train a candidate and register it

Training uses an **honest holdout**: it learns from history up to `train_end`, then scores the games after it, which it never sees, next to production and an
always-home baseline. A candidate is a candidate, not a recommendation: a backtest is what tells you whether it is distinguishable from production.
"""),
    code("""
from mllab import nb
path = nb.new_spec("train_v11_d2", "my_candidate")
print(path.read_text())
"""),
    md("Edit the copy if you want different parameters or a different `train_end`, then train it."),
    code("""
job = nb.submit("train", path)
nb.wait(job)
nb.report(job)
"""),
    md("""
## Register it

Registering records the trained model as a **new, immutable version** with its lineage (`--parent`), its feature list, a hash of the exact input data, and the
pinned library versions. It deploys nothing. Choose a version name that does not exist yet; run this cell deliberately.
"""),
    code("""
NEW_VERSION = "v12"        # must be new: versions are immutable, so a name can be used only once
nb.models()[["model", "version", "channels", "created_at"]]
"""),
    code("""nb.mllab("register-job", job, "hankstank", "game_outcome", NEW_VERSION, "--parent", "v10")"""),
    code("""nb.mllab("card", "hankstank", "game_outcome", NEW_VERSION)   # lineage, data, metrics, and the backtests that included it"""),
    md("""
## Making it live

This notebook deliberately stops here.

1. **Point a channel at it**: in the control app, *Models* → *Point channel…*. Moving the `prod` channel asks you to type the phrase `game_outcome@prod`, and it is
   audited. A channel is only a pointer: nothing is deployed by moving it.
2. **Deploy it**: `mllab deploy cloudfn-swap …` from where you hold GCP credentials. It shows a dry run first, takes a typed confirmation, backs up the
   current model object, and can roll back.
"""),
    code("nb.app('#/models')"),
]

NOTEBOOKS["05_tune_models.ipynb"] = [
    md("""
# Tune a model, and know whether it helped

Tuning is a search, and **the best of N tries looks good partly by luck**. The lab's tuning has three safeguards built in, and this notebook keeps them:

1. The search uses **only earlier seasons**, ranked by an inner walk-forward.
2. The winner is **frozen**, then scored on later seasons that no candidate ever saw.
3. Every contender is compared on the **same games**, with a paired confidence interval for the *difference*. The untuned version stays in the comparison.

The winner's own search score (the "inner" score) is best-of-N and optimistic by construction. **Only the outer numbers count.** For football, 2025 is the repo's
touch-once holdout: the lab never scores it, and neither should a spec.

Runs are lab jobs under hard memory, CPU and time limits, so you can close the notebook while one runs; `nb.wait(job)` just follows it.
"""),
    code("""
from mllab import nb
nb.specs()
"""),
    md("""
## NFL: a tuned twin of the headline model

First, the settings the repo's model uses today. They are the starting point the search is compared with:
"""),
    code("""nb.contenders("nfl_walkforward")[["name", "kind", "family", "params", "seeds"]]"""),
    md("""
`tune_spec` copies the spec and adds `xgb_pure_tuned`: the same model, with a random search over `space` (the starter space is below; edit it, or pass your own).
`seeds` averages several random seeds, which matters because a single XGBoost fit moves a season by a point or two on its own.
"""),
    code("""
nb.space("xgboost")        # the starter search space: {param: {low, high, [log], [type: int]}} or {param: [choices]}
"""),
    code("""
nfl_path = nb.tune_spec("nfl_walkforward", "my_nfl_tune", "xgb_pure", trials=40, seeds=[1, 2, 3])
print(nfl_path)
nb.contenders(nfl_path)[["name", "tuned", "seeds"]]
"""),
    code("""
nfl_job = nb.submit("backtest", nfl_path)
nb.wait(nfl_job)
"""),
    md("""
### Read it

`vs` is Elo by default. A **negative** difference is better than Elo. The verdict comes from the confidence interval: only *better* and *worse* mean anything, and
*not distinguishable* is the usual honest answer.
"""),
    code("nb.results(nfl_job)"),
    md("What the search found, and how much of its inner gain is probably luck:"),
    code("nb.tuned(nfl_job)"),
    code("""nb.trials(nfl_job, "xgb_pure_tuned")   # if the top few are all close, the surface is flat and the 'best' is not special"""),
    md("""
Compare the tuned model with its **own untuned twin**: `nb.results(nfl_job, vs="xgb_pure")`. If that interval spans zero, tuning did not help, and the simpler
default stands. (On the lab's earlier run of this experiment it did not.)
"""),
    code("""nb.results(nfl_job, vs="xgb_pure")"""),
    md("""
## CFB

The same, for FBS. (FCS has its own spec, `cfb_fcs_walkforward`.) College football has fewer seasons of history in the export, so expect wider intervals.
"""),
    code("""
cfb_path = nb.tune_spec("cfb_fbs_walkforward", "my_cfb_tune", "xgb", trials=30, seeds=[1, 2, 3])
cfb_job = nb.submit("backtest", cfb_path)
nb.wait(cfb_job)
"""),
    code("""nb.results(cfb_job, vs="xgb")"""),
    code("nb.tuned(cfb_job)"),
    md("""
## MLB: a small grid of fixed settings

MLB is scored only on 2026 games, so there are no whole seasons to hold out and search over. The honest tool is a **small grid**: each combination is a contender, all
scored on the same games against production (`prod`). `grid_spec` refuses a grid above 12 combinations, because every extra combination adds minutes and adds to the
chance that the winner is luck.
"""),
    code("""nb.contenders("mlb_v10_param_sweep")[["name", "kind", "family", "params"]]"""),
    code("""
mlb_path = nb.grid_spec("mlb_v10_param_sweep", "my_mlb_grid", "xgb_d2", {"max_depth": [2, 3], "learning_rate": [0.03, 0.05]})
nb.contenders(mlb_path)[["name", "params"]]
"""),
    code("""
mlb_job = nb.submit("backtest", mlb_path)
nb.wait(mlb_job)
"""),
    code("""nb.results(mlb_job, vs="prod")"""),
    md("""
With about 2,000 games, only differences of two or three accuracy points are detectable, so most MLB grids come back *not distinguishable from prod*. That is a result:
it says the settings are not what limits the model.

## If something does win

* Treat it as a **candidate**. Confirm it on games none of the contenders has seen (for MLB, wait for more games and re-run the backtest).
* Then train it properly and register it in `03_train_and_register`, and point a channel at it in the control app.
* Do not tune on the football 2025 holdout, and do not re-run a search until a result you like appears: that is the same luck, taken again.
"""),
]

NOTEBOOKS["04_pipeline_and_control.ipynb"] = [
    md("""
# The pipeline, and moving it home

The prediction jobs are moving from Cloud Functions (run by Cloud Scheduler) to this homelab. Each job has a **mode**:

| mode | what it does |
|---|---|
| **off** | the timer fires and does nothing. Every job starts here. |
| **dry** | runs on its schedule and does all its work, but writes nothing. (A few jobs refuse a dry run; for those it is a preflight only.) |
| **live** | runs and writes to production BigQuery. |

This notebook only **reads**. Changing a mode is done in the control app (`Pipeline`), which refuses to go live while the job's Cloud Scheduler twin is still enabled.
"""),
    code("""
from mllab import nb
pl = nb.pipeline()
print("as of", pl.attrs.get("taken_at"))
pl
"""),
    code("""pl.groupby(["bundle", "mode"]).size().unstack(fill_value=0)"""),
    md("Anything that is on and whose last run failed:"),
    code("""pl[(pl["mode"].isin(["dry", "live"])) & (pl["last_ok"] == False)]   # noqa: E712"""),
    md("""
## Moving one job to live

Do these in order, one job at a time. The control app enforces the check in step 2.

1. **Dry run for a few cycles.** Set the job to *Dry run* in the app, let it run on schedule, and read the log (click the job). It should do the work and write nothing.
2. **Pause its Cloud Scheduler twin**, from your laptop (audited there): `mllab control scheduler pause <job> --confirm <job>`. The app shows the exact command.
3. **Set it to Live** in the app. It re-checks Cloud Scheduler, then asks you to type the job's name.

Per-game MLB (`pregame`) is the exception: it is created from **two** places (the Scheduler job `mlb-2026-pregame-schedule` and the App Engine cron entry
`/api/lineup/schedule-today`), and its Cloud Tasks queues must be drained. The app lists every step and asks you to acknowledge each.

The Cloud Functions stay deployed, idle, as a manual fallback: there is no automatic failover, because two writers on one table is worse than one missed run.
"""),
    md("""
## What the model control plane does

Separate from the pipeline: it decides what the **site** shows and which model the pipeline runs, without a code change (hide a model, pin an exact artifact,
freeze a sport with a banner). Its settings live in BigQuery's `control` dataset and are changed with `mllab control …` from your laptop. See `docs/MODEL_CONTROL.md`.
"""),
    code("nb.app()"),
]


def render() -> dict[str, str]:
    return {name: notebook(cells) for name, cells in NOTEBOOKS.items()}


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    want = render()
    if a == ["--check"]:
        stale = [n for n, t in want.items() if not (OUT / n).is_file() or (OUT / n).read_text() != t]
        stale += [p.name for p in OUT.glob("*.ipynb") if p.name not in want and p.name != "quickstart.ipynb"]
        print("notebooks are current" if not stale else "stale: " + ", ".join(stale))
        return 1 if stale else 0
    OUT.mkdir(exist_ok=True)
    for n, t in want.items():
        (OUT / n).write_text(t)
    print("\n".join(want))
    return 0


if __name__ == "__main__":
    sys.exit(main())
