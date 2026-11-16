# The control app

The lab's front door: **`https://<homelab>.<tailnet>.ts.net:8899/`** opens it (the root redirects to `/app/`). JupyterLab is at `/lab`.

| Page | What it is for |
|---|---|
| **Overview** | everything that needs attention first (a job written from two places, a stale broker, failed runs, critical model findings), then a card per area |
| **Pipeline** | every job: mode (Off / Dry run / Live), schedule, next and last run, and what Cloud Scheduler says about its twin. Click a row for its log; run it once, dry or for real |
| **Data** | the data map: every table the pipeline reads or writes, per sport in pipeline order (raw → features → predictions → shadows → sims → rankings), with rows, last change, and a verdict against its own cadence. Flags a table that is stale or missing, and a **Live job that finished OK without changing its table**. Declared in `catalog/*.yaml`; table metadata only, so it costs nothing (`docs/CONTROL_CENTER.md`) |
| **Models** | the registry: versions, lineage, metrics, channels. *Point channel…* moves a pointer |
| **Jobs** | lab jobs (backtests, training) with logs, a cancel button, and the experiments they produced |
| **GCP** | Cloud Scheduler, read-only, with the exact audited command for each job |
| **Site** | Which ranking columns hankstank.com's power-rankings table shows: up to 3 of ours beside the media polls, with a preview, typed confirmation, and Back to default. Writes the `control` dataset (needs the `dataEditor` grant; see MODEL_CONTROL.md) |
| **Health** | the model-health findings and per-model status (the same data as the static dashboard at `/dash/`) |
| **Audit** | the append-only log of every change, from the app, the CLI and the broker |
| **Notebooks** | the notebooks in the work folder, and a link to JupyterLab |

## Rolling back, and running a job by hand

Every pipeline row has **▶ Run…** and **↶ Roll back**.

* **Run…** offers a *Dry run* (all the work, writes nothing; for the four jobs that refuse dry runs it is a preflight) or a *Real run*, which asks for the same
  confirmation as going Live. The job's mode does not change. The app then follows the job and tells you how it went: OK or FAILED, and how long it took.
* **Roll back** sets the job back to the mode it had before its latest change, as read from the audit log, and does it as another change (audited, and it goes
  through the same gates: rolling back *to* Live is going Live). It is disabled when the log shows no earlier mode, for instance a job changed by hand with
  `pipeline-ctl`, which is not in the log.
* **Models → ↶ Roll back to vN** points a channel back at the version it had before its latest move. The `prod` channel asks for the typed phrase. A channel is only a
  pointer, so this deploys nothing; undoing a deployment is `mllab deploy cloudfn-rollback`.

These change **the homelab only**. Rolling a Live job back stops it writing from the homelab, but its Cloud Scheduler twin is still paused, so for now nothing runs
that job in production. The dialog says so and shows the command that hands it back to Cloud Functions (`mllab control scheduler resume JOB --confirm JOB`), which stays
a laptop command by design.

## What it can and cannot change

The app controls **what runs on the homelab** and **the lab itself**. It only *reads* GCP, with one deliberate exception: the Site page writes the `control` dataset (and only that), once you have granted the lab's service account `dataEditor` on it.

* **Can:** set a pipeline job's mode; run a job once (dry, or for real); point a registry channel at a version; cancel a lab job.
* **Cannot, on purpose:** pause or resume a Cloud Scheduler job, write model-control events (hide / pin / freeze), or deploy a model. Those stay as
  audited commands on your laptop (`mllab control …`, `mllab deploy …`); the app shows the exact command next to the thing it applies to. The homelab's
  service account (`mllab-lab`) has no write access to those, and a web page should not be the thing that gets it.

## Safety model

A job must never write production from two places. So going **Live** (and a real *Run once*) is gated:

1. Cloud Scheduler is read **fresh**. The same-named job must be **PAUSED**. If it is enabled, or if Cloud Scheduler cannot be read, the app refuses. It does not guess.
2. You type the job's name.
3. `pregame` (per-game MLB) additionally needs two acknowledgements the app cannot verify (it cannot see the App Engine `cron.yaml` entry or the Cloud Tasks queues).

The app shows a **conflict** (red) whenever a job is Live here and still enabled in GCP, on every page.

The gate is enforced twice: in the web API, and again by the broker (the typed name for Live and for real runs). Dry runs and Off need neither.

### Who may do what

The web service listens on `127.0.0.1:8897`, which **any container on the homelab can reach without Caddy**, with any header it likes. So it does not trust
Caddy or the Tailscale header on their own:

| | needs |
|---|---|
| look at status | the allowed Tailscale login header (`MLLAB_ALLOWED_LOGIN`) |
| read run logs | the above + the **unlock cookie** |
| change anything | the above + the unlock cookie + a same-origin request carrying `X-Mllab-App: 1` |

The **unlock** is a password, once per browser: `MLLAB_SECRET` from `/etc/mllab/env`, which no container has. The cookie is an HMAC of the login under that
secret, `HttpOnly`, `SameSite=Strict`, 30 days. Five wrong passwords in a minute locks the prompt for a minute. A page on another website cannot make a change
for you: it cannot send the custom header, and its `Origin` would not match.

```
ssh -t homelab 'sudo grep MLLAB_SECRET /etc/mllab/env'
```

The page ships with `Content-Security-Policy: default-src 'none'; script-src 'self'; style-src 'self'; frame-ancestors 'none'`: its own script and stylesheet only.
Every element is built with `textContent`, so nothing the server returns can become markup.

### Turning the password off

If you trust everything that runs on the homelab, you can drop the password. Add this line to `/etc/mllab/env` and restart the API:

```
ssh -t homelab 'echo "MLLAB_APP_NO_UNLOCK=1" | sudo tee -a /etc/mllab/env >/dev/null && sudo systemctl restart mllab-api'
```

The header then shows *No password*. Only the exact value `1` counts; remove the line (and restart) to turn the lock back on. **What you give up:** a process on the homelab, such as a
compromised container, can forge the login header and change job modes, run jobs and move registry channels. **What stays:** the allowed-login check, the same-origin
check (another website cannot act for you), the typed job name, and the Cloud Scheduler gate for going Live, so the worst such a process can do is bounded by what you have
already paused in GCP.

## How it fits together

```
 browser ──tailscale serve──▶ Caddy :8899 ──forward_auth──▶ mllab API :8897   (user mllab: web page + /app-api)
                                                                │ reads                │ drops a request file
                                      /mnt/appdata/mllab/pipeline-status/              ▼   /mnt/appdata/mllab/spool/pipe-<id>.json
                                      state.json + logs/  ◀── pipeline-broker-snapshot (user pipeline, every 30 s)
                                                                       pipeline-broker-apply (root, path-triggered)
                                      pipeline-results/<id>.json ◀──────────┘   validates, then writes the mode file / starts the run
```

* The app's user **cannot read or write** `/etc/pipeline` or `/var/lib/pipeline`: they belong to the `pipeline` group. The broker is the only bridge.
* `pipeline-broker-snapshot` (as `pipeline`) publishes modes, last runs, per-game counts, the release and each job's latest log.
* `pipeline-broker-apply` (as **root**, only when a request file appears) is the one privileged piece. A request is **untrusted**: a regular file owned by `mllab`,
  under 2 KB, strict JSON with exactly the expected keys, a job from the job table (never free text), a mode from `off|dry|live`. A symlink is never followed.
  It writes the mode file, or starts a transient `pipeline-adhoc-…` unit as `pipeline` with the same limits as a scheduled run, then records the outcome and an
  audit line. Code: `src/mllab/pipeline/broker.py`; rules pinned in `tests/test_pipeline_broker.py`.
* The page and API: `src/mllab/webapp/` (`session.py` access rules, `pipeline_view.py` gates and requests, `views.py` the read side, `routes.py`, `static/`).

## Install / update

From the homelab, with the repo synced to `~/mllab`. Order matters: the app first, then the broker.

```
cd ~/mllab/host && sudo ./install.sh --dry-run              # look first
cd ~/mllab/host && sudo ./install.sh                        # mllab: code, units, Caddyfile, starter notebooks; restarts mllab-api / mllab-caddy only if they changed

cd ~/mllab/host/pipeline && sudo ./install.sh --dry-run --ref <release>
cd ~/mllab/host/pipeline && sudo ./install.sh --ref <release>   # the broker: its directories and units, plus the pipeline lib
```

**Pin `--ref` to the release that is already installed** (`readlink /opt/pipeline/current` shows it, e.g. `dad6ca7bd8fc`). The pipeline installer always stages a release,
and without `--ref` that is the latest `main` of `hanks_tank_ml`, which would change the pipeline's code as a side effect of adding the broker.
Every job mode is kept, a job installed for the first time starts **off**, and Jupyter is never restarted by either installer.

Verify:

```
systemctl is-active pipeline-broker-snapshot.timer pipeline-broker-apply.path mllab-api mllab-caddy
ls -l /mnt/appdata/mllab/pipeline-status/            # state.json refreshed every 30 s
```

The app's header shows **Broker Ns** (green while the snapshot is under 2.5 minutes old). If it says *Broker stale*, the snapshot timer is not running.

## Starter notebooks

`notebooks/` is copied into the Jupyter work folder by `host/install.sh`, **only if a file of that name is absent**: a notebook you have edited is never overwritten.
They are generated by `scripts/make_notebooks.py` (tests fail if the committed files drift) and stay thin: the work is in `mllab.nb`, shared with the app.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Blank white page on the tailnet URL | Caddy's site was addressed `127.0.0.1`, which does not match the tailnet `Host`. Fixed (`http://:8899`); pinned by `tests/test_caddyfile.py` |
| *Broker stale* / empty pipeline | `pipeline-broker-snapshot.timer` is not running, or `pipeline-status/` is missing: re-run the full pipeline install |
| A click says "no answer from the broker after 45 s" | `pipeline-broker-apply.path` is not running, or the request was rejected: see the **Audit** page and `journalctl -u pipeline-broker-apply` |
| Going Live refuses with "cannot read Cloud Scheduler" | the lab's service account cannot list Scheduler jobs, or the API is down. The gate fails closed on purpose |
| Password rejected | it is `MLLAB_SECRET`, not the Jupyter password/token |
| `401 locked` | the cookie expired or this is a new browser: unlock again |

## Develop it

```
PYTHONPATH=src .venv/bin/python scripts/dev_app.py        # http://127.0.0.1:8890/app/   lab password: devpass
```

Runs the real app against fake data (a fake Cloud Scheduler with a deliberate two-writers case, and a fake broker that applies requests), touching nothing real.
