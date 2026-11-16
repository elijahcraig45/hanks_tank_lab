"""Helpers for the lab's notebooks.

The notebooks are the workbench; the control app (/app/) is where production is watched and changed. They share their code: everything here that
shows state reads through the same functions the app uses, so a notebook and the app cannot disagree, and a fix lands once.

    from mllab import nb
    nb.status()                      # one-screen summary of the lab
    nb.pipeline(), nb.models(), nb.experiments(), nb.jobs()     # DataFrames
    nb.specs()                       # the experiment specs that ship with the lab
    path = nb.new_spec("mlb_v10_param_sweep", "my_sweep")       # an editable copy under work/specs
    nb.mllab("backtest", "run", str(path))                      # the CLI, in-process
    nb.wait(job_id)                  # follow a job until it ends

Nothing here changes production: it can read the pipeline's published status, and it can submit lab jobs and register models, exactly as the CLI can."""
from __future__ import annotations

import contextlib
import io
import json
import os
import time
from pathlib import Path

import pandas as pd
import yaml

from . import runs as R
from .jobs import LocalRunner
from .registry import Registry
from .webapp import views

TERMINAL = ("finished", "failed", "terminated", "lost")


def root() -> Path:
    return Path(os.environ.get("MLLAB_ROOT", "/mnt/appdata/mllab"))


def registry() -> Registry:
    return Registry(os.environ.get("MLLAB_REGISTRY", root() / "registry"))


def runner() -> LocalRunner:
    return LocalRunner(root())


def app_url(path: str = "") -> str:
    """Relative: it resolves against whichever host name this notebook was opened from (the tailnet name, normally)."""
    return "/app/" + path.lstrip("/")


def app(path: str = ""):
    """A link to the control app. Displays as a clickable link in Jupyter, prints the URL anywhere else."""
    try:
        from IPython.display import HTML, display
        display(HTML(f'<a href="{app_url(path)}" target="_blank">Open the control app ↗</a>'))
    except ImportError:
        print(app_url(path))


# ------------------------------------------------------------------------------------------------------------------------------ state
def pipeline() -> pd.DataFrame:
    """Every pipeline job, its mode, schedule and last run, as the broker last published it. Empty (with a note) if the broker is not running."""
    p = Path(os.environ.get("BROKER_STATUS_DIR", root() / "pipeline-status")) / "state.json"
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        print(f"No pipeline status at {p}: the pipeline broker is not publishing (pipeline-broker-snapshot.timer on the homelab).")
        return pd.DataFrame(columns=["job", "bundle", "mode", "schedule", "last_ok", "last_finished"])
    rows = [{"job": j["name"], "bundle": j["bundle"], "mode": j["mode"] if not j["paused"] else "paused", "schedule": j["cron"],
             "last_ok": (j["last"] or {}).get("ok"), "last_finished": (j["last"] or {}).get("finished")} for j in raw["jobs"]]
    pg = raw.get("pregame") or {}
    rows.append({"job": "pregame", "bundle": "mlb", "mode": pg.get("mode"), "schedule": "every minute", "last_ok": (pg.get("last") or {}).get("ok"),
                 "last_finished": (pg.get("last") or {}).get("finished")})
    df = pd.DataFrame(rows)
    df.attrs["taken_at"] = raw.get("taken_at")
    return df


def models() -> pd.DataFrame:
    """One row per registered version, newest first within each model."""
    rows = []
    for pr in views.models(registry())["projects"]:
        for mo in pr["models"]:
            for v in mo["versions"]:
                rows.append({"project": pr["project"], "model": mo["model"], "version": v["version"], "channels": ",".join(v["channels"]), "parent": v["parent"],
                             "created_at": v["created_at"], "features": v["features"], "metrics": v["metrics"], "sha256": v["sha256"]})
    return pd.DataFrame(rows, columns=["project", "model", "version", "channels", "parent", "created_at", "features", "metrics", "sha256"])


def experiments(limit: int = 30) -> pd.DataFrame:
    """Backtests and training runs that produced results, each with its own verdict. Runs are never ranked against each other: they use different games."""
    return pd.DataFrame(R.list_runs(runner(), limit=limit))


def jobs(limit: int = 40) -> pd.DataFrame:
    return pd.DataFrame(views.jobs(runner(), limit)["jobs"])


def status() -> None:
    """The lab at a glance. Read-only."""
    pl = pipeline()
    if len(pl):
        print("PIPELINE (live / dry / off):", " / ".join(str(int((pl["mode"] == m).sum())) for m in ("live", "dry", "off")), f"  (as of {pl.attrs.get('taken_at', '?')})")
        bad = pl[(pl["mode"] != "off") & (pl["last_ok"] == False)]  # noqa: E712  a missing last_ok is NaN/None, not a failure
        for _, r in bad.iterrows():
            print(f"  ! {r['job']}: last run FAILED")
    m = models()
    print("MODELS:", ", ".join(f"{r.model} {r.version}{' [' + r.channels + ']' if r.channels else ''}" for r in m.itertuples() if r.channels) or "none on a channel")
    ex = experiments(5)
    print("EXPERIMENTS: " + (f"{len(ex)} recent; latest {ex.iloc[0]['name']} ({ex.iloc[0]['kind']})" if len(ex) else "none yet"))
    print("Control app:", app_url())


# ---------------------------------------------------------------------------------------------------------------------------------- specs
def _spec_dirs() -> list[Path]:
    dirs = [Path(p) for p in os.environ.get("MLLAB_SPECS", "").split(":") if p]
    return dirs + [Path("/opt/mllab/src/specs"), Path(__file__).resolve().parents[2] / "specs", root() / "work" / "specs"]


def _find_spec(name: str) -> Path:
    for d in _spec_dirs():
        for cand in (d / name, d / f"{name}.yaml"):
            if cand.is_file():
                return cand
    raise FileNotFoundError(f"no spec named {name!r}; see nb.specs()")


def specs() -> pd.DataFrame:
    """The experiment specs available: the ones that ship with the lab, and your own copies under work/specs."""
    rows, seen = [], set()
    for d in _spec_dirs():
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.yaml")):
            if p.stem in seen:
                continue
            seen.add(p.stem)
            try:
                raw = yaml.safe_load(p.read_text()) or {}
            except yaml.YAMLError:
                raw = {}
            kind = "backtest" if "retrain" in raw or "models" in raw or "contenders" in raw else "train" if "family" in raw else "check" if "table" in raw else "other"
            first = next((ln.lstrip("# ").strip() for ln in p.read_text().splitlines() if ln.startswith("#")), "")
            rows.append({"spec": p.stem, "kind": kind, "where": "yours" if "work" in p.parts else "lab", "about": first[:110]})
    return pd.DataFrame(rows, columns=["spec", "kind", "where", "about"])


def new_spec(base: str, name: str) -> Path:
    """Copy a shipped spec to work/specs/<name>.yaml, ready to edit. Relative paths (the SQL file, the environment lock) are made absolute so the copy runs from anywhere."""
    src = _find_spec(base)
    raw = yaml.safe_load(src.read_text())
    data = raw.get("data")
    if isinstance(data, dict) and data.get("sql_file"):
        data["sql_file"] = str((src.parent / data["sql_file"]).resolve())
    if raw.get("env_lock"):  # football specs name a requirements lock relative to the spec; a copy elsewhere would lose it
        raw["env_lock"] = str((src.parent / raw["env_lock"]).resolve())
    raw["name"] = name
    dest = root() / "work" / "specs" / f"{name}.yaml"
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; pick another name or edit it directly")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(f"# copied from {src.name}; edit freely\n" + yaml.safe_dump(raw, sort_keys=False))
    return dest


# ------------------------------------------------------------------------------------------------------------------------------- tuning
SPACES = {
    "xgboost": {"max_depth": {"low": 2, "high": 5, "type": "int"}, "learning_rate": {"low": 0.01, "high": 0.1, "log": True},
                "n_estimators": {"low": 100, "high": 600, "type": "int"}, "min_child_weight": {"low": 5, "high": 40, "type": "int"},
                "subsample": {"low": 0.6, "high": 1.0}, "colsample_bytree": {"low": 0.3, "high": 1.0}, "gamma": {"low": 0.0, "high": 2.0},
                "reg_lambda": {"low": 1.0, "high": 20.0, "log": True}},
    "lightgbm": {"num_leaves": {"low": 4, "high": 31, "type": "int"}, "learning_rate": {"low": 0.01, "high": 0.1, "log": True},
                 "n_estimators": {"low": 100, "high": 600, "type": "int"}, "min_child_samples": {"low": 20, "high": 100, "type": "int"},
                 "subsample": {"low": 0.6, "high": 1.0}, "colsample_bytree": {"low": 0.3, "high": 1.0}},
    "logreg": {"C": {"low": 0.001, "high": 10.0, "log": True}},
}


def space(family: str = "xgboost") -> dict:
    """A starting search space for `tune_spec`, to edit: {param: {low, high, [log], [type: int]}} or {param: [choices]}."""
    import copy
    if family not in SPACES:
        raise KeyError(f"no starter space for {family!r}; choose from {sorted(SPACES)}")
    return copy.deepcopy(SPACES[family])


def _load_spec(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _save_spec(path: Path, raw: dict, note: str) -> None:
    path.write_text(f"# {note}\n" + yaml.safe_dump(raw, sort_keys=False))


def _validated(path: Path, raw: dict, note: str) -> Path:
    """Write `raw` to `path` only if the backtest loader accepts it, so a bad edit never leaves a broken spec behind."""
    import copy

    from . import backtest as B
    B.validate(copy.deepcopy(raw))  # raises BacktestError with the reason
    _save_spec(path, raw, note)
    return path


def contenders(spec: str | Path) -> pd.DataFrame:
    """What a spec compares: each contender's family, parameters, seeds, walk-forward and whether it is tuned. A starting point for your own settings.
    `spec` is a path, or the name of a shipped or copied spec."""
    raw = _load_spec(spec if Path(str(spec)).is_file() else _find_spec(str(spec)))
    rows = [{"name": c["name"], "kind": "retrain", "family": c.get("family", "xgboost"), "params": c.get("params", {}), "seeds": c.get("seeds"),
             "walk_forward": c.get("walk_forward"), "tuned": bool(c.get("tune"))} for c in raw.get("retrain", [])]
    rows += [{"name": c["name"], "kind": "registered model", "family": "", "params": {}, "seeds": None, "walk_forward": None, "tuned": False} for c in raw.get("models", [])]
    rows += [{"name": c["name"], "kind": "baseline column", "family": "", "params": {}, "seeds": None, "walk_forward": None, "tuned": False} for c in raw.get("columns", [])]
    return pd.DataFrame(rows, columns=["name", "kind", "family", "params", "seeds", "walk_forward", "tuned"])


def _space_for(family: str) -> dict:  # `space` is shadowed by tune_spec's own argument of that name
    return globals()["space"](family)


def tune_spec(base: str, name: str, contender: str, *, space: dict | None = None, trials: int = 40, inner_folds: int = 3, seeds: list[int] | None = None,
              seed: int = 7) -> Path:
    """A copy of `base` with a tuned twin of one contender added: `<contender>_tuned` runs a random search over `space`.

    The search uses ONLY the seasons before the evaluation window, ranked by an inner walk-forward, and the winner is frozen and then scored on seasons no
    candidate ever saw. The untuned contender stays in the spec so the comparison is fair. The winner's inner score is best-of-N and optimistic: only the outer
    numbers (see `results`) count. Needs a spec that walks forward by whole groups (the football specs do)."""
    import copy
    path = new_spec(base, name)
    try:
        raw = _load_spec(path)
        src = next((c for c in raw.get("retrain", []) if c["name"] == contender), None)
        if src is None:
            raise KeyError(f"{contender!r} is not a retrain contender of {base}; choose from {[c['name'] for c in raw.get('retrain', [])]}")
        tuned = copy.deepcopy(src)
        tuned["name"] = f"{contender}_tuned"[:41]
        if any(c["name"] == tuned["name"] for c in raw["retrain"]):
            raise ValueError(f"{base} already has {tuned['name']}")
        fam = tuned.get("family", "xgboost")
        tuned["tune"] = {"trials": int(trials), "inner_folds": int(inner_folds), "seed": int(seed), "space": space if space is not None else _space_for(fam)}
        if seeds:
            tuned["seeds"] = list(seeds)
        raw["retrain"].append(tuned)
        raw["name"] = name
        return _validated(path, raw, f"{contender} + a tuned twin ({trials} random trials); copied from {Path(base).stem}")
    except Exception:
        path.unlink(missing_ok=True)  # never leave a half-made copy that would block the next try with the same name
        raise


def _slug(key: str, value) -> str:
    abbr = "".join(w[0] for w in key.split("_"))
    return f"{abbr}{str(value).replace('.', 'p').replace('-', 'm')}"


def grid_spec(base: str, name: str, template: str, grid: dict[str, list], *, max_contenders: int = 12) -> Path:
    """A copy of `base` with one contender per combination of `grid`, each a copy of `template` with those parameters changed.

    Every contender is scored on the same games with paired confidence intervals. With many combinations, the best one looks good partly by luck: treat the
    winner as a candidate and confirm it on games none of them has seen, and keep `max_contenders` small. A grid of 12 already needs differences of a couple of
    accuracy points to mean anything on about 2,000 games."""
    import copy
    import itertools
    if not grid or any(not v for v in grid.values()):
        raise ValueError("grid needs at least one parameter with at least one value, e.g. {'max_depth': [2, 3], 'learning_rate': [0.03, 0.05]}")
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    if len(combos) > max_contenders:
        raise ValueError(f"{len(combos)} combinations is more than max_contenders={max_contenders}: each one adds minutes and adds to the chance that the winner is luck")
    path = new_spec(base, name)
    try:
        raw = _load_spec(path)
        tpl = next((c for c in raw.get("retrain", []) if c["name"] == template), None)
        if tpl is None:
            raise KeyError(f"{template!r} is not a retrain contender of {base}; choose from {[c['name'] for c in raw.get('retrain', [])]}")
        added = []
        for combo in combos:
            c = copy.deepcopy(tpl)
            c.pop("tune", None)
            c["params"] = {**c.get("params", {}), **dict(zip(keys, combo))}
            c["name"] = (template + "_" + "_".join(_slug(k, v) for k, v in zip(keys, combo)))[:41]
            if c["name"] in [x["name"] for x in raw["retrain"]] + [a["name"] for a in added]:
                raise ValueError(f"two combinations would both be called {c['name']!r}: shorten the template's name or the values")
            added.append(c)
        raw["retrain"] = [c for c in raw["retrain"] if c["name"] != template] + [tpl] + added
        raw["name"] = name
        return _validated(path, raw, f"a {len(added)}-point grid over {', '.join(keys)} around {template}; copied from {Path(base).stem}")
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _results_json(job_id: str) -> dict:
    try:
        p = next((p for p in runner().outputs(job_id) if p.name == "results.json"), None)
    except (KeyError, OSError, ValueError):  # a job that has not got as far as writing its spec's outputs
        p = None
    if p is None:
        try:
            state = runner().status(job_id).get("state")
        except Exception:  # noqa: BLE001  a status lookup must not hide the message that matters
            state = "in an unknown state"
        raise FileNotFoundError(f"job {job_id} has no results.json: it is {state}. See nb.report({job_id!r}) or its log.")
    return json.loads(p.read_text())


def results(job_id: str, vs: str | None = None) -> pd.DataFrame:
    """Every contender of a finished backtest, best log loss first, with the paired difference against `vs` and a plain-words verdict.

    `vs` defaults to Elo if present, else the always-home prior. A negative difference means better than `vs`. The verdict comes from the confidence interval:
    only "better" or "worse" mean anything; "not distinguishable" is the usual honest answer."""
    res = _results_json(job_id)
    m = res["metrics"]
    vs = vs or ("elo" if "elo" in m else "home_prior")
    if vs not in m:
        raise KeyError(f"{vs!r} is not in this backtest; choose from {sorted(m)}")
    rows = []
    for name, v in m.items():
        row = {"contender": name, "log_loss": v["log_loss"], "accuracy": v["accuracy"], "brier": v.get("brier"), "auc": v.get("auc"), "vs": vs}
        d = R._paired(res, name, vs) if name != vs else None
        if d:
            lo, hi = d["ci"]
            row.update(diff=d["diff"], ci_low=lo, ci_high=hi, verdict="better" if hi < 0 else "worse" if lo > 0 else "not distinguishable")
        else:
            row.update(diff=None, ci_low=None, ci_high=None, verdict="(reference)" if name == vs else "")
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    df.attrs.update(games=res.get("games_compared"), date_range=res.get("date_range"))
    return df


def tuned(job_id: str) -> pd.DataFrame:
    """For each tuned contender: what the search found, and how much of its inner gain is probably luck (best-of-N is optimistic by construction)."""
    res = _results_json(job_id)
    rows = []
    for name, info in res.get("info", {}).items():
        t = info.get("tuned")
        if t:
            rows.append({"contender": name, "trials": t["n_trials"], "default_inner_loss": t["default_inner_log_loss"], "best_inner_loss": t["best_inner_log_loss"],
                         "inner_gain": t["default_inner_log_loss"] - t["best_inner_log_loss"], "tuned_on": " to ".join(t["tuning_window"]), "best_params": t["best_params"]})
    return pd.DataFrame(rows, columns=["contender", "trials", "default_inner_loss", "best_inner_loss", "inner_gain", "tuned_on", "best_params"])


def trials(job_id: str, contender: str) -> pd.DataFrame:
    """The best few candidates the search tried for one tuned contender, one parameter per column. If the top few are all close, the surface is flat and the
    'best' is not special."""
    t = (_results_json(job_id).get("info", {}).get(contender) or {}).get("tuned")
    if not t:
        raise KeyError(f"{contender!r} was not tuned in this job; see nb.tuned({job_id!r})")
    return pd.DataFrame([{"inner_log_loss": x["inner_log_loss"], **x["params"]} for x in t["top_trials"]])


# ------------------------------------------------------------------------------------------------------------------------------------ cli
def mllab(*argv: str) -> int:
    """Run `mllab <argv>` in this process and show its output. Same code path as the command line, same audit log, same safeguards."""
    from . import cli
    try:
        return cli.main(list(argv)) or 0
    except SystemExit as e:  # argparse exits on a usage error; a notebook should show the message and carry on
        return int(e.code or 0)


def capture(*argv: str) -> tuple[int, str]:
    """Like mllab(), but returns (exit code, text) instead of printing."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = mllab(*argv)
    return rc, buf.getvalue()


def submit(kind: str, spec: str | Path, *, max_gb: float | None = None) -> str:
    """Submit a backtest or training job from a spec file and return its job id. `kind` is "backtest" or "train". Raises with the CLI's message if it refuses."""
    if kind not in ("backtest", "train"):
        raise ValueError('kind must be "backtest" or "train"')
    rc, out = capture(kind, "run", str(spec), *(["--max-gb", str(max_gb)] if max_gb else []))
    if rc != 0 or not out.strip():
        raise RuntimeError(out.strip()[-800:] or f"mllab {kind} run failed (exit {rc})")
    return out.strip().splitlines()[0]


def report(job_id: str) -> None:
    """Print a finished backtest or training job's report; if it has none, say how the job ended and show the end of its log."""
    r = runner()
    rp = next((p for p in r.outputs(job_id) if p.name == "report.md"), None) if r.status(job_id).get("state") == "finished" else None
    if rp:
        print(rp.read_text())
    else:
        print(f"No report. The job is {r.status(job_id).get('state')}. End of its log:\n")
        print(r.logs(job_id)[0][-3000:])


def wait(job_id: str, *, every: float = 15.0, timeout: float = 7200.0, show=print, sleep=time.sleep, clock=time.monotonic) -> str:
    """Follow a lab job until it ends, printing each change of state. Returns the final state. Stopping the cell does not stop the job."""
    r, t0, last = runner(), clock(), None
    while True:
        st = r.status(job_id).get("state", "?")
        if st != last:
            show(f"{job_id}: {st}")
            last = st
        if st in TERMINAL:
            return st
        if clock() - t0 > timeout:
            show(f"still {st} after {int(timeout)} s; it keeps running. Check again with nb.jobs().")
            return st
        sleep(every)
