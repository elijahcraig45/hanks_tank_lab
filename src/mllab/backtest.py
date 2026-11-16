"""Backtest orchestration: build a job from a spec, run it in the models' pinned environment, read results.

spec.yaml:
  name: v10_vs_retrains
  data:   {sql_file: sql/mlb_v10_pregame.sql}     # runs through bq()'s cost + result guards, cached as parquet
          # or {file: features.parquet}           # already under the lab's data dir
  label: home_win        date: game_date        id: game_pk
  start: 2026-04-18      end: 2026-09-27
  models:   [{version: {project: hankstank, model: game_outcome, version: v10}, name: v10, parity: true}]
  columns:  [{name: prod, column: prod_prob, reference: true}, {name: elo, column: elo_home_win_prob}]
  history:  {file: mlb_history_2015_2025.parquet, label: home_won}   # optional warm start for retrains (older games)
  retrain:  [{name: depth2, features_from: v10, params: {max_depth: 2}, walk_forward: {min_train: 0, step: 100}}]
  tiers / thresholds / bootstrap / seed: optional
  env: lock (default; per-lock venv) | host (this machine's python, for dev)
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from .jobs import JobError, JobSpec, LocalRunner
from .registry import Registry, RegistryError

WORKER = Path(__file__).with_name("backtest_worker.py")
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,40}$")
ALLOWED = {"name", "data", "history", "env_lock", "filter", "label", "date", "id", "start", "end", "models", "columns", "retrain", "baseline",
           "tiers", "thresholds", "bootstrap", "seed", "env", "timeout_s", "mem_mb"}


class BacktestError(Exception):
    pass


@dataclass
class Prepared:
    job_spec: JobSpec
    workdir: Path
    worker_spec: dict


def load(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return validate(raw)


def _iso(v):
    """YAML turns `start: 2026-04-18` into a date object; the worker spec is JSON, so store an ISO string."""
    if v is None:
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()[:10]
    return str(v)


def validate(raw: dict) -> dict:
    for k in ("start", "end"):
        if k in raw:
            raw[k] = _iso(raw[k])
    extra = set(raw) - ALLOWED
    if extra:
        raise BacktestError(f"unknown spec key(s): {sorted(extra)}")
    if not NAME.match(str(raw.get("name", ""))):
        raise BacktestError("spec needs a name like 'v10_vs_retrains'")
    d = raw.get("data") or {}
    if bool(d.get("sql_file")) == bool(d.get("file")):
        raise BacktestError("data needs exactly one of sql_file or file")
    if not (raw.get("models") or raw.get("columns") or raw.get("retrain")):
        raise BacktestError("nothing to compare: add models, columns or retrain")
    flt = raw.get("filter")
    if flt is not None and not (isinstance(flt, dict) and isinstance(flt.get("column"), str) and ("equals" in flt) != ("in" in flt)
                                and (isinstance(flt.get("in"), list) or "equals" in flt)):
        raise BacktestError("filter must be {column: NAME, equals: VALUE} or {column: NAME, in: [VALUES]}")
    hist = raw.get("history")
    if hist is not None and (not isinstance(hist, dict) or not hist.get("file") or "/" in str(hist["file"])):
        raise BacktestError("history needs {file: name.parquet} directly under the lab's data dir")
    names = [c["name"] for k in ("models", "columns", "retrain") for c in raw.get(k, [])]
    if len(names) != len(set(names)) or "home_prior" in names:
        raise BacktestError("contender names must be unique and not 'home_prior'")
    for n in names:
        if not NAME.match(n):
            raise BacktestError(f"bad contender name {n!r}")
    for m in raw.get("models", []):
        if not isinstance(m.get("version"), dict) or not {"project", "model", "version"} <= set(m["version"]):
            raise BacktestError(f"model {m.get('name')}: version needs project, model, version")
    ref = [c for c in raw.get("columns", []) if c.get("reference")]
    if any(m.get("parity") for m in raw.get("models", [])) and len(ref) != 1:
        raise BacktestError("parity needs exactly one column marked reference: true")
    for r in raw.get("retrain", []):
        f = r.get("features")
        if not (f or r.get("features_from")):
            raise BacktestError(f"retrain {r['name']}: give features or features_from")
        if isinstance(f, dict) and f.get("exclude_pattern"):
            try:
                re.compile(f["exclude_pattern"])
            except re.error as e:
                raise BacktestError(f"retrain {r['name']}: bad exclude_pattern: {e}") from e
        if isinstance(f, dict) and not f.get("all_numeric"):
            raise BacktestError(f"retrain {r['name']}: features as a mapping must be {{all_numeric: true, exclude: [...]}}")
        tn = r.get("tune")
        if tn is not None:
            if not (isinstance(tn, dict) and isinstance(tn.get("space"), dict) and tn["space"]):
                raise BacktestError(f"retrain {r['name']}: tune needs a non-empty space mapping")
            if r.get("family", "xgboost") == "logreg" and "C" not in tn["space"]:
                raise BacktestError(f"retrain {r['name']}: logreg can only tune C")
            if not (1 <= int(tn.get("trials", 20)) <= 200):
                raise BacktestError(f"retrain {r['name']}: tune.trials must be 1..200")
            if not (r.get("walk_forward") or {}).get("group"):
                raise BacktestError(f"retrain {r['name']}: tune needs walk_forward.group and min_groups >= 3")
            for k, v in tn["space"].items():
                ok = isinstance(v, list) and v or (isinstance(v, dict) and {"low", "high"} <= set(v) and v["low"] < v["high"])
                if not ok:
                    raise BacktestError(f"retrain {r['name']}: tune.space.{k} must be a list of choices or {{low, high, [log], [type: int]}}")
                if isinstance(v, dict) and v.get("log") and v["low"] <= 0:
                    raise BacktestError(f"retrain {r['name']}: tune.space.{k}: log scale needs low > 0")
        sd = r.get("seeds")
        if sd is not None and (not isinstance(sd, list) or not sd or len(sd) > 20 or not all(isinstance(x, int) for x in sd)):
            raise BacktestError(f"retrain {r['name']}: seeds must be a list of 1-20 integers")
        if sd and r.get("family", "xgboost") == "logreg":
            raise BacktestError(f"retrain {r['name']}: logreg is deterministic; seeds only apply to xgboost/lightgbm")
        if r.get("impute", "median") not in ("median", "zero"):
            raise BacktestError(f"retrain {r['name']}: impute must be median or zero")
        if r.get("features_from") and r["features_from"] not in [m["name"] for m in raw.get("models", [])]:
            raise BacktestError(f"retrain {r['name']}: features_from must name one of the models")
    return raw


def prepare_data(raw: dict, data_dir: Path, *, bq_kwargs: dict | None = None, base: Path = Path(".")) -> str:
    """Return the file name (under data_dir) the job should read. SQL goes through bq()'s guards."""
    d = raw["data"]
    data_dir.mkdir(parents=True, exist_ok=True)
    if d.get("file"):
        if "/" in d["file"] or not (data_dir / d["file"]).is_file():
            raise BacktestError(f"data file must exist directly under {data_dir}: {d['file']!r}")
        return d["file"]
    from .bq import bq

    sql = (base / d["sql_file"]).read_text()
    df = bq(sql, **(bq_kwargs or {}))
    out = f"{raw['name']}.parquet"
    tmp = data_dir / f".{out}.tmp"
    df.to_parquet(tmp)
    tmp.replace(data_dir / out)
    return out


def _stage_plugin(ref: str, base: Path, workdir: Path, name: str) -> str:
    """Copy a predictor plugin into the job's code dir; the job runs with that dir as its cwd."""
    path, _, fn = ref.partition(":")
    src = (base / path).resolve()
    if not src.is_file() or src.suffix != ".py":
        raise BacktestError(f"predictor plugin not found (relative to the spec): {path}")
    (workdir / "plugins").mkdir(exist_ok=True)
    shutil.copy2(src, workdir / "plugins" / f"{name}.py")
    return f"plugins/{name}.py:{fn or 'load'}"


def prepare(raw: dict, reg: Registry, data_file: str, *, workdir: Path | None = None, base: Path = Path(".")) -> Prepared:
    validate(raw)
    workdir = Path(workdir or tempfile.mkdtemp(prefix="mllab-bt-"))
    workdir.mkdir(parents=True, exist_ok=True)
    models, locks = [], set()
    lock_src = None
    for m in raw.get("models", []):
        v = m["version"]
        try:
            pkl = reg.verify(v["project"], v["model"], v["version"])
            meta = reg.meta(v["project"], v["model"], v["version"])
        except RegistryError as e:
            raise BacktestError(str(e)) from e
        vdir = reg.version_dir(v["project"], v["model"], v["version"])
        locks.add(meta.lock_hash)
        lock_src = lock_src or (vdir / "requirements.lock" if (vdir / "requirements.lock").is_file() else None)
        models.append({"name": m["name"], "pickle": str(pkl), "sha256": meta.sha256, "parity": bool(m.get("parity")),
                       "code": [str(vdir / "code")] if (vdir / "code").is_dir() else [],
                       **({"predictor": _stage_plugin(m["predictor"], base, workdir, m["name"])} if m.get("predictor") else {})})
    if len(locks) > 1:
        raise BacktestError("these models were registered with different requirements locks; a single job has one environment. "
                            "Backtest them separately, or re-register with a shared lock.")
    env = raw.get("env", "lock")
    if raw.get("env_lock"):  # an explicit lock wins; it is the only way for a retrain-only spec to get a pinned environment
        lock_src = (base / raw["env_lock"]).resolve()
        if not lock_src.is_file():
            raise BacktestError(f"env_lock not found (relative to the spec): {raw['env_lock']}")
    if env == "lock" and models and lock_src is None:
        raise BacktestError("models have no requirements.lock; register with --lock, or set env: host")
    if env == "lock" and not models and lock_src is None:
        raise BacktestError("retrain-only backtests need env: host, env_lock: FILE, or a model so its lock is used")
    wspec = {k: raw[k] for k in ("label", "date", "id", "start", "end", "filter", "columns", "retrain", "baseline",
                                 "tiers", "thresholds", "bootstrap", "seed") if k in raw}
    wspec.update(data=f"../inputs/{data_file}", models=models, out_dir="outputs")
    inputs = [data_file]
    if raw.get("history"):
        wspec["history"] = {**raw["history"], "file": f"../inputs/{raw['history']['file']}"}
        inputs.append(raw["history"]["file"])
    shutil.copy2(WORKER, workdir / "backtest_worker.py")
    (workdir / "spec.json").write_text(json.dumps(wspec, indent=2))
    if env == "lock":
        shutil.copy2(lock_src, workdir / "requirements.lock")
    js = JobSpec(name=raw["name"], script="backtest_worker.py", args=["spec.json"], code_dir=str(workdir),
                 timeout_s=int(raw.get("timeout_s", 1800)), mem_mb=int(raw.get("mem_mb", 3072)), cpu_pct=200,
                 lock="requirements.lock" if env == "lock" else None, inputs=inputs)
    return Prepared(js, workdir, wspec)


def submit(raw: dict, reg: Registry, runner: LocalRunner, *, bq_kwargs: dict | None = None, base: Path = Path(".")) -> str:
    data_file = prepare_data(raw, runner.data_root, bq_kwargs=bq_kwargs, base=base)
    prep = prepare(raw, reg, data_file, base=base)
    try:
        return runner.submit(prep.job_spec)
    finally:
        shutil.rmtree(prep.workdir, ignore_errors=True)


def results(runner: LocalRunner, job_id: str) -> dict:
    st = runner.status(job_id)
    if st["state"] != "finished":
        raise JobError(f"job {job_id} is {st['state']}; no results (see `mllab job logs {job_id}`)")
    for p in runner.outputs(job_id):
        if p.name == "results.json":
            return json.loads(p.read_text())
    raise JobError(f"job {job_id} finished without results.json")


def report_path(runner: LocalRunner, job_id: str) -> Path | None:
    return next((p for p in runner.outputs(job_id) if p.name == "report.md"), None)
