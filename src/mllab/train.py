"""Train a model as a job, then register the result as a new immutable version.

spec.yaml:
  name: v11_d2
  data:    {sql_file: ../sql/mlb_v10_pregame.sql}     # same guarded data path as backtests
  history: {file: mlb_history_2015_2025.parquet, label: home_won}
  label: home_win   date: game_date   id: game_pk   start: 2026-04-18
  features_from: {project: hankstank, model: game_outcome, version: v10}      # or: features: [a, b, c]
  family: xgboost | lightgbm | logreg        params: {max_depth: 2}
  train_end: 2026-08-31            # games after this are an untouched holdout, scored next to `columns` and a home prior
  columns: [{name: prod, column: prod_prob}]
  env: lock (default; uses the features_from model's requirements.lock) | host
Then:  mllab train run spec.yaml --wait   ->   mllab register-job JOB hankstank game_outcome v11_d2 --parent v10
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import yaml

from . import backtest as B
from .backtest import BacktestError, Prepared
from .jobs import JobError, JobSpec, LocalRunner
from .registry import Meta, Registry, RegistryError

WORKER = Path(__file__).with_name("train_worker.py")
ALLOWED = {"name", "data", "history", "label", "date", "id", "start", "train_end", "features", "features_from", "family",
           "params", "columns", "tiers", "thresholds", "env", "env_lock", "filter", "impute", "feature_set", "timeout_s", "mem_mb"}


def load(path: str | Path) -> dict:
    return validate(yaml.safe_load(Path(path).read_text()) or {})


def validate(raw: dict) -> dict:
    for k in ("start", "train_end"):
        if k in raw:
            raw[k] = B._iso(raw[k])
    extra = set(raw) - ALLOWED
    if extra:
        raise BacktestError(f"unknown spec key(s): {sorted(extra)}")
    if not B.NAME.match(str(raw.get("name", ""))):
        raise BacktestError("spec needs a name like 'v11_d2'")
    d = raw.get("data") or {}
    if bool(d.get("sql_file")) == bool(d.get("file")):
        raise BacktestError("data needs exactly one of sql_file or file")
    if (raw.get("features") is None) == (raw.get("features_from") is None):
        raise BacktestError("give exactly one of features or features_from")
    f = raw.get("features")
    if isinstance(f, dict):
        if not f.get("all_numeric"):
            raise BacktestError("features as a mapping must be {all_numeric: true, exclude: [...], exclude_pattern: regex}")
    elif f is not None and not (isinstance(f, list) and f):
        raise BacktestError("features must be a non-empty list or an {all_numeric: true, ...} mapping")
    if raw.get("impute", "median") not in ("median", "zero"):
        raise BacktestError("impute must be median or zero")
    if raw.get("feature_set") not in (None, "v10", "v8"):
        raise BacktestError("feature_set must be v10 or v8: it tells the production pipeline which feature table the artifact expects")
    flt = raw.get("filter")
    if flt is not None and not (isinstance(flt, dict) and isinstance(flt.get("column"), str) and ("equals" in flt) != ("in" in flt)):
        raise BacktestError("filter must be {column: NAME, equals: VALUE} or {column: NAME, in: [VALUES]}")
    ff = raw.get("features_from")
    if ff is not None and not (isinstance(ff, dict) and {"project", "model", "version"} <= set(ff)):
        raise BacktestError("features_from needs project, model, version")
    if raw.get("family", "xgboost") not in ("xgboost", "lightgbm", "logreg"):
        raise BacktestError("family must be xgboost, lightgbm or logreg")
    hist = raw.get("history")
    if hist is not None and (not isinstance(hist, dict) or not hist.get("file") or "/" in str(hist["file"])):
        raise BacktestError("history needs {file: name.parquet} directly under the lab's data dir")
    return raw


def prepare(raw: dict, reg: Registry, data_file: str, *, workdir: Path | None = None, base: Path = Path(".")) -> Prepared:
    validate(raw)
    workdir = Path(workdir or tempfile.mkdtemp(prefix="mllab-train-"))
    workdir.mkdir(parents=True, exist_ok=True)
    env = raw.get("env", "lock")
    wspec = {k: raw[k] for k in ("name", "label", "date", "id", "start", "train_end", "family", "params", "columns",
                                 "tiers", "thresholds", "features", "filter", "impute", "feature_set") if k in raw}
    wspec.update(data=f"../inputs/{data_file}", out_dir="outputs")
    lock_src = None
    if raw.get("features_from"):
        v = raw["features_from"]
        try:
            pkl = reg.verify(v["project"], v["model"], v["version"])
            meta = reg.meta(v["project"], v["model"], v["version"])
        except RegistryError as e:
            raise BacktestError(str(e)) from e
        vdir = reg.version_dir(v["project"], v["model"], v["version"])
        wspec["features_from"] = {"name": "base", "pickle": str(pkl), "sha256": meta.sha256,
                                  "code": [str(vdir / "code")] if (vdir / "code").is_dir() else []}
        lock_src = vdir / "requirements.lock" if (vdir / "requirements.lock").is_file() else None
    inputs = [data_file]
    if raw.get("history"):
        wspec["history"] = {**raw["history"], "file": f"../inputs/{raw['history']['file']}"}
        inputs.append(raw["history"]["file"])
    if raw.get("env_lock"):  # explicit lock: the way a spec with explicit features (no registry model to take one from) gets a pinned env
        lock_src = (base / raw["env_lock"]).resolve()
        if not lock_src.is_file():
            raise BacktestError(f"env_lock not found (relative to the spec): {raw['env_lock']}")
    if env == "lock" and lock_src is None:
        raise BacktestError("no requirements.lock to build the environment from; use features_from a model registered with "
                            "--lock, give env_lock: FILE, or set env: host")
    shutil.copy2(WORKER, workdir / "train_worker.py")
    shutil.copy2(B.WORKER, workdir / "backtest_worker.py")
    (workdir / "spec.json").write_text(json.dumps(wspec, indent=2))
    if env == "lock":
        shutil.copy2(lock_src, workdir / "requirements.lock")
    js = JobSpec(name=f"train-{raw['name']}"[:60], script="train_worker.py", args=["spec.json"], code_dir=str(workdir),
                 timeout_s=int(raw.get("timeout_s", 1800)), mem_mb=int(raw.get("mem_mb", 3072)), cpu_pct=200,
                 lock="requirements.lock" if env == "lock" else None, inputs=inputs)
    return Prepared(js, workdir, wspec)


def submit(raw: dict, reg: Registry, runner: LocalRunner, *, bq_kwargs: dict | None = None, base: Path = Path(".")) -> str:
    data_file = B.prepare_data({**raw, "name": raw["name"]}, runner.data_root, bq_kwargs=bq_kwargs, base=base)
    prep = prepare(raw, reg, data_file, base=base)
    try:
        return runner.submit(prep.job_spec)
    finally:
        shutil.rmtree(prep.workdir, ignore_errors=True)


def _sha(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.name):
        h.update(p.name.encode())
        with open(p.resolve(), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def register_from_job(reg: Registry, runner: LocalRunner, job_id: str, project: str, model: str, version: str,
                      *, parent: str | None = None) -> Meta:
    st = runner.status(job_id)
    if st["state"] != "finished":
        raise JobError(f"job {job_id} is {st['state']}; only a finished training job can be registered")
    jd = runner.jobs / job_id
    out = jd / "code" / json.loads((jd / "spec.json").read_text()).get("outputs", "outputs")
    pkl, mj = out / "model.pkl", out / "metrics.json"
    if not pkl.is_file() or not mj.is_file():
        raise JobError(f"job {job_id} has no model.pkl/metrics.json; was it a training job?")
    m = json.loads(mj.read_text())
    compact = {"train_rows": m["train_rows"], "train_end": m["train_end"], "holdout_games": m.get("holdout_games_compared"),
               "family": m["family"], "params": m["params"], "job": job_id,
               "holdout": {k: {x: v[x] for x in ("accuracy", "log_loss", "brier", "auc", "n")} for k, v in m["holdout"].items()}}
    lock = jd / "code" / "requirements.lock"
    return reg.register(project, model, version, pkl, parent=parent, features=m["features"], metrics=compact,
                        data_hash=_sha([p for p in (jd / "inputs").iterdir()]), lockfile=lock if lock.is_file() else None,
                        train_script=jd / "code" / "train_worker.py", source=f"job:{job_id}")
