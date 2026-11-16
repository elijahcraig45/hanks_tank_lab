"""The starter notebooks are product, so they are tested like it: current with their generator, free of saved output, every cell valid Python, every helper they
call real, every %%bq header parsed by the real magic, and the offline cells actually executed against a lab on disk."""
import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

from mllab import nb
from mllab.magics import _parse
from mllab.pipeline import broker, runner as prunner

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb"))


def _gen():
    spec = importlib.util.spec_from_file_location("make_notebooks", ROOT / "scripts" / "make_notebooks.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cells(path, kind="code"):
    return ["".join(c["source"]) for c in json.loads(path.read_text())["cells"] if c["cell_type"] == kind]


def python_of(src: str) -> str | None:
    """The cell as plain Python, or None for a %%cell magic (its body is SQL, not Python). Line magics like %load_ext are dropped."""
    if src.lstrip().startswith("%%"):
        return None
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("%"))


def test_the_committed_notebooks_are_what_the_generator_writes():
    assert _gen().main(["--check"]) == 0, "run: PYTHONPATH=src python3 scripts/make_notebooks.py"


def test_there_is_a_start_here_and_the_quickstart():
    names = {p.name for p in NOTEBOOKS}
    assert {"00_start_here.ipynb", "quickstart.ipynb"} <= names and len(names) >= 6


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_a_notebook_is_clean_and_valid(path):
    raw = json.loads(path.read_text())
    assert raw["nbformat"] == 4 and raw["cells"] and raw["cells"][0]["cell_type"] == "markdown"
    for c in raw["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None, "saved output: it may carry data or secrets"
        if python_of("".join(c["source"])) is not None and c["cell_type"] == "code":
            ast.parse(python_of("".join(c["source"])))  # a syntax error would only show when someone runs the cell


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_nb_helper_a_notebook_calls_exists(path):
    for src in cells(path):
        py = python_of(src)
        if py is None:
            continue
        for attr in re.findall(r"\bnb\.([A-Za-z_]\w*)", py):
            assert hasattr(nb, attr), f"{path.name} calls nb.{attr}, which does not exist"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_bq_magic_header_parses_with_the_real_parser(path):
    for src in cells(path):
        if src.lstrip().startswith("%%bq"):
            opts = _parse(src.splitlines()[0][len("%%bq"):])
            assert opts["var"], "name the DataFrame, or the result is only displayed"
            assert opts["max_gb"] is not None, "state the scan cap explicitly: the guard's default is not a decision"


def test_the_sql_in_the_explore_notebook_uses_columns_the_table_specs_declare():
    src = "\n".join(s for s in cells(NB_DIR / "01_explore_data.ipynb") if s.lstrip().startswith("%%bq"))
    for col in ("game_pk", "home_win_probability", "predicted_at", "game_time_utc", "home_score", "away_score", "status", "synced_at"):
        assert col in src
        specs = "".join((ROOT / "specs" / "bq" / f).read_text() for f in ("game_predictions.yaml", "games.yaml"))
        assert col in specs, f"{col} is not in the table specs: the query may be wrong"


# --------------------------------------------------------------------------------------------------------------------------- executed
@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("MLLAB_ROOT", str(tmp_path))
    for k in ("MLLAB_REGISTRY", "BROKER_STATUS_DIR", "MLLAB_SPECS"):
        monkeypatch.delenv(k, raising=False)
    for d in ("registry", "jobs", "work", "pipeline-status", "pj", "ps"):
        (tmp_path / d).mkdir()
    prunner.write_mode("nfl-weekly-predict", "dry", tmp_path / "pj")
    prunner.record("nfl-weekly-predict", {"started": "2026-10-03T06:00:00+00:00", "finished": "2026-10-03T06:01:00+00:00", "ok": False, "seconds": 3, "error": "x"}, tmp_path / "ps")
    broker.cmd_snapshot(jobs_dir=tmp_path / "pj", state_dir=tmp_path / "ps", release=tmp_path / "none", show=lambda u: {}, status_dir=tmp_path / "pipeline-status")
    return tmp_path


def execute(name, stop_before=None, ns=None):
    ns = ns if ns is not None else {}
    for src in cells(NB_DIR / name):
        py = python_of(src)
        if py is None:
            continue
        if stop_before and re.search(stop_before, py):
            break
        exec(compile(py, name, "exec"), ns)
    return ns


@pytest.mark.parametrize("name", ["00_start_here.ipynb", "04_pipeline_and_control.ipynb"])
def test_the_offline_notebooks_run_end_to_end(lab, name, capsys):
    execute(name)
    out = capsys.readouterr().out
    assert "Traceback" not in out and len(out) > 20


def test_the_pipeline_notebook_shows_the_failed_job(lab):
    ns = execute("04_pipeline_and_control.ipynb")
    assert "nfl-weekly-predict" in " ".join(ns["pl"]["job"][(ns["pl"]["last_ok"] == False)])  # noqa: E712


@pytest.mark.parametrize("name,kind,spec", [("02_backtest_models.ipynb", "backtest", "mlb_v10_param_sweep"), ("03_train_and_register.ipynb", "train", "train_v11_d2")])
def test_the_spec_a_notebook_copies_is_accepted_by_the_real_loader(lab, name, kind, spec):
    ns = execute(name, stop_before=r"nb\.submit")
    path = ns["path"]
    assert path.is_file() and path.parent == lab / "work" / "specs"
    if kind == "backtest":
        from mllab import backtest as B
        raw = B.load(path)
    else:
        from mllab import train as T
        raw = T.load(path)
    assert raw["name"] == path.stem
    assert Path(raw["data"]["sql_file"]).is_absolute() and Path(raw["data"]["sql_file"]).is_file()  # the copy runs from work/specs, not from the repo


def test_the_tuning_notebook_builds_valid_specs_for_nfl_cfb_and_mlb(lab, monkeypatch):
    """Run every cell with only the long-running calls stubbed, so all three spec-building calls run for real and pass the real validator."""
    import pandas as pd
    from mllab import backtest as B
    monkeypatch.setenv("MLLAB_SPECS", str(ROOT / "specs"))
    monkeypatch.setattr(nb, "submit", lambda kind, spec, **kw: f"stub-{Path(spec).stem}")
    monkeypatch.setattr(nb, "wait", lambda *a, **k: "finished")
    for fn in ("results", "tuned", "trials"):
        monkeypatch.setattr(nb, fn, lambda *a, **k: pd.DataFrame())
    ns = execute("05_tune_models.ipynb")
    for var, contender in (("nfl_path", "xgb_pure_tuned"), ("cfb_path", "xgb_tuned"), ("mlb_path", "xgb_d2_md3_lr0p05")):
        raw = B.load(ns[var])
        assert contender in [c["name"] for c in raw["retrain"]], (var, [c["name"] for c in raw["retrain"]])
    assert ns["nfl_job"].startswith("stub-") and ns["mlb_job"].startswith("stub-")
