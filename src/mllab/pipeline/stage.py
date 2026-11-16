"""Build the three function bundles from a hanks_tank_ml checkout, exactly as the deploy scripts stage them.

The functions are not deployed from the repo layout. scripts/gcp/nfl/deploy_nfl.sh and scripts/gcp/cfb/deploy_cfb.sh *flatten* the tree: they
copy nfl/*.py into the function root, rename rankings/http.py to http_transport.py, and for CFB rename cfb_config.py to config.py (NFL and CFB
each have their own config.py). The code imports itself that way, so a git checkout cannot be imported as-is. This reproduces the copies, one
bundle per function, so each job runs against the same module layout it has in production. Each bundle runs in its own process, which is also
what keeps the two config.py files apart.

If the deploy scripts change, this must change with them: tests/test_pipeline_stage.py pins the rules, and `verify` checks a real checkout."""
from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Written by deploy_cfb.sh as a heredoc; there is no requirements file for CFB in the repo.
CFB_REQUIREMENTS = """functions-framework==3.*
google-cloud-bigquery==3.39.0
db-dtypes==1.5.0
pandas==2.3.3
numpy==2.4.0
pyarrow==22.0.0
scikit-learn==1.8.0
xgboost==3.1.3
requests==2.32.5
scipy==1.16.3
"""
# resolves sm:// references in the runner; the functions get secrets injected by Cloud Functions, the homelab fetches them
EXTRA_REQUIREMENTS = ["google-cloud-secret-manager>=2.20"]
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "tests")


def _py(src: Path, dst: Path) -> None:
    for f in sorted(src.glob("*.py")):
        shutil.copy2(f, dst / f.name)


def _tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst / src.name, ignore=_IGNORE, dirs_exist_ok=True)


def _packages(src_root: Path, dst: Path) -> None:
    # Kept as packages: they import each other by package path (`from rankings import core`), which flattening breaks.
    for pkg in ("rankings", "stats", "season_sim"):
        _tree(src_root / pkg, dst)
    shutil.copy2(src_root / "rankings" / "http.py", dst / "http_transport.py")


class StageDrift(Exception):
    """The staged bundle lacks something the deploy script copies: the deploy scripts changed and this file did not follow."""


DEPLOY_SCRIPTS = {"nfl": "scripts/gcp/nfl/deploy_nfl.sh", "cfb": "scripts/gcp/cfb/deploy_cfb.sh"}


def deploy_script_copies(text: str) -> list[tuple[list[str], str]]:
    """Every `cp SRC... DST` in a deploy script, as ([sources], destination). Flags (-R) are dropped; quoting is shell quoting."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("cp "):
            continue
        try:
            toks = [t for t in shlex.split(line, comments=True)[1:] if not t.startswith("-")]
        except ValueError:
            continue
        if len(toks) >= 2:
            out.append((toks[:-1], toks[-1]))
    return out


def check_deploy_parity(repo: Path, dest: Path) -> list[str]:
    """What the deploy scripts put into each function's source root that the staged bundle does not have.

    A copy into a directory (dst ends in '/') names each source file; a copy to a path renames it, so the destination's basename is what
    must exist. Globs are skipped (the *.py copies are done in bulk). The deploy scripts abort on a failed cp (set -e); the code they stage
    often fails OPEN when a file is missing, so the omission would otherwise show up as behaviour that is quietly absent."""
    missing = []
    for bundle, rel in DEPLOY_SCRIPTS.items():
        script = repo / rel
        if not script.is_file():
            continue
        for srcs, dst in deploy_script_copies(script.read_text()):
            names = [Path(x).name for x in srcs if "*" not in x] if dst.endswith("/") else [Path(dst).name]
            for name in names:
                if not (dest / bundle / name).exists():
                    missing.append(f"{bundle}/{name}   (copied by {rel})")
    return sorted(set(missing))


def stage_release(repo: Path, dest: Path) -> dict[str, list[str]]:
    """Create dest/{mlb,nfl,cfb} from the checkout at `repo`. Returns {bundle: [relative files]}."""
    src, nfl_dir, cfb_dir = repo / "src", repo / "src" / "nfl", repo / "src" / "cfb"
    if not src.is_dir():
        raise FileNotFoundError(f"{repo} has no src/: not a hanks_tank_ml checkout")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # mlb-2026-daily-pipeline and mlb-2026-sim-blend: the source root is src/ itself (PYTHONPATH=/workspace in the function)
    shutil.copytree(src, dest / "mlb", ignore=_IGNORE)

    # nfl-weekly-pipeline  (deploy_nfl.sh)
    nfl = dest / "nfl"
    nfl.mkdir()
    _py(nfl_dir, nfl)
    shutil.copy2(nfl_dir / "requirements.txt", nfl / "requirements.txt")
    _packages(src, nfl)
    # Read-only model control plane client. Both deploy scripts stage it ("staging it is what makes the control plane work"); the code
    # fails open when it is absent, so leaving it out is silent: a pause or pin set in the lab would simply not apply.
    shutil.copy2(src / "model_control.py", nfl / "model_control.py")
    req = (nfl / "requirements.txt").read_text()
    if not re.search(r"^scipy", req, re.M):  # the Bradley-Terry fit builds a sparse design matrix
        (nfl / "requirements.txt").write_text(req.rstrip("\n") + "\nscipy==1.16.3\n")

    # cfb-weekly-pipeline  (deploy_cfb.sh)
    cfb = dest / "cfb"
    cfb.mkdir()
    _py(cfb_dir, cfb)
    for name in ("features.py", "models.py", "margin_ridge.py", "drive_sim.py"):  # the shared football core and the shared drive simulator
        shutil.copy2(nfl_dir / name, cfb / name)
    shutil.copy2(cfb_dir / "cfbd_team_crosswalk.json", cfb / "cfbd_team_crosswalk.json")
    # features.py reads its default Elo constants from a module named `config`; in the CFB bundle that name is CFB's own
    shutil.copy2(cfb_dir / "cfb_config.py", cfb / "config.py")
    _packages(src, cfb)
    shutil.copy2(src / "model_control.py", cfb / "model_control.py")
    (cfb / "requirements.txt").write_text(CFB_REQUIREMENTS)

    drift = check_deploy_parity(repo, dest)
    if drift:
        raise StageDrift("the deploy scripts copy files this stager does not stage:\n  " + "\n  ".join(drift)
                         + "\nUpdate mllab/src/mllab/pipeline/stage.py to match the deploy scripts.")
    return {b: sorted(str(p.relative_to(dest / b)) for p in (dest / b).rglob("*") if p.is_file()) for b in ("mlb", "nfl", "cfb")}


# ------------------------------------------------------------------------------------------------------------------------ requirements
def _name(line: str) -> str:
    return re.split(r"[\s=<>~!\[;]", line.strip(), maxsplit=1)[0].lower().replace("_", "-")


def _lines(path: Path) -> list[str]:
    out = []
    for raw in path.read_text().splitlines():
        line = raw.split(" #")[0].strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def merged_requirements(release: Path) -> list[str]:
    """One venv serves all three bundles. Every package must resolve to a single spec: an exact pin beats a range, and two different exact
    pins for one package is an error, because the functions are separate precisely so that they do not have to agree."""
    chosen: dict[str, str] = {}
    sources = [release / b / "requirements.txt" for b in ("mlb", "nfl", "cfb")]
    for path in sources:
        for line in _lines(path) + (EXTRA_REQUIREMENTS if path == sources[-1] else []):
            n = _name(line)
            have = chosen.get(n)
            exact = "==" in line and "*" not in line.split("==", 1)[1]
            if have is None:
                chosen[n] = line
            elif have != line:
                have_exact = "==" in have and "*" not in have.split("==", 1)[1]
                if exact and have_exact:
                    raise ValueError(f"conflicting pins for {n}: {have!r} vs {line!r} ({path.parent.name})")
                if exact:
                    chosen[n] = line
    return sorted(chosen.values(), key=_name)


def write_release(repo: Path, dest: Path, ref: str = "") -> dict:
    files = stage_release(repo, dest)
    reqs = merged_requirements(dest)
    (dest / "requirements.lock.txt").write_text("\n".join(reqs) + "\n")
    meta = {"ref": ref, "staged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bundles": {b: len(f) for b, f in files.items()}, "requirements": len(reqs)}
    (dest / "RELEASE.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--ref", default="")
    a = ap.parse_args(argv)
    print(json.dumps(write_release(a.repo, a.dest, a.ref)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
