"""The stager must reproduce deploy_nfl.sh and deploy_cfb.sh, because the football code imports itself the way those scripts lay it out."""
import json
from pathlib import Path

import pytest

from mllab.pipeline import stage as S


# the cp lines of the real deploy scripts that matter to staging, as they are written there
NFL_SCRIPT = '''
    cp "$NFL_DIR"/*.py "$NFL_DIR/requirements.txt" "$SRC_DIR"/
    cp -R "$ROOT_SRC/rankings" "$ROOT_SRC/stats" "$ROOT_SRC/season_sim" "$SRC_DIR"/
    cp "$ROOT_SRC/rankings/http.py" "$SRC_DIR/http_transport.py"
    cp "$ROOT_SRC/model_control.py" "$SRC_DIR"/
'''
CFB_SCRIPT = '''
    cp "$CFB_DIR"/*.py "$STAGE"/
    cp "$NFL_DIR/features.py" "$NFL_DIR/models.py" "$NFL_DIR/margin_ridge.py" "$STAGE"/
    cp "$NFL_DIR/drive_sim.py" "$CFB_DIR/cfbd_team_crosswalk.json" "$STAGE"/
    cp "$CFB_DIR/cfb_config.py" "$STAGE/config.py"
    cp -R "$SRC_DIR/rankings" "$SRC_DIR/stats" "$SRC_DIR/season_sim" "$STAGE"/
    cp "$SRC_DIR/rankings/http.py" "$STAGE/http_transport.py"
    cp "$SRC_DIR/model_control.py" "$STAGE"/
'''


def make_repo(tmp_path):
    r = tmp_path / "repo"
    files = {
        "src/cloud_function_main.py": "MLB", "src/requirements.txt": "pandas==2.3.3\nxgboost==3.1.3\nrequests==2.32.5\nfunctions-framework>=3.0.0\n",
        "src/nfl/main.py": "NFL_MAIN", "src/nfl/config.py": "NFL_CONFIG", "src/nfl/features.py": "FEATURES", "src/nfl/models.py": "MODELS",
        "src/nfl/margin_ridge.py": "RIDGE", "src/nfl/drive_sim.py": "DRIVESIM",
        "src/nfl/requirements.txt": "functions-framework==3.*\npandas==2.3.3\npolars==1.*\nnflreadpy==0.1.5\n",
        "src/cfb/main.py": "CFB_MAIN", "src/cfb/cfb_config.py": "CFB_CONFIG", "src/cfb/pipeline.py": "CFB_PIPE",
        "src/cfb/cfbd_team_crosswalk.json": "{}",
        "src/rankings/__init__.py": "", "src/rankings/core.py": "CORE", "src/rankings/http.py": "HTTP",
        "src/stats/__init__.py": "", "src/stats/pickem.py": "PICKEM", "src/season_sim/__init__.py": "", "src/season_sim/store.py": "STORE",
        "src/rankings/__pycache__/core.cpython-313.pyc": "junk", "src/tests/test_x.py": "t",
        "src/model_control.py": "MODEL_CONTROL",
        "scripts/gcp/nfl/deploy_nfl.sh": NFL_SCRIPT, "scripts/gcp/cfb/deploy_cfb.sh": CFB_SCRIPT,
    }
    for rel, text in files.items():
        p = r / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return r


def read(p):
    return p.read_text()


def test_nfl_bundle_is_flattened_with_packages_kept(tmp_path):
    out = tmp_path / "rel"
    S.stage_release(make_repo(tmp_path), out)
    nfl = out / "nfl"
    assert read(nfl / "main.py") == "NFL_MAIN" and read(nfl / "config.py") == "NFL_CONFIG"
    assert read(nfl / "http_transport.py") == "HTTP"  # rankings/http.py, renamed
    assert read(nfl / "rankings" / "core.py") == "CORE" and (nfl / "stats" / "pickem.py").exists() and (nfl / "season_sim" / "store.py").exists()
    assert not (nfl / "cfb").exists() and not (nfl / "pipeline.py").exists()  # nothing from CFB


def test_nfl_requirements_gain_scipy_exactly_once(tmp_path):
    out = tmp_path / "rel"
    S.stage_release(make_repo(tmp_path), out)
    reqs = read(out / "nfl" / "requirements.txt").splitlines()
    assert reqs.count("scipy==1.16.3") == 1 and "polars==1.*" in reqs


def test_cfb_bundle_has_its_own_config_and_the_shared_football_core(tmp_path):
    out = tmp_path / "rel"
    S.stage_release(make_repo(tmp_path), out)
    cfb = out / "cfb"
    assert read(cfb / "config.py") == "CFB_CONFIG"  # cfb_config.py renamed: features.py imports a module called `config`
    assert read(cfb / "cfb_config.py") == "CFB_CONFIG"
    for name, text in {"features.py": "FEATURES", "models.py": "MODELS", "margin_ridge.py": "RIDGE", "drive_sim.py": "DRIVESIM"}.items():
        assert read(cfb / name) == text
    assert (cfb / "cfbd_team_crosswalk.json").exists() and read(cfb / "http_transport.py") == "HTTP"
    assert read(cfb / "main.py") == "CFB_MAIN" and read(out / "nfl" / "main.py") == "NFL_MAIN"  # two main.py, kept apart
    assert read(cfb / "requirements.txt") == S.CFB_REQUIREMENTS


def test_mlb_bundle_is_the_src_tree_without_junk(tmp_path):
    out = tmp_path / "rel"
    files = S.stage_release(make_repo(tmp_path), out)
    assert read(out / "mlb" / "cloud_function_main.py") == "MLB"
    assert not [f for f in files["mlb"] if "__pycache__" in f or f.endswith(".pyc") or f.startswith("tests/")]


def test_restaging_replaces_the_destination(tmp_path):
    out = tmp_path / "rel"
    repo = make_repo(tmp_path)
    S.stage_release(repo, out)
    (out / "stale.txt").write_text("old")
    S.stage_release(repo, out)
    assert not (out / "stale.txt").exists()


def test_not_a_checkout_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        S.stage_release(tmp_path, tmp_path / "rel")


def test_merged_requirements_keep_one_spec_per_package(tmp_path):
    out = tmp_path / "rel"
    S.stage_release(make_repo(tmp_path), out)
    reqs = S.merged_requirements(out)
    names = [S._name(r) for r in reqs]
    assert len(names) == len(set(names))
    assert "pandas==2.3.3" in reqs and "polars==1.*" in reqs and "nflreadpy==0.1.5" in reqs and "scipy==1.16.3" in reqs
    assert any(r.startswith("google-cloud-secret-manager") for r in reqs)  # the runner resolves sm:// references


def test_an_exact_pin_beats_a_range(tmp_path):
    out = tmp_path / "rel"
    S.stage_release(make_repo(tmp_path), out)
    # the MLB tree says functions-framework>=3.0.0; make the NFL bundle pin it exactly
    (out / "nfl" / "requirements.txt").write_text("functions-framework==3.8.1\n")
    assert "functions-framework==3.8.1" in S.merged_requirements(out)


def test_two_different_exact_pins_are_an_error_not_a_coin_flip(tmp_path):
    out = tmp_path / "rel"
    S.stage_release(make_repo(tmp_path), out)
    (out / "nfl" / "requirements.txt").write_text("pandas==2.2.0\n")
    with pytest.raises(ValueError, match="pandas"):
        S.merged_requirements(out)


def test_write_release_records_what_it_staged(tmp_path):
    out = tmp_path / "rel"
    meta = S.write_release(make_repo(tmp_path), out, ref="abc123")
    assert meta["ref"] == "abc123" and set(meta["bundles"]) == {"mlb", "nfl", "cfb"}
    assert json.loads((out / "RELEASE.json").read_text())["requirements"] == meta["requirements"]
    assert (out / "requirements.lock.txt").read_text().count("\n") == meta["requirements"]


# ---------------------------------------------------------------------------------------------- the control plane client and drift
def test_both_football_bundles_carry_the_control_plane_client(tmp_path):
    # Without it the football code fails open: a pause or pin set in the lab silently does not apply
    out = tmp_path / "rel"
    S.stage_release(make_repo(tmp_path), out)
    assert read(out / "nfl" / "model_control.py") == "MODEL_CONTROL" and read(out / "cfb" / "model_control.py") == "MODEL_CONTROL"
    assert (out / "mlb" / "model_control.py").exists()  # the mlb bundle is all of src/


def test_staging_requires_the_control_plane_client_like_the_deploy_scripts_cp_does(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "src" / "model_control.py").unlink()
    with pytest.raises(FileNotFoundError):
        S.stage_release(repo, tmp_path / "rel")


def test_deploy_script_copies_are_parsed_as_the_shell_reads_them():
    got = S.deploy_script_copies(CFB_SCRIPT)
    assert (["$CFB_DIR/cfb_config.py"], "$STAGE/config.py") in got  # a rename
    assert (["$NFL_DIR/features.py", "$NFL_DIR/models.py", "$NFL_DIR/margin_ridge.py"], "$STAGE/") in got  # several sources, one directory
    assert (["$SRC_DIR/rankings", "$SRC_DIR/stats", "$SRC_DIR/season_sim"], "$STAGE/") in got  # -R is dropped
    assert (["$CFB_DIR/*.py"], "$STAGE/") in got


def test_parity_passes_when_the_bundles_have_everything_the_scripts_copy(tmp_path):
    out = tmp_path / "rel"
    repo = make_repo(tmp_path)
    S.stage_release(repo, out)
    assert S.check_deploy_parity(repo, out) == []


def test_a_new_cp_in_a_deploy_script_makes_staging_fail_loudly_not_silently(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "src" / "new_helper.py").write_text("x")
    script = repo / "scripts" / "gcp" / "nfl" / "deploy_nfl.sh"
    script.write_text(script.read_text() + '    cp "$ROOT_SRC/new_helper.py" "$SRC_DIR"/\n')
    with pytest.raises(S.StageDrift, match=r"nfl/new_helper.py"):
        S.stage_release(repo, tmp_path / "rel")


def test_parity_notices_a_renamed_destination_that_is_missing(tmp_path):
    out = tmp_path / "rel"
    repo = make_repo(tmp_path)
    S.stage_release(repo, out)
    (out / "cfb" / "config.py").unlink()
    assert S.check_deploy_parity(repo, out) == ["cfb/config.py   (copied by scripts/gcp/cfb/deploy_cfb.sh)"]


REAL_REPO = Path.home() / "Documents" / "personalDev" / "mlb" / "hanks_tank_ml"


@pytest.mark.skipif(not (REAL_REPO / "scripts" / "gcp" / "nfl" / "deploy_nfl.sh").is_file(), reason="no local hanks_tank_ml checkout")
def test_the_real_checkout_stages_without_drift(tmp_path):
    """Against the real deploy scripts: if they change and stage.py does not follow, this fails on a developer machine too."""
    out = tmp_path / "rel"
    S.stage_release(REAL_REPO, out)
    assert S.check_deploy_parity(REAL_REPO, out) == []
    for b in ("mlb", "nfl", "cfb"):
        assert (out / b / "model_control.py").exists()
