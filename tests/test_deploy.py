import hashlib
import shlex

import numpy as np
import pytest

from mllab.deploy import cloudfn, cloudrun
from mllab.deploy.confirm import ConfirmationError
from mllab.registry import Registry, RegistryError

OBJ = "models/vertex/game_outcome_2026_v10/model.pkl"


class Precondition(Exception):
    pass


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket_, self.name, self.metadata = bucket, name, None

    @property
    def generation(self):
        return self.bucket_.objs[self.name]["gen"]

    def download_to_filename(self, path):
        open(path, "wb").write(self.bucket_.objs[self.name]["data"])

    def upload_from_filename(self, path, if_generation_match=None):
        b = self.bucket_
        cur = b.objs.get(self.name, {}).get("gen", 0)
        if if_generation_match is not None and if_generation_match != cur:
            raise Precondition("generation mismatch")
        b.put(self.name, open(path, "rb").read(), self.metadata)


class FakeBucket:
    def __init__(self):
        self.objs, self._gen, self.corrupt_next_upload = {}, 100, False

    def put(self, name, data, metadata=None):
        self._gen += 1
        if self.corrupt_next_upload and metadata:
            data, self.corrupt_next_upload = b"corrupted", False
        self.objs[name] = {"data": data, "gen": self._gen, "meta": metadata}

    def get_blob(self, name):
        if name not in self.objs:
            return None
        b = FakeBlob(self, name); b.metadata = self.objs[name]["meta"]; return b

    def blob(self, name):
        return FakeBlob(self, name)

    def copy_blob(self, src, dest, new_name, if_generation_match=None):
        cur = dest.objs.get(new_name, {}).get("gen", 0)
        if if_generation_match is not None and if_generation_match != cur:
            raise Precondition("generation mismatch")
        dest.put(new_name, src.bucket_.objs[src.name]["data"], self.objs[src.name]["meta"])


class FakeClient:
    def __init__(self, bucket):
        self.b = bucket
    def bucket(self, name):
        return self.b


@pytest.fixture
def setup(tmp_path):
    reg = Registry(tmp_path / "reg")
    old = tmp_path / "old.pkl"; old.write_bytes(b"old-v10")
    new = tmp_path / "new.pkl"; new.write_bytes(b"new-v11")
    reg.register("hankstank", "game", "v10", old)
    reg.register("hankstank", "game", "v11", new, parent="v10")
    reg.promote("hankstank", "game", "v10")
    bucket = FakeBucket(); bucket.put(OBJ, b"old-v10")
    return reg, bucket, FakeClient(bucket), tmp_path / "audit.log"


def plan(reg, client, v="v11"):
    return cloudfn.plan_swap(reg, "hankstank", "game", v, bucket="hanks_tank_data", object_name=OBJ, client=client)


def test_plan_detects_change_and_noop(setup):
    reg, b, c, _ = setup
    p = plan(reg, c)
    assert not p.noop and "current:" in p.diff() and p.backup_object.startswith(cloudfn.BACKUP_PREFIX)
    assert plan(reg, c, "v10").noop  # live object already is v10


def test_swap_requires_typed_confirmation(setup):
    reg, b, c, ap = setup
    with pytest.raises(ConfirmationError):
        cloudfn.apply_swap(plan(reg, c), reg, client=c, confirm=None, audit_path=ap)
    with pytest.raises(ConfirmationError):
        cloudfn.apply_swap(plan(reg, c), reg, client=c, confirm="yes", audit_path=ap)
    assert b.objs[OBJ]["data"] == b"old-v10"


def test_dry_run_changes_nothing(setup):
    reg, b, c, ap = setup
    r = cloudfn.apply_swap(plan(reg, c), reg, client=c, confirm=None, dry_run=True, audit_path=ap)
    assert r["dry_run"] and b.objs[OBJ]["data"] == b"old-v10" and len(b.objs) == 1


def test_swap_backs_up_uploads_verifies_and_promotes(setup):
    reg, b, c, ap = setup
    p = plan(reg, c)
    r = cloudfn.apply_swap(p, reg, client=c, confirm=p.confirm_phrase, audit_path=ap)
    assert r["changed"] and r["previous_version"] == "v10"
    assert b.objs[OBJ]["data"] == b"new-v11"
    assert b.objs[p.backup_object]["data"] == b"old-v10"  # rollback source exists
    assert reg.channels("hankstank", "game")["prod"] == "v11"
    assert reg.deployments("hankstank", "game")[-1]["target"] == "cloudfn-model-swap"
    assert "cloudfn_swap" in open(ap).read()


def test_swap_refuses_if_object_changed_since_plan(setup):
    reg, b, c, ap = setup
    p = plan(reg, c)
    b.put(OBJ, b"someone-else-wrote-this")
    with pytest.raises(RegistryError, match="changed since the plan"):
        cloudfn.apply_swap(p, reg, client=c, confirm=p.confirm_phrase, audit_path=ap)
    assert b.objs[OBJ]["data"] == b"someone-else-wrote-this"


def test_post_upload_verification_failure_is_loud(setup):
    reg, b, c, ap = setup
    p = plan(reg, c); b.corrupt_next_upload = True
    with pytest.raises(RegistryError, match="verification failed"):
        cloudfn.apply_swap(p, reg, client=c, confirm=p.confirm_phrase, audit_path=ap)
    assert reg.channels("hankstank", "game")["prod"] == "v10"  # not promoted


def test_tampered_registry_artifact_is_never_uploaded(setup):
    reg, b, c, ap = setup
    vdir = reg.version_dir("hankstank", "game", "v11"); vdir.chmod(0o755)
    (vdir / "model.pkl").chmod(0o644); (vdir / "model.pkl").write_bytes(b"evil")
    with pytest.raises(RegistryError, match="hash mismatch"):
        plan(reg, c)


def test_rollback_restores_previous_object_and_channel(setup):
    reg, b, c, ap = setup
    p = plan(reg, c)
    cloudfn.apply_swap(p, reg, client=c, confirm=p.confirm_phrase, audit_path=ap)
    with pytest.raises(ConfirmationError):
        cloudfn.rollback_swap(reg, "hankstank", "game", bucket="hanks_tank_data", object_name=OBJ, client=c, confirm=None, audit_path=ap)
    r = cloudfn.rollback_swap(reg, "hankstank", "game", bucket="hanks_tank_data", object_name=OBJ, client=c,
                              confirm="rollback hankstank/game", audit_path=ap)
    assert b.objs[OBJ]["data"] == b"old-v10" and r["channel"] == "v10"
    assert reg.channels("hankstank", "game")["prod"] == "v10"


def test_rollback_with_nothing_to_roll_back(setup):
    reg, b, c, ap = setup
    with pytest.raises(RegistryError, match="no backed-up swap"):
        cloudfn.rollback_swap(reg, "hankstank", "game", bucket="x", object_name=OBJ, client=c, confirm=None)


def test_deploy_script_wrapper_defaults_safe(tmp_path):
    seen = {}
    cmd = cloudfn.run_deploy_script("deploy_v10.sh", [], runner=lambda c, **kw: seen.update(cmd=c, env=kw["env"]))
    assert "--skip-model-upload" in cmd and "--dry-run" in cmd
    assert seen["env"]["CLOUDSDK_CORE_ACCOUNT"] == "you@example.com"
    with pytest.raises(RegistryError):
        cloudfn.run_deploy_script("x.sh", [], account="someone@blocked.example")


# --- Cloud Run -------------------------------------------------------------------------------

@pytest.fixture
def served(tmp_path):
    reg = Registry(tmp_path / "reg")
    lock = tmp_path / "requirements.lock"; lock.write_text("pandas==2.3.3\nnumpy==2.4.0\nscikit-learn==1.8.0\nxgboost==3.1.3\n")
    art = tmp_path / "m.pkl"; art.write_bytes(_real_pickle())
    helper = tmp_path / "model_classes.py"; helper.write_text("class X: pass\n")
    reg.register("hankstank", "game", "v10", art, lockfile=lock, features=["a", "b"], code_snapshot=[helper])
    return reg, tmp_path


def test_bundle_contents(served):
    reg, tmp = served
    out = cloudrun.render_bundle(reg, "hankstank", "game", "v10", tmp / "bundle")
    names = {p.name for p in out.iterdir()}
    assert {"Dockerfile", "app.py", "model.pkl", "features.json", "requirements.txt", "code"} <= names
    reqs = (out / "requirements.txt").read_text().split()
    assert "pandas==2.3.3" in reqs and "fastapi" in reqs and "uvicorn" in reqs
    assert (out / "code" / "model_classes.py").exists()
    compile((out / "app.py").read_text(), "app.py", "exec")  # generated server is valid Python


def test_bundle_needs_lockfile(tmp_path):
    reg = Registry(tmp_path / "r"); a = tmp_path / "m.pkl"; a.write_bytes(b"x")
    reg.register("p", "m", "v1", a)
    with pytest.raises(RegistryError, match="requirements.lock"):
        cloudrun.render_bundle(reg, "p", "m", "v1", tmp_path / "b")


def test_deploy_command_enforces_cost_and_privacy_defaults(served):
    reg, tmp = served
    r = cloudrun.deploy(reg, "hankstank", "game", "v10", service="game-v10", work_dir=tmp / "b", confirm=None)
    cmd = shlex.split(r["command"])
    assert r["dry_run"] and "--no-allow-unauthenticated" in cmd
    assert cmd[cmd.index("--min-instances") + 1] == "0" and cmd[cmd.index("--max-instances") + 1] == "2"


@pytest.mark.parametrize("opts, msg", [({"min_instances": 1}, "scale to zero"), ({"max_instances": 50}, "max_instances")])
def test_deploy_rejects_costly_options(served, opts, msg):
    reg, tmp = served
    with pytest.raises(RegistryError, match=msg):
        cloudrun.deploy(reg, "hankstank", "game", "v10", service="svc", work_dir=tmp / "b", confirm=None, **opts)


def test_deploy_needs_confirmation_then_runs_once(served):
    reg, tmp = served
    calls = []
    run = lambda cmd, **kw: calls.append((cmd, kw["env"]["CLOUDSDK_CORE_ACCOUNT"]))
    with pytest.raises(ConfirmationError):
        cloudrun.deploy(reg, "hankstank", "game", "v10", service="svc", work_dir=tmp / "b", confirm="nope", dry_run=False, runner=run)
    assert calls == []
    cloudrun.deploy(reg, "hankstank", "game", "v10", service="svc", work_dir=tmp / "b", confirm="deploy svc",
                    dry_run=False, runner=run, audit_path=tmp / "a.log")
    assert len(calls) == 1 and calls[0][1] == "you@example.com"
    assert reg.deployments("hankstank", "game")[-1]["target"] == "cloudrun"


@pytest.mark.parametrize("bad", ["Bad_Name", "-x", "x" * 60, "a b"])
def test_service_names_validated(served, bad):
    reg, tmp = served
    with pytest.raises(RegistryError, match="invalid Cloud Run service"):
        cloudrun.deploy(reg, "hankstank", "game", "v10", service=bad, work_dir=tmp / "b", confirm=None)


def test_rollback_command_validated():
    assert "svc-00003-abc=100" in cloudrun.rollback_command("svc", "p", "us-central1", "svc-00003-abc")
    with pytest.raises(RegistryError):
        cloudrun.rollback_command("svc", "p", "us-central1", "x; rm -rf /")


def test_bundle_can_be_rendered_twice(served):
    reg, tmp = served
    cloudrun.render_bundle(reg, "hankstank", "game", "v10", tmp / "again")
    cloudrun.render_bundle(reg, "hankstank", "game", "v10", tmp / "again")  # second deploy must not crash


def test_rollback_of_a_first_ever_swap_clears_the_channel_pointer(setup):
    """Regression: with no previous pointer, rollback used to restore the object but leave the channel on the new version."""
    reg, b, c, ap = setup
    p = plan(reg, c)
    cloudfn.apply_swap(p, reg, client=c, confirm=p.confirm_phrase, channel="staging", audit_path=ap)
    assert reg.channels("hankstank", "game")["staging"] == "v11"
    cloudfn.rollback_swap(reg, "hankstank", "game", bucket="hanks_tank_data", object_name=OBJ, client=c,
                          confirm="rollback hankstank/game", channel="staging", audit_path=ap)
    assert b.objs[OBJ]["data"] == b"old-v10"
    assert "staging" not in reg.channels("hankstank", "game")  # pointer removed, matching the restored object
    assert reg.channels("hankstank", "game")["prod"] == "v10"  # other channels untouched


def test_second_rollback_does_not_undo_the_same_swap_twice(setup):
    """Regression: rollback used to pick the last swap even when it was already rolled back, restoring a stale backup."""
    reg, b, c, ap = setup
    p = plan(reg, c)
    cloudfn.apply_swap(p, reg, client=c, confirm=p.confirm_phrase, audit_path=ap)
    rb = dict(bucket="hanks_tank_data", object_name=OBJ, client=c, confirm="rollback hankstank/game", audit_path=ap)
    cloudfn.rollback_swap(reg, "hankstank", "game", **rb)
    assert b.objs[OBJ]["data"] == b"old-v10"
    b.put(OBJ, b"a-later-legitimate-write")  # someone updates the object after the rollback
    with pytest.raises(RegistryError, match="no backed-up swap"):
        cloudfn.rollback_swap(reg, "hankstank", "game", **rb)
    assert b.objs[OBJ]["data"] == b"a-later-legitimate-write"  # NOT overwritten by the stale backup


def test_rollbacks_unwind_swaps_in_reverse_order(setup, tmp_path):
    reg, b, c, ap = setup
    third = tmp_path / "v12.pkl"; third.write_bytes(b"new-v12"); reg.register("hankstank", "game", "v12", third, parent="v11")
    p1 = plan(reg, c, "v11"); cloudfn.apply_swap(p1, reg, client=c, confirm=p1.confirm_phrase, audit_path=ap)
    p2 = plan(reg, c, "v12"); cloudfn.apply_swap(p2, reg, client=c, confirm=p2.confirm_phrase, audit_path=ap)
    rb = dict(bucket="hanks_tank_data", object_name=OBJ, client=c, confirm="rollback hankstank/game", audit_path=ap)
    cloudfn.rollback_swap(reg, "hankstank", "game", **rb); assert b.objs[OBJ]["data"] == b"new-v11"
    cloudfn.rollback_swap(reg, "hankstank", "game", **rb); assert b.objs[OBJ]["data"] == b"old-v10"
    with pytest.raises(RegistryError):
        cloudfn.rollback_swap(reg, "hankstank", "game", **rb)


def test_rollback_is_recorded_in_the_deployment_history(setup):
    reg, b, c, ap = setup
    p = plan(reg, c); cloudfn.apply_swap(p, reg, client=c, confirm=p.confirm_phrase, audit_path=ap)
    cloudfn.rollback_swap(reg, "hankstank", "game", bucket="hanks_tank_data", object_name=OBJ, client=c, confirm="rollback hankstank/game", audit_path=ap)
    assert [d["target"] for d in reg.deployments("hankstank", "game")] == ["cloudfn-model-swap", "cloudfn-rollback"]


def _real_pickle():
    import pickle as pk
    from xgboost import XGBClassifier
    X = np.random.default_rng(0).normal(size=(40, 2))
    return pk.dumps({"model": XGBClassifier(n_estimators=2).fit(X, (X[:, 0] > 0).astype(int)), "features": ["a", "b"], "fill_values": {}})


LOCK = ["# comment", "google-cloud-aiplatform==1.132.0", "pandas==2.3.3", "numpy==2.4.0", "scikit-learn==1.8.0", "xgboost==3.1.3",
        "pybaseball==2.2.7", "functions-framework>=3.0.0", "lightgbm==4.6.0", "catboost==1.2.10", "MLB-StatsAPI==1.9.0", "scipy==1.16.3", "db-dtypes==1.5.0"]


def test_lean_installs_only_what_the_pickle_references_at_exact_pins():
    got = cloudrun.lean_requirements(LOCK, {"xgboost", "numpy", "sklearn", "collections"})
    # no catboost, no lightgbm; xgboost becomes its CPU-only distribution at the SAME pin (no ~470 MB of CUDA libraries on a GPU-less host)
    assert sorted(got) == ["numpy==2.4.0", "pandas==2.3.3", "scikit-learn==1.8.0", "scipy==1.16.3", "xgboost-cpu==3.1.3"]


def test_lean_includes_each_referenced_model_library():
    got = cloudrun.lean_requirements(LOCK, {"lightgbm", "catboost", "numpy"})
    assert "lightgbm==4.6.0" in got and "catboost==1.2.10" in got and not any(x.startswith("xgboost") for x in got)


def test_lean_refuses_unknown_or_unpinned_dependencies():
    with pytest.raises(RegistryError, match="does not know how to install"):
        cloudrun.lean_requirements(LOCK, {"xgboost", "torch"})
    with pytest.raises(RegistryError, match="does not pin it"):
        cloudrun.lean_requirements(["pandas==2.3.3", "numpy==2.4.0"], {"xgboost"})
    with pytest.raises(RegistryError, match="could not tell which model library"):
        cloudrun.lean_requirements(LOCK, {"numpy"})


def test_local_helper_modules_are_not_treated_as_third_party():
    got = cloudrun.lean_requirements(LOCK, {"xgboost", "model_classes"}, local_modules={"model_classes"})
    assert "xgboost-cpu==3.1.3" in got


def test_pickle_scan_finds_real_libraries(tmp_path):
    import pickle as pk
    from xgboost import XGBClassifier
    X = np.random.default_rng(0).normal(size=(50, 2)); y = (X[:, 0] > 0).astype(int)
    p = tmp_path / "m.pkl"; p.write_bytes(pk.dumps({"model": XGBClassifier(n_estimators=3).fit(X, y), "features": ["a", "b"]}))
    mods = cloudrun.pickle_modules(p)
    assert "xgboost" in mods and "lightgbm" not in mods and "catboost" not in mods


def test_pickle_scan_never_executes_the_pickles_code(tmp_path):
    """A hostile pickle whose REDUCE would run a shell command: scanning must record `os` and run nothing."""
    import os as _os
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (_os.system, (f"touch {marker}",))

    p = tmp_path / "evil.pkl"; p.write_bytes(__import__("pickle").dumps(Evil()))
    assert "posix" in cloudrun.pickle_modules(p) or "os" in cloudrun.pickle_modules(p)
    assert not marker.exists()  # the command never ran


def test_bundle_is_lean_by_default_and_full_on_request(tmp_path):
    import pickle as pk
    from xgboost import XGBClassifier
    X = np.random.default_rng(0).normal(size=(50, 2)); y = (X[:, 0] > 0).astype(int)
    reg = Registry(tmp_path / "r")
    lock = tmp_path / "l.lock"; lock.write_text("\n".join(LOCK) + "\n")
    art = tmp_path / "m.pkl"; art.write_bytes(pk.dumps({"model": XGBClassifier(n_estimators=3).fit(X, y), "features": ["a", "b"], "fill_values": {}}))
    reg.register("p", "m", "v1", art, lockfile=lock)
    lean = (cloudrun.render_bundle(reg, "p", "m", "v1", tmp_path / "a") / "requirements.txt").read_text().split()
    full = (cloudrun.render_bundle(reg, "p", "m", "v1", tmp_path / "b", full_lock=True) / "requirements.txt").read_text().split()
    assert "xgboost-cpu==3.1.3" in lean and "fastapi" in lean and "uvicorn" in lean
    assert not any(x.startswith(("pybaseball", "catboost", "lightgbm", "google-cloud-aiplatform")) for x in lean)
    assert "pybaseball==2.2.7" in full and "catboost==1.2.10" in full


def test_corrupt_pickle_gives_a_clear_error_and_full_lock_still_works(tmp_path):
    reg = Registry(tmp_path / "r")
    lock = tmp_path / "l.lock"; lock.write_text("\n".join(LOCK) + "\n")
    art = tmp_path / "junk.pkl"; art.write_bytes(b"this is not a pickle"); reg.register("p", "m", "v1", art, lockfile=lock)
    with pytest.raises(RegistryError, match="cannot read the model pickle.*--full-lock"):
        cloudrun.render_bundle(reg, "p", "m", "v1", tmp_path / "a")
    assert (cloudrun.render_bundle(reg, "p", "m", "v1", tmp_path / "b", full_lock=True) / "requirements.txt").exists()


def test_scan_survives_the_container_shapes_real_pickles_use(tmp_path):
    """Regression: the real V10 pickle rebuilds containers via item assignment/append/add, which crashed the first scanner."""
    import collections
    import pickle as pk
    payload = {"od": collections.OrderedDict(a=1, b=2), "dd": collections.defaultdict(list, {"x": [1, 2]}), "s": {1, 2, 3},
               "fs": frozenset({4}), "nested": [collections.OrderedDict(z=[1, {2}])], "arr": np.arange(5), "tuple": (1, "a")}
    p = tmp_path / "c.pkl"; p.write_bytes(pk.dumps(payload))
    assert {"collections", "numpy"} <= cloudrun.pickle_modules(p)


@pytest.mark.parametrize("kind, expected_lib", [("xgboost", "xgboost"), ("lightgbm", "lightgbm"), ("sklearn", "sklearn")])
def test_scan_recognises_each_real_model_family(tmp_path, kind, expected_lib):
    import pickle as pk
    X = np.random.default_rng(0).normal(size=(60, 3)); y = (X[:, 0] > 0).astype(int)
    if kind == "xgboost":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=3).fit(X, y)
    elif kind == "lightgbm":
        pytest.importorskip("lightgbm")
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=3, verbose=-1).fit(X, y)
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    p = tmp_path / "m.pkl"; p.write_bytes(pk.dumps({"model": m, "features": ["a", "b", "c"]}))
    mods = cloudrun.pickle_modules(p)
    assert expected_lib in mods
    others = {"xgboost", "lightgbm", "catboost"} - {expected_lib}
    assert not (others & mods)  # a model only pulls in its own library
    assert cloudrun.lean_requirements(LOCK, mods)  # and the lean lock resolves for it


def test_scan_recognises_catboost(tmp_path):
    pytest.importorskip("catboost")
    import pickle as pk
    from catboost import CatBoostClassifier
    X = np.random.default_rng(0).normal(size=(60, 3)); y = (X[:, 0] > 0).astype(int)
    m = CatBoostClassifier(iterations=3, verbose=0, allow_writing_files=False).fit(X, y)
    p = tmp_path / "m.pkl"; p.write_bytes(pk.dumps({"model": m}))
    assert "catboost" in cloudrun.pickle_modules(p)


def test_cpu_variant_keeps_extras_and_unpinned_lines_sane():
    got = cloudrun.lean_requirements(["xgboost>=3.0", "numpy==2.4.0", "pandas==2.3.3", "scikit-learn==1.8.0", "scipy==1.16.3"], {"xgboost"})
    assert "xgboost-cpu>=3.0" in got
