import os
import stat

import pytest

from mllab.registry import Registry, RegistryError, sha256_file


@pytest.fixture
def reg(tmp_path):
    return Registry(tmp_path / "registry")


@pytest.fixture
def artifact(tmp_path):
    p = tmp_path / "model.pkl"
    p.write_bytes(b"pretend-pickle-v1")
    return p


def test_register_writes_meta_and_hash(reg, artifact):
    meta = reg.register("hankstank", "game_outcome", "v10", artifact, features=["a", "b"], metrics={"auc": 0.6})
    assert meta.sha256 == sha256_file(artifact)
    assert reg.meta("hankstank", "game_outcome", "v10").features == ["a", "b"]
    assert reg.verify("hankstank", "game_outcome", "v10").read_bytes() == b"pretend-pickle-v1"


def test_versions_are_immutable(reg, artifact):
    reg.register("p", "m", "v1", artifact)
    with pytest.raises(RegistryError, match="immutable"):
        reg.register("p", "m", "v1", artifact)
    pkl = reg.version_dir("p", "m", "v1") / "model.pkl"
    assert not os.access(pkl, os.W_OK) or os.geteuid() == 0
    assert not pkl.stat().st_mode & stat.S_IWUSR


def test_verify_detects_tampering(reg, artifact):
    reg.register("p", "m", "v1", artifact)
    vdir = reg.version_dir("p", "m", "v1")
    vdir.chmod(0o755)
    pkl = vdir / "model.pkl"
    pkl.chmod(0o644)
    pkl.write_bytes(b"tampered")
    with pytest.raises(RegistryError, match="hash mismatch"):
        reg.verify("p", "m", "v1")


def test_parent_must_exist(reg, artifact):
    with pytest.raises(RegistryError, match="parent"):
        reg.register("p", "m", "v2", artifact, parent="v1")
    reg.register("p", "m", "v1", artifact)
    assert reg.register("p", "m", "v2", artifact, parent="v1").parent == "v1"


@pytest.mark.parametrize("bad", ["../evil", "a/b", "", ".hidden", "sp ace"])
def test_rejects_bad_names(reg, artifact, bad):
    with pytest.raises(RegistryError, match="invalid"):
        reg.register("p", "m", bad, artifact)


def test_promote_and_rollback_only_move_pointer(reg, artifact):
    reg.register("p", "m", "v1", artifact)
    reg.register("p", "m", "v2", artifact, parent="v1")
    assert reg.promote("p", "m", "v1") is None
    assert reg.promote("p", "m", "v2") == "v1"
    assert reg.channels("p", "m") == {"prod": "v2"}
    assert reg.promote("p", "m", "v1") == "v2"  # rollback
    assert reg.versions("p", "m") == ["v1", "v2"]


def test_promote_unknown_version_fails(reg):
    with pytest.raises(RegistryError, match="unknown"):
        reg.promote("p", "m", "nope")


def test_failed_register_leaves_no_partial_dir(reg, artifact, tmp_path):
    with pytest.raises(FileNotFoundError):
        reg.register("p", "m", "v1", artifact, lockfile=tmp_path / "missing.lock")
    assert reg.versions("p", "m") == []
    assert not list((reg.root / "p" / "m").glob(".*tmp"))
