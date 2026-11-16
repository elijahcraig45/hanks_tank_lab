import os
import subprocess
import time
from pathlib import Path

import pytest

from mllab.registry import Registry

SCRIPT = Path(__file__).resolve().parents[1] / "host" / "backup.sh"


def backup(root, dest, keep=14):
    env = {**os.environ, "MLLAB_ROOT": str(root), "MLLAB_BACKUP_DEST": str(dest), "MLLAB_BACKUP_KEEP": str(keep)}
    r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def snaps(dest):
    return sorted(p.name for p in Path(dest).iterdir() if p.name.startswith("20") and p.is_dir())


@pytest.fixture
def lab(tmp_path):
    root = tmp_path / "lab"
    reg = Registry(root / "registry")
    for v in ("v1", "v2"):
        a = tmp_path / f"{v}.pkl"; a.write_bytes(v.encode() * 1000)
        reg.register("p", "m", v, a, features=["a"])
    reg.promote("p", "m", "v2")
    (root / "audit").mkdir(); (root / "audit" / "deploys.log").write_text('{"a": 1}\n')
    return root, reg, tmp_path / "backups"


def test_same_day_reruns_make_separate_snapshots_never_nested(lab):
    """Regression: a second run on the same date moved the new snapshot INSIDE the first and left the top-level one stale."""
    root, reg, dest = lab
    backup(root, dest); time.sleep(1.1); backup(root, dest)
    s = snaps(dest)
    assert len(s) == 2 and s[0] != s[1]
    assert not list(Path(dest).rglob("*.partial")) and not list(Path(dest).rglob("*.tmp"))  # nothing nested, nothing half-built left behind


def test_a_new_version_appears_in_the_next_snapshot_not_a_stale_one(lab, tmp_path):
    root, reg, dest = lab
    backup(root, dest)
    a = tmp_path / "v3.pkl"; a.write_bytes(b"three" * 500); reg.register("p", "m", "v3", a)
    time.sleep(1.1); backup(root, dest)
    newest = Path(dest) / snaps(dest)[-1]
    assert (newest / "registry" / "p" / "m" / "v3" / "model.pkl").exists()


def test_a_restored_snapshot_is_a_working_registry_with_intact_hashes(lab):
    root, reg, dest = lab
    backup(root, dest)
    restored = Registry(Path(dest) / snaps(dest)[-1] / "registry")
    assert restored.versions("p", "m") == ["v1", "v2"] and restored.channels("p", "m") == {"prod": "v2"}
    for v in ("v1", "v2"):
        assert restored.verify("p", "m", v).read_bytes() == reg.verify("p", "m", v).read_bytes()  # verify() re-hashes against meta.json


def test_a_corrupted_backup_is_detected_on_restore(lab):
    root, reg, dest = lab
    backup(root, dest)
    snap = Path(dest) / snaps(dest)[-1] / "registry" / "p" / "m" / "v1"
    snap.chmod(0o755); (snap / "model.pkl").chmod(0o644); (snap / "model.pkl").write_bytes(b"bitrot")
    from mllab.registry import RegistryError
    with pytest.raises(RegistryError, match="hash mismatch"):
        Registry(Path(dest) / snaps(dest)[-1] / "registry").verify("p", "m", "v1")


def test_unchanged_files_are_hardlinked_so_snapshots_are_cheap(lab):
    root, reg, dest = lab
    backup(root, dest); time.sleep(1.1); backup(root, dest)
    a, b = snaps(dest)
    fa = Path(dest) / a / "registry" / "p" / "m" / "v1" / "model.pkl"; fb = Path(dest) / b / "registry" / "p" / "m" / "v1" / "model.pkl"
    assert os.stat(fa).st_ino == os.stat(fb).st_ino  # same inode: no second copy on disk


def test_retention_keeps_only_the_newest_n(lab):
    root, reg, dest = lab
    for _ in range(4):
        backup(root, dest, keep=2); time.sleep(1.1)
    s = snaps(dest)
    assert len(s) == 2 and s == sorted(s)


def test_nothing_to_back_up_is_an_error_not_an_empty_snapshot(tmp_path):
    env = {**os.environ, "MLLAB_ROOT": str(tmp_path / "empty"), "MLLAB_BACKUP_DEST": str(tmp_path / "b")}
    (tmp_path / "empty").mkdir()
    r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "nothing to back up" in r.stderr
    assert not [p for p in (tmp_path / "b").iterdir() if p.name.startswith("20")]


def test_pruning_never_loosens_the_live_registrys_file_permissions(lab):
    """Snapshots share inodes with the live registry via hardlinks; pruning may relax directories but must never chmod a file."""
    root, reg, dest = lab
    live = root / "registry" / "p" / "m" / "v1" / "model.pkl"
    mode_before = os.stat(live).st_mode & 0o777
    for _ in range(3):
        backup(root, dest, keep=1); time.sleep(1.1)
    assert os.stat(live).st_mode & 0o777 == mode_before and not os.access(live, os.W_OK)  # still read-only after pruning
    assert live.read_bytes() == b"v1" * 1000
