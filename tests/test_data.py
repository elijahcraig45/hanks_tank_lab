import sys

import pytest

from mllab import data as D
from mllab.jobs import DirectLauncher, JobSpec, LocalRunner


@pytest.fixture
def lab(tmp_path):
    r = LocalRunner(tmp_path / "lab", launcher=DirectLauncher(), python_bin=sys.executable)
    code = tmp_path / "code"; code.mkdir()
    (code / "s.py").write_text("import pathlib\npathlib.Path('outputs').mkdir(exist_ok=True)\npathlib.Path('outputs/x.parquet').write_bytes(b'PAR1data')\n"
                               "pathlib.Path('outputs/sub').mkdir(exist_ok=True)\npathlib.Path('outputs/sub/y.csv').write_text('a\\n1\\n')\n")
    jid = r.submit(JobSpec(name="j", script="s.py", code_dir=str(code), timeout_s=30))
    r.launcher.wait(jid)
    return r, jid, tmp_path


def test_promote_copies_and_lists(lab):
    r, jid, _ = lab
    dest = D.promote(r, jid, "x.parquet", "nfl_features.parquet")
    assert dest.read_bytes() == b"PAR1data"
    assert D.promote(r, jid, "sub/y.csv", "y.csv").read_text() == "a\n1\n"
    assert [n for n, _ in D.list_files(r.data_root)] == ["nfl_features.parquet", "y.csv"]


def test_no_overwrite_without_replace(lab):
    r, jid, _ = lab
    D.promote(r, jid, "x.parquet", "a.parquet")
    with pytest.raises(D.DataError, match="already exists"):
        D.promote(r, jid, "x.parquet", "a.parquet")
    D.promote(r, jid, "x.parquet", "a.parquet", replace=True)


@pytest.mark.parametrize("bad", ["../../etc/passwd", "/etc/passwd", "sub/../../x.parquet", "nope.parquet", "sub"])
def test_only_real_files_inside_outputs(lab, bad):
    r, jid, _ = lab
    with pytest.raises(D.DataError, match="not a file inside"):
        D.promote(r, jid, bad, "z.parquet")


@pytest.mark.parametrize("name", ["../x.parquet", "x.pkl", "x", "a/b.parquet", ".hidden.parquet", "x y.csv"])
def test_names_are_validated(lab, name):
    r, jid, _ = lab
    with pytest.raises(D.DataError, match="name must look like"):
        D.promote(r, jid, "x.parquet", name)


def test_symlink_escape_is_refused(lab, tmp_path):
    r, jid, _ = lab
    secret = tmp_path / "secret.parquet"; secret.write_bytes(b"secret")
    (r.jobs / jid / "code" / "outputs" / "link.parquet").symlink_to(secret)
    with pytest.raises(D.DataError, match="not a file inside"):
        D.promote(r, jid, "link.parquet", "stolen.parquet")


def test_unfinished_job_cannot_be_promoted(lab, tmp_path):
    r, _, _ = lab
    code = tmp_path / "slowcode"; code.mkdir(); (code / "s.py").write_text("import time; time.sleep(20)")
    jid = r.submit(JobSpec(name="s", script="s.py", code_dir=str(code), timeout_s=60))
    with pytest.raises(D.DataError, match="only a finished job"):
        D.promote(r, jid, "x.parquet", "a.parquet")
    r.launcher.stop(jid)
