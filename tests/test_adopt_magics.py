import base64
import hashlib
from types import SimpleNamespace

import pytest

from mllab.adopt import adopt, parse_gcs_uri
from mllab.magics import _parse
from mllab.registry import Registry, RegistryError

DATA = b"deployed-v10-bytes"
MD5 = base64.b64encode(hashlib.md5(DATA).digest()).decode()


class FakeBlob:
    def __init__(self, data=DATA, md5=MD5):
        self.data, self.md5_hash, self.generation = data, md5, 1712345
    def download_to_filename(self, path):
        open(path, "wb").write(self.data)


class FakeClient:
    def __init__(self, blob):
        self._blob = blob
    def bucket(self, name):
        return SimpleNamespace(get_blob=lambda n: self._blob)


def test_adopt_registers_and_records_source(tmp_path):
    reg = Registry(tmp_path / "r")
    m = adopt(reg, "hankstank", "game_outcome", "v10", "gs://hanks_tank_data/models/vertex/game_outcome_2026_v10/model.pkl",
              client=FakeClient(FakeBlob()))
    assert m.sha256 == hashlib.sha256(DATA).hexdigest()
    assert m.source.endswith("#1712345")
    assert reg.verify("hankstank", "game_outcome", "v10").read_bytes() == DATA


def test_adopt_rejects_corrupt_download(tmp_path):
    reg = Registry(tmp_path / "r")
    with pytest.raises(RegistryError, match="md5"):
        adopt(reg, "p", "m", "v1", "gs://b/o.pkl", client=FakeClient(FakeBlob(md5="AAAA")))
    assert reg.versions("p", "m") == []


def test_adopt_missing_object(tmp_path):
    with pytest.raises(RegistryError, match="not found"):
        adopt(Registry(tmp_path), "p", "m", "v1", "gs://b/o.pkl", client=FakeClient(None))


@pytest.mark.parametrize("bad", ["s3://b/o", "gs://b", "gs://b/", "gs:///o", "b/o"])
def test_bad_uris(bad):
    with pytest.raises(RegistryError):
        parse_gcs_uri(bad)


def test_magic_option_parsing():
    o = _parse("--project other --max-gb 2.5 --force games")
    assert (o["project"], o["max_gb"], o["force"], o["var"]) == ("other", 2.5, True, "games")
    with pytest.raises(ValueError):
        _parse("--bogus")
