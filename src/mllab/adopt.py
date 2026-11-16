"""Import an existing GCS model artifact (e.g. the deployed V10) into the registry."""
from __future__ import annotations

import base64
import hashlib
import tempfile
from pathlib import Path

from .registry import Meta, Registry, RegistryError


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://") or uri.count("/") < 3:
        raise RegistryError(f"expected gs://bucket/object, got {uri!r}")
    bucket, _, name = uri[5:].partition("/")
    if not bucket or not name or name.endswith("/"):
        raise RegistryError(f"expected gs://bucket/object, got {uri!r}")
    return bucket, name


def _md5_b64(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - GCS reports md5, used only to check the transfer
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def adopt(reg: Registry, project: str, model: str, version: str, gcs_uri: str, *, client=None, **register_kw) -> Meta:
    """Download gcs_uri, check it against GCS's md5, and register it as an immutable version."""
    bucket_name, name = parse_gcs_uri(gcs_uri)
    if client is None:
        from google.cloud import storage  # lazy: keeps this module importable without GCP libs

        client = storage.Client(project=project)
    blob = client.bucket(bucket_name).get_blob(name)
    if blob is None:
        raise RegistryError(f"{gcs_uri} not found")
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "model.pkl"
        blob.download_to_filename(str(local))
        if blob.md5_hash and _md5_b64(local) != blob.md5_hash:
            raise RegistryError(f"md5 mismatch downloading {gcs_uri}; refusing to register")
        return reg.register(project, model, version, local, source=f"{gcs_uri}#{blob.generation}", **register_kw)
