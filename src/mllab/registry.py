"""Immutable, hash-verified model registry.

Layout: <root>/<project>/<model>/<version>/{model.pkl,meta.json,...}
Channels (prod, staging, ...) are pointers in <root>/<project>/<model>/channels.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RESERVED_VERSIONS = {"cards"}  # sibling folders inside a model dir that are not versions


class RegistryError(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_name(kind: str, value: str) -> str:
    if not _NAME.match(value) or ".." in value:
        raise RegistryError(f"invalid {kind} name: {value!r}")
    return value


def git_state(path: Path) -> tuple[str | None, bool]:
    """Return (sha, dirty) for the repo containing path, or (None, False)."""
    cwd = path if path.is_dir() else path.parent
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


@dataclass
class Meta:
    project: str
    model: str
    version: str
    sha256: str
    created_at: str
    parent: str | None = None
    git_sha: str | None = None
    git_dirty: bool = False
    data_hash: str | None = None
    features: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    lock_hash: str | None = None
    code_snapshot: list[str] = field(default_factory=list)
    deployed_to: list[dict] = field(default_factory=list)
    source: str | None = None


class Registry:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)

    # -- paths ---------------------------------------------------------
    def _model_dir(self, project: str, model: str) -> Path:
        return self.root / _check_name("project", project) / _check_name("model", model)

    def version_dir(self, project: str, model: str, version: str) -> Path:
        return self._model_dir(project, model) / _check_name("version", version)

    # -- register ------------------------------------------------------
    def register(
        self,
        project: str,
        model: str,
        version: str,
        artifact: str | os.PathLike,
        *,
        parent: str | None = None,
        features: list[str] | None = None,
        metrics: dict | None = None,
        data_hash: str | None = None,
        lockfile: str | os.PathLike | None = None,
        train_script: str | os.PathLike | None = None,
        code_snapshot: list[str | os.PathLike] | None = None,
        source: str | None = None,
    ) -> Meta:
        artifact = Path(artifact)
        if not artifact.is_file():
            raise RegistryError(f"artifact not found: {artifact}")
        if version in RESERVED_VERSIONS:
            raise RegistryError(f"{version!r} is reserved and cannot be used as a version name")
        vdir = self.version_dir(project, model, version)
        if vdir.exists():
            raise RegistryError(f"{project}/{model}/{version} already exists (versions are immutable)")
        if parent is not None and not self.version_dir(project, model, parent).exists():
            raise RegistryError(f"parent version {parent!r} does not exist")

        digest = sha256_file(artifact)
        sha, dirty = git_state(Path(train_script) if train_script else artifact)
        tmp = vdir.with_name(f".{version}.tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            shutil.copy2(artifact, tmp / "model.pkl")
            snaps: list[str] = []
            if lockfile:
                shutil.copy2(lockfile, tmp / "requirements.lock")
            if train_script:
                shutil.copy2(train_script, tmp / "train.py")
            for extra in code_snapshot or []:
                extra = Path(extra)
                dest = tmp / "code" / extra.name
                dest.parent.mkdir(exist_ok=True)
                shutil.copy2(extra, dest)
                snaps.append(extra.name)
            (tmp / "features.json").write_text(json.dumps(features or [], indent=2))
            meta = Meta(
                project=project,
                model=model,
                version=version,
                sha256=digest,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                parent=parent,
                git_sha=sha,
                git_dirty=dirty,
                data_hash=data_hash,
                features=features or [],
                metrics=metrics or {},
                lock_hash=sha256_file(tmp / "requirements.lock") if lockfile else None,
                code_snapshot=snaps,
                source=source,
            )
            (tmp / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
            tmp.rename(vdir)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        self._lock(vdir)
        return meta

    @staticmethod
    def _lock(vdir: Path) -> None:
        for p in sorted(vdir.rglob("*"), reverse=True):
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        vdir.chmod(vdir.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    # -- read ----------------------------------------------------------
    def meta(self, project: str, model: str, version: str) -> Meta:
        path = self.version_dir(project, model, version) / "meta.json"
        if not path.is_file():
            raise RegistryError(f"unknown version {project}/{model}/{version}")
        return Meta(**json.loads(path.read_text()))

    def verify(self, project: str, model: str, version: str) -> Path:
        """Check the pickle hash and return its path. Load pickles only via this."""
        meta = self.meta(project, model, version)
        pkl = self.version_dir(project, model, version) / "model.pkl"
        if sha256_file(pkl) != meta.sha256:
            raise RegistryError(f"hash mismatch for {project}/{model}/{version}: artifact was modified")
        return pkl

    def versions(self, project: str, model: str) -> list[str]:
        d = self._model_dir(project, model)
        if not d.is_dir():
            return []
        found = [p for p in d.iterdir() if p.is_dir() and not p.name.startswith(".") and (p / "meta.json").exists()]
        found.sort(key=lambda p: (p / "meta.json").stat().st_mtime if (p / "meta.json").exists() else 0)
        return [p.name for p in found]

    # -- channels ------------------------------------------------------
    def _channels_path(self, project: str, model: str) -> Path:
        return self._model_dir(project, model) / "channels.json"

    def channels(self, project: str, model: str) -> dict[str, str]:
        p = self._channels_path(project, model)
        return json.loads(p.read_text()) if p.is_file() else {}

    def promote(self, project: str, model: str, version: str, channel: str = "prod") -> str | None:
        """Point channel at version. Returns the previous version (for rollback)."""
        _check_name("channel", channel)
        self.verify(project, model, version)
        ch = self.channels(project, model)
        previous = ch.get(channel)
        ch[channel] = version
        path = self._channels_path(project, model)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ch, indent=2))
        os.replace(tmp, path)
        return previous

    def clear_channel(self, project: str, model: str, channel: str) -> str | None:
        """Remove a channel pointer (used when a rollback undoes the channel's first-ever promotion)."""
        _check_name("channel", channel)
        ch = self.channels(project, model)
        previous = ch.pop(channel, None)
        path = self._channels_path(project, model)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ch, indent=2))
        os.replace(tmp, path)
        return previous

    # -- deployments ---------------------------------------------------
    def record_deploy(self, project: str, model: str, entry: dict) -> None:
        """Append to <model>/deployments.jsonl. Kept outside version dirs, which are immutable."""
        path = self._model_dir(project, model) / "deployments.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
        try:
            os.write(fd, (json.dumps(entry, sort_keys=True) + "\n").encode())
        finally:
            os.close(fd)

    def deployments(self, project: str, model: str) -> list[dict]:
        path = self._model_dir(project, model) / "deployments.jsonl"
        return [json.loads(l) for l in path.read_text().splitlines()] if path.is_file() else []
