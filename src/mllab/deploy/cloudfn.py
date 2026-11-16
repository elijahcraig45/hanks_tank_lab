"""Cloud Function model swap.

The function loads its model from a fixed GCS object (predict_today_games.py:75), so a swap means
replacing that object. Every swap backs up the current object first, guards against concurrent
writers with if_generation_match, and re-downloads to verify. Rollback restores the backup.
Code redeploys stay with the existing deploy_*.sh scripts (see run_deploy_script).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .. import audit
from ..auth import is_blocked_account
from ..registry import Registry, RegistryError
from .confirm import require

BACKUP_PREFIX = "models/vertex/_mllab_backups"


@dataclass
class SwapPlan:
    project: str
    model: str
    version: str
    bucket: str
    object_name: str
    new_sha256: str
    current_sha256: str | None
    current_generation: int | None
    backup_object: str | None

    @property
    def noop(self) -> bool:
        return self.current_sha256 == self.new_sha256

    @property
    def confirm_phrase(self) -> str:
        return f"deploy {self.project}/{self.model}/{self.version}"

    def diff(self) -> str:
        if self.noop:
            return f"no change: gs://{self.bucket}/{self.object_name} already holds {self.version} ({self.new_sha256[:12]})"
        cur = f"{self.current_sha256[:12]} (gen {self.current_generation})" if self.current_sha256 else "(missing)"
        return (f"gs://{self.bucket}/{self.object_name}\n  current: {cur}\n  new:     {self.new_sha256[:12]} = {self.version}\n"
                f"  backup:  gs://{self.bucket}/{self.backup_object}")


def _sha256_blob(blob) -> str:
    meta = (blob.metadata or {}).get("mllab_sha256")
    if meta:
        return meta
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x"
        blob.download_to_filename(str(p))
        return hashlib.sha256(p.read_bytes()).hexdigest()


def plan_swap(reg: Registry, project: str, model: str, version: str, *, bucket: str, object_name: str, client) -> SwapPlan:
    pkl = reg.verify(project, model, version)  # hash-checked before anything else
    new = hashlib.sha256(pkl.read_bytes()).hexdigest()
    blob = client.bucket(bucket).get_blob(object_name)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if blob is None:
        return SwapPlan(project, model, version, bucket, object_name, new, None, None, None)
    name = object_name.strip("/").replace("/", "__")
    return SwapPlan(project, model, version, bucket, object_name, new, _sha256_blob(blob), blob.generation,
                    f"{BACKUP_PREFIX}/{name}/{ts}-gen{blob.generation}.pkl")


def apply_swap(plan: SwapPlan, reg: Registry, *, client, confirm: str | None, dry_run: bool = False,
               channel: str = "prod", audit_path: Path | None = None) -> dict:
    if plan.noop:
        return {"changed": False, "detail": plan.diff()}
    if dry_run:
        return {"changed": False, "dry_run": True, "detail": plan.diff()}
    require(plan.confirm_phrase, confirm)

    bucket = client.bucket(plan.bucket)
    live = bucket.get_blob(plan.object_name)
    live_gen = live.generation if live else None
    if live_gen != plan.current_generation:
        raise RegistryError(f"{plan.object_name} changed since the plan (gen {plan.current_generation} -> {live_gen}); re-plan")
    if live is not None and plan.backup_object:
        bucket.copy_blob(live, bucket, new_name=plan.backup_object)

    pkl = reg.verify(plan.project, plan.model, plan.version)
    blob = bucket.blob(plan.object_name)
    blob.metadata = {"mllab_sha256": plan.new_sha256, "mllab_version": plan.version}
    blob.upload_from_filename(str(pkl), if_generation_match=live_gen or 0)

    check = bucket.get_blob(plan.object_name)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x"
        check.download_to_filename(str(p))
        if hashlib.sha256(p.read_bytes()).hexdigest() != plan.new_sha256:
            raise RegistryError("post-upload verification failed; restore with rollback_swap")

    previous = reg.promote(plan.project, plan.model, plan.version, channel)
    entry = {**asdict(plan), "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "target": "cloudfn-model-swap", "previous_channel_version": previous}
    reg.record_deploy(plan.project, plan.model, entry)
    audit.record("cloudfn_swap", path=audit_path, project=plan.project, model=plan.model, version=plan.version,
                 object=plan.object_name, backup=plan.backup_object, previous=previous)
    return {"changed": True, "backup": plan.backup_object, "previous_version": previous}


def pending_swaps(deployments: list[dict], object_name: str) -> list[dict]:
    """Swaps of this object that have not been rolled back yet, oldest first. A rollback undoes the most recent swap,
    so this is a stack: swap A, swap B, rollback -> [A]; rollback again -> []."""
    stack: list[dict] = []
    for d in deployments:
        if d.get("object_name") != object_name:
            continue
        if d.get("target") == "cloudfn-model-swap" and d.get("backup_object"):
            stack.append(d)
        elif d.get("target") == "cloudfn-rollback" and stack:
            stack.pop()
    return stack


def rollback_swap(reg: Registry, project: str, model: str, *, bucket: str, object_name: str, client,
                  confirm: str | None, dry_run: bool = False, channel: str = "prod", audit_path: Path | None = None) -> dict:
    swaps = pending_swaps(reg.deployments(project, model), object_name)
    if not swaps:
        raise RegistryError("no backed-up swap to roll back")
    last = swaps[-1]
    phrase = f"rollback {project}/{model}"
    if dry_run:
        return {"changed": False, "dry_run": True, "detail": f"would restore gs://{bucket}/{last['backup_object']} -> {object_name}"}
    require(phrase, confirm)
    b = client.bucket(bucket)
    src, live = b.get_blob(last["backup_object"]), b.get_blob(object_name)
    if src is None:
        raise RegistryError(f"backup object missing: {last['backup_object']}")
    b.copy_blob(src, b, new_name=object_name, if_generation_match=live.generation if live else 0)
    prev = last.get("previous_channel_version")
    if prev:
        reg.promote(project, model, prev, channel)
    else:  # the swap created this channel's pointer; undoing it must not leave it pointing at the rolled-back version
        reg.clear_channel(project, model, channel)
    reg.record_deploy(project, model, {"target": "cloudfn-rollback", "object_name": object_name, "bucket": bucket,
                                       "restored_from": last["backup_object"], "undid_version": last.get("version"),
                                       "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    audit.record("cloudfn_rollback", path=audit_path, project=project, model=model, object=object_name,
                 restored_from=last["backup_object"], channel_now=prev)
    return {"changed": True, "restored_from": last["backup_object"], "channel": prev}


def run_deploy_script(script: str | os.PathLike, flags: list[str], *, account: str = "you@example.com",
                      dry_run: bool = True, runner=subprocess.run) -> list[str]:
    """Code redeploy stays with the existing deploy_*.sh (tracked-files-only staging, scheduler recreation).

    Defaults to --dry-run and always passes --skip-model-upload, so the script can never overwrite the
    model object that apply_swap manages.
    """
    if is_blocked_account(account):
        raise RegistryError("refusing work account")
    cmd = ["bash", str(script), "--skip-model-upload", *flags, *(["--dry-run"] if dry_run else [])]
    runner(cmd, check=True, env={**os.environ, "CLOUDSDK_CORE_ACCOUNT": account})
    return cmd
