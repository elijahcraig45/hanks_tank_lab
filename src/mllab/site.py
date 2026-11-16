"""What the live site says its models are: the public Models API already lists every model with its key and role, so the control tool can
validate targets and recognise production keys from the real registry instead of a copy that could drift. Read-only GETs."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://hankstank.com"
CACHE_TTL_S = float(os.environ.get("MLLAB_SITE_TTL", "3600"))


class SiteError(Exception):
    pass


def _fetch(url: str, timeout: float = 20.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "mllab-control/1.0", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https base, read-only
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, TimeoutError) as e:
        raise SiteError(f"could not read {url}: {e}") from e


def registry(sport: str, *, base: str = BASE, fetch=_fetch) -> list[dict]:
    """[{key, label, role, available}] for a sport, as the site currently lists them."""
    if sport not in ("mlb", "nfl", "cfb"):
        raise SiteError(f"unknown sport {sport!r}")
    d = fetch(f"{base}/api/models/{sport}/compare")
    models = ((d or {}).get("data") or {}).get("models")
    if not isinstance(models, list):
        raise SiteError("the models API returned an unexpected shape")
    return [{"key": m["key"], "label": m.get("label", m["key"]), "role": m.get("role"), "available": bool(m.get("available", True))} for m in models if "key" in m]


def _cache_file(sport: str, cache_dir=None) -> Path:
    return Path(cache_dir or os.environ.get("MLLAB_CACHE", "/mnt/appdata/mllab/cache")) / "site" / f"registry-{sport}.json"


def cached_registry(sport: str, *, ttl: float | None = None, cache_dir=None, now=time.time, **kw) -> list[dict]:
    """registry(), reusing a copy fetched within `ttl` seconds (MLLAB_SITE_TTL, default an hour).

    For the health timer. It runs every 10 minutes, and each live fetch wakes the site's App Engine instance, which then stays billed for
    ~15 minutes: a 10-minute poll kept it up around the clock. Anything that validates a change before writing it (the control CLI) must keep
    calling registry() for a fresh read. A failed fetch still raises SiteError, so an outage is seen at the next refresh; the cache never
    answers for a failed read. An unwritable cache directory only costs the saving."""
    ttl = CACHE_TTL_S if ttl is None else ttl
    base = kw.get("base", BASE)
    path = _cache_file(sport, cache_dir)
    try:
        c = json.loads(path.read_text())
        if c["base"] == base and isinstance(c["models"], list) and 0 <= now() - float(c["fetched_at"]) < ttl:
            return c["models"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    models = registry(sport, **kw)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"base": base, "fetched_at": now(), "models": models}))
        os.replace(tmp, path)
    except OSError:
        pass
    return models


def known_models(sport: str, **kw) -> list[str]:
    return [m["key"] for m in registry(sport, **kw)]


def production_keys(sport: str, **kw) -> set[str]:
    return {m["key"] for m in registry(sport, **kw) if m["role"] == "production"}
