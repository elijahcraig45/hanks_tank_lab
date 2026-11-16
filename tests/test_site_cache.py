"""cached_registry: the health timer reuses a recent copy of the site's model list so it stops waking the site every 10 minutes."""
import pytest

from mllab import site as S

MODELS = [{"key": "v10", "label": "V10", "role": "production", "available": True}]


class Fetch:
    def __init__(self, models=MODELS, fail=False):
        self.calls, self.models, self.fail = [], models, fail

    def __call__(self, url, *a, **k):
        self.calls.append(url)
        if self.fail:
            raise S.SiteError("down")
        return {"data": {"models": self.models}}


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def get(fetch, clock, tmp_path, **kw):
    return S.cached_registry("mlb", ttl=kw.pop("ttl", 3600), cache_dir=tmp_path, now=clock, fetch=fetch, **kw)


def test_second_read_within_ttl_does_not_fetch(tmp_path):
    f, c = Fetch(), Clock()
    first = get(f, c, tmp_path)
    c.t += 3599
    assert get(f, c, tmp_path) == first == MODELS
    assert len(f.calls) == 1


def test_expired_copy_is_refetched_and_replaced(tmp_path):
    f, c = Fetch(), Clock()
    get(f, c, tmp_path)
    c.t += 3600
    f.models = [{"key": "v11", "label": "V11", "role": "production", "available": True}]
    assert [m["key"] for m in get(f, c, tmp_path)] == ["v11"]
    assert len(f.calls) == 2
    c.t += 10
    assert [m["key"] for m in get(f, c, tmp_path)] == ["v11"] and len(f.calls) == 2


def test_failed_fetch_raises_and_leaves_no_cache(tmp_path):
    f, c = Fetch(fail=True), Clock()
    with pytest.raises(S.SiteError):
        get(f, c, tmp_path)
    assert not S._cache_file("mlb", tmp_path).exists()


def test_failed_refresh_after_expiry_raises_not_stale(tmp_path):
    f, c = Fetch(), Clock()
    get(f, c, tmp_path)
    c.t += 7200
    f.fail = True
    with pytest.raises(S.SiteError):
        get(f, c, tmp_path)


def test_unwritable_cache_dir_still_returns_a_fresh_read(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    f, c = Fetch(), Clock()
    assert get(f, c, blocker / "sub") == MODELS and len(f.calls) == 1


def test_corrupt_cache_is_ignored(tmp_path):
    f, c = Fetch(), Clock()
    p = S._cache_file("mlb", tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text("{not json")
    assert get(f, c, tmp_path) == MODELS and len(f.calls) == 1


def test_different_base_url_does_not_reuse_the_copy(tmp_path):
    f, c = Fetch(), Clock()
    get(f, c, tmp_path, base="https://a.example")
    get(f, c, tmp_path, base="https://b.example")
    assert f.calls == ["https://a.example/api/models/mlb/compare", "https://b.example/api/models/mlb/compare"]


def test_each_sport_has_its_own_copy(tmp_path):
    f, c = Fetch(), Clock()
    for sp in ("mlb", "nfl", "cfb"):
        S.cached_registry(sp, ttl=3600, cache_dir=tmp_path, now=c, fetch=f)
    assert len(f.calls) == 3


def test_clock_going_backwards_refetches(tmp_path):
    f, c = Fetch(), Clock()
    get(f, c, tmp_path)
    c.t -= 500
    get(f, c, tmp_path)
    assert len(f.calls) == 2


def test_uncached_registry_always_fetches(tmp_path):
    f = Fetch()
    S.registry("mlb", fetch=f)
    S.registry("mlb", fetch=f)
    assert len(f.calls) == 2


def test_health_snapshot_uses_the_cached_registry_and_control_uses_the_live_one():
    import inspect

    from mllab import cli

    health = inspect.getsource(cli._health_snapshot)
    assert "site.cached_registry" in health and "registry_fn=site.registry" not in health
    assert "known_models=site.known_models" in inspect.getsource(cli)
