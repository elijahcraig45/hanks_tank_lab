from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")
pytest.importorskip("google.cloud.bigquery")

from mllab import auth
from mllab.bq import QueryTooExpensive, ResultTooLarge, bq, cache_key


class FakeClient:
    def __init__(self, est, result_bytes=100, rows=2, cols=1):
        self.est, self.result_bytes, self.rows, self.cols = est, result_bytes, rows, cols
        self.runs = 0
        self.downloaded = 0

    def query(self, sql, job_config=None):
        if job_config.dry_run:
            return SimpleNamespace(total_bytes_processed=self.est)
        self.runs += 1

        def to_df(**kw):
            self.kw = kw
            self.downloaded += 1
            return pd.DataFrame({"x": [1, 2]})

        return SimpleNamespace(result=lambda: SimpleNamespace(total_rows=self.rows, schema=[0] * self.cols),
                               destination="tmp.table", to_dataframe=to_df)

    def get_table(self, dest):
        return SimpleNamespace(num_bytes=self.result_bytes)


def test_over_cap_is_refused(tmp_path):
    with pytest.raises(QueryTooExpensive):
        bq("select 1", max_bytes=100, cache_dir=tmp_path, client=FakeClient(500))


def test_force_overrides_cap(tmp_path):
    c = FakeClient(500)
    assert len(bq("select 1", max_bytes=100, force=True, cache_dir=tmp_path, client=c)) == 2
    assert c.runs == 1


def test_huge_result_is_refused_before_download(tmp_path):
    """Cheap to scan, too big for RAM: the statcast SELECT * case (5 GiB scanned, 7.8M rows x 121 cols)."""
    c = FakeClient(est=10, result_bytes=5 * 1024**3)
    with pytest.raises(ResultTooLarge, match="nothing was downloaded"):
        bq("select * from big", max_bytes=100, cache_dir=tmp_path, client=c)
    assert c.downloaded == 0  # never pulled into memory
    assert not list((tmp_path / "bq").glob("*.parquet"))  # and nothing half-cached


def test_result_cap_configurable_and_force_overrides(tmp_path):
    c = FakeClient(est=10, result_bytes=2 * 1024**3)
    assert len(bq("q1", max_bytes=100, max_result_bytes=3 * 1024**3, cache_dir=tmp_path, client=c)) == 2
    with pytest.raises(ResultTooLarge):
        bq("q2", max_bytes=100, cache_dir=tmp_path, client=c)
    assert len(bq("q2", max_bytes=100, force=True, cache_dir=tmp_path, client=c)) == 2


def test_row_count_floor_catches_zero_reported_size(tmp_path):
    """BigQuery can report numBytes=0 for a result; 500M rows x 10 cols must still be refused."""
    c = FakeClient(est=10, result_bytes=0, rows=500_000_000, cols=10)
    with pytest.raises(ResultTooLarge):
        bq("select * from huge", max_bytes=100, cache_dir=tmp_path, client=c)
    assert c.downloaded == 0


def test_result_cap_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MLLAB_BQ_MAX_RESULT_BYTES", "50")
    with pytest.raises(ResultTooLarge):
        bq("q3", max_bytes=100, cache_dir=tmp_path, client=FakeClient(est=10, result_bytes=100))


def test_rest_download_by_default_storage_api_opt_in(tmp_path, monkeypatch):
    c = FakeClient(10)
    bq("a1", max_bytes=100, cache_dir=tmp_path, client=c)
    assert c.kw == {"create_bqstorage_client": False}  # works without readsessions.create
    monkeypatch.setenv("MLLAB_BQ_STORAGE_API", "1")
    bq("a2", max_bytes=100, cache_dir=tmp_path, client=c)
    assert c.kw == {"create_bqstorage_client": True}


def test_second_call_hits_cache(tmp_path):
    c = FakeClient(10)
    bq("select   1", max_bytes=100, cache_dir=tmp_path, client=c)
    bq("select 1", max_bytes=100, cache_dir=tmp_path, client=c)  # whitespace-insensitive key
    assert c.runs == 1


def test_cache_key_differs_by_project():
    assert cache_key("select 1", "a") != cache_key("select 1", "b")


def test_auth_needs_both_factors():
    kw = dict(allowed_login="henry@example.com", expected_secret="s3cret")
    assert auth.check("Henry@Example.com", "s3cret", **kw)
    assert not auth.check("henry@example.com", None, **kw)
    assert not auth.check(None, "s3cret", **kw)
    assert not auth.check("other@example.com", "s3cret", **kw)
    assert not auth.check("henry@example.com", "wrong", **kw)


def test_auth_fails_closed_when_unconfigured():
    assert not auth.check("a", "b", allowed_login="", expected_secret="")


def test_work_account_blocked():
    with pytest.raises(RuntimeError):
        auth.assert_not_work_account("someone@blocked.example")
    auth.assert_not_work_account("you@example.com")


def test_blocked_domains_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("MLLAB_BLOCKED_ACCOUNT_DOMAINS", " One.example , @two.example ,")
    assert auth.blocked_domains() == ("one.example", "two.example")
    assert auth.is_blocked_account("x@TWO.example") and not auth.is_blocked_account("x@three.example")
    monkeypatch.delenv("MLLAB_BLOCKED_ACCOUNT_DOMAINS")
    assert auth.blocked_domains() == () and not auth.is_blocked_account("someone@blocked.example")
    assert not auth.is_blocked_account(None) and not auth.is_blocked_account("no-at-sign")
