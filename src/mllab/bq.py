"""BigQuery access with a cost guard and a parquet cache.

Every query is dry-run first; anything estimated over the cap is refused unless force=True.
Results are cached as parquet keyed by the SQL hash and read back through DuckDB.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

DEFAULT_MAX_BYTES = 10 * 1024**3  # 10 GiB scanned, about $0.06 at on-demand pricing
DEFAULT_MAX_RESULT_BYTES = 1 * 1024**3  # 1 GiB returned: a DataFrame is several times larger in RAM


class QueryTooExpensive(Exception):
    pass


class ResultTooLarge(Exception):
    """The query is cheap to scan but returns more than fits comfortably in memory."""


def _fmt(n: int) -> str:
    return f"{n / 1024**3:.2f} GiB"


def cache_key(sql: str, project: str) -> str:
    return hashlib.sha256(f"{project}\n{' '.join(sql.split())}".encode()).hexdigest()[:24]


def _default_client(project: str):
    from google.cloud import bigquery  # lazy: keeps the guard testable without GCP libs

    return bigquery.Client(project=project)


def bq(
    sql: str,
    *,
    project: str = "hankstank",
    max_bytes: int | None = None,
    max_result_bytes: int | None = None,
    force: bool = False,
    refresh: bool = False,
    cache_dir: str | os.PathLike | None = None,
    client=None,
):
    """Run sql and return a pandas DataFrame."""
    import duckdb

    cap = max_bytes if max_bytes is not None else int(os.environ.get("MLLAB_BQ_MAX_BYTES", DEFAULT_MAX_BYTES))
    cache = Path(cache_dir or os.environ.get("MLLAB_CACHE", "/mnt/appdata/mllab/cache")) / "bq"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{cache_key(sql, project)}.parquet"

    if path.exists() and not refresh:
        print(f"[bq] cache hit {path.name} (0 bytes billed)")
        return duckdb.sql(f"SELECT * FROM read_parquet('{path}')").df()

    from google.cloud import bigquery

    client = client or _default_client(project)
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    est = dry.total_bytes_processed
    print(f"[bq] estimated {_fmt(est)} (cap {_fmt(cap)})")
    if est > cap and not force:
        raise QueryTooExpensive(f"estimated {_fmt(est)} exceeds cap {_fmt(cap)}; pass force=True to override")

    limit = max(est, cap) if force else cap
    job = client.query(sql, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=limit))
    rows = job.result()  # wait; results now sit in BigQuery's temp table, nothing is in our RAM yet
    result_cap = (max_result_bytes if max_result_bytes is not None
                  else int(os.environ.get("MLLAB_BQ_MAX_RESULT_BYTES", DEFAULT_MAX_RESULT_BYTES)))
    # num_bytes can read 0 for small results, so floor it with rows x cols x 8 (pandas' minimum per cell)
    size = max(client.get_table(job.destination).num_bytes or 0,
               (rows.total_rows or 0) * len(rows.schema) * 8)
    print(f"[bq] result {_fmt(size)} (cap {_fmt(result_cap)})")
    if size > result_cap and not force:
        raise ResultTooLarge(
            f"result is {_fmt(size)} (cap {_fmt(result_cap)}); nothing was downloaded. Select fewer columns, "
            "aggregate or LIMIT in SQL, or pass force=True if the kernel has the memory. "
            "Re-running the same query within 24h is served from BigQuery's cache at no charge."
        )
    # REST download by default: the Storage API needs bigquery.readsessions.create (roles/bigquery.readSessionUser),
    # which the lab's service account deliberately does not have. Set MLLAB_BQ_STORAGE_API=1 once it is granted.
    df = job.to_dataframe(create_bqstorage_client=os.environ.get("MLLAB_BQ_STORAGE_API") == "1")
    df.to_parquet(path)
    return df
