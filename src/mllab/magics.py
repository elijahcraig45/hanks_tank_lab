"""IPython cell magic:  %%bq [--project P] [--max-gb N] [--max-result-gb N] [--force] [--refresh] [varname]

Usage in a notebook:
    %load_ext mllab.magics
    %%bq games
    SELECT * FROM `hankstank.mlb_2026_season.game_predictions` LIMIT 100
"""
from __future__ import annotations

import shlex


def _parse(line: str) -> dict:
    opts = {"project": "hankstank", "max_gb": None, "max_result_gb": None, "force": False, "refresh": False, "var": None}
    toks = shlex.split(line)
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "--project":
            opts["project"] = toks[i + 1]; i += 2
        elif t == "--max-gb":
            opts["max_gb"] = float(toks[i + 1]); i += 2
        elif t == "--max-result-gb":
            opts["max_result_gb"] = float(toks[i + 1]); i += 2
        elif t == "--force":
            opts["force"] = True; i += 1
        elif t == "--refresh":
            opts["refresh"] = True; i += 1
        elif t.startswith("-"):
            raise ValueError(f"unknown option {t}")
        else:
            opts["var"] = t; i += 1
    return opts


def bq_magic(line: str, cell: str, ns: dict | None = None):
    from .bq import bq

    o = _parse(line)
    df = bq(
        cell,
        project=o["project"],
        max_bytes=int(o["max_gb"] * 1024**3) if o["max_gb"] is not None else None,
        max_result_bytes=int(o["max_result_gb"] * 1024**3) if o["max_result_gb"] is not None else None,
        force=o["force"],
        refresh=o["refresh"],
    )
    if o["var"]:
        if ns is not None:
            ns[o["var"]] = df
        return None
    return df


def ask_magic(line: str, cell: str):
    """%%ask [--with-table DATASET.TABLE ...]  then the question as the cell body. Prints advice; runs nothing."""
    from . import ask as A

    toks, tables, i = shlex.split(line), [], 0
    while i < len(toks):
        if toks[i] == "--with-table" and i + 1 < len(toks):
            tables.append(toks[i + 1]); i += 2
        else:
            raise ValueError(f"unknown option {toks[i]}")
    print(A.ask(cell, context=A.build_context(tables=tables)))


def load_ipython_extension(ip):
    ip.register_magic_function(lambda line, cell: bq_magic(line, cell, ip.user_ns), "cell", "bq")
    ip.register_magic_function(ask_magic, "cell", "ask")
