"""SQL: SELECT key, SUM(value) FROM t GROUP BY key."""

import duckdb
import numpy as np
import pandas as pd
import torch

from lab.case import Case, Tolerance


class GroupSum(torch.nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, keys, vals):
        # scatter_add exports as ScatterElements. index_add would export as ScatterND, whose CPU kernel
        # loses updates under multithreading in onnxruntime 1.32 (microsoft/onnxruntime#34274).
        return torch.zeros(self.groups, dtype=vals.dtype, device=vals.device).scatter_add(0, keys, vals)


def cases(small: bool = False) -> list[Case]:
    rows, groups = (100_000, 100) if small else (10_000_000, 1000)
    rng = np.random.default_rng(0)
    keys = rng.integers(0, groups, rows).astype(np.int64)
    vals = rng.random(rows).astype(np.float32)
    con = duckdb.connect()

    def duck(k, v):
        con.register("t", pd.DataFrame({"k": k, "v": v}))
        r = con.execute("SELECT k, SUM(v) AS s FROM t GROUP BY k").fetchnumpy()
        out = np.zeros(groups)
        out[r["k"]] = r["s"]
        return out

    return [
        Case(
            "groupby",
            GroupSum(groups),
            {"keys": keys, "vals": vals},
            {
                "numpy.bincount": lambda k, v: np.bincount(k, weights=v, minlength=groups),
                "pandas": lambda k, v: pd.Series(v).groupby(k).sum().reindex(range(groups), fill_value=0).to_numpy(),
                "duckdb": duck,
            },
            Tolerance("close", rtol=1e-3),
            f"{rows:,} rows, {groups:,} groups",
        )
    ]
