"""SQL: SELECT key, SUM(value) FROM t GROUP BY key — 10M rows, 1000 groups."""

import duckdb
import numpy as np
import pandas as pd
import torch

from lab.harness import run_experiment

N, K = 10_000_000, 1000


class GroupSum(torch.nn.Module):
    def forward(self, keys, vals):
        # scatter_add -> ScatterElements(reduction=add). index_add exports ScatterND, which races in ORT 1.32 CPU.
        return torch.zeros(K, dtype=vals.dtype, device=vals.device).scatter_add(
            0, keys, vals
        )


def main():
    rng = np.random.default_rng(0)
    keys = rng.integers(0, K, N).astype(np.int64)
    vals = rng.random(N).astype(np.float32)
    con = duckdb.connect()

    def duck(k, v):
        con.register("t", pd.DataFrame({"k": k, "v": v}))
        r = con.execute("SELECT k, SUM(v) FROM t GROUP BY k ORDER BY k").fetchnumpy()
        out = np.zeros(K, np.float64)
        out[r["k"]] = r["sum(v)"]
        return out

    run_experiment(
        "groupby_sum",
        GroupSum(),
        (keys, vals),
        ["keys", "vals"],
        baselines={
            "numpy.bincount": lambda k, v: np.bincount(k, weights=v, minlength=K),
            "pandas.groupby": lambda k, v: (
                pd.Series(v).groupby(k).sum().reindex(range(K), fill_value=0).to_numpy()
            ),
            "duckdb": duck,
        },
        check=lambda out, ref: np.allclose(out, ref, rtol=1e-3),
        size_note=f"{N:,} rows, {K} groups",
    )


if __name__ == "__main__":
    main()
