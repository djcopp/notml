"""Graphs: PageRank (30 power iterations) and all-pairs shortest paths (min-plus squaring)."""

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import floyd_warshall

from lab.harness import run_experiment

PR_N, PR_DEG, PR_ITERS, DAMP = 4000, 20, 30, 0.85
SP_N = 256
LOG_STEPS = int(np.ceil(np.log2(SP_N)))


class PageRank(torch.nn.Module):
    """Sparse PageRank as an edge list: gather contributions at src, scatter-add into dst."""

    def forward(self, src, dst, w):  # [E] int64, [E] int64, [E] float (1/outdeg[src])
        r = torch.full((PR_N,), 1.0 / PR_N, device=w.device)
        for _ in range(PR_ITERS):
            contrib = r[src] * w
            r = (
                DAMP * torch.zeros(PR_N, device=w.device).scatter_add(0, dst, contrib)
                + (1 - DAMP) / PR_N
            )
        return r


class APSP(torch.nn.Module):
    def forward(self, d):  # d: [N, N] with inf for no edge
        for _ in range(LOG_STEPS):
            d = torch.minimum(d, (d.unsqueeze(2) + d.unsqueeze(0)).amin(dim=1))
        return d


def main():
    rng = np.random.default_rng(0)
    # PageRank on a random sparse digraph (each node links to PR_DEG random nodes)
    rows = np.repeat(np.arange(PR_N), PR_DEG)
    cols = rng.integers(0, PR_N, PR_N * PR_DEG)
    A = sp.csr_matrix(
        (np.ones(len(rows), np.float32), (cols, rows)), shape=(PR_N, PR_N)
    )
    M = (
        (A.multiply(1.0 / np.asarray(A.sum(0)).ravel()[None, :]))
        .tocsr()
        .astype(np.float32)
    )
    coo = M.tocoo()
    src, dst, w = (
        coo.col.astype(np.int64),
        coo.row.astype(np.int64),
        coo.data.astype(np.float32),
    )

    def scipy_pr(*_):
        r = np.full(PR_N, 1 / PR_N, np.float32)
        for _ in range(PR_ITERS):
            r = DAMP * (M @ r) + (1 - DAMP) / PR_N
        return r

    run_experiment(
        "pagerank",
        PageRank(),
        (src, dst, w),
        ["src", "dst", "w"],
        baselines={"scipy.sparse": scipy_pr},
        check=lambda out, ref: np.allclose(out, ref, rtol=1e-3, atol=1e-7),
        size_note=f"{PR_N} nodes, {PR_N * PR_DEG:,} edges, {PR_ITERS} iters (edge-list gather/scatter)",
    )

    # APSP on a random weighted graph
    W = np.full((SP_N, SP_N), np.inf, np.float32)
    mask = rng.random((SP_N, SP_N)) < 0.05
    W[mask] = rng.random(mask.sum()).astype(np.float32) * 10
    np.fill_diagonal(W, 0)
    run_experiment(
        "apsp_minplus",
        APSP(),
        (W,),
        ["d"],
        baselines={
            "scipy floyd_warshall": lambda w: floyd_warshall(w, directed=True).astype(
                np.float32
            )
        },
        check=lambda out, ref: np.allclose(out, ref, rtol=1e-4),
        size_note=f"{SP_N} nodes, dense min-plus",
    )


if __name__ == "__main__":
    main()
