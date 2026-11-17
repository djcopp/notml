"""Graphs: PageRank as an edge list, and all-pairs shortest paths by min-plus squaring."""

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import floyd_warshall

from lab.case import Case, Tolerance

DAMPING, ITERATIONS = 0.85, 30


class PageRank(torch.nn.Module):
    """Each iteration gathers rank at the source of every edge and scatter-adds it into the target."""

    def __init__(self, nodes: int):
        super().__init__()
        self.nodes = nodes

    def forward(self, src, dst, weight):  # weight = 1 / out_degree(src)
        n = self.nodes
        rank = torch.full((n,), 1.0 / n, device=weight.device)
        for _ in range(ITERATIONS):
            incoming = torch.zeros(n, device=weight.device).scatter_add(0, dst, rank[src] * weight)
            rank = DAMPING * incoming + (1 - DAMPING) / n
        return rank


class ShortestPaths(torch.nn.Module):
    """Repeated min-plus squaring: after k steps, d holds the shortest paths using up to 2^k edges."""

    def __init__(self, nodes: int):
        super().__init__()
        self.steps = int(np.ceil(np.log2(nodes)))

    def forward(self, d):  # [N, N], inf where there is no edge
        for _ in range(self.steps):
            d = torch.minimum(d, (d.unsqueeze(2) + d.unsqueeze(0)).amin(dim=1))
        return d


def pagerank_case(nodes: int, degree: int = 20) -> Case:
    rng = np.random.default_rng(0)
    src = np.repeat(np.arange(nodes), degree)
    dst = rng.integers(0, nodes, nodes * degree)
    adjacency = sp.csr_matrix((np.ones(len(src), np.float32), (dst, src)), shape=(nodes, nodes))
    transition = adjacency.multiply(1.0 / np.asarray(adjacency.sum(0)).ravel()[None, :]).tocsr().astype(np.float32)
    edges = transition.tocoo()

    def scipy_pagerank(*_):
        rank = np.full(nodes, 1 / nodes, np.float32)
        for _ in range(ITERATIONS):
            rank = DAMPING * (transition @ rank) + (1 - DAMPING) / nodes
        return rank

    return Case(
        "pagerank",
        PageRank(nodes),
        {
            "src": edges.col.astype(np.int64),
            "dst": edges.row.astype(np.int64),
            "weight": edges.data.astype(np.float32),
        },
        {"scipy.sparse": scipy_pagerank},
        Tolerance("close", rtol=1e-3, atol=1e-7),
        f"{nodes:,} nodes, {edges.nnz:,} edges, {ITERATIONS} iterations",
    )


def shortest_paths_case(nodes: int, density: float = 0.05) -> Case:
    rng = np.random.default_rng(0)
    w = np.full((nodes, nodes), np.inf, np.float32)
    mask = rng.random((nodes, nodes)) < density
    w[mask] = rng.random(mask.sum()).astype(np.float32) * 10
    np.fill_diagonal(w, 0)
    return Case(
        "shortest_paths",
        ShortestPaths(nodes),
        {"d": w},
        {"scipy floyd_warshall": lambda d: floyd_warshall(d, directed=True).astype(np.float32)},
        Tolerance("close", rtol=1e-4),
        f"{nodes} nodes, all pairs",
    )


def cases(small: bool = False) -> list[Case]:
    return [pagerank_case(500 if small else 4000), shortest_paths_case(64 if small else 256)]
