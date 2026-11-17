"""Fuzzy matching: Levenshtein distance for 200k string pairs (fixed width 12, NUL-padded)."""

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist

from lab.harness import run_experiment

N, L = 200_000, 12


class Lev(torch.nn.Module):
    def forward(self, a, b, la, lb):  # a, b: [N, L] int64; la, lb: [N] lengths
        n = a.shape[0]
        idx = torch.arange(L + 1, dtype=torch.int64, device=a.device)
        prev = idx.unsqueeze(0).expand(n, L + 1)  # row 0: 0..L
        rows = [prev]
        for i in range(1, L + 1):
            cur = [torch.full((n,), i, dtype=torch.int64, device=a.device)]
            for j in range(1, L + 1):
                cost = (a[:, i - 1] != b[:, j - 1]).to(torch.int64)
                v = torch.minimum(
                    torch.minimum(prev[:, j] + 1, cur[j - 1] + 1), prev[:, j - 1] + cost
                )
                cur.append(v)
            prev = torch.stack(cur, dim=1)
            rows.append(prev)
        table = torch.stack(rows, dim=1)  # [N, L+1, L+1]
        return table[torch.arange(n, device=a.device), la, lb]


def main():
    rng = np.random.default_rng(0)
    alpha = np.frombuffer(b"abcdefghij", np.uint8)
    la = rng.integers(4, L + 1, N)
    lb = np.clip(la + rng.integers(-2, 3, N), 1, L)
    a = np.zeros((N, L), np.int64)
    b = np.zeros((N, L), np.int64)
    sa, sb = [], []
    for k in range(N):
        x = rng.choice(alpha, la[k])
        y = x.copy()[: lb[k]]
        if len(y) < lb[k]:
            y = np.concatenate([y, rng.choice(alpha, lb[k] - len(y))])
        m = rng.random(len(y)) < 0.25
        y[m] = rng.choice(alpha, m.sum())
        a[k, : la[k]] = x
        b[k, : lb[k]] = y
        sa.append(bytes(x).decode())
        sb.append(bytes(y).decode())
    run_experiment(
        "levenshtein",
        Lev(),
        (a, b, la.astype(np.int64), lb.astype(np.int64)),
        ["a", "b", "la", "lb"],
        baselines={
            "rapidfuzz loop": lambda *_: np.array(
                [Levenshtein.distance(x, y) for x, y in zip(sa, sb)]
            ),
            "rapidfuzz.cpdist (C++, all cores)": lambda *_: cpdist(
                sa, sb, scorer=Levenshtein.distance, workers=-1
            ),
        },
        check=lambda out, ref: np.array_equal(out.ravel(), ref.ravel()),
        size_note=f"{N:,} pairs, <= {L} chars",
    )


if __name__ == "__main__":
    main()
