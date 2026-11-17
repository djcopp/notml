"""Fuzzy matching: Levenshtein distance for many short string pairs, as a fully unrolled DP table.

This one loses badly (see the README). It's kept because it shows where the approach stops working:
an unrolled 12x12 dynamic program is ~1,500 tiny nodes, and launch overhead swamps the arithmetic.
"""

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist

from lab.case import Case, Tolerance

WIDTH = 12  # strings are NUL-padded to this length


class EditDistance(torch.nn.Module):
    def forward(self, a, b, len_a, len_b):  # a, b: [N, WIDTH]; len_a, len_b: [N]
        n = a.shape[0]
        prev = torch.arange(WIDTH + 1, device=a.device).unsqueeze(0).expand(n, WIDTH + 1)
        rows = [prev]
        for i in range(1, WIDTH + 1):
            cur = [torch.full((n,), i, dtype=torch.int64, device=a.device)]
            for j in range(1, WIDTH + 1):
                cost = (a[:, i - 1] != b[:, j - 1]).to(torch.int64)
                cur.append(torch.minimum(torch.minimum(prev[:, j] + 1, cur[j - 1] + 1), prev[:, j - 1] + cost))
            prev = torch.stack(cur, dim=1)
            rows.append(prev)
        table = torch.stack(rows, dim=1)  # [N, WIDTH+1, WIDTH+1]
        return table[torch.arange(n, device=a.device), len_a, len_b]


def make_pairs(n: int, rng) -> tuple[list[str], list[str]]:
    """Random strings over a 10-letter alphabet, each paired with a copy that has about 25% of letters changed."""
    alphabet = np.frombuffer(b"abcdefghij", np.uint8)
    xs, ys = [], []
    for _ in range(n):
        la = rng.integers(4, WIDTH + 1)
        lb = int(np.clip(la + rng.integers(-2, 3), 1, WIDTH))
        x = rng.choice(alphabet, la)
        y = np.concatenate([x[:lb], rng.choice(alphabet, max(0, lb - la))])
        edit = rng.random(lb) < 0.25
        y[edit] = rng.choice(alphabet, edit.sum())
        xs.append(bytes(x).decode())
        ys.append(bytes(y).decode())
    return xs, ys


def encode(strings: list[str]) -> np.ndarray:
    out = np.zeros((len(strings), WIDTH), np.int64)
    for i, s in enumerate(strings):
        out[i, : len(s)] = np.frombuffer(s.encode(), np.uint8)
    return out


def cases(small: bool = False) -> list[Case]:
    n = 2_000 if small else 200_000
    xs, ys = make_pairs(n, np.random.default_rng(0))
    return [
        Case(
            "levenshtein",
            EditDistance(),
            {
                "a": encode(xs),
                "b": encode(ys),
                "len_a": np.array([len(s) for s in xs], np.int64),
                "len_b": np.array([len(s) for s in ys], np.int64),
            },
            {
                "rapidfuzz (all cores)": lambda *_: cpdist(xs, ys, scorer=Levenshtein.distance, workers=-1),
                "rapidfuzz (loop)": lambda *_: np.array(
                    [Levenshtein.distance(x, y) for x, y in zip(xs, ys, strict=True)]
                ),
            },
            Tolerance("exact"),
            f"{n:,} pairs, up to {WIDTH} characters",
        )
    ]
