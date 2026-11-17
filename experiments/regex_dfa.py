"""Text: validate strings against an email-like pattern by running a DFA as table lookups.

Pattern: [a-z0-9._]+@[a-z0-9]+\\.[a-z]{2,6}

Two versions of the same graph. The int64 one is what PyTorch gives you by default. The int32 one is
identical apart from the integer width, and it is the one that stays on WebGPU, which has no int64.
"""

import re

import numpy as np
import torch

from lab.case import Case, Tolerance

WIDTH = 32  # strings are NUL-padded to this length
PATTERN = re.compile(r"[a-z0-9._]+@[a-z0-9]+\.[a-z]{2,6}")


def transition_table() -> tuple[np.ndarray, np.ndarray]:
    """States: 0 start, 1 local part, 2 after @, 3 domain, 4 after dot, 5-10 one to six TLD letters, 11 dead."""
    lower = [ord(c) for c in "abcdefghijklmnopqrstuvwxyz"]
    digit = [ord(c) for c in "0123456789"]
    states, dead = 12, 11
    table = np.full((states, 256), dead, np.int64)
    for c in lower + digit + [ord("."), ord("_")]:
        table[0, c] = table[1, c] = 1
    table[1, ord("@")] = 2
    for c in lower + digit:
        table[2, c] = table[3, c] = 3
    table[3, ord(".")] = 4
    for c in lower:
        table[4, c] = 5
        for s in range(5, 10):
            table[s, c] = s + 1
    accept = np.zeros(states, bool)
    accept[6:11] = True  # two to six TLD letters
    table[6:11, 0] = np.arange(6, 11)  # NUL padding leaves an accepting state where it is
    return table.reshape(-1), accept


TABLE, ACCEPT = transition_table()


class DFA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.from_numpy(TABLE))
        self.register_buffer("accept", torch.from_numpy(ACCEPT))

    def forward(self, chars):  # [N, WIDTH] int64
        state = torch.zeros(chars.shape[0], dtype=torch.int64, device=chars.device)
        for p in range(WIDTH):
            state = self.table[state * 256 + chars[:, p]]
        return self.accept[state]


class DFA32(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.from_numpy(TABLE.astype(np.int32)))
        self.register_buffer("accept", torch.from_numpy(ACCEPT.astype(np.int32)))

    def forward(self, chars):  # [N, WIDTH] int32
        # index_select exports as Gather, which accepts int32 indices. Plain indexing exports as GatherND,
        # which the exporter emits with int32 indices even though the spec requires int64
        # (fixed in microsoft/onnxscript#3221).
        state = torch.zeros(chars.shape[0], dtype=torch.int32, device=chars.device)
        for p in range(WIDTH):
            state = self.table.index_select(0, state * 256 + chars[:, p])
        return self.accept.index_select(0, state)


def make_strings(n: int, rng) -> list[str]:
    """Random address-shaped strings; every third one has its @ replaced so it should be rejected."""
    alphabet = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz0123456789._", np.uint8)
    out = []
    for i in range(n):
        local = bytes(rng.choice(alphabet, rng.integers(1, 10)))
        domain = bytes(rng.choice(alphabet[:36], rng.integers(1, 8)))
        tld = bytes(rng.choice(alphabet[:26], rng.integers(1, 8)))
        s = local + b"@" + domain + b"." + tld
        if i % 3 == 0:
            s = s.replace(b"@", b"#")
        out.append(s[:WIDTH].decode())
    return out


def encode(strings: list[str], dtype) -> np.ndarray:
    out = np.zeros((len(strings), WIDTH), dtype)
    for i, s in enumerate(strings):
        out[i, : len(s)] = np.frombuffer(s.encode(), np.uint8)
    return out


def numpy_dfa(chars):
    state = np.zeros(chars.shape[0], np.int64)
    for p in range(WIDTH):
        state = TABLE[state * 256 + chars[:, p]]
    return ACCEPT[state]


def cases(small: bool = False) -> list[Case]:
    n = 10_000 if small else 1_000_000
    strings = make_strings(n, np.random.default_rng(0))

    def python_re(_):
        return np.array([PATTERN.fullmatch(s) is not None for s in strings])

    size = f"{n:,} strings, up to {WIDTH} characters"
    return [
        Case(
            "regex_dfa_int64",
            DFA(),
            {"chars": encode(strings, np.int64)},
            {"python re": python_re, "numpy": numpy_dfa},
            Tolerance("exact"),
            size,
        ),
        Case(
            "regex_dfa_int32",
            DFA32(),
            {"chars": encode(strings, np.int32)},
            {"python re": python_re, "numpy": numpy_dfa},
            Tolerance("exact"),
            size,
        ),
    ]
