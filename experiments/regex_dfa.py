"""Text: validate 1M strings against ^[a-z0-9._]+@[a-z0-9]+\\.[a-z]{2,6}$ via a DFA run as tensor gathers."""

import re

import numpy as np
import torch

from lab.harness import run_experiment

N, W = 1_000_000, 32
PATTERN = re.compile(r"[a-z0-9._]+@[a-z0-9]+\.[a-z]{2,6}")

# states: 0 start, 1 local, 2 after @, 3 domain, 4 after dot, 5..10 = 1..6 tld letters, 11 dead
LOWER = [ord(c) for c in "abcdefghijklmnopqrstuvwxyz"]
DIGIT = [ord(c) for c in "0123456789"]
S, DEAD = 12, 11
table = np.full((S, 256), DEAD, np.int64)
for c in LOWER + DIGIT + [ord("."), ord("_")]:
    table[0, c] = table[1, c] = 1
table[1, ord("@")] = 2
for c in LOWER + DIGIT:
    table[2, c] = table[3, c] = 3
table[3, ord(".")] = 4
for c in LOWER:
    table[4, c] = 5
    for s in range(5, 10):
        table[s, c] = s + 1
ACCEPT = np.zeros(S, bool)
ACCEPT[6:11] = True  # 2..6 tld letters
for s in range(6, 11):
    table[s, 0] = s  # NUL padding keeps accepting states
TABLE_FLAT = table.reshape(-1)


class DFA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.from_numpy(TABLE_FLAT))
        self.register_buffer("accept", torch.from_numpy(ACCEPT))

    def forward(self, chars):  # [N, W] int64
        state = torch.zeros(chars.shape[0], dtype=torch.int64, device=chars.device)
        for p in range(W):
            state = self.table[state * 256 + chars[:, p]]
        return self.accept[state]


def make_strings(rng):
    alpha = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz0123456789._", np.uint8)
    out = []
    for i in range(N):
        a = bytes(rng.choice(alpha, rng.integers(1, 10)))
        d = bytes(rng.choice(alpha[:36], rng.integers(1, 8)))
        t = bytes(rng.choice(alpha[:26], rng.integers(1, 8)))
        s = a + b"@" + d + b"." + t
        if i % 3 == 0:  # corrupt a third of them
            s = s.replace(b"@", b"#")
        out.append(s[:W])
    return out


def main():
    rng = np.random.default_rng(0)
    strs = make_strings(rng)
    chars = np.zeros((N, W), np.int64)
    for i, s in enumerate(strs):
        chars[i, : len(s)] = np.frombuffer(s, np.uint8)
    pystrs = [s.decode() for s in strs]

    def numpy_dfa(c):
        st = np.zeros(c.shape[0], np.int64)
        for p in range(W):
            st = TABLE_FLAT[st * 256 + c[:, p]]
        return ACCEPT[st]

    run_experiment(
        "regex_dfa",
        DFA(),
        (chars,),
        ["chars"],
        baselines={
            "python re.fullmatch": lambda c: np.array(
                [PATTERN.fullmatch(s) is not None for s in pystrs]
            ),
            "numpy DFA": numpy_dfa,
        },
        check=lambda out, ref: np.array_equal(out.astype(bool), ref.astype(bool)),
        size_note=f"{N:,} strings x {W} chars",
    )


if __name__ == "__main__":
    main()


class DFA32(torch.nn.Module):
    """Same DFA with int32 states/chars: WebGPU has no native int64, so int64 graphs fall off the GPU."""

    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.from_numpy(TABLE_FLAT.astype(np.int32)))
        self.register_buffer("accept", torch.from_numpy(ACCEPT.astype(np.int32)))

    def forward(self, chars):  # [N, W] int32
        # index_select -> Gather (int32 indices allowed). Plain tensor indexing exports GatherND, which the
        # torch exporter emits with int32 indices although the ONNX spec only allows int64 (invalid model).
        state = torch.zeros(chars.shape[0], dtype=torch.int32, device=chars.device)
        for p in range(W):
            state = self.table.index_select(0, state * 256 + chars[:, p])
        return self.accept.index_select(0, state)


def main_i32():
    rng = np.random.default_rng(0)
    strs = make_strings(rng)
    chars = np.zeros((N, W), np.int32)
    for i, s in enumerate(strs):
        chars[i, : len(s)] = np.frombuffer(s, np.uint8)
    pystrs = [s.decode() for s in strs]
    run_experiment(
        "regex_dfa_i32",
        DFA32(),
        (chars,),
        ["chars"],
        baselines={
            "python re.fullmatch": lambda c: np.array(
                [PATTERN.fullmatch(s) is not None for s in pystrs]
            )
        },
        check=lambda out, ref: np.array_equal(out.astype(bool), ref.astype(bool)),
        size_note=f"{N:,} strings x {W} chars, int32",
    )
