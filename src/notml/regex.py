"""Compile a regular expression into an ONNX graph that checks a whole batch of strings at once.

    model = notml.regex.compile(r"\\d{4}-\\d{2}-\\d{2}", width=32)
    chars = notml.regex.encode(["2024-01-31", "not a date"], width=32)
    session.run(None, {"chars": chars})[0]  # -> [True, False]

The pattern becomes a deterministic finite automaton (via ``interegular``), and the automaton becomes a
graph of standard ONNX ops: table lookups and integer arithmetic, all int32, so it runs unchanged
on onnxruntime's CPU, CUDA and CoreML providers and on WebGPU in a browser.

Matching is ``re.fullmatch`` on ASCII text. Strings go in as a uint8 array padded with NUL bytes to
``width``; NUL is reserved as padding and must not appear in the strings themselves.

Two ways to run the automaton:

``stride``  Step through the string ``stride`` characters at a time, one lookup per step. Precomputing the
            transitions for character pairs (stride 2) or triples halves or thirds the number of
            sequential steps, at the cost of a larger table.
``scan``    Treat each character as a function from states to states and compose those functions in a
            balanced tree (a parallel prefix over the string). That takes log2(width) steps instead of
            width, but does about ``states`` times more work, so it pays off for long strings and small
            automata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import interegular
import numpy as np
import onnx
from interegular.fsm import anything_else

from notml.graph import Graph

PAD = 0  # the byte used for padding; it leaves every state unchanged


@dataclass(frozen=True)
class DFA:
    """A byte-level automaton. State 0 is dead; ``table[state, cls]`` is the next state."""

    byte_class: np.ndarray  # [256] int32, byte -> character class; class 0 is padding
    table: np.ndarray  # [states, classes] int32
    start: int
    accepting: np.ndarray  # [states] bool

    @property
    def states(self) -> int:
        return self.table.shape[0]

    @property
    def classes(self) -> int:
        return self.table.shape[1]

    def fullmatch(self, s: str) -> bool:
        state = self.start
        for b in s.encode():
            state = self.table[state, self.byte_class[b]]
        return bool(self.accepting[state])


def dfa(pattern: str) -> DFA:
    """Build the minimal byte-level DFA for ``pattern``."""
    fsm = interegular.parse_pattern(pattern).to_fsm().reduce()
    symbols = fsm.alphabet._symbol_mapping  # char (or anything_else) -> symbol id
    other = symbols.get(anything_else)

    # Our classes: 0 = padding, 1 + symbol id for interegular's symbols, and one extra class for
    # characters the pattern never mentions when interegular has no "anything else" symbol.
    n_symbols = len(set(symbols.values()))
    unmatched = n_symbols + 1
    byte_class = np.full(256, unmatched if other is None else 1 + other, np.int32)
    for ch, sym in symbols.items():
        if ch is not anything_else and len(ch) == 1 and ord(ch) < 128:
            byte_class[ord(ch)] = 1 + sym
    byte_class[PAD] = 0

    # States: 0 = dead, then interegular's states renumbered from 1.
    ids = {s: i + 1 for i, s in enumerate(sorted(fsm.states))}
    classes = unmatched + 1
    table = np.zeros((len(ids) + 1, classes), np.int32)
    table[:, 0] = np.arange(len(ids) + 1)  # padding keeps the state
    for s, row in fsm.map.items():
        for sym, nxt in row.items():
            table[ids[s], 1 + sym] = ids[nxt]
    accepting = np.zeros(len(ids) + 1, bool)
    accepting[[ids[s] for s in fsm.finals]] = True
    return DFA(byte_class, table, ids[fsm.initial], accepting)


def encode(strings: list[str], width: int) -> np.ndarray:
    """ASCII strings as a ``[len(strings), width]`` uint8 array, NUL-padded."""
    out = np.zeros((len(strings), width), np.uint8)
    for i, s in enumerate(strings):
        b = s.encode("ascii")
        if len(b) > width:
            raise ValueError(f"string {i} is {len(b)} bytes, longer than width={width}")
        if b"\0" in b:
            raise ValueError(f"string {i} contains a NUL byte, which is reserved for padding")
        out[i, : len(b)] = np.frombuffer(b, np.uint8)
    return out


def _stride_table(d: DFA, stride: int) -> np.ndarray:
    """Transitions over ``stride`` characters at once: ``[S, C**stride]``, combined class in row-major order."""
    table = d.table
    for _ in range(stride - 1):
        table = d.table[table].reshape(d.states, -1)  # next[s, a * C + c] = table[first[s, a], c]
    return table


def compile(  # noqa: A001 - mirrors re.compile
    pattern: str,
    width: int = 32,
    method: Literal["stride", "scan"] = "stride",
    stride: int = 2,
) -> onnx.ModelProto:
    """ONNX model: input ``chars`` uint8 ``[N, width]`` (from :func:`encode`), output ``match`` bool ``[N]``."""
    d = dfa(pattern)
    g = Graph(f"regex_{method}")
    chars = g.op("Cast", g.input("chars", np.uint8, ["N", width]), to=onnx.TensorProto.INT32)
    cls = g.op("Gather", g.const(d.byte_class), chars)  # [N, width] character classes

    if method == "stride":
        if width % stride:
            raise ValueError("width must be a multiple of stride")
        table = _stride_table(d, stride)
        flat = g.const(table.reshape(-1).astype(np.int32))
        steps = width // stride
        # combined class for each step: c0 * C^(k-1) + c1 * C^(k-2) + ... (row-major over the stride)
        grouped = g.op("Reshape", cls, g.const([0, steps, stride], np.int64))
        weights = g.const((d.classes ** np.arange(stride - 1, -1, -1)).astype(np.int32))
        combined = g.op("ReduceSum", g.op("Mul", grouped, weights), g.const([2], np.int64), keepdims=0)
        row = g.const(np.int32(table.shape[1]))
        state = None
        for p in range(steps):
            col = g.op("Gather", combined, g.const(np.int64(p)), axis=1)  # [N]
            if state is None:  # the first step starts from the start state, so its row is a constant
                state = g.op("Gather", g.const(table[d.start].astype(np.int32)), col)
            else:
                state = g.op("Gather", flat, g.op("Add", g.op("Mul", state, row), col))
        final = state
    elif method == "scan":
        levels = int(np.log2(width))
        if 2**levels != width:
            raise ValueError("scan needs a power-of-two width")
        # Each character becomes a function from states to states: [N, width, S].
        funcs = g.op("Gather", g.const(np.ascontiguousarray(d.table.T)), cls)
        for _ in range(levels):
            pairs = g.op("Reshape", funcs, g.const([0, -1, 2, d.states], np.int64))
            first = g.op("Gather", pairs, g.const(np.int64(0)), axis=2)  # [N, L/2, S]
            then = g.op("Gather", pairs, g.const(np.int64(1)), axis=2)
            funcs = g.op("GatherElements", then, first, axis=2)  # then(first(s))
        # funcs: [N, 1, S]; apply to the start state
        funcs = g.op("Reshape", funcs, g.const([0, d.states], np.int64))
        final = g.op("Gather", funcs, g.const(np.int64(d.start)), axis=1)
    else:
        raise ValueError(f"unknown method {method!r}")

    match = g.op("Gather", g.const(d.accepting), final)
    g.output(match, "match", np.bool_, ["N"])
    return g.model()


def match(pattern: str, strings: list[str], width: int = 32, providers=None, **options) -> np.ndarray:
    """Convenience wrapper: compile, run with onnxruntime, return a bool array."""
    import onnxruntime as ort

    session = ort.InferenceSession(compile(pattern, width, **options).SerializeToString(), providers=providers)
    return session.run(None, {"chars": encode(strings, width)})[0]
