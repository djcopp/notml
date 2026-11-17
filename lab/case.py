"""The unit of work: one workload, its inputs, the baselines it races, and how to judge the answer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import onnx
import torch


@dataclass(frozen=True)
class Tolerance:
    """How an output is compared with the reference. The browser bench implements the same four rules.

    exact       every element equal
    close       numpy.allclose(out, ref, rtol, atol)
    max_rel     max |out - ref| / max |ref| < rtol   (for outputs with a few near-zero elements)
    scalar_rel  |out[0] - ref[0]| / |ref[0]| < rtol  (Monte Carlo estimates)
    """

    kind: Literal["exact", "close", "max_rel", "scalar_rel"]
    rtol: float = 0.0
    atol: float = 0.0

    def check(self, out, ref) -> bool:
        out = np.asarray(out, np.float64).ravel()
        ref = np.asarray(ref, np.float64).ravel()
        if out.shape != ref.shape:
            return False
        if self.kind == "exact":
            return bool(np.array_equal(out, ref))
        if self.kind == "close":
            return bool(np.allclose(out, ref, rtol=self.rtol, atol=self.atol))
        if self.kind == "max_rel":
            return bool(np.abs(out - ref).max() / np.abs(ref).max() < self.rtol)
        if self.kind == "scalar_rel":
            return bool(abs(out[0] - ref[0]) / abs(ref[0]) < self.rtol)
        raise ValueError(self.kind)


@dataclass(frozen=True)
class Loop:
    """Call the model repeatedly, feeding outputs back in. ``feedback`` maps input name -> output index."""

    calls: int
    feedback: dict[str, int]


@dataclass
class Case:
    name: str
    model: torch.nn.Module | onnx.ModelProto
    inputs: dict[str, np.ndarray]
    baselines: dict[str, Callable[..., np.ndarray]]  # the first one is the reference answer
    tolerance: Tolerance
    size: str
    loop: Loop | None = None
    known_bad: str | None = None  # kept on purpose as a negative result; tests skip it
    js_regexp: str | None = None  # the browser bench also times this JavaScript RegExp on the same strings

    @property
    def reference(self) -> Callable[..., np.ndarray]:
        return next(iter(self.baselines.values()))
