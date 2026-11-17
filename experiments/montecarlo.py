"""Finance: price an arithmetic-average Asian call option by Monte Carlo under geometric Brownian motion.

Three ways to get the random numbers inside the graph:

torch     ``torch.randn`` exports as RandomNormalLike. Correct, but GPU providers run it on the CPU, so the
          whole model ends up no faster than numpy.
pcg       A counter-based generator: every normal is a pure function of its index (PCG hash -> two
          uniforms -> Box-Muller). Statistically sound and fully elementwise, but it needs BitShift and
          BitwiseXor, which onnxruntime-web's WebGPU provider doesn't implement yet.
muldiv    The same idea with only multiply, add and divide, which WebGPU does have. It runs entirely on
          the GPU, and it is biased: prices come out ~14% low. Kept as a negative result.
"""

from typing import Literal

import numpy as np
import onnx
import torch
from onnx import TensorProto

from lab.case import Case, Tolerance
from lab.graph import Graph

S0, STRIKE, RATE, SIGMA, TENOR = 100.0, 100.0, 0.03, 0.25, 1.0


def gbm_terms(steps: int) -> tuple[float, float]:
    dt = TENOR / steps
    return (RATE - 0.5 * SIGMA**2) * dt, SIGMA * np.sqrt(dt)


class AsianCall(torch.nn.Module):
    def __init__(self, paths: int, steps: int):
        super().__init__()
        self.paths, self.steps = paths, steps
        self.drift, self.vol = gbm_terms(steps)

    def forward(self, s0):
        z = torch.randn(self.paths, self.steps, device=s0.device)
        log_path = torch.cumsum(self.drift + self.vol * z, dim=1)
        average = (s0 * torch.exp(log_path)).mean(dim=1)
        return (torch.clamp(average - STRIKE, min=0.0).mean() * np.exp(-RATE * TENOR)).reshape(1)


def numpy_price(s0, *_, paths: int, steps: int, seed: int = 0):
    drift, vol = gbm_terms(steps)
    z = np.random.default_rng(seed).standard_normal((paths, steps), dtype=np.float32)
    average = (s0 * np.exp(np.cumsum(drift + vol * z, axis=1))).mean(axis=1)
    return np.array([np.maximum(average - STRIKE, 0).mean() * np.exp(-RATE * TENOR)], np.float32)


def pcg_hash(g: Graph, x: str) -> str:
    """PCG-RXS-M-XS 32-bit hash (Jarzynski & Olano, 2020)."""

    def u32(v):
        return g.const(v, np.uint32)

    state = g.op("Add", g.op("Mul", x, u32(747796405)), u32(2891336453))
    shift = g.op("Add", g.op("BitShift", state, u32(28), direction="RIGHT"), u32(4))
    word = g.op("Mul", g.op("BitwiseXor", g.op("BitShift", state, shift, direction="RIGHT"), state), u32(277803737))
    return g.op("BitwiseXor", g.op("BitShift", word, u32(22), direction="RIGHT"), word)


def muldiv_hash(g: Graph, x: str, rounds: int = 4) -> str:
    """Multiply-add mixing where unsigned Div stands in for a right shift. No bitwise ops."""

    def u32(v):
        return g.const(v, np.uint32)

    for r, mult in enumerate([0x7FEB352D, 0x846CA68B, 0x9E3779B1, 0x85EBCA77][:rounds]):
        x = g.op("Add", x, g.op("Div", x, u32(1 << (16 - r))))
        x = g.op("Mul", x, u32(mult))
        x = g.op("Add", x, g.op("Div", x, u32(1 << 15)))
    return x


def asian_call_graph(
    paths: int, steps: int, rng: Literal["pcg", "muldiv"], output: Literal["price", "normals"] = "price"
) -> onnx.ModelProto:
    """The Asian option written directly in ONNX with a hash-based generator."""
    g = Graph(f"asian_call_{rng}")
    s0 = g.input("s0", np.float32, [1])
    seed = g.input("seed", np.int32, [1])

    def f32(v):
        return g.const(v, np.float32)

    # Element (i, j) uses counters c and c + 1 with c = 2 * (i * steps + j) + seed. Built from two broadcast
    # constants rather than Range, which WebGPU doesn't implement either.
    rows = g.const((np.arange(paths, dtype=np.uint32) * np.uint32(2 * steps))[:, None])
    cols = g.const((np.arange(steps, dtype=np.uint32) * np.uint32(2))[None, :])
    counter = g.op("Add", g.op("Add", rows, cols), g.op("Cast", seed, to=TensorProto.UINT32))
    hash_fn = pcg_hash if rng == "pcg" else muldiv_hash

    def uniform(x: str) -> str:  # top 24 bits -> (0, 1)
        top = g.op("Div", hash_fn(g, x), g.const(256, np.uint32))
        return g.op("Mul", g.op("Add", g.op("Cast", top, to=TensorProto.FLOAT), f32(0.5)), f32(2.0**-24))

    u1 = uniform(counter)
    u2 = uniform(g.op("Add", counter, g.const(1, np.uint32)))
    radius = g.op("Sqrt", g.op("Mul", f32(-2.0), g.op("Log", u1)))
    z = g.op("Mul", radius, g.op("Cos", g.op("Mul", u2, f32(2 * np.pi))))
    if output == "normals":
        g.output(z, "normals", np.float32, [paths, steps])
        return g.model()

    drift, vol = gbm_terms(steps)
    log_path = g.op("CumSum", g.op("Add", g.op("Mul", z, f32(vol)), f32(drift)), g.const(1, np.int32))
    average = g.op("ReduceMean", g.op("Mul", g.op("Exp", log_path), s0), g.const([1], np.int64), keepdims=0)
    payoff = g.op("Relu", g.op("Sub", average, f32(STRIKE)))
    price = g.op("Mul", g.op("ReduceMean", payoff, keepdims=1), f32(np.exp(-RATE * TENOR)))
    g.output(price, "price", np.float32, [1])
    return g.model()


def normal_quality(rng: Literal["pcg", "muldiv"], paths: int = 4096, steps: int = 64) -> dict[str, float]:
    """Moments and lag-1 autocorrelation of the normals a generator produces."""
    import onnxruntime as ort

    model = asian_call_graph(paths, steps, rng, output="normals")
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    z = session.run(None, {"s0": np.array([S0], np.float32), "seed": np.array([7], np.int32)})[0].ravel()
    c = z - z.mean()
    return {
        "mean": float(z.mean()),
        "std": float(z.std()),
        "skew": float((c**3).mean() / z.std() ** 3),
        "kurtosis": float((c**4).mean() / z.std() ** 4),
        "lag1_autocorr": float(np.corrcoef(z[:-1], z[1:])[0, 1]),
    }


def cases(small: bool = False) -> list[Case]:
    paths, steps = (20_000, 64) if small else (1_000_000, 64)
    # Small runs have more Monte Carlo noise (about 1.4% between two estimates), so allow more slack.
    tolerance = Tolerance("scalar_rel", rtol=0.05 if small else 0.01)
    s0 = np.array([S0], np.float32)
    seed = np.array([12345], np.int32)
    size = f"{paths:,} paths x {steps} steps"

    def reference(*args):
        return numpy_price(args[0], paths=paths, steps=steps)

    return [
        Case("montecarlo_torch_rng", AsianCall(paths, steps), {"s0": s0}, {"numpy": reference}, tolerance, size),
        Case(
            "montecarlo_pcg",
            asian_call_graph(paths, steps, "pcg"),
            {"s0": s0, "seed": seed},
            {"numpy": reference},
            tolerance,
            size,
        ),
        Case(
            "montecarlo_muldiv",
            asian_call_graph(paths, steps, "muldiv"),
            {"s0": s0, "seed": seed},
            {"numpy": reference},
            tolerance,
            size,
            known_bad="multiply/divide hash is biased; prices come out about 14% low",
        ),
    ]
