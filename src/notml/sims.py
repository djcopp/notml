"""Simulations as ONNX graphs. Each model advances the state by ``steps`` and returns the new state, so a
page can call it once per animation frame and feed the output straight back in.

All three run on a torus: the grid (or space) wraps at the edges.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto

from notml.graph import Graph


def _wrap_pad(g: Graph, x: str, k: int) -> str:
    """Pad the last two axes of an NCHW tensor by ``k`` cells, copying from the opposite edge."""
    i64 = lambda v: g.const(np.array(v, np.int64))  # noqa: E731
    for axis in (2, 3):
        head = g.op("Slice", x, i64([0]), i64([k]), i64([axis]))
        tail = g.op("Slice", x, i64([-k]), i64([np.iinfo(np.int64).max]), i64([axis]))
        x = g.op("Concat", tail, x, head, axis=axis)
    return x


def life(height: int, width: int, steps: int = 8) -> onnx.ModelProto:
    """Conway's Game of Life. Input and output ``cells``: float32 ``[1, 1, H, W]``, 1 alive and 0 dead.
    Also outputs ``image``: uint8 ``[H, W]``, 255 for live cells, ready to draw.

    The rules fold into one convolution. With v = neighbours + cell / 2, a cell is alive next step exactly
    when 2.5 <= v <= 3.5, so the kernel computes u = 2v - 6 (centre weight 1, neighbours 2, bias -6) and
    the next state is clip(2 - |u|, 0, 1). That is three passes per step instead of eight comparisons.
    """
    g = Graph("life")
    x = g.input("cells", np.float32, [1, 1, height, width])
    kernel = np.full((1, 1, 3, 3), 2.0, np.float32)
    kernel[0, 0, 1, 1] = 1.0
    k, bias = g.const(kernel), g.const(np.array([-6.0], np.float32))
    two, zero, one = (g.const(np.float32(v)) for v in (2, 0, 1))
    for _ in range(steps):
        u = g.op("Conv", _wrap_pad(g, x, 1), k, bias, pads=[0, 0, 0, 0])
        x = g.op("Clip", g.op("Sub", two, g.op("Abs", u)), zero, one)
    g.output(x, "cells_out", np.float32, [1, 1, height, width])
    flat = g.op("Reshape", x, g.const(np.array([height, width], np.int64)))
    image = g.op("Cast", g.op("Mul", flat, g.const(np.float32(255))), to=TensorProto.UINT8)
    g.output(image, "image", np.uint8, [height, width])
    return g.model()


# Lenia parameters close to the "Orbium" creature (Chan, 2019): kernel radius 13, growth centre 0.15.
LENIA = {"radius": 13, "mu": 0.15, "sigma": 0.017, "dt": 0.1}


def lenia_kernel(radius: int) -> np.ndarray:
    """Smooth ring: exp(4 - 1 / (r (1 - r))) for r in (0, 1), normalised to sum to 1."""
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    r = np.sqrt(x**2 + y**2) / radius
    inside = (r > 0) & (r < 1)
    k = np.zeros_like(r)
    k[inside] = np.exp(4 - 1 / (r[inside] * (1 - r[inside])))
    return (k / k.sum()).astype(np.float32)


def lenia(height: int, width: int, steps: int = 4, **params) -> onnx.ModelProto:
    """Lenia, a continuous Game of Life. Input and output ``world``: float32 ``[1, 1, H, W]`` in [0, 1].
    Each step convolves with a 27 x 27 ring kernel (729 multiply-adds per cell) and applies a smooth
    growth function. Also outputs ``image``: uint8 ``[H, W]``."""
    p = {**LENIA, **params}
    g = Graph("lenia")
    a = g.input("world", np.float32, [1, 1, height, width])
    k = g.const(lenia_kernel(p["radius"])[None, None])
    f32 = lambda v: g.const(np.float32(v))  # noqa: E731
    for _ in range(steps):
        u = g.op("Conv", _wrap_pad(g, a, p["radius"]), k, pads=[0, 0, 0, 0])
        z = g.op("Div", g.op("Sub", u, f32(p["mu"])), f32(p["sigma"]))
        growth = g.op("Sub", g.op("Mul", f32(2), g.op("Exp", g.op("Mul", f32(-0.5), g.op("Mul", z, z)))), f32(1))
        a = g.op("Clip", g.op("Add", a, g.op("Mul", f32(p["dt"]), growth)), f32(0), f32(1))
    g.output(a, "world_out", np.float32, [1, 1, height, width])
    flat = g.op("Reshape", a, g.const(np.array([height, width], np.int64)))
    g.output(g.op("Cast", g.op("Mul", flat, f32(255)), to=TensorProto.UINT8), "image", np.uint8, [height, width])
    return g.model()


# Gray-Scott parameters that give self-replicating spots ("mitosis"); try F=0.037, k=0.06 for labyrinths.
GRAY_SCOTT = {"feed": 0.0367, "kill": 0.0649, "du": 1.0, "dv": 0.5}
LAPLACIAN = np.array([[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]], np.float32)


def gray_scott(height: int, width: int, steps: int = 16, feed: float | None = None, kill: float | None = None):
    """Gray-Scott reaction-diffusion. Input and output ``uv``: float32 ``[1, 2, H, W]`` (the two chemical
    concentrations). Also outputs ``image``: uint8 ``[H, W]``, the concentration of the second chemical."""
    p = {**GRAY_SCOTT, **({"feed": feed} if feed is not None else {}), **({"kill": kill} if kill is not None else {})}
    g = Graph("gray_scott")
    uv = g.input("uv", np.float32, [1, 2, height, width])
    lap = g.const(np.stack([LAPLACIAN, LAPLACIAN])[:, None])  # depthwise: one kernel per chemical
    diffusion = g.const(np.array([p["du"], p["dv"]], np.float32).reshape(1, 2, 1, 1))
    i64 = lambda v: g.const(np.array(v, np.int64))  # noqa: E731
    f32 = lambda v: g.const(np.float32(v))  # noqa: E731
    for _ in range(steps):
        laplace = g.op("Conv", _wrap_pad(g, uv, 1), lap, group=2, pads=[0, 0, 0, 0])
        u = g.op("Slice", uv, i64([0]), i64([1]), i64([1]))
        v = g.op("Slice", uv, i64([1]), i64([2]), i64([1]))
        uvv = g.op("Mul", u, g.op("Mul", v, v))
        du = g.op("Add", g.op("Sub", f32(0), uvv), g.op("Mul", f32(p["feed"]), g.op("Sub", f32(1), u)))
        dv = g.op("Sub", uvv, g.op("Mul", f32(p["feed"] + p["kill"]), v))
        react = g.op("Concat", du, dv, axis=1)
        uv = g.op("Add", uv, g.op("Add", g.op("Mul", diffusion, laplace), react))
    g.output(uv, "uv_out", np.float32, [1, 2, height, width])
    v = g.op("Reshape", g.op("Slice", uv, i64([1]), i64([2]), i64([1])), i64([height, width]))
    image = g.op("Cast", g.op("Clip", g.op("Mul", v, f32(3 * 255)), f32(0), f32(255)), to=TensorProto.UINT8)
    g.output(image, "image", np.uint8, [height, width])
    return g.model()


def particle_life(
    n: int,
    types: int = 6,
    steps: int = 1,
    radius: float = 0.08,
    beta: float = 0.3,
    friction: float = 0.707,
    dt: float = 0.02,
    force: float = 10.0,
) -> onnx.ModelProto:
    """Particle Life: every particle attracts or repels every other, with a strength set by their two types.

    Inputs: ``pos`` and ``vel`` float32 ``[n, 2]`` in a unit torus, ``kind`` float32 ``[n, types]`` (one-hot)
    and ``rules`` float32 ``[types, types]`` (attraction between types, -1 to 1). Outputs: new ``pos``, ``vel``.
    Every step is O(n²): all pairs interact, which is exactly the shape of work a GPU is good at.
    """
    g = Graph("particle_life")
    pos = g.input("pos", np.float32, [n, 2])
    vel = g.input("vel", np.float32, [n, 2])
    kind = g.input("kind", np.float32, [n, types])
    rules = g.input("rules", np.float32, [types, types])
    i64 = lambda v: g.const(np.array(v, np.int64))  # noqa: E731
    f32 = lambda v: g.const(np.float32(v))  # noqa: E731
    # attraction[i, j] = rules[kind_i, kind_j]
    attraction = g.op("MatMul", g.op("MatMul", kind, rules), g.op("Transpose", kind, perm=[1, 0]))
    for _ in range(steps):
        d = g.op("Sub", g.op("Unsqueeze", pos, i64([0])), g.op("Unsqueeze", pos, i64([1])))  # [n, n, 2], j - i
        d = g.op("Sub", d, g.op("Floor", g.op("Add", d, f32(0.5))))  # shortest way round the torus
        r = g.op("Sqrt", g.op("Add", g.op("ReduceSum", g.op("Mul", d, d), i64([2]), keepdims=0), f32(1e-9)))
        x = g.op("Div", r, f32(radius))  # distance in units of the interaction radius
        repel = g.op("Sub", g.op("Div", x, f32(beta)), f32(1))
        bump = g.op(
            "Mul",
            attraction,
            g.op(
                "Sub",
                f32(1),
                g.op("Div", g.op("Abs", g.op("Sub", g.op("Mul", f32(2), x), f32(1 + beta))), f32(1 - beta)),
            ),
        )
        strength = g.op(
            "Where", g.op("Less", x, f32(beta)), repel, g.op("Where", g.op("Less", x, f32(1)), bump, f32(0))
        )
        direction = g.op("Div", d, g.op("Unsqueeze", r, i64([2])))
        acc = g.op("ReduceSum", g.op("Mul", direction, g.op("Unsqueeze", strength, i64([2]))), i64([1]), keepdims=0)
        acc = g.op("Mul", acc, f32(force * radius))
        vel = g.op("Add", g.op("Mul", vel, f32(friction)), g.op("Mul", acc, f32(dt)))
        pos = g.op("Add", pos, g.op("Mul", vel, f32(dt)))
        pos = g.op("Sub", pos, g.op("Floor", pos))  # wrap into [0, 1)
    g.output(pos, "pos_out", np.float32, [n, 2])
    g.output(vel, "vel_out", np.float32, [n, 2])
    return g.model()
