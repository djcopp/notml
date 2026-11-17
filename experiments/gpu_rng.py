"""Monte Carlo with a GPU-native, counter-based RNG built from standard ONNX integer ops.

RandomNormalLike runs on CPU in CoreML/WebGPU, which capped Monte Carlo at ~1x. Here each element's
normal is a pure function of its index: PCG-hash(index) -> two uniforms -> Box-Muller. Every op is
elementwise, so the whole graph can stay on the accelerator. Same Asian-option workload as montecarlo.py.
"""

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh

from experiments.montecarlo import (
    DRIFT,
    K_STRIKE,
    PATHS,
    S0,
    STEPS,
    VOL,
    R,
    numpy_price,
)
from experiments.montecarlo import T as TENOR
from lab import harness

U32 = T.UINT32


def pcg_hash(nodes, inits, x, tag):
    """PCG-RXS-M-XS 32-bit hash (Jarzynski & Olano 2020), uint32 -> uint32."""

    def c(name, v):
        inits.append(nh.from_array(np.array(v, np.uint32), f"{tag}_{name}"))
        return f"{tag}_{name}"

    n = lambda op, *ins, **kw: (
        nodes.append(h.make_node(op, list(ins), [f"{tag}_{len(nodes)}"], **kw)),
        f"{tag}_{len(nodes) - 1}",
    )[1]
    state = n("Add", n("Mul", x, c("m1", 747796405)), c("a1", 2891336453))
    shift = n(
        "Add", n("BitShift", state, c("s28", 28), direction="RIGHT"), c("four", 4)
    )
    word = n(
        "Mul",
        n("BitwiseXor", n("BitShift", state, shift, direction="RIGHT"), state),
        c("m2", 277803737),
    )
    return n("BitwiseXor", n("BitShift", word, c("s22", 22), direction="RIGHT"), word)


def build_model(paths=PATHS, steps=STEPS):
    nodes, inits = [], []
    n = lambda op, *ins, out=None, **kw: (
        nodes.append(h.make_node(op, list(ins), [out or f"n{len(nodes)}"], **kw)),
        out or f"n{len(nodes) - 1}",
    )[1]
    cst = lambda name, v, dt: (
        inits.append(nh.from_array(np.array(v, dt), name)),
        name,
    )[1]
    count = paths * steps
    # counters: 2*i and 2*i+1 for the two uniforms of element i (seed folded into the counter)
    idx = n(
        "Range",
        cst("zero", 0, np.int32),
        cst("cnt", count, np.int32),
        cst("one", 1, np.int32),
    )
    idx = n(
        "Add",
        n("Mul", n("Cast", idx, to=U32), cst("two", 2, np.uint32)),
        n("Cast", "seed", to=U32),
    )
    idx2 = n("Add", idx, cst("oneu", 1, np.uint32))
    to_unit = lambda hsh, tag: n(
        "Mul",
        n(
            "Add",
            n(
                "Cast",
                n("BitShift", hsh, cst(f"s8{tag}", 8, np.uint32), direction="RIGHT"),
                to=T.FLOAT,
            ),
            cst(f"half{tag}", 0.5, np.float32),
        ),
        cst(f"inv24{tag}", 2.0**-24, np.float32),
    )  # (0,1)
    u1 = to_unit(pcg_hash(nodes, inits, idx, "ha"), "a")
    u2 = to_unit(pcg_hash(nodes, inits, idx2, "hb"), "b")
    radius = n("Sqrt", n("Mul", cst("m2f", -2.0, np.float32), n("Log", u1)))
    z = n("Mul", radius, n("Cos", n("Mul", u2, cst("tau", 2 * np.pi, np.float32))))
    z = n("Reshape", z, cst("shape", [paths, steps], np.int64))
    logp = n(
        "CumSum",
        n(
            "Add",
            n("Mul", z, cst("vol", VOL, np.float32)),
            cst("drift", DRIFT, np.float32),
        ),
        cst("ax1", 1, np.int64),
    )
    avg = n(
        "ReduceMean",
        n("Mul", n("Exp", logp), "s0"),
        cst("axes1", [1], np.int64),
        keepdims=0,
    )
    pay = n("Relu", n("Sub", avg, cst("k", K_STRIKE, np.float32)))
    price = n(
        "Mul",
        n("ReduceMean", pay, keepdims=1),
        cst("disc", np.exp(-R * TENOR), np.float32),
        out="price",
    )
    g = h.make_graph(
        nodes,
        "mc_pcg",
        [
            h.make_tensor_value_info("s0", T.FLOAT, [1]),
            h.make_tensor_value_info("seed", T.INT32, [1]),
        ],
        [h.make_tensor_value_info("price", T.FLOAT, [1])],
        initializer=inits,
    )
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 18)])
    m.ir_version = 9  # onnxruntime-web 1.30 rejects the newer IR version the latest onnx helper stamps
    onnx.checker.check_model(m, full_check=True)
    return m


def main():
    import onnxruntime as ort

    m = build_model()
    path = harness.BUILD / "montecarlo_pcg.onnx"
    onnx.save(m, path)
    # RNG quality sanity check on a small graph's intermediate normals
    small = build_model(paths=1000, steps=1000)
    inputs = (np.array([S0], np.float32), np.array([12345], np.int32))
    print(
        f"montecarlo_pcg: {len(m.graph.node)} nodes, ops: {sorted({nd.op_type for nd in m.graph.node})}"
    )
    for prov in ("CPUExecutionProvider", "CoreMLExecutionProvider"):
        opts = {"ModelFormat": "MLProgram"} if prov.startswith("CoreML") else {}
        s = ort.InferenceSession(
            str(path), providers=[(prov, opts), "CPUExecutionProvider"]
        )
        run = lambda: s.run(None, {"s0": inputs[0], "seed": inputs[1]})[0]
        price = float(run()[0])
        ms = harness.timeit(run)
        print(f"  onnx:{prov[:-17]:8s} price {price:.4f}  {ms:8.1f} ms")
    ref = [float(numpy_price(inputs[0])[0]) for _ in range(3)]
    print(
        f"  numpy reference price {np.mean(ref):.4f} (+/- {np.std(ref):.4f} across runs)  {harness.timeit(lambda: numpy_price(inputs[0])):.1f} ms"
    )
    return m, inputs


if __name__ == "__main__":
    main()


def mix_hash(nodes, inits, x, tag, rounds=3):
    """Hash using only Mul/Add/Div on int32 (all have WebGPU kernels; BitShift/BitwiseXor don't).
    Each round: multiply-add (wrapping) then fold the high half into the low half with x + x // 65536."""

    def c(name, v):
        inits.append(nh.from_array(np.array(v, np.int32), f"{tag}_{name}"))
        return f"{tag}_{name}"

    n = lambda op, *ins, **kw: (
        nodes.append(h.make_node(op, list(ins), [f"{tag}_{len(nodes)}"], **kw)),
        f"{tag}_{len(nodes) - 1}",
    )[1]
    mults = [
        -1640531535,
        1103515245,
        2146121005 - 2**32 if 2146121005 >= 2**31 else 2146121005,
    ]
    for r in range(rounds):
        x = n("Add", n("Mul", x, c(f"m{r}", mults[r % 3])), c(f"a{r}", 12345 + 977 * r))
        x = n("Add", x, n("Div", x, c(f"d{r}", 65536)))
    return x


def build_model_gpu(paths=PATHS, steps=STEPS):
    """Same Asian option, RNG built only from ops with WebGPU kernels; indices from two small broadcast ranges."""
    nodes, inits = [], []
    n = lambda op, *ins, out=None, **kw: (
        nodes.append(h.make_node(op, list(ins), [out or f"n{len(nodes)}"], **kw)),
        out or f"n{len(nodes) - 1}",
    )[1]
    cst = lambda name, v, dt: (
        inits.append(nh.from_array(np.array(v, dt), name)),
        name,
    )[1]
    rows = cst(
        "rows",
        np.arange(paths, dtype=np.int32).reshape(paths, 1) * (2 * steps),
        np.int32,
    )  # [P,1]
    cols = cst(
        "cols", (np.arange(steps, dtype=np.int32) * 2).reshape(1, steps), np.int32
    )  # [1,S]
    idx = n("Add", n("Add", rows, cols), "seed")  # [P,S] on GPU
    two24 = cst("two24", 1 << 24, np.int32)

    def unit(
        x, tag
    ):  # int32 -> (0,1): take 24 bits via Mod (handles negatives with a second Mod)
        m = n("Mod", n("Add", n("Mod", x, two24), two24), two24)
        return n(
            "Mul",
            n("Add", n("Cast", m, to=T.FLOAT), cst(f"half{tag}", 0.5, np.float32)),
            cst(f"inv{tag}", 2.0**-24, np.float32),
        )

    u1 = unit(mix_hash(nodes, inits, idx, "ha"), "a")
    u2 = unit(mix_hash(nodes, inits, n("Add", idx, cst("one", 1, np.int32)), "hb"), "b")
    z = n(
        "Mul",
        n("Sqrt", n("Mul", cst("m2f", -2.0, np.float32), n("Log", u1))),
        n("Cos", n("Mul", u2, cst("tau", 2 * np.pi, np.float32))),
    )
    logp = n(
        "CumSum",
        n(
            "Add",
            n("Mul", z, cst("vol", VOL, np.float32)),
            cst("drift", DRIFT, np.float32),
        ),
        cst("ax1", 1, np.int32),
    )
    avg = n(
        "ReduceMean",
        n("Mul", n("Exp", logp), "s0"),
        cst("axes1", [1], np.int64),
        keepdims=0,
    )
    pay = n("Relu", n("Sub", avg, cst("k", K_STRIKE, np.float32)))
    n(
        "Mul",
        n("ReduceMean", pay, keepdims=1),
        cst("disc", np.exp(-R * TENOR), np.float32),
        out="price",
    )
    g = h.make_graph(
        nodes,
        "mc_gpu_hash",
        [
            h.make_tensor_value_info("s0", T.FLOAT, [1]),
            h.make_tensor_value_info("seed", T.INT32, [1]),
        ],
        [h.make_tensor_value_info("price", T.FLOAT, [1])],
        initializer=inits,
    )
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 18)])
    m.ir_version = 9
    onnx.checker.check_model(m, full_check=True)
    return m


def rng_quality(model_builder):
    """Moments + lag-1 autocorrelation of the normals the hash produces (via a tiny graph that outputs z)."""
    import onnxruntime as ort

    m = model_builder(paths=4096, steps=64)
    zname = next(nd.output[0] for nd in m.graph.node if nd.op_type == "Cos")
    # rebuild z = radius * cos as an extra output by exposing the Mul consuming Cos
    zmul = next(nd for nd in m.graph.node if nd.op_type == "Mul" and zname in nd.input)
    m.graph.output.append(h.make_tensor_value_info(zmul.output[0], T.FLOAT, None))
    s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    z = s.run(
        [zmul.output[0]],
        {"s0": np.array([S0], np.float32), "seed": np.array([7], np.int32)},
    )[0].ravel()
    ac = np.corrcoef(z[:-1], z[1:])[0, 1]
    return dict(
        mean=float(z.mean()),
        std=float(z.std()),
        skew=float(((z - z.mean()) ** 3).mean() / z.std() ** 3),
        kurt=float(((z - z.mean()) ** 4).mean() / z.std() ** 4),
        lag1_autocorr=float(ac),
    )


def mix_hash_u32(nodes, inits, x, tag, rounds=4):
    """uint32 multiply-add-shift mixing with Div (== logical shift for unsigned). No BitShift/BitwiseXor."""

    def c(name, v):
        inits.append(nh.from_array(np.array(v, np.uint32), f"{tag}_{name}"))
        return f"{tag}_{name}"

    n = lambda op, *ins, **kw: (
        nodes.append(h.make_node(op, list(ins), [f"{tag}_{len(nodes)}"], **kw)),
        f"{tag}_{len(nodes) - 1}",
    )[1]
    mults = [0x7FEB352D, 0x846CA68B, 0x9E3779B1, 0x85EBCA77]
    for r in range(rounds):
        x = n("Add", x, n("Div", x, c(f"s{r}a", 1 << (16 - r))))  # x += x >> (16 - r)
        x = n("Mul", x, c(f"m{r}", mults[r]))
        x = n("Add", x, n("Div", x, c(f"s{r}b", 1 << 15)))  # x += x >> 15
    return x


def build_model_gpu_u32(paths=PATHS, steps=STEPS):
    nodes, inits = [], []
    n = lambda op, *ins, out=None, **kw: (
        nodes.append(h.make_node(op, list(ins), [out or f"n{len(nodes)}"], **kw)),
        out or f"n{len(nodes) - 1}",
    )[1]
    cst = lambda name, v, dt: (
        inits.append(nh.from_array(np.array(v, dt), name)),
        name,
    )[1]
    rows = cst(
        "rows",
        (np.arange(paths, dtype=np.uint32) * np.uint32(2 * steps)).reshape(paths, 1),
        np.uint32,
    )
    cols = cst(
        "cols",
        (np.arange(steps, dtype=np.uint32) * np.uint32(2)).reshape(1, steps),
        np.uint32,
    )
    idx = n("Add", n("Add", rows, cols), n("Cast", "seed", to=U32))

    def unit(x, tag):  # top 24 bits -> (0,1)
        return n(
            "Mul",
            n(
                "Add",
                n("Cast", n("Div", x, cst(f"d8{tag}", 256, np.uint32)), to=T.FLOAT),
                cst(f"half{tag}", 0.5, np.float32),
            ),
            cst(f"inv{tag}", 2.0**-24, np.float32),
        )

    u1 = unit(mix_hash_u32(nodes, inits, idx, "ha"), "a")
    u2 = unit(
        mix_hash_u32(nodes, inits, n("Add", idx, cst("one", 1, np.uint32)), "hb"), "b"
    )
    z = n(
        "Mul",
        n("Sqrt", n("Mul", cst("m2f", -2.0, np.float32), n("Log", u1))),
        n("Cos", n("Mul", u2, cst("tau", 2 * np.pi, np.float32))),
    )
    logp = n(
        "CumSum",
        n(
            "Add",
            n("Mul", z, cst("vol", VOL, np.float32)),
            cst("drift", DRIFT, np.float32),
        ),
        cst("ax1", 1, np.int32),
    )
    avg = n(
        "ReduceMean",
        n("Mul", n("Exp", logp), "s0"),
        cst("axes1", [1], np.int64),
        keepdims=0,
    )
    pay = n("Relu", n("Sub", avg, cst("k", K_STRIKE, np.float32)))
    n(
        "Mul",
        n("ReduceMean", pay, keepdims=1),
        cst("disc", np.exp(-R * TENOR), np.float32),
        out="price",
    )
    g = h.make_graph(
        nodes,
        "mc_gpu_hash_u32",
        [
            h.make_tensor_value_info("s0", T.FLOAT, [1]),
            h.make_tensor_value_info("seed", T.INT32, [1]),
        ],
        [h.make_tensor_value_info("price", T.FLOAT, [1])],
        initializer=inits,
    )
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 18)])
    m.ir_version = 9
    onnx.checker.check_model(m, full_check=True)
    return m
