"""Shared benchmark harness for onnx-lab experiments.

An experiment provides:
  * a torch.nn.Module computing the workload (exported to ONNX)
  * example numpy inputs
  * baselines: {name: callable(*numpy_inputs) -> numpy output}
  * an equivalence check

The harness exports the module to ONNX, runs it on each available onnxruntime execution provider,
times everything with the same method (median of N after warm-up), checks outputs agree with the first
baseline, and appends a row per runner to build/results.jsonl.
"""

from __future__ import annotations

import json
import pathlib
import platform
import statistics
import time
import warnings
from collections.abc import Callable

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)
RESULTS = BUILD / "results.jsonl"


def timeit(
    fn: Callable[[], object], warmup: int = 2, reps: int = 7, budget_s: float = 20.0
) -> float:
    """Median wall time in ms. Stops early if the budget is exhausted (slow pure-Python baselines)."""
    for _ in range(warmup):
        fn()
    times = []
    t_start = time.perf_counter()
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
        if time.perf_counter() - t_start > budget_s and len(times) >= 1:
            break
    return statistics.median(times)


def export(
    module: torch.nn.Module, inputs: tuple, path: pathlib.Path, input_names: list[str]
) -> onnx.ModelProto:
    module.eval()
    tins = tuple(torch.from_numpy(np.ascontiguousarray(x)) for x in inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prog = torch.onnx.export(
            module, tins, dynamo=True, input_names=input_names, verbose=False
        )
    prog.optimize()
    prog.save(str(path))
    m = onnx.load(str(path))
    return m


def ort_runner(path: pathlib.Path, provider: str, input_names: list[str]):
    opts = {"CoreMLExecutionProvider": {"ModelFormat": "MLProgram"}}.get(provider, {})
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(
        str(path), so, providers=[(provider, opts), "CPUExecutionProvider"]
    )

    def run(*xs):
        return sess.run(None, dict(zip(input_names, xs, strict=True)))[0]

    return run


def torch_runner(module: torch.nn.Module, device: str):
    m = module.to(device).eval()

    def run(*xs):
        with torch.no_grad():
            out = m(*(torch.from_numpy(np.ascontiguousarray(x)).to(device) for x in xs))
            if device == "mps":
                torch.mps.synchronize()
            return out.cpu().numpy()

    return run


def run_experiment(
    name: str,
    module: torch.nn.Module,
    inputs: tuple,
    input_names: list[str],
    baselines: dict[str, Callable],
    check: Callable[[np.ndarray, np.ndarray], bool],
    size_note: str,
    torch_devices: tuple[str, ...] = ("mps",),
    providers: tuple[str, ...] = ("CPUExecutionProvider", "CoreMLExecutionProvider"),
    reps: int = 7,
) -> list[dict]:
    path = BUILD / f"{name}.onnx"
    t0 = time.perf_counter()
    model = export(module, inputs, path, input_names)
    export_s = time.perf_counter() - t0
    n_nodes = len(model.graph.node)
    ops = sorted({n.op_type for n in model.graph.node})
    print(
        f"\n=== {name}  [{size_note}]  onnx: {n_nodes} nodes, {path.stat().st_size / 1e3:.0f} kB, "
        f"export {export_s:.1f}s\n    ops: {', '.join(ops)}"
    )

    runners: dict[str, Callable] = dict(baselines)
    for p in providers:
        if p in ort.get_available_providers():
            try:
                runners[f"onnx:{p.replace('ExecutionProvider', '')}"] = ort_runner(
                    path, p, input_names
                )
            except Exception as e:  # noqa: BLE001
                print(f"    {p} failed to load: {str(e)[:150]}")
    for d in torch_devices:
        if d == "mps" and not torch.backends.mps.is_available():
            continue
        runners[f"torch:{d}"] = torch_runner(module, d)

    ref_name = next(iter(baselines))
    ref = None
    rows = []
    for rname, fn in runners.items():
        try:
            out = fn(*inputs)
            ok = True if ref is None else bool(check(np.asarray(out), ref))
            if ref is None:
                ref = np.asarray(out)
            ms = timeit(lambda fn=fn: fn(*inputs), reps=reps)
        except Exception as e:  # noqa: BLE001
            print(f"    {rname:22s} ERROR {str(e)[:160]}")
            continue
        rows.append(
            dict(
                experiment=name,
                runner=rname,
                ms=ms,
                correct=ok,
                size=size_note,
                nodes=n_nodes,
                machine=platform.machine(),
                ts=time.time(),
            )
        )
    base_ms = rows[0]["ms"] if rows else None
    for r in rows:
        speed = base_ms / r["ms"] if base_ms else float("nan")
        print(
            f"    {r['runner']:22s} {r['ms']:10.2f} ms   {speed:7.2f}x vs {ref_name}   "
            f"{'ok' if r['correct'] else 'MISMATCH'}"
        )
    with RESULTS.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows
