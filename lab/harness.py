"""Export a case to ONNX, run it on every runtime available, check each answer and time it.

Every runner is timed the same way (median of N runs after warm-up) and checked against the case's
reference baseline. One JSON line per runner goes to build/results.jsonl.
"""

from __future__ import annotations

import json
import logging
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

from lab.case import Case
from lab.graph import IR_VERSION

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
RESULTS = BUILD / "results.jsonl"

# The exporter logs a warning per missing torchvision op on every export.
logging.getLogger("torch.onnx").setLevel(logging.ERROR)

PROVIDERS = ("CPUExecutionProvider", "CoreMLExecutionProvider", "CUDAExecutionProvider")


def timeit(fn: Callable[[], object], warmup: int = 2, reps: int = 7, budget_s: float = 20.0) -> float:
    """Median wall time in ms. Stops early once ``budget_s`` is spent, so slow baselines don't stall a run."""
    for _ in range(warmup):
        fn()
    times: list[float] = []
    start = time.perf_counter()
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
        if time.perf_counter() - start > budget_s:
            break
    return statistics.median(times)


def save_onnx(case: Case, path: pathlib.Path) -> onnx.ModelProto:
    """Write the case's model to ``path``, exporting it from PyTorch first if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(case.model, onnx.ModelProto):
        onnx.save(case.model, path)
        return case.model
    case.model.eval()
    args = tuple(torch.from_numpy(np.ascontiguousarray(x)) for x in case.inputs.values())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = torch.onnx.export(case.model, args, dynamo=True, input_names=list(case.inputs), verbose=False)
    program.optimize()
    program.save(str(path))
    model = onnx.load(str(path))
    model.ir_version = min(model.ir_version, IR_VERSION)
    onnx.save(model, path)
    return model


def onnx_runner(case: Case, path: pathlib.Path, provider: str) -> Callable[[], np.ndarray]:
    options = {"CoreMLExecutionProvider": {"ModelFormat": "MLProgram"}}.get(provider, {})
    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = [(provider, options)] + (["CPUExecutionProvider"] if provider != "CPUExecutionProvider" else [])
    session = ort.InferenceSession(str(path), so, providers=providers)

    def run() -> np.ndarray:
        feeds = dict(case.inputs)
        for _ in range(case.loop.calls if case.loop else 1):
            outs = session.run(None, feeds)
            if case.loop:
                feeds.update({name: outs[i] for name, i in case.loop.feedback.items()})
        return outs[0]

    return run


def torch_runner(case: Case, device: str) -> Callable[[], np.ndarray]:
    module = case.model.to(device).eval()
    names = list(case.inputs)
    tensors = [torch.from_numpy(np.ascontiguousarray(x)).to(device) for x in case.inputs.values()]

    def run() -> np.ndarray:
        args = list(tensors)
        with torch.no_grad():
            for _ in range(case.loop.calls if case.loop else 1):
                out = module(*args)
                if case.loop:
                    for name, i in case.loop.feedback.items():
                        args[names.index(name)] = out[i]
            first = out[0] if isinstance(out, tuple) else out
            if device == "mps":
                torch.mps.synchronize()
            return first.cpu().numpy()

    return run


def run_case(case: Case, reps: int = 7, torch_devices: tuple[str, ...] = ("mps", "cuda")) -> list[dict]:
    path = BUILD / f"{case.name}.onnx"
    model = save_onnx(case, path)
    ops = sorted({n.op_type for n in model.graph.node})
    print(f"\n{case.name}  [{case.size}]  {len(model.graph.node)} nodes, {path.stat().st_size / 1e3:.0f} kB")
    print(f"  ops: {', '.join(ops)}")
    if case.known_bad:
        print(f"  note: {case.known_bad}")

    inputs = list(case.inputs.values())
    runners: dict[str, Callable[[], np.ndarray]] = {
        name: (lambda fn=fn: fn(*inputs)) for name, fn in case.baselines.items()
    }
    for provider in PROVIDERS:
        if provider in ort.get_available_providers():
            runners[f"onnx:{provider.removesuffix('ExecutionProvider')}"] = onnx_runner(case, path, provider)
    if isinstance(case.model, torch.nn.Module):
        for device in torch_devices:
            if (device == "mps" and torch.backends.mps.is_available()) or (
                device == "cuda" and torch.cuda.is_available()
            ):
                runners[f"torch:{device}"] = torch_runner(case, device)

    reference = None
    rows = []
    for runner, fn in runners.items():
        try:
            out = np.asarray(fn())
            correct = True if reference is None else case.tolerance.check(out, reference)
            reference = out if reference is None else reference
            ms = timeit(fn, reps=reps)
        except Exception as e:  # noqa: BLE001 - a runtime refusing a model is a result, not a crash
            print(f"  {runner:24s} error: {str(e)[:160]}")
            continue
        rows.append(
            {
                "case": case.name,
                "runner": runner,
                "ms": ms,
                "correct": correct,
                "size": case.size,
                "machine": f"{platform.system()} {platform.machine()}",
                "onnxruntime": ort.__version__,
                "ts": time.time(),
            }
        )

    base = rows[0]["ms"] if rows else float("nan")
    for r in rows:
        status = "ok" if r["correct"] else "WRONG"
        print(f"  {r['runner']:24s} {r['ms']:10.2f} ms  {base / r['ms']:7.2f}x  {status}")
    BUILD.mkdir(exist_ok=True)
    with RESULTS.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return rows
