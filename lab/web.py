"""Bundle cases for the browser benchmark: build/web/<case>/{model.onnx, in_<i>.bin, ref.bin} plus manifest.json."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil

import numpy as np

from lab.case import Case
from lab.harness import BUILD, ROOT, save_onnx

WEB = BUILD / "web"


def _dtype(x: np.ndarray) -> str:
    return "bool" if x.dtype == np.bool_ else str(x.dtype)


def export(cases: list[Case], out: pathlib.Path = WEB) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for case in cases:
        folder = out / case.name
        folder.mkdir(exist_ok=True)
        save_onnx(case, folder / "model.onnx")
        inputs = []
        for i, (name, x) in enumerate(case.inputs.items()):
            x = np.ascontiguousarray(x)
            x.tofile(folder / f"in_{i}.bin")
            inputs.append({"name": name, "dtype": _dtype(x), "shape": list(x.shape), "file": f"in_{i}.bin"})
        ref = np.asarray(case.reference(*case.inputs.values()))
        ref = ref.astype(np.float32) if ref.dtype == np.float64 else ref
        ref.tofile(folder / "ref.bin")
        manifest[case.name] = {
            "size": case.size,
            "inputs": inputs,
            "ref": {"dtype": _dtype(ref), "shape": list(ref.shape), "file": "ref.bin"},
            "tolerance": dataclasses.asdict(case.tolerance),
            "loop": dataclasses.asdict(case.loop) if case.loop else None,
        }
        mb = sum((folder / i["file"]).stat().st_size for i in inputs) / 1e6
        print(f"exported {case.name} ({mb:.1f} MB of inputs)")
    manifest_path.write_text(json.dumps(manifest, indent=1))
    shutil.copy(ROOT / "web" / "bench.html", out / "index.html")
