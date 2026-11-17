"""Export each experiment's ONNX model + inputs + reference output for the browser benchmark page.

Writes build/web/<name>/{model.onnx, in_<i>.bin, ref.bin} and build/web/manifest.json.
"""

import importlib
import json
import shutil
import sys

import numpy as np

from lab import harness

WEB = harness.BUILD / "web"
EXPERIMENTS = sys.argv[1:] or [
    "nbody",
    "montecarlo",
    "groupby",
    "graphs",
    "regex_dfa",
    "levenshtein",
]
DT = {
    np.dtype("int32"): "int32",
    np.dtype("float32"): "float32",
    np.dtype("int64"): "int64",
    np.dtype("bool"): "bool",
    np.dtype("float64"): "float64",
}


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    manifest = (
        json.loads((WEB / "manifest.json").read_text())
        if (WEB / "manifest.json").exists()
        else {}
    )
    for spec in EXPERIMENTS:
        exp, _, entry = spec.partition(":")
        entry = entry or "main"
        captured = []
        mod = importlib.import_module(f"experiments.{exp}")
        mod.run_experiment = lambda *a, **k: captured.append((a, k))
        getattr(mod, entry)()
        for args, kw in captured:
            name, module, inputs, names = args[:4]
            baselines = kw["baselines"]
            d = WEB / name
            d.mkdir(exist_ok=True)
            harness.export(module, inputs, d / "model.onnx", names)
            ref = np.asarray(next(iter(baselines.values()))(*inputs))
            ref = ref.astype(np.float32) if ref.dtype == np.float64 else ref
            ins = []
            for i, (n, x) in enumerate(zip(names, inputs, strict=True)):
                x = np.ascontiguousarray(x)
                x.tofile(d / f"in_{i}.bin")
                ins.append(
                    dict(
                        name=n,
                        dtype=DT[x.dtype],
                        shape=list(x.shape),
                        file=f"in_{i}.bin",
                    )
                )
            ref.tofile(d / "ref.bin")
            manifest[name] = dict(
                inputs=ins,
                ref=dict(dtype=DT[ref.dtype], shape=list(ref.shape), file="ref.bin"),
                size=kw.get("size_note", ""),
                stochastic=name.startswith("montecarlo"),
                rel_to_max=name.startswith("nbody"),
            )
            print(
                f"exported {name}: {sum((d / i['file']).stat().st_size for i in ins) / 1e6:.1f} MB inputs"
            )
    (WEB / "manifest.json").write_text(json.dumps(manifest, indent=1))
    shutil.copy(harness.ROOT / "web" / "bench.html", WEB / "index.html")


if __name__ == "__main__":
    main()
