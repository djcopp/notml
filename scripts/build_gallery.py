"""Build the ONNX models the gallery page loads.

usage: uv run python scripts/build_gallery.py [out_dir]   (default: gallery/models)
"""

import json
import pathlib
import sys

import onnx

from notml import sims

SIMS = {
    "life": {"size": 2048, "steps": 4, "build": lambda c: sims.life(c["size"], c["size"], c["steps"])},
    "grayscott": {"size": 768, "steps": 24, "build": lambda c: sims.gray_scott(c["size"], c["size"], c["steps"])},
    "particles": {
        "n": 4096,
        "types": 6,
        "steps": 1,
        "build": lambda c: sims.particle_life(c["n"], c["types"], c["steps"]),
    },
}

if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "gallery/models")
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name, config in SIMS.items():
        model = config["build"](config)
        onnx.save(model, out / f"{name}.onnx")
        meta[name] = {k: v for k, v in config.items() if k != "build"} | {
            "bytes": (out / f"{name}.onnx").stat().st_size
        }
        print(f"{name}: {meta[name]}")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
