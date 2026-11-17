"""Measure steps per second for every simulation and engine in the gallery, in headless Chromium.

usage: uv run python scripts/bench_gallery.py [--json docs/results/gallery.json] [--seconds 6]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lab.serve import serve_in_background  # noqa: E402

SIMS = ["life", "grayscott", "lenia", "particles"]
ENGINES = ["webgpu", "wasm", "js"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    parser.add_argument("--seconds", type=float, default=6)
    parser.add_argument("--retry", type=int, default=2, help="re-measure a run that reported nothing")
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_gallery.py")], check=True, capture_output=True)
    server = serve_in_background(ROOT / "gallery")
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=os.environ.get("CHROMIUM_PATH") or None, args=["--enable-unsafe-webgpu"]
        )
        env = None
        for sim in SIMS:
            for engine in ENGINES:
                # a fresh page per run: onnxruntime-web 1.30 can't run WASM after WebGPU in the same page
                page = browser.new_page(viewport={"width": 1200, "height": 900})
                page.on("pageerror", lambda e: print("  page error:", str(e)[:300]))
                page.goto(f"http://127.0.0.1:{server.server_port}/index.html?sim={sim}&engine={engine}")
                env = env or page.evaluate(
                    "({threads: navigator.hardwareConcurrency, isolated: self.crossOriginIsolated})"
                )
                last = None
                for _ in range(1 + args.retry):  # the first load may reload itself to install the service worker
                    page.wait_for_timeout(args.seconds * 1000)  # the page reports a fresh rate every second
                    last = page.evaluate("window.benchLast")
                    if last:
                        break
                results.append(last or {"sim": sim, "engine": engine, "error": "no measurement"})
                print(f"  {sim:10s} {engine:7s} {last}", flush=True)
                page.close()
        browser.close()
    server.shutdown()
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps({"env": env, "results": results}, indent=1))


if __name__ == "__main__":
    main()
