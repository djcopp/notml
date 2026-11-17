"""Run the browser bench in a scripted Chromium so browser numbers are repeatable.

The page is served with cross-origin isolation (multithreaded WASM) and run in a foreground headless tab,
which avoids the throttling a hidden or background tab gets. Results go to build/browser_results.json.
"""

from __future__ import annotations

import json
import os
import time

from playwright.sync_api import sync_playwright

from lab.harness import BUILD
from lab.serve import serve_in_background
from lab.web import WEB

CHROMIUM_FLAGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--disable-background-timer-throttling"]


def _executable() -> str | None:
    """CHROMIUM_PATH if set, otherwise the browser Playwright installed (``playwright install chromium``)."""
    return os.environ.get("CHROMIUM_PATH") or None


def run(ort_version: str = "1.30.0", only: list[str] | None = None, reps: int = 7, timeout_s: int = 1800) -> dict:
    server = serve_in_background(WEB)
    url = f"http://127.0.0.1:{server.server_port}/index.html?ort={ort_version}&reps={reps}"
    if only:
        url += "&only=" + ",".join(only)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=_executable(), headless=True, args=CHROMIUM_FLAGS)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_function("window.benchResults !== undefined", timeout=timeout_s * 1000, polling=1000)
        report = page.evaluate("window.benchResults")
        browser.close()
    server.shutdown()
    report["ts"] = time.time()
    _print(report)
    out = BUILD / "browser_results.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")
    return report


def _print(report: dict) -> None:
    if "error" in report:
        raise RuntimeError(report["error"])
    env = report["env"]
    print(
        f"onnxruntime-web {env['ort']}, WebGPU: {env['webgpu']}, threads: {env['threads']}, isolated: {env['isolated']}"
    )
    for r in report["results"]:
        if "error" in r:
            print(f"  {r['case']:22s} {r['backend']:7s} error: {r['error'][:100]}")
        else:
            print(f"  {r['case']:22s} {r['backend']:7s} {r['ms']:10.2f} ms  {'ok' if r['correct'] else 'WRONG'}")
