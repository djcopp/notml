"""Draw the README figures from the committed benchmark results in docs/results/.

usage: uv run python scripts/figures.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import svg  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "results"
OUT = ROOT / "docs" / "img"

# Which baseline each native case is measured against: the tool people would reach for first.
NATIVE = [
    ("nbody", "N-body step", "4,096 bodies", "numpy"),
    ("regex_dfa_int32", "Regex validation", "1M strings", "python re"),
    ("vm", "Register VM", "262k programs", "numpy"),
    ("groupby", "GROUP BY sum", "10M rows", "numpy.bincount"),
    ("montecarlo_pcg", "Monte Carlo, PCG RNG", "1M paths", "numpy"),
    ("montecarlo_torch_rng", "Monte Carlo, RandomNormal", "1M paths", "numpy"),
    ("shortest_paths", "All-pairs shortest path", "256 nodes", "scipy floyd_warshall"),
    ("pagerank", "PageRank", "80k edges", "scipy.sparse"),
    ("levenshtein", "Levenshtein", "200k pairs", "rapidfuzz (all cores)"),
]
BROWSER = [
    ("pagerank", "PageRank"),
    ("regex_dfa_int32", "Regex validation, int32"),
    ("nbody", "N-body step"),
    ("vm", "Register VM"),
    ("shortest_paths", "All-pairs shortest path"),
    ("groupby", "GROUP BY sum"),
    ("montecarlo_torch_rng", "Monte Carlo, RandomNormal"),
    ("montecarlo_pcg", "Monte Carlo, PCG RNG"),
    ("levenshtein", "Levenshtein"),
    ("regex_dfa_int64", "Regex validation, int64"),
]


def latest_native() -> dict[tuple[str, str], float]:
    rows = [json.loads(line) for line in (RESULTS / "native.jsonl").read_text().splitlines()]
    out = {}
    for r in rows:  # later lines win
        if r["correct"]:
            out[(r["case"], r["runner"])] = r["ms"]
    return out


def browser_ms() -> dict[tuple[str, str], float]:
    report = json.loads((RESULTS / "browser.json").read_text())
    return {(r["case"], r["backend"]): r["ms"] for r in report["results"] if "ms" in r}


def native_chart(theme: svg.Theme) -> svg.Canvas:
    ms = latest_native()
    rows = []
    for case, label, size, baseline in NATIVE:
        base = ms[(case, baseline)]
        values = {}
        for series, runner in [("onnxruntime CPU", "onnx:CPU"), ("onnxruntime CoreML", "onnx:CoreML")]:
            if (case, runner) in ms:
                values[series] = base / ms[(case, runner)]
        rows.append((label, f"{size} · vs {baseline}", values))
    return svg.speedup_chart(
        rows, ["onnxruntime CPU", "onnxruntime CoreML"], theme, "speed-up over the baseline, log scale", lo=0.01
    )


def browser_chart(theme: svg.Theme) -> svg.Canvas:
    ms = browser_ms()
    rows = []
    for case, label in BROWSER:
        if (case, "webgpu") in ms and (case, "wasm") in ms:
            gpu, cpu = ms[(case, "webgpu")], ms[(case, "wasm")]
            rows.append((label, f"{svg.fmt_ms(gpu)} on WebGPU · {svg.fmt_ms(cpu)} on WASM", {"WebGPU": cpu / gpu}))
    rows.sort(key=lambda r: -r[2]["WebGPU"])
    return svg.speedup_chart(rows, ["WebGPU"], theme, "WebGPU speed-up over 10-thread WASM, log scale", lo=0.1, hi=10)


def int_width_chart(theme: svg.Theme) -> svg.Canvas:
    ms = browser_ms()
    rows = [
        ("int64, WebGPU", ms[("regex_dfa_int64", "webgpu")], 1),
        ("int64, WASM", ms[("regex_dfa_int64", "wasm")], None),
        ("int32, WebGPU", ms[("regex_dfa_int32", "webgpu")], 0),
        ("int32, WASM", ms[("regex_dfa_int32", "wasm")], None),
    ]
    return svg.time_chart(rows, theme, "time to check 1M strings, log scale (shorter is better)", log=True, label_w=150)


GALLERY = [
    ("lenia", "Lenia", "512 × 512, 27 × 27 kernel"),
    ("life", "Game of Life", "2048 × 2048"),
    ("grayscott", "Reaction–diffusion", "768 × 768"),
    ("particles", "Particle Life", "4,096 particles"),
]


def gallery_chart(theme: svg.Theme) -> svg.Canvas:
    report = json.loads((RESULTS / "gallery.json").read_text())
    rate = {(r["sim"], r["engine"]): r["steps_per_s"] for r in report["results"] if "steps_per_s" in r}
    rows = []
    for sim, label, size in GALLERY:
        js = rate[(sim, "js")]
        values = {"ONNX on WASM": rate[(sim, "wasm")] / js, "ONNX on WebGPU": rate[(sim, "webgpu")] / js}
        rows.append((label, f"{size} · plain JS {js:.1f} steps/s", values))
    return svg.speedup_chart(
        rows,
        ["ONNX on WASM", "ONNX on WebGPU"],
        theme,
        "steps per second relative to plain JavaScript, log scale",
        lo=0.1,
        hi=300,
    )


def portability_diagram(theme: svg.Theme) -> svg.Canvas:
    t = theme
    c = svg.Canvas(880, 300, t)
    c.box(0, 40, 180, 70, "PyTorch module", ["exported with torch.onnx"])
    c.box(0, 150, 180, 70, "Hand-written graph", ["onnx.helper, for RNGs"])
    c.box(270, 95, 150, 70, "workload.onnx", ["standard ops only"], accent=True, mono_title=True)
    c.arrow([(180, 75), (225, 75), (225, 120), (268, 120)])
    c.arrow([(180, 185), (225, 185), (225, 140), (268, 140)])
    lanes = [
        ("onnxruntime CPU", "Mac, Linux, Windows", False),
        ("CoreML", "Apple GPU / Neural Engine", False),
        ("WebGPU", "any browser tab", False),
        ("WASM, 10 threads", "any browser tab", False),
        ("CUDA", "not measured yet", True),
    ]
    for i, (name, sub, dashed) in enumerate(lanes):
        y = 4 + i * 58
        c.box(560, y, 220, 48, name, [sub], dashed=dashed, title_size=13)
        c.arrow([(420, 130), (490, 130), (490, y + 24), (558, y + 24)], dash="4 4" if dashed else None)
    c.text(800, 130, "same file,", size=12, color=t.muted)
    c.text(800, 148, "same answer", size=12, color=t.muted)
    return c


def vm_diagram(theme: svg.Theme) -> svg.Canvas:
    t = theme
    c = svg.Canvas(880, 250, t)
    steps = [
        ("fetch", ["one-hot pc × program"]),
        ("decode", ["op, a, b, c"]),
        ("read", ["one-hot register sums"]),
        ("execute", ["every op, then Where"]),
        ("write back", ["masked register update"]),
    ]
    w, gap, y = 150, 32, 40
    for i, (name, lines) in enumerate(steps):
        x = i * (w + gap)
        c.box(x, y, w, 66, name, lines, accent=i == 3)
        if i:
            c.arrow([(x - gap, y + 33), (x - 2, y + 33)])
    right = 4 * (w + gap) + w
    c.arrow([(right - 75, y + 66), (right - 75, 150), (75, 150), (75, y + 68)])
    c.text(440, 164, "16 cycles unrolled per model call, all int32", size=12, color=t.muted, anchor="middle")
    c.box(0, 192, 400, 50, "lane i: its own program, pc and 8 registers", [], title_size=13)
    c.box(480, 192, 400, 50, "262,144 lanes step together on the GPU", [], title_size=13)
    return c


def results_tables() -> str:
    """Markdown tables of every runner, for docs/results.md."""
    rows: dict[tuple[str, str], dict] = {}
    for line in (RESULTS / "native.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[(r["case"], r["runner"])] = r  # later lines win
    out = [
        "### Native\n",
        "| case | size | runner | ms | vs first baseline | answer |",
        "|---|---|---|---:|---:|---|",
    ]
    for case in dict.fromkeys(c for c, _ in rows):
        runners = [r for (c, _), r in rows.items() if c == case]
        base = runners[0]["ms"]
        for i, r in enumerate(runners):
            label, size = (f"`{case}`", r["size"]) if i == 0 else ("", "")
            answer = "correct" if r["correct"] else "**wrong**"
            out.append(f"| {label} | {size} | {r['runner']} | {r['ms']:,.1f} | {base / r['ms']:.2f}× | {answer} |")
    report = json.loads((RESULTS / "browser.json").read_text())
    env = report["env"]
    out += [
        "",
        "### Browser\n",
        f"onnxruntime-web {env['ort']}, headless Chromium, WebGPU adapter `{env['webgpu']}`, "
        f"{env['threads']} WASM threads, cross-origin isolated: {env['isolated']}.\n",
        "| case | WebGPU ms | WASM ms | WebGPU vs WASM | answer |",
        "|---|---:|---:|---:|---|",
    ]
    by_case: dict[str, dict] = {}
    for r in report["results"]:
        by_case.setdefault(r["case"], {})[r["backend"]] = r
    for case, rs in by_case.items():
        gpu, cpu = rs.get("webgpu", {}), rs.get("wasm", {})
        if "ms" in gpu and "ms" in cpu:
            ok = "correct" if gpu["correct"] and cpu["correct"] else "wrong"
            out.append(f"| `{case}` | {gpu['ms']:,.1f} | {cpu['ms']:,.1f} | {cpu['ms'] / gpu['ms']:.2f}× | {ok} |")
        else:
            out.append(f"| `{case}` | {gpu.get('ms', 'error')} | {cpu.get('ms', 'error')} | | |")
    return "\n".join(out) + "\n"


def main() -> None:
    svg.write(portability_diagram, "portability", "One ONNX file runs on every runtime", OUT)
    svg.write(native_chart, "native", "Native speed-ups over the usual tool", OUT)
    svg.write(browser_chart, "browser", "WebGPU vs WASM in the browser", OUT)
    svg.write(int_width_chart, "int-width", "The same DFA with int64 and int32", OUT)
    svg.write(vm_diagram, "vm", "How one VM cycle maps to tensor ops", OUT)
    print(f"wrote {len(list(OUT.glob('*.svg')))} files to {OUT.relative_to(ROOT)}")
    doc = ROOT / "docs" / "results.md"
    head, marker, _ = doc.read_text().partition("<!-- tables -->")
    doc.write_text(head + marker + "\n\n" + results_tables())


if __name__ == "__main__":
    main()
