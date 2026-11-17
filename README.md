# onnx-lab

Exploring non-ML workloads (analytics, text, simulation, graphs, finance) as portable ONNX graphs, measured against
the tools people already use, natively and in the browser (WebGPU / WASM). See [FINDINGS.md](FINDINGS.md).

```bash
uv sync
uv run python -m experiments.nbody          # any of: groupby, montecarlo, regex_dfa, levenshtein, nbody, graphs
uv run python -m lab.export_web             # bundle models + data into build/web for the browser bench
python3 -m http.server 8765 --directory build/web   # open http://localhost:8765/?ort=1.30.0
```
