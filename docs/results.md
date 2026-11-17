# Full results

All numbers come from one machine: an Apple M5 MacBook (10 cores) on macOS 26.5, with onnxruntime 1.32.0,
PyTorch 2.15 and numpy 2.5.5. Browser runs use onnxruntime-web 1.30.0 in headless Chromium, driven by
`python -m lab browser`.

## How the numbers were taken

- **Timing.** Each runner is called twice to warm up, then timed seven times; the table shows the median.
  The browser bench does the same inside the page with `performance.now()`.
- **Correctness.** Every runner's answer is compared with the first baseline using the case's tolerance
  (exact for integer outputs, relative tolerances for floating point). A fast wrong answer is reported as
  wrong, not as a speed-up.
- **Baselines.** The first baseline for each case is what most people would use for that job: numpy,
  pandas, DuckDB, Python's `re`, SciPy or rapidfuzz.
- **Browser fairness.** The bench page is served with cross-origin isolation headers, so the WASM backend
  really does get all 10 threads. It runs in a foreground headless tab, so timers aren't throttled.
- **Run-to-run spread.** Two full browser passes agreed to within about 10% for WebGPU and 15% for WASM.
  The tables show the second pass.
- **Loops.** The VM runs as 12 calls of 16 cycles. On WebGPU the state stays in GPU buffers between calls
  (`preferredOutputLocation: "gpu-buffer"`); natively it round-trips through numpy.

## Random number quality

Moments and lag-1 autocorrelation of 262,144 normals from each in-graph generator (ideal: 0, 1, 0, 3, 0):

| generator | mean | std | skew | kurtosis | lag-1 autocorrelation | option price |
|---|---:|---:|---:|---:|---:|---|
| PCG hash + Box-Muller | 0.0001 | 0.9999 | 0.0007 | 2.9994 | -0.0004 | matches numpy |
| multiply/divide hash | 0.0020 | 0.9948 | 0.0048 | 2.9317 | -0.0054 | about 14% low |

The multiply/divide generator looks almost fine on these moments, yet prices come out well below the
reference. My best guess is weak mixing between neighbouring counters, which compounds along each 64-step
path; I haven't confirmed it. Simple moment checks would not have caught it. Pricing a known option did.

<!-- tables -->

### Native

| case | size | runner | ms | vs first baseline | answer |
|---|---|---|---:|---:|---|
| `nbody` | 4,096 bodies | numpy | 262.6 | 1.00× | correct |
|  |  | onnx:CPU | 86.9 | 3.02× | correct |
|  |  | onnx:CoreML | 5.1 | 51.89× | correct |
|  |  | torch:mps | 33.2 | 7.90× | correct |
| `groupby` | 10,000,000 rows, 1,000 groups | numpy.bincount | 12.4 | 1.00× | correct |
|  |  | pandas | 34.3 | 0.36× | correct |
|  |  | duckdb | 12.5 | 0.99× | correct |
|  |  | onnx:CPU | 10.6 | 1.16× | correct |
|  |  | onnx:CoreML | 10.7 | 1.16× | correct |
|  |  | torch:mps | 2.4 | 5.05× | correct |
| `montecarlo_torch_rng` | 1,000,000 paths x 64 steps | numpy | 402.7 | 1.00× | correct |
|  |  | onnx:CPU | 468.3 | 0.86× | correct |
|  |  | onnx:CoreML | 467.0 | 0.86× | correct |
|  |  | torch:mps | 27.6 | 14.58× | correct |
| `montecarlo_pcg` | 1,000,000 paths x 64 steps | numpy | 401.0 | 1.00× | correct |
|  |  | onnx:CPU | 331.0 | 1.21× | correct |
|  |  | onnx:CoreML | 323.6 | 1.24× | correct |
| `montecarlo_muldiv` | 1,000,000 paths x 64 steps | numpy | 411.7 | 1.00× | correct |
|  |  | onnx:CPU | 538.6 | 0.76× | **wrong** |
|  |  | onnx:CoreML | 543.1 | 0.76× | **wrong** |
| `regex_dfa_int64` | 1,000,000 strings, up to 32 characters | python re | 129.8 | 1.00× | correct |
|  |  | numpy | 178.8 | 0.73× | correct |
|  |  | onnx:CPU | 95.8 | 1.36× | correct |
|  |  | onnx:CoreML | 178.6 | 0.73× | correct |
|  |  | torch:mps | 36.2 | 3.58× | correct |
| `regex_dfa_int32` | 1,000,000 strings, up to 32 characters | python re | 129.0 | 1.00× | correct |
|  |  | numpy | 191.1 | 0.67× | correct |
|  |  | onnx:CPU | 87.3 | 1.48× | correct |
|  |  | onnx:CoreML | 84.6 | 1.52× | correct |
|  |  | torch:mps | 26.6 | 4.85× | correct |
| `pagerank` | 4,000 nodes, 79,804 edges, 30 iterations | scipy.sparse | 1.2 | 1.00× | correct |
|  |  | onnx:CPU | 18.5 | 0.06× | correct |
|  |  | onnx:CoreML | 20.3 | 0.06× | correct |
|  |  | torch:mps | 3.4 | 0.34× | correct |
| `shortest_paths` | 256 nodes, all pairs | scipy floyd_warshall | 8.4 | 1.00× | correct |
|  |  | onnx:CPU | 16.3 | 0.51× | correct |
|  |  | onnx:CoreML | 28.8 | 0.29× | correct |
|  |  | torch:mps | 36.1 | 0.23× | correct |
| `levenshtein` | 200,000 pairs, up to 12 characters | rapidfuzz (all cores) | 4.3 | 1.00× | correct |
|  |  | rapidfuzz (loop) | 21.3 | 0.20× | correct |
|  |  | onnx:CPU | 96.2 | 0.04× | correct |
|  |  | onnx:CoreML | 317.8 | 0.01× | correct |
|  |  | torch:mps | 127.9 | 0.03× | correct |
| `vm` | 262,144 programs x 192 cycles | numpy | 2,437.3 | 1.00× | correct |
|  |  | onnx:CPU | 4,251.7 | 0.57× | correct |
|  |  | onnx:CoreML | 13,939.7 | 0.17× | correct |
|  |  | torch:mps | 1,969.2 | 1.24× | correct |

### Browser

onnxruntime-web 1.30.0, headless Chromium, WebGPU adapter `apple`, 10 WASM threads, cross-origin isolated: True.

| case | WebGPU ms | WASM ms | WebGPU vs WASM | answer |
|---|---:|---:|---:|---|
| `nbody` | 50.4 | 163.7 | 3.25× | correct |
| `groupby` | 47.8 | 19.6 | 0.41× | correct |
| `montecarlo_torch_rng` | 538.6 | 574.4 | 1.07× | correct |
| `montecarlo_pcg` | 1,175.4 | 935.9 | 0.80× | correct |
| `montecarlo_muldiv` | 366.7 | 1,265.4 | 3.45× | wrong |
| `regex_dfa_int64` | 214.1 | 184.1 | 0.86× | correct |
| `regex_dfa_int32` | 61.8 | 231.4 | 3.74× | correct |
| `pagerank` | 7.7 | 39.2 | 5.11× | correct |
| `shortest_paths` | 111.8 | 39.2 | 0.35× | correct |
| `levenshtein` | 588.6 | 223.1 | 0.38× | correct |
| `vm` | 5,655.4 | 9,348.3 | 1.65× | correct |
