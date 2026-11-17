# onnx-lab findings — non-ML workloads as portable ONNX graphs

Machine: Apple Silicon Mac (10 cores). Native: onnxruntime 1.32.0 (CPU, CoreML ML Program), PyTorch 2.15 MPS as a
GPU reference. Browser: the same `.onnx` files in onnxruntime-web (WebGPU vs WASM, 10 threads) inside Chromium.
Every number is a median of repeated runs, and every runner's output was checked against the reference.

## Results

### Native (desktop)

| workload | size | baseline | ONNX CPU | ONNX CoreML | torch MPS |
|---|---|---|---|---|---|
| N-body accel step | 4096 bodies | numpy 251 ms | 85 ms (3.0×) | **4.9 ms (51×)** | 34 ms (7.4×) |
| GROUP BY sum | 10M rows / 1k groups | numpy.bincount 11.8 ms (DuckDB 12.5, pandas 31) | 10.4 ms (1.1×) | 10.3 ms | 5.8 ms (2.0×) |
| Monte Carlo Asian option | 1M paths × 64 steps | numpy 392 ms | 461 ms (0.85×) | 450 ms | 26 ms (15×) |
| Regex validation (DFA), int32 | 1M strings × 32 | Python `re` 120 ms | 82 ms (1.46×) | 83 ms | 30 ms (4.1×) |
| PageRank (edge list) | 4k nodes / 80k edges / 30 it | scipy.sparse 1.2 ms | 17 ms | 19 ms | 3.3 ms |
| All-pairs shortest path (min-plus) | 256 nodes | scipy Floyd–Warshall 8 ms | 16 ms | 28 ms | 35 ms |
| Levenshtein (naive DP) | 200k pairs ≤12 chars | rapidfuzz cpdist 4.2 ms | 102 ms | 299 ms | 124 ms |

### Browser (same files, onnxruntime-web 1.30.0)

| workload | WebGPU | WASM | WebGPU vs WASM |
|---|---|---|---|
| Regex DFA **int32** | **57.9 ms** | 341 ms | **5.9×** (and faster than native Python `re`) |
| Regex DFA int64 | 1673 ms | 341 ms | 0.2× |
| PageRank | **8.8 ms** | 133 ms | **15×** (faster than native ORT CPU) |
| N-body | **48.8 ms** | 383 ms | **7.8×** |
| GROUP BY | 70.7 ms | 75.9 ms | 1.1× |
| Monte Carlo | 534 ms | 545 ms | 1.0× (RNG op runs on CPU) |
| Levenshtein | 250 ms | 159 ms | 0.6× |

onnxruntime-web 1.22 → 1.30: N-body WebGPU 108 → 49 ms; PageRank 24 → 8.8 ms; GROUP BY WASM 35 → 76 ms (regression).

## What we learned

1. **On CPU, ONNX is numpy-class, not magic.** 0.85–3× vs numpy; it wins when it can fuse/multithread elementwise
   math (N-body) and loses to specialised C++ (rapidfuzz) or better algorithms (sparse SciPy, Floyd–Warshall).
2. **The wins are on GPUs — and portability is the point.** The same file gives 51× (CoreML) natively and 6–15× over
   WASM in a browser tab, with no Python, no CUDA, no WGSL.
3. **Integer width decides GPU placement.** WebGPU has no native int64; exporters emit int64 indices by default, so
   index-heavy graphs (gathers, scatters, state machines) silently run slowly. Switching the DFA to int32 made it
   **29× faster on WebGPU**. A tool that rewrites int64 → int32 where values provably fit is an obvious win.
4. **Algorithm shape matters more than the runtime.** Dense formulations of sparse problems (PageRank as matmul:
   700× slower) or O(N³ log N) min-plus vs O(N³) Floyd lose. Gather/scatter edge lists work; long chains of tiny
   ops (naive Levenshtein, 1,520 nodes) don't.
5. **Randomness needs a GPU-native formulation.** `RandomNormalLike` runs on CPU in CoreML/WebGPU, capping Monte
   Carlo at ~1×. A counter-based RNG built from integer ops would keep it on the GPU (torch MPS shows 15× is there).

## Ecosystem bugs found (to verify minimal repros before reporting upstream)

* **onnxruntime 1.32.0 CPU: `ScatterND(reduction="add")` gives wrong results when multithreaded** — totals ~11–13%
  low at ≥1M updates (data race on duplicate indices). Correct with `intra_op_num_threads=1`; `ScatterElements` is
  correct and faster. `torch.Tensor.index_add` exports to the broken op.
* **torch.onnx (dynamo) exporter emits `GatherND` with int32 indices**, which the ONNX spec forbids → invalid model.
  Workaround: `index_select` (→ `Gather`).
* **onnxruntime CoreML EP (ML Program) rejects `Conv` without an explicit `pads` attribute** (valid ONNX).
* **onnxruntime-web 1.30 WASM is ~2× slower than 1.22 on ScatterElements-heavy GROUP BY** (needs a cleaner repro).

## Promising directions

* **WebGPU "doctor" + rewriter:** detect nodes that fall off the GPU (int64, unsupported ops, RNG) and rewrite them
  (int32 narrowing, decompositions, GPU RNG). Measured upside so far: 29× on an index-heavy graph.
* **Batched state machines on GPU:** DFAs (validation, tokenising, log parsing) and simulations (N-body, cellular
  automata, game logic) map well — both beat native CPU baselines from inside a browser.
* **Graph analytics in the browser:** edge-list PageRank is 15× faster on WebGPU than WASM.
* **Bit-parallel string algorithms** (Myers' Levenshtein on uint64 bitwise ops) instead of naive DP — not yet tried.
