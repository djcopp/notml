# notml

Machine learning tooling, used for programs that are not ML.

ONNX is a file format for neural networks, but its operators add up to a general array language. This
repo takes eight ordinary programs (a SQL aggregate, regex validation, a physics step, PageRank, shortest
paths, an option pricer, fuzzy string matching and a small virtual machine), writes each one as an ONNX
graph, and races it against the tool you would normally use. Each race runs natively and in a browser tab.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/portability-dark.svg">
  <img alt="A PyTorch module or a hand-written graph becomes one .onnx file, which runs on onnxruntime CPU, CoreML, WebGPU and WASM, with CUDA still to be measured." src="docs/img/portability-light.svg">
</picture>

## Why

An `.onnx` file is self-contained. It runs on onnxruntime on any OS, on Apple's GPU through CoreML, on
NVIDIA through CUDA or TensorRT, and in any modern browser through WebGPU, with no Python, no CUDA toolkit
and no shader code. Nearly all of the attention goes to running neural networks this way. I wanted to know
what else it is good for: which everyday workloads get faster when written as a graph, which get slower,
and why.

## Results

### In the browser

The same files in onnxruntime-web, WebGPU against WASM with all 10 threads. Both run in the same tab, so
this is the choice a web app actually faces.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/browser-dark.svg">
  <img alt="WebGPU speed-up over 10-thread WASM for each workload. PageRank, int32 regex validation and the N-body step are 3 to 5 times faster; the VM is 1.7 times faster; shortest paths, GROUP BY, Levenshtein and the int64 regex are slower." src="docs/img/browser-light.svg">
</picture>

Workloads that do a lot of independent arithmetic per element get 3-5× from the GPU: PageRank over 80,000
edges, checking a million strings against a regex, and an N-body step. Workloads that are mostly
scattered writes (GROUP BY), long chains of small dependent steps (Levenshtein's dynamic program) or a
poor algorithmic fit for tensors (all-pairs shortest paths) are faster on the CPU.

### Natively

Against the library you would normally reach for, on the same Mac:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/native-dark.svg">
  <img alt="Speed-up of onnxruntime on CPU and on CoreML over each workload's usual library. The N-body step is 52 times faster on CoreML; regex validation, GROUP BY and Monte Carlo with the PCG generator are 1.2 to 1.5 times faster; the VM, shortest paths, PageRank and Levenshtein are slower." src="docs/img/native-light.svg">
</picture>

On the CPU, onnxruntime behaves like a good numpy: 1.2-3× faster where it can fuse and multithread
elementwise maths, and well behind specialised libraries (rapidfuzz, SciPy's sparse routines) or better
algorithms (Floyd-Warshall instead of repeated min-plus squaring). The large native win is the N-body step
on Apple's GPU through CoreML: 52× faster than numpy, and faster than the same model in PyTorch on MPS.

### Integer width decides whether WebGPU helps

WebGPU has no 64-bit integers, and PyTorch's exporter uses int64 for anything index-like. The regex DFA
shows what that costs. The int64 version is no faster on WebGPU than on the CPU. The same graph with int32
states and characters is 3.5× faster on WebGPU, and 3.7× faster than WASM.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/int-width-dark.svg">
  <img alt="Time to check one million strings: int64 on WebGPU 214 ms, int64 on WASM 184 ms, int32 on WebGPU 62 ms, int32 on WASM 231 ms." src="docs/img/int-width-light.svg">
</picture>

Getting an int32 graph out of PyTorch takes care: `int32.sum()` silently promotes to int64, and int32
tensor indexing exports an invalid model (fixed in microsoft/onnxscript#3221).

### A virtual machine inside a graph

Can a graph run arbitrary programs, branches and all? [`experiments/vm.py`](experiments/vm.py) compiles a
small register machine into ONNX: 11 instructions, 8 registers, 16-instruction programs. Every lane has its
own program counter, registers and code, so a single call steps hundreds of thousands of unrelated
programs at once.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/vm-dark.svg">
  <img alt="One VM cycle: fetch with a one-hot program counter, decode, read registers with one-hot sums, compute every operation and select with Where, then write back with a mask. Sixteen cycles are unrolled per model call." src="docs/img/vm-light.svg">
</picture>

The demo runs 262,144 programs for 192 cycles, half computing GCDs and half counting Collatz steps. Every
lane matches a reference interpreter. WebGPU takes 5.7 s and WASM 9.3 s; natively, a vectorised numpy
interpreter takes 2.4 s. So it works, and it is not fast yet. Branch-free fetch and decode touch the whole
program and register file on every cycle, which comes to about 1,300 nodes per call.

Every number, with the method behind it, is in [docs/results.md](docs/results.md).

## What I took away

- **Portability is the reason to do this, not CPU speed.** On a CPU, onnxruntime is roughly as fast as
  numpy. The gains come from running one file on a GPU you don't otherwise have easy access to, and above
  all a GPU inside a browser tab.
- **The shape of the work matters more than the runtime.** Wide, independent, arithmetic-heavy work wins.
  Scatters, long chains of tiny dependent ops, and problems where a CPU library uses a smarter algorithm
  lose, by up to 70×.
- **Check every answer.** Two of the fastest results in this repo were wrong: a multithreaded `ScatterND`
  that lost updates, and a GPU-friendly random number generator that priced options 14% low.
- **Randomness is the awkward part.** `RandomNormalLike` runs on the CPU under GPU providers. A
  counter-based generator built from integer ops fixes that natively, but onnxruntime-web's WebGPU
  provider lacks the bitwise kernels it needs, and the bit-op-free substitute is biased.

## Bugs found on the way

Pushing these runtimes outside their usual workloads turned up three bugs. Fixes are submitted upstream:

| where | what goes wrong | fix |
|---|---|---|
| onnxruntime CPU | `ScatterND` with a reduction loses updates under multithreading; sums come out 11-13% low | [microsoft/onnxruntime#34274](https://github.com/microsoft/onnxruntime/pull/34274) |
| onnxruntime CoreML | a `Conv` with no `pads` attribute (valid ONNX) fails to compile as an ML Program | [microsoft/onnxruntime#34275](https://github.com/microsoft/onnxruntime/pull/34275) |
| PyTorch exporter | int32 tensor indexing exports an invalid `GatherND` | [microsoft/onnxscript#3221](https://github.com/microsoft/onnxscript/pull/3221) |

Two gaps without fixes yet: onnxruntime-web's WebGPU provider has no kernels for `BitShift`,
`BitwiseXor` or `Range`, and onnxruntime-web 1.30 rejects files stamped with the IR version that current
`onnx` releases write (pin `ir_version = 9`).

## Running it

```bash
uv sync
uv run python -m lab run                     # every case, natively, against its baselines
uv run python -m lab run vm regex_dfa_int32  # or just some of them
uv run pytest                                # every case at a small size, checked against its reference

uv run python -m lab web                     # bundle models and data for the browser
uv run playwright install chromium           # once
uv run python -m lab browser                 # run the bundle in headless Chromium
uv run python -m lab serve                   # or open it yourself at http://127.0.0.1:8766
```

Each experiment in [`experiments/`](experiments) is a short file that returns one or more `Case`s: a model
(a PyTorch module or a hand-built ONNX graph), its inputs, the baselines it races, and the tolerance its
answer must meet. The harness in [`lab/`](lab) does the rest. Adding a workload means writing one of those
files.

## Limits

All numbers come from one Apple M5 laptop. NVIDIA GPUs, through CUDA natively and WebGPU in Chrome on
Windows and Linux, are the obvious next measurement. Native GPU numbers come from CoreML, which accepts
some graphs and quietly runs others on the CPU, so a weak CoreML number can mean "fell back" rather than
"slow on the GPU". The browser bench reports which backend ran, but not how a backend split the graph.

MIT licence.
