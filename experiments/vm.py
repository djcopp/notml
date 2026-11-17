"""A tiny register virtual machine compiled into ONNX: millions of programs run in parallel, one per lane.

Design (applies the earlier lessons):
  * int32 everywhere — no int64 tensors, so WebGPU keeps it on the GPU
  * branch-free and gather-free: instruction fetch, register reads and writes are one-hot masks + reductions
  * each lane has its own program counter, registers and (optionally) its own program (MIMD-style)
  * one model call = STEPS_PER_CALL unrolled VM cycles; the host loops calls (state stays as plain tensors)

ISA (each instruction = 4 int32s: op, a, b, c):
  0 HALT                      1 LI   ra = c              2 ADD  ra = rb + rc        3 SUB  ra = rb - rc
  4 MUL  ra = rb * rc         5 DIV  ra = rb // rc       6 MOD  ra = rb % rc        7 ADDI ra = rb + c
  8 JNZ  if ra != 0: pc = c   9 JMP  pc = c             10 LT   ra = rb < rc
"""

from __future__ import annotations

import numpy as np
import torch

from lab.harness import timeit

N_REGS, PROG_LEN, STEPS_PER_CALL = 8, 16, 16
HALT, LI, ADD, SUB, MUL, DIV, MOD, ADDI, JNZ, JMP, LT = range(11)

# gcd(r0, r1) -> r0, via Euclid. r2 = tmp. Halts with result in r0.
GCD = [
    (JNZ, 1, 0, 3),  # 0: if r1 != 0 goto 3
    (HALT, 0, 0, 0),  # 1: done, r0 = gcd
    (HALT, 0, 0, 0),  # 2: (unused)
    (MOD, 2, 0, 1),  # 3: r2 = r0 % r1
    (ADD, 0, 1, 7),  # 4: r0 = r1 + r7 (r7 = 0)
    (ADD, 1, 2, 7),  # 5: r1 = r2 + r7
    (JMP, 0, 0, 0),  # 6: goto 0
]
# collatz step count of r0 -> r1 (bounded by the step budget)
COLLATZ = [
    (LI, 3, 0, 1),  # 0: r3 = 1
    (LI, 4, 0, 2),  # 1: r4 = 2
    (LI, 5, 0, 3),  # 2: r5 = 3
    (SUB, 2, 0, 3),  # 3: r2 = n - 1
    (JNZ, 2, 0, 6),  # 4: if n != 1 goto 6
    (HALT, 0, 0, 0),  # 5
    (MOD, 2, 0, 4),  # 6: r2 = n % 2
    (JNZ, 2, 0, 10),  # 7: odd -> 10
    (DIV, 0, 0, 4),  # 8: n = n / 2
    (JMP, 0, 0, 12),  # 9
    (MUL, 0, 0, 5),  # 10: n = 3n
    (ADDI, 0, 0, 1),  # 11: n = n + 1
    (ADDI, 1, 1, 1),  # 12: steps += 1
    (JMP, 0, 0, 3),  # 13
]


def assemble(prog) -> np.ndarray:
    p = np.zeros((PROG_LEN, 4), np.int32)
    p[: len(prog)] = np.asarray(prog, np.int32)
    return p


class VM(torch.nn.Module):
    def forward(self, prog, regs, pc):
        # prog: [B or 1, L, 4] int32; regs: [B, R] int32; pc: [B] int32
        lanes = torch.arange(PROG_LEN, dtype=torch.int32, device=regs.device)
        regidx = torch.arange(N_REGS, dtype=torch.int32, device=regs.device)
        zero = torch.zeros_like(pc)
        for _ in range(STEPS_PER_CALL):
            fetch = (pc.unsqueeze(1) == lanes).to(torch.int32)  # [B, L] one-hot
            ins = (fetch.unsqueeze(2) * prog).sum(
                1, dtype=torch.int32
            )  # [B, 4] (sum defaults to int64!)
            op, a, b, c = ins.unbind(1)  # Split, not int64 Gather
            oh_a = a.unsqueeze(1) == regidx  # [B, R]
            ra = (oh_a.to(torch.int32) * regs).sum(1, dtype=torch.int32)
            rb = ((b.unsqueeze(1) == regidx).to(torch.int32) * regs).sum(
                1, dtype=torch.int32
            )
            rc = ((c.unsqueeze(1) == regidx).to(torch.int32) * regs).sum(
                1, dtype=torch.int32
            )
            safe = torch.where(rc == 0, torch.ones_like(rc), rc)
            res = torch.where(op == LI, c, ra)
            res = torch.where(op == ADD, rb + rc, res)
            res = torch.where(op == SUB, rb - rc, res)
            res = torch.where(op == MUL, rb * rc, res)
            res = torch.where(
                op == DIV, torch.div(rb, safe, rounding_mode="trunc"), res
            )
            res = torch.where(op == MOD, torch.remainder(rb, safe), res)
            res = torch.where(op == ADDI, rb + c, res)
            res = torch.where(op == LT, (rb < rc).to(torch.int32), res)
            writes = (op == LI) | ((op >= ADD) & (op <= ADDI)) | (op == LT)
            regs = torch.where(oh_a & writes.unsqueeze(1), res.unsqueeze(1), regs)
            nxt = pc + 1
            nxt = torch.where((op == JNZ) & (ra != zero), c, nxt)
            nxt = torch.where(op == JMP, c, nxt)
            pc = torch.where(op == HALT, pc, nxt)
        return regs, pc


def py_run(prog, regs, max_cycles):
    """Reference interpreter (one program, pure Python)."""
    regs = list(regs)
    pc = 0
    for _ in range(max_cycles):
        op, a, b, c = prog[pc] if pc < len(prog) else (HALT, 0, 0, 0)
        if op == HALT:
            break
        rb = regs[b] if b < N_REGS else 0  # matches the one-hot read in the tensor VM
        rc = regs[c] if c < N_REGS else 0
        nxt = pc + 1
        if op == LI:
            regs[a] = c
        elif op == ADD:
            regs[a] = rb + rc
        elif op == SUB:
            regs[a] = rb - rc
        elif op == MUL:
            regs[a] = rb * rc
        elif op == DIV:
            regs[a] = int(rb / (rc or 1))
        elif op == MOD:
            regs[a] = rb % (rc or 1)
        elif op == ADDI:
            regs[a] = rb + c
        elif op == LT:
            regs[a] = int(rb < rc)
        elif op == JNZ and regs[a] != 0 or op == JMP:
            nxt = c
        pc = nxt
    return regs, pc


def main():
    import onnxruntime as ort

    from lab import harness

    path = harness.BUILD / "vm.onnx"
    B = 1_000_000
    rng = np.random.default_rng(0)
    regs = np.zeros((B, N_REGS), np.int32)
    regs[:, 0] = rng.integers(1, 1_000_000, B)
    regs[:, 1] = rng.integers(1, 1_000_000, B)
    # MIMD: even lanes run GCD, odd lanes run Collatz (on r0 only)
    progs = np.stack([assemble(GCD), assemble(COLLATZ)])
    lane_prog = progs[np.arange(B) % 2]
    regs[1::2, 0] = rng.integers(1, 1000, B // 2)
    regs[1::2, 1] = 0
    pc = np.zeros(B, np.int32)
    harness.export(VM(), (lane_prog, regs, pc), path, ["prog", "regs", "pc"])
    import onnx

    m = onnx.load(path)
    print(
        f"vm.onnx: {len(m.graph.node)} nodes, {STEPS_PER_CALL} cycles/call, ops: {sorted({n.op_type for n in m.graph.node})}"
    )
    types = {
        vi.type.tensor_type.elem_type
        for vi in list(m.graph.input) + list(m.graph.output)
    }
    print(
        "  int64 anywhere in graph:",
        any(t.data_type == 7 for t in m.graph.initializer),
        "| io elem types:",
        types,
    )

    CALLS = 12  # 192 VM cycles: enough for every GCD (<= ~30 iterations * 4) and Collatz(n<1000) bounded
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def run_onnx():
        r, p = regs, pc
        for _ in range(CALLS):
            r, p = s.run(None, {"prog": lane_prog, "regs": r, "pc": p})
        return r, p

    r_out, p_out = run_onnx()
    # verify against the reference interpreter on a sample of lanes
    sample = rng.choice(B, 2000, replace=False)
    bad = 0
    for i in sample:
        prog = GCD if i % 2 == 0 else COLLATZ
        rr, _ = py_run(prog, regs[i].tolist(), CALLS * STEPS_PER_CALL)
        bad += int(rr[0] != r_out[i, 0] or rr[1] != r_out[i, 1])
    gcd_ok = np.array_equal(r_out[0::2, 0], np.gcd(regs[0::2, 0], regs[0::2, 1]))
    halted = float(np.mean(lane_prog[np.arange(B), p_out, 0] == HALT))
    print(
        f"  correctness: {bad} mismatches vs Python interpreter on 2000 sampled lanes; all GCDs == np.gcd: {gcd_ok}; "
        f"halted lanes: {halted:.1%}"
    )
    ms = timeit(run_onnx, warmup=1, reps=3)
    cycles = B * CALLS * STEPS_PER_CALL
    print(
        f"  onnx:CPU  {B:,} programs x {CALLS * STEPS_PER_CALL} cycles: {ms:8.0f} ms  ({cycles / ms / 1e6:.2f} G VM-cycles/s)"
    )
    # Python interpreter throughput on a slice, extrapolated
    n_py = 2000
    t = timeit(
        lambda: [
            py_run(
                GCD if i % 2 == 0 else COLLATZ, regs[i].tolist(), CALLS * STEPS_PER_CALL
            )
            for i in range(n_py)
        ],
        warmup=0,
        reps=1,
    )
    print(
        f"  python interpreter: {t / n_py * B:8.0f} ms extrapolated for {B:,} programs"
    )
    return lane_prog, regs, pc, r_out, p_out


if __name__ == "__main__":
    main()
