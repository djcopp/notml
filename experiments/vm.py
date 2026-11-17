"""A register virtual machine compiled into ONNX. Every lane is an independent program with its own
program counter, registers and code, so one model call steps hundreds of thousands of programs at once.

Design choices, all to keep the graph on a GPU:
  * int32 throughout (WebGPU has no int64)
  * no gathers: instruction fetch and register reads/writes are one-hot masks and sums
  * one call runs CYCLES_PER_CALL unrolled cycles; the host feeds (regs, pc) back for more

ISA, one instruction = four int32s (op, a, b, c):
   0 HALT                 1 LI    ra = c            2 ADD   ra = rb + rc      3 SUB  ra = rb - rc
   4 MUL  ra = rb * rc    5 DIV   ra = rb / rc      6 MOD   ra = rb % rc      7 ADDI ra = rb + c
   8 JNZ  if ra: pc = c   9 JMP   pc = c           10 LT    ra = rb < rc
Division truncates toward zero, modulo takes the sign of the divisor, and dividing by zero divides by one.
Register operands outside r0-r7 read as zero.
"""

import numpy as np
import torch

from lab.case import Case, Loop, Tolerance

N_REGS, PROG_LEN, CYCLES_PER_CALL = 8, 16, 16
HALT, LI, ADD, SUB, MUL, DIV, MOD, ADDI, JNZ, JMP, LT = range(11)

# r0 <- gcd(r0, r1), Euclid's algorithm. r7 is always 0 and serves as a move.
GCD = [
    (JNZ, 1, 0, 3),  # 0  if r1 != 0 goto 3
    (HALT, 0, 0, 0),  # 1
    (HALT, 0, 0, 0),  # 2
    (MOD, 2, 0, 1),  # 3  r2 = r0 % r1
    (ADD, 0, 1, 7),  # 4  r0 = r1
    (ADD, 1, 2, 7),  # 5  r1 = r2
    (JMP, 0, 0, 0),  # 6
]
# r1 <- number of Collatz steps from r0 down to 1
COLLATZ = [
    (LI, 3, 0, 1),  # 0  r3 = 1
    (LI, 4, 0, 2),  # 1  r4 = 2
    (LI, 5, 0, 3),  # 2  r5 = 3
    (SUB, 2, 0, 3),  # 3  r2 = n - 1
    (JNZ, 2, 0, 6),  # 4  if n != 1 goto 6
    (HALT, 0, 0, 0),  # 5
    (MOD, 2, 0, 4),  # 6  r2 = n % 2
    (JNZ, 2, 0, 10),  # 7  odd: goto 10
    (DIV, 0, 0, 4),  # 8  n = n / 2
    (JMP, 0, 0, 12),  # 9
    (MUL, 0, 0, 5),  # 10 n = 3n
    (ADDI, 0, 0, 1),  # 11 n = n + 1
    (ADDI, 1, 1, 1),  # 12 steps += 1
    (JMP, 0, 0, 3),  # 13
]


def assemble(program) -> np.ndarray:
    out = np.zeros((PROG_LEN, 4), np.int32)
    out[: len(program)] = program
    return out


class VM(torch.nn.Module):
    def forward(self, prog, regs, pc):  # prog [B, PROG_LEN, 4], regs [B, N_REGS], pc [B]; all int32
        slots = torch.arange(PROG_LEN, dtype=torch.int32, device=regs.device)
        reg_ids = torch.arange(N_REGS, dtype=torch.int32, device=regs.device)
        one = torch.ones((), dtype=torch.int32, device=regs.device)  # ones_like() would be folded to [B]

        def read(index):
            mask = (index.unsqueeze(1) == reg_ids).to(torch.int32)
            return (mask * regs).sum(1, dtype=torch.int32)  # sum() would promote to int64 without dtype

        for _ in range(CYCLES_PER_CALL):
            fetch = (pc.unsqueeze(1) == slots).to(torch.int32)
            op, a, b, c = (fetch.unsqueeze(2) * prog).sum(1, dtype=torch.int32).unbind(1)
            ra, rb, rc = read(a), read(b), read(c)
            divisor = torch.where(rc == 0, one, rc)
            result = torch.where(op == LI, c, ra)
            result = torch.where(op == ADD, rb + rc, result)
            result = torch.where(op == SUB, rb - rc, result)
            result = torch.where(op == MUL, rb * rc, result)
            result = torch.where(op == DIV, torch.div(rb, divisor, rounding_mode="trunc"), result)
            result = torch.where(op == MOD, torch.remainder(rb, divisor), result)
            result = torch.where(op == ADDI, rb + c, result)
            result = torch.where(op == LT, (rb < rc).to(torch.int32), result)
            writes = (op == LI) | ((op >= ADD) & (op <= ADDI)) | (op == LT)
            regs = torch.where((a.unsqueeze(1) == reg_ids) & writes.unsqueeze(1), result.unsqueeze(1), regs)
            nxt = torch.where((op == JNZ) & (ra != 0), c, pc + 1)
            nxt = torch.where(op == JMP, c, nxt)
            pc = torch.where(op == HALT, pc, nxt)
        return regs, pc


def numpy_vm(prog, regs, pc, cycles: int):
    """The same machine written the way you would in numpy: fancy indexing instead of one-hot masks."""
    regs, pc = regs.copy(), pc.copy()
    lanes = np.arange(len(pc))

    def read(index):
        return np.where(index < N_REGS, regs[lanes, np.minimum(index, N_REGS - 1)], 0)

    for _ in range(cycles):
        op, a, b, c = prog[lanes, pc].T
        ra, rb, rc = read(a), read(b), read(c)
        divisor = np.where(rc == 0, 1, rc)
        quotient = np.sign(rb) * np.sign(divisor) * (np.abs(rb) // np.abs(divisor))
        result = np.select(
            [op == LI, op == ADD, op == SUB, op == MUL, op == DIV, op == MOD, op == ADDI, op == LT],
            [c, rb + rc, rb - rc, rb * rc, quotient, rb % divisor, rb + c, (rb < rc).astype(np.int32)],
            ra,
        ).astype(np.int32)
        writes = (op == LI) | ((op >= ADD) & (op <= ADDI)) | (op == LT)
        regs[lanes[writes], a[writes]] = result[writes]
        nxt = np.where((op == JNZ) & (ra != 0), c, pc + 1)
        nxt = np.where(op == JMP, c, nxt)
        pc = np.where(op == HALT, pc, nxt).astype(np.int32)
    return regs, pc


def python_vm(program, regs, cycles: int):
    """Reference interpreter for a single program, used by the tests."""
    regs, pc = list(regs), 0

    def reg(i):
        return regs[i] if i < N_REGS else 0

    for _ in range(cycles):
        op, a, b, c = program[pc] if pc < len(program) else (HALT, 0, 0, 0)
        if op == HALT:
            break
        rb, rc = reg(b), reg(c)
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
        elif (op == JNZ and regs[a] != 0) or op == JMP:
            nxt = c
        pc = nxt
    return regs, pc


def workload(lanes: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Even lanes compute a GCD of two numbers below a million; odd lanes count Collatz steps for n < 1000."""
    rng = np.random.default_rng(seed)
    prog = np.stack([assemble(GCD), assemble(COLLATZ)])[np.arange(lanes) % 2]
    regs = np.zeros((lanes, N_REGS), np.int32)
    regs[0::2, 0] = rng.integers(1, 1_000_000, (lanes + 1) // 2)
    regs[0::2, 1] = rng.integers(1, 1_000_000, (lanes + 1) // 2)
    regs[1::2, 0] = rng.integers(1, 1000, lanes // 2)
    return {"prog": prog, "regs": regs, "pc": np.zeros(lanes, np.int32)}


CALLS = 12  # 192 cycles: enough for every GCD here and almost every Collatz run below 1000


def cases(small: bool = False) -> list[Case]:
    lanes = 4_096 if small else 262_144
    cycles = CALLS * CYCLES_PER_CALL
    return [
        Case(
            "vm",
            VM(),
            workload(lanes),
            {"numpy": lambda prog, regs, pc: numpy_vm(prog, regs, pc, cycles)[0]},
            Tolerance("exact"),
            f"{lanes:,} programs x {cycles} cycles",
            loop=Loop(calls=CALLS, feedback={"regs": 0, "pc": 1}),
        )
    ]
