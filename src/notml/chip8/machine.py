"""CHIP-8 as an ONNX graph. Every console in a batch is a row; every instruction is branch-free tensor code.

State tensors, all int32, ``B`` consoles:

    mem     [B, 4096]  memory bytes (font at 0x000, program at 0x200)
    v       [B, 16]    registers V0-VF
    ctl     [B, 6]     PC, I, SP, delay timer, sound timer, random state
    stack   [B, 16]    return addresses
    screen  [B, 2048]  64 x 32 pixels, row-major, 0 or 1
    keys    [B, 16]    which of the 16 keys are held (input only)

Outputs are the same state with ``_out`` appended, plus ``image``: uint8 [B, 2048], 255 where lit.

There are no control-flow ops. Each cycle decodes the instruction, computes the result of every
instruction type, and keeps the one that applies with ``Where``. Bitwise AND/OR/XOR and shifts are done
with Div and Mod on bits, because onnxruntime-web's WebGPU backend has no bitwise kernels.
"""

from __future__ import annotations

import numpy as np
import onnx

from notml.chip8.reference import FONT, START
from notml.graph import Expr, Graph, expr_input

PC, I, SP, DT, ST, RNG = range(6)  # noqa: E741 - CHIP-8 calls the index register I
MSB_FIRST = np.array([128, 64, 32, 16, 8, 4, 2, 1], np.int32)


class _Machine:
    def __init__(self, g: Graph):
        self.g = g
        self.k16 = self.const(np.arange(16, dtype=np.int32)[None, :])  # [1, 16]
        self.pow2 = self.const(MSB_FIRST[None, :])  # [1, 8]
        self.r15 = self.const(np.arange(15, dtype=np.int32)[None, :])  # [1, 15] sprite rows
        self.c8 = self.const(np.arange(8, dtype=np.int32)[None, :])  # [1, 8] sprite columns

    def const(self, value) -> Expr:
        value = np.asarray(value)
        return Expr(self.g, self.g.const(value), value.dtype)

    def col(self, t: Expr, i: int) -> Expr:
        """Column ``i`` of a [B, n] tensor as [B, 1]."""
        g = self.g
        idx = lambda v: g.const(np.array([v], np.int64))  # noqa: E731
        return Expr(g, g.op("Slice", t.name, idx(i), idx(i + 1), idx(1)))

    def concat(self, parts: list[Expr]) -> Expr:
        return Expr(self.g, self.g.op("Concat", *[p.name for p in parts], axis=1))

    def bits(self, a: Expr) -> Expr:
        """[B, 1] byte -> [B, 8] bits, most significant first."""
        return (a // self.pow2) % 2

    def from_bits(self, b: Expr) -> Expr:
        return (b * self.pow2).sum(axis=1, keepdims=True)

    def cycle(self, s: dict[str, Expr], keys: Expr) -> dict[str, Expr]:
        mem, v, ctl, stack, screen = s["mem"], s["v"], s["ctl"], s["stack"], s["screen"]
        pc, i, sp, dt, st, rng = (self.col(ctl, k) for k in range(6))

        # fetch and decode
        op = mem.take(pc) * 256 + mem.take((pc + 1) % 4096)  # [B, 1]
        n1, x, y, n = op // 4096, op // 256 % 16, op // 16 % 16, op % 16
        nn, nnn = op % 256, op % 4096
        vx, vy, v0 = v.take(x), v.take(y), self.col(v, 0)
        key_vx = keys.take(vx % 16)

        def is_(top, low=None, low_byte=None):
            cond = n1 == top
            if low is not None:
                cond = cond & (n == low)
            if low_byte is not None:
                cond = cond & (nn == low_byte)
            return cond

        is_cls, is_ret = op == 0x00E0, op == 0x00EE
        is_jp, is_call, is_jpv0 = is_(1), is_(2), is_(0xB)
        is_ld_i, is_rnd, is_drw = is_(0xA), is_(0xC), is_(0xD)
        is_skp, is_sknp = is_(0xE, low_byte=0x9E), is_(0xE, low_byte=0xA1)
        is_ld_dt_to, is_ld_k = is_(0xF, low_byte=0x07), is_(0xF, low_byte=0x0A)
        is_ld_dt, is_ld_st = is_(0xF, low_byte=0x15), is_(0xF, low_byte=0x18)
        is_add_i, is_ld_f, is_ld_b = is_(0xF, low_byte=0x1E), is_(0xF, low_byte=0x29), is_(0xF, low_byte=0x33)
        is_store, is_load = is_(0xF, low_byte=0x55), is_(0xF, low_byte=0x65)
        alu = [is_(8, k) for k in (0, 1, 2, 3, 4, 5, 6, 7, 14)]

        # keys and random numbers
        any_key = keys.max(axis=1, keepdims=True) > 0
        first_key = ((keys > 0).where(self.k16, 16)).minimum(16)
        first_key = Expr(
            self.g, self.g.op("ReduceMin", first_key.name, self.g.const(np.array([1], np.int64)), keepdims=1)
        )
        rng_next = rng * 75 % 65537
        rng = is_rnd.where(rng_next, rng)

        # the value each register-writing instruction would put in VX
        bx, by = self.bits(vx), self.bits(vy)
        new_vx = is_(6).where(nn, vx)
        new_vx = is_(7).where((vx + nn) % 256, new_vx)
        alu_values = [
            vy,
            self.from_bits(bx + by - bx * by),
            self.from_bits(bx * by),
            self.from_bits((bx + by) % 2),
            (vx + vy) % 256,
            (vx - vy) % 256,
            vx // 2,
            (vy - vx) % 256,
            vx * 2 % 256,
        ]
        for cond, value in zip(alu, alu_values, strict=True):
            new_vx = cond.where(value, new_vx)
        new_vx = is_rnd.where(self.from_bits(self.bits(rng_next % 256) * self.bits(nn)), new_vx)
        new_vx = is_ld_dt_to.where(dt, new_vx)
        new_vx = is_ld_k.where(first_key, new_vx)
        writes_vx = is_(6) | is_(7) | is_rnd | is_ld_dt_to | (is_ld_k & any_key)
        for cond in alu:
            writes_vx = writes_vx | cond
        v = ((self.k16 == x) & writes_vx).where(new_vx, v)

        # VF flags from the ALU, written after VX so they win when X is F
        flag = alu[4].where((vx + vy > 255).int(), 0)
        flag = alu[5].where((vx >= vy).int(), flag)
        flag = alu[6].where(vx % 2, flag)
        flag = alu[7].where((vy >= vx).int(), flag)
        flag = alu[8].where(vx // 128, flag)
        writes_vf = alu[4] | alu[5] | alu[6] | alu[7] | alu[8]

        # FX65: load V0..VX from memory at I
        addr16 = (i + self.k16) % 4096  # [B, 16]
        loaded = mem.take(addr16)
        v = ((self.k16 <= x) & is_load).where(loaded, v)

        # DXYN: draw an N-row sprite from memory at I, XOR it onto the screen, VF = collision.
        # Only the sprite's own 15 x 8 patch is computed, then scattered into the screen; pixels that fall
        # off the edge (or aren't being drawn) go to a spare cell at index 2048.
        x0, y0 = vx % 64, vy % 32
        ys, xs = y0 + self.r15, x0 + self.c8  # [B, 15], [B, 8]
        sprite = mem.take((i + self.r15) % 4096)  # [B, 15] sprite bytes
        lit = (sprite.unsqueeze(2) // self.pow2.unsqueeze(1)) % 2  # [B, 15, 8]
        on_screen = ((self.r15 < n) & (ys < 32) & is_drw).unsqueeze(2) & (xs < 64).unsqueeze(1)
        pos = on_screen.where(ys.unsqueeze(2) * 64 + xs.unsqueeze(1), 2048).reshape([0, 120])
        lit = on_screen.where(lit, 0).reshape([0, 120])
        old = screen.take(pos)
        collision = (old * lit).max(axis=1, keepdims=True)
        screen = screen.put(pos, on_screen.reshape([0, 120]).where((old + lit) % 2, 0))
        screen = is_cls.where(0, screen)
        flag = is_drw.where(collision, flag)
        v = ((self.k16 == 15) & (writes_vf | is_drw)).where(flag, v)

        # FX33 (BCD of VX) and FX55 (store V0..VX) write up to 16 bytes at I
        bcd = (self.k16 == 0).where(vx // 100, (self.k16 == 1).where(vx // 10 % 10, vx % 10))
        values = is_ld_b.where(bcd, v)
        enabled = (is_store & (self.k16 <= x)) | (is_ld_b & (self.k16 < 3))
        mem = mem.put(addr16, enabled.where(values, loaded))

        # stack
        ret_addr = stack.take((sp - 1) % 16)
        stack = ((self.k16 == sp) & is_call).where((pc + 2) % 4096, stack)
        sp = is_call.where((sp + 1) % 16, is_ret.where((sp - 1) % 16, sp))

        # next PC
        skip = (
            (is_(3) & (vx == nn))
            | (is_(4) & (vx != nn))
            | (is_(5, 0) & (vx == vy))
            | (is_(9, 0) & (vx != vy))
            | (is_skp & (key_vx > 0))
            | (is_sknp & (key_vx == 0))
        )
        next_pc = skip.where((pc + 4) % 4096, (pc + 2) % 4096)
        next_pc = (is_jp | is_call).where(nnn, next_pc)
        next_pc = is_jpv0.where((nnn + v0) % 4096, next_pc)
        next_pc = is_ret.where(ret_addr, next_pc)
        next_pc = (is_ld_k & ~any_key).where(pc, next_pc)

        # I and timers
        i = is_ld_i.where(nnn, i)
        i = is_add_i.where((i + vx) % 4096, i)
        i = is_ld_f.where(vx % 16 * 5, i)
        dt = is_ld_dt.where(vx, dt)
        st = is_ld_st.where(vx, st)

        ctl = self.concat([next_pc, i, sp, dt, st, rng])
        return {"mem": mem, "v": v, "ctl": ctl, "stack": stack, "screen": screen}


SHAPES = {"mem": 4096, "v": 16, "ctl": 6, "stack": 16, "screen": 2048}


def build(cycles: int = 10, batch: int | str = "B") -> onnx.ModelProto:
    """A model that runs ``cycles`` instructions on every console, then ticks the timers once."""
    g = Graph("chip8")
    m = _Machine(g)
    state = {name: expr_input(g, name, np.int32, [batch, width]) for name, width in SHAPES.items()}
    keys = expr_input(g, "keys", np.int32, [batch, 16])
    state["screen"] = m.concat([state["screen"], m.col(state["ctl"], PC) * 0])  # spare cell for clipped pixels
    for _ in range(cycles):
        state = m.cycle(state, keys)
    sl = lambda v: g.const(np.array([v], np.int64))  # noqa: E731
    state["screen"] = Expr(g, g.op("Slice", state["screen"].name, sl(0), sl(2048), sl(1)))
    ctl = state["ctl"]
    timers = m.concat([m.col(ctl, DT), m.col(ctl, ST)])
    timers = (timers - 1).maximum(0)
    state["ctl"] = m.concat(
        [m.col(ctl, PC), m.col(ctl, I), m.col(ctl, SP), m.col(timers, 0), m.col(timers, 1), m.col(ctl, RNG)]
    )
    for name, width in SHAPES.items():
        g.output(state[name].name, f"{name}_out", np.int32, [batch, width])
    image = (state["screen"] * 255).cast(np.uint8)
    g.output(image.name, "image", np.uint8, [batch, 2048])
    return g.model()


def initial_state(roms: list[bytes], seeds: list[int] | None = None) -> dict[str, np.ndarray]:
    """Fresh consoles with the font loaded and one ROM per console (repeat a ROM to run copies of it)."""
    b = len(roms)
    mem = np.zeros((b, 4096), np.int32)
    mem[:, : len(FONT)] = FONT
    for k, rom in enumerate(roms):
        mem[k, START : START + len(rom)] = list(rom)
    ctl = np.zeros((b, 6), np.int32)
    ctl[:, PC] = START
    ctl[:, RNG] = seeds if seeds is not None else np.arange(1, b + 1)
    return {
        "mem": mem,
        "v": np.zeros((b, 16), np.int32),
        "ctl": ctl,
        "stack": np.zeros((b, 16), np.int32),
        "screen": np.zeros((b, 2048), np.int32),
    }
