"""A straightforward CHIP-8 interpreter, one console at a time, to test the ONNX version against."""

from __future__ import annotations

import numpy as np

FONT = [
    0xF0, 0x90, 0x90, 0x90, 0xF0, 0x20, 0x60, 0x20, 0x20, 0x70, 0xF0, 0x10, 0xF0, 0x80, 0xF0,
    0xF0, 0x10, 0xF0, 0x10, 0xF0, 0x90, 0x90, 0xF0, 0x10, 0x10, 0xF0, 0x80, 0xF0, 0x10, 0xF0,
    0xF0, 0x80, 0xF0, 0x90, 0xF0, 0xF0, 0x10, 0x20, 0x40, 0x40, 0xF0, 0x90, 0xF0, 0x90, 0xF0,
    0xF0, 0x90, 0xF0, 0x10, 0xF0, 0xF0, 0x90, 0xF0, 0x90, 0x90, 0xE0, 0x90, 0xE0, 0x90, 0xE0,
    0xF0, 0x80, 0x80, 0x80, 0xF0, 0xE0, 0x90, 0x90, 0x90, 0xE0, 0xF0, 0x80, 0xF0, 0x80, 0xF0,
    0xF0, 0x80, 0xF0, 0x80, 0x80,
]  # fmt: skip
WIDTH, HEIGHT, START = 64, 32, 0x200


class Chip8:
    def __init__(self, rom: bytes, seed: int = 1):
        self.mem = np.zeros(4096, np.int64)
        self.mem[: len(FONT)] = FONT
        self.mem[START : START + len(rom)] = list(rom)
        self.v = np.zeros(16, np.int64)
        self.pc, self.i, self.sp, self.dt, self.st, self.rng = START, 0, 0, 0, 0, seed
        self.stack = np.zeros(16, np.int64)
        self.screen = np.zeros((HEIGHT, WIDTH), np.int64)

    def step(self, keys: np.ndarray) -> None:
        mem, v = self.mem, self.v
        op = int(mem[self.pc % 4096]) * 256 + int(mem[(self.pc + 1) % 4096])
        n1, x, y, n, nn, nnn = op >> 12, (op >> 8) & 15, (op >> 4) & 15, op & 15, op & 255, op & 4095
        vx, vy = int(v[x]), int(v[y])
        pc = (self.pc + 2) % 4096
        if op == 0x00E0:
            self.screen[:] = 0
        elif op == 0x00EE:
            self.sp = (self.sp - 1) % 16
            pc = int(self.stack[self.sp])
        elif n1 == 1:
            pc = nnn
        elif n1 == 2:
            self.stack[self.sp] = pc
            self.sp = (self.sp + 1) % 16
            pc = nnn
        elif n1 == 3 and vx == nn or n1 == 4 and vx != nn or n1 == 5 and n == 0 and vx == vy:
            pc = (pc + 2) % 4096
        elif n1 == 9 and n == 0 and vx != vy:
            pc = (pc + 2) % 4096
        elif n1 == 6:
            v[x] = nn
        elif n1 == 7:
            v[x] = (vx + nn) % 256
        elif n1 == 8:
            flag = None
            if n == 0:
                v[x] = vy
            elif n == 1:
                v[x] = vx | vy
            elif n == 2:
                v[x] = vx & vy
            elif n == 3:
                v[x] = vx ^ vy
            elif n == 4:
                v[x], flag = (vx + vy) % 256, int(vx + vy > 255)
            elif n == 5:
                v[x], flag = (vx - vy) % 256, int(vx >= vy)
            elif n == 6:
                v[x], flag = vx >> 1, vx & 1
            elif n == 7:
                v[x], flag = (vy - vx) % 256, int(vy >= vx)
            elif n == 14:
                v[x], flag = (vx << 1) % 256, vx >> 7
            if flag is not None:
                v[15] = flag
        elif n1 == 0xA:
            self.i = nnn
        elif n1 == 0xB:
            pc = (nnn + int(v[0])) % 4096
        elif n1 == 0xC:
            self.rng = self.rng * 75 % 65537
            v[x] = (self.rng % 256) & nn
        elif n1 == 0xD:
            self._draw(vx % WIDTH, vy % HEIGHT, n)
        elif n1 == 0xE and nn == 0x9E and keys[vx % 16] or n1 == 0xE and nn == 0xA1 and not keys[vx % 16]:
            pc = (pc + 2) % 4096
        elif n1 == 0xF:
            if nn == 0x07:
                v[x] = self.dt
            elif nn == 0x0A:
                pressed = np.flatnonzero(keys)
                if len(pressed):
                    v[x] = int(pressed[0])
                else:
                    pc = self.pc
            elif nn == 0x15:
                self.dt = vx
            elif nn == 0x18:
                self.st = vx
            elif nn == 0x1E:
                self.i = (self.i + vx) % 4096
            elif nn == 0x29:
                self.i = (vx % 16) * 5
            elif nn == 0x33:
                for k, digit in enumerate((vx // 100, vx // 10 % 10, vx % 10)):
                    mem[(self.i + k) % 4096] = digit
            elif nn == 0x55:
                for k in range(x + 1):
                    mem[(self.i + k) % 4096] = v[k]
            elif nn == 0x65:
                for k in range(x + 1):
                    v[k] = mem[(self.i + k) % 4096]
        self.pc = pc

    def _draw(self, x0: int, y0: int, rows: int) -> None:
        collision = 0
        for r in range(rows):
            y = y0 + r
            if y >= HEIGHT:
                break
            byte = int(self.mem[(self.i + r) % 4096])
            for c in range(8):
                x = x0 + c
                if x < WIDTH and byte >> (7 - c) & 1:
                    collision |= int(self.screen[y, x])
                    self.screen[y, x] ^= 1
        self.v[15] = collision

    def frame(self, keys: np.ndarray, cycles: int) -> None:
        """Run ``cycles`` instructions, then tick the timers once (one 60 Hz frame)."""
        for _ in range(cycles):
            self.step(keys)
        self.dt, self.st = max(self.dt - 1, 0), max(self.st - 1, 0)
