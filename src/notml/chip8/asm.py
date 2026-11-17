"""A small two-pass CHIP-8 assembler using the classic mnemonics.

    start:  LD V0, 10        ; comments start with a semicolon
            LD I, sprite
            DRW V0, V1, 5
            JP start
    sprite: DB 0xF0, 0x90, 0x90, 0x90, 0xF0

Numbers can be decimal, ``0x`` hex or ``0b`` binary. Labels resolve to addresses from 0x200.
"""

from __future__ import annotations

import re

START = 0x200


def _num(token: str, labels: dict[str, int]) -> int:
    token = token.strip()
    if token in labels:
        return labels[token]
    try:
        return int(token, 0)
    except ValueError:
        raise ValueError(f"unknown label or number: {token!r}") from None


def _reg(token: str) -> int | None:
    m = re.fullmatch(r"[vV]([0-9a-fA-F])", token.strip())
    return int(m.group(1), 16) if m else None


def _encode(mnemonic: str, args: list[str], labels: dict[str, int]) -> list[int]:
    m, a = mnemonic.upper(), [s.strip() for s in args]
    up = [s.upper() for s in a]
    r = [_reg(s) for s in a]

    def word(w: int) -> list[int]:
        return [w >> 8 & 0xFF, w & 0xFF]

    def n(i: int, bits: int = 8) -> int:
        value = _num(a[i], labels)
        if not 0 <= value < 1 << bits:
            raise ValueError(f"{m} {', '.join(a)}: {value} doesn't fit in {bits} bits")
        return value

    if m == "DB":
        return [n(i) for i in range(len(a))]
    if m == "CLS":
        return word(0x00E0)
    if m == "RET":
        return word(0x00EE)
    if m == "JP":
        return word(0xB000 | n(1, 12)) if up[0] == "V0" and len(a) == 2 else word(0x1000 | n(0, 12))
    if m == "CALL":
        return word(0x2000 | n(0, 12))
    if m in ("SE", "SNE"):
        if r[1] is not None:
            return word((0x5000 if m == "SE" else 0x9000) | r[0] << 8 | r[1] << 4)
        return word((0x3000 if m == "SE" else 0x4000) | r[0] << 8 | n(1))
    if m in ("OR", "AND", "XOR", "SUB", "SUBN"):
        code = {"OR": 1, "AND": 2, "XOR": 3, "SUB": 5, "SUBN": 7}[m]
        return word(0x8000 | r[0] << 8 | r[1] << 4 | code)
    if m in ("SHR", "SHL"):
        return word(0x8000 | r[0] << 8 | (6 if m == "SHR" else 0xE))
    if m == "RND":
        return word(0xC000 | r[0] << 8 | n(1))
    if m == "DRW":
        return word(0xD000 | r[0] << 8 | r[1] << 4 | n(2, 4))
    if m == "SKP":
        return word(0xE09E | r[0] << 8)
    if m == "SKNP":
        return word(0xE0A1 | r[0] << 8)
    if m == "ADD":
        if up[0] == "I":
            return word(0xF01E | r[1] << 8)
        if r[1] is not None:
            return word(0x8004 | r[0] << 8 | r[1] << 4)
        return word(0x7000 | r[0] << 8 | n(1))
    if m == "LD":
        dst, src = up
        if dst == "I":
            return word(0xA000 | n(1, 12))
        if dst == "DT":
            return word(0xF015 | r[1] << 8)
        if dst == "ST":
            return word(0xF018 | r[1] << 8)
        if dst == "F":
            return word(0xF029 | r[1] << 8)
        if dst == "B":
            return word(0xF033 | r[1] << 8)
        if dst == "[I]":
            return word(0xF055 | r[1] << 8)
        if src == "DT":
            return word(0xF007 | r[0] << 8)
        if src == "K":
            return word(0xF00A | r[0] << 8)
        if src == "[I]":
            return word(0xF065 | r[0] << 8)
        if r[1] is not None:
            return word(0x8000 | r[0] << 8 | r[1] << 4)
        return word(0x6000 | r[0] << 8 | n(1))
    raise ValueError(f"unknown instruction {mnemonic!r}")


def _size(mnemonic: str, args: list[str]) -> int:
    return len(args) if mnemonic.upper() == "DB" else 2


def _lines(source: str):
    for lineno, raw in enumerate(source.splitlines(), 1):
        line = raw.split(";", 1)[0].strip()
        label = None
        if ":" in line:
            label, line = (s.strip() for s in line.split(":", 1))
        if line:
            mnemonic, _, rest = line.partition(" ")
            args = [s for s in rest.split(",") if s.strip()] if rest.strip() else []
        else:
            mnemonic, args = None, []
        yield lineno, label, mnemonic, args


def assemble(source: str) -> bytes:
    labels, addr = {}, START
    for _, label, mnemonic, args in _lines(source):
        if label:
            labels[label] = addr
        if mnemonic:
            addr += _size(mnemonic, args)
    out: list[int] = []
    for lineno, _, mnemonic, args in _lines(source):
        if mnemonic:
            try:
                out += _encode(mnemonic, args, labels)
            except (ValueError, IndexError, TypeError) as e:
                raise ValueError(f"line {lineno}: {e}") from None
    return bytes(out)
