"""Batch regex validation with notml.regex: three realistic patterns, each compiled with every method.

Half of each batch matches. The other half are near misses (one character changed, dropped or added), so
the automaton can't reject them early on the first character.
"""

from __future__ import annotations

import re
import string

import numpy as np

from lab.case import Case, Tolerance
from notml import regex

PATTERNS = {
    "email": (r"[a-z0-9._]+@[a-z0-9]+\.[a-z]{2,6}", 32),
    "timestamp": (r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z", 32),
    "uuid": (r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", 64),
}
METHODS = {"stride1": ("stride", 1), "stride2": ("stride", 2), "stride4": ("stride", 4)}

# Long lines are where the log-depth scan should help: 1,024 sequential steps become 10.
LOG_PATTERN = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} (INFO|WARN|ERROR) [a-z ]+"
LOG_WIDTH = 1024


def _positive(kind: str, rng) -> str:
    def pick(chars: str, n: int) -> str:
        return "".join(rng.choice(list(chars), n))

    lower, hexd = string.ascii_lowercase, "0123456789abcdef"
    if kind == "email":
        local = pick(lower + "._", rng.integers(1, 12))
        return f"{local}@{pick(lower, rng.integers(2, 9))}.{pick(lower, rng.integers(2, 5))}"
    if kind == "timestamp":
        frac = f".{pick(string.digits, rng.integers(1, 7))}" if rng.random() < 0.5 else ""
        ranges = [(1970, 2100), (1, 13), (1, 29), (0, 24), (0, 60), (0, 60)]
        y, mo, d, h, mi, s = (rng.integers(lo, hi) for lo, hi in ranges)
        return f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:02d}{frac}Z"
    version, variant = rng.integers(1, 6), rng.choice(list("89ab"))
    return f"{pick(hexd, 8)}-{pick(hexd, 4)}-{version}{pick(hexd, 3)}-{variant}{pick(hexd, 3)}-{pick(hexd, 12)}"


def _near_miss(s: str, rng, alphabet: str, width: int) -> str:
    i = int(rng.integers(0, len(s)))
    op = rng.integers(0, 3)
    if op == 0:
        return s[:i] + rng.choice(list(alphabet)) + s[i + 1 :]
    if op == 1:
        return s[:i] + s[i + 1 :]
    return (s[:i] + rng.choice(list(alphabet)) + s[i:])[:width]


def make_strings(kind: str, n: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    _, width = PATTERNS[kind]
    # email allows almost any lowercase character anywhere, so its near misses use characters it rejects
    alphabet = "#! " if kind == "email" else string.ascii_lowercase + string.digits + ".-_@:TZ"
    out = []
    for i in range(n):
        s = _positive(kind, rng)
        out.append(s if i % 2 == 0 else _near_miss(s, rng, alphabet, width))
    return out


def log_lines(n: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        body = "".join(rng.choice(list(string.ascii_lowercase + " "), rng.integers(200, LOG_WIDTH - 40)))
        level = rng.choice(["INFO", "WARN", "ERROR"])
        line = (
            f"2026-{rng.integers(1, 13):02d}-{rng.integers(1, 29):02d} {rng.integers(0, 24):02d}:00:00 {level} {body}"
        )
        if i % 2:  # near miss deep inside the line, so nothing can stop early
            j = int(rng.integers(len(line) // 2, len(line)))
            line = line[:j] + "!" + line[j + 1 :]
        out.append(line)
    return out


def long_line_cases(small: bool) -> list[Case]:
    n = 512 if small else 8_192
    strings = log_lines(n)
    compiled = re.compile(LOG_PATTERN)
    chars = regex.encode(strings, LOG_WIDTH)

    def python_re(_):
        return np.array([compiled.fullmatch(s) is not None for s in strings])

    return [
        Case(
            f"regex_loglines_{label}",
            regex.compile(LOG_PATTERN, LOG_WIDTH, method=method, stride=stride),
            {"chars": chars},
            {"python re": python_re},
            Tolerance("exact"),
            f"{n:,} lines, width {LOG_WIDTH}",
            js_regexp=f"^(?:{LOG_PATTERN})$" if label == "stride4" else None,
        )
        for label, (method, stride) in {"stride4": ("stride", 4), "scan": ("scan", 1)}.items()
    ]


def cases(small: bool = False) -> list[Case]:
    n = 20_000 if small else 1_000_000
    out = []
    for kind, (pattern, width) in PATTERNS.items():
        strings = make_strings(kind, n)
        compiled = re.compile(pattern)
        chars = regex.encode(strings, width)

        def python_re(_, strings=strings, compiled=compiled):
            return np.array([compiled.fullmatch(s) is not None for s in strings])

        for label, (method, stride) in METHODS.items():
            if method == "stride" and width % stride:
                continue
            out.append(
                Case(
                    f"regex_{kind}_{label}",
                    regex.compile(pattern, width, method=method, stride=stride),
                    {"chars": chars},
                    {"python re": python_re},
                    Tolerance("exact"),
                    f"{n:,} strings, width {width}",
                    js_regexp=f"^(?:{pattern})$" if label == "stride1" else None,  # time it once per pattern
                )
            )
    return out + long_line_cases(small)
