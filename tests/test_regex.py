"""notml.regex must agree with Python's re.fullmatch, for every method."""

import re

import numpy as np
import onnxruntime as ort
import pytest

from notml import regex

PATTERNS = [
    r"[a-z0-9._]+@[a-z0-9]+\.[a-z]{2,6}",
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z",
    r"(ab|a)c?",
    r"a*",
    r"(foo|bar)+baz?",
    r"[^x]{2,5}",
    r"x?",
    r"\w+\s\w+",
]
METHODS = [("stride", 1), ("stride", 2), ("stride", 4), ("scan", 1)]


def strings(n: int = 2000, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    alphabet = list("abcfoxyz019.@-_ :TZr")
    out = ["".join(rng.choice(alphabet, rng.integers(0, 15))) for _ in range(n)]
    return out + ["", "2024-01-31T12:00:00Z", "2024-01-31T12:00:00.123Z", "a.b@c.io", "foobarbaz", "hello world"]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_dfa_matches_re(pattern):
    d = regex.dfa(pattern)
    for s in strings():
        assert d.fullmatch(s) == (re.fullmatch(pattern, s) is not None), s


@pytest.mark.parametrize("method,stride", METHODS)
@pytest.mark.parametrize("pattern", PATTERNS)
def test_onnx_matches_re(pattern, method, stride):
    data = strings()
    expected = np.array([re.fullmatch(pattern, s) is not None for s in data])
    model = regex.compile(pattern, width=32, method=method, stride=stride)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    assert np.array_equal(session.run(None, {"chars": regex.encode(data, 32)})[0], expected)


def test_encode_rejects_long_and_nul_strings():
    with pytest.raises(ValueError):
        regex.encode(["x" * 33], 32)
    with pytest.raises(ValueError):
        regex.encode(["a\0b"], 32)


def test_match_helper():
    assert regex.match(r"\d+", ["123", "12a", ""]).tolist() == [True, False, False]
