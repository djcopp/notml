"""A small builder for writing ONNX graphs by hand, for the cases the PyTorch exporter can't express."""

from __future__ import annotations

import itertools

import numpy as np
import onnx
from onnx import helper, numpy_helper

# onnxruntime-web 1.30 refuses the IR version that recent `onnx` releases stamp by default.
IR_VERSION = 9


class Graph:
    """``g.op("Add", a, b)`` appends a node and returns the name of its output."""

    def __init__(self, name: str, opset: int = 18):
        self.name, self.opset = name, opset
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.inputs: list[onnx.ValueInfoProto] = []
        self.outputs: list[onnx.ValueInfoProto] = []
        self._ids = itertools.count()

    def _name(self, hint: str) -> str:
        return f"{hint}_{next(self._ids)}"

    def input(self, name: str, dtype, shape) -> str:
        elem = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
        self.inputs.append(helper.make_tensor_value_info(name, elem, shape))
        return name

    def output(self, tensor: str, name: str, dtype, shape) -> str:
        self.op("Identity", tensor, out=name)
        elem = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
        self.outputs.append(helper.make_tensor_value_info(name, elem, shape))
        return name

    def const(self, value, dtype=None) -> str:
        name = self._name("c")
        self.initializers.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(self, op_type: str, *inputs: str, out: str | None = None, **attrs) -> str:
        out = out or self._name(op_type.lower())
        self.nodes.append(helper.make_node(op_type, list(inputs), [out], **attrs))
        return out

    def model(self) -> onnx.ModelProto:
        graph = helper.make_graph(self.nodes, self.name, self.inputs, self.outputs, initializer=self.initializers)
        m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", self.opset)], producer_name="notml")
        m.ir_version = IR_VERSION
        onnx.checker.check_model(m, full_check=True)
        return m


class Expr:
    """A tensor in a :class:`Graph` with numpy-style operators, so graph code reads like array code.

    Python ints and floats are turned into constants of the other operand's type (int32 by default).
    Comparisons return bool tensors; ``&``, ``|`` and ``~`` work on bools.
    """

    def __init__(self, g: Graph, name: str, dtype=np.int32):
        self.g, self.name, self.dtype = g, name, np.dtype(dtype)

    def _wrap(self, other) -> Expr:
        if isinstance(other, Expr):
            return other
        if isinstance(other, np.ndarray):  # explicit arrays (axes, shapes) keep their own type
            return Expr(self.g, self.g.const(other), other.dtype)
        return Expr(self.g, self.g.const(np.asarray(other, self.dtype)), self.dtype)

    def _op(self, op_type, *others, dtype=None, **attrs) -> Expr:
        names = [self.name] + [self._wrap(o).name for o in others]
        return Expr(self.g, self.g.op(op_type, *names, **attrs), dtype or self.dtype)

    def __add__(self, o):
        return self._op("Add", o)  # noqa: E704

    def __radd__(self, o):
        return self._wrap(o)._op("Add", self)  # noqa: E704

    def __sub__(self, o):
        return self._op("Sub", o)  # noqa: E704

    def __rsub__(self, o):
        return self._wrap(o)._op("Sub", self)  # noqa: E704

    def __mul__(self, o):
        return self._op("Mul", o)  # noqa: E704

    def __rmul__(self, o):
        return self._wrap(o)._op("Mul", self)  # noqa: E704

    def __floordiv__(self, o):
        return self._op("Div", o)  # noqa: E704  (exact for the non-negative values used here)

    def __mod__(self, o):
        return self._op("Mod", o)  # noqa: E704  (sign follows the divisor, like Python)

    def __neg__(self):
        return self._op("Neg")  # noqa: E704

    def __eq__(self, o):
        return self._op("Equal", o, dtype=np.bool_)  # noqa: E704

    def __ne__(self, o):
        return ~(self == o)  # noqa: E704

    def __lt__(self, o):
        return self._op("Less", o, dtype=np.bool_)  # noqa: E704

    def __le__(self, o):
        return self._op("LessOrEqual", o, dtype=np.bool_)  # noqa: E704

    def __gt__(self, o):
        return self._op("Greater", o, dtype=np.bool_)  # noqa: E704

    def __ge__(self, o):
        return self._op("GreaterOrEqual", o, dtype=np.bool_)  # noqa: E704

    def __and__(self, o):
        return self._op("And", o)  # noqa: E704

    def __or__(self, o):
        return self._op("Or", o)  # noqa: E704

    def __invert__(self):
        return self._op("Not")  # noqa: E704

    __hash__ = None

    def int(self) -> Expr:
        return self.cast(np.int32)

    def cast(self, dtype) -> Expr:
        to = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
        return self._op("Cast", to=to, dtype=dtype)

    def where(self, a, b) -> Expr:
        """``a`` where this bool tensor is true, else ``b``."""
        a = a if isinstance(a, Expr) else Expr(self.g, self.g.const(np.asarray(a, np.int32)))
        b = a._wrap(b)
        return self._op("Where", a, b, dtype=a.dtype)

    def unsqueeze(self, axis: int) -> Expr:
        return self._op("Unsqueeze", np.array([axis], np.int64))

    def take(self, index: Expr, axis: int = 1) -> Expr:
        """Per-row lookup: ``out[b, j] = self[b, index[b, j]]`` (GatherElements)."""
        return self._op("GatherElements", index, axis=axis)

    def put(self, index: Expr, values: Expr, axis: int = 1) -> Expr:
        """Per-row write: ``self[b, index[b, j]] = values[b, j]`` (ScatterElements, no duplicates assumed)."""
        return self._op("ScatterElements", index, values, axis=axis)

    def sum(self, axis: int, keepdims: bool = False) -> Expr:
        return self._op("ReduceSum", np.array([axis], np.int64), keepdims=int(keepdims))

    def max(self, axis: int, keepdims: bool = False) -> Expr:
        return self._op("ReduceMax", np.array([axis], np.int64), keepdims=int(keepdims))

    def minimum(self, o) -> Expr:
        return self._op("Min", o)

    def maximum(self, o) -> Expr:
        return self._op("Max", o)

    def reshape(self, shape) -> Expr:
        return self._op("Reshape", np.array(shape, np.int64))


def expr_input(g: Graph, name: str, dtype, shape) -> Expr:
    return Expr(g, g.input(name, dtype, shape), dtype)
