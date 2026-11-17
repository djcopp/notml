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
