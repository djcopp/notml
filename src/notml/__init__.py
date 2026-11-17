"""notml: ordinary programs compiled to ONNX graphs, so they run on any GPU, including in a browser."""

from notml import regex
from notml.graph import Graph

__all__ = ["Graph", "regex"]
__version__ = "0.2.0"
