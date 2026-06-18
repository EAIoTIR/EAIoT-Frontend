"""Convert quantised ONNX models to C source code."""

from .onnx_to_c import convert_to_c
from .lstm_linear import replace_lstm_with_linear
from .export_onnx import export_to_onnx

__all__ = ['convert_to_c', 'replace_lstm_with_linear', 'export_to_onnx']