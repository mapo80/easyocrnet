"""Helper utilities for EasyOCRNet benchmarks."""

from .onnx_infer import run as run_onnx  # noqa: F401
from .openvino_infer import run as run_openvino  # noqa: F401

__all__ = ["run_onnx", "run_openvino"]
