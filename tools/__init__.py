"""Helper utilities for EasyOCRNet benchmarks."""

from .onnx_infer import run as run_onnx  # noqa: F401

try:  # pragma: no cover - optional dependency guard
    from .openvino_infer import run as run_openvino  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - executed when OpenVINO is absent

    def run_openvino(*_, **__):
        raise ModuleNotFoundError(
            "OpenVINO runtime is not installed. Install the `openvino` package to use "
            "the OpenVINO inference helpers."
        )


__all__ = ["run_onnx", "run_openvino"]
