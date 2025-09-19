"""Inference helpers using OpenVINO IR models."""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from openvino.runtime import Core

from .onnx_infer import (
    detector_postprocess,
    detector_preprocess,
    load_charset,
    recognizer_preprocess,
)


def _infer(compiled_model, input_data: np.ndarray) -> np.ndarray:
    """Run a single inference on the compiled model."""
    input_tensor = compiled_model.input(0)
    output_tensor = compiled_model.output(0)
    result = compiled_model({input_tensor.any_name: input_data})
    return result[output_tensor]


def run(
    detector_model: str,
    recognizer_model: str,
    image_path: str,
    charset: str,
    device: str = "CPU",
) -> Tuple[str, List[str]]:
    core = Core()
    det_compiled = core.compile_model(detector_model, device)
    rec_compiled = core.compile_model(recognizer_model, device)

    img = cv2.imread(image_path)
    det_in = detector_preprocess(img)
    det_out = _infer(det_compiled, det_in)
    bbox = detector_postprocess(det_out)
    if bbox is None:
        return "", [f"OpenVINO:{device}"]

    x_min, y_min, x_max, y_max = bbox
    crop = img[y_min:y_max, x_min:x_max]
    rec_in = recognizer_preprocess(crop)
    rec_out = _infer(rec_compiled, rec_in)

    charset_txt = load_charset(charset)
    prev = 0
    text = []
    for t in range(rec_out.shape[1]):
        idx = int(rec_out[0, t].argmax())
        if idx > 0 and idx != prev:
            ci = idx - 1
            if ci < len(charset_txt):
                text.append(charset_txt[ci])
        prev = idx

    return "".join(text), [f"OpenVINO:{device}"]


__all__ = ["run"]
