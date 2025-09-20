#!/usr/bin/env python3
"""Extract text from a single image using TorchfreeEasyOCR ONNX models."""
from __future__ import annotations

import argparse
from pathlib import Path

from run_onnx_examples import (
    MODEL_EXTENSIONS,
    PROVIDERS,
    guess_languages,
    resolve_model,
)
from tools import onnx_infer

try:  # pragma: no cover - optional dependency guard
    from tools import openvino_infer
except ModuleNotFoundError:  # pragma: no cover
    openvino_infer = None


def ensure_model(path: Path, provider_key: str) -> None:
    if path.exists():
        return

    if provider_key == "cpu":
        try:
            from tools.download_torchfree_models import download_models
        except Exception as exc:  # pragma: no cover - defensive import guard
            raise FileNotFoundError(path) from exc

        download_models(path.parent)
        if path.exists():
            return

    hint = "python tools/download_torchfree_models.py"
    raise FileNotFoundError(f"Missing model: {path}. Run `{hint}` first.")


def infer_text(
    image_path: Path,
    models_dir: Path,
    provider_key: str,
) -> str:
    model_ext = MODEL_EXTENSIONS[provider_key]
    detection_path = models_dir / f"detection{model_ext}"
    ensure_model(detection_path, provider_key)

    languages = guess_languages(image_path)
    model_cfg = resolve_model(languages)
    recognizer_path = models_dir / f"{model_cfg['recognizer']}{model_ext}"
    ensure_model(recognizer_path, provider_key)

    if provider_key == "cpu":
        text, _ = onnx_infer.run(
            str(detection_path),
            str(recognizer_path),
            str(image_path),
            charset=model_cfg["charset"],
            providers=PROVIDERS[provider_key],
        )
    else:
        if openvino_infer is None:
            raise ModuleNotFoundError(
                "OpenVINO runtime is not installed. Install the `openvino` package to use "
                "the OpenVINO inference helpers."
            )
        text, _ = openvino_infer.run(
            str(detection_path),
            str(recognizer_path),
            str(image_path),
            charset=model_cfg["charset"],
        )

    return text.strip()


def write_output(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the image to process.")
    parser.add_argument(
        "--models",
        type=Path,
        default=None,
        help="Directory containing TorchfreeEasyOCR ONNX models.",
    )
    parser.add_argument(
        "--provider",
        choices=tuple(PROVIDERS.keys()),
        default="cpu",
        help="Execution provider for inference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output text file path. Defaults to <image>.torchonnx.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path: Path = args.image
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    models_dir = args.models or Path("models") / args.provider
    text = infer_text(image_path, models_dir, args.provider)

    if args.output is None:
        args.output = image_path.with_suffix(".torchonnx.txt")

    write_output(text, args.output)
    print(f"Saved OCR text to {args.output}")


if __name__ == "__main__":
    main()
