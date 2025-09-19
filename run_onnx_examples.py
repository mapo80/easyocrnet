#!/usr/bin/env python3
"""Run EasyOCR ONNX models (from TorchfreeEasyOCR releases) on the local samples."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from tools import onnx_infer
from tools import openvino_infer

LANGUAGE_HINTS: Dict[str, Sequence[str]] = {
    "chinese": ("ch_sim",),
    "japanese": ("ja",),
    "korean": ("ko",),
    "thai": ("th",),
    "french": ("fr",),
}

LANGUAGE_MODELS: Dict[str, Dict[str, str]] = {
    "en": {"recognizer": "english_g2_rec", "charset": "en"},
    "fr": {"recognizer": "latin_g2_rec", "charset": "fr"},
    "ch_sim": {"recognizer": "zh_sim_g2_rec", "charset": "ch_sim"},
    "ja": {"recognizer": "japanese_g2_rec", "charset": "ja"},
    "ko": {"recognizer": "korean_g2_rec", "charset": "ko"},
    "th": {"recognizer": "thai_g1_rec", "charset": "th"},
}

MODEL_EXTENSIONS = {
    "cpu": ".onnx",
    "openvino": ".xml",
}

PROVIDERS: Dict[str, List[str]] = {
    "cpu": ["CPUExecutionProvider"],
    "openvino": ["OpenVINO"],
}


def guess_languages(image_path: Path) -> Tuple[str, ...]:
    stem = image_path.stem.lower()
    for key, languages in LANGUAGE_HINTS.items():
        if key in stem:
            return tuple(languages)
    return ("en",)


def resolve_model(languages: Tuple[str, ...]) -> Dict[str, str]:
    primary = languages[0] if languages else "en"
    return LANGUAGE_MODELS.get(primary, LANGUAGE_MODELS["en"])


def run_examples(
    example_dir: Path,
    model_root: Path,
    provider_key: str,
) -> List[Dict[str, object]]:
    model_ext = MODEL_EXTENSIONS[provider_key]
    detection_path = model_root / f"detection{model_ext}"
    if not detection_path.exists():
        hint = "python tools/download_torchfree_models.py"
        raise FileNotFoundError(
            f"Missing detection model: {detection_path}. Run `{hint}` first."
        )

    providers = PROVIDERS[provider_key]
    results: List[Dict[str, object]] = []

    for image_path in sorted(example_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        languages = guess_languages(image_path)
        model_cfg = resolve_model(languages)
        recognizer_path = model_root / f"{model_cfg['recognizer']}{model_ext}"
        if not recognizer_path.exists():
            hint = "python tools/download_torchfree_models.py"
            raise FileNotFoundError(
                f"Missing recognizer model: {recognizer_path}. Run `{hint}` first."
            )

        if provider_key == "cpu":
            text, providers_used = onnx_infer.run(
                str(detection_path),
                str(recognizer_path),
                str(image_path),
                charset=model_cfg["charset"],
                providers=providers,
            )
        else:
            text, providers_used = openvino_infer.run(
                str(detection_path),
                str(recognizer_path),
                str(image_path),
                charset=model_cfg["charset"],
            )

        results.append(
            {
                "image": str(image_path),
                "languages": list(languages),
                "recognizer": f"{model_cfg['recognizer']}{model_ext}",
                "charset": model_cfg["charset"],
                "providers": providers_used,
                "text": text,
            }
        )

    return results


def save_results(results: Iterable[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(list(results), f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("examples"),
        help="Directory containing sample images.",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=None,
        help="Directory containing ONNX models (detection/recognition).",
    )
    parser.add_argument(
        "--provider",
        choices=tuple(PROVIDERS.keys()),
        default="cpu",
        help="Execution provider to request from ONNX Runtime.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to store run results.",
    )
    args = parser.parse_args()

    model_dir = args.models or Path("models") / args.provider
    results = run_examples(args.examples_dir, model_dir, args.provider)

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output = Path("runs") / "onnx" / f"examples_{args.provider}_{timestamp}.json"
    save_results(results, args.output)

    print(f"Saved {len(results)} OCR results to {args.output}")


if __name__ == "__main__":
    main()
