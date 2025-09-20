#!/usr/bin/env python3
"""Benchmark TorchfreeEasyOCR inference on a single image."""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from run_onnx_examples import PROVIDERS
from torchfreeeasyocr_extract import infer_text

DEFAULT_RUNS = 6
DEFAULT_DISCARD = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("examples/english.png"),
        help="Image to benchmark. Defaults to examples/english.png",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=None,
        help="Directory that stores TorchfreeEasyOCR ONNX models (defaults to models/<provider>).",
    )
    parser.add_argument(
        "--provider",
        choices=tuple(PROVIDERS.keys()),
        default="cpu",
        help="Execution provider to benchmark (cpu or openvino).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Number of runs to execute (default: 6).",
    )
    parser.add_argument(
        "--discard",
        type=int,
        default=DEFAULT_DISCARD,
        help="Number of initial warm-up runs to discard from the average (default: 1).",
    )
    return parser.parse_args()


def run_benchmark(image_path: Path, models_dir: Path, provider: str, runs: int, discard: int) -> tuple[str, float]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if runs <= 0:
        raise ValueError("--runs must be greater than zero")
    if discard < 0 or discard >= runs:
        raise ValueError("--discard must be between 0 (inclusive) and runs - 1")

    print(f"Benchmarking {image_path} with provider '{provider}' and models in {models_dir}")
    print(f"Total runs: {runs} (discarding the first {discard} run{'s' if discard != 1 else ''} from averaging)")

    durations: list[float] = []
    recognized_text: str = ""

    for i in range(runs):
        start = time.perf_counter()
        recognized_text = infer_text(image_path, models_dir, provider)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(f"Run {i + 1}: {elapsed_ms:.2f} ms")
        if i >= discard:
            durations.append(elapsed_ms)

    if not durations:
        raise RuntimeError("No runs available for averaging; check the --runs/--discard values")

    average = statistics.fmean(durations)
    minimum = min(durations)
    maximum = max(durations)
    print()
    print(
        f"Average over {len(durations)} runs (discarding first {discard}): "
        f"{average:.2f} ms (min {minimum:.2f} ms | max {maximum:.2f} ms)"
    )

    return recognized_text, average


def main() -> None:
    args = parse_args()
    models_dir = args.models or Path("models") / args.provider
    text, _ = run_benchmark(args.image, models_dir, args.provider, args.runs, args.discard)
    print()
    print("Recognized text preview:")
    print(text)


if __name__ == "__main__":
    main()
