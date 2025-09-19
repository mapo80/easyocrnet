"""Download TorchfreeEasyOCR ONNX models into models/cpu."""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from typing import Dict
from urllib.request import urlopen

TORCHFREE_RELEASE = "https://github.com/SeldonHZ/TorchfreeEasyOCR/releases/download/pre-v1.1.0"

MODELS: Dict[str, Dict[str, str]] = {
    "detection.onnx": {
        "url": f"{TORCHFREE_RELEASE}/detection.zip",
        "md5": "c8fa14f85030d87c52f8990db50d68ef",
    },
    "english_g2_rec.onnx": {
        "url": f"{TORCHFREE_RELEASE}/english_g2_rec.zip",
        "md5": "8deccfa817467f834edb79b39220312e",
    },
    "latin_g2_rec.onnx": {
        "url": f"{TORCHFREE_RELEASE}/latin_g2_rec.zip",
        "md5": "613a143cb017110c0cbadda32165b580",
    },
    "zh_sim_g2_rec.onnx": {
        "url": f"{TORCHFREE_RELEASE}/zh_sim_g2_rec.zip",
        "md5": "9ba3fee6bfcca1d590d1cefcd862f43c",
    },
    "japanese_g2_rec.onnx": {
        "url": f"{TORCHFREE_RELEASE}/japanese_g2_rec.zip",
        "md5": "c3f65a6ef8fdb9947ae8bfdcf559947d",
    },
    "korean_g2_rec.onnx": {
        "url": f"{TORCHFREE_RELEASE}/korean_g2_rec.zip",
        "md5": "de1a84cab05f9da31851c7e99fe6a62b",
    },
    "thai_g1_rec.onnx": {
        "url": f"{TORCHFREE_RELEASE}/thai_g1_rec.zip",
        "md5": "15388c67adea8c93b982fc44bcffff53",
    },
}


def _md5(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_models(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    for filename, meta in MODELS.items():
        target = destination / filename
        if target.exists():
            current = _md5(target)
            if current == meta["md5"]:
                print(f"✓ {filename} already present (md5 ok)")
                continue
            print(f"! {filename} exists but checksum mismatch, re-downloading")

        print(f"↓ Downloading {filename}...")
        with urlopen(meta["url"]) as response:
            data = io.BytesIO(response.read())
        with zipfile.ZipFile(data) as archive:
            inner = [name for name in archive.namelist() if name.endswith(".onnx")]
            if not inner:
                raise RuntimeError(f"No .onnx file found in archive for {filename}")
            if len(inner) > 1:
                raise RuntimeError(f"Multiple .onnx files in archive for {filename}: {inner}")
            with archive.open(inner[0]) as src, target.open("wb") as dst:
                dst.write(src.read())

        checksum = _md5(target)
        if checksum != meta["md5"]:
            raise RuntimeError(
                f"Checksum mismatch for {filename}: expected {meta['md5']} got {checksum}"
            )
        print(f"✓ Saved {filename} ({target.stat().st_size / (1024 * 1024):.1f} MB)")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download TorchfreeEasyOCR ONNX models")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models") / "cpu",
        help="Destination directory for ONNX models",
    )
    args = parser.parse_args()

    download_models(args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
