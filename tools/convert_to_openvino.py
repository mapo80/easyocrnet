"""Convert ONNX models from models/cpu into OpenVINO IR format."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from openvino.runtime import serialize
from openvino.tools.mo import convert_model


def convert_single(onnx_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = convert_model(onnx_path)
    xml_path = output_dir / (onnx_path.stem + ".xml")
    bin_path = output_dir / (onnx_path.stem + ".bin")
    serialize(model, xml_path, bin_path)
    return xml_path


def convert_all(source_dir: Path, output_dir: Path) -> Iterable[Path]:
    for onnx_path in sorted(source_dir.glob("*.onnx")):
        print(f"Converting {onnx_path.name} -> IR")
        yield convert_single(onnx_path, output_dir)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert ONNX models into OpenVINO IR")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("models") / "cpu",
        help="Directory containing ONNX models",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models") / "openvino",
        help="Destination directory for OpenVINO IR files",
    )
    args = parser.parse_args()

    for xml_path in convert_all(args.source, args.output):
        print(f"✓ Saved {xml_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
