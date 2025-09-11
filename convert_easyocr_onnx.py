import argparse
from pathlib import Path

import torch
import onnx
import easyocr


class DetectorWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.model(x)
        return y


class RecognizerWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        text = torch.zeros(b, 26, dtype=torch.long)
        return self.model(x, text)


def export_models(det_path: Path, rec_path: Path, lang: str, model_dir: Path) -> None:
    reader = easyocr.Reader([lang], gpu=False, model_storage_directory=str(model_dir), quantize=False)

    det_model = DetectorWrapper(reader.detector)
    rec_model = RecognizerWrapper(reader.recognizer)

    det_input = torch.randn(1, 3, 608, 800)
    rec_input = torch.randn(1, 1, 64, 1000)

    torch.onnx.export(
        det_model,
        det_input,
        det_path,
        input_names=["image"],
        output_names=["output"],
        opset_version=11,
    )
    from torch.onnx import ExportOptions
    ep = torch.onnx.dynamo_export(rec_model, rec_input, export_options=ExportOptions(dynamic_shapes=True))
    model = ep.model_proto
    model.graph.input[0].name = "image"
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name == "x":
                node.input[i] = "image"
    model.graph.node[-1].output[0] = "output"
    model.graph.output[0].name = "output"
    onnx.save(model, rec_path)
    print(f"Saved detector to {det_path}")
    print(f"Saved recognizer to {rec_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert EasyOCR PyTorch models to ONNX")
    parser.add_argument("--lang", default="en", help="language code to load models for")
    parser.add_argument("--output-det", default="models/EasyOCRDetector.onnx")
    parser.add_argument("--output-rec", default="models/EasyOCRRecognizer.onnx")
    parser.add_argument("--model-dir", default="models", help="directory to cache downloaded weights")
    args = parser.parse_args()

    export_models(Path(args.output_det), Path(args.output_rec), args.lang, Path(args.model_dir))


if __name__ == "__main__":
    main()
