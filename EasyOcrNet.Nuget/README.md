# EasyOcrNet NuGet Package

This folder contains the source for the **EasyOcrNet** NuGet package, which includes ONNX models and character sets required for the EasyOcrNet OCR library.

## Included Models

- **detection.onnx** (79 MB) - CRAFT text detection model
- **latin_g2_rec.onnx** (15 MB) - Recognition model for Latin-based languages (Italian, English, Spanish, French, German, etc.)
- **english_g2_rec.onnx** (14 MB) - Recognition model optimized for English

## Included Character Sets

- **en_charset.txt** - English character set
- **it_charset.txt** - Italian character set
- **latin_char.txt** - Latin character set (fallback)

## Total Package Size

Approximately 108 MB

## Usage

After installing this package, the models and character files will be automatically copied to your output directory.

```csharp
using var engine = new OcrEngine(
    detectorPath: "models/detection.onnx",
    recognizerPath: "models/latin_g2_rec.onnx",
    language: "it",
    charsetDirectory: "character"
);
```

## License

Models are derived from [TorchfreeEasyOCR](https://github.com/SeldonHZ/TorchfreeEasyOCR).
