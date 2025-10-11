using SkiaSharp;

namespace EasyOcrNet;

public sealed record OcrResult(string Text, SKRect BoundingBox, double Confidence = 1.0);
