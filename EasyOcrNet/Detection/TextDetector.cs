using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace EasyOcrNet.Detection;

internal sealed class TextDetector
{
    private readonly IOcrBackend _backend;

    public TextDetector(IOcrBackend backend)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
    }

    public IReadOnlyList<SKRect> Detect(SKBitmap image)
    {
        if (image is null)
        {
            throw new ArgumentNullException(nameof(image));
        }

        DetectorInput detectorInput = DetectorInputBuilder.Build(image);
        ModelOutput detectorOutput = _backend.RunDetector(detectorInput.Tensor);

        var components = DetectionPostProcessor.ExtractBoundingBoxes(detectorOutput, detectorInput.ResizedWidth, detectorInput.ResizedHeight);
        if (components.Count == 0)
        {
            components.Add(new SKRect(0, 0, detectorInput.ResizedWidth, detectorInput.ResizedHeight));
        }

        var lineBoxes = TextComponentGrouper.GroupIntoLines(components);
        var scaled = new List<SKRect>(lineBoxes.Count);
        foreach (var line in lineBoxes)
        {
            var scaledRect = new SKRect(
                line.Left * detectorInput.ScaleX,
                line.Top * detectorInput.ScaleY,
                line.Right * detectorInput.ScaleX,
                line.Bottom * detectorInput.ScaleY);
            scaled.Add(scaledRect);
        }

        return scaled;
    }
}
