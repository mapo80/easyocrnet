using EasyOcrNet;
using EasyOcrNet.Configuration;
using EasyOcrNet.Detection;
using EasyOcrNet.Recognition;
using SkiaSharp;
using System.IO;
using System.Linq;
using Xunit;

namespace EasyOcrNet.Tests;

public class OnnxBackendTests
{
    [Fact]
    public void RunDetectorAndRecognizerProduceOutputs()
    {
        var resources = OcrModelCatalog.ResolveResources(new OcrOptions(TestPaths.ModelsDirectory));
        using var backend = new OnnxBackend(resources.DetectionPath, resources.RecognitionPath);

        using var bitmap = SKBitmap.Decode(Path.Combine(TestPaths.ExamplesDirectory, "english.png"));
        Assert.NotNull(bitmap);

        var detectorInput = DetectorInputBuilder.Build(bitmap!);
        var detection = backend.RunDetector(detectorInput.Tensor);

        Assert.Equal(4, detection.Rank);
        Assert.Equal(1, detection[0]);
        Assert.Equal(2, detection[3]);

        var recognitionInput = RecognitionInputBuilder.Build(bitmap!, new SKRect(0, 0, bitmap.Width, bitmap.Height));
        var recognition = backend.RunRecognizer(recognitionInput);

        Assert.Equal(3, recognition.Rank);
        Assert.Equal(1, recognition[0]);
        Assert.Contains(recognition.Data, v => !float.IsNaN(v));
    }
}
