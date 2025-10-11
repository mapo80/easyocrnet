using EasyOcrNet;
using EasyOcrNet.Detection;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using Xunit;

namespace EasyOcrNet.Tests;

public class TextDetectorTests
{
    private sealed class DetectorStub : IOcrBackend
    {
        public ModelOutput DetectorOutput { get; set; }

        public string Provider => "stub";

        public void Dispose()
        {
        }

        public ModelOutput RunDetector(DenseTensor<float> input)
        {
            return DetectorOutput;
        }

        public ModelOutput RunRecognizer(DenseTensor<float> input)
        {
            throw new NotSupportedException();
        }
    }

    [Fact]
    public void DetectFallsBackToFullImageWhenNoComponents()
    {
        var backend = new DetectorStub
        {
            DetectorOutput = new ModelOutput(Array.Empty<float>(), new[] { 1, 1, 1 })
        };
        var detector = new TextDetector(backend);

        using var bitmap = new SKBitmap(120, 60);
        var rectangles = detector.Detect(bitmap);

        Assert.Single(rectangles);
        var rect = rectangles[0];
        Assert.Equal(0f, rect.Left);
        Assert.Equal(0f, rect.Top);
        Assert.InRange(rect.Right, 119.9f, 120.1f);
        Assert.InRange(rect.Bottom, 59.9f, 60.1f);
    }

    [Fact]
    public void DetectScalesComponentBoundingBoxes()
    {
        var data = new float[1 * 2 * 2 * 2];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = 0f;
        }

        data[0] = 0.9f; // strong text in top-left cell

        var backend = new DetectorStub
        {
            DetectorOutput = new ModelOutput(data, new[] { 1, 2, 2, 2 })
        };

        var detector = new TextDetector(backend);
        using var bitmap = new SKBitmap(800, 608);
        var rectangles = detector.Detect(bitmap);

        Assert.Single(rectangles);
        var rect = rectangles[0];
        Assert.Equal(0f, rect.Left);
        Assert.Equal(0f, rect.Top);
        Assert.Equal(800f, rect.Right, 3);
        Assert.Equal(608f, rect.Bottom, 3);
    }
}
