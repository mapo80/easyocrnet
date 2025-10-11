using System.Collections.Generic;
using System.Linq;
using EasyOcrNet;
using EasyOcrNet.Configuration;
using EasyOcrNet.Detection;
using SkiaSharp;
using Xunit;
using Xunit.Abstractions;

namespace EasyOcrNet.Tests;

public class DetectionPostProcessorTests
{
    private readonly ITestOutputHelper _output;

    public DetectionPostProcessorTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void ExtractBoundingBoxes_SingleComponent_ReturnsExpectedRectangle()
    {
        var output = CreateDetectorOutput(6, 6, new[] { (1, 1, 4, 4) });
        var boxes = DetectionPostProcessor.ExtractBoundingBoxes(output, 600, 600);

        var box = Assert.Single(boxes);
        _output.WriteLine($"Box: L={box.Left:F1}, T={box.Top:F1}, R={box.Right:F1}, B={box.Bottom:F1}");
        Assert.InRange(box.Left, 100f - 1f, 100f + 1f);
        Assert.InRange(box.Top, 100f - 1f, 100f + 1f);
        Assert.InRange(box.Right, 400f - 1f, 400f + 1f);
        Assert.InRange(box.Bottom, 400f - 1f, 400f + 1f);
    }

    [Fact]
    public void ExtractBoundingBoxes_FiltersComponentsBelowScoreThreshold()
    {
        var output = CreateDetectorOutput(6, 6, new[] { (1, 1, 4, 4) }, textScore: 0.5f);
        var boxes = DetectionPostProcessor.ExtractBoundingBoxes(output, 600, 600);
        Assert.Empty(boxes);
    }

    [Fact]
    public void ExtractBoundingBoxes_MultipleComponents_ReturnsAllRegions()
    {
        var output = CreateDetectorOutput(
            10,
            8,
            new[]
            {
                (1, 1, 4, 5),
                (6, 2, 8, 6)
            });

        var boxes = DetectionPostProcessor.ExtractBoundingBoxes(output, 800, 640)
            .OrderBy(b => b.Left)
            .ToList();

        Assert.Equal(2, boxes.Count);
        _output.WriteLine($"Box A: L={boxes[0].Left:F1}, T={boxes[0].Top:F1}, R={boxes[0].Right:F1}, B={boxes[0].Bottom:F1}");
        _output.WriteLine($"Box B: L={boxes[1].Left:F1}, T={boxes[1].Top:F1}, R={boxes[1].Right:F1}, B={boxes[1].Bottom:F1}");

        Assert.InRange(boxes[0].Left, 80f - 1f, 80f + 1f);
        Assert.InRange(boxes[0].Top, 80f - 1f, 80f + 1f);
        Assert.InRange(boxes[0].Right, 320f - 1f, 320f + 1f);
        Assert.InRange(boxes[0].Bottom, 400f - 1f, 400f + 1f);

        Assert.InRange(boxes[1].Left, 480f - 1f, 480f + 1f);
        Assert.InRange(boxes[1].Top, 160f - 1f, 160f + 1f);
        Assert.InRange(boxes[1].Right, 640f - 1f, 640f + 1f);
        Assert.InRange(boxes[1].Bottom, 480f - 1f, 480f + 1f);
    }

    private static ModelOutput CreateDetectorOutput(int width, int height, IReadOnlyList<(int X0, int Y0, int X1, int Y1)> blocks, float textScore = 0.9f, float linkScore = 0.8f)
    {
        const int channels = 2;
        var data = new float[width * height * channels];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = (y * width + x) * channels;
                data[index] = 0f;
                data[index + 1] = 0f;
            }
        }

        foreach (var block in blocks)
        {
            for (int y = block.Y0; y <= block.Y1; y++)
            {
                for (int x = block.X0; x <= block.X1; x++)
                {
                    int index = (y * width + x) * channels;
                    data[index] = textScore;
                    data[index + 1] = linkScore;
                }
            }
        }

        var shape = new[] { 1, height, width, channels };
        return new ModelOutput(data, shape);
    }
}
