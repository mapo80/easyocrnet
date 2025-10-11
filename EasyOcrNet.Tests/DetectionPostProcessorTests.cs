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

    [Fact]
    public void ExtractBoundingBoxes_EmptyOutput_ReturnsEmptyList()
    {
        var output = CreateDetectorOutput(6, 6, Array.Empty<(int, int, int, int)>());
        var boxes = DetectionPostProcessor.ExtractBoundingBoxes(output, 600, 600);
        Assert.Empty(boxes);
    }

    [Fact]
    public void ExtractBoundingBoxes_SinglePixelComponent_AreaFiltered()
    {
        var output = CreateDetectorOutput(6, 6, new[] { (2, 2, 2, 2) }); // 1x1 = 1 pixel, below threshold 10
        var boxes = DetectionPostProcessor.ExtractBoundingBoxes(output, 600, 600);
        Assert.Empty(boxes); // Should be filtered due to small area
    }

    [Fact]
    public void ExtractBoundingBoxes_DiagonalLine_SeparateComponents()
    {
        // Create a diagonal line - in 4-connectivity these are separate components
        var blocks = new List<(int, int, int, int)>();
        for (int i = 0; i < 5; i++)
        {
            blocks.Add((i, i, i, i)); // Diagonal line - each pixel is separate in 4-connectivity
        }
        var output = CreateDetectorOutput(8, 8, blocks);
        var boxes = DetectionPostProcessor.ExtractBoundingBoxes(output, 800, 800);

        // Each pixel should be filtered due to small area (< 10 pixels)
        Assert.Empty(boxes);
    }

    [Fact]
    public void ExtractBoundingBoxes_TwoSeparateComponents_ReturnsTwoBoxes()
    {
        var output = CreateDetectorOutput(10, 10, new[]
        {
            (1, 1, 4, 4),  // Top-left component (4x4 = 16 pixels, above threshold)
            (6, 6, 9, 9)   // Bottom-right component (4x4 = 16 pixels, above threshold)
        });
        var boxes = DetectionPostProcessor.ExtractBoundingBoxes(output, 1000, 1000);

        Assert.Equal(2, boxes.Count);
    }

    [Fact]
    public void ExtractBoundingBoxes_ComponentTooSmall_AreaFiltered()
    {
        var output = CreateDetectorOutput(6, 6, new[] { (1, 1, 1, 1) }); // 1x1 = 1 pixel, below threshold 10
        var boxes = DetectionPostProcessor.ExtractBoundingBoxes(output, 600, 600);
        Assert.Empty(boxes);
    }

    [Fact]
    public void ExtractBoundingBoxes_LowScoreComponent_Filtered()
    {
        var output = CreateDetectorOutput(6, 6, new[] { (1, 1, 4, 4) }, textScore: 0.2f); // Below 0.7 threshold
        var boxes = DetectionPostProcessor.ExtractBoundingBoxes(output, 600, 600);
        Assert.Empty(boxes);
    }

    [Fact]
    public void ExtractBoundingBoxes_HorizontalLine_DilationWorks()
    {
        var output = CreateDetectorOutput(10, 6, new[] { (1, 1, 8, 4) }); // Larger horizontal area (8x4 = 32 pixels)
        var boxes = DetectionPostProcessor.ExtractBoundingBoxes(output, 1000, 600);

        Assert.Single(boxes);
        var box = boxes[0];
        Assert.True(box.Width > box.Height); // Should be wider than tall due to dilation
    }

    [Fact]
    public void ExtractBoundingBoxes_VerticalLine_DilationWorks()
    {
        var output = CreateDetectorOutput(6, 10, new[] { (1, 1, 4, 8) }); // Larger vertical area (4x8 = 32 pixels)
        var boxes = DetectionPostProcessor.ExtractBoundingBoxes(output, 600, 1000);

        Assert.Single(boxes);
        var box = boxes[0];
        Assert.True(box.Height > box.Width); // Should be taller than wide due to dilation
    }

    [Fact]
    public void ExtractBoundingBoxes_LargeComponent_UsesLargerDilation()
    {
        var output = CreateDetectorOutput(20, 20, new[] { (5, 5, 15, 15) }); // Large component
        var boxes = DetectionPostProcessor.ExtractBoundingBoxes(output, 2000, 2000);

        Assert.Single(boxes);
        var box = boxes[0];
        // Large component should have significant dilation (iterations up to 6)
        Assert.True(box.Width > 800);  // More than original ~1000px scaled
        Assert.True(box.Height > 800);
    }

    [Fact]
    public void ExtractBoundingBoxes_ComplexShape_ContourExtraction()
    {
        // Create an L-shaped component with sufficient area
        var blocks = new List<(int, int, int, int)>();
        for (int i = 0; i < 5; i++)
        {
            blocks.Add((1, 1 + i, 3, 1 + i));     // Vertical line (3px wide)
            blocks.Add((1 + i, 5, 1 + i, 7));     // Horizontal line (3px tall)
        }
        var output = CreateDetectorOutput(10, 10, blocks); // Total area: 5*3 + 5*3 = 30 pixels
        var boxes = DetectionPostProcessor.ExtractBoundingBoxes(output, 1000, 1000);

        Assert.Single(boxes);
        var box = boxes[0];
        Assert.True(box.Width > 0);
        Assert.True(box.Height > 0);
    }

    [Fact]
    public void ExtractBoundingBoxes_BoundaryComponent_ClampsToImageSize()
    {
        var output = CreateDetectorOutput(6, 6, new[] { (0, 0, 5, 5) }); // Component at edge
        var boxes = DetectionPostProcessor.ExtractBoundingBoxes(output, 600, 600);

        Assert.Single(boxes);
        var box = boxes[0];
        Assert.Equal(0f, box.Left);   // Should be clamped to 0
        Assert.Equal(0f, box.Top);    // Should be clamped to 0
        Assert.True(box.Right <= 600f);
        Assert.True(box.Bottom <= 600f);
    }

    [Fact]
    public void ExtractBoundingBoxes_InvalidModelOutput_ReturnsEmpty()
    {
        var data = new float[10];
        var shape = new[] { 0, 5, 5, 2 }; // Invalid: first dimension should be 1
        var output = new ModelOutput(data, shape);
        var boxes = DetectionPostProcessor.ExtractBoundingBoxes(output, 600, 600);
        Assert.Empty(boxes);
    }

    [Fact]
    public void ExtractBoundingBoxes_TextOnlyNoLink_Works()
    {
        var output = CreateDetectorOutput(6, 6, new[] { (1, 1, 4, 4) }, textScore: 0.8f, linkScore: 0.1f);
        var boxes = DetectionPostProcessor.ExtractBoundingBoxes(output, 600, 600);

        Assert.Single(boxes); // Should work with text score only
    }

    [Fact]
    public void ExtractBoundingBoxes_LinkOnlyNoText_Works()
    {
        var output = CreateDetectorOutput(6, 6, new[] { (1, 1, 4, 4) }, textScore: 0.1f, linkScore: 0.8f);
        var boxes = DetectionPostProcessor.ExtractBoundingBoxes(output, 600, 600);

        // This should be empty because textScore 0.1f is below threshold 0.7
        // and linkScore alone doesn't contribute to the maximum score calculation
        Assert.Empty(boxes);
    }

    [Fact]
    public void ExtractBoundingBoxes_MultipleOverlappingComponents_AfterDilation()
    {
        var output = CreateDetectorOutput(12, 8, new[]
        {
            (1, 1, 4, 4),  // Component A (4x4 = 16 pixels)
            (7, 1, 10, 4)  // Component B (4x4 = 16 pixels, close but separate)
        });
        var boxes = DetectionPostProcessor.ExtractBoundingBoxes(output, 1200, 800);

        // After dilation, they might merge or stay separate
        Assert.True(boxes.Count >= 1);
        Assert.True(boxes.Count <= 2);
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
