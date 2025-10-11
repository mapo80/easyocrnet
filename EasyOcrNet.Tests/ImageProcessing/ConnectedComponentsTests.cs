using EasyOcrNet.ImageProcessing;
using Xunit;

namespace EasyOcrNet.Tests.ImageProcessing;

public class ConnectedComponentsTests
{
    [Fact]
    public void Analyze_EmptyImage_ReturnsZeroComponents()
    {
        var empty = new bool[0, 0];
        var result = ConnectedComponentsAnalyzer.Analyze(empty);

        Assert.Equal(0, result.Count);
        Assert.Empty(result.Stats);
    }

    [Fact]
    public void Analyze_AllBackground_ReturnsZeroComponents()
    {
        var image = new bool[5, 5]; // All false
        var result = ConnectedComponentsAnalyzer.Analyze(image);

        Assert.Equal(0, result.Count);
        Assert.Empty(result.Stats);
    }

    [Fact]
    public void Analyze_SinglePixel_ReturnsOneComponent()
    {
        var image = new bool[3, 3];
        image[1, 1] = true; // Single pixel at center

        var result = ConnectedComponentsAnalyzer.Analyze(image);

        Assert.Equal(1, result.Count);
        Assert.Single(result.Stats);

        var comp = result.Stats[0];
        Assert.Equal(1, comp.Label);
        Assert.Equal(1, comp.X);
        Assert.Equal(1, comp.Y);
        Assert.Equal(1, comp.Width);
        Assert.Equal(1, comp.Height);
        Assert.Equal(1, comp.Area);
        Assert.Equal(1.0, comp.CentroidX);
        Assert.Equal(1.0, comp.CentroidY);
    }

    [Fact]
    public void Analyze_TwoSeparateComponents_ReturnsTwoLabels()
    {
        // Image:
        // X . .
        // . . .
        // . . X
        var image = new bool[3, 3];
        image[0, 0] = true; // Top-left
        image[2, 2] = true; // Bottom-right

        var result = ConnectedComponentsAnalyzer.Analyze(image);

        Assert.Equal(2, result.Count);
        Assert.Equal(2, result.Stats.Length);

        // Verify both components have area 1
        Assert.All(result.Stats, stat => Assert.Equal(1, stat.Area));

        // Verify labels are assigned (1 and 2)
        Assert.Contains(result.Stats, s => s.Label == 1);
        Assert.Contains(result.Stats, s => s.Label == 2);
    }

    [Fact]
    public void Analyze_ConnectedHorizontalLine_ReturnsSingleComponent()
    {
        // Image:
        // X X X X
        // . . . .
        var image = new bool[2, 4];
        for (int x = 0; x < 4; x++)
            image[0, x] = true;

        var result = ConnectedComponentsAnalyzer.Analyze(image);

        Assert.Equal(1, result.Count);
        var comp = result.Stats[0];
        Assert.Equal(4, comp.Area);
        Assert.Equal(0, comp.X);
        Assert.Equal(0, comp.Y);
        Assert.Equal(4, comp.Width);
        Assert.Equal(1, comp.Height);
    }

    [Fact]
    public void Analyze_ConnectedVerticalLine_ReturnsSingleComponent()
    {
        // Image:
        // X .
        // X .
        // X .
        var image = new bool[3, 2];
        for (int y = 0; y < 3; y++)
            image[y, 0] = true;

        var result = ConnectedComponentsAnalyzer.Analyze(image);

        Assert.Equal(1, result.Count);
        var comp = result.Stats[0];
        Assert.Equal(3, comp.Area);
        Assert.Equal(0, comp.X);
        Assert.Equal(0, comp.Y);
        Assert.Equal(1, comp.Width);
        Assert.Equal(3, comp.Height);
    }

    [Fact]
    public void Analyze_LShapedComponent_ReturnsSingleComponent()
    {
        // Image:
        // X X .
        // X . .
        // X . .
        var image = new bool[3, 3];
        image[0, 0] = true;
        image[0, 1] = true;
        image[1, 0] = true;
        image[2, 0] = true;

        var result = ConnectedComponentsAnalyzer.Analyze(image);

        Assert.Equal(1, result.Count);
        var comp = result.Stats[0];
        Assert.Equal(4, comp.Area);
        Assert.Equal(0, comp.X);
        Assert.Equal(0, comp.Y);
        Assert.Equal(2, comp.Width);
        Assert.Equal(3, comp.Height);
    }

    [Fact]
    public void Analyze_DiagonalPixels_ReturnsTwoComponents()
    {
        // Image (4-connectivity, diagonals don't connect):
        // X . .
        // . X .
        // . . X
        var image = new bool[3, 3];
        image[0, 0] = true;
        image[1, 1] = true;
        image[2, 2] = true;

        var result = ConnectedComponentsAnalyzer.Analyze(image);

        // With 4-connectivity, diagonal pixels are separate components
        Assert.Equal(3, result.Count);
        Assert.All(result.Stats, stat => Assert.Equal(1, stat.Area));
    }

    [Fact]
    public void Analyze_ComplexShape_CorrectStats()
    {
        // Image:
        // . X X .
        // X X X X
        // . X X .
        var image = new bool[3, 4];
        image[0, 1] = true;
        image[0, 2] = true;
        image[1, 0] = true;
        image[1, 1] = true;
        image[1, 2] = true;
        image[1, 3] = true;
        image[2, 1] = true;
        image[2, 2] = true;

        var result = ConnectedComponentsAnalyzer.Analyze(image);

        Assert.Equal(1, result.Count);
        var comp = result.Stats[0];
        Assert.Equal(8, comp.Area);
        Assert.Equal(0, comp.X);
        Assert.Equal(0, comp.Y);
        Assert.Equal(4, comp.Width);
        Assert.Equal(3, comp.Height);

        // Centroid should be near center
        Assert.InRange(comp.CentroidX, 1.0, 2.0);
        Assert.InRange(comp.CentroidY, 0.8, 1.2);
    }

    [Fact]
    public void Analyze_MultipleTextLines_ReturnsMultipleComponents()
    {
        // Simulate 3 text lines
        // X X X . . .
        // . . . . . .
        // . . . X X X
        // . . . . . .
        // X X . . . .
        var image = new bool[5, 6];

        // Line 1
        image[0, 0] = true;
        image[0, 1] = true;
        image[0, 2] = true;

        // Line 2
        image[2, 3] = true;
        image[2, 4] = true;
        image[2, 5] = true;

        // Line 3
        image[4, 0] = true;
        image[4, 1] = true;

        var result = ConnectedComponentsAnalyzer.Analyze(image);

        Assert.Equal(3, result.Count);

        // Verify areas
        Assert.Contains(result.Stats, s => s.Area == 3);
        Assert.Contains(result.Stats, s => s.Area == 2);
    }

    [Fact]
    public void Analyze_LabelMap_CorrectlyLabelsPixels()
    {
        // Image:
        // X . X
        // . . .
        // X . X
        var image = new bool[3, 3];
        image[0, 0] = true;
        image[0, 2] = true;
        image[2, 0] = true;
        image[2, 2] = true;

        var result = ConnectedComponentsAnalyzer.Analyze(image);

        Assert.Equal(4, result.Count);

        // Verify label map
        Assert.True(result.Labels[0, 0] > 0); // Labeled
        Assert.Equal(0, result.Labels[0, 1]); // Background
        Assert.True(result.Labels[0, 2] > 0); // Labeled

        // Each corner should have different label
        Assert.NotEqual(result.Labels[0, 0], result.Labels[0, 2]);
        Assert.NotEqual(result.Labels[0, 0], result.Labels[2, 0]);
        Assert.NotEqual(result.Labels[0, 0], result.Labels[2, 2]);
    }
}
