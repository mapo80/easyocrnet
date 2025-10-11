using EasyOcrNet.ImageProcessing;
using Xunit;

namespace EasyOcrNet.Tests.ImageProcessing;

public class MorphologyTests
{
    [Fact]
    public void Dilate_EmptyImage_ReturnsEmpty()
    {
        var empty = new bool[0, 0];
        var result = MorphologyOps.Dilate(empty, 3, 3);

        Assert.Equal(0, result.GetLength(0));
        Assert.Equal(0, result.GetLength(1));
    }

    [Fact]
    public void Dilate_AllBackground_RemainsBackground()
    {
        var image = new bool[5, 5]; // All false
        var result = MorphologyOps.Dilate(image, 3, 3);

        // Should still be all background
        for (int y = 0; y < 5; y++)
            for (int x = 0; x < 5; x++)
                Assert.False(result[y, x]);
    }

    [Fact]
    public void Dilate_SinglePixel_Expands()
    {
        // Image:
        // . . . . .
        // . . . . .
        // . . X . .
        // . . . . .
        // . . . . .
        var image = new bool[5, 5];
        image[2, 2] = true;

        var result = MorphologyOps.Dilate(image, 3, 3);

        // After 3x3 dilation, should expand by 1 pixel in each direction:
        // . . . . .
        // . X X X .
        // . X X X .
        // . X X X .
        // . . . . .
        Assert.False(result[0, 2]);
        Assert.True(result[1, 1]);
        Assert.True(result[1, 2]);
        Assert.True(result[1, 3]);
        Assert.True(result[2, 1]);
        Assert.True(result[2, 2]);
        Assert.True(result[2, 3]);
        Assert.True(result[3, 1]);
        Assert.True(result[3, 2]);
        Assert.True(result[3, 3]);
        Assert.False(result[4, 2]);
    }

    [Fact]
    public void Dilate_HorizontalLine_ExpandsVertically()
    {
        // Image:
        // . . . . .
        // . X X X .
        // . . . . .
        var image = new bool[3, 5];
        image[1, 1] = true;
        image[1, 2] = true;
        image[1, 3] = true;

        var result = MorphologyOps.Dilate(image, 1, 3);

        // After 1x3 dilation (vertical), should expand vertically:
        // . X X X .
        // . X X X .
        // . X X X .
        for (int y = 0; y < 3; y++)
        {
            Assert.False(result[y, 0]);
            Assert.True(result[y, 1]);
            Assert.True(result[y, 2]);
            Assert.True(result[y, 3]);
            Assert.False(result[y, 4]);
        }
    }

    [Fact]
    public void Dilate_LargerKernel_ExpandsMore()
    {
        // Single pixel with 5x5 kernel
        var image = new bool[9, 9];
        image[4, 4] = true; // Center

        var result = MorphologyOps.Dilate(image, 5, 5);

        // Should expand by 2 pixels in each direction
        // Check center 5x5 area is all true
        for (int y = 2; y <= 6; y++)
            for (int x = 2; x <= 6; x++)
                Assert.True(result[y, x], $"Expected true at ({y},{x})");

        // Check corners are false
        Assert.False(result[0, 0]);
        Assert.False(result[0, 8]);
        Assert.False(result[8, 0]);
        Assert.False(result[8, 8]);
    }

    [Fact]
    public void Erode_SinglePixel_Disappears()
    {
        var image = new bool[3, 3];
        image[1, 1] = true;

        var result = MorphologyOps.Erode(image, 3, 3);

        // Single pixel cannot satisfy 3x3 erosion
        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 3; x++)
                Assert.False(result[y, x]);
    }

    [Fact]
    public void Erode_SolidBlock_Shrinks()
    {
        // Image: 5x5 solid block
        var image = new bool[7, 7];
        for (int y = 1; y <= 5; y++)
            for (int x = 1; x <= 5; x++)
                image[y, x] = true;

        var result = MorphologyOps.Erode(image, 3, 3);

        // After 3x3 erosion, only center 3x3 should remain
        // (5x5 block shrinks by 1 pixel on each edge)
        Assert.False(result[1, 1]); // Edge
        Assert.True(result[2, 2]);  // Inner
        Assert.True(result[3, 3]);  // Center
        Assert.True(result[4, 4]);  // Inner
        Assert.False(result[5, 5]); // Edge
    }

    [Fact]
    public void Open_RemovesSmallNoise()
    {
        // Image with main block and small isolated pixel (noise)
        // X X X X X
        // X X X X X
        // X X X X X
        // X X X X X
        // X X X X X
        // . . . . .
        // . X . . .  <- single pixel noise
        var image = new bool[7, 5];
        for (int y = 0; y < 5; y++)
            for (int x = 0; x < 5; x++)
                image[y, x] = true;
        image[6, 1] = true;  // Small noise (should be removed by erosion)

        var result = MorphologyOps.Open(image, 3, 3);

        // Noise pixel should be removed
        Assert.False(result[6, 1]);

        // Main block center should remain (5x5 block survives 3x3 open)
        Assert.True(result[2, 2]); // Center remains
    }

    [Fact]
    public void Close_FillsSmallHoles()
    {
        // Image with small hole in solid region
        // X X X X X
        // X X . X X  <- small hole
        // X X X X X
        var image = new bool[3, 5];
        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 5; x++)
                image[y, x] = true;
        image[1, 2] = false; // Small hole

        var result = MorphologyOps.Close(image, 3, 3);

        // Hole should be filled
        Assert.True(result[1, 2]);
    }

    [Fact]
    public void Dilate_AsymmetricKernel_WorksCorrectly()
    {
        // Single pixel with 5x3 kernel (wider than tall)
        var image = new bool[7, 9];
        image[3, 4] = true; // Center

        var result = MorphologyOps.Dilate(image, 5, 3);

        // Should expand more horizontally (±2) than vertically (±1)
        Assert.True(result[2, 4]); // Top (within ±1 vertically)
        Assert.True(result[3, 2]); // Left (within ±2 horizontally)
        Assert.True(result[3, 6]); // Right (within ±2 horizontally)
        Assert.True(result[4, 4]); // Bottom (within ±1 vertically)

        // Further out vertically should be false (beyond ±1)
        Assert.False(result[1, 4]); // Too far up
        Assert.False(result[5, 4]); // Too far down

        // Center area should be dilated
        Assert.True(result[3, 3]);
        Assert.True(result[3, 4]);
        Assert.True(result[3, 5]);
    }

    [Fact]
    public void Dilate_EdgePixels_HandledCorrectly()
    {
        // Pixel at edge
        // X . .
        // . . .
        var image = new bool[2, 3];
        image[0, 0] = true;

        var result = MorphologyOps.Dilate(image, 3, 3);

        // Should dilate within bounds
        Assert.True(result[0, 0]);
        Assert.True(result[0, 1]);
        Assert.True(result[1, 0]);
        Assert.True(result[1, 1]);
        Assert.False(result[1, 2]); // Beyond dilation range
    }

    [Fact]
    public void Dilate_InvalidKernelSize_ThrowsException()
    {
        var image = new bool[3, 3];

        // Even kernel size should throw
        Assert.Throws<ArgumentException>(() => MorphologyOps.Dilate(image, 2, 3));
        Assert.Throws<ArgumentException>(() => MorphologyOps.Dilate(image, 3, 4));

        // Zero or negative should throw
        Assert.Throws<ArgumentException>(() => MorphologyOps.Dilate(image, 0, 3));
        Assert.Throws<ArgumentException>(() => MorphologyOps.Dilate(image, 3, -1));
    }
}
