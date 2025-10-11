using EasyOcrNet.ImageProcessing;
using Xunit;
using System;
using System.Collections.Generic;

namespace EasyOcrNet.Tests.ImageProcessing;

public class GeometryUtilsTests
{
    [Fact]
    public void ExtractContour_EmptyMask_ReturnsEmpty()
    {
        var mask = new bool[0, 0];
        var contour = GeometryUtils.ExtractContour(mask);

        Assert.Empty(contour);
    }

    [Fact]
    public void ExtractContour_AllBackground_ReturnsEmpty()
    {
        var mask = new bool[5, 5]; // All false
        var contour = GeometryUtils.ExtractContour(mask);

        Assert.Empty(contour);
    }

    [Fact]
    public void ExtractContour_SinglePixel_ReturnsSinglePoint()
    {
        var mask = new bool[3, 3];
        mask[1, 1] = true;

        var contour = GeometryUtils.ExtractContour(mask);

        Assert.Single(contour);
        Assert.Contains(new Point(1, 1), contour);
    }

    [Fact]
    public void ExtractContour_SolidSquare_ReturnsOnlyBoundary()
    {
        // 3x3 solid square
        var mask = new bool[5, 5];
        for (int y = 1; y <= 3; y++)
            for (int x = 1; x <= 3; x++)
                mask[y, x] = true;

        var contour = GeometryUtils.ExtractContour(mask);

        // Only boundary pixels (3x3 has 8 boundary pixels, 1 internal)
        Assert.Equal(8, contour.Count);

        // Center should NOT be in contour
        Assert.DoesNotContain(new Point(2, 2), contour);

        // Edges should be in contour
        Assert.Contains(new Point(1, 1), contour); // Corner
        Assert.Contains(new Point(2, 1), contour); // Edge
    }

    [Fact]
    public void ConvexHull_LessThanThreePoints_ReturnsInput()
    {
        var points = new List<Point> { new Point(1, 1), new Point(2, 2) };
        var hull = GeometryUtils.ConvexHull(points);

        Assert.Equal(2, hull.Count);
    }

    [Fact]
    public void ConvexHull_Triangle_ReturnsSamePoints()
    {
        var points = new List<Point>
        {
            new Point(0, 0),
            new Point(4, 0),
            new Point(2, 3)
        };

        var hull = GeometryUtils.ConvexHull(points);

        Assert.Equal(3, hull.Count);
        Assert.Contains(new Point(0, 0), hull);
        Assert.Contains(new Point(4, 0), hull);
        Assert.Contains(new Point(2, 3), hull);
    }

    [Fact]
    public void ConvexHull_WithInteriorPoint_ExcludesInteriorPoint()
    {
        var points = new List<Point>
        {
            new Point(0, 0),
            new Point(4, 0),
            new Point(4, 4),
            new Point(0, 4),
            new Point(2, 2)  // Interior point
        };

        var hull = GeometryUtils.ConvexHull(points);

        // Hull should have 4 corners, not include interior
        Assert.Equal(4, hull.Count);
        Assert.DoesNotContain(new Point(2, 2), hull);
    }

    [Fact]
    public void ConvexHull_ComplexShape_ReturnsCorrectHull()
    {
        var points = new List<Point>
        {
            new Point(0, 0),
            new Point(1, 1),
            new Point(2, 0),
            new Point(3, 1),
            new Point(4, 0),
            new Point(4, 4),
            new Point(2, 3),
            new Point(0, 4)
        };

        var hull = GeometryUtils.ConvexHull(points);

        // Expected hull: (0,0), (4,0), (4,4), (0,4)
        Assert.Contains(new Point(0, 0), hull);
        Assert.Contains(new Point(4, 0), hull);
        Assert.Contains(new Point(4, 4), hull);
        Assert.Contains(new Point(0, 4), hull);

        // Interior points should be excluded
        Assert.DoesNotContain(new Point(1, 1), hull);
        Assert.DoesNotContain(new Point(2, 3), hull);
    }

    [Fact]
    public void MinAreaRect_EmptyPoints_ReturnsZeroRect()
    {
        var points = new List<Point>();
        var rect = GeometryUtils.MinAreaRect(points);

        Assert.Equal(0, rect.Width);
        Assert.Equal(0, rect.Height);
    }

    [Fact]
    public void MinAreaRect_SinglePoint_ReturnsZeroSizeRect()
    {
        var points = new List<Point> { new Point(5, 3) };
        var rect = GeometryUtils.MinAreaRect(points);

        Assert.Equal(5, rect.Center.X);
        Assert.Equal(3, rect.Center.Y);
        Assert.Equal(0, rect.Width);
        Assert.Equal(0, rect.Height);
    }

    [Fact]
    public void MinAreaRect_TwoPoints_ReturnsLineRect()
    {
        var points = new List<Point>
        {
            new Point(0, 0),
            new Point(4, 0)
        };

        var rect = GeometryUtils.MinAreaRect(points);

        Assert.Equal(2.0, rect.Center.X);
        Assert.Equal(0.0, rect.Center.Y);
        Assert.Equal(4.0, rect.Width);
        Assert.Equal(0.0, rect.Height);
    }

    [Fact]
    public void MinAreaRect_AxisAlignedSquare_ReturnsCorrectRect()
    {
        var points = new List<Point>
        {
            new Point(0, 0),
            new Point(4, 0),
            new Point(4, 4),
            new Point(0, 4)
        };

        var rect = GeometryUtils.MinAreaRect(points);

        Assert.Equal(2.0, rect.Center.X, 1);
        Assert.Equal(2.0, rect.Center.Y, 1);
        Assert.InRange(rect.Width, 3.9, 4.1);
        Assert.InRange(rect.Height, 3.9, 4.1);
    }

    [Fact]
    public void MinAreaRect_Rectangle_ReturnsCorrectDimensions()
    {
        var points = new List<Point>
        {
            new Point(0, 0),
            new Point(6, 0),
            new Point(6, 3),
            new Point(0, 3)
        };

        var rect = GeometryUtils.MinAreaRect(points);

        Assert.Equal(3.0, rect.Center.X, 1);
        Assert.Equal(1.5, rect.Center.Y, 1);

        // Width and height should be 6 and 3 (in some order)
        var dims = new[] { rect.Width, rect.Height };
        Array.Sort(dims);
        Assert.InRange(dims[0], 2.9, 3.1); // Smaller dimension
        Assert.InRange(dims[1], 5.9, 6.1); // Larger dimension
    }

    [Fact]
    public void MinAreaRect_RotatedSquare_FindsMinimalRect()
    {
        // Diamond shape (rotated 45°)
        // Points:
        //     (2,0)
        //  (0,2) (4,2)
        //     (2,4)
        var points = new List<Point>
        {
            new Point(2, 0),
            new Point(4, 2),
            new Point(2, 4),
            new Point(0, 2)
        };

        var rect = GeometryUtils.MinAreaRect(points);

        // Center should be at (2, 2)
        Assert.InRange(rect.Center.X, 1.8, 2.2);
        Assert.InRange(rect.Center.Y, 1.8, 2.2);

        // Both dimensions should be approximately sqrt(32) ≈ 5.66 (diagonal)
        // But rotating calipers should find the minimum box
        // For a 45° rotated square with corners at distance sqrt(8) from center,
        // the min area rect has sides sqrt(8) × sqrt(8) = 2.83 × 2.83
        Assert.InRange(rect.Width, 2.5, 3.0);
        Assert.InRange(rect.Height, 2.5, 3.0);
    }

    [Fact]
    public void MinAreaRect_LongThinRectangle_FindsCorrectOrientation()
    {
        // Very thin horizontal rectangle
        var points = new List<Point>
        {
            new Point(0, 5),
            new Point(10, 5),
            new Point(10, 6),
            new Point(0, 6)
        };

        var rect = GeometryUtils.MinAreaRect(points);

        // Should detect this is essentially horizontal
        var dims = new[] { rect.Width, rect.Height };
        Array.Sort(dims);

        Assert.InRange(dims[0], 0.9, 1.1);  // Height ≈ 1
        Assert.InRange(dims[1], 9.9, 10.1); // Width ≈ 10
    }

    [Fact]
    public void RotatedRect_GetCorners_ReturnsCorrectPoints()
    {
        // Axis-aligned 4x2 rectangle centered at (2, 1)
        var rect = new RotatedRect(2, 1, 4, 2, 0);
        var corners = rect.GetCorners();

        Assert.Equal(4, corners.Length);

        // Corners should be approximately (0,0), (4,0), (4,2), (0,2)
        Assert.Contains(corners, c => Math.Abs(c.X - 0) < 0.1 && Math.Abs(c.Y - 0) < 0.1);
        Assert.Contains(corners, c => Math.Abs(c.X - 4) < 0.1 && Math.Abs(c.Y - 0) < 0.1);
        Assert.Contains(corners, c => Math.Abs(c.X - 4) < 0.1 && Math.Abs(c.Y - 2) < 0.1);
        Assert.Contains(corners, c => Math.Abs(c.X - 0) < 0.1 && Math.Abs(c.Y - 2) < 0.1);
    }

    [Fact]
    public void RotatedRect_GetCorners_RotatedRect_ReturnsRotatedPoints()
    {
        // 2x2 square rotated 45° (π/4)
        var rect = new RotatedRect(0, 0, 2, 2, Math.PI / 4);
        var corners = rect.GetCorners();

        Assert.Equal(4, corners.Length);

        // Corners should be rotated
        // All points should be at distance sqrt(2) from origin (since 2x2 square rotated 45°)
        foreach (var corner in corners)
        {
            double dist = Math.Sqrt(corner.X * corner.X + corner.Y * corner.Y);
            Assert.InRange(dist, 1.3, 1.5); // sqrt(2) ≈ 1.414
        }
    }
}
