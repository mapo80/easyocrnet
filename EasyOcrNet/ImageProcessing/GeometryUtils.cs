using System;
using System.Collections.Generic;
using System.Linq;

namespace EasyOcrNet.ImageProcessing;

/// <summary>
/// Represents a 2D point with integer coordinates.
/// </summary>
public readonly struct Point
{
    public int X { get; }
    public int Y { get; }

    public Point(int x, int y)
    {
        X = x;
        Y = y;
    }

    public override string ToString() => $"({X},{Y})";
}

/// <summary>
/// Represents a rotated rectangle.
/// </summary>
public readonly struct RotatedRect
{
    /// <summary>
    /// Center point of the rectangle.
    /// </summary>
    public (double X, double Y) Center { get; }

    /// <summary>
    /// Width of the rectangle.
    /// </summary>
    public double Width { get; }

    /// <summary>
    /// Height of the rectangle.
    /// </summary>
    public double Height { get; }

    /// <summary>
    /// Rotation angle in radians.
    /// </summary>
    public double Angle { get; }

    public RotatedRect(double centerX, double centerY, double width, double height, double angle)
    {
        Center = (centerX, centerY);
        Width = width;
        Height = height;
        Angle = angle;
    }

    /// <summary>
    /// Gets the four corner points of the rotated rectangle.
    /// </summary>
    public (double X, double Y)[] GetCorners()
    {
        double cx = Center.X;
        double cy = Center.Y;
        double w = Width;
        double h = Height;
        double cos = Math.Cos(Angle);
        double sin = Math.Sin(Angle);

        // Half dimensions
        double hw = w / 2;
        double hh = h / 2;

        // Four corners relative to center, then rotated
        var corners = new (double X, double Y)[4];
        corners[0] = (cx + (-hw * cos - (-hh) * sin), cy + (-hw * sin + (-hh) * cos)); // Top-left
        corners[1] = (cx + (hw * cos - (-hh) * sin), cy + (hw * sin + (-hh) * cos));   // Top-right
        corners[2] = (cx + (hw * cos - hh * sin), cy + (hw * sin + hh * cos));         // Bottom-right
        corners[3] = (cx + (-hw * cos - hh * sin), cy + (-hw * sin + hh * cos));       // Bottom-left

        return corners;
    }

    public override string ToString() =>
        $"Center=({Center.X:F1},{Center.Y:F1}), Size=({Width:F1}×{Height:F1}), Angle={Angle * 180 / Math.PI:F1}°";
}

/// <summary>
/// Geometry utility functions for image processing.
/// </summary>
public static class GeometryUtils
{
    /// <summary>
    /// Extracts the contour (boundary pixels) from a binary mask.
    /// Returns pixels in counter-clockwise order starting from the top-left.
    /// </summary>
    /// <param name="mask">Binary mask where true = foreground.</param>
    /// <returns>List of contour points.</returns>
    public static List<Point> ExtractContour(bool[,] mask)
    {
        if (mask == null)
            throw new ArgumentNullException(nameof(mask));

        int height = mask.GetLength(0);
        int width = mask.GetLength(1);

        if (height == 0 || width == 0)
            return new List<Point>();

        var contour = new List<Point>();

        // Find all boundary pixels (foreground pixels with at least one background neighbor)
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (!mask[y, x])
                    continue;

                // Check if this is a boundary pixel (has background neighbor in 4-connectivity)
                bool isBoundary = false;

                // Check top
                if (y == 0 || !mask[y - 1, x])
                    isBoundary = true;
                // Check bottom
                else if (y == height - 1 || !mask[y + 1, x])
                    isBoundary = true;
                // Check left
                else if (x == 0 || !mask[y, x - 1])
                    isBoundary = true;
                // Check right
                else if (x == width - 1 || !mask[y, x + 1])
                    isBoundary = true;

                if (isBoundary)
                    contour.Add(new Point(x, y));
            }
        }

        return contour;
    }

    /// <summary>
    /// Computes the convex hull of a set of points using Graham's scan algorithm.
    /// </summary>
    /// <param name="points">Input points.</param>
    /// <returns>Points on the convex hull in counter-clockwise order.</returns>
    public static List<Point> ConvexHull(List<Point> points)
    {
        if (points == null)
            throw new ArgumentNullException(nameof(points));

        if (points.Count < 3)
            return new List<Point>(points);

        // Find the point with lowest Y (and leftmost if tie)
        var sorted = points.OrderBy(p => p.Y).ThenBy(p => p.X).ToList();
        var pivot = sorted[0];

        // Sort points by polar angle relative to pivot
        var angleSort = sorted.Skip(1)
            .OrderBy(p => Math.Atan2(p.Y - pivot.Y, p.X - pivot.X))
            .ThenBy(p => (p.X - pivot.X) * (p.X - pivot.X) + (p.Y - pivot.Y) * (p.Y - pivot.Y))
            .ToList();

        var hull = new List<Point> { pivot };

        foreach (var point in angleSort)
        {
            // Remove points that make a clockwise turn
            while (hull.Count >= 2)
            {
                var a = hull[hull.Count - 2];
                var b = hull[hull.Count - 1];
                var cross = CrossProduct(a, b, point);

                if (cross <= 0) // Clockwise or collinear
                    hull.RemoveAt(hull.Count - 1);
                else
                    break;
            }

            hull.Add(point);
        }

        return hull;
    }

    /// <summary>
    /// Computes the minimum area bounding rectangle for a set of points using Rotating Calipers.
    /// </summary>
    /// <param name="points">Input points (should ideally be convex hull for best results).</param>
    /// <returns>Minimum area rotated rectangle.</returns>
    public static RotatedRect MinAreaRect(List<Point> points)
    {
        if (points == null)
            throw new ArgumentNullException(nameof(points));

        if (points.Count == 0)
            return new RotatedRect(0, 0, 0, 0, 0);

        if (points.Count == 1)
        {
            var p = points[0];
            return new RotatedRect(p.X, p.Y, 0, 0, 0);
        }

        if (points.Count == 2)
        {
            var p1 = points[0];
            var p2 = points[1];
            double cx = (p1.X + p2.X) / 2.0;
            double cy = (p1.Y + p2.Y) / 2.0;
            double width = Math.Sqrt((p2.X - p1.X) * (p2.X - p1.X) + (p2.Y - p1.Y) * (p2.Y - p1.Y));
            double angle = Math.Atan2(p2.Y - p1.Y, p2.X - p1.X);
            return new RotatedRect(cx, cy, width, 0, angle);
        }

        // Compute convex hull first
        var hull = ConvexHull(points);

        if (hull.Count < 3)
        {
            // Fallback to axis-aligned bounding box
            return AxisAlignedBoundingBox(points);
        }

        // Rotating calipers algorithm
        double minArea = double.MaxValue;
        RotatedRect bestRect = default;

        int n = hull.Count;

        for (int i = 0; i < n; i++)
        {
            var p1 = hull[i];
            var p2 = hull[(i + 1) % n];

            // Edge vector
            double edgeX = p2.X - p1.X;
            double edgeY = p2.Y - p1.Y;
            double edgeLen = Math.Sqrt(edgeX * edgeX + edgeY * edgeY);

            if (edgeLen < 1e-10)
                continue;

            // Unit vector along edge
            double ux = edgeX / edgeLen;
            double uy = edgeY / edgeLen;

            // Perpendicular unit vector
            double vx = -uy;
            double vy = ux;

            // Project all points onto edge direction and perpendicular
            double minU = double.MaxValue, maxU = double.MinValue;
            double minV = double.MaxValue, maxV = double.MinValue;

            foreach (var p in hull)
            {
                double u = p.X * ux + p.Y * uy;
                double v = p.X * vx + p.Y * vy;

                minU = Math.Min(minU, u);
                maxU = Math.Max(maxU, u);
                minV = Math.Min(minV, v);
                maxV = Math.Max(maxV, v);
            }

            double width = maxU - minU;
            double height = maxV - minV;
            double area = width * height;

            if (area < minArea)
            {
                minArea = area;

                // Center in rotated coordinate system
                double cu = (minU + maxU) / 2;
                double cv = (minV + maxV) / 2;

                // Transform back to original coordinates
                double cx = cu * ux + cv * vx;
                double cy = cu * uy + cv * vy;

                double angle = Math.Atan2(uy, ux);

                bestRect = new RotatedRect(cx, cy, width, height, angle);
            }
        }

        return bestRect;
    }

    /// <summary>
    /// Computes an axis-aligned bounding box (fallback when convex hull fails).
    /// </summary>
    private static RotatedRect AxisAlignedBoundingBox(List<Point> points)
    {
        int minX = int.MaxValue, maxX = int.MinValue;
        int minY = int.MaxValue, maxY = int.MinValue;

        foreach (var p in points)
        {
            minX = Math.Min(minX, p.X);
            maxX = Math.Max(maxX, p.X);
            minY = Math.Min(minY, p.Y);
            maxY = Math.Max(maxY, p.Y);
        }

        double cx = (minX + maxX) / 2.0;
        double cy = (minY + maxY) / 2.0;
        double width = maxX - minX;
        double height = maxY - minY;

        return new RotatedRect(cx, cy, width, height, 0);
    }

    /// <summary>
    /// Computes the cross product of vectors (b-a) and (c-a).
    /// Positive = counter-clockwise, negative = clockwise, zero = collinear.
    /// </summary>
    private static long CrossProduct(Point a, Point b, Point c)
    {
        return (long)(b.X - a.X) * (c.Y - a.Y) - (long)(b.Y - a.Y) * (c.X - a.X);
    }
}
