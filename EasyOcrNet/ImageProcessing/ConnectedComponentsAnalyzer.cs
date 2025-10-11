using System;
using System.Collections.Generic;

namespace EasyOcrNet.ImageProcessing;

/// <summary>
/// Statistics for a single connected component.
/// </summary>
public readonly struct ComponentStats
{
    /// <summary>
    /// The label ID of this component.
    /// </summary>
    public int Label { get; }

    /// <summary>
    /// The left-most X coordinate of the bounding box.
    /// </summary>
    public int X { get; }

    /// <summary>
    /// The top-most Y coordinate of the bounding box.
    /// </summary>
    public int Y { get; }

    /// <summary>
    /// The width of the bounding box.
    /// </summary>
    public int Width { get; }

    /// <summary>
    /// The height of the bounding box.
    /// </summary>
    public int Height { get; }

    /// <summary>
    /// The total number of pixels in this component.
    /// </summary>
    public int Area { get; }

    /// <summary>
    /// The X coordinate of the centroid.
    /// </summary>
    public double CentroidX { get; }

    /// <summary>
    /// The Y coordinate of the centroid.
    /// </summary>
    public double CentroidY { get; }

    internal ComponentStats(int label, int x, int y, int width, int height, int area, double centroidX, double centroidY)
    {
        Label = label;
        X = x;
        Y = y;
        Width = width;
        Height = height;
        Area = area;
        CentroidX = centroidX;
        CentroidY = centroidY;
    }

    public override string ToString() =>
        $"Label={Label}, BBox=({X},{Y},{Width},{Height}), Area={Area}, Centroid=({CentroidX:F1},{CentroidY:F1})";
}

/// <summary>
/// Result of connected components analysis.
/// </summary>
public readonly struct ConnectedComponentsResult
{
    /// <summary>
    /// The number of components found (excluding background label 0).
    /// </summary>
    public int Count { get; }

    /// <summary>
    /// Label map where each pixel contains its component label (0 = background).
    /// </summary>
    public int[,] Labels { get; }

    /// <summary>
    /// Statistics for each component (indexed by label - 1, since label 0 is background).
    /// </summary>
    public ComponentStats[] Stats { get; }

    internal ConnectedComponentsResult(int count, int[,] labels, ComponentStats[] stats)
    {
        Count = count;
        Labels = labels;
        Stats = stats;
    }
}

/// <summary>
/// Performs connected components analysis on binary images using the Two-Pass algorithm.
/// </summary>
public static class ConnectedComponentsAnalyzer
{
    /// <summary>
    /// Analyzes connected components in a binary image using 4-connectivity.
    /// </summary>
    /// <param name="binaryImage">Binary image where true = foreground, false = background.</param>
    /// <returns>Connected components analysis result.</returns>
    public static ConnectedComponentsResult Analyze(bool[,] binaryImage)
    {
        if (binaryImage == null)
            throw new ArgumentNullException(nameof(binaryImage));

        int height = binaryImage.GetLength(0);
        int width = binaryImage.GetLength(1);

        if (height == 0 || width == 0)
            return new ConnectedComponentsResult(0, new int[0, 0], Array.Empty<ComponentStats>());

        var labels = new int[height, width];
        int nextLabel = 1;
        var unionFind = new UnionFind(width * height); // Over-allocate for safety

        // First pass: assign temporary labels and record equivalences
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (!binaryImage[y, x])
                {
                    labels[y, x] = 0; // Background
                    continue;
                }

                // Check 4-connected neighbors (top and left only in first pass)
                int topLabel = (y > 0) ? labels[y - 1, x] : 0;
                int leftLabel = (x > 0) ? labels[y, x - 1] : 0;

                if (topLabel == 0 && leftLabel == 0)
                {
                    // No labeled neighbors: assign new label
                    labels[y, x] = nextLabel++;
                }
                else if (topLabel != 0 && leftLabel == 0)
                {
                    // Only top neighbor labeled
                    labels[y, x] = topLabel;
                }
                else if (topLabel == 0 && leftLabel != 0)
                {
                    // Only left neighbor labeled
                    labels[y, x] = leftLabel;
                }
                else
                {
                    // Both neighbors labeled
                    labels[y, x] = Math.Min(topLabel, leftLabel);

                    // Record equivalence if different labels
                    if (topLabel != leftLabel)
                    {
                        unionFind.Union(topLabel, leftLabel);
                    }
                }
            }
        }

        // Second pass: relabel using union-find roots
        var labelMapping = new Dictionary<int, int>();
        int finalLabel = 1;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int label = labels[y, x];
                if (label == 0)
                    continue;

                // Find root label
                int root = unionFind.Find(label);

                // Map root to final consecutive label
                if (!labelMapping.TryGetValue(root, out int mappedLabel))
                {
                    mappedLabel = finalLabel++;
                    labelMapping[root] = mappedLabel;
                }

                labels[y, x] = mappedLabel;
            }
        }

        // Calculate statistics for each component
        int numComponents = finalLabel - 1;
        var stats = CalculateStats(labels, numComponents, width, height);

        return new ConnectedComponentsResult(numComponents, labels, stats);
    }

    private static ComponentStats[] CalculateStats(int[,] labels, int numComponents, int width, int height)
    {
        if (numComponents == 0)
            return Array.Empty<ComponentStats>();

        // Initialize accumulators for each component
        var areas = new int[numComponents];
        var minX = new int[numComponents];
        var minY = new int[numComponents];
        var maxX = new int[numComponents];
        var maxY = new int[numComponents];
        var sumX = new long[numComponents];
        var sumY = new long[numComponents];

        for (int i = 0; i < numComponents; i++)
        {
            minX[i] = width;
            minY[i] = height;
            maxX[i] = -1;
            maxY[i] = -1;
        }

        // Accumulate statistics
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int label = labels[y, x];
                if (label == 0)
                    continue;

                int idx = label - 1; // Convert to 0-based index
                areas[idx]++;
                sumX[idx] += x;
                sumY[idx] += y;

                if (x < minX[idx]) minX[idx] = x;
                if (x > maxX[idx]) maxX[idx] = x;
                if (y < minY[idx]) minY[idx] = y;
                if (y > maxY[idx]) maxY[idx] = y;
            }
        }

        // Build stats array
        var stats = new ComponentStats[numComponents];
        for (int i = 0; i < numComponents; i++)
        {
            int area = areas[i];
            double centroidX = area > 0 ? (double)sumX[i] / area : 0;
            double centroidY = area > 0 ? (double)sumY[i] / area : 0;

            int bboxWidth = maxX[i] >= minX[i] ? maxX[i] - minX[i] + 1 : 0;
            int bboxHeight = maxY[i] >= minY[i] ? maxY[i] - minY[i] + 1 : 0;

            stats[i] = new ComponentStats(
                label: i + 1,
                x: minX[i],
                y: minY[i],
                width: bboxWidth,
                height: bboxHeight,
                area: area,
                centroidX: centroidX,
                centroidY: centroidY);
        }

        return stats;
    }
}
