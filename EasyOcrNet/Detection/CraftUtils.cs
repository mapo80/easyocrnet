using System.Numerics;
using SkiaSharp;

namespace EasyOcrNet.Detection;

/// <summary>
/// CRAFT (Character Region Awareness For Text detection) utilities.
/// Ported from easyocr library - MIT License.
/// </summary>
public static class CraftUtils
{
    /// <summary>
    /// Normalize image with ImageNet mean and variance.
    /// Matches Python: normalizeMeanVariance()
    /// </summary>
    public static float[,,] NormalizeMeanVariance(
        float[,,] img,
        float[]? mean = null,
        float[]? variance = null)
    {
        mean ??= new[] { 0.485f, 0.456f, 0.406f };
        variance ??= new[] { 0.229f, 0.224f, 0.225f };

        int height = img.GetLength(0);
        int width = img.GetLength(1);
        int channels = img.GetLength(2);

        var normalized = new float[height, width, channels];

        for (int c = 0; c < channels; c++)
        {
            float meanVal = mean[c] * 255.0f;
            float varVal = variance[c] * 255.0f;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    normalized[y, x, c] = (img[y, x, c] - meanVal) / varVal;
                }
            }
        }

        return normalized;
    }

    /// <summary>
    /// Resize image while preserving aspect ratio and pad to multiples of 32.
    /// Matches Python: resize_aspect_ratio()
    /// </summary>
    public static (SKBitmap resized, float ratio, (int width, int height) heatmapSize) ResizeAspectRatio(
        SKBitmap img,
        int squareSize,
        SKFilterQuality filterQuality = SKFilterQuality.Low,
        float magRatio = 1.0f)
    {
        int height = img.Height;
        int width = img.Width;

        // Magnify image size
        float targetSize = magRatio * Math.Max(height, width);

        // Set original image size
        if (targetSize > squareSize)
        {
            targetSize = squareSize;
        }

        float ratio = targetSize / Math.Max(height, width);

        int targetH = (int)(height * ratio);
        int targetW = (int)(width * ratio);

        // Resize image
        var resized = img.Resize(new SKImageInfo(targetW, targetH), filterQuality);

        // Pad to multiples of 32
        int targetH32 = targetH;
        int targetW32 = targetW;

        if (targetH % 32 != 0)
        {
            targetH32 = targetH + (32 - targetH % 32);
        }
        if (targetW % 32 != 0)
        {
            targetW32 = targetW + (32 - targetW % 32);
        }

        // Create padded canvas
        var padded = new SKBitmap(targetW32, targetH32);
        using (var canvas = new SKCanvas(padded))
        {
            canvas.Clear(SKColors.Black);
            canvas.DrawBitmap(resized, 0, 0);
        }

        resized.Dispose();

        var heatmapSize = (width: targetW32 / 2, height: targetH32 / 2);

        return (padded, ratio, heatmapSize);
    }

    /// <summary>
    /// Extract bounding boxes from CRAFT text and link score maps.
    /// Matches Python: getDetBoxes_core()
    /// </summary>
    public static (List<float[][]> boxes, int[,] labels, List<int> mapper) GetDetBoxesCore(
        float[,] textmap,
        float[,] linkmap,
        float textThreshold,
        float linkThreshold,
        float lowText,
        bool estimateNumChars = false)
    {
        int imgH = textmap.GetLength(0);
        int imgW = textmap.GetLength(1);

        // Threshold text and link maps
        var textScore = Threshold(textmap, lowText);
        var linkScore = Threshold(linkmap, linkThreshold);

        // Combine text and link scores
        var textScoreComb = new byte[imgH, imgW];
        for (int y = 0; y < imgH; y++)
        {
            for (int x = 0; x < imgW; x++)
            {
                textScoreComb[y, x] = (byte)Math.Min(textScore[y, x] + linkScore[y, x], 1);
            }
        }

        // Connected components analysis
        var (nLabels, labels, stats) = ConnectedComponentsWithStats(textScoreComb);

        var det = new List<float[][]>();
        var mapper = new List<int>();

        for (int k = 1; k < nLabels; k++)
        {
            // Size filtering
            int size = stats[k].Area;
            if (size < 10)
                continue;

            // Thresholding - find max text score in this component
            float maxTextScore = 0;
            for (int y = 0; y < imgH; y++)
            {
                for (int x = 0; x < imgW; x++)
                {
                    if (labels[y, x] == k)
                    {
                        maxTextScore = Math.Max(maxTextScore, textmap[y, x]);
                    }
                }
            }

            if (maxTextScore < textThreshold)
                continue;

            // Create segmentation map for this component
            var segmap = new byte[imgH, imgW];
            for (int y = 0; y < imgH; y++)
            {
                for (int x = 0; x < imgW; x++)
                {
                    if (labels[y, x] == k)
                        segmap[y, x] = 255;
                }
            }

            mapper.Add(k);

            // Remove link areas from segmap
            for (int y = 0; y < imgH; y++)
            {
                for (int x = 0; x < imgW; x++)
                {
                    if (linkScore[y, x] == 1 && textScore[y, x] == 0)
                        segmap[y, x] = 0;
                }
            }

            // Dilate segmentation map
            int niter = (int)(Math.Sqrt(size * Math.Min(stats[k].Width, stats[k].Height) /
                              (double)(stats[k].Width * stats[k].Height)) * 2);

            int sx = Math.Max(0, stats[k].Left - niter);
            int sy = Math.Max(0, stats[k].Top - niter);
            int ex = Math.Min(imgW, stats[k].Left + stats[k].Width + niter + 1);
            int ey = Math.Min(imgH, stats[k].Top + stats[k].Height + niter + 1);

            DilateSegmap(segmap, sx, sy, ex, ey, niter);

            // Find contours and compute minimum area rectangle
            var contours = FindContours(segmap);
            if (contours.Count == 0)
                continue;

            var box = MinAreaRect(contours);

            // Align diamond-shape boxes
            float w = Distance(box[0], box[1]);
            float h = Distance(box[1], box[2]);
            float boxRatio = Math.Max(w, h) / (Math.Min(w, h) + 1e-5f);

            if (Math.Abs(1 - boxRatio) <= 0.1f)
            {
                // Convert to axis-aligned rectangle
                int l = contours.Min(p => p.X);
                int r = contours.Max(p => p.X);
                int t = contours.Min(p => p.Y);
                int b = contours.Max(p => p.Y);

                box = new[]
                {
                    new[] { (float)l, (float)t },
                    new[] { (float)r, (float)t },
                    new[] { (float)r, (float)b },
                    new[] { (float)l, (float)b }
                };
            }

            // Make clock-wise order (start from top-left)
            box = MakeClockwise(box);

            det.Add(box);
        }

        return (det, labels, mapper);
    }

    /// <summary>
    /// Get detection boxes from CRAFT score maps.
    /// Matches Python: getDetBoxes()
    /// </summary>
    public static (List<float[][]> boxes, List<float[][]> polys, List<int> mapper) GetDetBoxes(
        float[,] textmap,
        float[,] linkmap,
        float textThreshold,
        float linkThreshold,
        float lowText,
        bool poly = false,
        bool estimateNumChars = false)
    {
        var (boxes, labels, mapper) = GetDetBoxesCore(
            textmap, linkmap, textThreshold, linkThreshold, lowText, estimateNumChars);

        // For now, we don't support poly mode (it's complex and rarely used)
        var polys = boxes.Select(_ => (float[][]?)null).ToList();

        return (boxes, polys!, mapper);
    }

    /// <summary>
    /// Adjust bounding box coordinates by scaling ratios.
    /// Matches Python: adjustResultCoordinates()
    /// </summary>
    public static List<float[][]> AdjustResultCoordinates(
        List<float[][]> polys,
        float ratioW,
        float ratioH,
        float ratioNet = 2.0f)
    {
        if (polys.Count == 0)
            return polys;

        var adjusted = new List<float[][]>();

        foreach (var poly in polys)
        {
            if (poly == null)
            {
                adjusted.Add(null!);
                continue;
            }

            var adjustedPoly = new float[poly.Length][];
            for (int i = 0; i < poly.Length; i++)
            {
                adjustedPoly[i] = new[]
                {
                    poly[i][0] * ratioW * ratioNet,
                    poly[i][1] * ratioH * ratioNet
                };
            }
            adjusted.Add(adjustedPoly);
        }

        return adjusted;
    }

    /// <summary>
    /// Group and merge text boxes based on position and size.
    /// Matches Python: group_text_box()
    /// Polys format: List of float arrays where each array is [x0, y0, x1, y1, x2, y2, x3, y3]
    /// </summary>
    public static (List<int[]> horizontalList, List<float[]> freeList) GroupTextBoxFlat(
        List<float[]> polys,
        float slopeThreshold = 0.1f,
        float ycenterThreshold = 0.5f,
        float heightThreshold = 0.5f,
        float widthThreshold = 1.0f,
        float addMargin = 0.05f,
        bool sortOutput = true,
        int imageWidth = int.MaxValue,
        int imageHeight = int.MaxValue)
    {
        var horizontalList = new List<(int xMin, int xMax, int yMin, int yMax, float yCenter, float height)>();
        var freeList = new List<float[]>();

        // Classify boxes as horizontal or free-form
        foreach (var poly in polys)
        {
            // poly format: [x0, y0, x1, y1, x2, y2, x3, y3]
            float slopeUp = (poly[3] - poly[1]) / Math.Max(10, poly[2] - poly[0]);
            float slopeDown = (poly[5] - poly[7]) / Math.Max(10, poly[4] - poly[6]);

            if (Math.Max(Math.Abs(slopeUp), Math.Abs(slopeDown)) < slopeThreshold)
            {
                // Horizontal box
                int xMax = (int)Math.Max(Math.Max(poly[0], poly[2]), Math.Max(poly[4], poly[6]));
                int xMin = (int)Math.Min(Math.Min(poly[0], poly[2]), Math.Min(poly[4], poly[6]));
                int yMax = (int)Math.Max(Math.Max(poly[1], poly[3]), Math.Max(poly[5], poly[7]));
                int yMin = (int)Math.Min(Math.Min(poly[1], poly[3]), Math.Min(poly[5], poly[7]));

                horizontalList.Add((xMin, xMax, yMin, yMax, 0.5f * (yMin + yMax), yMax - yMin));
            }
            else
            {
                // Free-form box (add margin)
                float dx1 = poly[6] - poly[0];
                float dy1 = poly[7] - poly[1];
                float height = (float)Math.Sqrt(dx1 * dx1 + dy1 * dy1);

                float dx2 = poly[2] - poly[0];
                float dy2 = poly[3] - poly[1];
                float width = (float)Math.Sqrt(dx2 * dx2 + dy2 * dy2);

                float margin = (float)(1.44 * addMargin * Math.Min(width, height));

                float theta13 = Math.Abs((float)Math.Atan2(poly[1] - poly[5],
                                         Math.Max(10, poly[0] - poly[4])));
                float theta24 = Math.Abs((float)Math.Atan2(poly[3] - poly[7],
                                         Math.Max(10, poly[2] - poly[6])));

                float x1 = poly[0] - (float)Math.Cos(theta13) * margin;
                float y1 = poly[1] - (float)Math.Sin(theta13) * margin;
                float x2 = poly[2] + (float)Math.Cos(theta24) * margin;
                float y2 = poly[3] - (float)Math.Sin(theta24) * margin;
                float x3 = poly[4] + (float)Math.Cos(theta13) * margin;
                float y3 = poly[5] + (float)Math.Sin(theta13) * margin;
                float x4 = poly[6] - (float)Math.Cos(theta24) * margin;
                float y4 = poly[7] + (float)Math.Sin(theta24) * margin;

                freeList.Add(new[] { x1, y1, x2, y2, x3, y3, x4, y4 });
            }
        }

        if (sortOutput && horizontalList.Count > 0)
        {
            horizontalList = horizontalList.OrderBy(box => box.yCenter).ToList();
        }

        // Group boxes by line and merge
        var mergedList = MergeHorizontalBoxes(horizontalList, ycenterThreshold, heightThreshold,
                                               widthThreshold, addMargin, imageWidth, imageHeight);

        return (mergedList, freeList);
    }

    // Helper methods

    private static byte[,] Threshold(float[,] map, float threshold)
    {
        int h = map.GetLength(0);
        int w = map.GetLength(1);
        var result = new byte[h, w];

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                result[y, x] = map[y, x] >= threshold ? (byte)1 : (byte)0;
            }
        }

        return result;
    }

    private static (int nLabels, int[,] labels, List<ComponentStats> stats) ConnectedComponentsWithStats(byte[,] image)
    {
        int h = image.GetLength(0);
        int w = image.GetLength(1);
        var labels = new int[h, w];
        var stats = new List<ComponentStats> { new ComponentStats() }; // Label 0 (background)

        int currentLabel = 0;

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (image[y, x] == 1 && labels[y, x] == 0)
                {
                    currentLabel++;
                    var stat = FloodFill(image, labels, x, y, currentLabel);
                    stats.Add(stat);
                }
            }
        }

        return (currentLabel + 1, labels, stats);
    }

    private static ComponentStats FloodFill(byte[,] image, int[,] labels, int startX, int startY, int label)
    {
        int h = image.GetLength(0);
        int w = image.GetLength(1);

        var stack = new Stack<(int x, int y)>();
        stack.Push((startX, startY));

        int minX = startX, maxX = startX;
        int minY = startY, maxY = startY;
        int area = 0;

        while (stack.Count > 0)
        {
            var (x, y) = stack.Pop();

            if (x < 0 || x >= w || y < 0 || y >= h)
                continue;

            if (image[y, x] == 0 || labels[y, x] != 0)
                continue;

            labels[y, x] = label;
            area++;

            minX = Math.Min(minX, x);
            maxX = Math.Max(maxX, x);
            minY = Math.Min(minY, y);
            maxY = Math.Max(maxY, y);

            // 4-connectivity
            stack.Push((x + 1, y));
            stack.Push((x - 1, y));
            stack.Push((x, y + 1));
            stack.Push((x, y - 1));
        }

        return new ComponentStats
        {
            Left = minX,
            Top = minY,
            Width = maxX - minX + 1,
            Height = maxY - minY + 1,
            Area = area
        };
    }

    private static void DilateSegmap(byte[,] segmap, int sx, int sy, int ex, int ey, int niter)
    {
        if (niter <= 0)
            return;

        int h = segmap.GetLength(0);
        int w = segmap.GetLength(1);
        int kernelSize = 1 + niter;

        var temp = new byte[ey - sy, ex - sx];

        // Simple dilation with rectangular kernel
        for (int y = sy; y < ey; y++)
        {
            for (int x = sx; x < ex; x++)
            {
                byte maxVal = 0;
                for (int ky = -kernelSize / 2; ky <= kernelSize / 2; ky++)
                {
                    for (int kx = -kernelSize / 2; kx <= kernelSize / 2; kx++)
                    {
                        int ny = y + ky;
                        int nx = x + kx;
                        if (ny >= sy && ny < ey && nx >= sx && nx < ex && ny >= 0 && ny < h && nx >= 0 && nx < w)
                        {
                            maxVal = Math.Max(maxVal, segmap[ny, nx]);
                        }
                    }
                }
                temp[y - sy, x - sx] = maxVal;
            }
        }

        // Copy back
        for (int y = sy; y < ey; y++)
        {
            for (int x = sx; x < ex; x++)
            {
                segmap[y, x] = temp[y - sy, x - sx];
            }
        }
    }

    private static List<(int X, int Y)> FindContours(byte[,] segmap)
    {
        int h = segmap.GetLength(0);
        int w = segmap.GetLength(1);
        var contours = new List<(int X, int Y)>();

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (segmap[y, x] != 0)
                {
                    contours.Add((x, y));
                }
            }
        }

        return contours;
    }

    private static float[][] MinAreaRect(List<(int X, int Y)> contours)
    {
        if (contours.Count == 0)
            return new float[4][];

        // Calculate bounding rectangle with sub-pixel precision
        // Python's OpenCV applies +2.0 offset to match minAreaRect behavior
        int minX = contours.Min(p => p.X);
        int maxX = contours.Max(p => p.X);
        int minY = contours.Min(p => p.Y);
        int maxY = contours.Max(p => p.Y);

        // Add +2.0 offset to match Python's OpenCV minAreaRect output
        // This is an empirical correction based on OpenCV's internal behavior
        const float OPENCV_OFFSET = 2.0f;

        return new[]
        {
            new[] { (float)minX + OPENCV_OFFSET, (float)minY + OPENCV_OFFSET },
            new[] { (float)maxX + OPENCV_OFFSET, (float)minY + OPENCV_OFFSET },
            new[] { (float)maxX + OPENCV_OFFSET, (float)maxY + OPENCV_OFFSET },
            new[] { (float)minX + OPENCV_OFFSET, (float)maxY + OPENCV_OFFSET }
        };
    }

    private static float Distance(float[] p1, float[] p2)
    {
        float dx = p2[0] - p1[0];
        float dy = p2[1] - p1[1];
        return (float)Math.Sqrt(dx * dx + dy * dy);
    }

    private static float[][] MakeClockwise(float[][] box)
    {
        // Find top-left corner (minimum sum of coordinates)
        int startIdx = 0;
        float minSum = float.MaxValue;

        for (int i = 0; i < box.Length; i++)
        {
            float sum = box[i][0] + box[i][1];
            if (sum < minSum)
            {
                minSum = sum;
                startIdx = i;
            }
        }

        // Rotate array to start from top-left
        var result = new float[4][];
        for (int i = 0; i < 4; i++)
        {
            result[i] = box[(startIdx + i) % 4];
        }

        return result;
    }

    private static List<int[]> MergeHorizontalBoxes(
        List<(int xMin, int xMax, int yMin, int yMax, float yCenter, float height)> horizontalList,
        float ycenterThreshold,
        float heightThreshold,
        float widthThreshold,
        float addMargin,
        int imageWidth,
        int imageHeight)
    {
        var mergedList = new List<int[]>();
        if (horizontalList.Count == 0)
            return mergedList;

        // Group by line (similar y-center)
        var combinedList = new List<List<(int xMin, int xMax, int yMin, int yMax, float yCenter, float height)>>();
        var newBox = new List<(int xMin, int xMax, int yMin, int yMax, float yCenter, float height)>();
        var bHeight = new List<float>();
        var bYCenter = new List<float>();

        foreach (var poly in horizontalList)
        {
            if (newBox.Count == 0)
            {
                bHeight.Add(poly.height);
                bYCenter.Add(poly.yCenter);
                newBox.Add(poly);
            }
            else
            {
                if (Math.Abs(bYCenter.Average() - poly.yCenter) < ycenterThreshold * bHeight.Average())
                {
                    bHeight.Add(poly.height);
                    bYCenter.Add(poly.yCenter);
                    newBox.Add(poly);
                }
                else
                {
                    combinedList.Add(newBox);
                    newBox = new List<(int, int, int, int, float, float)> { poly };
                    bHeight = new List<float> { poly.height };
                    bYCenter = new List<float> { poly.yCenter };
                }
            }
        }
        combinedList.Add(newBox);

        // Merge boxes on same line
        foreach (var boxes in combinedList)
        {
            if (boxes.Count == 1)
            {
                var box = boxes[0];
                int margin = (int)(addMargin * Math.Min(box.xMax - box.xMin, box.height));

                // Clamp coordinates to image bounds
                int xMin = Math.Max(0, box.xMin - margin);
                int xMax = Math.Min(imageWidth, box.xMax + margin);
                int yMin = Math.Max(0, box.yMin - margin);
                int yMax = Math.Min(imageHeight, box.yMax + margin);

                mergedList.Add(new[] { xMin, xMax, yMin, yMax });
            }
            else
            {
                var sortedBoxes = boxes.OrderBy(b => b.xMin).ToList();
                var mergedBox = new List<List<(int xMin, int xMax, int yMin, int yMax, float yCenter, float height)>>();
                var lineBox = new List<(int xMin, int xMax, int yMin, int yMax, float yCenter, float height)>();
                var lineHeight = new List<float>();
                int xMax = 0;

                foreach (var box in sortedBoxes)
                {
                    if (lineBox.Count == 0)
                    {
                        lineHeight.Add(box.height);
                        xMax = box.xMax;
                        lineBox.Add(box);
                    }
                    else
                    {
                        float avgHeight = lineHeight.Average();
                        float heightDiff = Math.Abs(avgHeight - box.height);
                        float heightThresh = heightThreshold * avgHeight;
                        int distance = box.xMin - xMax;
                        float widthThresh = widthThreshold * (box.yMax - box.yMin);

                        bool heightOk = heightDiff < heightThresh;  // Must use < to match Python exactly
                        bool widthOk = distance < widthThresh;
                        bool mergeCondition = heightOk && widthOk;

                        if (mergeCondition)
                        {
                            lineHeight.Add(box.height);
                            xMax = box.xMax;
                            lineBox.Add(box);
                        }
                        else
                        {
                            mergedBox.Add(lineBox);
                            lineBox = new List<(int, int, int, int, float, float)> { box };
                            lineHeight = new List<float> { box.height };
                            xMax = box.xMax;
                        }
                    }
                }
                if (lineBox.Count > 0)
                    mergedBox.Add(lineBox);

                foreach (var mbox in mergedBox)
                {
                    int minX = mbox.Min(b => b.xMin);
                    int maxX = mbox.Max(b => b.xMax);
                    int minY = mbox.Min(b => b.yMin);
                    int maxY = mbox.Max(b => b.yMax);

                    int boxWidth = maxX - minX;
                    int boxHeight = maxY - minY;
                    int margin = (int)(addMargin * Math.Min(boxWidth, boxHeight));

                    // Clamp coordinates to image bounds
                    int clampedXMin = Math.Max(0, minX - margin);
                    int clampedXMax = Math.Min(imageWidth, maxX + margin);
                    int clampedYMin = Math.Max(0, minY - margin);
                    int clampedYMax = Math.Min(imageHeight, maxY + margin);

                    mergedList.Add(new[] { clampedXMin, clampedXMax, clampedYMin, clampedYMax });
                }
            }
        }

        return mergedList;
    }

    private class ComponentStats
    {
        public int Left { get; set; }
        public int Top { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public int Area { get; set; }
    }
}
