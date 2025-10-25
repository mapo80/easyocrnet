using System;
using System.Collections.Generic;
using System.Linq;

namespace EasyOcrNet.Detection;

/// <summary>
/// Text box grouping and merging - EXACT translation from Python craft_utils.py
/// </summary>
public static class GroupTextBox
{
    /// <summary>
    /// Group and merge text boxes based on their position and size.
    /// Traduzione 1:1 da craft_utils.py group_text_box()
    /// </summary>
    /// <param name="polys">List of polygons in flattened format [x0,y0,x1,y1,x2,y2,x3,y3]</param>
    /// <param name="slopeThreshold">Slope threshold for horizontal classification (default 0.1)</param>
    /// <param name="ycenterThreshold">Y-center threshold for grouping (default 0.5)</param>
    /// <param name="heightThreshold">Height threshold for merging (default 0.5)</param>
    /// <param name="widthThreshold">Width threshold for merging (default 1.0)</param>
    /// <param name="addMargin">Margin to add around boxes (default 0.05)</param>
    /// <param name="sortOutput">Sort by y-center (default true)</param>
    /// <returns>(horizontalList, freeList)</returns>
    public static (List<float[]> horizontalList, List<float[][]> freeList) Group(
        List<float[]> polys,
        float slopeThreshold = 0.1f,
        float ycenterThreshold = 0.5f,
        float heightThreshold = 0.5f,
        float widthThreshold = 1.0f,
        float addMargin = 0.05f,
        bool sortOutput = true)
    {
        var horizontalList = new List<float[]>();
        var freeList = new List<float[][]>();
        var combinedList = new List<List<float[]>>();
        var mergedList = new List<float[]>();

        // Step 1: Classify boxes as horizontal or free-form based on slope
        // Python lines 231-258
        foreach (var poly in polys)
        {
            // poly format: [x0, y0, x1, y1, x2, y2, x3, y3]
            float slopeUp = (poly[3] - poly[1]) / Math.Max(10f, poly[2] - poly[0]);
            float slopeDown = (poly[5] - poly[7]) / Math.Max(10f, poly[4] - poly[6]);

            if (Math.Max(Math.Abs(slopeUp), Math.Abs(slopeDown)) < slopeThreshold)
            {
                // Horizontal box
                float xMax = Math.Max(Math.Max(poly[0], poly[2]), Math.Max(poly[4], poly[6]));
                float xMin = Math.Min(Math.Min(poly[0], poly[2]), Math.Min(poly[4], poly[6]));
                float yMax = Math.Max(Math.Max(poly[1], poly[3]), Math.Max(poly[5], poly[7]));
                float yMin = Math.Min(Math.Min(poly[1], poly[3]), Math.Min(poly[5], poly[7]));

                // Format: [xMin, xMax, yMin, yMax, yCenter, height]
                horizontalList.Add(new[] { xMin, xMax, yMin, yMax, 0.5f * (yMin + yMax), yMax - yMin });
            }
            else
            {
                // Free-form box (not horizontal)
                float dx1 = poly[6] - poly[0];
                float dy1 = poly[7] - poly[1];
                float height = MathF.Sqrt(dx1 * dx1 + dy1 * dy1);

                float dx2 = poly[2] - poly[0];
                float dy2 = poly[3] - poly[1];
                float width = MathF.Sqrt(dx2 * dx2 + dy2 * dy2);

                int margin = (int)(1.44f * addMargin * Math.Min(width, height));

                float theta13 = MathF.Abs(MathF.Atan((poly[1] - poly[5]) / Math.Max(10f, poly[0] - poly[4])));
                float theta24 = MathF.Abs(MathF.Atan((poly[3] - poly[7]) / Math.Max(10f, poly[2] - poly[6])));

                float x1 = poly[0] - MathF.Cos(theta13) * margin;
                float y1 = poly[1] - MathF.Sin(theta13) * margin;
                float x2 = poly[2] + MathF.Cos(theta24) * margin;
                float y2 = poly[3] - MathF.Sin(theta24) * margin;
                float x3 = poly[4] + MathF.Cos(theta13) * margin;
                float y3 = poly[5] + MathF.Sin(theta13) * margin;
                float x4 = poly[6] - MathF.Cos(theta24) * margin;
                float y4 = poly[7] + MathF.Sin(theta24) * margin;

                freeList.Add(new[] { new[] { x1, y1 }, new[] { x2, y2 }, new[] { x3, y3 }, new[] { x4, y4 } });
            }
        }

        // Step 2: Sort horizontal boxes by y-center
        // Python lines 260-261
        if (sortOutput)
        {
            horizontalList = horizontalList.OrderBy(box => box[4]).ToList();  // Sort by yCenter (index 4)
        }

        // Step 3: Group boxes by line (similar y_center)
        // Python lines 263-280
        var newBox = new List<float[]>();
        var bHeight = new List<float>();
        var bYcenter = new List<float>();

        foreach (var poly in horizontalList)
        {
            if (newBox.Count == 0)
            {
                bHeight.Add(poly[5]);  // height
                bYcenter.Add(poly[4]); // yCenter
                newBox.Add(poly);
            }
            else
            {
                // Check if this box belongs to the same line
                if (Math.Abs(bYcenter.Average() - poly[4]) < ycenterThreshold * bHeight.Average())
                {
                    bHeight.Add(poly[5]);
                    bYcenter.Add(poly[4]);
                    newBox.Add(poly);
                }
                else
                {
                    // Start new line
                    combinedList.Add(newBox);
                    newBox = new List<float[]> { poly };
                    bHeight = new List<float> { poly[5] };
                    bYcenter = new List<float> { poly[4] };
                }
            }
        }
        combinedList.Add(newBox);

        // Step 4: Merge boxes on the same line
        // Python lines 282-329
        foreach (var boxes in combinedList)
        {
            if (boxes.Count == 1)
            {
                // Single box on this line
                var box = boxes[0];
                int margin = (int)(addMargin * Math.Min(box[1] - box[0], box[5]));
                mergedList.Add(new[] { box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin });
            }
            else
            {
                // Multiple boxes on same line - sort by x position
                var sortedBoxes = boxes.OrderBy(b => b[0]).ToList();

                var mergedBox = new List<List<float[]>>();
                newBox = new List<float[]>();
                bHeight = new List<float>();
                float xMax = 0;

                foreach (var box in sortedBoxes)
                {
                    if (newBox.Count == 0)
                    {
                        bHeight.Add(box[5]);
                        xMax = box[1];
                        newBox.Add(box);
                    }
                    else
                    {
                        // Check if boxes should be merged
                        float avgHeight = bHeight.Average();
                        bool heightOk = Math.Abs(avgHeight - box[5]) < heightThreshold * avgHeight;
                        bool widthOk = (box[0] - xMax) < widthThreshold * (box[3] - box[2]);

                        if (heightOk && widthOk)
                        {
                            bHeight.Add(box[5]);
                            xMax = box[1];
                            newBox.Add(box);
                        }
                        else
                        {
                            mergedBox.Add(newBox);
                            newBox = new List<float[]> { box };
                            bHeight = new List<float> { box[5] };
                            xMax = box[1];
                        }
                    }
                }
                if (newBox.Count > 0)
                {
                    mergedBox.Add(newBox);
                }

                // Create final merged boxes
                foreach (var mbox in mergedBox)
                {
                    if (mbox.Count != 1)
                    {
                        // Adjacent boxes - merge them
                        float xMin = mbox.Min(b => b[0]);
                        float xMaxFinal = mbox.Max(b => b[1]);
                        float yMin = mbox.Min(b => b[2]);
                        float yMax = mbox.Max(b => b[3]);

                        float boxWidth = xMaxFinal - xMin;
                        float boxHeight = yMax - yMin;
                        int margin = (int)(addMargin * Math.Min(boxWidth, boxHeight));

                        mergedList.Add(new[] { xMin - margin, xMaxFinal + margin, yMin - margin, yMax + margin });
                    }
                    else
                    {
                        // Non-adjacent box
                        var box = mbox[0];
                        float boxWidth = box[1] - box[0];
                        float boxHeight = box[3] - box[2];
                        int margin = (int)(addMargin * Math.Min(boxWidth, boxHeight));

                        mergedList.Add(new[] { box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin });
                    }
                }
            }
        }

        return (mergedList, freeList);
    }

    /// <summary>
    /// Wrapper method compatible with OcrEngine signature.
    /// Converts float[] boxes to int[] boxes for horizontal list.
    /// </summary>
    public static (List<int[]> horizontalList, List<float[]> freeList) GroupFlat(
        List<float[]> polys,
        float slopeThreshold = 0.1f,
        float ycenterThreshold = 0.5f,
        float heightThreshold = 0.5f,
        float widthThreshold = 1.0f,
        float addMargin = 0.05f,
        bool sortOutput = true,
        int imageWidth = 0,
        int imageHeight = 0)
    {
        // Call the main grouping function
        var (horizontalFloat, freeListNested) = Group(
            polys,
            slopeThreshold,
            ycenterThreshold,
            heightThreshold,
            widthThreshold,
            addMargin,
            sortOutput);

        // Convert horizontal list from float[] to int[] and clamp to image bounds
        var horizontalInt = new List<int[]>();
        foreach (var box in horizontalFloat)
        {
            // box format: [xMin, xMax, yMin, yMax]
            int xMin = Math.Max(0, (int)Math.Round(box[0]));
            int xMax = imageWidth > 0 ? Math.Min(imageWidth, (int)Math.Round(box[1])) : (int)Math.Round(box[1]);
            int yMin = Math.Max(0, (int)Math.Round(box[2]));
            int yMax = imageHeight > 0 ? Math.Min(imageHeight, (int)Math.Round(box[3])) : (int)Math.Round(box[3]);

            horizontalInt.Add(new[] { xMin, xMax, yMin, yMax });
        }

        // Flatten free list from float[][] to List<float[]>
        var freeListFlat = new List<float[]>();
        foreach (var freeBox in freeListNested)
        {
            // freeBox is already in format [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
            // We need to flatten to [x0, y0, x1, y1, x2, y2, x3, y3]
            var flat = new float[8];
            for (int i = 0; i < 4; i++)
            {
                flat[i * 2] = freeBox[i][0];
                flat[i * 2 + 1] = freeBox[i][1];
            }
            freeListFlat.Add(flat);
        }

        return (horizontalInt, freeListFlat);
    }
}
