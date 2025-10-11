using System;
using System.Collections.Generic;
using EasyOcrNet.Configuration;
using EasyOcrNet.ImageProcessing;
using SkiaSharp;

namespace EasyOcrNet.Detection;

internal static class DetectionPostProcessor
{
    public static List<SKRect> ExtractBoundingBoxes(ModelOutput output, int width, int height)
    {
        if (output.Rank != 4 || output[0] != 1)
        {
            return new List<SKRect>();
        }

        int detH = output[1];
        int detW = output[2];
        int channels = output[3];
        if (detH <= 0 || detW <= 0 || channels <= 0)
        {
            return new List<SKRect>();
        }

        var data = output.Data;
        var textMap = new float[detH, detW];
        var linkMap = new float[detH, detW];
        var combinedMask = new bool[detH, detW];

        for (int y = 0; y < detH; y++)
        {
            for (int x = 0; x < detW; x++)
            {
                int index = (y * detW + x) * channels;
                float textScore = data[index];
                float linkScore = channels > 1 ? data[index + 1] : 0f;

                textMap[y, x] = textScore;
                linkMap[y, x] = linkScore;

                bool textActive = textScore >= OcrConstants.DetectorLowTextThreshold;
                bool linkActive = linkScore >= OcrConstants.DetectorTextLinkThreshold;

                combinedMask[y, x] = textActive || linkActive;
            }
        }

        var components = ConnectedComponentsAnalyzer.Analyze(combinedMask);
        if (components.Count == 0)
        {
            return new List<SKRect>();
        }

        float scaleX = width / (float)detW;
        float scaleY = height / (float)detH;

        var results = new List<SKRect>(components.Count);
        var labels = components.Labels;

        foreach (var component in components.Stats)
        {
            if (component.Area < 10)
            {
                continue;
            }

            if (GetMaximumScore(textMap, labels, component) < OcrConstants.DetectorTextScoreThreshold)
            {
                continue;
            }

            var mask = ExtractComponentMask(labels, component);
            var dilated = DilateMask(mask, component);
            var contour = GeometryUtils.ExtractContour(dilated);
            if (contour.Count == 0)
            {
                continue;
            }

            var rect = GeometryUtils.MinAreaRect(contour);
            var bounds = ConvertToImageSpace(rect, component, scaleX, scaleY, width, height);
            if (bounds.Width > 0 && bounds.Height > 0)
            {
                results.Add(bounds);
            }
        }

        return results;
    }

    private static float GetMaximumScore(float[,] textMap, int[,] labels, ComponentStats component)
    {
        float max = 0f;
        int endY = component.Y + component.Height;
        int endX = component.X + component.Width;

        for (int y = component.Y; y < endY; y++)
        {
            for (int x = component.X; x < endX; x++)
            {
                if (labels[y, x] == component.Label)
                {
                    float score = textMap[y, x];
                    if (score > max)
                    {
                        max = score;
                    }
                }
            }
        }

        return max;
    }

    private static bool[,] ExtractComponentMask(int[,] labels, ComponentStats component)
    {
        var mask = new bool[component.Height, component.Width];
        int endY = component.Y + component.Height;
        int endX = component.X + component.Width;

        for (int y = component.Y; y < endY; y++)
        {
            for (int x = component.X; x < endX; x++)
            {
                if (labels[y, x] == component.Label)
                {
                    mask[y - component.Y, x - component.X] = true;
                }
            }
        }

        return mask;
    }

    private static bool[,] DilateMask(bool[,] mask, ComponentStats component)
    {
        int longestSide = Math.Max(component.Width, component.Height);
        int iterations = Math.Clamp(longestSide / 10, 1, 6);
        int kernelSize = EnsureOdd(iterations * 2 + 1);

        return MorphologyOps.Dilate(mask, kernelSize, kernelSize);
    }

    private static int EnsureOdd(int value)
    {
        if (value <= 0)
        {
            return 1;
        }

        return (value % 2 == 0) ? value + 1 : value;
    }

    private static SKRect ConvertToImageSpace(RotatedRect rect, ComponentStats component, float scaleX, float scaleY, int width, int height)
    {
        var corners = rect.GetCorners();

        float minX = float.PositiveInfinity;
        float maxX = float.NegativeInfinity;
        float minY = float.PositiveInfinity;
        float maxY = float.NegativeInfinity;

        for (int i = 0; i < corners.Length; i++)
        {
            double offsetX = corners[i].X + component.X;
            double offsetY = corners[i].Y + component.Y;

            float scaledX = (float)(offsetX * scaleX);
            float scaledY = (float)(offsetY * scaleY);

            if (scaledX < minX) minX = scaledX;
            if (scaledX > maxX) maxX = scaledX;
            if (scaledY < minY) minY = scaledY;
            if (scaledY > maxY) maxY = scaledY;
        }

        minX = Math.Clamp(minX, 0f, width);
        maxX = Math.Clamp(maxX, 0f, width);
        minY = Math.Clamp(minY, 0f, height);
        maxY = Math.Clamp(maxY, 0f, height);

        if (maxX <= minX || maxY <= minY)
        {
            return SKRect.Empty;
        }

        return new SKRect(minX, minY, maxX, maxY);
    }
}
