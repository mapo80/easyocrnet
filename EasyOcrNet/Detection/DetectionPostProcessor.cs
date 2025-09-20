using EasyOcrNet.Configuration;
using SkiaSharp;
using System.Buffers;

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
        var data = output.Data;

        float scaleX = width / (float)detW;
        float scaleY = height / (float)detH;

        int totalCells = detH * detW;
        var visited = new bool[totalCells];
        var components = new List<SKRect>(Math.Max(4, totalCells / 32));
        var queueBuffer = ArrayPool<int>.Shared.Rent(totalCells);

        var offsetsY = new[] { -1, -1, -1, 0, 0, 1, 1, 1 };
        var offsetsX = new[] { -1, 0, 1, -1, 1, -1, 0, 1 };

        try
        {
            for (int index = 0; index < totalCells; index++)
            {
                if (visited[index])
                {
                    continue;
                }

                float scoreText = data[index * channels];
                if (scoreText < OcrConstants.DetectorLowTextThreshold)
                {
                    continue;
                }

                visited[index] = true;

                int head = 0;
                int tail = 0;
                queueBuffer[tail++] = index;

                int minX = index % detW;
                int maxX = minX;
                int minY = index / detW;
                int maxY = minY;
                bool hasStrongText = scoreText >= OcrConstants.DetectorTextScoreThreshold;

                while (head < tail)
                {
                    int current = queueBuffer[head++];
                    int cy = current / detW;
                    int cx = current - cy * detW;
                    int currentIndex = current * channels;
                    float currentText = data[currentIndex];
                    float currentLink = channels > 1 ? data[currentIndex + 1] : 0f;

                    if (currentText >= OcrConstants.DetectorTextScoreThreshold)
                    {
                        hasStrongText = true;
                    }

                    if (currentLink >= OcrConstants.DetectorTextLinkThreshold)
                    {
                        hasStrongText = true;
                    }

                    if (cx < minX) minX = cx;
                    if (cx > maxX) maxX = cx;
                    if (cy < minY) minY = cy;
                    if (cy > maxY) maxY = cy;

                    for (int direction = 0; direction < offsetsY.Length; direction++)
                    {
                        int ny = cy + offsetsY[direction];
                        int nx = cx + offsetsX[direction];
                        if ((uint)ny >= (uint)detH || (uint)nx >= (uint)detW)
                        {
                            continue;
                        }

                        int neighbor = ny * detW + nx;
                        if (visited[neighbor])
                        {
                            continue;
                        }

                        int neighborIndex = neighbor * channels;
                        float neighborText = data[neighborIndex];
                        float neighborLink = channels > 1 ? data[neighborIndex + 1] : 0f;
                        if (neighborText >= OcrConstants.DetectorLowTextThreshold || neighborLink >= OcrConstants.DetectorTextLinkThreshold)
                        {
                            visited[neighbor] = true;
                            queueBuffer[tail++] = neighbor;
                        }
                    }
                }

                if (hasStrongText)
                {
                    float left = MathF.Max(0f, minX * scaleX);
                    float top = MathF.Max(0f, minY * scaleY);
                    float right = MathF.Min(width, (maxX + 1) * scaleX);
                    float bottom = MathF.Min(height, (maxY + 1) * scaleY);
                    components.Add(new SKRect(left, top, right, bottom));
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(queueBuffer);
        }

        return components;
    }
}
