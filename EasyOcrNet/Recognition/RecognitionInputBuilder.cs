using EasyOcrNet.Configuration;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System.Buffers;

namespace EasyOcrNet.Recognition;

internal static class RecognitionInputBuilder
{
    public static DenseTensor<float> Build(SKBitmap source, SKRect rect)
    {
        if (source is null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        int left = Math.Max(0, (int)Math.Floor(rect.Left) - OcrConstants.RecognitionPadding);
        int top = Math.Max(0, (int)Math.Floor(rect.Top) - OcrConstants.RecognitionPadding);
        int right = Math.Min(source.Width, (int)Math.Ceiling(rect.Right) + OcrConstants.RecognitionPadding);
        int bottom = Math.Min(source.Height, (int)Math.Ceiling(rect.Bottom) + OcrConstants.RecognitionPadding);

        if (right <= left)
        {
            right = Math.Min(source.Width, left + 1);
        }

        if (bottom <= top)
        {
            bottom = Math.Min(source.Height, top + 1);
        }

        int width = Math.Max(1, right - left);
        int height = Math.Max(1, bottom - top);

        using var cropped = new SKBitmap(width, height, source.ColorType, source.AlphaType);
        using (var canvas = new SKCanvas(cropped))
        {
            canvas.DrawBitmap(source, new SKRect(left, top, right, bottom), new SKRect(0, 0, width, height));
        }

        int targetHeight = OcrConstants.RecognizerInputHeight;
        float scale = targetHeight / (float)height;
        int targetWidth = Math.Clamp((int)Math.Ceiling(width * scale), 1, OcrConstants.RecognizerMaxWidth);

        var grayscaleBuffer = ArrayPool<float>.Shared.Rent(width * height);
        try
        {
            FillGrayscale(cropped, grayscaleBuffer.AsSpan(0, width * height));

            var resizedBuffer = ArrayPool<float>.Shared.Rent(targetWidth * targetHeight);
            try
            {
                Resize(grayscaleBuffer.AsSpan(0, width * height), width, height, resizedBuffer.AsSpan(0, targetWidth * targetHeight), targetWidth, targetHeight);

                var normalized = GC.AllocateUninitializedArray<float>(targetHeight * OcrConstants.RecognizerMaxWidth);
                Normalize(resizedBuffer.AsSpan(0, targetWidth * targetHeight), targetWidth, normalized, targetHeight);

                return new DenseTensor<float>(normalized, new[] { 1, 1, targetHeight, OcrConstants.RecognizerMaxWidth });
            }
            finally
            {
                ArrayPool<float>.Shared.Return(resizedBuffer);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(grayscaleBuffer);
        }
    }

    private static void FillGrayscale(SKBitmap bitmap, Span<float> destination)
    {
        var pixels = bitmap.Pixels;
        int count = Math.Min(pixels.Length, destination.Length);
        for (int i = 0; i < count; i++)
        {
            var pixel = pixels[i];
            destination[i] = 0.299f * pixel.Red + 0.587f * pixel.Green + 0.114f * pixel.Blue;
        }
    }

    private static void Resize(Span<float> source, int width, int height, Span<float> destination, int targetWidth, int targetHeight)
    {
        float yScale = height / (float)targetHeight;
        float xScale = width / (float)targetWidth;

        for (int row = 0; row < targetHeight; row++)
        {
            float srcY = (row + 0.5f) * yScale - 0.5f;
            int y0 = (int)MathF.Floor(srcY);
            float yLerp = srcY - y0;
            if (y0 < 0)
            {
                y0 = 0;
                yLerp = 0f;
            }

            int y1 = Math.Min(y0 + 1, height - 1);

            for (int col = 0; col < targetWidth; col++)
            {
                float srcX = (col + 0.5f) * xScale - 0.5f;
                int x0 = (int)MathF.Floor(srcX);
                float xLerp = srcX - x0;
                if (x0 < 0)
                {
                    x0 = 0;
                    xLerp = 0f;
                }

                int x1 = Math.Min(x0 + 1, width - 1);

                float topLeft = source[y0 * width + x0];
                float topRight = source[y0 * width + x1];
                float bottomLeft = source[y1 * width + x0];
                float bottomRight = source[y1 * width + x1];

                float upper = topLeft + (topRight - topLeft) * xLerp;
                float lower = bottomLeft + (bottomRight - bottomLeft) * xLerp;
                destination[row * targetWidth + col] = upper + (lower - upper) * yLerp;
            }
        }
    }

    private static void Normalize(Span<float> source, int sourceWidth, float[] destination, int targetHeight)
    {
        const float scale = 1f / 127.5f;
        int paddedWidth = OcrConstants.RecognizerMaxWidth;

        for (int row = 0; row < targetHeight; row++)
        {
            int destOffset = row * paddedWidth;
            int srcOffset = row * sourceWidth;
            int col = 0;
            float lastValue = source[srcOffset + Math.Max(0, sourceWidth - 1)];
            float padded = lastValue * scale - 1f;

            for (; col < sourceWidth; col++)
            {
                destination[destOffset + col] = source[srcOffset + col] * scale - 1f;
            }

            for (; col < paddedWidth; col++)
            {
                destination[destOffset + col] = padded;
            }
        }
    }
}
