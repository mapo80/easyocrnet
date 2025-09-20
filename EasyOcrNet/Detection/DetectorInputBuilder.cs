using EasyOcrNet.Configuration;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace EasyOcrNet.Detection;

internal static class DetectorInputBuilder
{
    private static readonly float[] Mean = { 0.485f * 255f, 0.456f * 255f, 0.406f * 255f };
    private static readonly float[] Std = { 0.229f * 255f, 0.224f * 255f, 0.225f * 255f };

    public static DetectorInput Build(SKBitmap source)
    {
        if (source is null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        using var resized = new SKBitmap(OcrConstants.DetectorInputWidth, OcrConstants.DetectorInputHeight, source.ColorType, source.AlphaType);
        var sampling = new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None);
        source.ScalePixels(resized, sampling);

        int width = OcrConstants.DetectorInputWidth;
        int height = OcrConstants.DetectorInputHeight;
        int planeSize = width * height;
        var buffer = GC.AllocateUninitializedArray<float>(planeSize * 3);
        var pixelSpan = resized.Pixels;
        float invStd0 = 1f / Std[0];
        float invStd1 = 1f / Std[1];
        float invStd2 = 1f / Std[2];

        for (int y = 0; y < height; y++)
        {
            int rowOffset = y * width;
            for (int x = 0; x < width; x++)
            {
                int pixelIndex = rowOffset + x;
                var pixel = pixelSpan[pixelIndex];
                buffer[pixelIndex] = ((float)pixel.Red - Mean[0]) * invStd0;
                buffer[planeSize + pixelIndex] = ((float)pixel.Green - Mean[1]) * invStd1;
                buffer[planeSize * 2 + pixelIndex] = ((float)pixel.Blue - Mean[2]) * invStd2;
            }
        }

        var tensor = new DenseTensor<float>(buffer, new[] { 1, 3, height, width });
        var scaleX = source.Width / (float)width;
        var scaleY = source.Height / (float)height;

        return new DetectorInput(tensor, width, height, scaleX, scaleY);
    }
}

internal readonly record struct DetectorInput(DenseTensor<float> Tensor, int ResizedWidth, int ResizedHeight, float ScaleX, float ScaleY);
