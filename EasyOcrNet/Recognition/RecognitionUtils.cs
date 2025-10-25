using SkiaSharp;

namespace EasyOcrNet.Recognition;

/// <summary>
/// Recognition preprocessing and decoding utilities.
/// Matches Python ocr_process.py implementation exactly.
/// </summary>
public static class RecognitionUtils
{
    /// <summary>
    /// Preprocess cropped region for recognition.
    /// Matches Python: recognizer_preprocess()
    /// </summary>
    /// <param name="crop">Cropped SKBitmap (BGR format)</param>
    /// <param name="imgH">Target height (default 64)</param>
    /// <param name="imgW">Target width (default 200)</param>
    /// <returns>Preprocessed tensor ready for ONNX inference (1, 1, H, W)</returns>
    public static float[,,,] PreprocessCrop(SKBitmap crop, int imgH = 64, int imgW = 200)
    {
        // 1. Convert to grayscale
        var grayscale = ConvertToGrayscale(crop);

        int h = grayscale.GetLength(0);
        int w = grayscale.GetLength(1);

        // 2. Calculate resize width maintaining aspect ratio
        float ratio = w / (float)h;
        int resizedW = (int)(imgH * ratio) > imgW ? imgW : (int)(imgH * ratio);

        // 3. Resize with bilinear interpolation (matches cv2.INTER_LINEAR)
        var resized = ResizeBilinear(grayscale, imgH, resizedW);

        // 4. Normalize: [0,255] -> [0,1] -> [-1,1]
        var normalized = new float[imgH, resizedW];
        for (int y = 0; y < imgH; y++)
        {
            for (int x = 0; x < resizedW; x++)
            {
                float val = resized[y, x] / 255.0f;
                normalized[y, x] = (val - 0.5f) / 0.5f;
            }
        }

        // 5. Create padded output with last column repeated
        var padded = new float[imgH, imgW];
        for (int y = 0; y < imgH; y++)
        {
            // Copy resized content
            for (int x = 0; x < resizedW; x++)
            {
                padded[y, x] = normalized[y, x];
            }

            // Pad the rest with last column repeated
            if (resizedW < imgW)
            {
                float lastCol = normalized[y, resizedW - 1];
                for (int x = resizedW; x < imgW; x++)
                {
                    padded[y, x] = lastCol;
                }
            }
        }

        // 6. Add batch and channel dimensions: (1, 1, H, W)
        var tensor = new float[1, 1, imgH, imgW];
        for (int y = 0; y < imgH; y++)
        {
            for (int x = 0; x < imgW; x++)
            {
                tensor[0, 0, y, x] = padded[y, x];
            }
        }

        return tensor;
    }

    /// <summary>
    /// Adjust contrast of grayscale image to target level.
    /// Matches Python: adjust_contrast_grey()
    /// </summary>
    public static byte[,] AdjustContrastGrey(byte[,] img, float target = 0.4f)
    {
        int h = img.GetLength(0);
        int w = img.GetLength(1);

        // Calculate contrast
        var (contrast, high, low) = ContrastGrey(img);

        if (contrast >= target)
            return img;

        // Adjust contrast
        var adjusted = new byte[h, w];
        float ratio = 200.0f / Math.Max(10, high - low);

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int val = (int)((img[y, x] - low + 25) * ratio);
                adjusted[y, x] = (byte)Math.Clamp(val, 0, 255);
            }
        }

        return adjusted;
    }

    /// <summary>
    /// Calculate contrast of grayscale image.
    /// Matches Python: contrast_grey()
    /// </summary>
    private static (float contrast, float high, float low) ContrastGrey(byte[,] img)
    {
        int h = img.GetLength(0);
        int w = img.GetLength(1);
        int totalPixels = h * w;

        // Collect all pixel values
        var pixels = new List<byte>(totalPixels);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                pixels.Add(img[y, x]);
            }
        }

        pixels.Sort();

        // Calculate 10th and 90th percentiles
        int lowIdx = (int)(totalPixels * 0.1);
        int highIdx = (int)(totalPixels * 0.9);

        float low = pixels[lowIdx];
        float high = pixels[highIdx];

        float contrast = (high - low) / Math.Max(10, high + low);

        return (contrast, high, low);
    }

    /// <summary>
    /// Decode recognition output using CTC greedy decoding.
    /// Matches Python: decode_recognition()
    /// </summary>
    public static string DecodeRecognition(float[,,] output, List<string> charsetList)
    {
        int batch = output.GetLength(0);
        int timeSteps = output.GetLength(1);
        int numClasses = output.GetLength(2);

        // 1. Greedy decoding: take argmax
        var predsIndex = new int[timeSteps];
        for (int t = 0; t < timeSteps; t++)
        {
            int maxIdx = 0;
            float maxVal = output[0, t, 0];

            for (int c = 1; c < numClasses; c++)
            {
                if (output[0, t, c] > maxVal)
                {
                    maxVal = output[0, t, c];
                    maxIdx = c;
                }
            }

            predsIndex[t] = maxIdx;
        }

        // 2. CTC decoding: remove consecutive duplicates and blanks
        var result = new List<string>();
        int? prevIdx = null;

        for (int t = 0; t < timeSteps; t++)
        {
            int idx = predsIndex[t];

            // Skip blank (index 0)
            if (idx == 0)
            {
                prevIdx = idx;
                continue;
            }

            // Skip consecutive duplicates
            if (prevIdx.HasValue && idx == prevIdx.Value)
            {
                continue;
            }

            // Add character
            if (idx < charsetList.Count)
            {
                result.Add(charsetList[idx]);
            }

            prevIdx = idx;
        }

        return string.Concat(result);
    }

    /// <summary>
    /// Calculate confidence score from model output.
    /// Matches Python: calculate_confidence()
    /// </summary>
    public static float CalculateConfidence(float[,,] output, List<string> charsetList)
    {
        int timeSteps = output.GetLength(1);
        int numClasses = output.GetLength(2);

        // 1. Apply softmax
        var probabilities = Softmax(output);

        // 2. Get predicted indices
        var predsIndex = new int[timeSteps];
        for (int t = 0; t < timeSteps; t++)
        {
            int maxIdx = 0;
            float maxVal = output[0, t, 0];

            for (int c = 1; c < numClasses; c++)
            {
                if (output[0, t, c] > maxVal)
                {
                    maxVal = output[0, t, c];
                    maxIdx = c;
                }
            }

            predsIndex[t] = maxIdx;
        }

        // 3. Collect max probabilities for non-blank, non-duplicate positions
        var maxProbsList = new List<float>();
        int? prevIdx = null;

        for (int t = 0; t < timeSteps; t++)
        {
            int idx = predsIndex[t];

            // Skip blank and duplicates
            if (idx != 0 && (!prevIdx.HasValue || idx != prevIdx.Value))
            {
                maxProbsList.Add(probabilities[0, t, idx]);
            }

            prevIdx = idx;
        }

        // 4. Calculate custom mean: prod^(2/sqrt(n))
        if (maxProbsList.Count == 0)
            return 0.0f;

        return CustomMean(maxProbsList);
    }

    /// <summary>
    /// Calculate custom mean confidence score.
    /// Matches Python: custom_mean()
    /// Formula: prod(x)^(2.0/sqrt(len(x)))
    /// </summary>
    private static float CustomMean(List<float> values)
    {
        if (values.Count == 0)
            return 0.0f;

        // Calculate product of all values
        double product = 1.0;
        foreach (var val in values)
        {
            product *= val;
        }

        // Calculate exponent: 2.0 / sqrt(len)
        double exponent = 2.0 / Math.Sqrt(values.Count);

        // Result: prod^exponent
        return (float)Math.Pow(product, exponent);
    }

    /// <summary>
    /// Apply softmax to 3D output tensor.
    /// Matches Python: softmax(x, axis=2)
    /// </summary>
    private static float[,,] Softmax(float[,,] input)
    {
        int batch = input.GetLength(0);
        int timeSteps = input.GetLength(1);
        int numClasses = input.GetLength(2);

        var output = new float[batch, timeSteps, numClasses];

        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < timeSteps; t++)
            {
                // Find max for numerical stability
                float max = input[b, t, 0];
                for (int c = 1; c < numClasses; c++)
                {
                    if (input[b, t, c] > max)
                        max = input[b, t, c];
                }

                // Calculate exp(x - max) and sum
                float sum = 0;
                var expValues = new float[numClasses];
                for (int c = 0; c < numClasses; c++)
                {
                    expValues[c] = (float)Math.Exp(input[b, t, c] - max);
                    sum += expValues[c];
                }

                // Normalize
                for (int c = 0; c < numClasses; c++)
                {
                    output[b, t, c] = expValues[c] / sum;
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Convert SKBitmap to grayscale byte array.
    /// Matches cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    /// </summary>
    private static byte[,] ConvertToGrayscale(SKBitmap bitmap)
    {
        int h = bitmap.Height;
        int w = bitmap.Width;
        var grayscale = new byte[h, w];

        unsafe
        {
            var ptr = (byte*)bitmap.GetPixels().ToPointer();
            int bytesPerPixel = bitmap.BytesPerPixel;

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int offset = (y * w + x) * bytesPerPixel;

                    byte r, g, b;

                    // Handle different color types
                    if (bitmap.ColorType == SKColorType.Bgra8888 || bitmap.ColorType == SKColorType.Rgba8888)
                    {
                        b = ptr[offset + 0];
                        g = ptr[offset + 1];
                        r = ptr[offset + 2];
                    }
                    else if (bitmap.ColorType == SKColorType.Rgb888x)
                    {
                        r = ptr[offset + 0];
                        g = ptr[offset + 1];
                        b = ptr[offset + 2];
                    }
                    else if (bitmap.ColorType == SKColorType.Gray8)
                    {
                        grayscale[y, x] = ptr[offset];
                        continue;
                    }
                    else
                    {
                        // Fallback: use GetPixel (slower but works)
                        var color = bitmap.GetPixel(x, y);
                        r = color.Red;
                        g = color.Green;
                        b = color.Blue;
                    }

                    // ITU-R BT.601 conversion (same as OpenCV)
                    grayscale[y, x] = (byte)(0.299 * r + 0.587 * g + 0.114 * b);
                }
            }
        }

        return grayscale;
    }

    /// <summary>
    /// Resize grayscale image using bilinear interpolation.
    /// Matches cv2.resize(..., interpolation=cv2.INTER_LINEAR)
    /// </summary>
    private static byte[,] ResizeBilinear(byte[,] src, int dstH, int dstW)
    {
        int srcH = src.GetLength(0);
        int srcW = src.GetLength(1);

        var dst = new byte[dstH, dstW];

        float scaleX = srcW / (float)dstW;
        float scaleY = srcH / (float)dstH;

        for (int y = 0; y < dstH; y++)
        {
            for (int x = 0; x < dstW; x++)
            {
                float srcX = (x + 0.5f) * scaleX - 0.5f;
                float srcY = (y + 0.5f) * scaleY - 0.5f;

                int x0 = (int)Math.Floor(srcX);
                int y0 = (int)Math.Floor(srcY);
                int x1 = Math.Min(x0 + 1, srcW - 1);
                int y1 = Math.Min(y0 + 1, srcH - 1);

                x0 = Math.Max(0, x0);
                y0 = Math.Max(0, y0);

                float dx = srcX - x0;
                float dy = srcY - y0;

                // Bilinear interpolation
                float val = (1 - dy) * ((1 - dx) * src[y0, x0] + dx * src[y0, x1])
                          + dy * ((1 - dx) * src[y1, x0] + dx * src[y1, x1]);

                dst[y, x] = (byte)Math.Clamp(val, 0, 255);
            }
        }

        return dst;
    }
}
