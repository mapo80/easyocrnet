using EasyOcrNet.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace EasyOcrNet.Detection;

/// <summary>
/// CRAFT (Character Region Awareness For Text detection) detector.
/// Uses ONNX Runtime for inference with SkiaSharp for image processing.
/// </summary>
public class CraftDetector : IDetector
{
    private readonly InferenceSession _session;
    private readonly OcrConfig _config;
    private bool _disposed;

    /// <summary>
    /// Initialize CRAFT detector with ONNX model
    /// </summary>
    /// <param name="modelPath">Path to ONNX detection model</param>
    /// <param name="config">OCR configuration</param>
    public CraftDetector(string modelPath, OcrConfig config)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Detection model not found: {modelPath}");

        _config = config;

        // Configure ONNX Runtime session
        var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };

        _session = new InferenceSession(modelPath, sessionOptions);
    }

    /// <summary>
    /// Detect text regions in image
    /// </summary>
    public async Task<List<DetectionResult>> DetectAsync(SKBitmap bitmap)
    {
        return await Task.Run(() => Detect(bitmap));
    }

    private List<DetectionResult> Detect(SKBitmap bitmap)
    {
        // 1. Preprocess image
        var (input, ratio, heatmapSize) = PreprocessImage(bitmap);

        // 2. Run ONNX inference
        var scoreMap = RunInference(input);

        // 3. Post-process to extract bounding boxes
        var detections = PostprocessDetection(scoreMap, ratio);

        return detections;
    }

    /// <summary>
    /// Preprocess image for detection model.
    /// Matches Python: detector_preprocess()
    /// </summary>
    private (Tensor<float> input, float ratio, (int width, int height) heatmapSize) PreprocessImage(SKBitmap bitmap)
    {
        // 1. Convert BGR to RGB (SkiaSharp uses RGBA, extract RGB)
        using var rgbBitmap = ConvertToRgb(bitmap);

        // 2. Resize with aspect ratio preservation and pad to multiples of 32
        var (resized, ratio, heatmapSize) = CraftUtils.ResizeAspectRatio(
            rgbBitmap,
            squareSize: 2560,
            filterQuality: SKFilterQuality.Low, // Matches cv2.INTER_LINEAR
            magRatio: 1.0f);

        // 3. Convert to float array (H, W, C)
        var imgArray = BitmapToFloatArray(resized);
        resized.Dispose();

        // 4. Normalize with ImageNet mean and variance
        var normalized = CraftUtils.NormalizeMeanVariance(imgArray);

        // 5. Transpose from (H, W, C) to (C, H, W)
        int height = normalized.GetLength(0);
        int width = normalized.GetLength(1);
        int channels = normalized.GetLength(2);

        var transposed = new float[channels, height, width];
        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    transposed[c, y, x] = normalized[y, x, c];
                }
            }
        }

        // 6. Add batch dimension: (1, C, H, W)
        var dimensions = new[] { 1, channels, height, width };
        var tensorData = new float[1 * channels * height * width];

        int idx = 0;
        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    tensorData[idx++] = transposed[c, y, x];
                }
            }
        }

        var tensor = new DenseTensor<float>(tensorData, dimensions);

        return (tensor, ratio, heatmapSize);
    }

    /// <summary>
    /// Run ONNX inference on preprocessed input
    /// </summary>
    private float[,,,] RunInference(Tensor<float> input)
    {
        // Create input container
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_session.InputNames[0], input)
        };

        // Run inference
        using var results = _session.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();

        // Get output shape (should be: 1, H/2, W/2, 2)
        var outputMeta = _session.OutputMetadata[_session.OutputNames[0]];
        var shape = outputMeta.Dimensions;

        int batch = (int)shape[0];
        int height = (int)(input.Dimensions[2] / 2); // Output is half the input size
        int width = (int)(input.Dimensions[3] / 2);
        int channels = 2; // text and link channels

        // Reshape to 4D array
        var scoreMap = new float[batch, height, width, channels];
        int idx = 0;

        for (int b = 0; b < batch; b++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        scoreMap[b, y, x, c] = output[idx++];
                    }
                }
            }
        }

        return scoreMap;
    }

    /// <summary>
    /// Post-process detection output to extract bounding boxes.
    /// Matches Python: detector_postprocess()
    /// </summary>
    private List<DetectionResult> PostprocessDetection(float[,,,] scoreMap, float ratio)
    {
        // Extract text and link score maps
        int height = scoreMap.GetLength(1);
        int width = scoreMap.GetLength(2);

        var scoreText = new float[height, width];
        var scoreLink = new float[height, width];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                scoreText[y, x] = scoreMap[0, y, x, 0];
                scoreLink[y, x] = scoreMap[0, y, x, 1];
            }
        }

        // Get detection boxes using CRAFT post-processing
        var (boxes, _, _) = CraftUtils.GetDetBoxes(
            scoreText,
            scoreLink,
            textThreshold: _config.TextThreshold,
            linkThreshold: _config.LinkThreshold,
            lowText: _config.LowText,
            poly: false,
            estimateNumChars: false);

        if (boxes.Count == 0)
            return new List<DetectionResult>();

        // Scale back to original image coordinates
        float ratioW = 1.0f / ratio;
        float ratioH = 1.0f / ratio;
        boxes = CraftUtils.AdjustResultCoordinates(boxes, ratioW, ratioH);

        // Convert boxes to flat format for GroupTextBox
        // From [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] to [x1, y1, x2, y2, x3, y3, x4, y4]
        var flatPolys = new List<float[]>();
        foreach (var box in boxes)
        {
            var flat = new float[8];
            flat[0] = box[0][0]; // x1
            flat[1] = box[0][1]; // y1
            flat[2] = box[1][0]; // x2
            flat[3] = box[1][1]; // y2
            flat[4] = box[2][0]; // x3
            flat[5] = box[2][1]; // y3
            flat[6] = box[3][0]; // x4
            flat[7] = box[3][1]; // y4
            flatPolys.Add(flat);
        }

        // Group text boxes (merge adjacent boxes on same line)
        // This matches Python's run_ocr() implementation
        var (horizontalList, freeList) = CraftUtils.GroupTextBoxFlat(
            flatPolys,
            slopeThreshold: _config.SlopeThreshold,
            ycenterThreshold: _config.YCenterThreshold,
            heightThreshold: _config.HeightThreshold,
            widthThreshold: _config.WidthThreshold,
            addMargin: _config.AddMargin,
            sortOutput: true);

        // Convert grouped boxes to DetectionResult format
        var detections = new List<DetectionResult>();

        // Add horizontal boxes (merged rectangles)
        foreach (var region in horizontalList)
        {
            // region format: [xMin, xMax, yMin, yMax]
            var boundingBox = new BoundingBox(
                TopLeft: new Point2D(region[0], region[2]),
                TopRight: new Point2D(region[1], region[2]),
                BottomRight: new Point2D(region[1], region[3]),
                BottomLeft: new Point2D(region[0], region[3])
            );

            detections.Add(new DetectionResult(boundingBox, Confidence: 1.0f));
        }

        // Add free-form boxes (rotated/non-horizontal)
        foreach (var box in freeList)
        {
            if (box == null || box.Length < 8)
                continue;

            // box format: [x0, y0, x1, y1, x2, y2, x3, y3]
            var boundingBox = new BoundingBox(
                TopLeft: new Point2D((int)box[0], (int)box[1]),
                TopRight: new Point2D((int)box[2], (int)box[3]),
                BottomRight: new Point2D((int)box[4], (int)box[5]),
                BottomLeft: new Point2D((int)box[6], (int)box[7])
            );

            detections.Add(new DetectionResult(boundingBox, Confidence: 1.0f));
        }

        return detections;
    }

    /// <summary>
    /// Convert SkiaSharp bitmap to RGB (removing alpha channel if present)
    /// </summary>
    private SKBitmap ConvertToRgb(SKBitmap bitmap)
    {
        // Create RGB888 bitmap (no alpha)
        var rgbInfo = new SKImageInfo(
            bitmap.Width,
            bitmap.Height,
            SKColorType.Rgb888x,
            SKAlphaType.Opaque);

        var rgbBitmap = new SKBitmap(rgbInfo);

        using (var canvas = new SKCanvas(rgbBitmap))
        {
            canvas.Clear(SKColors.White);
            canvas.DrawBitmap(bitmap, 0, 0);
        }

        return rgbBitmap;
    }

    /// <summary>
    /// Convert bitmap to float array (H, W, C) with values in [0, 255]
    /// </summary>
    private float[,,] BitmapToFloatArray(SKBitmap bitmap)
    {
        int height = bitmap.Height;
        int width = bitmap.Width;
        var array = new float[height, width, 3];

        unsafe
        {
            var ptr = (byte*)bitmap.GetPixels().ToPointer();
            int bytesPerPixel = bitmap.BytesPerPixel;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int offset = (y * width + x) * bytesPerPixel;

                    // SkiaSharp RGB888x format: R, G, B, X
                    array[y, x, 0] = ptr[offset + 0]; // R
                    array[y, x, 1] = ptr[offset + 1]; // G
                    array[y, x, 2] = ptr[offset + 2]; // B
                }
            }
        }

        return array;
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _session?.Dispose();
        _disposed = true;
    }
}
