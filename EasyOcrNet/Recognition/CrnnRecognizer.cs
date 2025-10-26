using EasyOcrNet.Models;
using EasyOcrNet.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace EasyOcrNet.Recognition;

/// <summary>
/// CRNN (Convolutional Recurrent Neural Network) text recognizer.
/// Uses ONNX Runtime for inference with SkiaSharp for image processing.
/// </summary>
public class CrnnRecognizer : IRecognizer
{
    private readonly InferenceSession _session;
    private readonly List<string> _charsetList;
    private readonly OcrConfig _config;
    private readonly int _imgW;
    private bool _disposed;

    /// <summary>
    /// Initialize CRNN recognizer with ONNX model and charset
    /// </summary>
    /// <param name="modelPath">Path to ONNX recognition model</param>
    /// <param name="language">Language code (e.g., "en", "it")</param>
    /// <param name="config">OCR configuration</param>
    /// <param name="charsetDirectory">Directory containing charset files</param>
    public CrnnRecognizer(string modelPath, string language, OcrConfig config, string charsetDirectory = "character")
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Recognition model not found: {modelPath}");

        _config = config;

        // Load charset
        var charset = CharsetLoader.Load(language, charsetDirectory);
        _charsetList = CharsetLoader.LoadAsListWithBlank(language, charsetDirectory);

        // Use imgW=896 for all models (larger = more details preserved)
        // Smaller images are handled fine with padding
        _imgW = 896;

        // Configure ONNX Runtime session for maximum performance
        var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_PARALLEL,
            InterOpNumThreads = 1,  // Use 1 thread for inter-op (simpler models)
            IntraOpNumThreads = Environment.ProcessorCount  // Use all cores for matrix ops
        };

        _session = new InferenceSession(modelPath, sessionOptions);
    }

    /// <summary>
    /// Recognize text from image with detection result
    /// </summary>
    public async Task<RecognitionResult> RecognizeAsync(SKBitmap bitmap, DetectionResult detection)
    {
        return await Task.Run(() =>
        {
            // Extract crop from bounding box
            var crop = ExtractCrop(bitmap, detection.BoundingBox);

            // Recognize crop
            return RecognizeCrop(crop);
        });
    }

    /// <summary>
    /// Recognize text from pre-cropped image
    /// </summary>
    public async Task<RecognitionResult> RecognizeCropAsync(SKBitmap crop)
    {
        return await Task.Run(() => RecognizeCrop(crop));
    }

    private RecognitionResult RecognizeCrop(SKBitmap crop)
    {
        // 1. Preprocess crop
        var input = RecognitionUtils.PreprocessCrop(crop, imgH: 64, imgW: _imgW);

        // 2. Convert to ONNX tensor
        var tensor = ConvertToTensor(input);

        // 3. Run inference
        var output = RunInference(tensor);

        // 4. Decode text using CTC
        var text = RecognitionUtils.DecodeRecognition(output, _charsetList);

        // 5. Calculate confidence
        var confidence = RecognitionUtils.CalculateConfidence(output, _charsetList);

        return new RecognitionResult(text, confidence);
    }

    /// <summary>
    /// Extract crop from bitmap using bounding box
    /// </summary>
    private SKBitmap ExtractCrop(SKBitmap bitmap, BoundingBox bbox)
    {
        // Calculate bounding rectangle (round float coordinates to int pixels)
        int minX = (int)Math.Round(Math.Min(Math.Min(bbox.TopLeft.X, bbox.TopRight.X),
                            Math.Min(bbox.BottomLeft.X, bbox.BottomRight.X)));
        int maxX = (int)Math.Round(Math.Max(Math.Max(bbox.TopLeft.X, bbox.TopRight.X),
                            Math.Max(bbox.BottomLeft.X, bbox.BottomRight.X)));
        int minY = (int)Math.Round(Math.Min(Math.Min(bbox.TopLeft.Y, bbox.TopRight.Y),
                            Math.Min(bbox.BottomLeft.Y, bbox.BottomRight.Y)));
        int maxY = (int)Math.Round(Math.Max(Math.Max(bbox.TopLeft.Y, bbox.TopRight.Y),
                            Math.Max(bbox.BottomLeft.Y, bbox.BottomRight.Y)));

        // Clamp to image bounds
        minX = Math.Max(0, minX);
        minY = Math.Max(0, minY);
        maxX = Math.Min(bitmap.Width, maxX);
        maxY = Math.Min(bitmap.Height, maxY);

        int width = maxX - minX;
        int height = maxY - minY;

        if (width <= 0 || height <= 0)
        {
            // Return empty 1x1 bitmap if invalid
            return new SKBitmap(1, 1);
        }

        // Extract crop
        var cropInfo = new SKImageInfo(width, height, bitmap.ColorType, bitmap.AlphaType);
        var crop = new SKBitmap(cropInfo);

        using (var canvas = new SKCanvas(crop))
        {
            canvas.Clear(SKColors.White);
            var srcRect = new SKRect(minX, minY, maxX, maxY);
            var dstRect = new SKRect(0, 0, width, height);
            canvas.DrawBitmap(bitmap, srcRect, dstRect);
        }

        return crop;
    }

    /// <summary>
    /// Convert 4D float array to ONNX tensor
    /// </summary>
    private Tensor<float> ConvertToTensor(float[,,,] input)
    {
        int batch = input.GetLength(0);
        int channels = input.GetLength(1);
        int height = input.GetLength(2);
        int width = input.GetLength(3);

        var dimensions = new[] { batch, channels, height, width };
        var tensorData = new float[batch * channels * height * width];

        int idx = 0;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        tensorData[idx++] = input[b, c, h, w];
                    }
                }
            }
        }

        return new DenseTensor<float>(tensorData, dimensions);
    }

    /// <summary>
    /// Run ONNX inference on preprocessed input
    /// </summary>
    private float[,,] RunInference(Tensor<float> input)
    {
        // Create input container
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_session.InputNames[0], input)
        };

        // Run inference
        using var results = _session.Run(inputs);
        var outputTensor = results.First();
        var outputArray = outputTensor.AsEnumerable<float>().ToArray();

        // Get actual shape from ONNX output tensor
        var tensorInfo = outputTensor.AsTensor<float>();
        var dimensions = tensorInfo.Dimensions.ToArray();

        // Output shape is (batch, time_steps, num_classes)
        int batch = dimensions[0];
        int timeSteps = dimensions[1];
        int numClasses = dimensions[2];

        // Validation: ensure reshape is valid
        if (outputArray.Length != batch * timeSteps * numClasses)
        {
            throw new InvalidOperationException(
                $"ONNX output reshape mismatch! " +
                $"Output length: {outputArray.Length}, " +
                $"Expected: {batch * timeSteps * numClasses} " +
                $"(batch={batch} × timeSteps={timeSteps} × numClasses={numClasses})");
        }

        // Reshape to 3D array
        var output = new float[batch, timeSteps, numClasses];
        int idx = 0;

        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < timeSteps; t++)
            {
                for (int c = 0; c < numClasses; c++)
                {
                    output[b, t, c] = outputArray[idx++];
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Recognize text from multiple detections (sequential with optimized ONNX session)
    /// </summary>
    public async Task<List<RecognitionResult>> RecognizeBatchAsync(SKBitmap bitmap, List<DetectionResult> detections)
    {
        return await Task.Run(() =>
        {
            if (detections == null || detections.Count == 0)
                return new List<RecognitionResult>();

            var results = new List<RecognitionResult>();

            foreach (var detection in detections)
            {
                var crop = ExtractCrop(bitmap, detection.BoundingBox);
                var result = RecognizeCrop(crop);
                results.Add(result);
                crop.Dispose();
            }

            return results;
        });
    }


    public void Dispose()
    {
        if (_disposed)
            return;

        _session?.Dispose();
        _disposed = true;
    }
}
