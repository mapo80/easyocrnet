using EasyOcrNet.Detection;
using EasyOcrNet.Models;
using EasyOcrNet.Recognition;
using SkiaSharp;

namespace EasyOcrNet;

/// <summary>
/// Complete OCR engine that orchestrates detection, grouping, and recognition.
/// Matches Python run_ocr() architecture:
/// 1. Detection (raw bounding boxes)
/// 2. Grouping (merge adjacent boxes)
/// 3. Recognition (extract and recognize text)
/// </summary>
public class OcrEngine : IDisposable
{
    private readonly IDetector _detector;
    private readonly IRecognizer _recognizer;
    private readonly OcrConfig _config;
    private bool _disposed;

    /// <summary>
    /// Initialize OCR engine with detector and recognizer
    /// </summary>
    /// <param name="detectorPath">Path to ONNX detection model</param>
    /// <param name="recognizerPath">Path to ONNX recognition model</param>
    /// <param name="language">Language code (e.g., "en", "it")</param>
    /// <param name="config">OCR configuration</param>
    /// <param name="charsetDirectory">Directory containing charset files</param>
    public OcrEngine(
        string detectorPath,
        string recognizerPath,
        string language,
        OcrConfig? config = null,
        string charsetDirectory = "character")
    {
        _config = config ?? new OcrConfig(Language: language);
        _detector = new CraftDetector(detectorPath, _config);
        _recognizer = new CrnnRecognizer(recognizerPath, language, _config, charsetDirectory);
    }

    /// <summary>
    /// Process image through complete OCR pipeline
    /// </summary>
    /// <param name="bitmap">Input image</param>
    /// <returns>List of OCR results with text and bounding boxes</returns>
    public async Task<List<OcrResult>> ProcessImageAsync(SKBitmap bitmap)
    {
        // 1. Detection (RAW bounding boxes, no grouping)
        var detections = await _detector.DetectAsync(bitmap);

        if (detections.Count == 0)
            return new List<OcrResult>();

        // 2. Group text boxes (merge adjacent boxes on same line)
        // This is where Python does grouping in run_ocr()
        var groupedDetections = GroupDetections(detections, bitmap.Width, bitmap.Height);

        // 3. Recognition (extract crops and recognize text)
        var results = new List<OcrResult>();
        foreach (var detection in groupedDetections)
        {
            var recognition = await _recognizer.RecognizeAsync(bitmap, detection);
            results.Add(new OcrResult(
                BoundingBox: detection.BoundingBox,
                Text: recognition.Text,
                Confidence: recognition.Confidence));
        }

        return results;
    }

    /// <summary>
    /// Group raw detections using text box merging algorithm.
    /// Matches Python: group_text_box() in craft_utils.py
    /// </summary>
    private List<DetectionResult> GroupDetections(List<DetectionResult> detections, int imageWidth, int imageHeight)
    {
        // Convert DetectionResult to flat format for grouping
        // Format: [x0, y0, x1, y1, x2, y2, x3, y3]
        var flatPolys = new List<float[]>();
        foreach (var det in detections)
        {
            var bbox = det.BoundingBox;
            flatPolys.Add(new[] {
                (float)bbox.TopLeft.X, (float)bbox.TopLeft.Y,
                (float)bbox.TopRight.X, (float)bbox.TopRight.Y,
                (float)bbox.BottomRight.X, (float)bbox.BottomRight.Y,
                (float)bbox.BottomLeft.X, (float)bbox.BottomLeft.Y
            });
        }

        // Apply grouping with thresholds from config
        var (horizontalList, freeList) = CraftUtils.GroupTextBoxFlat(
            flatPolys,
            slopeThreshold: _config.SlopeThreshold,
            ycenterThreshold: _config.YCenterThreshold,
            heightThreshold: _config.HeightThreshold,
            widthThreshold: _config.WidthThreshold,
            addMargin: _config.AddMargin,
            sortOutput: true,
            imageWidth: imageWidth,
            imageHeight: imageHeight);

        // Convert grouped boxes back to DetectionResult with coordinate clamping
        var convertedBoxes = ConvertGroupedBoxes(horizontalList, freeList, imageWidth, imageHeight);

        // DEBUG: Log boxes before filter
        Console.WriteLine($"[DEBUG] Before min_size filter: {convertedBoxes.Count} boxes");
        foreach (var box in convertedBoxes)
        {
            var bbox = box.BoundingBox;
            Console.WriteLine($"[DEBUG]   ({bbox.MinX},{bbox.MinY},{bbox.MaxX},{bbox.MaxY}) size={bbox.Width}x{bbox.Height}");
        }

        // Filter by min_size (matches Python: readtext() min_size filter)
        // Filter boxes where max(width, height) > minSize
        if (_config.MinSize > 0)
        {
            convertedBoxes = convertedBoxes
                .Where(det => Math.Max(det.BoundingBox.Width, det.BoundingBox.Height) > _config.MinSize)
                .ToList();
        }

        Console.WriteLine($"[DEBUG] After min_size filter: {convertedBoxes.Count} boxes");

        return convertedBoxes;
    }

    /// <summary>
    /// Convert grouped boxes to DetectionResult format with coordinate clamping
    /// </summary>
    private List<DetectionResult> ConvertGroupedBoxes(
        List<int[]> horizontalList,
        List<float[]> freeList,
        int imageWidth,
        int imageHeight)
    {
        var results = new List<DetectionResult>();

        // Add horizontal boxes (merged rectangles)
        // Format: [xMin, xMax, yMin, yMax]
        foreach (var region in horizontalList)
        {
            // Clamp coordinates to image bounds
            int xMin = Math.Max(0, region[0]);
            int xMax = Math.Min(imageWidth, region[1]);
            int yMin = Math.Max(0, region[2]);
            int yMax = Math.Min(imageHeight, region[3]);

            var boundingBox = new BoundingBox(
                TopLeft: new Point2D(xMin, yMin),
                TopRight: new Point2D(xMax, yMin),
                BottomRight: new Point2D(xMax, yMax),
                BottomLeft: new Point2D(xMin, yMax)
            );

            results.Add(new DetectionResult(boundingBox, Confidence: 1.0f));
        }

        // Add free-form boxes (rotated/non-horizontal)
        // Format: [x0, y0, x1, y1, x2, y2, x3, y3]
        foreach (var box in freeList)
        {
            if (box == null || box.Length < 8)
                continue;

            // Clamp coordinates to image bounds
            int x0 = Math.Clamp((int)box[0], 0, imageWidth);
            int y0 = Math.Clamp((int)box[1], 0, imageHeight);
            int x1 = Math.Clamp((int)box[2], 0, imageWidth);
            int y1 = Math.Clamp((int)box[3], 0, imageHeight);
            int x2 = Math.Clamp((int)box[4], 0, imageWidth);
            int y2 = Math.Clamp((int)box[5], 0, imageHeight);
            int x3 = Math.Clamp((int)box[6], 0, imageWidth);
            int y3 = Math.Clamp((int)box[7], 0, imageHeight);

            var boundingBox = new BoundingBox(
                TopLeft: new Point2D(x0, y0),
                TopRight: new Point2D(x1, y1),
                BottomRight: new Point2D(x2, y2),
                BottomLeft: new Point2D(x3, y3)
            );

            results.Add(new DetectionResult(boundingBox, Confidence: 1.0f));
        }

        return results;
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        (_detector as IDisposable)?.Dispose();
        (_recognizer as IDisposable)?.Dispose();
        _disposed = true;
    }
}
