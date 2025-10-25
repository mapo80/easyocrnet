using System.CommandLine;
using System.Text;
using EasyOcrNet.Detection;
using EasyOcrNet.Models;
using SkiaSharp;

namespace EasyOcrNet.Cli;

class Program
{
    static async Task<int> Main(string[] args)
    {
        var rootCommand = new RootCommand("EasyOcrNet CLI - OCR testing and comparison tool");

        // Detect command for Phase 2 testing
        var detectCommand = new Command("detect", "Run detection only (Phase 2 test)");

        var imageOption = new Option<string>(
            name: "--image",
            description: "Path to input image")
        { IsRequired = true };

        var modelOption = new Option<string>(
            name: "--model",
            description: "Path to detection ONNX model",
            getDefaultValue: () => "models/EasyOCRDetector.onnx");

        var outputOption = new Option<string?>(
            name: "--output",
            description: "Output file path (default: <image>.ocr.csharp.txt)");

        var compareOption = new Option<string?>(
            name: "--compare",
            description: "Python output file to compare against");

        var textThresholdOption = new Option<float>(
            name: "--text-threshold",
            description: "Text confidence threshold",
            getDefaultValue: () => 0.7f);

        var linkThresholdOption = new Option<float>(
            name: "--link-threshold",
            description: "Link confidence threshold",
            getDefaultValue: () => 0.4f);

        var lowTextOption = new Option<float>(
            name: "--low-text",
            description: "Low text threshold",
            getDefaultValue: () => 0.4f);

        detectCommand.AddOption(imageOption);
        detectCommand.AddOption(modelOption);
        detectCommand.AddOption(outputOption);
        detectCommand.AddOption(compareOption);
        detectCommand.AddOption(textThresholdOption);
        detectCommand.AddOption(linkThresholdOption);
        detectCommand.AddOption(lowTextOption);

        detectCommand.SetHandler(
            DetectAsync,
            imageOption,
            modelOption,
            outputOption,
            compareOption,
            textThresholdOption,
            linkThresholdOption,
            lowTextOption);

        rootCommand.AddCommand(detectCommand);

        return await rootCommand.InvokeAsync(args);
    }

    static async Task DetectAsync(
        string imagePath,
        string modelPath,
        string? outputPath,
        string? comparePath,
        float textThreshold,
        float linkThreshold,
        float lowText)
    {
        try
        {
            Console.WriteLine($"EasyOcrNet Detection Test (Phase 2)");
            Console.WriteLine($"==================================");
            Console.WriteLine($"Image: {imagePath}");
            Console.WriteLine($"Model: {modelPath}");
            Console.WriteLine($"Text threshold: {textThreshold}");
            Console.WriteLine($"Link threshold: {linkThreshold}");
            Console.WriteLine($"Low text: {lowText}");
            Console.WriteLine();

            // Validate inputs
            if (!File.Exists(imagePath))
            {
                Console.WriteLine($"Error: Image file not found: {imagePath}");
                return;
            }

            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"Error: Model file not found: {modelPath}");
                return;
            }

            // Set default output path
            outputPath ??= imagePath + ".ocr.csharp.txt";

            // Load image
            Console.WriteLine("Loading image...");
            using var bitmap = SKBitmap.Decode(imagePath);
            if (bitmap == null)
            {
                Console.WriteLine("Error: Failed to decode image");
                return;
            }
            Console.WriteLine($"Image size: {bitmap.Width}x{bitmap.Height}");

            // Create detector
            var config = new OcrConfig(
                TextThreshold: textThreshold,
                LinkThreshold: linkThreshold,
                LowText: lowText);

            Console.WriteLine("Initializing detector...");
            using var detector = new CraftDetector(modelPath, config);

            // Run detection
            Console.WriteLine("Running detection...");
            var startTime = DateTime.Now;
            var detections = await detector.DetectAsync(bitmap);
            var elapsed = (DateTime.Now - startTime).TotalSeconds;

            Console.WriteLine($"Detected {detections.Count} text regions in {elapsed:F2}s");
            Console.WriteLine();

            // Write results to file
            await WriteDetectionResults(outputPath, detections);
            Console.WriteLine($"Results written to: {outputPath}");

            // Compare with Python if requested
            if (!string.IsNullOrEmpty(comparePath))
            {
                await CompareResults(outputPath, comparePath);
            }
            else
            {
                // Display first few detections
                Console.WriteLine();
                Console.WriteLine("First 5 detections:");
                for (int i = 0; i < Math.Min(5, detections.Count); i++)
                {
                    var det = detections[i];
                    Console.WriteLine($"{i + 1}. {det.BoundingBox.ToOutputString()}");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }

    static async Task WriteDetectionResults(string outputPath, List<DetectionResult> detections)
    {
        var sb = new StringBuilder();
        foreach (var detection in detections)
        {
            // Format: (x1,y1) (x2,y2) (x3,y3) (x4,y4)
            sb.AppendLine(detection.BoundingBox.ToOutputString());
        }

        await File.WriteAllTextAsync(outputPath, sb.ToString());
    }

    static async Task CompareResults(string csharpPath, string pythonPath)
    {
        Console.WriteLine();
        Console.WriteLine("=== COMPARISON: C# vs Python ===");
        Console.WriteLine();

        if (!File.Exists(pythonPath))
        {
            Console.WriteLine($"Error: Python results file not found: {pythonPath}");
            return;
        }

        var csharpLines = await File.ReadAllLinesAsync(csharpPath);
        var pythonLines = await File.ReadAllLinesAsync(pythonPath);

        Console.WriteLine($"C# detections:     {csharpLines.Length}");
        Console.WriteLine($"Python detections: {pythonLines.Length}");
        Console.WriteLine();

        if (csharpLines.Length != pythonLines.Length)
        {
            Console.WriteLine($"WARNING: Detection count mismatch!");
            Console.WriteLine($"  C#:     {csharpLines.Length}");
            Console.WriteLine($"  Python: {pythonLines.Length}");
            Console.WriteLine();
        }

        // Compare bounding boxes
        int matchCount = 0;
        int totalCompared = Math.Min(csharpLines.Length, pythonLines.Length);
        var differences = new List<(int index, string csharp, string python)>();

        for (int i = 0; i < totalCompared; i++)
        {
            var csharpLine = csharpLines[i].Trim();
            var pythonLine = pythonLines[i].Trim();

            // Extract just the bounding box part (before any | separator)
            var csharpBox = csharpLine.Split('|')[0].Trim();
            var pythonBox = pythonLine.Split('|')[0].Trim();

            // Parse coordinates for tolerance comparison
            var csharpCoords = ParseBoundingBox(csharpBox);
            var pythonCoords = ParseBoundingBox(pythonBox);

            if (csharpCoords != null && pythonCoords != null)
            {
                var maxDiff = MaxCoordinateDifference(csharpCoords, pythonCoords);

                if (maxDiff <= 2.0) // Allow 2 pixel tolerance
                {
                    matchCount++;
                }
                else
                {
                    differences.Add((i + 1, csharpBox, pythonBox));
                }
            }
            else if (csharpBox == pythonBox)
            {
                matchCount++;
            }
            else
            {
                differences.Add((i + 1, csharpBox, pythonBox));
            }
        }

        double accuracy = totalCompared > 0 ? (matchCount * 100.0 / totalCompared) : 0;

        Console.WriteLine($"Bounding Box Accuracy: {accuracy:F2}% ({matchCount}/{totalCompared} match)");
        Console.WriteLine();

        if (differences.Count > 0)
        {
            Console.WriteLine($"First {Math.Min(5, differences.Count)} differences:");
            foreach (var (index, csharp, python) in differences.Take(5))
            {
                Console.WriteLine($"{index}. DIFFERENT:");
                Console.WriteLine($"   C#:  {csharp}");
                Console.WriteLine($"   Py:  {python}");

                var csharpCoords = ParseBoundingBox(csharp);
                var pythonCoords = ParseBoundingBox(python);
                if (csharpCoords != null && pythonCoords != null)
                {
                    var maxDiff = MaxCoordinateDifference(csharpCoords, pythonCoords);
                    Console.WriteLine($"   Max diff: {maxDiff:F1} pixels");
                }
                Console.WriteLine();
            }
        }
        else
        {
            Console.WriteLine("All bounding boxes match (within 2px tolerance)!");
        }
    }

    static List<(int x, int y)>? ParseBoundingBox(string bbox)
    {
        try
        {
            // Format: (x1,y1) (x2,y2) (x3,y3) (x4,y4)
            var points = bbox.Trim()
                .Split(new[] { '(', ')', ' ' }, StringSplitOptions.RemoveEmptyEntries);

            var coords = new List<(int x, int y)>();
            foreach (var point in points)
            {
                var parts = point.Split(',');
                if (parts.Length == 2)
                {
                    coords.Add((int.Parse(parts[0]), int.Parse(parts[1])));
                }
            }

            return coords.Count == 4 ? coords : null;
        }
        catch
        {
            return null;
        }
    }

    static double MaxCoordinateDifference(List<(int x, int y)> coords1, List<(int x, int y)> coords2)
    {
        if (coords1.Count != coords2.Count)
            return double.MaxValue;

        double maxDiff = 0;
        for (int i = 0; i < coords1.Count; i++)
        {
            var dx = Math.Abs(coords1[i].x - coords2[i].x);
            var dy = Math.Abs(coords1[i].y - coords2[i].y);
            var dist = Math.Sqrt(dx * dx + dy * dy);
            maxDiff = Math.Max(maxDiff, dist);
        }

        return maxDiff;
    }
}
