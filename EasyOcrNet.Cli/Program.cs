using System.CommandLine;
using System.IO;
using System.Text;
using EasyOcrNet.Assets;
using EasyOcrNet.Detection;
using EasyOcrNet.Recognition;
using EasyOcrNet.Models;
using SkiaSharp;

namespace EasyOcrNet.Cli;

class Program
{
    private static string ResolvePath(string path)
    {
        if (Path.IsPathRooted(path))
            return path;

        var baseDirectory = AppContext.BaseDirectory;
        return Path.GetFullPath(Path.Combine(baseDirectory, path));
    }

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

        // OCR command for Phase 3 testing (full pipeline)
        var ocrCommand = new Command("ocr", "Run full OCR pipeline (detection + recognition)");

        var ocrImageOption = new Option<string>(
            name: "--image",
            description: "Path to input image")
        { IsRequired = true };

        var detectorModelOption = new Option<string>(
            name: "--detector",
            description: "Path to detection ONNX model",
            getDefaultValue: () => "models/cpu/detection.onnx");

        var recognizerModelOption = new Option<string>(
            name: "--recognizer",
            description: "Path to recognition ONNX model",
            getDefaultValue: () => "models/cpu/english_g2_rec.onnx");

        var languageOption = new Option<string>(
            name: "--lang",
            description: "Language code (en, it, etc.)",
            getDefaultValue: () => "en");

        var ocrOutputOption = new Option<string?>(
            name: "--output",
            description: "Output file path (default: <image>.ocr.csharp.txt)");

        var ocrCompareOption = new Option<string?>(
            name: "--compare",
            description: "Python output file to compare against");

        ocrCommand.AddOption(ocrImageOption);
        ocrCommand.AddOption(detectorModelOption);
        ocrCommand.AddOption(recognizerModelOption);
        ocrCommand.AddOption(languageOption);
        ocrCommand.AddOption(ocrOutputOption);
        ocrCommand.AddOption(ocrCompareOption);

        ocrCommand.SetHandler(
            OcrAsync,
            ocrImageOption,
            detectorModelOption,
            recognizerModelOption,
            languageOption,
            ocrOutputOption,
            ocrCompareOption);

        rootCommand.AddCommand(ocrCommand);

        var downloadCommand = new Command("download", "Scarica i modelli ONNX dalla release GitHub di EasyOcrNet");

        var downloadDirOption = new Option<string>(
            name: "--dir",
            description: "Directory di destinazione per i modelli ONNX",
            getDefaultValue: () => "models/cpu");

        var repositoryOption = new Option<string>(
            name: "--repo",
            description: "Repository GitHub che contiene la release",
            getDefaultValue: () => "mapo80/easyocrnet");

        var releaseTagOption = new Option<string>(
            name: "--tag",
            description: "Tag della release da cui scaricare gli asset",
            getDefaultValue: () => "v2025.09.19");

        downloadCommand.AddOption(downloadDirOption);
        downloadCommand.AddOption(repositoryOption);
        downloadCommand.AddOption(releaseTagOption);

        downloadCommand.SetHandler(
            async (string dir, string repo, string tag) =>
            {
                var resolvedDir = ResolvePath(dir);

                Console.WriteLine("EasyOcrNet Asset Downloader");
                Console.WriteLine("===========================");
                Console.WriteLine($"Repository: {repo}");
                Console.WriteLine($"Release tag: {tag}");
                Console.WriteLine($"Directory destinazione: {resolvedDir}");
                Console.WriteLine();

                try
                {
                    var options = new GithubReleaseOptions(repo, tag);
                    await OcrReleaseDownloader.EnsureAllModelsAsync(
                        resolvedDir,
                        options,
                        Console.WriteLine);

                    Console.WriteLine();
                    Console.WriteLine("✓ Download completato. I file charset sono già integrati in EasyOcrNet.");
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Errore durante il download: {ex.Message}");
                }
            },
            downloadDirOption,
            repositoryOption,
            releaseTagOption);

        rootCommand.AddCommand(downloadCommand);

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
            var resolvedModelPath = ResolvePath(modelPath);

            Console.WriteLine($"EasyOcrNet Detection Test (Phase 2)");
            Console.WriteLine($"==================================");
            Console.WriteLine($"Image: {imagePath}");
            Console.WriteLine($"Model: {resolvedModelPath}");
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

            await OcrReleaseDownloader.EnsureModelAsync(
                resolvedModelPath,
                logger: Console.WriteLine);

            if (!File.Exists(resolvedModelPath))
            {
                Console.WriteLine($"Error: Model file not found: {resolvedModelPath}");
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
            using var detector = new CraftDetector(resolvedModelPath, config);

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

    static async Task OcrAsync(
        string imagePath,
        string detectorPath,
        string recognizerPath,
        string language,
        string? outputPath,
        string? comparePath)
    {
        try
        {
            var resolvedDetectorPath = ResolvePath(detectorPath);
            var resolvedRecognizerPath = ResolvePath(recognizerPath);

            Console.WriteLine($"EasyOcrNet Full OCR Test (Phase 3)");
            Console.WriteLine($"==================================");
            Console.WriteLine($"Image: {imagePath}");
            Console.WriteLine($"Detector: {resolvedDetectorPath}");
            Console.WriteLine($"Recognizer: {resolvedRecognizerPath}");
            Console.WriteLine($"Language: {language}");
            Console.WriteLine();

            // Validate inputs
            if (!File.Exists(imagePath))
            {
                Console.WriteLine($"Error: Image file not found: {imagePath}");
                return;
            }

            await OcrReleaseDownloader.EnsureModelAsync(
                resolvedDetectorPath,
                logger: Console.WriteLine);

            if (!File.Exists(resolvedDetectorPath))
            {
                Console.WriteLine($"Error: Detector model not found: {resolvedDetectorPath}");
                return;
            }

            await OcrReleaseDownloader.EnsureModelAsync(
                resolvedRecognizerPath,
                logger: Console.WriteLine);

            if (!File.Exists(resolvedRecognizerPath))
            {
                Console.WriteLine($"Error: Recognizer model not found: {resolvedRecognizerPath}");
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

            // Create configuration with default thresholds
            var config = new OcrConfig(Language: language);

            // Initialize OCR engine (handles detection + grouping + recognition)
            Console.WriteLine("Initializing OCR engine...");
            using var ocrEngine = new OcrEngine(resolvedDetectorPath, resolvedRecognizerPath, language, config);

            // Run full OCR pipeline
            Console.WriteLine("Running OCR pipeline (detection -> grouping -> recognition)...");
            var startTime = DateTime.Now;
            var ocrResults = await ocrEngine.ProcessImageAsync(bitmap);
            var totalTime = (DateTime.Now - startTime).TotalSeconds;

            Console.WriteLine($"Detected and recognized {ocrResults.Count} text regions in {totalTime:F2}s");
            Console.WriteLine($"Total time: {totalTime:F2}s");
            Console.WriteLine();

            // Write results to file
            await WriteOcrResults(outputPath, ocrResults);
            Console.WriteLine($"Results written to: {outputPath}");

            // Compare with Python if requested
            if (!string.IsNullOrEmpty(comparePath))
            {
                await CompareOcrResults(outputPath, comparePath);
            }
            else
            {
                // Display first few results
                Console.WriteLine();
                Console.WriteLine("First 3 results:");
                for (int i = 0; i < Math.Min(3, ocrResults.Count); i++)
                {
                    var result = ocrResults[i];
                    Console.WriteLine($"{i + 1}. {result.Text} ({result.Confidence:F4})");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }

    static async Task WriteOcrResults(string outputPath, List<OcrResult> results)
    {
        var sb = new StringBuilder();
        foreach (var result in results)
        {
            sb.AppendLine(result.ToOutputString());
        }

        await File.WriteAllTextAsync(outputPath, sb.ToString());
    }

    static async Task CompareOcrResults(string csharpPath, string pythonPath)
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

        Console.WriteLine($"C# results:     {csharpLines.Length}");
        Console.WriteLine($"Python results: {pythonLines.Length}");
        Console.WriteLine();

        // Compare text and confidence
        int exactMatches = 0;
        int bboxMatches = 0;
        var differences = new List<(int index, string csharp, string python)>();

        int totalCompared = Math.Min(csharpLines.Length, pythonLines.Length);

        for (int i = 0; i < totalCompared; i++)
        {
            var csharpParts = csharpLines[i].Split('|');
            var pythonParts = pythonLines[i].Split('|');

            if (csharpParts.Length < 3 || pythonParts.Length < 3)
                continue;

            var csharpBbox = csharpParts[0].Trim();
            var pythonBbox = pythonParts[0].Trim();
            var csharpText = csharpParts[1].Trim();
            var pythonText = pythonParts[1].Trim();

            // Check bbox match (with tolerance)
            var csharpCoords = ParseBoundingBox(csharpBbox);
            var pythonCoords = ParseBoundingBox(pythonBbox);
            bool bboxMatch = false;

            if (csharpCoords != null && pythonCoords != null)
            {
                var maxDiff = MaxCoordinateDifference(csharpCoords, pythonCoords);
                if (maxDiff <= 2.0)
                {
                    bboxMatches++;
                    bboxMatch = true;
                }
            }

            // Check text match
            if (csharpText == pythonText && bboxMatch)
            {
                exactMatches++;
            }
            else
            {
                differences.Add((i + 1, csharpLines[i], pythonLines[i]));
            }
        }

        double textAccuracy = totalCompared > 0 ? (exactMatches * 100.0 / totalCompared) : 0;
        double bboxAccuracy = totalCompared > 0 ? (bboxMatches * 100.0 / totalCompared) : 0;

        Console.WriteLine($"Bounding Box Accuracy: {bboxAccuracy:F2}% ({bboxMatches}/{totalCompared})");
        Console.WriteLine($"Text Accuracy: {textAccuracy:F2}% ({exactMatches}/{totalCompared})");
        Console.WriteLine();

        if (differences.Count > 0 && differences.Count <= 10)
        {
            Console.WriteLine($"Showing {differences.Count} differences:");
            foreach (var (index, csharp, python) in differences)
            {
                Console.WriteLine($"{index}. DIFFERENT:");
                Console.WriteLine($"   C#: {csharp}");
                Console.WriteLine($"   Py: {python}");
                Console.WriteLine();
            }
        }
        else if (differences.Count > 10)
        {
            Console.WriteLine($"First 5 of {differences.Count} differences:");
            foreach (var (index, csharp, python) in differences.Take(5))
            {
                Console.WriteLine($"{index}. DIFFERENT:");
                Console.WriteLine($"   C#: {csharp}");
                Console.WriteLine($"   Py: {python}");
                Console.WriteLine();
            }
        }
        else
        {
            Console.WriteLine("Perfect match! All results identical.");
        }
    }
}
