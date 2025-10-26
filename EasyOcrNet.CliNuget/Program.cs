using EasyOcrNet;
using SkiaSharp;
using System.Diagnostics;

Console.WriteLine("EasyOcrNet CLI - Using NuGet Package");
Console.WriteLine("=====================================\n");

// Check if dataset path is provided
string datasetPath = args.Length > 0 ? args[0] : "dataset/it";

if (!Directory.Exists(datasetPath))
{
    Console.WriteLine($"Dataset directory not found: {datasetPath}");
    Console.WriteLine("Usage: dotnet run [dataset-path]");
    return 1;
}

// Get all image files
var imageFiles = Directory.GetFiles(datasetPath, "*.png")
    .Concat(Directory.GetFiles(datasetPath, "*.jpg"))
    .Concat(Directory.GetFiles(datasetPath, "*.jpeg"))
    .ToList();

if (imageFiles.Count == 0)
{
    Console.WriteLine($"No images found in: {datasetPath}");
    return 1;
}

Console.WriteLine($"Found {imageFiles.Count} images in {datasetPath}\n");

// Initialize OCR Engine with models from NuGet package
Console.WriteLine("Initializing OCR Engine...");
Console.WriteLine("Models source: NuGet package EasyOcrNet");
Console.WriteLine("- Detector: models/detection.onnx");
Console.WriteLine("- Recognizer: models/latin_g2_rec.onnx");
Console.WriteLine("- Language: Italian (it)\n");

using var engine = new OcrEngine(
    detectorPath: "models/detection.onnx",
    recognizerPath: "models/latin_g2_rec.onnx",
    language: "it",
    charsetDirectory: "character"
);

Console.WriteLine("OCR Engine initialized successfully!\n");
Console.WriteLine("Processing images...\n");

var totalStopwatch = Stopwatch.StartNew();
var results = new List<(string filename, int detections, long ms)>();

foreach (var imagePath in imageFiles)
{
    var filename = Path.GetFileName(imagePath);
    Console.WriteLine($"Processing: {filename}");

    try
    {
        using var bitmap = SKBitmap.Decode(imagePath);

        var sw = Stopwatch.StartNew();
        var ocrResults = await engine.ProcessImageAsync(bitmap);
        sw.Stop();

        Console.WriteLine($"  Size: {bitmap.Width}x{bitmap.Height}");
        Console.WriteLine($"  Detections: {ocrResults.Count}");
        Console.WriteLine($"  Time: {sw.ElapsedMilliseconds}ms\n");

        results.Add((filename, ocrResults.Count, sw.ElapsedMilliseconds));

        // Display OCR results
        if (ocrResults.Count > 0)
        {
            Console.WriteLine("  OCR Results:");
            foreach (var result in ocrResults.Take(5)) // Show first 5
            {
                Console.WriteLine($"    - \"{result.Text}\" (confidence: {result.Confidence:F4})");
            }
            if (ocrResults.Count > 5)
            {
                Console.WriteLine($"    ... and {ocrResults.Count - 5} more");
            }
            Console.WriteLine();
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  ERROR: {ex.Message}\n");
    }
}

totalStopwatch.Stop();

// Summary
Console.WriteLine("\n========================================");
Console.WriteLine("SUMMARY");
Console.WriteLine("========================================");
Console.WriteLine($"Total images processed: {results.Count}");
Console.WriteLine($"Total detections: {results.Sum(r => r.detections)}");
Console.WriteLine($"Total time: {totalStopwatch.ElapsedMilliseconds}ms");
Console.WriteLine($"Average time per image: {(results.Count > 0 ? totalStopwatch.ElapsedMilliseconds / results.Count : 0)}ms");
Console.WriteLine();

if (results.Count > 0)
{
    Console.WriteLine("Per-image results:");
    foreach (var (filename, detections, ms) in results)
    {
        Console.WriteLine($"  {filename}: {detections} detections in {ms}ms");
    }
}

Console.WriteLine("\nDone!");
return 0;
