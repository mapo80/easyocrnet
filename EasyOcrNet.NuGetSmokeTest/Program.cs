using EasyOcrNet;
using EasyOcrNet.Configuration;
using EasyOcrNet.Languages;
using SkiaSharp;

static string LocateModels(string backend)
{
    var baseDir = AppContext.BaseDirectory;
    var candidates = new[]
    {
        Path.Combine(baseDir, "contentFiles", "any", "any", "models", backend),
        Path.Combine(baseDir, "models", backend),
    };

    foreach (var candidate in candidates)
    {
        if (Directory.Exists(candidate))
        {
            return candidate;
        }
    }

    throw new DirectoryNotFoundException(
        $"Could not locate packaged '{backend}' models under {string.Join(", ", candidates)}."
    );
}

static string LocateSample()
{
    var baseDir = AppContext.BaseDirectory;
    var candidates = new[]
    {
        Path.Combine(baseDir, "..", "..", "..", "..", "examples", "generated_1.png"),
        Path.Combine(Environment.CurrentDirectory, "examples", "generated_1.png"),
    };

    foreach (var candidate in candidates)
    {
        var path = Path.GetFullPath(candidate);
        if (File.Exists(path))
        {
            return path;
        }
    }

    throw new FileNotFoundException("Sample image not found in expected locations.");
}

var models = LocateModels("onnx");
var image = LocateSample();

using var bitmap = SKBitmap.Decode(image);
var options = new OcrOptions(models, OcrLanguage.Italian, InferenceBackend.Onnx);
using var engine = new EasyOcr(options);

var results = engine.Read(bitmap);
Console.WriteLine($"Recognised {results.Count} lines using models from: {models}");
foreach (var line in results)
{
    Console.WriteLine(line.Text);
}
