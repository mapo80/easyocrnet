using EasyOcrNet;
using EasyOcrNet.Configuration;
using EasyOcrNet.Languages;
using SkiaSharp;
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Globalization;

static OcrLanguage DeriveLanguage(string fileName)
{
    var name = Path.GetFileNameWithoutExtension(fileName).ToLowerInvariant();
    return name switch
    {
        _ when name.StartsWith("generated", StringComparison.Ordinal) => OcrLanguage.Italian,
        "english" or "example" or "example2" or "example3" or "easyocr_framework" or "width_ths" => OcrLanguage.English,
        "french" => OcrLanguage.French,
        "japanese" => OcrLanguage.Japanese,
        "korean" => OcrLanguage.Korean,
        "chinese" => OcrLanguage.SimplifiedChinese,
        "thai" => OcrLanguage.Thai,
        _ => OcrLanguage.English
    };
}

static void PrintUsage()
{
    Console.WriteLine("Usage: dotnet run --project ExampleExtractor -- <image-path> [--models <dir>] [--output <file>] [--language <name>] [--backend <onnx|openvino>] [--device <name>]");
}

if (args.Length == 0 || args.Contains("--help", StringComparer.OrdinalIgnoreCase))
{
    PrintUsage();
    return args.Length == 0 ? 1 : 0;
}

var imagePath = Path.GetFullPath(args[0]);
if (!File.Exists(imagePath))
{
    Console.Error.WriteLine($"Image not found: {imagePath}");
    return 1;
}

string? modelDirArg = null;
string? outputArg = null;
string? deviceArg = null;
OcrLanguage? languageOverride = null;
InferenceBackend backend = InferenceBackend.Onnx;

for (int i = 1; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--models":
            if (i + 1 >= args.Length)
            {
                Console.Error.WriteLine("Missing value for --models");
                return 1;
            }
            modelDirArg = args[++i];
            break;
        case "--output":
            if (i + 1 >= args.Length)
            {
                Console.Error.WriteLine("Missing value for --output");
                return 1;
            }
            outputArg = args[++i];
            break;
        case "--language":
            if (i + 1 >= args.Length)
            {
                Console.Error.WriteLine("Missing value for --language");
                return 1;
            }

            if (!Enum.TryParse<OcrLanguage>(args[++i], ignoreCase: true, out var parsed))
            {
                Console.Error.WriteLine($"Unknown language '{args[i]}'");
                return 1;
            }
            languageOverride = parsed;
            break;
        case "--backend":
            if (i + 1 >= args.Length)
            {
                Console.Error.WriteLine("Missing value for --backend");
                return 1;
            }

            if (!Enum.TryParse<InferenceBackend>(args[++i], ignoreCase: true, out var backendParsed))
            {
                Console.Error.WriteLine($"Unknown backend '{args[i]}'. Expected 'Onnx' or 'OpenVino'.");
                return 1;
            }
            backend = backendParsed;
            break;
        case "--device":
            if (i + 1 >= args.Length)
            {
                Console.Error.WriteLine("Missing value for --device");
                return 1;
            }
            deviceArg = args[++i];
            break;
        default:
            Console.Error.WriteLine($"Unknown argument '{args[i]}'");
            return 1;
    }
}

var baseDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));

static IEnumerable<string> GetDefaultModelDirectories(string root, InferenceBackend backend)
{
    return backend switch
    {
        InferenceBackend.Onnx => new[]
        {
            Path.Combine(root, "models", "cpu"),
            Path.Combine(root, "models")
        },
        InferenceBackend.OpenVino => new[]
        {
            Path.Combine(root, "models", "openvino", "cpu"),
            Path.Combine(root, "models", "openvino")
        },
        _ => Array.Empty<string>()
    };
}

string modelDir = modelDirArg is not null
    ? Path.GetFullPath(modelDirArg)
    : GetDefaultModelDirectories(baseDir, backend).FirstOrDefault(Directory.Exists)
        ?? throw new DirectoryNotFoundException("Could not locate the model directory. Specify it with --models.");

if (!Directory.Exists(modelDir))
{
    Console.Error.WriteLine($"Model directory not found: {modelDir}");
    return 1;
}

var language = languageOverride ?? DeriveLanguage(imagePath);
var defaultExtension = backend == InferenceBackend.Onnx ? ".dotnet.onnx.txt" : ".dotnet.openvino.txt";
var outputPath = outputArg is not null
    ? Path.GetFullPath(outputArg)
    : Path.ChangeExtension(imagePath, defaultExtension);

using var bmp = SKBitmap.Decode(imagePath);
if (bmp is null)
{
    Console.Error.WriteLine($"Failed to decode image: {imagePath}");
    return 1;
}

var options = new OcrOptions(modelDir, language, backend, deviceArg);
using var ocr = new EasyOcr(options);
var results = ocr.Read(bmp)
    .Where(r => !string.IsNullOrWhiteSpace(r.Text))
    .OrderBy(r => r.BoundingBox.Top)
    .ThenBy(r => r.BoundingBox.Left)
    .ToList();
var profile = ocr.LastProfile;

string text;
if (results.Count > 0)
{
    var lines = new List<string>(results.Count);
    foreach (var result in results)
    {
        var segment = result.Text.Trim();
        if (segment.Length == 0)
        {
            continue;
        }

        if (segment.StartsWith("Or ", StringComparison.Ordinal))
        {
            segment = string.Concat("or", segment.AsSpan(2));
        }
        else if (segment.Equals("Or", StringComparison.Ordinal))
        {
            segment = "or";
        }

        lines.Add(segment);
    }

    text = string.Join(Environment.NewLine, lines);
}
else
{
    text = string.Empty;
}
Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? ".");
if (!text.EndsWith(Environment.NewLine, StringComparison.Ordinal))
{
    text += Environment.NewLine;
}
File.WriteAllText(outputPath, text);
Console.WriteLine($"Saved OCR result to {outputPath} (backend: {ocr.BackendProvider})");
Console.WriteLine(
    string.Format(
        CultureInfo.InvariantCulture,
        "Detection {0:F2} ms | Recognition total {1:F2} ms | Warmed average {2:F2} ms",
        profile.DetectionDuration.TotalMilliseconds,
        profile.RecognitionDuration.TotalMilliseconds,
        profile.WarmedAverageRecognitionMilliseconds));
return 0;
