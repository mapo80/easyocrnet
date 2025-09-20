using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using EasyOcrNet;
using EasyOcrNet.Configuration;
using EasyOcrNet.Languages;
using SkiaSharp;

const int DefaultRuns = 6;
const int DefaultDiscard = 1;

static void PrintUsage()
{
    Console.WriteLine("Usage: dotnet run --project EasyOcrNet.BenchmarkCli -- [--image <path>] [--models <dir>] [--language <name>] [--backend <onnx|openvino>] [--device <name>] [--runs <n>] [--discard <n>]");
    Console.WriteLine("Defaults: image=examples/english.png, backend=Onnx, runs=6, discard=1 (warm-up).");
}

if (args.Contains("--help", StringComparer.OrdinalIgnoreCase))
{
    PrintUsage();
    return 0;
}

string? imagePathArg = null;
string? modelDirArg = null;
OcrLanguage? languageOverride = null;
string? deviceArg = null;
InferenceBackend backend = InferenceBackend.Onnx;
int runs = DefaultRuns;
int discard = DefaultDiscard;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--image":
            if (++i >= args.Length)
            {
                Console.Error.WriteLine("Missing value for --image");
                return 1;
            }
            imagePathArg = args[i];
            break;
        case "--models":
            if (++i >= args.Length)
            {
                Console.Error.WriteLine("Missing value for --models");
                return 1;
            }
            modelDirArg = args[i];
            break;
        case "--language":
            if (++i >= args.Length)
            {
                Console.Error.WriteLine("Missing value for --language");
                return 1;
            }

            if (!Enum.TryParse<OcrLanguage>(args[i], ignoreCase: true, out var parsed))
            {
                Console.Error.WriteLine($"Unknown language '{args[i]}'");
                return 1;
            }
            languageOverride = parsed;
            break;
        case "--backend":
            if (++i >= args.Length)
            {
                Console.Error.WriteLine("Missing value for --backend");
                return 1;
            }

            if (!Enum.TryParse<InferenceBackend>(args[i], ignoreCase: true, out var backendParsed))
            {
                Console.Error.WriteLine($"Unknown backend '{args[i]}'. Expected 'Onnx' or 'OpenVino'.");
                return 1;
            }
            backend = backendParsed;
            break;
        case "--device":
            if (++i >= args.Length)
            {
                Console.Error.WriteLine("Missing value for --device");
                return 1;
            }
            deviceArg = args[i];
            break;
        case "--runs":
            if (++i >= args.Length || !int.TryParse(args[i], NumberStyles.Integer, CultureInfo.InvariantCulture, out runs) || runs <= 0)
            {
                Console.Error.WriteLine("--runs expects a positive integer");
                return 1;
            }
            break;
        case "--discard":
            if (++i >= args.Length || !int.TryParse(args[i], NumberStyles.Integer, CultureInfo.InvariantCulture, out discard) || discard < 0)
            {
                Console.Error.WriteLine("--discard expects a non-negative integer");
                return 1;
            }
            break;
        default:
            Console.Error.WriteLine($"Unknown argument '{args[i]}'");
            PrintUsage();
            return 1;
    }
}

var baseDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));
var defaultImagePath = Path.Combine(baseDir, "examples", "english.png");
var imagePath = Path.GetFullPath(imagePathArg ?? defaultImagePath);
if (!File.Exists(imagePath))
{
    Console.Error.WriteLine($"Image not found: {imagePath}");
    return 1;
}

static IEnumerable<string> GetDefaultModelDirs(string root, InferenceBackend backend)
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
    : GetDefaultModelDirs(baseDir, backend).FirstOrDefault(Directory.Exists)
        ?? throw new DirectoryNotFoundException("Could not locate the model directory. Specify it with --models.");

if (!Directory.Exists(modelDir))
{
    Console.Error.WriteLine($"Model directory not found: {modelDir}");
    return 1;
}

if (discard >= runs)
{
    Console.Error.WriteLine("The number of discarded runs must be less than the total runs.");
    return 1;
}

var language = languageOverride ?? OcrLanguage.English;

using var bitmap = SKBitmap.Decode(imagePath);
if (bitmap is null)
{
    Console.Error.WriteLine($"Failed to decode image: {imagePath}");
    return 1;
}

var options = new OcrOptions(modelDir, language, backend, deviceArg);
using var engine = new EasyOcr(options);

Console.WriteLine($"Benchmarking {imagePath} using models in {modelDir}");
Console.WriteLine($"Language: {language}, Backend: {backend}, Device: {options.Device}");
Console.WriteLine($"Runs: {runs}, Discarded warm-up runs: {discard}");

var durations = new List<double>(Math.Max(0, runs - discard));
int segments = 0;

for (int i = 0; i < runs; i++)
{
    var sw = Stopwatch.StartNew();
    var results = engine.Read(bitmap);
    sw.Stop();

    segments = results.Count;
    var elapsedMs = sw.Elapsed.TotalMilliseconds;
    Console.WriteLine($"Run {i + 1}: {elapsedMs:F2} ms ({segments} segments)");

    if (i >= discard)
    {
        durations.Add(elapsedMs);
    }
}

if (durations.Count == 0)
{
    Console.Error.WriteLine("No runs available for averaging.");
    return 1;
}

var average = durations.Average();
var min = durations.Min();
var max = durations.Max();

Console.WriteLine();
Console.WriteLine($"Average over {durations.Count} runs (discarding first {discard}): {average:F2} ms");
Console.WriteLine($"Min: {min:F2} ms | Max: {max:F2} ms");
Console.WriteLine($"Detected segments per run: {segments}");

return 0;
