using EasyOcrNet.Configuration;
using EasyOcrNet.Detection;
using EasyOcrNet.Languages;
using EasyOcrNet.Recognition;
using SkiaSharp;
using System.Diagnostics;

namespace EasyOcrNet;

public sealed class EasyOcr : IDisposable
{
    private readonly IOcrBackend _backend;
    private readonly TextDetector _detector;
    private readonly TextRecognizer _recognizer;

    public EasyOcr(string modelDirectory)
        : this(new OcrOptions(modelDirectory))
    {
    }

    public EasyOcr(string modelDirectory, OcrLanguage language, InferenceBackend backend = InferenceBackend.Onnx, string? device = null)
        : this(new OcrOptions(modelDirectory, language, backend, device))
    {
    }

    public EasyOcr(OcrOptions options)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        Language = options.Language;
        ModelDirectory = options.ModelDirectory;

        var resources = OcrModelCatalog.ResolveResources(options);
        _backend = OcrBackendFactory.Create(options, resources);
        _detector = new TextDetector(_backend);
        _recognizer = new TextRecognizer(_backend, resources.Characters);
    }

    public string ModelDirectory { get; }

    public OcrLanguage Language { get; }

    public string BackendProvider => _backend.Provider;

    private OcrExecutionProfile _lastProfile = OcrExecutionProfile.Empty;

    public OcrExecutionProfile LastProfile => _lastProfile;

    public IReadOnlyList<OcrResult> Read(SKBitmap image)
    {
        if (image is null)
        {
            throw new ArgumentNullException(nameof(image));
        }

        var overallWatch = Stopwatch.StartNew();
        var detectionWatch = Stopwatch.StartNew();
        var detections = _detector.Detect(image);
        detectionWatch.Stop();

        var results = new List<OcrResult>(detections.Count);
        var recognitionDurations = detections.Count == 0 ? Array.Empty<double>() : new double[detections.Count];

        var recognitionWatch = Stopwatch.StartNew();
        for (int i = 0; i < detections.Count; i++)
        {
            var rect = detections[i];
            var runWatch = Stopwatch.StartNew();
            var (text, confidence) = _recognizer.Recognize(image, rect);
            runWatch.Stop();
            recognitionDurations[i] = runWatch.Elapsed.TotalMilliseconds;
            results.Add(new OcrResult(text, rect, confidence));
        }
        recognitionWatch.Stop();
        overallWatch.Stop();

        _lastProfile = OcrExecutionProfile.Create(
            detectionWatch.Elapsed,
            recognitionWatch.Elapsed,
            overallWatch.Elapsed,
            recognitionDurations,
            BackendProvider);

        return results;
    }

    public void Dispose()
    {
        _backend.Dispose();
    }
}
