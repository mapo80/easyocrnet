using EasyOcrNet.Languages;

namespace EasyOcrNet.Configuration;

/// <summary>
/// Defines the configuration for the OCR engine.
/// </summary>
public sealed class OcrOptions
{
    public OcrOptions(
        string modelDirectory,
        OcrLanguage language = OcrLanguage.English,
        InferenceBackend backend = InferenceBackend.Onnx,
        string? device = null)
    {
        if (string.IsNullOrWhiteSpace(modelDirectory))
        {
            throw new ArgumentException("Model directory must be provided", nameof(modelDirectory));
        }

        ModelDirectory = Path.GetFullPath(modelDirectory);
        Language = language;
        Backend = backend;
        Device = string.IsNullOrWhiteSpace(device) ? "CPU" : device;
    }

    public string ModelDirectory { get; }

    public OcrLanguage Language { get; }

    public InferenceBackend Backend { get; }

    public string Device { get; }
}
