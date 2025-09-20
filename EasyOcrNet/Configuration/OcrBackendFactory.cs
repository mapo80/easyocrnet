namespace EasyOcrNet.Configuration;

internal static class OcrBackendFactory
{
    public static IOcrBackend Create(OcrOptions options, OcrModelResources resources)
    {
        return options.Backend switch
        {
            InferenceBackend.Onnx => new OnnxBackend(resources.DetectionPath, resources.RecognitionPath),
            InferenceBackend.OpenVino => new OpenVinoBackend(resources.DetectionPath, resources.RecognitionPath, options.Device),
            _ => throw new ArgumentOutOfRangeException(nameof(options.Backend), options.Backend, null),
        };
    }
}
