using EasyOcrNet.Resources;

namespace EasyOcrNet.Configuration;

internal static class OcrModelCatalog
{
    public static OcrModelResources ResolveResources(OcrOptions options)
    {
        var metadata = OcrLanguageCatalog.GetMetadata(options.Language);

        var detectionFile = options.Backend switch
        {
            InferenceBackend.Onnx => "detection.onnx",
            InferenceBackend.OpenVino => "detection.xml",
            _ => throw new ArgumentOutOfRangeException(nameof(options.Backend), options.Backend, null),
        };

        var recognizerExtension = options.Backend switch
        {
            InferenceBackend.Onnx => ".onnx",
            InferenceBackend.OpenVino => ".xml",
            _ => throw new ArgumentOutOfRangeException(nameof(options.Backend), options.Backend, null),
        };

        var detectionPath = Path.Combine(options.ModelDirectory, detectionFile);
        if (!File.Exists(detectionPath) && options.Backend == InferenceBackend.Onnx)
        {
            var legacy = Path.Combine(options.ModelDirectory, "EasyOCRDetector.onnx");
            if (File.Exists(legacy))
            {
                detectionPath = legacy;
            }
        }

        if (!File.Exists(detectionPath))
        {
            throw new FileNotFoundException($"Detection model not found at '{detectionPath}'.", detectionPath);
        }

        var characterKey = metadata.CharacterSetKey;
        var recognizerPath = Path.Combine(options.ModelDirectory, metadata.RecognizerModelKey + recognizerExtension);
        if (!File.Exists(recognizerPath) && options.Backend == InferenceBackend.Onnx)
        {
            var legacy = Path.Combine(options.ModelDirectory, "EasyOCRRecognizer.onnx");
            if (File.Exists(legacy))
            {
                recognizerPath = legacy;
                characterKey = "english_g2";
            }
        }

        if (!File.Exists(recognizerPath))
        {
            throw new FileNotFoundException($"Recognizer model not found for language '{options.Language}'. Expected at '{recognizerPath}'.", recognizerPath);
        }

        var characters = CharacterSetCatalog.GetCharacters(characterKey);
        return new OcrModelResources(detectionPath, recognizerPath, characters);
    }
}

internal sealed record OcrModelResources(string DetectionPath, string RecognitionPath, string Characters);
