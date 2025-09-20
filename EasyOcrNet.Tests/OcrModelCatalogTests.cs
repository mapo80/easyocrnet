using EasyOcrNet.Configuration;
using EasyOcrNet.Languages;
using System;
using System.IO;
using Xunit;

namespace EasyOcrNet.Tests;

public class OcrModelCatalogTests
{
    [Fact]
    public void ResolveResourcesUsesLegacyOnnxNamesWhenPresent()
    {
        var tempDir = Directory.CreateDirectory(Path.Combine(Path.GetTempPath(), "easyocr-models-" + Guid.NewGuid().ToString("N")));
        try
        {
            var detector = Path.Combine(tempDir.FullName, "EasyOCRDetector.onnx");
            var recognizer = Path.Combine(tempDir.FullName, "EasyOCRRecognizer.onnx");
            File.WriteAllText(detector, string.Empty);
            File.WriteAllText(recognizer, string.Empty);

            var options = new OcrOptions(tempDir.FullName, OcrLanguage.English, InferenceBackend.Onnx);
            var resources = OcrModelCatalog.ResolveResources(options);

            Assert.EndsWith("EasyOCRDetector.onnx", resources.DetectionPath);
            Assert.EndsWith("EasyOCRRecognizer.onnx", resources.RecognitionPath);
            Assert.False(string.IsNullOrEmpty(resources.Characters));
        }
        finally
        {
            tempDir.Delete(recursive: true);
        }
    }

    [Fact]
    public void ResolveResourcesThrowsWhenRecognizerMissing()
    {
        var tempDir = Directory.CreateDirectory(Path.Combine(Path.GetTempPath(), "easyocr-models-" + Guid.NewGuid().ToString("N")));
        try
        {
            var detector = Path.Combine(tempDir.FullName, "detection.onnx");
            File.WriteAllText(detector, string.Empty);

            var options = new OcrOptions(tempDir.FullName, OcrLanguage.English, InferenceBackend.Onnx);
            var exception = Assert.Throws<FileNotFoundException>(() => OcrModelCatalog.ResolveResources(options));
            Assert.Contains("Recognizer", exception.Message);
        }
        finally
        {
            tempDir.Delete(recursive: true);
        }
    }
}
