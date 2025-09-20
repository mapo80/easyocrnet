using EasyOcrNet;
using EasyOcrNet.Configuration;
using EasyOcrNet.Languages;
using System;
using System.IO;
using Xunit;

namespace EasyOcrNet.Tests;

public class BackendFactoryTests
{
    [Fact]
    public void CreateThrowsForUnknownBackend()
    {
        var options = new OcrOptions(TestPaths.ModelsDirectory, OcrLanguage.English, (InferenceBackend)123);
        var resources = new OcrModelResources("det", "rec", "abc");

        var exception = Assert.Throws<ArgumentOutOfRangeException>(() => OcrBackendFactory.Create(options, resources));
        Assert.Equal("Backend", exception.ParamName);
    }

    [Fact]
    public void CreateOpenVinoBackendThrowsWhenResourcesMissing()
    {
        var options = new OcrOptions(TestPaths.ModelsDirectory, OcrLanguage.English, InferenceBackend.OpenVino);
        var resources = new OcrModelResources("missing.xml", "missing.xml", "abc");

        Assert.Throws<FileNotFoundException>(() => OcrBackendFactory.Create(options, resources));
    }

    [Fact]
    public void OnnxBackendThrowsWhenDetectorMissing()
    {
        var exception = Assert.Throws<FileNotFoundException>(() => new OnnxBackend("missing-detector.onnx", "recognizer.onnx"));
        Assert.Contains("Detector", exception.Message);
    }

    [Fact]
    public void OnnxBackendThrowsWhenRecognizerMissing()
    {
        var resources = OcrModelCatalog.ResolveResources(new OcrOptions(TestPaths.ModelsDirectory));
        var missingRecognizer = Path.Combine(TestPaths.ModelsDirectory, "__missing__.onnx");

        var exception = Assert.Throws<FileNotFoundException>(() => new OnnxBackend(resources.DetectionPath, missingRecognizer));
        Assert.Contains("Recognizer", exception.Message);
    }
}
