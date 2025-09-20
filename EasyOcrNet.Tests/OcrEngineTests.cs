using EasyOcrNet;
using EasyOcrNet.Configuration;
using EasyOcrNet.Detection;
using EasyOcrNet.Languages;
using EasyOcrNet.Recognition;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.IO;
using Xunit;

namespace EasyOcrNet.Tests;

public class OcrEngineTests
{
    [Fact]
    public void OptionsRequireModelDirectory()
    {
        var exception = Assert.Throws<ArgumentException>(() => new OcrOptions(""));
        Assert.Equal("modelDirectory", exception.ParamName);
    }

    [Fact]
    public void LanguageCatalogReturnsLatinSetForFrench()
    {
        var metadata = OcrLanguageCatalog.GetMetadata(OcrLanguage.French);
        Assert.Equal("latin_g2_rec", metadata.RecognizerModelKey);
        Assert.Equal("latin_g2", metadata.CharacterSetKey);
    }

    [Fact]
    public void ModelCatalogResolvesOpenVinoResources()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "easyocr-openvino-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);

        try
        {
            var detectorXml = Path.Combine(tempDir, "detection.xml");
            var detectorBin = Path.Combine(tempDir, "detection.bin");
            File.WriteAllText(detectorXml, "<net></net>");
            File.WriteAllBytes(detectorBin, new byte[16]);

            var recognizerXml = Path.Combine(tempDir, "english_g2_rec.xml");
            var recognizerBin = Path.Combine(tempDir, "english_g2_rec.bin");
            File.WriteAllText(recognizerXml, "<net></net>");
            File.WriteAllBytes(recognizerBin, new byte[16]);

            var options = new OcrOptions(tempDir, OcrLanguage.English, InferenceBackend.OpenVino);
            var resources = OcrModelCatalog.ResolveResources(options);

            Assert.EndsWith("detection.xml", resources.DetectionPath);
            Assert.EndsWith("english_g2_rec.xml", resources.RecognitionPath);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void TextComponentGrouperClustersByBaseline()
    {
        var components = new List<SKRect>
        {
            new(10, 10, 100, 40),
            new(110, 12, 200, 42),
            new(15, 120, 90, 150),
            new(100, 122, 190, 152)
        };

        var lines = TextComponentGrouper.GroupIntoLines(components);
        Assert.Equal(2, lines.Count);
        Assert.True(lines[0].Bottom <= lines[1].Top);
    }

    [Fact]
    public void RecognitionInputBuilderProducesNormalizedTensor()
    {
        using var bitmap = new SKBitmap(50, 20);
        using (var canvas = new SKCanvas(bitmap))
        {
            canvas.Clear(SKColors.White);
            using var font = new SKFont { Size = 14f };
            using var paint = new SKPaint { Color = SKColors.Black, IsAntialias = true };
            canvas.DrawText("AB", 5, 15, SKTextAlign.Left, font, paint);
        }

        var tensor = RecognitionInputBuilder.Build(bitmap, new SKRect(0, 0, bitmap.Width, bitmap.Height));
        Assert.Equal(new[] { 1, 1, OcrConstants.RecognizerInputHeight, OcrConstants.RecognizerMaxWidth }, tensor.Dimensions.ToArray());
        Assert.InRange(tensor[0, 0, 0, 0], -2f, 2f);
    }

    [Fact]
    public void EasyOcrMatchesTorchfreeOutputForEnglishExample()
    {
        var dotnetPath = Path.Combine(TestPaths.ExamplesDirectory, "english.dotnet.onnx.txt");
        var torchPath = Path.Combine(TestPaths.ExamplesDirectory, "english.torchonnx.txt");

        Assert.True(File.Exists(dotnetPath));
        Assert.True(File.Exists(torchPath));

        var dotnet = File.ReadAllText(dotnetPath);
        var torch = File.ReadAllText(torchPath);

        Assert.Equal(torch, dotnet);
    }

    [Fact]
    public void EasyOcrConstructedFromDirectoryUsesDefaults()
    {
        using var engine = new EasyOcr(TestPaths.ModelsDirectory);
        Assert.Equal(Path.GetFullPath(TestPaths.ModelsDirectory), engine.ModelDirectory);
        Assert.Equal(OcrLanguage.English, engine.Language);
    }

    [Fact]
    public void EasyOcrThrowsWhenBitmapIsNull()
    {
        using var engine = new EasyOcr(new OcrOptions(TestPaths.ModelsDirectory));
        Assert.Throws<ArgumentNullException>(() => engine.Read(null!));
    }

    [Fact]
    public void EngineProducesExecutionProfile()
    {
        using var bitmap = SKBitmap.Decode(Path.Combine(TestPaths.ExamplesDirectory, "english.png"));
        Assert.NotNull(bitmap);

        var options = new OcrOptions(TestPaths.ModelsDirectory, OcrLanguage.English, InferenceBackend.Onnx);
        using var engine = new EasyOcr(options);

        var results = engine.Read(bitmap!);
        var profile = engine.LastProfile;

        Assert.True(profile.TotalDuration >= profile.DetectionDuration);
        Assert.Equal(engine.BackendProvider, profile.Provider);
        Assert.Equal(results.Count, profile.RecognitionDurations.Count);

        if (results.Count > 1)
        {
            Assert.True(profile.WarmedAverageRecognitionMilliseconds >= 0d);
        }
    }
}
