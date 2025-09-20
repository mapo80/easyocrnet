using EasyOcrNet;
using EasyOcrNet.Recognition;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using Xunit;

namespace EasyOcrNet.Tests;

public class TextRecognizerTests
{
    private sealed class StubBackend : IOcrBackend
    {
        public ModelOutput RecognizerOutput { get; set; }

        public string Provider => "stub";

        public void Dispose()
        {
        }

        public ModelOutput RunDetector(DenseTensor<float> input)
        {
            throw new NotSupportedException();
        }

        public ModelOutput RunRecognizer(DenseTensor<float> input)
        {
            return RecognizerOutput;
        }
    }

    [Fact]
    public void RecognizeDecodesSequencesUsingDecoder()
    {
        var backend = new StubBackend();
        var characters = "ABC";
        var recognizer = new TextRecognizer(backend, characters);

        var data = new float[1 * 3 * 4];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = float.NegativeInfinity;
        }

        data[1] = 1.0f;       // timestep 0 -> 'A'
        data[4] = 1.0f;       // timestep 1 -> blank (index 0)
        data[8 + 2] = 1.0f;   // timestep 2 -> 'B'

        backend.RecognizerOutput = new ModelOutput(data, new[] { 1, 3, 4 });

        using var bitmap = new SKBitmap(64, 32);
        using (var canvas = new SKCanvas(bitmap))
        {
            canvas.Clear(SKColors.White);
            using var paint = new SKPaint { Color = SKColors.Black };
            canvas.DrawRect(new SKRect(5, 5, 30, 20), paint);
        }

        var result = recognizer.Recognize(bitmap, new SKRect(0, 0, bitmap.Width, bitmap.Height));
        Assert.Equal("AB", result);
    }

    [Fact]
    public void RecognizeThrowsWhenImageIsNull()
    {
        var backend = new StubBackend { RecognizerOutput = new ModelOutput(Array.Empty<float>(), new[] { 0, 0, 0 }) };
        var recognizer = new TextRecognizer(backend, "A");
        Assert.Throws<ArgumentNullException>(() => recognizer.Recognize(null!, new SKRect()));
    }
}
