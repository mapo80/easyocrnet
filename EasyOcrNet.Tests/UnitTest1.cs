using SkiaSharp;
using Xunit;

namespace EasyOcrNet.Tests;

public class OcrTests
{
    [Fact]
    public void RecognizesSimpleText()
    {
        using var bmp = new SKBitmap(800, 608);
        using (var canvas = new SKCanvas(bmp))
        {
            canvas.Clear(SKColors.White);
            var paint = new SKPaint
            {
                Color = SKColors.Black,
                TextSize = 200,
                IsAntialias = true
            };
            canvas.DrawText("HELLO", 50, 300, paint);
        }

        var modelDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "models"));
        using var ocr = new EasyOcr(modelDir);
        var results = ocr.Read(bmp).ToList();
        Assert.NotEmpty(results.First().Text);
    }
}
