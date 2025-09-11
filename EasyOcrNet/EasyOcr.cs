using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace EasyOcrNet;

public record OcrResult(string Text, SKRect BoundingBox);

public class EasyOcr : IDisposable
{
    private readonly InferenceSession _detector;
    private readonly InferenceSession _recognizer;
    private readonly string _charset;

    private static readonly HashSet<Charset> LatinCharsets = new()
    {
        Charset.af, Charset.az, Charset.bs, Charset.cs, Charset.cy, Charset.da,
        Charset.de, Charset.en, Charset.es, Charset.et, Charset.fr, Charset.ga,
        Charset.hr, Charset.hu, Charset.id, Charset.@is, Charset.it, Charset.ku,
        Charset.la, Charset.lt, Charset.lv, Charset.mi, Charset.ms, Charset.mt,
        Charset.nl, Charset.no, Charset.oc, Charset.pi, Charset.pl, Charset.pt,
        Charset.ro, Charset.rs_latin, Charset.sk, Charset.sl, Charset.sq,
        Charset.sv, Charset.sw, Charset.tl, Charset.tr, Charset.uz, Charset.vi
    };

    public EasyOcr(string modelDirectory, Charset charset = Charset.en)
    {
        _detector = new InferenceSession(Path.Combine(modelDirectory, "EasyOCRDetector.onnx"));
        _recognizer = new InferenceSession(Path.Combine(modelDirectory, "EasyOCRRecognizer.onnx"));
        _charset = LoadCharset(modelDirectory, charset);
    }

    public IEnumerable<OcrResult> Read(SKBitmap image)
    {
        using var resized = new SKBitmap(800, 608);
        image.ScalePixels(resized, SKFilterQuality.Medium);

        // Prepare detector input [1,3,608,800]
        var detTensor = new DenseTensor<float>(new[] { 1, 3, 608, 800 });
        var mean = new float[] { 0.485f * 255f, 0.456f * 255f, 0.406f * 255f };
        var std = new float[] { 0.229f * 255f, 0.224f * 255f, 0.225f * 255f };
        for (int y = 0; y < 608; y++)
        {
            for (int x = 0; x < 800; x++)
            {
                var c = resized.GetPixel(x, y);
                detTensor[0, 0, y, x] = ((float)c.Red - mean[0]) / std[0];
                detTensor[0, 1, y, x] = ((float)c.Green - mean[1]) / std[1];
                detTensor[0, 2, y, x] = ((float)c.Blue - mean[2]) / std[2];
            }
        }
        using var detResults = _detector.Run(new[] { NamedOnnxValue.CreateFromTensor("image", detTensor) });
        var bbox = GetBboxFromDetector(detResults, resized.Width, resized.Height);
        var text = Recognize(resized, bbox);
        return new[] { new OcrResult(text, bbox) };
    }

    private string Recognize(SKBitmap image, SKRect rect)
    {
        int x = (int)rect.Left;
        int y = (int)rect.Top;
        int w = (int)rect.Width;
        int h = (int)rect.Height;

        using var cropped = new SKBitmap(w, h);
        using (var canvas = new SKCanvas(cropped))
        {
            canvas.DrawBitmap(image, new SKRect(x, y, x + w, y + h), new SKRect(0, 0, w, h));
        }

        // Convert to grayscale
        var gray = new SKBitmap(w, h, SKColorType.Gray8, SKAlphaType.Opaque);
        var pixels = cropped.Pixels;
        var grayPix = gray.Pixels;
        for (int i = 0; i < pixels.Length; i++)
        {
            var p = pixels[i];
            byte g = (byte)(0.299 * p.Red + 0.587 * p.Green + 0.114 * p.Blue);
            grayPix[i] = new SKColor(g, g, g);
        }

        // Resize keeping aspect ratio to height 64
        float scale = 64f / h;
        int newW = Math.Min(1000, (int)Math.Ceiling(w * scale));
        using var resized = new SKBitmap(newW, 64, SKColorType.Gray8, SKAlphaType.Opaque);
        gray.ScalePixels(resized, SKFilterQuality.Medium);

        // Prepare recognizer input [1,1,64,1000]
        var tensor = new DenseTensor<float>(new[] { 1, 1, 64, 1000 });
        for (int row = 0; row < 64; row++)
        {
            for (int col = 0; col < 1000; col++)
            {
                byte val;
                if (col < newW)
                    val = resized.GetPixel(col, row).Red;
                else
                    val = resized.GetPixel(newW - 1, row).Red;
                float norm = (val / 255f - 0.5f) / 0.5f;
                tensor[0, 0, row, col] = norm;
            }
        }

        using var results = _recognizer.Run(new[] { NamedOnnxValue.CreateFromTensor("image", tensor) });
        var output = results.First().AsTensor<float>();
        int T = output.Dimensions[1];
        int C = output.Dimensions[2];
        var sb = new StringBuilder();
        int prev = 0;
        for (int t = 0; t < T; t++)
        {
            int maxIdx = 0;
            float maxVal = float.NegativeInfinity;
            for (int c = 0; c < C; c++)
            {
                float v = output[0, t, c];
                if (v > maxVal)
                {
                    maxVal = v;
                    maxIdx = c;
                }
            }
            if (maxIdx > 0 && maxIdx != prev)
            {
                int charIndex = maxIdx - 1;
                if (charIndex >= 0 && charIndex < _charset.Length)
                    sb.Append(_charset[charIndex]);
            }
            prev = maxIdx;
        }
        return sb.ToString();
    }

    private static string LoadCharset(string modelDirectory, Charset charset)
    {
        var rootDir = Path.GetFullPath(Path.Combine(modelDirectory, ".."));
        var charDir = Path.Combine(rootDir, "character");
        if (LatinCharsets.Contains(charset))
        {
            const string latinChars =
                "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ " +
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" +
                "ÀÁÂÃÄÅÆÇÈÉÊËÍÎÑÒÓÔÕÖØÚÛÜÝÞß" +
                "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿąęĮįıŁłŒœŠšųŽž\n";
            return latinChars;
        }

        var file = Path.Combine(charDir, $"{charset}_char.txt");
        if (!File.Exists(file))
            file = Path.Combine(charDir, $"{charset}.txt");

        // EasyOCR augments language-specific characters with digits, punctuation,
        // spaces, and newlines. Reproduce that behaviour so output indices align
        // with the original Python implementation.
        var lang = File.ReadAllText(file, Encoding.UTF8)
            .Replace("\r", string.Empty)
            .Replace("\n", string.Empty);
        const string symbols = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \n";
        return symbols + lang;
    }

    private SKRect GetContentBox(SKBitmap img)
    {
        int minX = img.Width, minY = img.Height, maxX = 0, maxY = 0;
        bool found = false;
        for (int y = 0; y < img.Height; y++)
        {
            for (int x = 0; x < img.Width; x++)
            {
                var c = img.GetPixel(x, y);
                if (c.Red < 250 || c.Green < 250 || c.Blue < 250)
                {
                    found = true;
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                }
            }
        }
        return found ? new SKRect(minX, minY, maxX, maxY) : new SKRect(0, 0, img.Width, img.Height);
    }

    private SKRect GetBboxFromDetector(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> detResults, int width, int height)
    {
        var tensor = detResults.First(v => v.Name == "results").AsTensor<float>();
        int detH = tensor.Dimensions[1];
        int detW = tensor.Dimensions[2];
        float scaleX = (float)width / detW;
        float scaleY = (float)height / detH;
        int minX = detW, minY = detH, maxX = 0, maxY = 0;
        bool found = false;
        for (int y = 0; y < detH; y++)
        {
            for (int x = 0; x < detW; x++)
            {
                float p = tensor[0, y, x, 0];
                if (p > 0.3f)
                {
                    found = true;
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                }
            }
        }
        return found
            ? new SKRect(minX * scaleX, minY * scaleY, (maxX + 1) * scaleX, (maxY + 1) * scaleY)
            : new SKRect(0, 0, width, height);
    }

    public void Dispose()
    {
        _detector.Dispose();
        _recognizer.Dispose();
    }
}
