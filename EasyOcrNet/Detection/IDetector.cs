using EasyOcrNet.Models;
using SkiaSharp;

namespace EasyOcrNet.Detection;

/// <summary>
/// Interface for text detection in images using SkiaSharp
/// </summary>
public interface IDetector : IDisposable
{
    /// <summary>
    /// Detect text regions in image
    /// </summary>
    /// <param name="bitmap">Input image as SKBitmap</param>
    /// <returns>List of detected text regions with confidence</returns>
    Task<List<DetectionResult>> DetectAsync(SKBitmap bitmap);
}
