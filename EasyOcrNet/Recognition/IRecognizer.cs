using EasyOcrNet.Models;
using SkiaSharp;

namespace EasyOcrNet.Recognition;

/// <summary>
/// Interface for text recognition using SkiaSharp
/// </summary>
public interface IRecognizer : IDisposable
{
    /// <summary>
    /// Recognize text in detected region
    /// </summary>
    /// <param name="bitmap">Full image as SKBitmap</param>
    /// <param name="detection">Detected region</param>
    /// <returns>Recognition result with text and confidence</returns>
    Task<RecognitionResult> RecognizeAsync(SKBitmap bitmap, DetectionResult detection);

    /// <summary>
    /// Recognize text from cropped image region
    /// </summary>
    /// <param name="crop">Cropped image region as SKBitmap</param>
    /// <returns>Recognition result with text and confidence</returns>
    Task<RecognitionResult> RecognizeCropAsync(SKBitmap crop);
}
