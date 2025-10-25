namespace EasyOcrNet.PostProcessing;

/// <summary>
/// Interface for text post-processing
/// </summary>
public interface IPostProcessor
{
    /// <summary>
    /// Process recognized text to fix common OCR errors
    /// </summary>
    /// <param name="text">Raw OCR text</param>
    /// <returns>Processed text</returns>
    string Process(string text);
}
