namespace EasyOcrNet.Models;

/// <summary>
/// Represents a 2D point with integer coordinates
/// </summary>
public readonly record struct Point2D(int X, int Y)
{
    public override string ToString() => $"({X},{Y})";
}

/// <summary>
/// Represents a bounding box with four corner points (clockwise from top-left)
/// </summary>
public record BoundingBox(
    Point2D TopLeft,
    Point2D TopRight,
    Point2D BottomRight,
    Point2D BottomLeft)
{
    /// <summary>
    /// Convert bounding box to array of points
    /// </summary>
    public Point2D[] ToArray() => new[] { TopLeft, TopRight, BottomRight, BottomLeft };

    /// <summary>
    /// Get minimum X coordinate
    /// </summary>
    public int MinX => Math.Min(Math.Min(TopLeft.X, TopRight.X), Math.Min(BottomRight.X, BottomLeft.X));

    /// <summary>
    /// Get maximum X coordinate
    /// </summary>
    public int MaxX => Math.Max(Math.Max(TopLeft.X, TopRight.X), Math.Max(BottomRight.X, BottomLeft.X));

    /// <summary>
    /// Get minimum Y coordinate
    /// </summary>
    public int MinY => Math.Min(Math.Min(TopLeft.Y, TopRight.Y), Math.Min(BottomRight.Y, BottomLeft.Y));

    /// <summary>
    /// Get maximum Y coordinate
    /// </summary>
    public int MaxY => Math.Max(Math.Max(TopLeft.Y, TopRight.Y), Math.Max(BottomRight.Y, BottomLeft.Y));

    /// <summary>
    /// Get width of bounding box
    /// </summary>
    public int Width => MaxX - MinX;

    /// <summary>
    /// Get height of bounding box
    /// </summary>
    public int Height => MaxY - MinY;

    /// <summary>
    /// Format bounding box for output (matches Python format)
    /// </summary>
    public string ToOutputString() =>
        $"({TopLeft.X},{TopLeft.Y}) ({TopRight.X},{TopRight.Y}) ({BottomRight.X},{BottomRight.Y}) ({BottomLeft.X},{BottomLeft.Y})";
}

/// <summary>
/// Result from text detection
/// </summary>
public record DetectionResult(
    BoundingBox BoundingBox,
    float Confidence = 1.0f);

/// <summary>
/// Result from text recognition
/// </summary>
public record RecognitionResult(
    string Text,
    float Confidence);

/// <summary>
/// Complete OCR result combining detection and recognition
/// </summary>
public record OcrResult(
    BoundingBox BoundingBox,
    string Text,
    float Confidence)
{
    /// <summary>
    /// Format OCR result for output file (matches Python format exactly)
    /// Format: (x1,y1) (x2,y2) (x3,y3) (x4,y4) | text | confidence
    /// </summary>
    public string ToOutputString()
    {
        var cleanText = Text.Replace("\n", " ").Replace("\r", " ");
        return $"{BoundingBox.ToOutputString()} | {cleanText} | {Confidence:F4}";
    }
}

/// <summary>
/// Configuration for OCR processing
/// </summary>
public record OcrConfig(
    string Language = "en",
    bool EnableSpellCheck = false,
    int MinSize = 20,
    float SlopeThreshold = 0.1f,
    float YCenterThreshold = 0.5f,
    float HeightThreshold = 0.5f,
    float WidthThreshold = 0.5f,
    float AddMargin = 0.1f,
    float TextThreshold = 0.7f,
    float LinkThreshold = 0.4f,
    float LowText = 0.4f,
    float ContrastThreshold = 0.1f,
    float AdjustContrast = 0.5f);
