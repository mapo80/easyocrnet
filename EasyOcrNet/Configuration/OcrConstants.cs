namespace EasyOcrNet.Configuration;

internal static class OcrConstants
{
    public const int DetectorInputWidth = 800;
    public const int DetectorInputHeight = 608;
    public const int RecognizerInputHeight = 64;
    public const int RecognizerMaxWidth = 1000;
    public const int RecognitionPadding = 2;

    public const float DetectorTextScoreThreshold = 0.7f;
    public const float DetectorTextLinkThreshold = 0.4f;
    public const float DetectorLowTextThreshold = 0.4f;
}
