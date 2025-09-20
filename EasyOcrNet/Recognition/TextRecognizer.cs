using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace EasyOcrNet.Recognition;

internal sealed class TextRecognizer
{
    private readonly IOcrBackend _backend;
    private readonly SequenceDecoder _decoder;

    public TextRecognizer(IOcrBackend backend, string characters)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _decoder = new SequenceDecoder(characters);
    }

    public string Recognize(SKBitmap image, SKRect region)
    {
        if (image is null)
        {
            throw new ArgumentNullException(nameof(image));
        }

        DenseTensor<float> input = RecognitionInputBuilder.Build(image, region);
        ModelOutput output = _backend.RunRecognizer(input);
        return _decoder.Decode(output);
    }
}
