namespace EasyOcrNet.Recognition;

internal sealed class SequenceDecoder
{
    private readonly string _characters;

    public SequenceDecoder(string characters)
    {
        _characters = characters ?? throw new ArgumentNullException(nameof(characters));
    }

    public string Decode(ModelOutput output)
    {
        return DecodeCore(output).Text;
    }

    public (string Text, double Confidence) DecodeWithConfidence(ModelOutput output)
    {
        return DecodeCore(output);
    }

    private (string Text, double Confidence) DecodeCore(ModelOutput output)
    {
        if (output.Rank != 3)
        {
            return (string.Empty, 0d);
        }

        int timeSteps = output[1];
        int classes = output[2];
        var data = output.Data;

        var buffer = new char[timeSteps];
        int length = 0;
        int prev = 0;
        double confidenceSum = 0d;
        int confidenceCount = 0;

        for (int t = 0; t < timeSteps; t++)
        {
            int offset = t * classes;
            int maxIndex = 0;
            float maxLogit = float.NegativeInfinity;
            for (int c = 0; c < classes; c++)
            {
                float value = data[offset + c];
                if (value > maxLogit)
                {
                    maxLogit = value;
                    maxIndex = c;
                }
            }

            double probability = 0d;
            if (!float.IsNegativeInfinity(maxLogit))
            {
                double sumExp = 0d;
                for (int c = 0; c < classes; c++)
                {
                    sumExp += Math.Exp(data[offset + c] - maxLogit);
                }

                if (sumExp > 0d)
                {
                    probability = 1d / sumExp;
                }
            }

            if (maxIndex > 0 && maxIndex != prev)
            {
                int characterIndex = maxIndex - 1;
                if ((uint)characterIndex < (uint)_characters.Length)
                {
                    buffer[length++] = _characters[characterIndex];
                    confidenceSum += probability;
                    confidenceCount++;
                }
            }

            prev = maxIndex;
        }

        var text = length == 0 ? string.Empty : new string(buffer, 0, length);
        var confidence = confidenceCount > 0 ? confidenceSum / confidenceCount : 0d;
        return (text, confidence);
    }
}
