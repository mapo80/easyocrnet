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
        if (output.Rank != 3)
        {
            return string.Empty;
        }

        int timeSteps = output[1];
        int classes = output[2];
        var data = output.Data;

        var buffer = new char[timeSteps];
        int length = 0;
        int prev = 0;

        for (int t = 0; t < timeSteps; t++)
        {
            int offset = t * classes;
            int maxIndex = 0;
            float maxValue = float.NegativeInfinity;
            for (int c = 0; c < classes; c++)
            {
                float value = data[offset + c];
                if (value > maxValue)
                {
                    maxValue = value;
                    maxIndex = c;
                }
            }

            if (maxIndex > 0 && maxIndex != prev)
            {
                int characterIndex = maxIndex - 1;
                if ((uint)characterIndex < (uint)_characters.Length)
                {
                    buffer[length++] = _characters[characterIndex];
                }
            }

            prev = maxIndex;
        }

        return length == 0 ? string.Empty : new string(buffer, 0, length);
    }
}
