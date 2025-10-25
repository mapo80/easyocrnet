using System.Text;

namespace EasyOcrNet.Utils;

/// <summary>
/// Loads character sets for different languages
/// Matches Python load_charset() function exactly
/// </summary>
public static class CharsetLoader
{
    /// <summary>
    /// English Gen2 charset (hardcoded to match torchfree-ocr exactly)
    /// </summary>
    private const string EnglishCharset = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    /// <summary>
    /// Load charset for specified language
    /// </summary>
    /// <param name="language">Language code (en, it, etc.)</param>
    /// <param name="charsetDirectory">Directory containing charset files (default: "character")</param>
    /// <returns>Charset string with all characters</returns>
    /// <exception cref="FileNotFoundException">When charset file not found</exception>
    public static string Load(string language, string charsetDirectory = "character")
    {
        // English uses hardcoded charset (matches Python exactly)
        if (language.Equals("en", StringComparison.OrdinalIgnoreCase))
        {
            return EnglishCharset;
        }

        // Try to load from file
        var charsetPath = Path.Combine(charsetDirectory, $"{language}_charset.txt");

        if (!File.Exists(charsetPath))
        {
            // Fallback to old naming convention
            charsetPath = Path.Combine(charsetDirectory, $"{language}_char.txt");
        }

        if (!File.Exists(charsetPath))
        {
            throw new FileNotFoundException($"Charset file not found for language '{language}'", charsetPath);
        }

        // Read all characters (one per line) and concatenate
        // Match Python: f.read().replace('\n', '')
        // IMPORTANT: File.ReadAllText() preserves all characters including spaces
        var content = File.ReadAllText(charsetPath, Encoding.UTF8);
        var charset = content.Replace("\n", "").Replace("\r", "");

        return charset;
    }

    /// <summary>
    /// Get charset as list with [blank] token at index 0
    /// Used for CTC decoding
    /// </summary>
    /// <param name="language">Language code</param>
    /// <param name="charsetDirectory">Directory containing charset files</param>
    /// <returns>List of characters with [blank] at index 0</returns>
    public static List<string> LoadAsListWithBlank(string language, string charsetDirectory = "character")
    {
        var charset = Load(language, charsetDirectory);
        var charList = new List<string> { "[blank]" };
        charList.AddRange(charset.Select(c => c.ToString()));
        return charList;
    }

    /// <summary>
    /// Get number of characters (including blank token)
    /// </summary>
    public static int GetCharCount(string language, string charsetDirectory = "character")
    {
        return Load(language, charsetDirectory).Length + 1; // +1 for blank token
    }
}
