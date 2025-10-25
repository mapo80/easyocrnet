using System;
using System.Text.RegularExpressions;

namespace EasyOcrNet.Recognition;

/// <summary>
/// Post-processing for OCR text recognition output.
/// Implements same logic as Python ocr_process.py postprocess_italian_text()
/// </summary>
public static class TextPostProcessor
{
    /// <summary>
    /// Complete post-processing pipeline for Italian text.
    /// Matches Python: postprocess_italian_text()
    /// </summary>
    public static string PostProcessItalian(string text)
    {
        if (string.IsNullOrEmpty(text))
            return text;

        // Step 1: Normalize punctuation
        text = NormalizePunctuation(text);

        // Step 2: Fix Unicode character substitutions (apostrophes)
        text = FixApostrophes(text);

        // Step 3: Separate compound words with missing apostrophes
        text = SeparateCompoundWords(text);

        // Step 4: Fix contextual errors (specific word patterns)
        text = FixContextualErrors(text);

        return text;
    }

    /// <summary>
    /// Normalize punctuation and apostrophes to standard forms.
    /// Matches Python: normalize_punctuation()
    /// </summary>
    private static string NormalizePunctuation(string text)
    {
        // Normalize typographic apostrophes to ASCII
        text = text.Replace('\u2019', '\'');  // ' → '
        text = text.Replace('\u2018', '\'');  // ' → '
        text = text.Replace('`', '\'');       // ` → '

        // Normalize quotes
        text = text.Replace('\u201C', '"');   // " → "
        text = text.Replace('\u201D', '"');   // " → "

        // Normalize dashes
        text = text.Replace('\u2013', '-');   // – → -
        text = text.Replace('\u2014', '-');   // — → -

        return text;
    }

    /// <summary>
    /// Fix Unicode character substitutions for apostrophes.
    /// Matches Python: Fix ľ (U+013E) → apostrophe logic
    /// </summary>
    private static string FixApostrophes(string text)
    {
        // Fix ľ (U+013E - L with caron) → apostrophe
        // This is a common OCR error where apostrophe is recognized as ľ
        // NOTE: Python uses [aeiou] without re.IGNORECASE, so uppercase letters are NOT matched
        // This is intentional - all-uppercase text with ľ is not processed

        // Fix 2a: ľ after 'l' (double consonant) → ll'
        // "delľ" → "dell'", "dalľ" → "dall'", "alľ" → "all'"
        text = Regex.Replace(text, @"([aeiou])lľ", "$1ll'");

        // Fix 2b: ľ after vowel (single consonant) → l'
        // "aľ" → "al'"
        text = Regex.Replace(text, @"([aeiou])ľ", "$1l'");

        // Fix 2c: ľ after other consonants → '
        text = Regex.Replace(text, @"([bcdfgmnpqrstvz])ľ", "$1'");

        // Fix 2d: ľ at start of word (after space/punctuation) → "l'"
        // " ľunico" → " l'unico", " ľora" → " l'ora"
        // NOTE: Python uses [a-z] without IGNORECASE, so it only matches lowercase
        text = Regex.Replace(text, @"(\s)ľ([a-z])", "$1l'$2");

        // Fix 2e: ľ at start of string → "l'"
        // NOTE: Python uses [a-z] without IGNORECASE, so it only matches lowercase
        text = Regex.Replace(text, @"^ľ([a-z])", "l'$1");

        // Fix 3: Other character substitutions
        text = text.Replace("iĪ", "il");  // iĪ → il (I with macron)
        text = text.Replace("Ī", "l");    // Ī → l
        text = text.Replace("Ē", "È");    // Ē → È (E with macron → E with grave)

        return text;
    }

    /// <summary>
    /// Separate common compound words in Italian where apostrophes were missed.
    /// Matches Python: separate_compound_words()
    /// </summary>
    private static string SeparateCompoundWords(string text)
    {
        // Common patterns where apostrophe is missing
        // Article + noun contractions (double consonant)
        text = Regex.Replace(text, @"\bdallu", "dall'u", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bdellu", "dell'u", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bnellu", "nell'u", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bsullu", "sull'u", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\ballU", "all'U", RegexOptions.IgnoreCase);

        // Article + noun contractions (single consonant - OCR error)
        text = Regex.Replace(text, @"\bdalu", "dall'u", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bdelu", "dell'u", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bnelu", "nell'u", RegexOptions.IgnoreCase);

        // Verbs with missing apostrophe
        text = Regex.Replace(text, @"\bcè\b", "c'è", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bcera\b", "c'era", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bcerano\b", "c'erano", RegexOptions.IgnoreCase);

        // Pronoun contractions
        text = Regex.Replace(text, @"\bcenè\b", "ce n'è", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bcene\b", "ce ne", RegexOptions.IgnoreCase);

        // Common expressions
        text = Regex.Replace(text, @"\bunarm", "un'arm", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bunaltr", "un'altr", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bunult", "un'ult", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bunennesimo\b", "un'ennesimo", RegexOptions.IgnoreCase);

        // Verb + pronoun
        text = Regex.Replace(text, @"\bcomè\b", "com'è", RegexOptions.IgnoreCase);

        return text;
    }

    /// <summary>
    /// Fix common contextual OCR errors specific to Italian text.
    /// Matches Python: fix_contextual_errors()
    /// </summary>
    private static string FixContextualErrors(string text)
    {
        // Dictionary of common OCR mistakes and their corrections
        text = Regex.Replace(text, @"\bprogerto\b", "progetto", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bdallalto\b", "dall'alto", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bbensi\b", "bensì", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bcaffe\b", "caffè", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bcitta\b", "città", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bpiu\b", "più", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bgia\b", "già", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bperche\b", "perché", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"\bcosi\b", "così", RegexOptions.IgnoreCase);

        return text;
    }
}
