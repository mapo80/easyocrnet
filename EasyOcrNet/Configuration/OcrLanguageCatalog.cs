using EasyOcrNet.Languages;
using EasyOcrNet.Resources;

namespace EasyOcrNet.Configuration;

internal static class OcrLanguageCatalog
{
    private const string EnglishModelKey = "english_g2_rec";
    private const string LatinModelKey = "latin_g2_rec";

    private static readonly HashSet<OcrLanguage> LatinLanguages = new()
    {
        OcrLanguage.Afrikaans,
        OcrLanguage.Albanian,
        OcrLanguage.Basque,
        OcrLanguage.Catalan,
        OcrLanguage.Croatian,
        OcrLanguage.Czech,
        OcrLanguage.Danish,
        OcrLanguage.Dutch,
        OcrLanguage.Estonian,
        OcrLanguage.Filipino,
        OcrLanguage.Finnish,
        OcrLanguage.French,
        OcrLanguage.Galician,
        OcrLanguage.German,
        OcrLanguage.Hungarian,
        OcrLanguage.Icelandic,
        OcrLanguage.Indonesian,
        OcrLanguage.Irish,
        OcrLanguage.Italian,
        OcrLanguage.Kurdish,
        OcrLanguage.Latin,
        OcrLanguage.Latvian,
        OcrLanguage.Lithuanian,
        OcrLanguage.Maori,
        OcrLanguage.Malay,
        OcrLanguage.Maltese,
        OcrLanguage.Norwegian,
        OcrLanguage.Polish,
        OcrLanguage.Portuguese,
        OcrLanguage.Romanian,
        OcrLanguage.SerbianLatin,
        OcrLanguage.Slovak,
        OcrLanguage.Slovenian,
        OcrLanguage.Spanish,
        OcrLanguage.Swahili,
        OcrLanguage.Swedish,
        OcrLanguage.Turkish,
        OcrLanguage.Uzbek,
        OcrLanguage.Vietnamese,
    };

    private static readonly IReadOnlyDictionary<OcrLanguage, string> ModelOverrides = new Dictionary<OcrLanguage, string>
    {
        [OcrLanguage.English] = EnglishModelKey,
        [OcrLanguage.SimplifiedChinese] = "zh_sim_g2_rec",
        [OcrLanguage.Japanese] = "japanese_g2_rec",
        [OcrLanguage.Korean] = "korean_g2_rec",
        [OcrLanguage.Thai] = "thai_g1_rec",
    };

    public static OcrLanguageMetadata GetMetadata(OcrLanguage language)
    {
        if (!ModelOverrides.TryGetValue(language, out var modelKey))
        {
            modelKey = LatinLanguages.Contains(language) ? LatinModelKey : EnglishModelKey;
        }

        var characterKey = language switch
        {
            OcrLanguage.English => "english_g2",
            OcrLanguage.SimplifiedChinese => "ch_sim",
            OcrLanguage.Japanese => "ja",
            OcrLanguage.Korean => "ko",
            OcrLanguage.Thai => "th",
            _ => LatinLanguages.Contains(language) ? "latin_g2" : "english_g2",
        };

        return new OcrLanguageMetadata(language, modelKey, characterKey);
    }
}

internal sealed record OcrLanguageMetadata(OcrLanguage Language, string RecognizerModelKey, string CharacterSetKey);
