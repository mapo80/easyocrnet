using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace EasyOcrNet.Resources;

internal static class CharacterSetCatalog
{
    private static readonly ConcurrentDictionary<string, string> Cache = new(StringComparer.OrdinalIgnoreCase);
    private static readonly IReadOnlyDictionary<string, string> BuiltIn = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
    {
        ["english_g2"] = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        ["latin_g2"] = PredefinedCharacterSets.Latin,
        ["ch_sim"] = PredefinedCharacterSets.ChineseSimplified,
        ["ja"] = PredefinedCharacterSets.Japanese,
        ["ko"] = PredefinedCharacterSets.Korean,
        ["th"] = PredefinedCharacterSets.Thai,
    };
    private static readonly Lazy<string> CharacterRoot = new(() =>
    {
        var baseDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));
        var candidate = Path.Combine(baseDir, "character");
        if (!Directory.Exists(candidate))
        {
            throw new DirectoryNotFoundException($"Character directory not found at '{candidate}'.");
        }

        return candidate;
    });

    public static string GetCharacters(string key)
    {
        if (string.IsNullOrWhiteSpace(key))
        {
            throw new ArgumentException("Character set key must be provided", nameof(key));
        }

        return Cache.GetOrAdd(key, LoadCharacters);
    }

    private static string LoadCharacters(string key)
    {
        if (BuiltIn.TryGetValue(key, out var builtIn))
        {
            return builtIn;
        }

        var fileName = key.EndsWith("_char", StringComparison.OrdinalIgnoreCase) ? key : key + "_char";
        var path = Path.Combine(CharacterRoot.Value, fileName + ".txt");
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Character set file not found for key '{key}'.", path);
        }

        var builder = new StringBuilder();
        foreach (var line in File.ReadLines(path))
        {
            if (line.Length == 0)
            {
                continue;
            }

            builder.Append(line[0]);
        }

        return builder.ToString();
    }
}
