using EasyOcrNet.Resources;
using System;
using System.IO;
using Xunit;

namespace EasyOcrNet.Tests;

public class CharacterSetCatalogTests
{
    [Fact]
    public void ReturnsBuiltInCharacterSetFromCache()
    {
        var first = CharacterSetCatalog.GetCharacters("latin_g2");
        var second = CharacterSetCatalog.GetCharacters("LATIN_G2");

        Assert.False(string.IsNullOrEmpty(first));
        Assert.Same(first, second);
    }

    [Fact]
    public void LoadsCharactersFromFileWhenNotBuiltIn()
    {
        var key = "temp_" + Guid.NewGuid().ToString("N");
        var filePath = Path.Combine(TestPaths.SolutionRoot, "character", key + "_char.txt");

        Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
        File.WriteAllLines(filePath, new[] { "A", "b", string.Empty, "C" });

        try
        {
            var characters = CharacterSetCatalog.GetCharacters(key);
            Assert.Equal("AbC", characters);
        }
        finally
        {
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }
        }
    }

    [Fact]
    public void ThrowsWhenCharacterFileMissing()
    {
        var key = "missing_" + Guid.NewGuid().ToString("N");
        var exception = Assert.Throws<FileNotFoundException>(() => CharacterSetCatalog.GetCharacters(key));
        Assert.Contains(key, exception.Message);
    }
}
