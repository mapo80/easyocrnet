using EasyOcrNet.Utils;
using Xunit;

namespace EasyOcrNet.Tests;

public class CharsetLoaderTests
{
    [Theory]
    [InlineData("en")]
    [InlineData("it")]
    [InlineData("latin")]
    public void LoadsEmbeddedCharset(string language)
    {
        var charset = CharsetLoader.Load(language);
        Assert.False(string.IsNullOrWhiteSpace(charset));
        Assert.True(charset.Length > 10);
    }
}
