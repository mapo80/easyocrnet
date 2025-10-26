namespace EasyOcrNet.Assets;

/// <summary>
/// Options that identify the GitHub release containing EasyOcrNet assets.
/// </summary>
public sealed record GithubReleaseOptions(
    string Repository = "mapo80/easyocrnet",
    string Tag = "v2025.09.19")
{
    /// <summary>
    /// Build the base download URL for the release.
    /// </summary>
    public string BuildBaseUrl() => $"https://github.com/{Repository}/releases/download/{Tag}";
}
