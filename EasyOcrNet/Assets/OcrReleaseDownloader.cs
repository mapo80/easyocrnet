using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace EasyOcrNet.Assets;

/// <summary>
/// Downloads EasyOcrNet runtime assets (ONNX models, etc.) from a GitHub release.
/// </summary>
public static class OcrReleaseDownloader
{
    private sealed record ReleaseAsset(
        string FileName,
        string ReleaseName,
        string Md5,
        string? InnerEntry = null)
    {
        public bool IsArchive => ReleaseName.EndsWith(".zip", StringComparison.OrdinalIgnoreCase);
        public string InnerFile => InnerEntry ?? FileName;
    }

    private static readonly IReadOnlyDictionary<string, ReleaseAsset> ModelAssets =
        new Dictionary<string, ReleaseAsset>(StringComparer.OrdinalIgnoreCase)
        {
            ["detection.onnx"] = new(
                FileName: "detection.onnx",
                ReleaseName: "detection.onnx",
                Md5: "c8fa14f85030d87c52f8990db50d68ef"),
            ["english_g2_rec.onnx"] = new(
                FileName: "english_g2_rec.onnx",
                ReleaseName: "english_g2_rec.onnx",
                Md5: "8deccfa817467f834edb79b39220312e"),
            ["latin_g2_rec.onnx"] = new(
                FileName: "latin_g2_rec.onnx",
                ReleaseName: "latin_g2_rec.onnx",
                Md5: "613a143cb017110c0cbadda32165b580"),
            ["zh_sim_g2_rec.onnx"] = new(
                FileName: "zh_sim_g2_rec.onnx",
                ReleaseName: "zh_sim_g2_rec.onnx",
                Md5: "9ba3fee6bfcca1d590d1cefcd862f43c"),
            ["japanese_g2_rec.onnx"] = new(
                FileName: "japanese_g2_rec.onnx",
                ReleaseName: "japanese_g2_rec.onnx",
                Md5: "c3f65a6ef8fdb9947ae8bfdcf559947d"),
            ["korean_g2_rec.onnx"] = new(
                FileName: "korean_g2_rec.onnx",
                ReleaseName: "korean_g2_rec.onnx",
                Md5: "de1a84cab05f9da31851c7e99fe6a62b"),
            ["thai_g1_rec.onnx"] = new(
                FileName: "thai_g1_rec.onnx",
                ReleaseName: "thai_g1_rec.onnx",
                Md5: "15388c67adea8c93b982fc44bcffff53"),
        };

    private static readonly HttpClient Http;

    static OcrReleaseDownloader()
    {
        Http = new HttpClient
        {
            Timeout = TimeSpan.FromMinutes(10)
        };
        Http.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("EasyOcrNet", "1.0"));
    }

    private static readonly ConcurrentDictionary<string, SemaphoreSlim> Locks = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Ensure that a specific model file exists locally, downloading it if necessary.
    /// </summary>
    /// <param name="modelPath">Destination path for the ONNX model.</param>
    /// <param name="options">GitHub release options (repository + tag).</param>
    /// <param name="logger">Optional logging callback.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public static async Task EnsureModelAsync(
        string modelPath,
        GithubReleaseOptions? options = null,
        Action<string>? logger = null,
        CancellationToken cancellationToken = default)
    {
        var fileName = Path.GetFileName(modelPath);
        if (!ModelAssets.TryGetValue(fileName, out var asset))
        {
            return;
        }

        var semaphore = Locks.GetOrAdd(Path.GetFullPath(modelPath), _ => new SemaphoreSlim(1, 1));
        await semaphore.WaitAsync(cancellationToken);
        try
        {
            if (File.Exists(modelPath))
            {
                var currentHash = ComputeMd5(modelPath);
                if (currentHash.Equals(asset.Md5, StringComparison.OrdinalIgnoreCase))
                {
                    logger?.Invoke($"✓ {fileName} già presente (md5 ok)");
                    return;
                }

                logger?.Invoke($"! {fileName} trovato ma checksum md5 errato, nuovo download in corso");
            }

            var baseUrl = (options ?? new GithubReleaseOptions()).BuildBaseUrl();
            var assetUrl = $"{baseUrl}/{asset.ReleaseName}";
            Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);
            await DownloadAssetAsync(assetUrl, asset, modelPath, logger, cancellationToken);

            var checksum = ComputeMd5(modelPath);
            if (!checksum.Equals(asset.Md5, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException(
                    $"Checksum md5 errato per {fileName}: atteso {asset.Md5}, ottenuto {checksum}");
            }

            var sizeMb = new FileInfo(modelPath).Length / (1024d * 1024d);
            logger?.Invoke($"✓ {fileName} scaricato ({sizeMb:F1} MB)");
        }
        finally
        {
            semaphore.Release();
        }
    }

    /// <summary>
    /// Synchronous wrapper around <see cref="EnsureModelAsync"/>.
    /// </summary>
    public static void EnsureModel(
        string modelPath,
        GithubReleaseOptions? options = null,
        Action<string>? logger = null,
        CancellationToken cancellationToken = default)
    {
        EnsureModelAsync(modelPath, options, logger, cancellationToken).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Ensure that all supported models exist under the provided directory.
    /// </summary>
    public static async Task EnsureAllModelsAsync(
        string destinationDirectory,
        GithubReleaseOptions? options = null,
        Action<string>? logger = null,
        CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(destinationDirectory);

        foreach (var asset in ModelAssets.Values)
        {
            var path = Path.Combine(destinationDirectory, asset.FileName);
            await EnsureModelAsync(path, options, logger, cancellationToken);
        }
    }

    /// <summary>
    /// Synchronous wrapper around <see cref="EnsureAllModelsAsync"/>.
    /// </summary>
    public static void EnsureAllModels(
        string destinationDirectory,
        GithubReleaseOptions? options = null,
        Action<string>? logger = null,
        CancellationToken cancellationToken = default)
    {
        EnsureAllModelsAsync(destinationDirectory, options, logger, cancellationToken).GetAwaiter().GetResult();
    }

    private static async Task DownloadAssetAsync(
        string assetUrl,
        ReleaseAsset asset,
        string destinationPath,
        Action<string>? logger,
        CancellationToken cancellationToken)
    {
        logger?.Invoke($"↓ Download {asset.FileName} da {assetUrl}");

        using var response = await Http.GetAsync(assetUrl, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();

        if (asset.IsArchive)
        {
            await using var memory = new MemoryStream();
            await response.Content.CopyToAsync(memory, cancellationToken);
            memory.Position = 0;

            using var archive = new ZipArchive(memory, ZipArchiveMode.Read, leaveOpen: false, entryNameEncoding: Encoding.UTF8);
            var entry = archive.Entries.FirstOrDefault(e =>
                e.FullName.EndsWith(asset.InnerFile, StringComparison.OrdinalIgnoreCase));

            if (entry == null)
            {
                throw new InvalidOperationException(
                    $"Il file {asset.InnerFile} non è presente nell'archivio {asset.ReleaseName}");
            }

            await using var entryStream = entry.Open();
            await using var target = File.Create(destinationPath);
            await entryStream.CopyToAsync(target, cancellationToken);
        }
        else
        {
            await using var target = File.Create(destinationPath);
            await response.Content.CopyToAsync(target, cancellationToken);
        }
    }

    private static string ComputeMd5(string path)
    {
        using var stream = File.OpenRead(path);
        var hash = MD5.HashData(stream);
        var builder = new StringBuilder(hash.Length * 2);
        foreach (var b in hash)
        {
            builder.Append(b.ToString("x2"));
        }
        return builder.ToString();
    }
}
