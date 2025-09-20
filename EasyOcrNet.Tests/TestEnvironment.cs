using System;
using System.IO;
using System.Linq;

namespace EasyOcrNet.Tests;

internal static class TestPaths
{
    public static string SolutionRoot { get; } = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));

    public static string ModelsDirectory { get; } = new[]
    {
        Path.Combine(SolutionRoot, "models", "cpu"),
        Path.Combine(SolutionRoot, "models")
    }.FirstOrDefault(Directory.Exists)
        ?? throw new DirectoryNotFoundException("Could not locate the OCR model directory under the repository.");

    public static string ExamplesDirectory { get; } = Path.Combine(SolutionRoot, "examples");
}
