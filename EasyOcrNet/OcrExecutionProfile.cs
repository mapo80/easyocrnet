using System;
using System.Collections.Generic;

namespace EasyOcrNet;

public sealed class OcrExecutionProfile
{
    private static readonly IReadOnlyList<double> EmptyDurations = Array.AsReadOnly(Array.Empty<double>());

    public static OcrExecutionProfile Empty { get; } = new(
        TimeSpan.Zero,
        TimeSpan.Zero,
        TimeSpan.Zero,
        0d,
        EmptyDurations,
        string.Empty);

    private readonly IReadOnlyList<double> _recognitionDurations;

    private OcrExecutionProfile(
        TimeSpan detectionDuration,
        TimeSpan recognitionDuration,
        TimeSpan totalDuration,
        double warmedAverage,
        IReadOnlyList<double> recognitionDurations,
        string provider)
    {
        DetectionDuration = detectionDuration;
        RecognitionDuration = recognitionDuration;
        TotalDuration = totalDuration;
        WarmedAverageRecognitionMilliseconds = warmedAverage;
        _recognitionDurations = recognitionDurations;
        Provider = provider;
    }

    public TimeSpan DetectionDuration { get; }

    public TimeSpan RecognitionDuration { get; }

    public TimeSpan TotalDuration { get; }

    public double WarmedAverageRecognitionMilliseconds { get; }

    public string Provider { get; }

    public IReadOnlyList<double> RecognitionDurations => _recognitionDurations;

    internal static OcrExecutionProfile Create(
        TimeSpan detectionDuration,
        TimeSpan recognitionDuration,
        TimeSpan totalDuration,
        double[] recognitionDurations,
        string provider)
    {
        if (recognitionDurations.Length == 0)
        {
            return new OcrExecutionProfile(
                detectionDuration,
                recognitionDuration,
                totalDuration,
                0d,
                EmptyDurations,
                provider ?? string.Empty);
        }

        var snapshot = (double[])recognitionDurations.Clone();
        var readOnly = Array.AsReadOnly(snapshot);

        double warmedAverage;
        if (snapshot.Length <= 1)
        {
            warmedAverage = snapshot[0];
        }
        else
        {
            double sum = 0d;
            for (int i = 1; i < snapshot.Length; i++)
            {
                sum += snapshot[i];
            }
            warmedAverage = sum / (snapshot.Length - 1);
        }

        return new OcrExecutionProfile(
            detectionDuration,
            recognitionDuration,
            totalDuration,
            warmedAverage,
            readOnly,
            provider ?? string.Empty);
    }
}
