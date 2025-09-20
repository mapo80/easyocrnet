using EasyOcrNet;
using System;
using Xunit;

namespace EasyOcrNet.Tests;

public class OcrExecutionProfileTests
{
    [Fact]
    public void EmptyDurationsReturnEmptyProfile()
    {
        var profile = OcrExecutionProfile.Create(TimeSpan.FromMilliseconds(1), TimeSpan.FromMilliseconds(2), TimeSpan.FromMilliseconds(3), Array.Empty<double>(), null!);

        Assert.Equal(TimeSpan.FromMilliseconds(1), profile.DetectionDuration);
        Assert.Equal(TimeSpan.FromMilliseconds(2), profile.RecognitionDuration);
        Assert.Equal(TimeSpan.FromMilliseconds(3), profile.TotalDuration);
        Assert.Equal(0d, profile.WarmedAverageRecognitionMilliseconds);
        Assert.Equal(string.Empty, profile.Provider);
        Assert.Empty(profile.RecognitionDurations);
    }

    [Fact]
    public void WarmedAverageSkipsFirstDuration()
    {
        var durations = new[] { 10d, 6d, 4d };
        var profile = OcrExecutionProfile.Create(TimeSpan.Zero, TimeSpan.Zero, TimeSpan.Zero, durations, "test");

        durations[0] = 0d; // ensure a copy was made

        Assert.Equal(5d, profile.WarmedAverageRecognitionMilliseconds);
        Assert.Equal(3, profile.RecognitionDurations.Count);
        Assert.Equal(10d, profile.RecognitionDurations[0]);
        Assert.Equal("test", profile.Provider);
    }

    [Fact]
    public void SingleDurationUsesExactValue()
    {
        var profile = OcrExecutionProfile.Create(TimeSpan.Zero, TimeSpan.Zero, TimeSpan.Zero, new[] { 42d }, "provider");
        Assert.Equal(42d, profile.WarmedAverageRecognitionMilliseconds);
        Assert.Single(profile.RecognitionDurations);
    }
}
