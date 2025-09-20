using EasyOcrNet;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using Xunit;

namespace EasyOcrNet.Tests;

public class OpenVinoBackendTests
{
    [Fact]
    public void ThrowsWhenDetectorXmlMissing()
    {
        var exception = Assert.Throws<FileNotFoundException>(() => new OpenVinoBackend("missing.xml", "recognizer.xml", "CPU"));
        Assert.Contains("Detector XML", exception.Message);
    }

    [Fact]
    public void ThrowsWhenDetectorBinMissing()
    {
        var tempDir = Directory.CreateDirectory(Path.Combine(Path.GetTempPath(), "openvino-test-" + Guid.NewGuid().ToString("N")));
        try
        {
            var detectorXml = Path.Combine(tempDir.FullName, "detection.xml");
            File.WriteAllText(detectorXml, "<net></net>");

            var exception = Assert.Throws<FileNotFoundException>(() => new OpenVinoBackend(detectorXml, Path.Combine(tempDir.FullName, "recognizer.xml"), "CPU"));
            Assert.Contains("Detector BIN", exception.Message);
        }
        finally
        {
            tempDir.Delete(recursive: true);
        }
    }

    [Fact]
    public void ThrowsWhenRecognizerXmlMissing()
    {
        var tempDir = Directory.CreateDirectory(Path.Combine(Path.GetTempPath(), "openvino-test-" + Guid.NewGuid().ToString("N")));
        try
        {
            var detectorXml = Path.Combine(tempDir.FullName, "detection.xml");
            File.WriteAllText(detectorXml, "<net></net>");
            File.WriteAllBytes(Path.ChangeExtension(detectorXml, ".bin"), new byte[8]);

            var exception = Assert.Throws<FileNotFoundException>(() => new OpenVinoBackend(detectorXml, Path.Combine(tempDir.FullName, "recognizer.xml"), "CPU"));
            Assert.Contains("Recognizer XML", exception.Message);
        }
        finally
        {
            tempDir.Delete(recursive: true);
        }
    }

    [Fact]
    public void ThrowsWhenRecognizerBinMissing()
    {
        var tempDir = Directory.CreateDirectory(Path.Combine(Path.GetTempPath(), "openvino-test-" + Guid.NewGuid().ToString("N")));
        try
        {
            var detectorXml = Path.Combine(tempDir.FullName, "detection.xml");
            File.WriteAllText(detectorXml, "<net></net>");
            File.WriteAllBytes(Path.ChangeExtension(detectorXml, ".bin"), new byte[8]);

            var recognizerXml = Path.Combine(tempDir.FullName, "recognizer.xml");
            File.WriteAllText(recognizerXml, "<net></net>");

            var exception = Assert.Throws<FileNotFoundException>(() => new OpenVinoBackend(detectorXml, recognizerXml, "CPU"));
            Assert.Contains("Recognizer BIN", exception.Message);
        }
        finally
        {
            tempDir.Delete(recursive: true);
        }
    }

    [Fact]
    public void UsesRuntimeToRunDetectorAndRecognizer()
    {
        var tempDir = Directory.CreateDirectory(Path.Combine(Path.GetTempPath(), "openvino-test-" + Guid.NewGuid().ToString("N")));
        try
        {
            var detectorXml = Path.Combine(tempDir.FullName, "detection.xml");
            var recognizerXml = Path.Combine(tempDir.FullName, "recognizer.xml");
            File.WriteAllText(detectorXml, "<net></net>");
            File.WriteAllBytes(Path.ChangeExtension(detectorXml, ".bin"), new byte[8]);
            File.WriteAllText(recognizerXml, "<net></net>");
            File.WriteAllBytes(Path.ChangeExtension(recognizerXml, ".bin"), new byte[8]);

            var runtime = new FakeRuntime
            {
                DetectorResult = new ModelOutput(new[] { 1f, 2f, 3f }, new[] { 1, 3 }),
                RecognizerResult = new ModelOutput(new[] { 4f, 5f }, new[] { 1, 2 })
            };

            using (var backend = new OpenVinoBackend(detectorXml, recognizerXml, "GPU", () => runtime))
            {
                Assert.Equal("OpenVINO:GPU", backend.Provider);

                var detectorInput = new DenseTensor<float>(new float[1 * 3 * 608 * 800], new[] { 1, 3, 608, 800 });
                var recognizerInput = new DenseTensor<float>(new float[1 * 1 * 64 * 1000], new[] { 1, 1, 64, 1000 });

                var detectorOutput = backend.RunDetector(detectorInput);
                var recognizerOutput = backend.RunRecognizer(recognizerInput);

                Assert.Same(detectorInput, runtime.DetectorModel!.LastInput);
                Assert.Same(recognizerInput, runtime.RecognizerModel!.LastInput);
                Assert.Equal(runtime.DetectorResult.Data, detectorOutput.Data);
                Assert.Equal(runtime.DetectorResult.Shape, detectorOutput.Shape);
                Assert.Equal(runtime.RecognizerResult.Data, recognizerOutput.Data);
                Assert.Equal(runtime.RecognizerResult.Shape, recognizerOutput.Shape);
            }

            Assert.True(runtime.DetectorModel!.Disposed);
            Assert.True(runtime.RecognizerModel!.Disposed);
            Assert.True(runtime.Disposed);
        }
        finally
        {
            tempDir.Delete(recursive: true);
        }
    }

    [Fact]
    public void DisposesRuntimeWhenCompilationFails()
    {
        var tempDir = Directory.CreateDirectory(Path.Combine(Path.GetTempPath(), "openvino-test-" + Guid.NewGuid().ToString("N")));
        try
        {
            var detectorXml = Path.Combine(tempDir.FullName, "detection.xml");
            var recognizerXml = Path.Combine(tempDir.FullName, "recognizer.xml");
            File.WriteAllText(detectorXml, "<net></net>");
            File.WriteAllBytes(Path.ChangeExtension(detectorXml, ".bin"), new byte[8]);
            File.WriteAllText(recognizerXml, "<net></net>");
            File.WriteAllBytes(Path.ChangeExtension(recognizerXml, ".bin"), new byte[8]);

            var runtime = new FakeRuntime
            {
                DetectorResult = new ModelOutput(Array.Empty<float>(), Array.Empty<int>()),
                ThrowOnSecondCompile = true
            };

            Assert.Throws<InvalidOperationException>(() => new OpenVinoBackend(detectorXml, recognizerXml, "CPU", () => runtime));

            Assert.True(runtime.Disposed);
            Assert.True(runtime.DetectorModel!.Disposed);
        }
        finally
        {
            tempDir.Delete(recursive: true);
        }
    }

    private sealed class FakeRuntime : IOpenVinoRuntime
    {
        private int _compileCount;

        public FakeCompiledModel? DetectorModel { get; private set; }
        public FakeCompiledModel? RecognizerModel { get; private set; }
        public bool Disposed { get; private set; }
        public ModelOutput DetectorResult { get; set; }
        public ModelOutput RecognizerResult { get; set; }
        public bool ThrowOnSecondCompile { get; set; }

        public IOpenVinoCompiledModel CompileModel(string xmlPath, string binPath, long[] inputShape, string device)
        {
            _compileCount++;
            if (_compileCount == 2 && ThrowOnSecondCompile)
            {
                throw new InvalidOperationException("compile failed");
            }

            var result = _compileCount == 1 ? DetectorResult : RecognizerResult;
            var model = new FakeCompiledModel(inputShape.ToArray(), result);

            if (_compileCount == 1)
            {
                DetectorModel = model;
            }
            else
            {
                RecognizerModel = model;
            }

            return model;
        }

        public void Dispose()
        {
            Disposed = true;
        }
    }

    private sealed class FakeCompiledModel : IOpenVinoCompiledModel
    {
        private readonly long[] _expectedShape;
        private readonly ModelOutput _result;

        public FakeCompiledModel(long[] expectedShape, ModelOutput result)
        {
            _expectedShape = expectedShape;
            _result = result;
        }

        public DenseTensor<float>? LastInput { get; private set; }
        public bool Disposed { get; private set; }

        public ModelOutput Run(DenseTensor<float> tensor, long[] expectedShape)
        {
            LastInput = tensor;
            Assert.True(expectedShape.SequenceEqual(_expectedShape));
            return _result;
        }

        public void Dispose()
        {
            Disposed = true;
        }
    }
}
