using System;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenVinoSharp;
using OvTensor = OpenVinoSharp.Tensor;
using OvShape = OpenVinoSharp.Shape;

namespace EasyOcrNet;

internal readonly record struct ModelOutput(float[] Data, int[] Shape)
{
    public int Rank => Shape.Length;
    public int this[int index] => Shape[index];
}

internal interface IOcrBackend : IDisposable
{
    ModelOutput RunDetector(DenseTensor<float> input);
    ModelOutput RunRecognizer(DenseTensor<float> input);
    string Provider { get; }
}

internal sealed class OnnxBackend : IOcrBackend
{
    private readonly InferenceSession _detector;
    private readonly InferenceSession _recognizer;
    private readonly string _detectorInput;
    private readonly string _recognizerInput;

    public string Provider { get; }

    public OnnxBackend(string detectorPath, string recognizerPath)
    {
        if (!File.Exists(detectorPath))
            throw new FileNotFoundException($"Detector model not found at '{detectorPath}'", detectorPath);
        if (!File.Exists(recognizerPath))
            throw new FileNotFoundException($"Recognizer model not found at '{recognizerPath}'", recognizerPath);

        using var detectorOptions = CreateSessionOptions();
        using var recognizerOptions = CreateSessionOptions();

        _detector = new InferenceSession(detectorPath, detectorOptions);
        _recognizer = new InferenceSession(recognizerPath, recognizerOptions);

        _detectorInput = _detector.InputMetadata.Keys.Single();
        _recognizerInput = _recognizer.InputMetadata.Keys.Single();

        Provider = "ONNXRuntime";
    }

    private static SessionOptions CreateSessionOptions()
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_PARALLEL,
            IntraOpNumThreads = Math.Max(1, Environment.ProcessorCount),
            InterOpNumThreads = Math.Max(1, Environment.ProcessorCount / 2)
        };

        options.EnableCpuMemArena = true;
        options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
        options.AddSessionConfigEntry("session.intra_op.allow_spinning", "1");

        return options;
    }

    public ModelOutput RunDetector(DenseTensor<float> input)
    {
        using var outputs = _detector.Run(new[] { NamedOnnxValue.CreateFromTensor(_detectorInput, input) });
        var tensor = outputs[0].AsTensor<float>();
        return new ModelOutput(tensor.ToArray(), tensor.Dimensions.ToArray());
    }

    public ModelOutput RunRecognizer(DenseTensor<float> input)
    {
        using var outputs = _recognizer.Run(new[] { NamedOnnxValue.CreateFromTensor(_recognizerInput, input) });
        var tensor = outputs[0].AsTensor<float>();
        return new ModelOutput(tensor.ToArray(), tensor.Dimensions.ToArray());
    }

    public void Dispose()
    {
        _detector.Dispose();
        _recognizer.Dispose();
    }
}

internal interface IOpenVinoRuntime : IDisposable
{
    IOpenVinoCompiledModel CompileModel(string xmlPath, string binPath, long[] inputShape, string device);
}

internal interface IOpenVinoCompiledModel : IDisposable
{
    ModelOutput Run(DenseTensor<float> tensor, long[] expectedShape);
}

internal sealed class OpenVinoBackend : IOcrBackend
{
    private readonly IOpenVinoRuntime _runtime;
    private readonly IOpenVinoCompiledModel _detector;
    private readonly IOpenVinoCompiledModel _recognizer;
    private readonly long[] _detectorInputDims = { 1, 3, 608, 800 };
    private readonly long[] _recognizerInputDims = { 1, 1, 64, 1000 };

    public string Provider { get; }

    public OpenVinoBackend(string detectorXmlPath, string recognizerXmlPath, string device, Func<IOpenVinoRuntime>? runtimeFactory = null)
    {
        var detectorBin = Path.ChangeExtension(detectorXmlPath, ".bin");
        var recognizerBin = Path.ChangeExtension(recognizerXmlPath, ".bin");

        if (!File.Exists(detectorXmlPath))
            throw new FileNotFoundException($"Detector XML not found at '{detectorXmlPath}'", detectorXmlPath);
        if (!File.Exists(detectorBin))
            throw new FileNotFoundException($"Detector BIN not found at '{detectorBin}'", detectorBin);
        if (!File.Exists(recognizerXmlPath))
            throw new FileNotFoundException($"Recognizer XML not found at '{recognizerXmlPath}'", recognizerXmlPath);
        if (!File.Exists(recognizerBin))
            throw new FileNotFoundException($"Recognizer BIN not found at '{recognizerBin}'", recognizerBin);

        EnsureNativeRuntimeLoaded();

        _runtime = runtimeFactory?.Invoke() ?? CreateRuntime();
        IOpenVinoCompiledModel? detector = null;
        try
        {
            detector = _runtime.CompileModel(detectorXmlPath, detectorBin, _detectorInputDims, device);
            _detector = detector;
            _recognizer = _runtime.CompileModel(recognizerXmlPath, recognizerBin, _recognizerInputDims, device);
        }
        catch
        {
            detector?.Dispose();
            _runtime.Dispose();
            throw;
        }

        Provider = $"OpenVINO:{device}";
    }

    public ModelOutput RunDetector(DenseTensor<float> input)
    {
        return _detector.Run(input, _detectorInputDims);
    }

    public ModelOutput RunRecognizer(DenseTensor<float> input)
    {
        return _recognizer.Run(input, _recognizerInputDims);
    }

    public void Dispose()
    {
        _detector.Dispose();
        _recognizer.Dispose();
        _runtime.Dispose();
    }

    private static void EnsureNativeRuntimeLoaded()
    {
        var baseDir = AppContext.BaseDirectory;
        var nativeDir = Path.Combine(baseDir, "runtimes", "linux-x64", "native");
        if (!Directory.Exists(nativeDir))
        {
            return;
        }

        var currentLd = Environment.GetEnvironmentVariable("LD_LIBRARY_PATH") ?? string.Empty;
        if (!currentLd.Split(':', StringSplitOptions.RemoveEmptyEntries).Contains(nativeDir))
        {
            var updated = string.IsNullOrEmpty(currentLd) ? nativeDir : string.Concat(nativeDir, ":", currentLd);
            Environment.SetEnvironmentVariable("LD_LIBRARY_PATH", updated);
        }

        static void LoadIfExists(string path)
        {
            if (!File.Exists(path))
            {
                return;
            }

            try
            {
                NativeLibrary.Load(path);
            }
            catch (DllNotFoundException ex)
            {
                throw new InvalidOperationException($"Failed to preload OpenVINO native library '{path}'. Ensure runtime dependencies are present in the native folder.", ex);
            }
        }

        // Load the shared libraries explicitly to guarantee dependency resolution.
        LoadIfExists(Path.Combine(nativeDir, "libtbb.so.12"));
        LoadIfExists(Path.Combine(nativeDir, "libtbbmalloc.so"));
        LoadIfExists(Path.Combine(nativeDir, "libopenvino.so"));
        LoadIfExists(Path.Combine(nativeDir, "libopenvino_c.so"));
    }

    private static IOpenVinoRuntime CreateRuntime()
    {
        try
        {
            return new OpenVinoRuntime();
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException("OpenVINO native runtime could not be loaded. Ensure the OpenVINO runtime libraries are installed and available on PATH.", ex);
        }
    }

    private sealed class OpenVinoRuntime : IOpenVinoRuntime
    {
        private readonly Core _core;

        public OpenVinoRuntime()
        {
            _core = new Core();
        }

        public IOpenVinoCompiledModel CompileModel(string xmlPath, string binPath, long[] inputShape, string device)
        {
            using var model = _core.read_model(xmlPath, binPath);
            using var shape = new OvShape(inputShape);
            var partial = new PartialShape(shape);
            model.reshape(partial);
            var compiled = _core.compile_model(model, device);
            return new OpenVinoCompiledModel(compiled, inputShape);
        }

        public void Dispose()
        {
            _core.Dispose();
        }
    }

    private sealed class OpenVinoCompiledModel : IOpenVinoCompiledModel
    {
        private readonly CompiledModel _model;
        private readonly long[] _inputShape;

        public OpenVinoCompiledModel(CompiledModel model, long[] inputShape)
        {
            _model = model;
            _inputShape = inputShape;
        }

        public ModelOutput Run(DenseTensor<float> tensor, long[] expectedShape)
        {
            using var request = _model.create_infer_request();
            using var ovShape = new OvShape(_inputShape);
            using var ovTensor = new OvTensor(ovShape, tensor.ToArray());
            request.set_input_tensor(ovTensor);
            request.infer();
            using var outputTensor = request.get_output_tensor(0);
            var outputData = outputTensor.get_data<float>((int)outputTensor.get_size());
            var outputShape = outputTensor.get_shape().Select(dim => (int)dim).ToArray();
            return new ModelOutput(outputData, outputShape);
        }

        public void Dispose()
        {
            _model.Dispose();
        }
    }
}
