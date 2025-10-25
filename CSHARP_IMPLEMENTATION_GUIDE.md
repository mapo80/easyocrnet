# EasyOcrNet C# Implementation Guide

## Obiettivo
Portare l'implementazione Python completa in C# con **parità di output al 100%**.

## Architettura

```
EasyOcrNet/                  # Core library
├── Models/
│   ├── OcrResult.cs        # Result data structures
│   ├── DetectionResult.cs
│   └── RecognitionResult.cs
├── Detection/
│   ├── IDetector.cs        # Interface
│   ├── CraftDetector.cs    # CRAFT implementation
│   └── DetectionUtils.cs   # resize_aspect_ratio, normalizeMeanVariance, etc.
├── Recognition/
│   ├── IRecognizer.cs      # Interface
│   ├── OnnxRecognizer.cs   # Recognition implementation
│   └── RecognitionUtils.cs # Preprocessing utilities
├── PostProcessing/
│   ├── ItalianPostProcessor.cs
│   ├── ContextualCorrections.cs
│   └── SpellChecker.cs     # SymSpell integration
├── Utils/
│   ├── CraftUtils.cs       # Port of craft_utils.py
│   ├── ImageUtils.cs
│   └── CharsetLoader.cs
└── OcrEngine.cs            # Main entry point

EasyOcrNet.Tests/           # Unit tests
├── DetectionTests.cs       # Test vs Python output
├── RecognitionTests.cs     # Test vs Python output
├── PostProcessingTests.cs
└── IntegrationTests.cs     # Full pipeline tests

EasyOcrNet.Cli/             # CLI application
└── Program.cs              # Dataset processing CLI
```

## Dipendenze NuGet (Versioni Stabili)

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.17.0" />
<PackageReference Include="OpenCvSharp4" Version="4.9.0.20240103" />
<PackageReference Include="OpenCvSharp4.runtime.win" Version="4.9.0.20240103" />
<PackageReference Include="OpenCvSharp4.runtime.osx" Version="4.9.0.20240103" />
<PackageReference Include="SixLabors.ImageSharp" Version="3.1.3" />
<PackageReference Include="SymSpell" Version="6.7.2" />
<PackageReference Include="System.Text.Json" Version="8.0.1" />
```

## Step 1: Core Data Models

### EasyOcrNet/Models/OcrResult.cs
```csharp
namespace EasyOcrNet.Models;

public record Point2D(int X, int Y);

public record BoundingBox(
    Point2D TopLeft,
    Point2D TopRight,
    Point2D BottomRight,
    Point2D BottomLeft
)
{
    public Point2D[] ToArray() => new[] { TopLeft, TopRight, BottomRight, BottomLeft };
}

public record DetectionResult(
    BoundingBox BoundingBox,
    float Confidence
);

public record RecognitionResult(
    string Text,
    float Confidence
);

public record OcrResult(
    BoundingBox BoundingBox,
    string Text,
    float Confidence
);
```

## Step 2: Detection Pipeline

### EasyOcrNet/Detection/CraftDetector.cs

Key methods to implement:
1. `Preprocess(Mat image)` → Convert BGR→RGB, resize_aspect_ratio, normalize
2. `RunInference(float[] input)` → ONNX inference
3. `Postprocess(float[] output)` → getDetBoxes, adjustResultCoordinates

**Critical:** Use `cv2.INTER_LINEAR` for all resizing (matches Python exactly)

### EasyOcrNet/Utils/CraftUtils.cs

Port these Python functions:
- `resize_aspect_ratio()` - Aspect ratio preserving resize
- `normalizeMeanVariance()` - Normalize to [0,1]
- `getDetBoxes()` - Extract bounding boxes from heatmap
- `adjustResultCoordinates()` - Scale boxes back to original size
- `group_text_box()` - Group boxes into lines

## Step 3: Recognition Pipeline

### EasyOcrNet/Recognition/OnnxRecognizer.cs

Key methods:
1. `Preprocess(Mat crop)` → Grayscale, resize, normalize to [-1,1], pad
2. `RunInference(float[] input)` → ONNX inference
3. `Decode(float[] output, string charset)` → CTC greedy decoding
4. `CalculateConfidence(float[] output)` → Custom mean formula

**Critical preprocessing steps:**
```csharp
// 1. Convert to grayscale BEFORE cropping (order matters!)
var grayImage = new Mat();
Cv2.CvtColor(image, grayImage, ColorConversionCodes.BGR2GRAY);

// 2. Extract crop from grayscale
var crop = grayImage[yMin..yMax, xMin..xMax];

// 3. Resize with INTER_LINEAR
var resized = new Mat();
Cv2.Resize(crop, resized, new Size(resizedW, imgH), 0, 0, InterpolationFlags.Linear);

// 4. Normalize to [-1, 1]
resized.ConvertTo(normalized, MatType.CV_32F);
normalized = (normalized / 255.0f - 0.5f) / 0.5f;

// 5. Pad with last column repeated
// ... (see Python implementation)
```

## Step 4: Post-Processing

### EasyOcrNet/PostProcessing/ItalianPostProcessor.cs

Implement these transformations:
1. `NormalizePunctuation()` - ' → ', " → "
2. `FixUnicodeSubstitutions()` - ľ → l', Ī → l, Ē → È
3. `SeparateCompoundWords()` - nelluniversità → nell'università
4. `FixContextualErrors()` - progerto → progetto, etc.
5. `RestoreAccents()` - caffe → caffè, citta → città

### EasyOcrNet/PostProcessing/SpellChecker.cs

```csharp
using SymSpell;

public class SpellChecker
{
    private readonly SymSpell _symSpell;

    public SpellChecker(string dictionaryPath)
    {
        _symSpell = new SymSpell(maxDictionaryEditDistance: 2, prefixLength: 7);
        _symSpell.LoadDictionary(dictionaryPath, termIndex: 0, countIndex: 1);
    }

    public string CorrectText(string text, bool enableSpellCheck = false)
    {
        if (!enableSpellCheck) return text;

        // Split into tokens
        // Apply corrections with confidence threshold
        // Preserve capitalization
        // Return corrected text
    }
}
```

## Step 5: Main OCR Engine

### EasyOcrNet/OcrEngine.cs

```csharp
public class OcrEngine : IDisposable
{
    private readonly IDetector _detector;
    private readonly IRecognizer _recognizer;
    private readonly IPostProcessor _postProcessor;

    public OcrEngine(
        string detectorPath,
        string recognizerPath,
        string charset,
        string language = "en",
        bool enableSpellCheck = false)
    {
        _detector = new CraftDetector(detectorPath);
        _recognizer = new OnnxRecognizer(recognizerPath, charset);
        _postProcessor = PostProcessorFactory.Create(language, enableSpellCheck);
    }

    public async Task<List<OcrResult>> ProcessImageAsync(string imagePath)
    {
        // 1. Load image
        using var image = Cv2.ImRead(imagePath);

        // 2. Detection
        var detections = await _detector.DetectAsync(image);

        // 3. Recognition (with two-pass for low confidence)
        var results = new List<OcrResult>();
        foreach (var detection in detections)
        {
            var text = await _recognizer.RecognizeAsync(image, detection);

            // Apply post-processing
            text.Text = _postProcessor.Process(text.Text);

            results.Add(new OcrResult(
                detection.BoundingBox,
                text.Text,
                text.Confidence
            ));
        }

        return results;
    }
}
```

## Step 6: CLI Application

### EasyOcrNet.Cli/Program.cs

```csharp
using CommandLine;

public class Options
{
    [Option("dataset", Required = true)]
    public string Dataset { get; set; }

    [Option("models", Default = "models/cpu")]
    public string Models { get; set; }

    [Option("lang", Default = "en")]
    public string Language { get; set; }

    [Option("mode", Default = "all")]
    public string Mode { get; set; }

    [Option("spell-check", Default = false)]
    public bool EnableSpellCheck { get; set; }
}

class Program
{
    static async Task Main(string[] args)
    {
        await Parser.Default.ParseArguments<Options>(args)
            .WithParsedAsync(RunOcr);
    }

    static async Task RunOcr(Options opts)
    {
        // 1. Load models
        var detectorPath = Path.Combine(opts.Models, "detection.onnx");
        var recognizerPath = GetRecognizerPath(opts.Language, opts.Models);
        var charset = CharsetLoader.Load(opts.Language);

        // 2. Create OCR engine
        using var engine = new OcrEngine(
            detectorPath,
            recognizerPath,
            charset,
            opts.Language,
            opts.EnableSpellCheck);

        // 3. Process images
        var imageFiles = Directory.GetFiles(opts.Dataset, "*.png")
            .Where(f => !f.Contains(".ocr."));

        foreach (var imageFile in imageFiles)
        {
            var results = await engine.ProcessImageAsync(imageFile);

            // Save results (matching Python format)
            await SaveResultsAsync(imageFile, results, opts.Mode);
        }
    }
}
```

## Step 7: Tests

### EasyOcrNet.Tests/IntegrationTests.cs

```csharp
[Fact]
public async Task English_Output_Should_Match_Python_Exactly()
{
    // Arrange
    var pythonOutput = File.ReadAllLines("dataset/base/image.png.ocr.python.txt");

    using var engine = new OcrEngine(
        "models/cpu/detection.onnx",
        "models/cpu/english_g2_rec.onnx",
        CharsetLoader.Load("en"),
        "en");

    // Act
    var results = await engine.ProcessImageAsync("dataset/base/image.png");

    // Assert
    Assert.Equal(pythonOutput.Length, results.Count);

    for (int i = 0; i < results.Count; i++)
    {
        var pythonLine = ParsePythonOutput(pythonOutput[i]);
        var result = results[i];

        // Exact match required
        Assert.Equal(pythonLine.Text, result.Text);
        Assert.Equal(pythonLine.Confidence, result.Confidence, precision: 4);
    }
}

[Fact]
public async Task Italian_PostProcessing_Should_Match_Python()
{
    // Test contextual corrections
    // Test accent restoration
    // Test spell checking (if enabled)
}
```

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create new EasyOcrNet project structure
- [ ] Add NuGet dependencies
- [ ] Implement data models (OcrResult, BoundingBox, etc.)
- [ ] Port CraftUtils.cs from Python craft_utils.py

### Phase 2: Detection
- [ ] Implement CraftDetector.Preprocess
- [ ] Implement ONNX inference for detection
- [ ] Port getDetBoxes algorithm
- [ ] Port adjustResultCoordinates
- [ ] Port group_text_box
- [ ] Test detection output vs Python

### Phase 3: Recognition
- [ ] Implement recognizer preprocessing (exact order matters!)
- [ ] Implement CTC greedy decoding
- [ ] Implement confidence calculation
- [ ] Implement two-pass recognition (low confidence)
- [ ] Test recognition output vs Python

### Phase 4: Post-Processing
- [ ] Implement ItalianPostProcessor
- [ ] Implement contextual corrections
- [ ] Integrate SymSpell for spell checking
- [ ] Test post-processing output vs Python

### Phase 5: Integration
- [ ] Implement OcrEngine main class
- [ ] Implement charset loading
- [ ] Implement model path resolution
- [ ] Create CLI application
- [ ] Test full pipeline vs Python

### Phase 6: Validation
- [ ] Run on dataset/base (English) - must match 100%
- [ ] Run on dataset/it (Italian) - must match 100%
- [ ] Performance benchmarks
- [ ] Cross-platform testing (Windows, macOS, Linux)

## Critical Implementation Notes

### 1. Image Processing Order
**MUST MATCH EXACTLY:**
```
1. Load BGR image
2. Convert entire image to grayscale
3. Extract crops from grayscale image
4. Resize with INTER_LINEAR
5. Normalize to [-1, 1]
6. Pad with last column repeated
```

### 2. Numerical Precision
- Use `float32` everywhere (not double)
- Use same confidence formula: `custom_mean(x) = x.prod() ** (2.0 / sqrt(len(x)))`
- Round confidence to 4 decimal places

### 3. Character Encoding
- Load charset files with UTF-8 encoding
- Handle accented characters correctly
- Preserve exact character order from Python

### 4. Post-Processing Rules
- Apply in exact order: punctuation → unicode → compounds → contextual → spell
- Use case-insensitive regex where Python does
- Preserve capitalization in spell corrections

## Testing Strategy

1. **Unit Tests**: Each component vs Python reference output
2. **Integration Tests**: Full pipeline vs Python on known images
3. **Regression Tests**: Ensure changes don't break existing tests
4. **Performance Tests**: Measure vs Python baseline

## Expected Performance

- **English accuracy**: 100% match with Python (required)
- **Italian accuracy**: 100% match with Python (70.72% vs ground truth)
- **Speed**: Should be faster than Python (C# + ONNX Runtime optimization)
- **Memory**: Similar to Python (~500MB for models)

## Deliverables

1. ✅ EasyOcrNet.csproj - Core library
2. ✅ EasyOcrNet.Tests.csproj - Comprehensive tests
3. ✅ EasyOcrNet.Cli.csproj - CLI application
4. ✅ README.md - Usage documentation
5. ✅ All tests passing with 100% parity

## Next Steps

1. Start with Phase 1 (infrastructure)
2. Implement and test each phase sequentially
3. Validate against Python after each phase
4. Do NOT proceed to next phase until current phase matches Python exactly

## Support Files Needed

Create these files in the repository:
- `character/en_charset.txt` - English charset
- `character/it_charset.txt` - Italian charset
- `dictionaries/it_complete.txt` - Already exists
- `models/cpu/detection.onnx` - Already exists
- `models/cpu/english_g2_rec.onnx` - Already exists
- `models/cpu/latin_g2_rec.onnx` - Already exists
