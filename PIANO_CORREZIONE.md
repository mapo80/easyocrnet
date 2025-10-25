# 🔧 PIANO DI CORREZIONE - EasyOcrNet Parity Issues

**Data:** 2025-10-25
**Stato:** Implementazione funzionante ma con 0% di parità
**Target:** 100% parità con Python

---

## 🎯 OBIETTIVO

Raggiungere **100% di parità** tra l'implementazione Python e C# di EasyOcrNet:
- Stesso numero di detections
- Stessi bounding boxes (con tolleranza 2px)
- Stesso testo riconosciuto
- Stesso ordine di output

---

## 🔍 ROOT CAUSE ANALYSIS

### Problema 1: GROUPING SEMPRE ATTIVO ⚠️⚠️⚠️

**Gravità:** CRITICA
**Impatto:** Causa TUTTE le differenze

**Analisi:**
```csharp
// CraftDetector.PostprocessDetection() - LINEA 207-233
// Il grouping è SEMPRE eseguito, anche quando non dovrebbe!

var (horizontalList, freeList) = CraftUtils.GroupTextBoxFlat(
    flatPolys,
    slopeThreshold: _config.SlopeThreshold,  // <- Sempre eseguito!
    ...
);
```

**Problema:**
- Python fa grouping SOLO in `run_ocr()`, NON in `detector_postprocess()`
- C# fa grouping SEMPRE nel detector → risultati completamente diversi
- Il comando `detect` dovrebbe restituire bbox RAW senza grouping

**Evidenza:**
```
Python detection RAW:  17 bbox (da getDetBoxes)
Python dopo grouping:  10 bbox (in run_ocr)

C# detection:          13 bbox (grouping applicato sempre!)
Dovrebbe essere:       17 bbox (come Python RAW)
```

### Problema 2: Coordinate Negative nel Grouping 🐛

**Gravità:** ALTA
**Causa:** Bug nel calcolo del margine in `MergeHorizontalBoxes()`

```csharp
// LINEA ~697 CraftUtils.cs
int margin = (int)(addMargin * Math.Min(box.xMax - box.xMin, box.height));
mergedList.Add(new[] { box.xMin - margin, ... });  // <- Può diventare negativo!
```

**Fix:**
```csharp
int margin = (int)(addMargin * Math.Min(box.xMax - box.xMin, box.height));
int xMin = Math.Max(0, box.xMin - margin);  // Clamp a 0
int xMax = Math.Min(imageWidth, box.xMax + margin);
int yMin = Math.Max(0, box.yMin - margin);
int yMax = Math.Min(imageHeight, box.yMax + margin);
mergedList.Add(new[] { xMin, xMax, yMin, yMax });
```

### Problema 3: Ordinamento Detections Diverso 📐

**Gravità:** MEDIA
**Impatto:** Rende impossibile confronto linea-per-linea

**Python:**
```python
# craft_utils.py - group_text_box()
if sort_output:
    horizontal_list = sorted(horizontal_list, key=lambda item: item[4])  # Sort by y_center
```

**C#:**
```csharp
// CraftUtils.cs - LINEA 360
if (sortOutput && horizontalList.Count > 0)
{
    horizontalList = horizontalList.OrderBy(box => box.yCenter).ToList();  // OK!
}
```

**Nota:** L'ordinamento sembra corretto, ma viene applicato sui bbox DOPO grouping errato.

### Problema 4: Architettura Errata 🏗️

**Problema:** Il grouping è nel posto sbagliato della pipeline!

**Architettura Python (CORRETTA):**
```
1. detector_preprocess()
2. ONNX inference
3. detector_postprocess() → bbox RAW (17)
4. run_ocr():
   - group_text_box() → bbox merged (10)
   - crop extraction
   - recognition
```

**Architettura C# (ERRATA):**
```
1. PreprocessImage()
2. ONNX inference
3. PostprocessDetection():
   - GetDetBoxes → bbox RAW (17)
   - GroupTextBoxFlat → bbox merged (13) ← SBAGLIATO QUI!
4. Recognition
```

---

## 📋 PIANO D'AZIONE

### FASE 1: Separare Detection e Grouping ⭐⭐⭐ (PRIORITÀ MASSIMA)

**Obiettivo:** Rimuovere il grouping dal CraftDetector e metterlo dove dovrebbe essere.

**Step 1.1:** Modificare `CraftDetector.PostprocessDetection()`
```csharp
private List<DetectionResult> PostprocessDetection(float[,,,] scoreMap, float ratio)
{
    // ... extraction score maps ...

    var (boxes, _, _) = CraftUtils.GetDetBoxes(...);

    // Scale back to original coordinates
    boxes = CraftUtils.AdjustResultCoordinates(boxes, ratioW, ratioH);

    // ❌ RIMUOVERE IL GROUPING DA QUI!
    // var (horizontalList, freeList) = CraftUtils.GroupTextBoxFlat(...);

    // ✅ Restituire bbox RAW
    var detections = new List<DetectionResult>();
    foreach (var box in boxes)
    {
        var boundingBox = new BoundingBox(...);
        detections.Add(new DetectionResult(boundingBox, Confidence: 1.0f));
    }

    return detections;
}
```

**Step 1.2:** Creare `OcrEngine` per la pipeline completa

```csharp
// EasyOcrNet/OcrEngine.cs (NUOVO FILE)
public class OcrEngine : IDisposable
{
    private readonly IDetector _detector;
    private readonly IRecognizer _recognizer;
    private readonly OcrConfig _config;

    public async Task<List<OcrResult>> ProcessImageAsync(SKBitmap bitmap)
    {
        // 1. Detection (RAW)
        var detections = await _detector.DetectAsync(bitmap);

        // 2. Group text boxes (QUI!)
        var groupedDetections = GroupDetections(detections, bitmap.Width, bitmap.Height);

        // 3. Recognition
        var results = new List<OcrResult>();
        foreach (var detection in groupedDetections)
        {
            var recognition = await _recognizer.RecognizeAsync(bitmap, detection);
            results.Add(new OcrResult(detection.BoundingBox, recognition.Text, recognition.Confidence));
        }

        return results;
    }

    private List<DetectionResult> GroupDetections(List<DetectionResult> detections, int imageWidth, int imageHeight)
    {
        // Convert to flat format
        var flatPolys = new List<float[]>();
        foreach (var det in detections)
        {
            var bbox = det.BoundingBox;
            flatPolys.Add(new[] {
                bbox.TopLeft.X, bbox.TopLeft.Y,
                bbox.TopRight.X, bbox.TopRight.Y,
                bbox.BottomRight.X, bbox.BottomRight.Y,
                bbox.BottomLeft.X, bbox.BottomLeft.Y
            });
        }

        // Apply grouping
        var (horizontalList, freeList) = CraftUtils.GroupTextBoxFlat(
            flatPolys,
            _config.SlopeThreshold,
            _config.YCenterThreshold,
            _config.HeightThreshold,
            _config.WidthThreshold,
            _config.AddMargin,
            sortOutput: true);

        // Convert back with coordinate clamping
        return ConvertGroupedBoxes(horizontalList, freeList, imageWidth, imageHeight);
    }
}
```

**Step 1.3:** Aggiornare CLI per usare `OcrEngine`

```csharp
// EasyOcrNet.Cli/Program.cs - OcrAsync()

// PRIMA:
using var detector = new CraftDetector(detectorPath, config);
using var recognizer = new CrnnRecognizer(recognizerPath, language, config);
var detections = await detector.DetectAsync(bitmap);
// ... manual recognition loop ...

// DOPO:
using var ocrEngine = new OcrEngine(detectorPath, recognizerPath, language, config);
var results = await ocrEngine.ProcessImageAsync(bitmap);
```

**Risultato Atteso:**
- `detect` command: 17 bbox (come Python RAW)
- `ocr` command: 10 bbox (come Python dopo grouping)

---

### FASE 2: Fix Coordinate Negative ⭐⭐ (ALTA PRIORITÀ)

**Step 2.1:** Aggiungere clamping in `MergeHorizontalBoxes()`

```csharp
// CraftUtils.cs - MergeHorizontalBoxes()
// Aggiungere parametri imageWidth, imageHeight

private static List<int[]> MergeHorizontalBoxes(
    List<(int xMin, int xMax, int yMin, int yMax, float yCenter, float height)> horizontalList,
    float ycenterThreshold,
    float heightThreshold,
    float widthThreshold,
    float addMargin,
    int imageWidth,    // ← NUOVO
    int imageHeight)   // ← NUOVO
{
    // ... existing logic ...

    int margin = (int)(addMargin * Math.Min(boxWidth, boxHeight));

    // ✅ CLAMP COORDINATES
    int xMin = Math.Max(0, minX - margin);
    int xMax = Math.Min(imageWidth, maxX + margin);
    int yMin = Math.Max(0, minY - margin);
    int yMax = Math.Min(imageHeight, maxY + margin);

    mergedList.Add(new[] { xMin, xMax, yMin, yMax });
}
```

**Step 2.2:** Passare dimensioni immagine attraverso la call chain

```csharp
// GroupTextBoxFlat signature update
public static (List<int[]> horizontalList, List<float[]> freeList) GroupTextBoxFlat(
    List<float[]> polys,
    float slopeThreshold,
    float ycenterThreshold,
    float heightThreshold,
    float widthThreshold,
    float addMargin,
    bool sortOutput,
    int imageWidth,    // ← NUOVO
    int imageHeight)   // ← NUOVO
```

---

### FASE 3: Test e Validazione ⭐⭐ (ALTA PRIORITÀ)

**Step 3.1:** Test Detection RAW

```bash
# Python
python ocr_process.py --dataset dataset/base --mode detect

# C# (dopo fix Fase 1)
dotnet run --project EasyOcrNet.Cli detect --image dataset/base/HAL.2015.page_42.pdf_125176.png

# Confronto
# Python: 17 bbox
# C#:     17 bbox ← Target!
```

**Step 3.2:** Test Full OCR Pipeline

```bash
# Python
python ocr_process.py --dataset dataset/base --mode text

# C# (dopo fix Fase 1 + 2)
dotnet run --project EasyOcrNet.Cli ocr --image dataset/base/HAL.2015.page_42.pdf_125176.png --compare dataset/base/HAL.2015.page_42.pdf_125176.png.ocr.python.txt

# Target:
# - Bbox Accuracy: 100% (10/10 match)
# - Text Accuracy: 100% (10/10 match)
```

**Step 3.3:** Creare Unit Tests

```csharp
// EasyOcrNet.Tests/DetectionTests.cs
[Fact]
public async Task Detection_ShouldMatch_PythonRawOutput()
{
    // Arrange
    var image = SKBitmap.Decode("dataset/base/HAL.2015.page_42.pdf_125176.png");
    var detector = new CraftDetector("models/cpu/detection.onnx", new OcrConfig());

    // Act
    var detections = await detector.DetectAsync(image);

    // Assert
    Assert.Equal(17, detections.Count); // Python RAW count
}

[Fact]
public async Task OcrPipeline_ShouldMatch_PythonGroupedOutput()
{
    // Arrange
    var image = SKBitmap.Decode("dataset/base/HAL.2015.page_42.pdf_125176.png");
    var engine = new OcrEngine("models/cpu/detection.onnx", "models/cpu/english_g2_rec.onnx", "en", new OcrConfig());

    // Act
    var results = await engine.ProcessImageAsync(image);

    // Assert
    Assert.Equal(10, results.Count); // Python grouped count
}
```

---

### FASE 4: Verifiche Finali ⭐ (MEDIA PRIORITÀ)

**Step 4.1:** Test su multiple immagini
```bash
# Test su tutti i dataset
python ocr_process.py --dataset dataset/base --mode text
# C# equivalente per ogni immagine
```

**Step 4.2:** Calcolare metriche finali
- Bbox Accuracy: >98%
- Text Accuracy: >95%
- Performance: C# deve rimanere più veloce di Python

---

## 📊 METRICHE DI SUCCESSO

| Metrica | Stato Attuale | Target | Priorità |
|---------|---------------|--------|----------|
| Detection RAW count | ❌ 13 (con grouping) | ✅ 17 (senza grouping) | MASSIMA |
| OCR Grouped count | ❌ 13 | ✅ 10 | ALTA |
| Bbox Accuracy | ❌ 0% | ✅ 100% | ALTA |
| Text Accuracy | ❌ 0% | ✅ 100% | ALTA |
| Coordinate Negative | ❌ 2/13 (15%) | ✅ 0/10 (0%) | ALTA |
| Overall Similarity | ⚠️ 65.3% | ✅ 100% | MASSIMA |

---

## ⏱️ TIMELINE STIMATO

**Fase 1 (Refactoring architettura):** 2-3 ore
**Fase 2 (Fix coordinate):** 30 minuti
**Fase 3 (Testing):** 1 ora
**Fase 4 (Validazione):** 30 minuti

**TOTALE:** ~4-5 ore di lavoro

---

## 🚨 RISCHI E MITIGAZIONI

### Rischio 1: Breaking Changes
**Mitigazione:** Creare branch `fix-parity` prima di iniziare

### Rischio 2: Altri bug nascosti
**Mitigazione:** Test incrementale dopo ogni fase

### Rischio 3: Performance degradation
**Mitigazione:** Benchmark prima/dopo ogni modifica

---

## 📝 CHECKLIST

- [ ] **FASE 1.1:** Rimuovere grouping da CraftDetector
- [ ] **FASE 1.2:** Creare OcrEngine.cs
- [ ] **FASE 1.3:** Aggiornare CLI per usare OcrEngine
- [ ] **FASE 2.1:** Fix coordinate negative con clamping
- [ ] **FASE 2.2:** Aggiornare signature GroupTextBoxFlat
- [ ] **FASE 3.1:** Test detection RAW (target: 17 bbox)
- [ ] **FASE 3.2:** Test OCR pipeline (target: 10 bbox)
- [ ] **FASE 3.3:** Creare unit tests
- [ ] **FASE 4.1:** Test su multiple immagini
- [ ] **FASE 4.2:** Calcolare metriche finali
- [ ] **COMMIT:** Parità 100% raggiunta!

---

**Priorità di esecuzione:**
1. ⭐⭐⭐ FASE 1 (risolve il 90% dei problemi)
2. ⭐⭐ FASE 2 (risolve coordinate negative)
3. ⭐⭐ FASE 3 (valida le fix)
4. ⭐ FASE 4 (validazione finale)

---

*Piano creato: 2025-10-25*
*Ultimo aggiornamento: 2025-10-25*
