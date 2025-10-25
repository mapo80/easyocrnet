# Piano di Ottimizzazione Performance C# OCR

**Obiettivo**: Ridurre i tempi di esecuzione del 50% o più (da 2.99s a ≤1.50s)

**Baseline Attuale**: 2.99s per immagine 1024x1024px (Italian OCR, Release build)

---

## Analisi Profiling Iniziale

Prima di ottimizzare, dobbiamo misurare dove viene speso il tempo:

### 1. Profiling Dettagliato per Fase

**Action**: Aggiungere timing dettagliato per ogni fase

```
Pipeline OCR:
├── Detection (CRAFT)
│   ├── Image Load & Preprocessing      ~ ?ms
│   ├── ONNX Inference (Detection)      ~ ?ms
│   └── Post-processing (getDetBoxes)   ~ ?ms
├── Grouping                            ~ ?ms
├── Recognition (CRNN)
│   ├── Crop Extraction (16 crops)      ~ ?ms
│   ├── Preprocessing per crop          ~ ?ms
│   ├── ONNX Inference (Recognition)    ~ ?ms (× 16)
│   └── CTC Decoding                    ~ ?ms
└── Post-processing (Italian)           ~ ?ms
```

**Deliverable**: Script di profiling che mostra ms per ogni fase

---

## Piano di Ottimizzazione (Priorità)

### 🔴 PRIORITÀ ALTA - Quick Wins (Target: -40%)

#### 1. Batch Recognition Inference ⚡️
**Impatto Stimato**: -30-40% tempo totale
**Difficoltà**: Media

**Problema Attuale**:
- 16 crop riconosciute sequenzialmente
- 16 chiamate separate a ONNX session
- Ogni chiamata ha overhead di setup

**Soluzione**:
```csharp
// PRIMA (sequenziale):
foreach (var crop in crops)
{
    var result = _session.Run(inputs);  // 16 chiamate separate
}

// DOPO (batch):
var batchInput = CreateBatchTensor(crops);  // Shape: [16, 1, 64, 896]
var batchResults = _session.Run(batchInput); // 1 chiamata batch
```

**Implementazione**:
1. Modificare `CrnnRecognizer.cs` per supportare batch inference
2. Creare `CreateBatchTensor()` che stack tutti i crops
3. Modificare `DecodeText()` per processare batch output
4. Gestire casi con crop di dimensioni diverse (padding dinamico)

**File da Modificare**:
- `EasyOcrNet/Recognition/CrnnRecognizer.cs`
- `EasyOcrNet/Recognition/RecognitionUtils.cs`

---

#### 2. ONNX Session Optimization ⚡️
**Impatto Stimato**: -5-10% tempo totale
**Difficoltà**: Facile

**Problema Attuale**:
- Sessioni ONNX create ogni volta
- No graph optimization level specificato
- No intra/inter op threads configurati

**Soluzione**:
```csharp
var sessionOptions = new SessionOptions
{
    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
    IntraOpNumThreads = Environment.ProcessorCount,     // Parallelizza ops
    InterOpNumThreads = 1,                              // Sequential graphs
    ExecutionMode = ExecutionMode.ORT_PARALLEL,         // Parallel execution
    EnableMemoryPattern = true,                         // Reuse memory
    EnableCpuMemArena = true                           // Memory arena
};
```

**Implementazione**:
1. Aggiungere configurazione avanzata in `CraftDetector.cs`
2. Aggiungere configurazione avanzata in `CrnnRecognizer.cs`
3. Test A/B con diverse configurazioni thread

**File da Modificare**:
- `EasyOcrNet/Detection/CraftDetector.cs` (linea ~50)
- `EasyOcrNet/Recognition/CrnnRecognizer.cs` (linea ~44)

---

#### 3. SkiaSharp Image Processing Optimization ⚡️
**Impatto Stimato**: -5-10% tempo totale
**Difficoltà**: Facile

**Problema Attuale**:
- Crop extraction con SkiaSharp può essere lento
- Conversioni colore multiple
- Allocazioni memory inutili

**Soluzione**:
```csharp
// Usare Span<T> per evitare allocazioni
// Pre-allocare buffer per crop
// Riusare SKBitmap invece di creare nuovi
// Usare SKImage.FromPixelCopy invece di SKBitmap quando possibile
```

**Implementazione**:
1. Profiling SkiaSharp operations
2. Object pooling per SKBitmap
3. Ridurre conversioni colore
4. Usare `Span<byte>` per operazioni raw

**File da Modificare**:
- `EasyOcrNet/Recognition/RecognitionUtils.cs` (PreprocessCrop)
- `EasyOcrNet/Utils/ImageUtils.cs` (se esistente)

---

### 🟡 PRIORITÀ MEDIA - Ottimizzazioni Strutturali (Target: -15%)

#### 4. Parallel Crop Preprocessing ⚡️
**Impatto Stimato**: -5-8% tempo totale
**Difficoltà**: Facile

**Problema Attuale**:
- Preprocessing di 16 crop fatto sequenzialmente
- CPU multi-core non utilizzato

**Soluzione**:
```csharp
// PRIMA:
foreach (var crop in crops)
{
    var preprocessed = RecognitionUtils.PreprocessCrop(crop);
}

// DOPO:
var preprocessedCrops = crops.AsParallel()
    .WithDegreeOfParallelism(Environment.ProcessorCount)
    .Select(crop => RecognitionUtils.PreprocessCrop(crop))
    .ToArray();
```

**Implementazione**:
1. Usare `Parallel.ForEach` o PLINQ
2. Assicurarsi thread-safety
3. Benchmark ottimale degree of parallelism

**File da Modificare**:
- `EasyOcrNet/Recognition/CrnnRecognizer.cs`

---

#### 5. Memory Pool & Object Reuse 🔄
**Impatto Stimato**: -3-5% tempo totale
**Difficoltà**: Media

**Problema Attuale**:
- Allocazione/deallocazione continua di tensori
- GC pressure durante inferenza
- Array allocations per ogni crop

**Soluzione**:
```csharp
// Usare ArrayPool<T>
private static readonly ArrayPool<float> _tensorPool = ArrayPool<float>.Shared;

// Rent instead of new
var tensorArray = _tensorPool.Rent(totalSize);
try
{
    // Use tensor
}
finally
{
    _tensorPool.Return(tensorArray);
}
```

**Implementazione**:
1. `ArrayPool<float>` per tensori ONNX
2. `ObjectPool<SKBitmap>` per crop images
3. Pre-allocare buffer comuni
4. Ridurre allocazioni in hot path

**File da Modificare**:
- `EasyOcrNet/Recognition/CrnnRecognizer.cs`
- `EasyOcrNet/Detection/CraftDetector.cs`

---

#### 6. Span<T> & stackalloc Optimization ⚡️
**Impatto Stimato**: -2-4% tempo totale
**Difficoltà**: Media

**Problema Attuale**:
- Array allocations per operazioni temporanee
- Boxing/unboxing in alcuni punti
- Copie di memoria non necessarie

**Soluzione**:
```csharp
// PRIMA:
float[] tempBuffer = new float[256];  // Heap allocation

// DOPO:
Span<float> tempBuffer = stackalloc float[256];  // Stack allocation
```

**Implementazione**:
1. Identificare allocazioni temporanee < 1KB
2. Usare `stackalloc` dove possibile
3. Passare `Span<T>` invece di array
4. Ridurre copie con `Memory<T>`

**File da Modificare**:
- `EasyOcrNet/Recognition/RecognitionUtils.cs`
- `EasyOcrNet/Detection/CraftUtils.cs`

---

### 🟢 PRIORITÀ BASSA - Ottimizzazioni Avanzate (Target: -10%)

#### 7. SIMD Vectorization 🚀
**Impatto Stimato**: -3-5% tempo totale
**Difficoltà**: Alta

**Problema Attuale**:
- Operazioni su array fatte elemento per elemento
- CPU SIMD capabilities non utilizzate

**Soluzione**:
```csharp
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// Normalizzazione vettorizzata
for (int i = 0; i < array.Length; i += Vector<float>.Count)
{
    var vec = new Vector<float>(array, i);
    vec = (vec - meanVec) / stdVec;
    vec.CopyTo(array, i);
}
```

**Implementazione**:
1. Identificare loops con operazioni matematiche
2. Usare `Vector<T>` per auto-vectorization
3. Usare intrinsics X86/ARM per max performance
4. Fallback per piattaforme senza SIMD

**File da Modificare**:
- `EasyOcrNet/Recognition/RecognitionUtils.cs` (normalizzazione)
- `EasyOcrNet/Detection/CraftUtils.cs` (operazioni su score maps)

---

#### 8. Lazy Loading & Model Caching 💾
**Impatto Stimato**: -2-3% tempo totale (su run successive)
**Difficoltà**: Media

**Problema Attuale**:
- Modelli ONNX caricati ogni volta
- Charset ricaricato per ogni istanza
- Session options ricreate

**Soluzione**:
```csharp
private static readonly ConcurrentDictionary<string, InferenceSession> _sessionCache
    = new();

public static InferenceSession GetOrCreateSession(string modelPath, SessionOptions options)
{
    return _sessionCache.GetOrAdd(modelPath, path =>
        new InferenceSession(path, options));
}
```

**Implementazione**:
1. Cache statica per sessioni ONNX
2. Lazy loading per charset
3. Singleton pattern per configurazioni
4. Thread-safe access

**File da Modificare**:
- `EasyOcrNet/OcrEngine.cs`
- `EasyOcrNet/Utils/CharsetLoader.cs`

---

#### 9. GPU Acceleration (Opzionale) 🎮
**Impatto Stimato**: -50-70% tempo totale (se GPU disponibile)
**Difficoltà**: Media

**Problema Attuale**:
- Solo CPU execution provider
- GPU capabilities non utilizzate

**Soluzione**:
```csharp
var sessionOptions = new SessionOptions();

// Try GPU first, fallback to CPU
if (Cuda.IsAvailable())
    sessionOptions.AppendExecutionProvider_CUDA(0);
else if (DirectML.IsAvailable())  // Windows
    sessionOptions.AppendExecutionProvider_DML(0);
else
    sessionOptions.AppendExecutionProvider_CPU();
```

**Implementazione**:
1. Detect GPU availability
2. Configure CUDA/DirectML execution provider
3. Fallback gracefully to CPU
4. Test on various hardware

**File da Modificare**:
- `EasyOcrNet/Detection/CraftDetector.cs`
- `EasyOcrNet/Recognition/CrnnRecognizer.cs`

---

#### 10. OpenVINO Backend (Alternative) ⚡️
**Impatto Stimato**: -20-30% tempo totale
**Difficoltà**: Alta

**Problema Attuale**:
- ONNX Runtime non sempre ottimale per Intel CPU

**Soluzione**:
- Usare OpenVINO execution provider
- Conversione modelli a formato OpenVINO IR
- Ottimizzazioni specifiche Intel

**Implementazione**:
1. Setup OpenVINO toolkit
2. Convert ONNX → OpenVINO IR
3. Implementare backend alternativo
4. A/B test vs ONNX Runtime

**File da Modificare**:
- Creare `EasyOcrNet/Backends/OpenVinoBackend.cs`
- Modificare `OcrEngine.cs` per backend selection

---

## Timeline di Implementazione

### Fase 1: Quick Wins (Settimana 1)
1. ✅ Profiling dettagliato per identificare hotspots
2. ⚡ Batch inference per recognition (Priority #1)
3. ⚡ ONNX session optimization (Priority #2)
4. ⚡ SkiaSharp optimization (Priority #3)

**Target**: 2.99s → ~1.80s (-40%)

### Fase 2: Ottimizzazioni Strutturali (Settimana 2)
5. ⚡ Parallel crop preprocessing (Priority #4)
6. 🔄 Memory pooling (Priority #5)
7. ⚡ Span<T> optimization (Priority #6)

**Target**: 1.80s → ~1.30s (-55% totale)

### Fase 3: Ottimizzazioni Avanzate (Opzionale)
8. 🚀 SIMD vectorization (Priority #7)
9. 💾 Model caching (Priority #8)
10. 🎮 GPU/OpenVINO (Priority #9-10)

**Target**: 1.30s → ~1.00s (-67% totale)

---

## Metodologia di Testing

Per ogni ottimizzazione:

1. **Benchmark PRIMA**: Eseguire `python run_benchmarks.py` e salvare risultati
2. **Implementare ottimizzazione**: Una alla volta
3. **Benchmark DOPO**: Eseguire benchmark e confrontare
4. **Verificare correttezza**: Controllare che output OCR sia identico
5. **Committare se miglioramento**: Solo se speedup > 2%
6. **Aggiornare storico**: Aggiornare README.md con nuovi risultati

### Script di Validazione

```bash
# 1. Benchmark corrente
python run_benchmarks.py

# 2. Implementa ottimizzazione

# 3. Rebuild Release
dotnet build -c Release

# 4. Benchmark post-ottimizzazione
python run_benchmarks.py

# 5. Verifica correttezza output
dotnet run --project EasyOcrNet.Cli -c Release -- ocr \
  --image dataset/it/doc-it-01.png \
  --compare dataset/it/doc-it-01.png.ocr.python.txt
```

---

## Metriche di Successo

| Metrica | Baseline | Target Fase 1 | Target Fase 2 | Target Finale |
|---------|----------|---------------|---------------|---------------|
| **Tempo Medio** | 2.99s | 1.80s | 1.30s | 1.00s |
| **Speedup vs Python** | 6.43x | 10.7x | 14.8x | 19.2x |
| **Riduzione %** | 0% | -40% | -55% | -67% |
| **Correttezza Output** | 100% | 100% | 100% | 100% |

---

## Rischi e Mitigazioni

### Rischio 1: Batch Inference cambia output
**Mitigazione**: Test approfonditi su tutti dataset, confronto byte-per-byte

### Rischio 2: Memory pooling introduce bugs
**Mitigazione**: Unit tests estensivi, valgrind/dotMemory profiling

### Rischio 3: SIMD non disponibile su tutte piattaforme
**Mitigazione**: Fallback automatico, feature detection runtime

### Rischio 4: Ottimizzazioni rendono codice non mantenibile
**Mitigazione**: Commenti dettagliati, keep simple path come fallback

---

## Prossimi Passi Immediati

1. **Creare script di profiling dettagliato** (`profile_ocr.py`)
2. **Implementare Batch Recognition** (Priority #1)
3. **Benchmark e validare risultati**
4. **Aggiornare README con nuovi risultati**
5. **Procedere con prossima ottimizzazione**

---

**Note**: Questo piano è iterativo. Dopo ogni ottimizzazione, rivalutare le priorità basandosi sui nuovi profiling results.
