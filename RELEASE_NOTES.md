# EasyOcrNet v1.0.0 - Initial Release

🎉 **Prima release ufficiale di EasyOcrNet!**

EasyOcrNet è un port .NET di EasyOCR che fornisce capacità di riconoscimento ottico dei caratteri (OCR) senza dipendenze da PyTorch. Utilizza modelli ONNX per un'esecuzione leggera e ad alte prestazioni.

## 📦 Pacchetto NuGet

Questa release include il pacchetto NuGet **EasyOcrNet v1.0.0** che contiene:

### Modelli ONNX Inclusi
- **detection.onnx** (79 MB) - Modello CRAFT per la detection del testo
- **latin_g2_rec.onnx** (15 MB) - Modello di recognition per lingue latine (italiano, inglese, spagnolo, francese, tedesco, etc.)
- **english_g2_rec.onnx** (14 MB) - Modello di recognition ottimizzato per inglese

### Character Sets
- **en_charset.txt** - Set di caratteri inglese
- **it_charset.txt** - Set di caratteri italiano
- **latin_char.txt** - Set di caratteri latino (fallback)

**Dimensione totale pacchetto**: ~201 MB (include modelli duplicati per compatibilità)

## 🚀 Caratteristiche Principali

### Performance
- **~6.4x più veloce** dell'implementazione Python originale
- Tempo medio: **~4.3 secondi/immagine** (test su dataset italiano 1024x1024)
- Ottimizzato per CPU con ONNX Runtime

### Supporto Multi-Lingua
- Italiano, Inglese, Spagnolo, Francese, Tedesco
- Oltre 40 lingue europee tramite modello latino
- Supporto per lingue asiatiche (con modelli aggiuntivi)

### Architettura Modulare
- **Detection**: CRAFT (Character Region Awareness For Text detection)
- **Recognition**: CRNN (Convolutional Recurrent Neural Network)
- **Post-Processing**: Correzioni specifiche per lingua (es. apostrofi italiani)
- **Pipeline completo**: Detection → Grouping → Recognition → Post-Processing

## 📊 Risultati Benchmark

Test su **4 immagini italiane** (dataset di test):

```
Total images processed: 4
Total detections: 99
Total time: 17135ms
Average time per image: 4283ms

Dettagli per immagine:
  doc-it-04.png (2480x3508): 43 detections in 10107ms
  doc-it-01.png (1024x1024): 16 detections in 2162ms
  doc-it-02.png (1024x1024): 14 detections in 2018ms
  doc-it-03.png (1024x1024): 26 detections in 2700ms
```

**Confronto con Python**:
- Python (ONNX): 19.23s per immagine
- C# (ONNX Runtime): 2.99s per immagine
- **Speedup: 6.43x**

## 🛠️ Installazione

### Opzione 1: NuGet Package (Consigliato)

```bash
dotnet add package EasyOcrNet --version 1.0.0 --source <path-to-nupkg>
```

### Opzione 2: Download dalla Release

1. Scarica `EasyOcrNet.1.0.0.nupkg` da questa release
2. Aggiungi una sorgente NuGet locale:
   ```bash
   dotnet nuget add source /path/to/nupkgs --name local
   ```
3. Installa il pacchetto:
   ```bash
   dotnet add package EasyOcrNet --version 1.0.0
   ```

## 💻 Utilizzo Base

```csharp
using EasyOcrNet;
using SkiaSharp;

// Carica un'immagine
using var bitmap = SKBitmap.Decode("path/to/image.jpg");

// Crea il motore OCR
using var engine = new OcrEngine(
    detectorPath: "models/detection.onnx",
    recognizerPath: "models/latin_g2_rec.onnx",
    language: "it",
    charsetDirectory: "character"
);

// Esegui il riconoscimento OCR
var results = await engine.ProcessImageAsync(bitmap);

// Stampa i risultati
foreach (var result in results)
{
    Console.WriteLine($"Testo: {result.Text}");
    Console.WriteLine($"Confidence: {result.Confidence:F4}");
}
```

## 📋 Requisiti

- **.NET 8.0** o superiore
- **Dipendenze NuGet**:
  - Microsoft.ML.OnnxRuntime (1.18.0)
  - SkiaSharp (2.88.3)
  - SkiaSharp.NativeAssets.Linux (2.88.3) - per Linux

## 🌐 Piattaforme Supportate

- **Windows**: x86, x64, ARM
- **Linux**: x64, ARM
- **macOS**: x64, ARM64 (Apple Silicon)

## 📚 Documentazione

- [README principale](https://github.com/mapo80/easyocrnet/blob/main/README.md)
- [Guida CLI](https://github.com/mapo80/easyocrnet/blob/main/EasyOcrNet.CliNuget/README.md)
- [Documentazione NuGet](https://github.com/mapo80/easyocrnet/blob/main/EasyOcrNet.Nuget/README.md)

## 🎯 Progetti Inclusi

- **EasyOcrNet** - Libreria core
- **EasyOcrNet.Nuget** - Sorgente del pacchetto NuGet
- **EasyOcrNet.CliNuget** - Applicazione console di esempio
- **EasyOcrNet.Tests** - Suite di test xUnit

## 🙏 Crediti

Questo progetto si basa sul lavoro di:

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) by JaidedAI - OCR originale
- [TorchfreeEasyOCR](https://github.com/SeldonHZ/TorchfreeEasyOCR) by SeldonHZ - Conversione modelli a ONNX
- Microsoft.ML.OnnxRuntime - Runtime di inferenza ONNX
- SkiaSharp - Libreria di elaborazione immagini

## 📝 Note sulla Release

- Prima release stabile di EasyOcrNet
- Supporto completo per italiano con post-processing
- Modelli ONNX pre-compilati inclusi nel pacchetto
- Performance ottimizzate per CPU
- Architettura modulare e estensibile

## 🐛 Known Issues

- Il pacchetto NuGet è grande (~201 MB) a causa dei modelli ONNX inclusi
- ProjectReference ha priorità su PackageReference (vedere documentazione)

## 🔜 Future Improvements

- Supporto GPU (CUDA/DirectML)
- Batch processing ottimizzato
- Modelli quantizzati per ridurre dimensioni
- Supporto per più lingue asiatiche
- Cache dei modelli ONNX

---

**Download**: Scarica `EasyOcrNet.1.0.0.nupkg` dagli assets di questa release.

**Licenza**: MIT (vedere LICENSE file nel repository)
