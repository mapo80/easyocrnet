# EasyOcrNet

**EasyOcrNet** è un port .NET di [EasyOCR](https://github.com/JaidedAI/EasyOCR) che fornisce capacità di riconoscimento ottico dei caratteri (OCR) senza dipendenze da PyTorch. Il progetto utilizza modelli ONNX per un'esecuzione leggera e ad alte prestazioni.

## Caratteristiche Principali

- **Supporto Multi-Lingua**: Oltre 100 lingue supportate attraverso charset configurabili
- **Backend Multipli**: Supporto per ONNX Runtime e OpenVINO per un'inferenza ottimizzata
- **Lightweight**: ~272 MB contro ~1.52 GB della versione Python originale
- **Cross-Platform**: Windows, Linux e macOS supportati
- **Nessuna Dipendenza da PyTorch**: Utilizza modelli ONNX per eliminare le pesanti dipendenze (~1-2GB)
- **Avvio Rapido**: Tempi di startup e esecuzione più veloci rispetto alla versione originale

## Architettura

Il progetto è organizzato con un'architettura modulare multi-backend:

```
EasyOcrNet/
├── EasyOcrNet/              # Libreria core
│   ├── EasyOcr.cs          # API principale
│   ├── Backends.cs         # Astrazioni backend (ONNX, OpenVINO)
│   └── Charset.cs          # Gestione charset per 100+ lingue
├── EasyOcrNet.Tests/       # Test xUnit
├── ExampleExtractor/       # Applicazione di esempio
├── tools/                  # Script Python per gestione modelli
├── models/                 # Modelli ONNX e OpenVINO (git-ignored)
├── character/              # File di mappatura caratteri per ogni lingua
└── examples/               # Immagini di esempio per test
```

## Requisiti

- **.NET 8.0** o superiore
- Python 3.x (per gli script di download e conversione modelli)

### Dipendenze NuGet

| Pacchetto | Versione | Scopo |
|-----------|---------|-------|
| Microsoft.ML.OnnxRuntime | 1.18.0 | Inferenza modelli ONNX |
| OpenVINO.CSharp.API | 2025.0.0.1 | Inferenza modelli OpenVINO |
| SkiaSharp | 2.88.3 | Elaborazione immagini |
| SkiaSharp.NativeAssets.Linux | 2.88.3 | Supporto native Linux |
| System.Numerics.Tensors | 8.0.0 | Operazioni su tensori |

## Installazione

### 1. Clone del Repository

```bash
git clone https://github.com/tuouser/easyocrnet.git
cd easyocrnet
```

### 2. Download dei Modelli ONNX

I modelli ONNX derivano da [TorchfreeEasyOCR](https://github.com/SeldonHZ/TorchfreeEasyOCR).

Scarica i modelli necessari utilizzando lo script Python:

```bash
python tools/download_torchfree_models.py
```

Questo creerà i file `models/cpu/*.onnx` includendo:
- `detection.onnx` (83 MB) - Modello di detection del testo
- `english_g2_rec.onnx` - Recognizer per inglese
- `latin_g2_rec.onnx` - Recognizer per lingue latine (40+ lingue europee)
- `japanese_g2_rec.onnx` - Recognizer per giapponese
- `korean_g2_rec.onnx` - Recognizer per coreano
- `zh_sim_g2_rec.onnx` - Recognizer per cinese semplificato
- `thai_g1_rec.onnx` - Recognizer per thailandese

I file vengono verificati tramite checksum MD5 durante il download.

### 3. (Opzionale) Conversione a OpenVINO

Per utilizzare il backend OpenVINO ottimizzato per hardware Intel:

```bash
python tools/convert_to_openvino.py
```

Questo convertirà i modelli ONNX nel formato OpenVINO IR (`*.xml` + `*.bin`) sotto `models/openvino/`.

### 4. Build del Progetto

```bash
dotnet build EasyOcrNet.sln
```

## Utilizzo

### Esempio Base

```csharp
using EasyOcrNet;
using SkiaSharp;

// Carica un'immagine
using var image = SKBitmap.Decode("path/to/image.jpg");

// Crea l'istanza EasyOcr (backend ONNX, lingua inglese)
using var ocr = new EasyOcr(
    modelDirectory: "models/cpu",
    charset: Charset.en,
    backend: InferenceBackend.Onnx
);

// Esegui il riconoscimento OCR
var results = ocr.Read(image);

// Stampa i risultati
foreach (var result in results)
{
    Console.WriteLine($"Testo: {result.Text}");
    Console.WriteLine($"BBox: {result.BoundingBox}");
}
```

### Utilizzo con Backend OpenVINO

```csharp
using var ocr = new EasyOcr(
    modelDirectory: "models/openvino",
    charset: Charset.en,
    backend: InferenceBackend.OpenVino,
    device: "CPU"  // Opzioni: CPU, GPU, MYRIAD, etc.
);

var results = ocr.Read(image);
```

### Supporto Multi-Lingua

```csharp
// Italiano (usa charset latino)
using var ocrIt = new EasyOcr("models/cpu", Charset.it);

// Giapponese
using var ocrJa = new EasyOcr("models/cpu", Charset.ja);

// Coreano
using var ocrKo = new EasyOcr("models/cpu", Charset.ko);

// Cinese semplificato
using var ocrZh = new EasyOcr("models/cpu", Charset.ch_sim);

// Thailandese
using var ocrTh = new EasyOcr("models/cpu", Charset.th);
```

## Lingue Supportate

### Charset Latino (latin_g2_rec)

Il modello latino supporta **40+ lingue europee**:

| Codice | Lingua | Codice | Lingua |
|--------|--------|--------|--------|
| `af` | Afrikaans | `nl` | Olandese |
| `az` | Azero | `no` | Norvegese |
| `bs` | Bosniaco | `oc` | Occitano |
| `cs` | Ceco | `pl` | Polacco |
| `cy` | Gallese | `pt` | Portoghese |
| `da` | Danese | `ro` | Rumeno |
| `de` | Tedesco | `sk` | Slovacco |
| `en` | Inglese | `sl` | Sloveno |
| `es` | Spagnolo | `sq` | Albanese |
| `et` | Estone | `sv` | Svedese |
| `fr` | Francese | `sw` | Swahili |
| `hr` | Croato | `tl` | Tagalog |
| `hu` | Ungherese | `tr` | Turco |
| `id` | Indonesiano | `uz` | Uzbeko |
| `is` | Islandese | `vi` | Vietnamita |
| `it` | Italiano | `rs_latin` | Serbo (latino) |

### Charset Asiatici

| Modello | Charset | Lingue |
|---------|---------|--------|
| `english_g2_rec` | `en` | Inglese |
| `japanese_g2_rec` | `ja` | Giapponese |
| `korean_g2_rec` | `ko` | Coreano |
| `zh_sim_g2_rec` | `ch_sim` | Cinese semplificato |
| `thai_g1_rec` | `th` | Thailandese |

Oltre 100 lingue sono supportate tramite l'enum `Charset`. Vedi [Charset.cs](EasyOcrNet/Charset.cs) per l'elenco completo.

## Pipeline di Elaborazione

```
Immagine Input
    ↓
Resize (800x608)
    ↓
Normalizzazione (ImageNet mean/std)
    ↓
Conversione a Tensor
    ↓
[Backend ONNX o OpenVINO]
    ↓
Detection Model → BBox Extraction (threshold 0.3)
    ↓
Crop Regioni + Normalizzazione
    ↓
Recognition Model
    ↓
Character Decoding (argmax + deduplicazione)
    ↓
OcrResult (Text, BoundingBox)
```

## API Reference

### Classe `EasyOcr`

#### Constructor

```csharp
public EasyOcr(
    string modelDirectory,
    Charset charset = Charset.en,
    InferenceBackend backend = InferenceBackend.Onnx,
    string device = "CPU"
)
```

**Parametri:**
- `modelDirectory`: Percorso alla directory contenente i modelli ONNX o OpenVINO
- `charset`: Lingua/charset da utilizzare (default: `Charset.en`)
- `backend`: Backend di inferenza (`InferenceBackend.Onnx` o `InferenceBackend.OpenVino`)
- `device`: Dispositivo per OpenVINO (default: `"CPU"`, opzioni: `"GPU"`, `"MYRIAD"`, etc.)

#### Metodo `Read`

```csharp
public IEnumerable<OcrResult> Read(SKBitmap image)
```

Esegue l'OCR sull'immagine fornita.

**Ritorna:** Una collezione di `OcrResult` contenenti il testo riconosciuto e il bounding box.

### Record `OcrResult`

```csharp
public record OcrResult(string Text, SKRect BoundingBox);
```

- `Text`: Il testo riconosciuto
- `BoundingBox`: Il rettangolo che delimita la regione di testo (`SKRect`)

### Enum `InferenceBackend`

```csharp
public enum InferenceBackend
{
    Onnx,
    OpenVino
}
```

### Enum `Charset`

L'enum `Charset` contiene 106+ identificatori di lingua. Esempi:

```csharp
Charset.en      // Inglese
Charset.it      // Italiano
Charset.fr      // Francese
Charset.de      // Tedesco
Charset.es      // Spagnolo
Charset.ja      // Giapponese
Charset.ko      // Coreano
Charset.ch_sim  // Cinese semplificato
Charset.th      // Thailandese
// ... e molti altri
```

## Confronto con EasyOCR Originale

| Caratteristica | EasyOCR (Python) | EasyOcrNet (.NET) |
|---------------|------------------|-------------------|
| **Dimensione Pacchetto** | ~1.52 GB | ~272 MB |
| **Dipendenze** | PyTorch (~1-2 GB) | ONNX Runtime |
| **Linguaggio** | Python | C# (.NET 8.0) |
| **Backend** | PyTorch | ONNX, OpenVINO |
| **Startup** | Più lento | Più veloce |
| **Deployment** | Pesante | Leggero |
| **Multi-lingua** | 80+ lingue | 100+ lingue |

## Testing

Il progetto include test xUnit completi per verificare la correttezza del riconoscimento.

### Esecuzione Test

```bash
cd EasyOcrNet.Tests
dotnet test
```

### Esempi di Test

I test utilizzano immagini di esempio nella directory [examples/](examples/) che coprono:
- Inglese
- Francese
- Giapponese
- Coreano
- Cinese
- Thailandese

I risultati vengono confrontati con baseline Python per validare la correttezza:
- `.txt` - Risultati C# ONNX
- `.python.txt` - Risultati Python baseline
- `.diff.txt` - Differenze in formato unified diff

### Code Coverage

Il progetto utilizza Coverlet per la misurazione della copertura del codice:

```bash
dotnet test /p:CollectCoverage=true
```

## Script di Supporto

### `tools/download_torchfree_models.py`

Scarica i modelli ONNX da TorchfreeEasyOCR con verifica MD5.

```bash
python tools/download_torchfree_models.py
```

### `tools/convert_to_openvino.py`

Converte i modelli ONNX nel formato OpenVINO IR.

```bash
python tools/convert_to_openvino.py
```

### `run_onnx_examples.py`

Esegue OCR su batch di immagini di esempio e genera report JSON.

```bash
python run_onnx_examples.py
```

### `onnx_compare.py`

Confronta i risultati tra C# ONNX e baseline Python, generando report diff.

```bash
python onnx_compare.py
```

### `ocr_process.py`

Script unificato per processare immagini con OCR. Supporta generazione di file di testo, visualizzazioni con bounding box, e report JSON.

**Prerequisiti:**
```bash
pip install opencv-python onnxruntime easyocr
```

**Modalità disponibili:**
- `text`: Genera file `.ocr.python.txt` con risultati OCR
- `visualize`: Genera immagini `.ocr.bbox.png` con bounding box colorati
- `all`: Genera entrambi (default)

**Utilizzo base:**
```bash
# Genera file di testo + visualizzazioni (default)
python ocr_process.py

# Solo file di testo
python ocr_process.py --mode text

# Solo visualizzazioni
python ocr_process.py --mode visualize

# Con report JSON
python ocr_process.py --json results.json

# Lingua italiana
python ocr_process.py --lang it --dataset /path/to/images
```

**Opzioni:**
- `--dataset`: Directory immagini (default: `dataset/base`)
- `--lang`: Codice lingua (default: `en`)
- `--mode`: `text` | `visualize` | `all`
- `--json`: Salva report JSON
- `--no-text`: Non disegna etichette sui bbox
- `--thickness`: Spessore linee bbox (default: `2`)
- `--scale`: Fattore di scala per immagini bbox (es: `2.0` = 2x, `3.0` = 3x)
- `--overwrite`: Sovrascrive file esistenti
- `--extensions`: Estensioni immagini (default: `.png,.jpg,.jpeg`)

**Output generati:**

1. **File di testo** (`.ocr.python.txt`):
```
(0,15) (37,15) (37,27) (0,27) | Oil price | 0.8614
(42,16) (72,16) (72,24) (42,24) | AW | 0.0191
```

2. **Immagini con bbox** (`.ocr.bbox.png`):
- 🟢 Verde: Alta confidenza (>= 0.7)
- 🟡 Giallo: Media confidenza (0.4-0.7)
- 🔴 Rosso: Bassa confidenza (< 0.4)

## Modelli ONNX

I modelli ONNX presenti nella directory [models/](models/) derivano dal progetto:

**[TorchfreeEasyOCR](https://github.com/SeldonHZ/TorchfreeEasyOCR)**

TorchfreeEasyOCR è un port di EasyOCR che elimina le dipendenze da PyTorch convertendo i modelli in formato ONNX. Questo consente:
- Pacchetto più leggero (no PyTorch)
- Inferenza più veloce
- Migliore portabilità
- Supporto per runtime alternativi (ONNX Runtime, OpenVINO)

### Modelli Disponibili

| Modello | Dimensione | Scopo |
|---------|-----------|-------|
| `detection.onnx` | 83 MB | Detection delle regioni di testo (CRAFT) |
| `english_g2_rec.onnx` | ~15 MB | Recognition inglese |
| `latin_g2_rec.onnx` | ~30 MB | Recognition lingue latine |
| `japanese_g2_rec.onnx` | ~45 MB | Recognition giapponese |
| `korean_g2_rec.onnx` | ~30 MB | Recognition coreano |
| `zh_sim_g2_rec.onnx` | ~215 MB | Recognition cinese semplificato |
| `thai_g1_rec.onnx` | ~20 MB | Recognition thailandese |

### Note Tecniche

- **Detection Model (CRAFT)**: Character Region Awareness For Text detection
  - Input: `[1, 3, 608, 800]` (batch, canali, altezza, larghezza)
  - Output: Score map per le regioni di testo
  - Threshold: 0.3 per filtrare detection a bassa confidenza

- **Recognition Models**:
  - Input: `[1, 1, 64, 1000]` (immagine grayscale normalizzata)
  - Output: Sequenza di probabilità per carattere
  - Decoding: argmax + deduplicazione caratteri consecutivi

## Performance

### Dimensioni Pacchetto

- **Libreria Core**: Minimale (~100 KB)
- **Con Dipendenze NuGet**: ~50 MB
- **Modelli ONNX**: ~375 MB totali (scaricabili on-demand)
- **Modelli OpenVINO**: Variabile (generalmente più compatti)

### Piattaforme Supportate

- **Windows**: x86, x64, ARM
- **Linux**: x64, ARM (via SkiaSharp.NativeAssets.Linux)
- **macOS**: x64, ARM64 (Apple Silicon)

### Caratteristiche di Performance

- **Solo CPU**: Percorso di esecuzione ottimizzato per CPU
- **Detection**: Single bbox per immagine (semplificato)
- **Recognition**: Decoding carattere per carattere
- **Accelerazione GPU**: Disponibile tramite backend OpenVINO

## Sviluppo Recente

Commit recenti mostrano sviluppo attivo:
- **c9d89ac**: Fixes
- **7c6931f**: Aggiornamento dipendenze NuGet
- **0e54f12**: Supporto per output OCR con confidenza
- **74ba06f**: Implementazione post-processing multi-bbox CRAFT
- **dcf3009**: Aggiunta primitive di elaborazione immagini

## Struttura Progetto

```
EasyOcrNet.sln                    # Solution Visual Studio 2022+
├── EasyOcrNet/                   # Libreria principale (net8.0)
│   ├── EasyOcr.cs               # API pubblica principale (791 righe)
│   ├── Backends.cs              # Backend ONNX e OpenVINO
│   ├── Charset.cs               # Enum con 106 lingue
│   └── EasyOcrNet.csproj        # File di progetto
├── EasyOcrNet.Tests/             # Test xUnit (net8.0)
│   ├── UnitTest1.cs             # Test di riconoscimento
│   └── EasyOcrNet.Tests.csproj
├── ExampleExtractor/             # App console esempio (net9.0)
├── models/                       # Modelli (git-ignored)
│   ├── cpu/                     # Modelli ONNX
│   └── openvino/                # Modelli OpenVINO IR
├── character/                    # 106 file di mappatura caratteri
├── examples/                     # 57 immagini di test
├── tools/                        # Script Python di supporto
└── README.md                     # Questo file
```

## Contribuire

### Setup Ambiente di Sviluppo

1. Installa .NET 8.0 SDK o superiore
2. Installa Python 3.x
3. Clone del repository
4. Download dei modelli con `python tools/download_torchfree_models.py`
5. Build con `dotnet build`
6. Esegui test con `dotnet test`

### Linee Guida

- Utilizza C# 12 con nullable reference types
- Segui le convenzioni .NET standard
- Aggiungi test per nuove funzionalità
- Mantieni la compatibilità con .NET 8.0+

## Licenza

Da verificare - controllare i file di licenza dei progetti upstream:
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [TorchfreeEasyOCR](https://github.com/SeldonHZ/TorchfreeEasyOCR)

## Crediti e Attribuzioni

Questo progetto si basa sul lavoro di:

- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** by JaidedAI - OCR originale con supporto 80+ lingue
- **[TorchfreeEasyOCR](https://github.com/SeldonHZ/TorchfreeEasyOCR)** by SeldonHZ - Conversione modelli a ONNX
- **Microsoft.ML.OnnxRuntime** - Runtime di inferenza ONNX
- **OpenVINO** - Toolkit di inferenza ottimizzata Intel
- **SkiaSharp** - Libreria di elaborazione immagini cross-platform

## Risorse

- [Documentazione EasyOCR originale](https://www.jaided.ai/easyocr/documentation/)
- [TorchfreeEasyOCR Repository](https://github.com/SeldonHZ/TorchfreeEasyOCR)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [OpenVINO Documentation](https://docs.openvino.ai/)
- [CRAFT Paper](https://arxiv.org/abs/1904.01941) - Character Region Awareness For Text detection

## Contatti

Per bug, richieste di funzionalità o domande, apri una issue su GitHub.

---

**EasyOcrNet** - OCR .NET veloce, leggero e multi-lingua senza dipendenze PyTorch
