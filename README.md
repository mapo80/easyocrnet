# EasyOcrNet

**EasyOcrNet** è un port .NET di [EasyOCR](https://github.com/JaidedAI/EasyOCR) che fornisce capacità di riconoscimento ottico dei caratteri (OCR) senza dipendenze da PyTorch. Il progetto utilizza modelli ONNX per un'esecuzione leggera e ad alte prestazioni.

## Caratteristiche Principali

- **Supporto Multi-Lingua**: Lingue supportate attraverso charset configurabili
- **Backend ONNX Runtime**: Inferenza ottimizzata tramite ONNX Runtime
- **Lightweight**: Dimensioni ridotte rispetto alla versione Python con PyTorch
- **Cross-Platform**: Windows, Linux e macOS supportati
- **Nessuna Dipendenza da PyTorch**: Utilizza modelli ONNX per eliminare le pesanti dipendenze
- **Performance**: ~6.4x più veloce della versione Python su test italiani

## Architettura

Il progetto è organizzato con un'architettura modulare multi-backend:

```
EasyOcrNet/
├── EasyOcrNet/              # Libreria core
│   ├── OcrEngine.cs        # Motore OCR principale
│   ├── Detection/          # Moduli di detection (CRAFT)
│   ├── Recognition/        # Moduli di recognition (CRNN)
│   ├── PostProcessing/     # Post-processing del testo
│   ├── Models/             # Modelli di dati
│   └── Utils/              # Utility (charset loader)
├── EasyOcrNet.Tests/       # Test xUnit
├── EasyOcrNet.Cli/         # Applicazione console di esempio
├── tools/                  # Script Python per gestione modelli
├── models/                 # Modelli ONNX (git-ignored)
├── character/              # File di mappatura caratteri per ogni lingua
└── dataset/                # Dataset di test
```

## Requisiti

- **.NET 8.0** o superiore
- Python 3.x (per gli script di download e conversione modelli)

### Dipendenze NuGet

| Pacchetto | Versione | Scopo |
|-----------|---------|-------|
| Microsoft.ML.OnnxRuntime | 1.18.0 | Inferenza modelli ONNX |
| SkiaSharp | 2.88.3 | Elaborazione immagini |
| SkiaSharp.NativeAssets.Linux | 2.88.3 | Supporto native Linux |

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

### 3. Build del Progetto

```bash
dotnet build EasyOcrNet.sln
```

## Utilizzo

### Esempio Base

```csharp
using EasyOcrNet;
using SkiaSharp;

// Carica un'immagine
using var bitmap = SKBitmap.Decode("path/to/image.jpg");

// Crea il motore OCR
using var engine = new OcrEngine(
    detectorPath: "models/cpu/detection.onnx",
    recognizerPath: "models/cpu/latin_g2_rec.onnx",
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
    Console.WriteLine($"BBox: {result.BoundingBox}");
}
```

### Supporto Multi-Lingua

```csharp
// Inglese
using var ocrEn = new OcrEngine(
    "models/cpu/detection.onnx",
    "models/cpu/english_g2_rec.onnx",
    "en"
);

// Giapponese
using var ocrJa = new OcrEngine(
    "models/cpu/detection.onnx",
    "models/cpu/japanese_g2_rec.onnx",
    "ja"
);
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

I charset sono caricati dai file nella directory [character/](character/).

## Performance

EasyOcrNet offre prestazioni significativamente superiori rispetto all'implementazione Python originale grazie all'ottimizzazione del runtime .NET e all'uso efficiente di ONNX Runtime.

### Benchmark: Python vs C# (Italian OCR)

Test eseguiti su **dataset italiano** (1024x1024px) con **6 iterazioni** (prima esclusa come warmup):

| Piattaforma | Tempo Medio | Min | Max | Note |
|-------------|-------------|-----|-----|------|
| **Python (ONNX)** | 19.23s | 18.83s | 19.72s | Implementazione di riferimento |
| **C# (ONNX Runtime)** | 2.99s | 2.88s | 3.20s | Release build, .NET 8.0 |
| **Speedup** | **6.43x** | | | C# è ~6.4x più veloce |

#### Dettagli del Test

- **Hardware**: MacBook (Darwin 23.0.0)
- **Configurazione**:
  - Python: CPUExecutionProvider, modelli ONNX standard
  - C#: ONNX Runtime 1.18.0, Release build con ottimizzazioni
- **Dataset**: 4 immagini italiane (1024x1024px)
- **Metodologia**: 6 esecuzioni per immagine, prima esclusa (warmup), media delle rimanenti 5

#### Pipeline OCR Completa

La pipeline include tutte le fasi:

1. **Detection** (CRAFT): Individuazione bounding boxes del testo
2. **Grouping**: Merging di box adiacenti sulla stessa linea
3. **Recognition** (CRNN): Riconoscimento caratteri con CTC decoder
4. **Post-Processing**: Fix apostrofi, accenti, parole composte (solo per italiano)

### Storico Ottimizzazioni C#

Questa sezione traccia il progresso delle ottimizzazioni implementate nel tempo.

#### Baseline v1.0 (2025-01-24)

| Immagine | Python | C# | Speedup |
|----------|--------|-----|---------|
| doc-it-01.png | 19.23s | 2.99s | **6.43x** |

**Note**: Prima misurazione baseline con implementazione completa di post-processing italiano.

#### Performance Future

Gli obiettivi di ottimizzazione includono:

- **Inferenza parallela**: Batch processing per recognition di crop multiple
- **OpenVINO backend**: Ulteriore accelerazione CPU (~20-30% atteso)
- **GPU support**: Utilizzo GPU tramite CUDA/DirectML providers
- **Cache modelli**: Riutilizzo sessioni ONNX tra chiamate

### Come Eseguire i Benchmark

Per replicare i benchmark sul tuo sistema:

```bash
# Esegui benchmark completo (Python + C#)
python run_benchmarks.py

# I risultati vengono salvati in:
# - benchmark_results/python.json
# - benchmark_results/csharp.json
```

I benchmark utilizzano:
- **6 iterazioni** per ogni immagine
- **Prima iterazione scartata** (warmup)
- **Media delle restanti 5** iterazioni

## API Reference

### Classe `OcrEngine`

```csharp
public class OcrEngine : IDisposable
{
    public OcrEngine(
        string detectorPath,
        string recognizerPath,
        string language,
        OcrConfig? config = null,
        string charsetDirectory = "character"
    )

    public async Task<List<OcrResult>> ProcessImageAsync(SKBitmap bitmap)
}
```

**Parametri:**
- `detectorPath`: Percorso al modello ONNX di detection (es. `detection.onnx`)
- `recognizerPath`: Percorso al modello ONNX di recognition (es. `latin_g2_rec.onnx`)
- `language`: Codice lingua (es. `"en"`, `"it"`, `"ja"`)
- `config`: Configurazione OCR opzionale (threshold, dimensioni minime, etc.)
- `charsetDirectory`: Directory contenente i file charset

### Record `OcrResult`

```csharp
public record OcrResult(
    BoundingBox BoundingBox,
    string Text,
    float Confidence
);
```

- `BoundingBox`: Coordinate del box di testo (4 punti)
- `Text`: Il testo riconosciuto
- `Confidence`: Confidenza del riconoscimento (0.0 - 1.0)

## Confronto con EasyOCR Originale

| Caratteristica | EasyOCR (Python) | EasyOcrNet (.NET) |
|---------------|------------------|-------------------|
| **Performance** | 19.23s (baseline) | 2.99s (~6.4x più veloce) |
| **Dipendenze** | PyTorch (~1-2 GB) | ONNX Runtime |
| **Linguaggio** | Python | C# (.NET 8.0) |
| **Backend** | PyTorch | ONNX Runtime |
| **Deployment** | Pesante | Leggero |
| **Multi-lingua** | 80+ lingue | Lingue supportate tramite modelli ONNX |

## Testing

Il progetto include test xUnit completi per verificare la correttezza del riconoscimento.

### Esecuzione Test

```bash
cd EasyOcrNet.Tests
dotnet test
```

### Esempi di Test

I test utilizzano immagini di esempio nella directory [dataset/](dataset/) che coprono diverse lingue.

I risultati vengono confrontati con baseline Python per validare la correttezza.

## Script di Supporto

### `tools/download_torchfree_models.py`

Scarica i modelli ONNX da TorchfreeEasyOCR con verifica MD5.

```bash
python tools/download_torchfree_models.py
```

### `tools/convert_to_openvino.py`

Converte i modelli ONNX nel formato OpenVINO IR (non attualmente utilizzato nel codice C#).

### `run_benchmarks.py`

Esegue benchmark completi confrontando Python e C#.

```bash
python run_benchmarks.py
```

### `ocr_process.py` - Implementazione Python di Riferimento

Script Python autoconsistente che implementa l'intero pipeline OCR producendo risultati **identici al 100%** a torchfree-ocr.

**Questa implementazione serve come riferimento completo per port in altri linguaggi.**

#### Prerequisiti
```bash
pip install opencv-python numpy onnxruntime pillow
```

#### Utilizzo Base
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

#### Output Generati

1. **File di testo** (`.ocr.python.txt`):
```
(0,15) (37,15) (37,27) (0,27) | Oil price | 0.8614
(42,16) (72,16) (72,24) (42,24) | AW | 0.0191
```

2. **Immagini con bbox** (`.ocr.bbox.png`):
- 🟢 Verde: Alta confidenza (>= 0.7)
- 🟡 Giallo: Media confidenza (0.4-0.7)
- 🔴 Rosso: Bassa confidenza (< 0.4)

---

## Pipeline OCR

Il motore OCR implementa le seguenti fasi:

1. **Detection**: Individua le regioni di testo usando il modello CRAFT
2. **Grouping**: Raggruppa i box adiacenti sulla stessa linea
3. **Recognition**: Riconosce il testo in ogni regione usando modelli CRNN
4. **Post-Processing**: Applica correzioni specifiche per lingua (es. apostrofi italiani)

Per dettagli implementativi, consultare [ocr_process.py](ocr_process.py) che contiene l'implementazione di riferimento Python.

## Riferimenti Implementativi

L'implementazione C# è stata portata dall'implementazione Python di riferimento. I seguenti file Python documentano l'implementazione completa:

- **[ocr_process.py](ocr_process.py)**: Pipeline OCR completo
- **[craft_utils.py](craft_utils.py)**: Utilities per detection CRAFT
- **[CRAFT Paper](https://arxiv.org/abs/1904.01941)**: Algoritmo di detection

Il port C# mantiene la stessa architettura modulare con i seguenti componenti:
- **Detection**: [CraftDetector.cs](EasyOcrNet/Detection/CraftDetector.cs)
- **Recognition**: [CrnnRecognizer.cs](EasyOcrNet/Recognition/CrnnRecognizer.cs)
- **Grouping**: [GroupTextBox.cs](EasyOcrNet/Detection/GroupTextBox.cs)
- **Post-Processing**: [TextPostProcessor.cs](EasyOcrNet/Recognition/TextPostProcessor.cs)

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

## Dimensioni e Piattaforme

### Dimensioni Pacchetto

- **Libreria Core**: Minimale
- **Con Dipendenze NuGet**: ~50 MB
- **Modelli ONNX**: ~375 MB totali (scaricabili on-demand)

### Piattaforme Supportate

- **Windows**: x86, x64, ARM
- **Linux**: x64, ARM (via SkiaSharp.NativeAssets.Linux)
- **macOS**: x64, ARM64 (Apple Silicon)

## Struttura Progetto

```
EasyOcrNet.sln                    # Solution Visual Studio
├── EasyOcrNet/                   # Libreria principale (net8.0)
│   ├── OcrEngine.cs             # Motore OCR principale
│   ├── Detection/               # Moduli detection (CRAFT)
│   ├── Recognition/             # Moduli recognition (CRNN)
│   ├── PostProcessing/          # Post-processing testo
│   ├── Models/                  # Modelli di dati
│   ├── Utils/                   # Utility (charset loader)
│   └── EasyOcrNet.csproj        # File di progetto
├── EasyOcrNet.Tests/             # Test xUnit (net8.0)
├── EasyOcrNet.Cli/               # App console esempio
├── models/cpu/                   # Modelli ONNX (git-ignored)
├── character/                    # File di mappatura caratteri
├── dataset/                      # Dataset di test
├── tools/                        # Script Python di supporto
└── README.md                     # Questo file
```

## Pubblicazione Pacchetto NuGet

### Prerequisiti

1. **GitHub CLI installato** (`gh`)
   ```bash
   # Verifica installazione
   gh --version
   ```

2. **Modelli ONNX scaricati**
   ```bash
   python tools/download_torchfree_models.py
   ```

3. **Git configurato** con accesso al repository
   ```bash
   git remote -v  # Verifica remote
   ```

### Processo di Pubblicazione Completo

#### 1. Preparazione dei Modelli

I modelli devono essere copiati nella struttura del pacchetto NuGet:

```bash
# Copia modelli ONNX
cp models/cpu/detection.onnx EasyOcrNet.Nuget/contentFiles/any/any/models/
cp models/cpu/latin_g2_rec.onnx EasyOcrNet.Nuget/contentFiles/any/any/models/
cp models/cpu/english_g2_rec.onnx EasyOcrNet.Nuget/contentFiles/any/any/models/

# Copia character files
cp character/*.txt EasyOcrNet.Nuget/contentFiles/any/any/character/
```

#### 2. Aggiornamento Versione

Modifica la versione in `EasyOcrNet.Nuget/EasyOcrNet.Nuget.csproj`:

```xml
<PropertyGroup>
  <Version>1.0.0</Version>  <!-- Aggiorna questo numero -->
</PropertyGroup>
```

#### 3. Build del Pacchetto NuGet

```bash
cd EasyOcrNet.Nuget
dotnet pack -c Release
```

Il pacchetto viene creato in: `nupkgs/EasyOcrNet.{version}.nupkg`

**Dimensione attesa**: ~201 MB (include modelli ONNX duplicati per compatibilità)

#### 4. Verifica del Pacchetto (Opzionale)

```bash
# Ispeziona contenuto del pacchetto
unzip -l nupkgs/EasyOcrNet.1.0.0.nupkg | grep -E "build|contentFiles"

# Verifica che contenga:
# - build/EasyOcrNet.targets
# - build/models/*.onnx
# - build/character/*.txt
# - contentFiles/any/any/models/*.onnx
# - contentFiles/any/any/character/*.txt
```

#### 5. Test Locale (Raccomandato)

Testa il pacchetto prima di pubblicarlo:

```bash
# Pulisci cache NuGet
dotnet nuget locals all --clear

# Testa con EasyOcrNet.CliNuget
cd EasyOcrNet.CliNuget
dotnet restore
dotnet build -c Release

# Verifica che i modelli siano stati copiati
ls bin/Release/net9.0/models/
ls bin/Release/net9.0/character/

# Esegui un test
cd bin/Release/net9.0
./EasyOcrNet.CliNuget /path/to/test/images
```

#### 6. Creazione Release Notes

Crea o aggiorna `RELEASE_NOTES.md` con:
- Numero versione
- Novità e modifiche
- Benchmark e statistiche
- Known issues
- Breaking changes (se presenti)

Vedi esempio: [RELEASE_NOTES.md](RELEASE_NOTES.md)

#### 7. Creazione Tag Git

```bash
# Crea tag annotato
git tag -a v1.0.0 -m "EasyOcrNet v1.0.0 - Initial Release with NuGet package"

# Push del tag
git push origin v1.0.0
```

#### 8. Pubblicazione su GitHub Release

```bash
# Crea release con GitHub CLI
gh release create v1.0.0 \
  --title "EasyOcrNet v1.0.0 - Initial Release" \
  --notes-file RELEASE_NOTES.md \
  nupkgs/EasyOcrNet.1.0.0.nupkg

# Output: URL della release
# https://github.com/mapo80/easyocrnet/releases/tag/v1.0.0
```

#### 9. Verifica Release

```bash
# Visualizza dettagli release
gh release view v1.0.0

# Verifica che l'asset sia presente
gh release view v1.0.0 --json assets
```

### Pubblicazione su NuGet.org (Opzionale)

Per pubblicare su NuGet.org pubblico:

```bash
# 1. Ottieni API key da nuget.org
# Vai su: https://www.nuget.org/account/apikeys

# 2. Publica il pacchetto
dotnet nuget push nupkgs/EasyOcrNet.1.0.0.nupkg \
  --api-key YOUR_API_KEY \
  --source https://api.nuget.org/v3/index.json
```

**Nota**: Per NuGet.org considera di:
- Ridurre dimensioni pacchetto (modelli separati)
- Aggiungere icona pacchetto
- Includere README.md nel pacchetto
- Configurare repository URL

### Installazione da GitHub Release

Gli utenti possono installare il pacchetto dalla release:

```bash
# 1. Scarica il pacchetto dalla release
wget https://github.com/mapo80/easyocrnet/releases/download/v1.0.0/EasyOcrNet.1.0.0.nupkg

# 2. Aggiungi sorgente locale
dotnet nuget add source /path/to/downloaded --name easyocrnet-local

# 3. Installa il pacchetto
dotnet add package EasyOcrNet --version 1.0.0
```

### Checklist Pre-Release

- [ ] Modelli ONNX copiati in `EasyOcrNet.Nuget/contentFiles/`
- [ ] Versione aggiornata in `.csproj`
- [ ] Pacchetto NuGet buildato con successo
- [ ] Dimensione pacchetto verificata (~201 MB)
- [ ] Test locali completati con successo
- [ ] RELEASE_NOTES.md creato/aggiornato
- [ ] Tag git creato e pushato
- [ ] Release GitHub creata con asset
- [ ] Release verificata e funzionante

### Troubleshooting

**Problema**: Modelli non copiati nell'output
```bash
# Soluzione: Verifica che EasyOcrNet.targets sia nel pacchetto
unzip -l nupkgs/EasyOcrNet.1.0.0.nupkg | grep targets

# Se manca, ricostruisci il pacchetto
dotnet clean
dotnet pack -c Release
```

**Problema**: Pacchetto troppo grande
```bash
# Il pacchetto include modelli duplicati (contentFiles + build)
# Dimensione attesa: ~201 MB
# Per ridurre, considera di pubblicare modelli separatamente
```

**Problema**: ProjectReference impedisce uso del pacchetto
```bash
# Durante sviluppo, il ProjectReference ha priorità
# Per testare solo il pacchetto NuGet, rimuovi temporaneamente:
# <ProjectReference Include="..\EasyOcrNet\EasyOcrNet.csproj" />
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

- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** by JaidedAI - OCR originale
- **[TorchfreeEasyOCR](https://github.com/SeldonHZ/TorchfreeEasyOCR)** by SeldonHZ - Conversione modelli a ONNX
- **Microsoft.ML.OnnxRuntime** - Runtime di inferenza ONNX
- **SkiaSharp** - Libreria di elaborazione immagini cross-platform

## Risorse

- [Documentazione EasyOCR originale](https://www.jaided.ai/easyocr/documentation/)
- [TorchfreeEasyOCR Repository](https://github.com/SeldonHZ/TorchfreeEasyOCR)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [CRAFT Paper](https://arxiv.org/abs/1904.01941) - Character Region Awareness For Text detection

## Contatti

Per bug, richieste di funzionalità o domande, apri una issue su GitHub.

---

**EasyOcrNet** - OCR .NET veloce, leggero e multi-lingua senza dipendenze PyTorch

*README aggiornato per riflettere l'architettura attuale del progetto (Ottobre 2025)*
