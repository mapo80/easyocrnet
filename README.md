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

## 📋 Guida all'Implementazione OCR

Questa sezione documenta l'implementazione completa dell'OCR pipeline per facilitare il port in altri linguaggi di programmazione (C#, Java, Rust, etc.).

### ⚠️ Principi Fondamentali

1. **L'ordine delle operazioni è CRITICO** - anche piccole variazioni producono risultati diversi
2. **Precisione numerica** - usare float32 ovunque, evitare conversioni implicite
3. **Interpolazione corretta** - cv2.INTER_LINEAR (bilinear) per tutti i resize
4. **Two-pass recognition** - essenziale per risultati identici a torchfree

### 🔄 Pipeline Completo

```
Immagine BGR → Detection → Text Grouping → Per-Crop Recognition → Results
                  ↓              ↓                    ↓
              CRAFT Model    group_text_box    Two-Pass Processing
```

---

## 1️⃣ DETECTION - Fase 1: Preprocessing

### Input
- Immagine BGR da cv2.imread() o equivalente
- Dimensioni originali: qualsiasi (H, W, 3)

### Step 1.1: Conversione Grayscale
```python
# IMPORTANTE: Fare questo PRIMA dell'estrazione dei crop!
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
# Shape: (H, W) - uint8, range [0, 255]
```

### Step 1.2: Resize con Aspect Ratio Preservato

**ATTENZIONE**: Questo è diverso da un semplice resize!

```python
def resize_aspect_ratio(img, square_size=2560, interpolation=cv2.INTER_LINEAR, mag_ratio=1.0):
    """
    Resize mantenendo aspect ratio + padding a multipli di 32.

    CRITICAL: Il resize deve:
    1. Mantenere aspect ratio originale
    2. Scalare in modo che max(height, width) <= square_size * mag_ratio
    3. Paddare a multipli di 32 (necessario per CRAFT)
    """
    height, width = img.shape[:2]

    # Calcola target size
    target_size = mag_ratio * max(height, width)

    if target_size > square_size:
        target_size = square_size

    # Calcola ratio mantenendo aspect ratio
    ratio = target_size / max(height, width)

    target_h = int(height * ratio)
    target_w = int(width * ratio)

    # Resize con INTER_LINEAR
    img_resized = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # Padding a multipli di 32
    target_h32 = target_h if target_h % 32 == 0 else (target_h // 32 + 1) * 32
    target_w32 = target_w if target_w % 32 == 0 else (target_w // 32 + 1) * 32

    # Pad con zeri (nero)
    img_padded = np.zeros((target_h32, target_w32, 3), dtype=np.uint8)
    img_padded[:target_h, :target_w, :] = img_resized

    return img_padded, ratio, (target_h, target_w)
```

**Valori tipici**:
- Per immagini normali: risultato ~(608, 800) o simile
- ratio = dimensione_originale / dimensione_resized

### Step 1.3: Normalizzazione ImageNet

```python
def normalizeMeanVariance(img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    """
    Normalizzazione con ImageNet mean/std.

    CRITICAL: Ordine operazioni:
    1. Converti a float32 e dividi per 255
    2. Sottrai mean
    3. Dividi per std
    """
    img = img.astype(np.float32) / 255.0

    # Mean subtraction (per canale)
    img[:, :, 0] = (img[:, :, 0] - mean[0]) / variance[0]  # B
    img[:, :, 1] = (img[:, :, 1] - mean[1]) / variance[1]  # G
    img[:, :, 2] = (img[:, :, 2] - mean[2]) / variance[2]  # R

    return img
```

### Step 1.4: Conversione a Tensor

```python
# Transpose: (H, W, 3) → (3, H, W)
img_tensor = img_normalized.transpose(2, 0, 1)

# Add batch dimension: (3, H, W) → (1, 3, H, W)
input_tensor = img_tensor[np.newaxis, :, :, :]

# Assicurati che sia float32
input_tensor = input_tensor.astype(np.float32)
```

**Output finale**: `(1, 3, H_padded, W_padded)` float32

---

## 1️⃣ DETECTION - Fase 2: Inferenza CRAFT

### Modello
- File: `detection.onnx` (83 MB)
- Input shape: `[1, 3, H, W]` dove H, W sono multipli di 32
- Output shape: `[1, 2, H/2, W/2]` - score maps

### Inferenza

```python
import onnxruntime as ort

session = ort.InferenceSession('models/cpu/detection.onnx',
                               providers=['CPUExecutionProvider'])

output = session.run(None, {session.get_inputs()[0].name: input_tensor})[0]
# Shape: (1, 2, H/2, W/2)
```

**NOTA**: Output shape è la metà dell'input (stride=2)

---

## 1️⃣ DETECTION - Fase 3: Post-processing

### Step 3.1: Estrazione Bounding Boxes

```python
def getDetBoxes(score_map, text_threshold=0.7, link_threshold=0.4,
                low_text=0.4, poly=False):
    """
    Estrae bounding box da score map CRAFT.

    PARAMETERS:
    - text_threshold: soglia per region score (default: 0.7)
    - link_threshold: soglia per affinity score (default: 0.4)
    - low_text: soglia minima per region (default: 0.4)

    RETURNS:
    - boxes: lista di bbox [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    # Separa i due canali
    textmap = score_map[:, :, 0]  # Region score
    linkmap = score_map[:, :, 1]  # Affinity score

    # Threshold binario
    text_mask = textmap > low_text
    link_mask = linkmap > link_threshold

    # Connected components analysis
    # Trova regioni connesse in text_mask
    # ... (implementazione completa in craft_utils.py)

    return boxes
```

### Step 3.2: Scaling Coordinate

**CRITICAL**: Usa `ratio_net=2` perché lo score map è la metà dell'input!

```python
def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    """
    Scala coordinate da score map a immagine originale.

    CRITICAL: ratio_net=2 perché score map è H/2 x W/2
    """
    if len(polys) == 0:
        return polys

    for poly in polys:
        # Scala da score map (H/2, W/2) a immagine resized (H, W)
        poly *= ratio_net

        # Scala da immagine resized a immagine originale
        poly[:, 0] /= ratio_w
        poly[:, 1] /= ratio_h

    return polys
```

**Esempio**:
- Score map: (304, 400)
- Immagine resized: (608, 800)
- Immagine originale: (393, 568)
- ratio_net = 2
- ratio_w = 800 / 568
- ratio_h = 608 / 393

---

## 1️⃣ DETECTION - Fase 4: Text Grouping

**CRITICAL**: Questo step merge box adiacenti sulla stessa linea

```python
def group_text_box(polys, slope_ths=0.1, ycenter_ths=0.5,
                   height_ths=0.5, width_ths=0.5, add_margin=0.1):
    """
    Raggruppa detection box adiacenti.

    PARAMETERS (usare questi valori esatti):
    - slope_ths: 0.1 - threshold per pendenza linea
    - ycenter_ths: 0.5 - threshold per allineamento verticale
    - height_ths: 0.5 - threshold per altezza simile
    - width_ths: 0.5 - threshold per merge orizzontale
    - add_margin: 0.1 - margine aggiuntivo (10%)

    RETURNS:
    - horizontal_list: lista di [x_min, x_max, y_min, y_max] per testo orizzontale
    - free_list: lista di 4 punti per testo con orientamento libero
    """
    # ... (implementazione completa in craft_utils.py)
```

---

## 2️⃣ RECOGNITION - Fase 1: Estrazione Crop

### Step 1.1: Estrai Crop dall'Immagine Grayscale

**CRITICAL**: Usa `img_gray` (l'immagine grayscale creata all'inizio), NON l'immagine BGR!

```python
# Per horizontal text
for box in horizontal_list:
    x_min, x_max, y_min, y_max = box[:4]

    # Clip alle dimensioni immagine
    x_min = int(np.clip(x_min, 0, orig_w))
    x_max = int(np.clip(x_max, 0, orig_w))
    y_min = int(np.clip(y_min, 0, orig_h))
    y_max = int(np.clip(y_max, 0, orig_h))

    # Estrai crop dalla GRAYSCALE
    crop = img_gray[y_min:y_max, x_min:x_max]
    # Shape: (h, w) - uint8 grayscale
```

---

## 2️⃣ RECOGNITION - Fase 2: Preprocessing Per-Crop

### Step 2.1: Calcola imgW Dinamico

**CRITICAL**: Ogni crop ha il suo imgW basato sul suo aspect ratio!

```python
imgH = 64  # FISSO per tutti i crop

h, w = crop.shape
crop_ratio = w / float(h)

# Calcola imgW per QUESTO crop
imgW = math.ceil(crop_ratio) * imgH

# Calcola resized_w
resized_w = imgW if int(imgH * crop_ratio) > imgW else int(imgH * crop_ratio)
```

**Esempio**:
- Crop: (8, 30) → ratio = 3.75
- imgW = ceil(3.75) * 64 = 4 * 64 = 256
- resized_w = int(64 * 3.75) = 240

### Step 2.2: Resize

**CRITICAL**: Usa cv2.INTER_LINEAR (NO PIL, NO BICUBIC, NO LANCZOS)

```python
resized = cv2.resize(crop, (resized_w, imgH), interpolation=cv2.INTER_LINEAR)
# Shape: (64, resized_w) - uint8
```

### Step 2.3: Normalizzazione [-1, 1]

```python
# Converti a float32 e normalizza a [0, 1]
img_array = resized.astype(np.float32) / 255.0

# Scala a [-1, 1]
img_array = (img_array - 0.5) / 0.5
# Shape: (64, resized_w) - float32, range [-1, 1]
```

### Step 2.4: Padding con Last Column Repeat

**CRITICAL**: Ripeti l'ultima colonna, NON paddare con zeri!

```python
# Crea array paddato
padded = np.zeros((imgH, imgW), dtype=np.float32)

# Copia immagine
padded[:, :resized_w] = img_array

# Ripeti ultima colonna per riempire padding
if resized_w < imgW:
    last_col = img_array[:, -1:]  # Shape: (64, 1)
    padded[:, resized_w:] = np.tile(last_col, (1, imgW - resized_w))
```

### Step 2.5: Tensor Shape

```python
# Add batch + channel dimensions: (64, imgW) → (1, 1, 64, imgW)
input_tensor = padded[None, None, :, :].astype(np.float32)
```

---

## 2️⃣ RECOGNITION - Fase 3: Inferenza (First Pass)

### Modello
- File: `english_g2_rec.onnx` per inglese
- Input: `[1, 1, 64, imgW]` - imgW varia per crop!
- Output: `[1, T, 97]` dove T = sequenza temporale, 97 = num_classes

### Inferenza

```python
rec_session = ort.InferenceSession('models/cpu/english_g2_rec.onnx',
                                   providers=['CPUExecutionProvider'])

output = rec_session.run(None, {rec_session.get_inputs()[0].name: input_tensor})[0]
# Shape: (1, T, 97)
```

### Charset

**CRITICAL**: Per inglese, charset HARDCODED (non da file):

```python
charset_en = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# Lunghezza: 96 caratteri
# charset_list = ['[blank]'] + list(charset_en)  # 97 totali
```

---

## 2️⃣ RECOGNITION - Fase 4: CTC Decoding

**CRITICAL**: NO filtering/renormalization prima di argmax!

```python
def decode_recognition(output, charset):
    """
    CTC greedy decoding.

    CRITICAL: Argmax DIRETTAMENTE sui logit!
    """
    # 1. Argmax diretto (NO softmax prima!)
    preds_index = np.argmax(output, axis=2).reshape(-1)
    # Shape: (T,)

    # 2. Build charset list
    charset_list = ['[blank]'] + list(charset)

    # 3. Remove consecutive duplicates
    indices_list = []
    prev_idx = None
    for idx in preds_index:
        if idx != prev_idx:
            indices_list.append(idx)
            prev_idx = idx

    # 4. Remove blanks (index 0)
    indices_filtered = [idx for idx in indices_list if idx != 0]

    # 5. Map to characters
    text = ''.join([charset_list[idx] for idx in indices_filtered
                    if idx < len(charset_list)])

    return text
```

---

## 2️⃣ RECOGNITION - Fase 5: Confidence Calculation

```python
def calculate_confidence(output, charset):
    """
    Calcola confidence score.

    FORMULA: custom_mean(max_probs)
    dove custom_mean(x) = prod(x)^(2/sqrt(len(x)))
    """
    # Apply softmax
    preds_prob = softmax(output, axis=2)

    # Get indices
    preds_index = np.argmax(output, axis=2)[0]

    # Collect max probs for non-blank, non-duplicate
    max_probs_list = []
    prev_idx = None

    for i, idx in enumerate(preds_index):
        if idx != 0 and idx != prev_idx:  # not blank and not duplicate
            max_probs_list.append(preds_prob[0, i, idx])
        prev_idx = idx

    if len(max_probs_list) > 0:
        # Custom mean formula
        return np.prod(max_probs_list) ** (2.0 / np.sqrt(len(max_probs_list)))
    else:
        return 0.0
```

---

## 2️⃣ RECOGNITION - Fase 6: Two-Pass Processing

**CRITICAL**: Questo è essenziale per risultati identici!

### Step 6.1: Identify Low Confidence Results

```python
contrast_ths = 0.1  # Threshold fisso

low_conf_indices = [i for i, (_, _, conf, _) in enumerate(results)
                    if conf < contrast_ths]
```

### Step 6.2: Adjust Contrast

**CRITICAL**: Applica contrast adjustment al crop GIÀ RESIZED, non all'originale!

```python
def contrast_grey(img):
    """Calcola contrast."""
    high = np.percentile(img, 90)
    low = np.percentile(img, 10)
    return (high-low)/np.maximum(10, high+low), high, low


def adjust_contrast_grey(img, target=0.4):
    """
    Adjust contrast se sotto target.

    CRITICAL: Formula esatta (include +25 offset!)
    """
    contrast, high, low = contrast_grey(img)

    if contrast < target:
        img = img.astype(int)  # IMPORTANTE: int, non float32!
        ratio = 200./np.maximum(10, high-low)
        img = (img - low + 25)*ratio  # NOTA: +25 offset!
        img = np.maximum(np.full(img.shape, 0),
                        np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)

    return img
```

### Step 6.3: Second Pass Processing

```python
for idx in low_conf_indices:
    bbox, _, _, crop_original = first_pass_results[idx]

    # STEP 1: Resize crop originale (come first pass)
    h, w = crop_original.shape
    crop_ratio = w / float(h)
    imgW = math.ceil(crop_ratio) * imgH
    resized_w = imgW if int(imgH * crop_ratio) > imgW else int(imgH * crop_ratio)
    crop_resized = cv2.resize(crop_original, (resized_w, imgH),
                              interpolation=cv2.INTER_LINEAR)

    # STEP 2: Applica contrast adjustment al crop RESIZED
    crop_adjusted = adjust_contrast_grey(crop_resized, target=0.5)

    # STEP 3: Continua preprocessing normale
    img_array = crop_adjusted.astype(np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5

    # STEP 4-7: Padding, inference, decode, confidence
    # ... (stesso processo del first pass)
```

### Step 6.4: Merge Results

```python
# Per ogni crop, scegli il risultato con confidenza maggiore
for i in range(len(first_pass_results)):
    if i in second_pass_results:
        text1, conf1 = first_pass_results[i][1:3]
        text2, conf2 = second_pass_results[i]

        if conf1 > conf2:
            final_results.append((bbox, text1, conf1))
        else:
            final_results.append((bbox, text2, conf2))
    else:
        final_results.append(first_pass_results[i][:3])
```

---

## ⚠️ PUNTI CRITICI DI ATTENZIONE

### 1. Ordine Operazioni
- Grayscale conversion PRIMA di detection processing
- Crop extraction da immagine grayscale
- Contrast adjustment al crop RESIZED, non originale

### 2. Tipi Numerici
- SEMPRE usare float32 per tensori
- uint8 per immagini fino a normalizzazione
- int (non float32!) per contrast adjustment

### 3. Interpolazione
- cv2.INTER_LINEAR per TUTTI i resize
- MAI usare PIL resize, BICUBIC, o LANCZOS

### 4. Parametri Fissi
```python
# Detection
text_threshold = 0.7
link_threshold = 0.4
low_text = 0.4

# Text Grouping
slope_ths = 0.1
ycenter_ths = 0.5
height_ths = 0.5
width_ths = 0.5
add_margin = 0.1

# Recognition
imgH = 64  # FISSO
min_size = 20  # Filtra bbox troppo piccoli

# Two-pass
contrast_ths = 0.1
adjust_contrast = 0.5
```

### 5. Coordinate Scaling
- Detection: `ratio_net = 2` (CRITICO!)
- Use `ratio_w` e `ratio_h` dal resize originale

### 6. CTC Decoding
- NO softmax/filtering prima di argmax
- Remove duplicates DOPO argmax
- Remove blanks DOPO remove duplicates

### 7. Charset
- Inglese: hardcoded string (96 chars)
- Include spazio a posizione 42: `' '`
- Include simboli speciali: `!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ €`

---

## 🧪 Verifica Implementazione

### Test di Correttezza

1. **Detection Count Test**:
```python
# Dataset: dataset/base/HAL.2015.page_42.pdf_125176.png
# Expected: 10 detections
assert len(results) == 10
```

2. **Recognition Exact Match Test**:
```python
expected_texts = [
    "Oil price", "AW", "95i", "Oil price", "Brent",
    "52 16", "Omt", "108.71", "Nlwal @aSmicc", "Henry Hub"
]

for i, (bbox, text, conf) in enumerate(results):
    assert text == expected_texts[i], f"Mismatch at {i}: {text} != {expected_texts[i]}"
```

3. **Full Pipeline Test**:
```bash
# Confronta con torchfree-ocr
python ocr_process.py --dataset dataset/base --mode text
# Verifica che tutti i .ocr.python.txt siano identici a torchfree
```

---

## 📦 File di Supporto

- **[ocr_process.py](ocr_process.py)**: Implementazione completa di riferimento
- **[craft_utils.py](craft_utils.py)**: Utilities CRAFT autoconsistenti
- **[character/latin_char.txt](character/latin_char.txt)**: Charset latino (fallback)

---

## 🔗 Riferimenti

- [CRAFT Paper](https://arxiv.org/abs/1904.01941) - Detection algorithm
- [TorchfreeEasyOCR](https://github.com/SeldonHZ/TorchfreeEasyOCR) - Source dei modelli ONNX
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Progetto originale

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
