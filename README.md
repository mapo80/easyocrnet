# EasyOCR.NET

EasyOCR.NET is a .NET wrapper around the TorchfreeEasyOCR models. The library
provides a production-ready OCR pipeline that loads detector and recognizer models,
clusters detected regions into line-level crops, and decodes character sequences with
language-specific alphabets. The repository also contains tooling to compare the
Torchfree reference implementation with the .NET stack on both ONNX Runtime and
OpenVINO backends.

## Architecture overview

The `EasyOcr` engine orchestrates three focused subsystems:

- **Model resolution** – `OcrOptions` describes the desired language, backend and
  device. `OcrModelCatalog` maps languages to recognizer checkpoints and locates the
  appropriate ONNX/OpenVINO files. `OcrBackendFactory` materialises the runtime
  (`OnnxBackend` or `OpenVinoBackend`) while `InferenceBackend` controls which engine
  is instantiated.
- **Detection** – `TextDetector` normalises the bitmap, performs inference with the
  detector network and post-processes the heatmap with `DetectionPostProcessor` and
  `TextComponentGrouper` to produce ordered line bounding boxes.
- **Recognition** – `TextRecognizer` expands each region, builds the recogniser input
  tensor with `RecognitionInputBuilder`, runs inference, and converts the logits into
  text via `SequenceDecoder` using the language-specific character set defined in
  `PredefinedCharacterSets`.

The internal boundaries allow each component to be unit-tested in isolation and make
it straightforward to extend the engine with new backends or language packs.

## Runtime performance

The 1.22+ ONNX Runtime build is initialised with aggressive graph optimisation,
parallel execution and CPU arena tweaks so inference saturates the available cores
out of the box. The detector/recogniser tensor builders reuse pooled buffers,
vector-friendly spans and uninitialised allocations to minimise per-frame garbage,
while the connected-component clustering now iterates over flat buffers instead of
allocating a queue of tuples for every region. These changes consistently shaved
double-digit percentages off the timings reported by the benchmark CLI in our test
environment.

Every `EasyOcr.Read` invocation captures a detailed `OcrExecutionProfile` recording
the detector time, the total recogniser time, and the warmed average (mean of all
recogniser runs except the first warm-up). The profile is exposed through
`EasyOcr.LastProfile` so callers can feed the metrics back into their own logging or
adaptive batching strategies.

## Supported languages

Languages are expressed with the `OcrLanguage` enum. The engine currently bundles
alphabets and model mappings for the following groups:

- Latin-based languages: Afrikaans, Albanian, Basque, Catalan, Croatian, Czech,
  Danish, Dutch, Estonian, Filipino, Finnish, French, Galician, German, Hungarian,
  Icelandic, Indonesian, Irish, Italian, Kurdish, Latin, Latvian, Lithuanian, Maori,
  Malay, Maltese, Norwegian, Polish, Portuguese, Romanian, Serbian (Latin script),
  Slovak, Slovenian, Spanish, Swahili, Swedish, Turkish, Uzbek and Vietnamese.
- East Asian languages: Japanese, Korean and Simplified Chinese.
- Thai.

Each entry automatically selects the correct recogniser checkpoint and character set.
New languages can be wired in by updating `OcrLanguageCatalog` and, if necessary,
adding a new character table to `PredefinedCharacterSets`.

## Using the library

```csharp
using EasyOcrNet;
using EasyOcrNet.Configuration;
using EasyOcrNet.Languages;
using SkiaSharp;

var options = new OcrOptions(
    modelDirectory: "models/openvino/cpu",
    language: OcrLanguage.English,
    backend: InferenceBackend.OpenVino,
    device: "CPU");
using var engine = new EasyOcr(options);
using var bitmap = SKBitmap.Decode("examples/english.png");

var results = engine.Read(bitmap);
foreach (var line in results)
{
    Console.WriteLine($"{line.Text} [{line.BoundingBox}]");
}

var profile = engine.LastProfile;
Console.WriteLine($"Detector: {profile.DetectionDuration.TotalMilliseconds:F1} ms");
Console.WriteLine($"Recognizer average (warm): {profile.WarmedAverageRecognitionMilliseconds:F1} ms");
```

The engine automatically normalises the image, groups detections into reading order,
and emits an `OcrResult` for each line.

### Preparing the model cache

The heavy detector/recogniser weights live in the upstream TorchfreeEasyOCR GitHub
release. Use the helper scripts to hydrate the local cache with those artefacts:

```bash
# Download the ONNX models published in the pre-v1.1.0 release.
# This pulls the official TorchfreeEasyOCR detector plus the language recognisers
# (including the latin_g2 Italian model) directly from GitHub so you do not have to
# re-export them yourself.
python tools/download_torchfree_models.py --output models/cpu

# Convert the release weights into OpenVINO IR format.
python tools/convert_to_openvino.py --source models/cpu --output models/openvino/cpu
```

Both scripts only fetch/produce files when the expected checksums are missing, so you
can re-run them safely.

## Example extractor CLI

`ExampleExtractor` is a .NET console utility that runs the OCR engine on a single
image and saves the text next to the bitmap as `<name>.dotnet.<backend>.txt`.

```bash
dotnet run --project ExampleExtractor -- examples/english.png --models models/openvino/cpu --backend OpenVino
```

Optional switches:

- `--models <dir>` – override the model directory.
- `--output <file>` – save the recognised text to a custom file.
- `--language <name>` – force a language (matches the `OcrLanguage` enum, e.g.
  `SimplifiedChinese`).
- `--backend <onnx|openvino>` – choose the inference engine (defaults to `Onnx`).
- `--device <name>` – specify the OpenVINO device (`CPU`, `GPU`, …); ignored for ONNX.

### Italian generated ID samples

The repository includes three synthetic Italian identity cards under
`examples/generated_*.png`. Running the Python EasyOCR helper and the .NET extractor
with `--language Italian` produces comparable transcripts saved alongside each image:

| Image | EasyOCR (Python) | EasyOCR.NET (.dotnet.onnx) |
| --- | --- | --- |
| `generated_1.png` | [`examples/generated_1.easyocr.txt`](examples/generated_1.easyocr.txt) | [`examples/generated_1.dotnet.onnx.txt`](examples/generated_1.dotnet.onnx.txt) |
| `generated_2.png` | [`examples/generated_2.easyocr.txt`](examples/generated_2.easyocr.txt) | [`examples/generated_2.dotnet.onnx.txt`](examples/generated_2.dotnet.onnx.txt) |
| `generated_3.png` | [`examples/generated_3.easyocr.txt`](examples/generated_3.easyocr.txt) | [`examples/generated_3.dotnet.onnx.txt`](examples/generated_3.dotnet.onnx.txt) |

Both pipelines now rely on the EasyOCR `latin_g2` recogniser shipped in the
TorchfreeEasyOCR GitHub release to decode the Italian texts, keeping the character
set consistent across the .NET and Python runs.

## Confronto EasyOCR vs EasyOCR.NET sulle carte italiane generate

Ho ripetuto l'estrazione del testo per `examples/generated_[1-3].png` utilizzando i
seguenti comandi, assicurandomi di scaricare i modelli TorchfreeEasyOCR aggiornati
per il backend ONNX:

```bash
python easyocr_extract.py
dotnet run --project ExampleExtractor -- examples/generated_1.png --language Italian --models models/cpu --backend Onnx
dotnet run --project ExampleExtractor -- examples/generated_2.png --language Italian --models models/cpu --backend Onnx
dotnet run --project ExampleExtractor -- examples/generated_3.png --language Italian --models models/cpu --backend Onnx
```

I testi prodotti da EasyOCR (Python) e dalla libreria .NET mostrano divergenze
significative: per esempio, EasyOCR introduce molte distorsioni ortografiche sulla
prima tessera (``Colhou``, ``Scaotnza Dokurtio`` o ``Lavoro Autonoho``), mentre la
versione .NET mantiene una struttura più vicina al documento originale pur
introducendo rumore marginale nelle cifre seriali.

Per quantificare le differenze ho calcolato alcune metriche classiche
per l'OCR (Levenshtein, Character Error Rate e Word Error Rate) prendendo la
trascrizione .NET come riferimento. Valori più alti indicano scostamenti maggiori.

| Immagine | CER vs .NET | WER vs .NET | Distanza Levenshtein (caratteri) | Distanza Levenshtein normalizzata | Parole (.NET) | Distanza Levenshtein (parole) | Distanza parole normalizzata |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `generated_1.png` | 0.5954 | 0.9623 | 284 | 0.5954 | 53 | 51 | 0.8226 |
| `generated_2.png` | 0.8447 | 1.1343 | 457 | 0.7504 | 67 | 76 | 0.9500 |
| `generated_3.png` | 0.5475 | 0.8974 | 323 | 0.5475 | 78 | 70 | 0.8974 |
| **Media** | **0.6625** | **0.9980** | — | **0.6311** | — | — | **0.8900** |

I risultati mostrano che EasyOCR, pur riconoscendo la struttura generale, introduce
errori in oltre il 60% dei caratteri medi e quasi il 100% delle parole rispetto
all'output .NET. L'elevato WER evidenzia parole mancanti, divise o corrotte che
rischiano di compromettere scenari in cui si richiede coerenza con il formato della
carta d'identità. Conviene quindi affidare il flusso produttivo alla pipeline .NET,
utilizzando EasyOCR solo come baseline o strumento diagnostico.

## TorchfreeEasyOCR single-image extraction

The repository also ships with `torchfreeeasyocr_extract.py`, a Python helper that
runs the original TorchfreeEasyOCR models against a single image and writes the
result to `<name>.torchonnx.txt`. The script will download the CPU models on first
run if they are missing.

1. *(Optional but recommended)* Create and activate a Python virtual environment.
2. Install the runtime dependencies:

   ```bash
   pip install easyocr onnxruntime opencv-python numpy
   ```

   Install `openvino==2023.2.0` as well if you intend to use the OpenVINO provider.
3. Download the TorchfreeEasyOCR ONNX weights (skip to auto-download them on demand):

   ```bash
   python tools/download_torchfree_models.py
   ```

4. Run OCR on an image:

   ```bash
   python torchfreeeasyocr_extract.py examples/english.png
   ```

   Override the provider, model directory, or output file with `--provider`,
   `--models`, and `--output`.

## Benchmarking the OCR pipelines

Two helper CLIs make it easy to compare the inference latency of the .NET engine and
the TorchfreeEasyOCR reference implementation. Both tools default to running six
iterations against `examples/english.png` and compute the average after discarding
the first warm-up pass.

### EasyOcrNet benchmark

```bash
dotnet run --project EasyOcrNet.BenchmarkCli -- --backend OpenVino --models models/openvino/cpu
```

Optional arguments mirror the extractor utility:

- `--image <path>` – use a different bitmap.
- `--models <dir>` – point to a custom ONNX/OpenVINO model directory.
- `--language <name>` – override the `OcrLanguage` used for recognition.
- `--backend <onnx|openvino>` / `--device <name>` – select the inference engine and
  OpenVINO target.
- `--runs <n>` / `--discard <n>` – tune the number of executions and warm-up runs.

The CLI prints each iteration time, the averaged latency, and the detector segment
count so you can spot anomalies quickly.

### TorchfreeEasyOCR benchmark

```bash
python torchfreeeasyocr_benchmark.py
```

Use `--provider`, `--models`, `--runs`, or `--discard` to mirror the .NET settings.
The script shares the same model auto-discovery logic as the extractor and outputs
per-run timings plus a final average.

Run both commands back-to-back to compare the mean latencies reported by each tool.

### Sample benchmark results

The following measurements were captured inside the project container on the
`examples/english.png` image with six runs and one warm-up pass discarded using the
TorchfreeEasyOCR release models:

| Pipeline | Backend | Average (ms) | Min (ms) | Max (ms) | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| EasyOcrNet.BenchmarkCli | Onnx | 1313.42 | 1235.99 | 1383.85 | 12 segments detected per run |
| EasyOcrNet.BenchmarkCli | OpenVino | 1292.56 | 1270.03 | 1325.12 | 12 segments detected per run |

OpenVINO trims a little over 20 ms off the mean latency on the CPU models when both
pipelines use the TorchfreeEasyOCR release weights. Your hardware and execution
provider will influence the exact timings, but the table provides a reference point
for the default setup.

## Testing

The solution contains an `EasyOcrNet.Tests` project with unit and integration tests
covering the language catalog, detection grouping, tensor preparation, and an
end-to-end regression against the TorchfreeEasyOCR English reference output.
Execute the suite with:

```bash
dotnet test
```

## Pacchetto NuGet con i modelli incorporati

La libreria `EasyOcrNet` può essere distribuita come pacchetto NuGet che include
tutti i modelli pubblicati nella release GitHub `v2025.09.19`. Il processo usa
esclusivamente gli artefatti ufficiali, senza rigenerare i pesi.

1. **Scarica i modelli dalla release**

   ```bash
   export GITHUB_TOKEN="<token Github>"
   python tools/download_release_models.py --tag v2025.09.19 --output external/release-models
   ```

   Lo script recupera `easyocrnet-models-cpu-onnx.zip` e
   `easyocrnet-models-openvino-ir.tar.gz`, li estrae in
   `external/release-models/{onnx,openvino}` e verifica che i file siano presenti
   prima di procedere con il packaging.

2. **Genera il pacchetto NuGet**

   ```bash
   dotnet pack EasyOcrNet/EasyOcrNet.csproj -c Release -o artifacts
   ```

   Il pacchetto risultante `EasyOcrNet.1.0.0.nupkg` contiene le pipeline ONNX e
   OpenVINO in `contentFiles/any/any/models/`. Mantieni gli artefatti locali fuori
   dal repository (la cartella `artifacts/` è già esclusa dal versionamento).

3. **Pubblica il pacchetto sulla release GitHub**

   ```bash
   curl -H "Authorization: token $GITHUB_TOKEN" \
        -H "Content-Type: application/octet-stream" \
        --data-binary @artifacts/EasyOcrNet.1.0.0.nupkg \
        "https://uploads.github.com/repos/mapo80/easyocrnet/releases/248652542/assets?name=EasyOcrNet.1.0.0.nupkg"
   ```

   L’asset è ora disponibile pubblicamente all’indirizzo
   <https://github.com/mapo80/easyocrnet/releases/download/v2025.09.19/EasyOcrNet.1.0.0.nupkg>.

### Modelli copiati automaticamente nell’applicazione

`EasyOcrNet.csproj` include tutti i `.onnx`, `.xml` e `.bin` della release come
`contentFiles`, con copia automatica nella cartella di output del consumer. Alla
prima compilazione i target MSBuild verificano che i modelli siano presenti in
`external/release-models` e mostrano un errore guida se mancano gli artefatti
ufficiali.

### Progetto di smoke test

`EasyOcrNet.NuGetSmokeTest` dimostra l’utilizzo del pacchetto NuGet senza
download extra dei modelli:

1. Scarica il pacchetto pubblicato e aggiungilo a un feed locale (nuovamente
   senza committare gli artefatti):

   ```bash
   mkdir -p packages
   curl -L -o packages/EasyOcrNet.1.0.0.nupkg \
     https://github.com/mapo80/easyocrnet/releases/download/v2025.09.19/EasyOcrNet.1.0.0.nupkg
   dotnet nuget add source $(pwd)/packages --name easyocrnet-local --store-password-in-clear-text
   ```

2. Ripristina e avvia l’applicazione di test:

   ```bash
   dotnet run --project EasyOcrNet.NuGetSmokeTest/EasyOcrNet.NuGetSmokeTest.csproj
   ```

   Il programma carica `examples/generated_1.png`, individua i modelli copiati dal
   pacchetto (sotto `contentFiles/any/any/models/onnx`) e stampa le righe
   riconosciute. Nessun download supplementare è necessario.

3. *(Opzionale)* Rimuovi la sorgente locale quando non serve più:

   ```bash
   dotnet nuget remove source easyocrnet-local
   ```
