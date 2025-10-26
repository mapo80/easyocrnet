# EasyOcrNet.CliNuget

Applicazione console dimostrativa che utilizza **EasyOcrNet** tramite il pacchetto NuGet **EasyOcrNet**.

## Caratteristiche

- Utilizza il pacchetto NuGet `EasyOcrNet` che include:
  - Modelli ONNX (detection.onnx, latin_g2_rec.onnx, english_g2_rec.onnx)
  - File di charset per diverse lingue
- Processa batch di immagini da una directory
- Mostra statistiche e risultati OCR in tempo reale
- Supporto per italiano e altre lingue latine

## Requisiti

- .NET 9.0 o superiore
- Pacchetto NuGet locale: `EasyOcrNet` (versione 1.0.0+)

## Installazione

Il progetto è già configurato per utilizzare il pacchetto NuGet locale. I modelli vengono automaticamente copiati nella directory di output durante la build.

## Build

```bash
dotnet build -c Release
```

Questo compila il progetto e copia automaticamente i modelli ONNX (108MB) nella directory `bin/Release/net9.0/models/`.

## Utilizzo

```bash
# Eseguire dalla directory bin
cd bin/Release/net9.0
./EasyOcrNet.CliNuget /path/to/images

# Esempio con dataset italiano
./EasyOcrNet.CliNuget /Users/politom/Documents/Workspace/personal/easyocrnet/dataset/it
```

## Output

L'applicazione mostra:
- Numero di immagini trovate
- Per ogni immagine:
  - Dimensioni
  - Numero di rilevamenti
  - Tempo di elaborazione
  - Primi 5 risultati OCR con confidenza
- Statistiche finali:
  - Totale immagini processate
  - Totale rilevamenti
  - Tempo totale e medio per immagine

## Risultati Test (Dataset Italiano)

Test eseguiti su 4 immagini italiane (1024x1024 e 2480x3508):

```
Total images processed: 4
Total detections: 99
Total time: 17323ms
Average time per image: 4330ms

Per-image results:
  doc-it-04.png: 43 detections in 10040ms (2480x3508)
  doc-it-01.png: 16 detections in 2162ms (1024x1024)
  doc-it-02.png: 14 detections in 2046ms (1024x1024)
  doc-it-03.png: 26 detections in 2700ms (1024x1024)
```

## Struttura

- **Program.cs** - Logica principale dell'applicazione
- **EasyOcrNet.CliNuget.csproj** - File di progetto con riferimenti a:
  - `EasyOcrNet` (project reference per sviluppo)
  - `EasyOcrNet` (NuGet package)
  - `SkiaSharp` (per elaborazione immagini)

## Note

- I modelli ONNX sono inclusi nel pacchetto NuGet e pesano circa 108MB
- Il pacchetto viene gestito tramite la source locale `./nupkgs`
- I modelli vengono copiati automaticamente grazie al file `EasyOcrNet.targets`
- **ProjectReference vs PackageReference**: Il progetto include entrambi i riferimenti. Durante lo sviluppo, il `ProjectReference` ha priorità. Per testare il pacchetto NuGet standalone, rimuovere temporaneamente il `ProjectReference` dal file `.csproj`
