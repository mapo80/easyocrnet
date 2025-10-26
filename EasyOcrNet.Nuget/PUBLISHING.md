# Guida alla Pubblicazione del Pacchetto NuGet EasyOcrNet

Questa guida documenta il processo completo per creare e pubblicare una nuova versione del pacchetto NuGet EasyOcrNet.

## Indice

1. [Prerequisiti](#prerequisiti)
2. [Preparazione](#preparazione)
3. [Build del Pacchetto](#build-del-pacchetto)
4. [Test Locale](#test-locale)
5. [Pubblicazione GitHub Release](#pubblicazione-github-release)
6. [Pubblicazione NuGet.org](#pubblicazione-nugetorg-opzionale)
7. [Checklist](#checklist)
8. [Troubleshooting](#troubleshooting)

## Prerequisiti

### Software Necessario

- **Git** - Version control
- **GitHub CLI (`gh`)** - Per creare release
- **.NET SDK 8.0+** - Per build del pacchetto
- **Python 3.x** - Per script di download modelli

### Verifiche Iniziali

```bash
# Verifica Git
git --version

# Verifica GitHub CLI
gh --version
gh auth status

# Verifica .NET
dotnet --version

# Verifica Python
python --version
```

## Preparazione

### 1. Download Modelli ONNX

I modelli devono essere scaricati una sola volta (se non già presenti):

```bash
cd /path/to/easyocrnet
python tools/download_torchfree_models.py
```

Verifica che i modelli siano in `models/cpu/`:
- `detection.onnx` (79 MB)
- `latin_g2_rec.onnx` (15 MB)
- `english_g2_rec.onnx` (14 MB)

### 2. Copia Modelli nella Struttura NuGet

```bash
# Crea directory se non esistono
mkdir -p EasyOcrNet.Nuget/contentFiles/any/any/models
mkdir -p EasyOcrNet.Nuget/contentFiles/any/any/character

# Copia modelli ONNX
cp models/cpu/detection.onnx EasyOcrNet.Nuget/contentFiles/any/any/models/
cp models/cpu/latin_g2_rec.onnx EasyOcrNet.Nuget/contentFiles/any/any/models/
cp models/cpu/english_g2_rec.onnx EasyOcrNet.Nuget/contentFiles/any/any/models/

# Copia character files
cp character/en_charset.txt EasyOcrNet.Nuget/contentFiles/any/any/character/
cp character/it_charset.txt EasyOcrNet.Nuget/contentFiles/any/any/character/
cp character/latin_char.txt EasyOcrNet.Nuget/contentFiles/any/any/character/
```

### 3. Aggiornamento Versione

Modifica `EasyOcrNet.Nuget/EasyOcrNet.Nuget.csproj`:

```xml
<PropertyGroup>
  <TargetFramework>net8.0</TargetFramework>
  <PackageId>EasyOcrNet</PackageId>
  <Version>1.0.0</Version>  <!-- ⬅️ AGGIORNA QUESTO -->
  <Authors>EasyOcrNet</Authors>
  <Description>EasyOcrNet - .NET OCR library with ONNX models...</Description>
  ...
</PropertyGroup>
```

**Schema di versioning**: Usa [Semantic Versioning](https://semver.org/)
- `MAJOR.MINOR.PATCH` (es. 1.0.0)
- MAJOR: Breaking changes
- MINOR: Nuove funzionalità (backward compatible)
- PATCH: Bug fixes

## Build del Pacchetto

### 1. Clean Build

```bash
cd EasyOcrNet.Nuget

# Pulisci build precedenti
dotnet clean

# Build in Release
dotnet pack -c Release
```

### 2. Verifica Output

Il pacchetto viene creato in: `../nupkgs/EasyOcrNet.{version}.nupkg`

```bash
# Verifica esistenza
ls -lh ../nupkgs/EasyOcrNet.*.nupkg

# Output atteso:
# -rw-r--r--  1 user  staff   201M Oct 26 07:05 EasyOcrNet.1.0.0.nupkg
```

**Dimensione attesa**: ~201 MB

### 3. Ispeziona Contenuto Pacchetto

```bash
# Elenca contenuto
unzip -l ../nupkgs/EasyOcrNet.1.0.0.nupkg

# Verifica file critici
unzip -l ../nupkgs/EasyOcrNet.1.0.0.nupkg | grep -E "build|contentFiles" | grep -E "onnx|targets|txt"

# Deve contenere:
# ✅ build/EasyOcrNet.targets
# ✅ build/models/detection.onnx
# ✅ build/models/latin_g2_rec.onnx
# ✅ build/models/english_g2_rec.onnx
# ✅ build/character/*.txt
# ✅ contentFiles/any/any/models/*.onnx
# ✅ contentFiles/any/any/character/*.txt
```

## Test Locale

**⚠️ IMPORTANTE**: Testa sempre il pacchetto prima di pubblicarlo!

### 1. Pulisci Cache NuGet

```bash
dotnet nuget locals all --clear
```

### 2. Test con EasyOcrNet.CliNuget

```bash
cd ../EasyOcrNet.CliNuget

# Restore e build
dotnet restore
dotnet build -c Release

# Verifica copia modelli
ls -lh bin/Release/net9.0/models/
ls -lh bin/Release/net9.0/character/

# Output atteso:
# models/
#   detection.onnx (79M)
#   latin_g2_rec.onnx (15M)
#   english_g2_rec.onnx (14M)
# character/
#   en_charset.txt
#   it_charset.txt
#   latin_char.txt
```

### 3. Test Funzionale

```bash
cd bin/Release/net9.0

# Test su dataset italiano
./EasyOcrNet.CliNuget /path/to/dataset/it

# Output atteso:
# - Nessun errore di "file not found"
# - OCR funzionante
# - Risultati con confidence
```

### 4. Test su Nuovo Progetto (Opzionale)

Crea un progetto di test completamente nuovo:

```bash
# Crea nuovo progetto
mkdir /tmp/test-easyocr
cd /tmp/test-easyocr
dotnet new console

# Configura sorgente locale
dotnet nuget add source /path/to/easyocrnet/nupkgs --name test-local

# Aggiungi pacchetto
dotnet add package EasyOcrNet --version 1.0.0
dotnet add package SkiaSharp

# Build
dotnet build

# Verifica modelli
ls bin/Debug/net8.0/models/
```

## Pubblicazione GitHub Release

### 1. Crea RELEASE_NOTES.md

Crea o aggiorna `RELEASE_NOTES.md` nella root del progetto:

```markdown
# EasyOcrNet v1.0.0 - Release Title

## 🚀 Novità

- Feature 1
- Feature 2

## 🐛 Bug Fix

- Fix 1
- Fix 2

## 📊 Performance

- Benchmark results

## ⚠️ Breaking Changes

- Change 1 (se presente)

## 📋 Contenuto Pacchetto

- detection.onnx (79 MB)
- latin_g2_rec.onnx (15 MB)
- english_g2_rec.onnx (14 MB)
- Character sets
```

Vedi [RELEASE_NOTES.md](../RELEASE_NOTES.md) per esempio completo.

### 2. Commit e Push Modifiche

```bash
cd /path/to/easyocrnet

# Verifica stato
git status

# Aggiungi modifiche
git add EasyOcrNet.Nuget/EasyOcrNet.Nuget.csproj
git add RELEASE_NOTES.md
git add .  # Se ci sono altre modifiche

# Commit
git commit -m "Release v1.0.0: Update package version and release notes"

# Push
git push origin main  # o development, a seconda del branch
```

### 3. Crea Tag Git

```bash
# Crea tag annotato
git tag -a v1.0.0 -m "EasyOcrNet v1.0.0 - Release Title"

# Verifica tag
git tag -l

# Push tag
git push origin v1.0.0
```

**Convenzioni tag**:
- Formato: `v{MAJOR}.{MINOR}.{PATCH}` (es. v1.0.0)
- Usa tag annotati (`-a`) non lightweight
- Messaggio descrittivo

### 4. Crea GitHub Release

```bash
# Assicurati di essere nella root del repository
cd /path/to/easyocrnet

# Crea release con GitHub CLI
gh release create v1.0.0 \
  --title "EasyOcrNet v1.0.0 - Release Title" \
  --notes-file RELEASE_NOTES.md \
  nupkgs/EasyOcrNet.1.0.0.nupkg

# Output:
# https://github.com/mapo80/easyocrnet/releases/tag/v1.0.0
```

**Parametri**:
- `v1.0.0`: Tag della release
- `--title`: Titolo della release (visibile su GitHub)
- `--notes-file`: File con descrizione dettagliata
- Ultimo argomento: File da allegare (il pacchetto NuGet)

### 5. Verifica Release

```bash
# Visualizza release
gh release view v1.0.0

# Verifica asset allegato
gh release view v1.0.0 --json assets

# Output atteso:
# {
#   "assets": [
#     {
#       "name": "EasyOcrNet.1.0.0.nupkg",
#       "size": 210763776,  # ~201 MB
#       ...
#     }
#   ]
# }
```

### 6. Verifica su GitHub Web

Visita: `https://github.com/mapo80/easyocrnet/releases`

Verifica:
- ✅ Release visibile
- ✅ Titolo e descrizione corretti
- ✅ Asset `EasyOcrNet.1.0.0.nupkg` presente
- ✅ Download funzionante

## Pubblicazione NuGet.org (Opzionale)

Se vuoi pubblicare su NuGet.org pubblico:

### 1. Ottieni API Key

1. Vai su https://www.nuget.org
2. Registrati/Login
3. Vai su Account → API Keys
4. Crea nuova API key:
   - Name: `EasyOcrNet Publishing`
   - Scopes: `Push`
   - Packages: `EasyOcrNet`
   - Expiration: Scegli durata

### 2. Pubblica Pacchetto

```bash
dotnet nuget push nupkgs/EasyOcrNet.1.0.0.nupkg \
  --api-key YOUR_API_KEY_HERE \
  --source https://api.nuget.org/v3/index.json
```

### 3. Verifica Pubblicazione

Dopo alcuni minuti, visita:
`https://www.nuget.org/packages/EasyOcrNet/`

**⚠️ Note per NuGet.org**:

Il pacchetto attuale è **201 MB** - troppo grande per NuGet.org standard.

Considera:
- **Pacchetti separati**: `EasyOcrNet` (libreria) + `EasyOcrNet.Models` (modelli)
- **Download dinamico**: Scarica modelli al primo utilizzo
- **Modelli quantizzati**: Riduci dimensioni modelli

## Checklist

Prima di pubblicare, verifica:

### Pre-Build
- [ ] Modelli ONNX copiati in `EasyOcrNet.Nuget/contentFiles/`
- [ ] Character files copiati in `EasyOcrNet.Nuget/contentFiles/`
- [ ] Versione aggiornata in `EasyOcrNet.Nuget.csproj`
- [ ] RELEASE_NOTES.md creato/aggiornato

### Build
- [ ] `dotnet clean` eseguito
- [ ] `dotnet pack -c Release` completato senza errori
- [ ] Pacchetto creato in `nupkgs/`
- [ ] Dimensione pacchetto ~201 MB

### Test
- [ ] Cache NuGet pulita
- [ ] EasyOcrNet.CliNuget builda correttamente
- [ ] Modelli copiati in bin/Release/net9.0/
- [ ] Test funzionale su immagini completato
- [ ] Nessun errore "file not found"

### Git & GitHub
- [ ] Modifiche committate
- [ ] Push su remote completato
- [ ] Tag `v{version}` creato
- [ ] Tag pushato su GitHub
- [ ] Release GitHub creata
- [ ] Asset .nupkg allegato alla release
- [ ] Release verificata su GitHub web

### Post-Release
- [ ] Download pacchetto da release testato
- [ ] Installazione da release funzionante
- [ ] Documentazione aggiornata (README.md)

## Troubleshooting

### Problema: Modelli non vengono copiati

**Sintomo**: `bin/Release/net9.0/models/` è vuota dopo build

**Cause possibili**:
1. `EasyOcrNet.targets` non nel pacchetto
2. Path errati in targets file
3. ProjectReference ha priorità su PackageReference

**Soluzioni**:

```bash
# 1. Verifica targets nel pacchetto
unzip -l nupkgs/EasyOcrNet.1.0.0.nupkg | grep targets
# Deve mostrare: build/EasyOcrNet.targets

# 2. Verifica path in targets
cat EasyOcrNet.Nuget/EasyOcrNet.targets
# Path deve essere: $(MSBuildThisFileDirectory)..\contentFiles\any\any\models\

# 3. Rimuovi temporaneamente ProjectReference
# In EasyOcrNet.CliNuget.csproj:
# Commenta: <!-- <ProjectReference Include="..\EasyOcrNet\EasyOcrNet.csproj" /> -->

# 4. Rebuild
dotnet nuget locals all --clear
dotnet restore
dotnet build -c Release
```

### Problema: Pacchetto troppo grande

**Sintomo**: Pacchetto > 250 MB

**Causa**: Modelli duplicati in contentFiles e build

**È normale**: Il pacchetto include modelli 2 volte per compatibilità:
- `contentFiles/any/any/models/` - Standard NuGet
- `build/models/` - Per MSBuild targets

**Dimensione attesa**: ~201 MB (79+15+14 = 108 MB × 2 ≈ 200 MB + overhead)

**Soluzioni future**:
- Pubblicare modelli separatamente
- Download on-demand
- Modelli quantizzati

### Problema: "Package already exists" su NuGet.org

**Sintomo**: Errore durante push su NuGet.org

**Causa**: Versione già pubblicata (NuGet.org non permette override)

**Soluzione**:
```bash
# Incrementa versione
# In EasyOcrNet.Nuget.csproj:
# <Version>1.0.0</Version> → <Version>1.0.1</Version>

# Rebuild
dotnet pack -c Release

# Pubblica nuova versione
dotnet nuget push nupkgs/EasyOcrNet.1.0.1.nupkg ...
```

### Problema: GitHub CLI non autenticato

**Sintomo**: `gh` comandi falliscono con "authentication required"

**Soluzione**:
```bash
# Login con GitHub CLI
gh auth login

# Seleziona:
# - GitHub.com
# - HTTPS
# - Login with web browser
# - Segui istruzioni

# Verifica
gh auth status
```

### Problema: Tag già esistente

**Sintomo**: `fatal: tag 'v1.0.0' already exists`

**Soluzione**:
```bash
# Elimina tag locale
git tag -d v1.0.0

# Elimina tag remoto (SE necessario)
git push origin :refs/tags/v1.0.0

# Ricrea tag con versione corretta
git tag -a v1.0.0 -m "Message"
git push origin v1.0.0
```

## Script di Automazione (Futuro)

Considera di creare uno script `publish.sh`:

```bash
#!/bin/bash
VERSION=$1

if [ -z "$VERSION" ]; then
  echo "Usage: ./publish.sh <version>"
  echo "Example: ./publish.sh 1.0.0"
  exit 1
fi

# 1. Update version
sed -i '' "s/<Version>.*<\/Version>/<Version>$VERSION<\/Version>/" EasyOcrNet.Nuget/EasyOcrNet.Nuget.csproj

# 2. Build package
cd EasyOcrNet.Nuget
dotnet clean
dotnet pack -c Release
cd ..

# 3. Test
# ... test logic ...

# 4. Git tag
git tag -a "v$VERSION" -m "Release v$VERSION"
git push origin "v$VERSION"

# 5. Create release
gh release create "v$VERSION" \
  --title "EasyOcrNet v$VERSION" \
  --notes-file RELEASE_NOTES.md \
  "nupkgs/EasyOcrNet.$VERSION.nupkg"

echo "✅ Release v$VERSION pubblicata!"
```

## Riferimenti

- [NuGet Package Authoring Best Practices](https://learn.microsoft.com/en-us/nuget/create-packages/package-authoring-best-practices)
- [Semantic Versioning](https://semver.org/)
- [GitHub CLI Manual](https://cli.github.com/manual/)
- [MSBuild Targets](https://learn.microsoft.com/en-us/visualstudio/msbuild/msbuild-targets)

---

**Ultima revisione**: Ottobre 2025
**Versione documento**: 1.0
