# EasyOcrNet: Report di Parità Python vs C#

Data: 2025-10-25
Test image: `dataset/base/HAL.2015.page_42.pdf_125176.png` (393x56 px)

## 📊 Riepilogo Esecutivo

| Metrica | Python | C# | Stato |
|---------|--------|-----|-------|
| **Detections** | 10 | 13 | ⚠️ C# produce 3 detection in più |
| **Text Accuracy** | - | 0/10 (0.0%) | ❌ Nessun match esatto |
| **Bbox Accuracy** | - | 0/10 (0.0%) | ❌ Nessun bbox corrispondente |
| **Similarità Complessiva** | - | 65.3% | ⚠️ Moderata |
| **Tempo Totale** | ~0.23s | 0.19s | ✅ C# più veloce (17% faster) |
| **- Detection** | ~0.15s | 0.09s | ✅ C# 40% più veloce |
| **- Recognition** | ~0.08s | 0.10s | ⚠️ Python 20% più veloce |

## 🔍 Analisi Dettagliata delle Differenze

### 1. Numero di Detections

**Python:** 10 detections (con grouping completo)
**C#:** 13 detections (grouping parziale)

**Motivo:** L'algoritmo di grouping in C# non sta mergendo completamente i bounding box adiacenti come fa Python.

### 2. Confronto dei Primi 10 Risultati

| # | Python Text | C# Text | Similarità | Status |
|---|-------------|---------|------------|--------|
| 1 | Oil price | ZM | 0% | ❌ Completamente diverso |
| 2 | AW | Oillprice | 0% | ❌ Completamente diverso |
| 3 | 95i | M | 0% | ❌ Completamente diverso |
| 4 | Oil price | 64 | 0% | ❌ Completamente diverso |
| 5 | Brent | Vo.a | 0% | ❌ Completamente diverso |
| 6 | 52 16 | Oil price | 14% | ❌ Quasi nessuna somiglianza |
| 7 | Omt | Brent | 25% | ❌ Bassa somiglianza |
| 8 | 108.71 | 37 16 | 18% | ❌ Bassa somiglianza |
| 9 | Nlwal @aSmicc | ot | 0% | ❌ Completamente diverso |
| 10 | Henry Hub | 108.75 | 0% | ❌ Completamente diverso |

### 3. Output Completo

#### Python (10 detections):
```
(0,15) (37,15) (37,27) (0,27) | Oil price | 0.8614
(42,16) (72,16) (72,24) (42,24) | AW | 0.0191
(320,16) (344,16) (344,24) (320,24) | 95i | 0.4392
(0,29) (35,29) (35,41) (0,41) | Oil price | 0.6868
(42,30) (66,30) (66,38) (42,38) | Brent | 0.1578
(270,30) (292,30) (292,38) (270,38) | 52 16 | 0.2005
(318,30) (342,30) (342,38) (318,38) | Omt | 0.0079
(365,29) (391,29) (391,41) (365,41) | 108.71 | 0.1237
(0,46) (68,46) (68,54) (0,54) | Nlwal @aSmicc | 0.0883
(75,45) (123,45) (123,56) (75,56) | Henry Hub | 0.6348
```

#### C# (13 detections):
```
(360,0) (380,0) (380,8) (360,8) | ZM | 0,1762
(-1,13) (37,13) (37,27) (-1,27) | Oillprice | 0,3652          ⚠️ COORDINATE NEGATIVE
(42,16) (72,16) (72,24) (42,24) | M  | 0,0111
(282,16) (292,16) (292,24) (282,24) | 64 | 0,2352
(320,16) (344,16) (344,24) (320,24) | Vo.a | 0,0105
(-1,27) (35,27) (35,41) (-1,41) | Oil price | 0,9324          ⚠️ COORDINATE NEGATIVE
(42,30) (66,30) (66,38) (42,38) | Brent | 0,1348
(270,30) (292,30) (292,38) (270,38) | 37 16 | 0,0652
(318,30) (342,30) (342,38) (318,38) | ot | 0,0049
(363,27) (391,27) (391,41) (363,41) | 108.75 | 0,3120
(0,46) (68,46) (68,54) (0,54) | VM pAS LL | 0,0292
(76,46) (100,46) (100,54) (76,54) | Hcmr | 0,2846
(99,43) (123,43) (123,57) (99,57) | 'Hub | 0,2735
```

## 🐛 Problemi Critici Identificati

### 1. ⛔ BUG: Coordinate Negative nei Bounding Box

**Detections affette:**
- Detection #2: `(-1,13) (37,13) (37,27) (-1,27)` → "Oillprice"
- Detection #6: `(-1,27) (35,27) (35,41) (-1,41)` → "Oil price"

**Causa:** Bug nell'algoritmo `GroupTextBoxFlat()` durante il merging dei bbox adiacenti. Il calcolo del margine produce coordinate negative quando il bbox è vicino al bordo sinistro dell'immagine.

**Impatto:** I bbox con coordinate negative potrebbero causare errori durante il crop extraction per il recognition.

### 2. ⚠️ Grouping Incompleto

**Problema:** C# produce 13 detections invece di 10 perché alcuni bbox adiacenti non vengono mergati.

**Esempio:**
- Python: `(0,15) (37,15) (37,27) (0,27) | Oil price`
- C# split in:
  - `(-1,13) (37,13) (37,27) (-1,27) | Oillprice`
  - Altri frammenti separati

**Causa:** I parametri di threshold per il grouping (ycenter_ths, height_ths, width_ths) potrebbero non essere applicati correttamente nell'implementazione C#.

### 3. ⚠️ Ordine delle Detections

Le detections di C# non sono nello stesso ordine di Python, rendendo impossibile il confronto diretto indice-per-indice.

**Python ordina per:** Y-center (top-to-bottom, left-to-right)
**C# ordina per:** Non chiaro, sembra Y-center ma con differenze

## 📈 Progresso Implementazione

### ✅ Fasi Complete:

1. **Phase 1: Core Infrastructure** ✅
   - Models, Interfaces, CharsetLoader
   - SkiaSharp integration

2. **Phase 2: Detection Pipeline** ✅
   - CRAFT detector con ONNX
   - Preprocessing completo
   - Post-processing base

3. **Phase 3: Recognition Pipeline** ✅
   - CRNN recognizer con CTC decoding
   - Preprocessing grayscale + resize + normalize
   - Confidence calculation

4. **Bbox Grouping** ⚠️ (parziale)
   - Algoritmo `GroupTextBoxFlat()` implementato
   - Merging funziona ma con bug

### 🔨 Da Completare:

1. **Fix coordinate negative** ❌
   - Clamp coordinates a [0, image_size]
   - Fix margin calculation nel grouping

2. **Migliorare grouping** ❌
   - Debug parametri threshold
   - Verificare logica di merging
   - Test con diversi valori di threshold

3. **Ordinamento detections** ❌
   - Implementare ordinamento consistente
   - Ordinare per (Y, X) come Python

4. **Unit tests** ❌
   - Test per detection
   - Test per recognition
   - Test per grouping
   - Test end-to-end

## 🎯 Piano d'Azione

### Priorità Alta:
1. **Fix coordinate negative** (BUG critico)
2. **Fix ordinamento detections**
3. **Debug grouping algorithm**

### Priorità Media:
4. **Creare unit tests**
5. **Test su più immagini**
6. **Ottimizzare performance**

### Priorità Bassa:
7. **Documentazione API**
8. **Esempi d'uso**

## 📝 Note Tecniche

### Differenze Implementative:

**Python `group_text_box()`:**
- Usa NumPy per operazioni vettoriali
- Gestisce automaticamente edge cases
- Coordinate sempre valide

**C# `GroupTextBoxFlat()`:**
- Usa List<float[]> per polys
- Calcolo manuale delle distanze
- Possibili overflow/underflow nelle coordinate

### Performance:

**C# è più veloce** nel detection (-40%) grazie a:
- ONNX Runtime ottimizzato
- SkiaSharp efficiente
- Meno overhead Python

**Recognition simile** tra Python e C# perché entrambi usano ONNX Runtime.

## 📚 Conclusioni

**Stato Attuale:** ⚠️ Implementazione funzionante ma con differenze significative

**Similarità Complessiva:** 65.3% (testi combinati)

**Problemi Principali:**
1. Coordinate negative nei bbox (BUG)
2. Grouping incompleto
3. Ordinamento diverso

**Prossimi Step:**
1. Fix bug coordinate negative
2. Debug e fix grouping
3. Implementare ordinamento corretto
4. Creare test unitari
5. Raggiungere parità 100%

---

*Report generato automaticamente - 2025-10-25*
