# Data

# Spanish Literary Corpus (`data/`)

This directory contains raw, cleaned, and tokenized datasets for training and evaluating the Spanish Mini-GPT model.

_Note: Nothing inside `raw/`, `clean/`, or `tokenized/` is committed to Git. All dataset files are fully reproducible and regenerated on demand via `src/prepare_corpus.py` (see `REPRODUCE.md`)._

---

## Directory Structure

- `data/raw/` — Unmodified, cached plain-text files directly downloaded from Project Gutenberg.
- `data/clean/` — UTF-8 normalized texts with Gutenberg legal headers, footers, transcriber metadata, and footnotes removed.
  - `data/clean/train/` — Training split (25 complete works).
  - `data/clean/val/` — Validation split (5 held-out works: 1 major novel + 4 shorter poetry/story collections).
- `data/tokenized/` — BPE tokenizer and uint16 token streams (`train.bin`, `val.bin`, `tiny.bin`, `meta.json`, `tokenizer.json`); see *Tokenization* below.

---

## Sources & Corpus Catalog

| Work Title                         | Author                 | Gutenberg ID | Clean Filename                       |   Split   |
| :--------------------------------- | :--------------------- | :----------: | :----------------------------------- | :-------: |
| Don Quijote                        | Miguel de Cervantes    |    `2000`    | `donquijote.txt`                     |  `train`  |
| Novelas Ejemplares                 | Miguel de Cervantes    |   `61202`    | `novelasejemplares.txt`              |  `train`  |
| Obras Escogidas                    | Gustavo Adolfo Bécquer |   `53552`    | `obrasescogidas.txt`                 |  `train`  |
| Poesias Completas                  | Antonio Machado        |   `68525`    | `poesiascompletas.txt`               | **`val`** |
| Paginas Escogidas                  | Antonio Machado        |   `49437`    | `paginasescogidas.txt`               |  `train`  |
| Romancero Gitano                   | Federico García Lorca  |   `72665`    | `romancerogitano.txt`                |  `train`  |
| Libro de poemas                    | Federico García Lorca  |   `75703`    | `librodepoemas.txt`                  |  `train`  |
| Fortunata y Jacinta                | Benito Pérez Galdós    |   `17013`    | `fortunatayjacinta.txt`              |  `train`  |
| Misericordia                       | Benito Pérez Galdós    |   `21831`    | `misericordia.txt`                   | **`val`** |
| Trafalgar                          | Benito Pérez Galdós    |   `16961`    | `trafalgar.txt`                      |  `train`  |
| Tormento                           | Benito Pérez Galdós    |   `31465`    | `tormento.txt`                       |  `train`  |
| Marianela                          | Benito Pérez Galdós    |   `17340`    | `marianela.txt`                      |  `train`  |
| Los pazos de Ulloa                 | Emilia Pardo Bazán     |   `18005`    | `lospazosdeulloa.txt`                |  `train`  |
| Insolacion y Morriña               | Emilia Pardo Bazán     |   `52597`    | `insolacionymorrina.txt`             |  `train`  |
| La Tribuna                         | Emilia Pardo Bazán     |   `17491`    | `latribuna.txt`                      | **`val`** |
| Cuentos de Amor                    | Emilia Pardo Bazán     |   `55514`    | `cuentosdeamor.txt`                  |  `train`  |
| Los cuatro jinetes del apocalipsis | Vicente Blasco Ibáñez  |   `24536`    | `loscuatrojinetesdelapocalipsis.txt` |  `train`  |
| Mare Nostrum                       | Vicente Blasco Ibáñez  |   `23236`    | `marenostrum.txt`                    |  `train`  |
| Los Argonautas                     | Vicente Blasco Ibáñez  |   `25640`    | `losargonautas.txt`                  |  `train`  |
| Cañas y Barro                      | Vicente Blasco Ibáñez  |   `57781`    | `canasybarro.txt`                    | **`val`** |
| La Regenta                         | Leopoldo Alas          |   `17073`    | `laregenta.txt`                      |  `train`  |
| El Señor y los demás son Cuentos   | Leopoldo Alas          |   `62361`    | `elsenorylosdemassoncuentos.txt`     |  `train`  |
| Su Unico Hijo                      | Leopoldo Alas          |   `17341`    | `suunicohijo.txt`                    |  `train`  |
| Pepita Jimenez                     | Juan Valera            |   `17223`    | `pepitajimenez.txt`                  |  `train`  |
| Juanita la Larga                   | Juan Valera            |   `16484`    | `juanitalalarga.txt`                 |  `train`  |
| Cuentos y Dialogos                 | Juan Valera            |   `27214`    | `cuentosydialogos.txt`               | **`val`** |
| En las orillas del Sar             | Rosalía de Castro      |   `70984`    | `enlasorillasdelsar.txt`             |  `train`  |
| Sotileza                           | José María de Pereda   |   `49388`    | `sotileza.txt`                       |  `train`  |
| La Puchera                         | José María de Pereda   |   `55058`    | `lapuchera.txt`                      |  `train`  |
| La Montalvez                       | José María de Pereda   |   `25812`    | `lamontalvez.txt`                    |  `train`  |

---

## Split Strategy & Leakage Prevention

- **Train Set**: 25 works (~16.59 MB clean text) representing major Spanish Realist, Romantic, and Modernist literature.
- **Validation Set**: 5 works (~1.72 MB clean text) held out in their entirety to prevent data leakage during language model evaluation.

---

## Tokenization (step 5)

Byte-level BPE trained with Hugging Face `tokenizers` on the **training split only**
(`src/train_tokenizer.py`), then applied to both splits (`src/encode.py`).
Regenerate with `sbatch slurm/tokenize.sbatch`.

| Setting | Value |
| :-- | :-- |
| Model | Byte-level BPE (GPT-2 style pre-tokenizer, no normalizer, no `<unk>`) |
| Vocabulary size | 8,192 (incl. 1 special token) |
| Special token | `<\|endoftext\|>` = id 0, appended after every work |
| Trained on | 25 train works (16.587 MB); val works never seen |
| Tokenizer SHA-256 | `2fca7f76fb9d42d0…` (full hash in `meta.json`) |

### Output files (`data/tokenized/`)

| File | Contents | Tokens | Size |
| :-- | :-- | --: | --: |
| `train.bin` | 25 works + 25 EOT | 4,761,128 | 9.08 MB |
| `val.bin` | 5 works + 5 EOT | 518,383 | 0.99 MB |
| `tiny.bin` | first 1,500,000 tokens of `train.bin` | 1,500,000 | 2.86 MB |
| `tokenizer.json` | trained tokenizer | | |
| `meta.json` | vocab size, dtype, EOT id, per-file token counts, SHA-256 of every file | | |

Format: flat little-endian **uint16** token ids, no header. Load with
`np.memmap("data/tokenized/train.bin", dtype=np.uint16, mode="r")`.

Compression: 3.56 chars/token on train, 3.397 on val (val is
slightly worse since the tokenizer never saw it and it contains more poetry).

### Token counts per work

| File | Split | Chars | Tokens | Chars/token |
| :-- | :-: | --: | --: | --: |
| `cuentosdeamor.txt` | train | 375,269 | 107,730 | 3.48 |
| `donquijote.txt` | train | 2,110,726 | 583,362 | 3.62 |
| `elsenorylosdemassoncuentos.txt` | train | 298,525 | 85,285 | 3.50 |
| `enlasorillasdelsar.txt` | train | 161,078 | 46,820 | 3.44 |
| `fortunatayjacinta.txt` | train | 2,241,393 | 631,606 | 3.55 |
| `insolacionymorrina.txt` | train | 485,552 | 142,623 | 3.40 |
| `juanitalalarga.txt` | train | 413,708 | 116,331 | 3.56 |
| `lamontalvez.txt` | train | 602,297 | 167,742 | 3.59 |
| `lapuchera.txt` | train | 694,766 | 197,266 | 3.52 |
| `laregenta.txt` | train | 1,786,173 | 495,712 | 3.60 |
| `librodepoemas.txt` | train | 104,438 | 33,050 | 3.16 |
| `losargonautas.txt` | train | 1,223,629 | 336,643 | 3.63 |
| `loscuatrojinetesdelapocalipsis.txt` | train | 801,678 | 217,554 | 3.68 |
| `lospazosdeulloa.txt` | train | 493,670 | 142,312 | 3.47 |
| `marenostrum.txt` | train | 873,370 | 241,376 | 3.62 |
| `marianela.txt` | train | 293,980 | 82,892 | 3.55 |
| `novelasejemplares.txt` | train | 1,035,143 | 289,449 | 3.58 |
| `obrasescogidas.txt` | train | 553,182 | 156,270 | 3.54 |
| `paginasescogidas.txt` | train | 91,157 | 30,548 | 2.98 |
| `pepitajimenez.txt` | train | 315,000 | 89,362 | 3.52 |
| `romancerogitano.txt` | train | 36,477 | 12,446 | 2.93 |
| `sotileza.txt` | train | 664,706 | 192,370 | 3.46 |
| `suunicohijo.txt` | train | 515,816 | 143,617 | 3.59 |
| `tormento.txt` | train | 477,885 | 135,379 | 3.53 |
| `trafalgar.txt` | train | 300,188 | 83,358 | 3.60 |
| `canasybarro.txt` | val | 443,432 | 126,630 | 3.50 |
| `cuentosydialogos.txt` | val | 223,940 | 68,328 | 3.28 |
| `latribuna.txt` | val | 382,827 | 113,299 | 3.38 |
| `misericordia.txt` | val | 480,278 | 141,397 | 3.40 |
| `poesiascompletas.txt` | val | 230,684 | 68,724 | 3.36 |

### Verification

`python src/test_roundtrip.py` reads the `.bin` files back with `np.memmap`, splits on EOT,
decodes every work and checks it equals the clean text byte-for-byte; it also checks
`tiny.bin` is a prefix of `train.bin`, all ids < vocab size, and a set of edge-case strings
(accents, ñ, ¿¡, «», em-dash, emoji). Rerunning the pipeline gives byte-identical files.
