# MyUni Multimodal Sentiment Analysis POC

English-first proof of concept for analyzing social activity sentiment on MyUni.

## Current status (Milestone 3)

**Working today:**
- End-to-end **English text** sentiment
- Validated activity contract + **JSONL batch ingestion**
- **Image** activities with:
  - zero-shot **visual** sentiment (SigLIP 2)
  - **OCR** text extraction + OCR text sentiment (when Tesseract is available)
  - optional **caption** text sentiment kept as a separate modality
  - simple explainable **POC fusion** into `analysis.overall`

**Recognized but not analyzed yet:** video (reported as `unsupported`).

**Not implemented yet:** video/audio/ASR, SQLite storage, daily aggregation, Streamlit demo.

## Requirements

- Python 3.10+ recommended
- ~16 GB RAM; **CPU is supported**; GPU optional
- First text/visual analysis downloads Hugging Face weights (network once)
- Optional: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) binary for OCR (pipeline continues without it)

## Setup (Windows-friendly)

```powershell
cd D:\Work\myuni-sentiment-poc
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/generate_sample_image.py
```

### Tesseract OCR (optional, Windows)

1. Install the UB Mannheim Windows build: https://github.com/UB-Mannheim/tesseract/wiki
2. Ensure `tesseract.exe` is on `PATH` (installer option), or set:
   ```powershell
   $env:TESSDATA_PREFIX = "C:\Program Files\Tesseract-OCR\tessdata"
   ```
3. Verify: `tesseract --version`
4. If Tesseract is missing, image analysis still returns visual (+ caption) evidence and a clear `OCR unavailable` warning — it does **not** crash the pipeline.

## Models

### Text

`cardiffnlp/twitter-roberta-base-sentiment-latest`

- Lazy-loaded; score = `P(positive) - P(negative)` ≈ `[-1, +1]`

### Visual (zero-shot, not a trained sentiment classifier)

Exact checkpoint: **`google/siglip2-base-patch16-224`**

Candidate concept prompts (configurable in `src/config.py`):

- positive → `"a positive and pleasant situation"`
- neutral → `"a neutral everyday situation"`
- negative → `"a negative or unpleasant situation"`

Probabilities are a softmax over the three concept logits. Raw SigLIP sigmoid similarities are preserved under `details.raw_similarities`.

### OCR

`pytesseract` + system Tesseract. Meaningful OCR text is scored with the same text model; OCR evidence stays separate from caption (`modalities.text`) and visual (`modalities.visual`).

## Image overall fusion (POC-only)

Documented in `src/fusion.py` / `src/config.py`:

- confidence-weighted average of available modality scores (`text` caption, `visual`, `ocr`)
- label from fused score vs a small neutral band
- **Not** the future client business scoring methodology

## Activity contract (batch)

| Field | Rule |
| --- | --- |
| `activity_id` / `user_id` | Required non-blank |
| `activity_type` | `text` \| `image` \| `video` |
| `text` | Required for `text`; optional caption for `image`/`video` |
| `media_path` | Required for `image`/`video` |
| `created_at` | Required ISO-8601 |
| `metadata` / `content_kind` | Optional |

## CLI

```powershell
# Single text
python main.py "I really enjoyed today's workshop."

# Batch (text + image processed; video unsupported)
python main.py --batch data/samples/activities.jsonl
```

Sample image (synthetic, generated locally — not a copyrighted asset):

`data/samples/synthetic_sample.png` via `python scripts/generate_sample_image.py`

## Tests

```powershell
# Fast (skips real HF visual download)
pytest -q -m "not integration"

# Full (includes text + SigLIP smoke)
pytest -q
```

Image unit tests cover: openable sample, missing path, corrupt file, OCR unavailable, OCR empty, caption independence, schema shape. They do **not** assert brittle ML probabilities.

## Project layout

```text
src/
  analyzers/text.py
  analyzers/visual.py
  analyzers/ocr.py
  analyzers/image.py
  batch.py
  config.py
  fusion.py
  pipeline.py
  schemas.py
```

## Current limitations

- Video/audio not implemented
- SigLIP prompts are heuristic zero-shot concepts, not supervised sentiment labels
- OCR quality depends on image clarity and Tesseract install
- English only
- No SQLite / aggregation / Streamlit yet
- First visual load downloads ~weights and uses additional RAM alongside the text model
