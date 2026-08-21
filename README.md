# MyUni Multimodal Sentiment Analysis POC

English-first proof of concept for analyzing social activity sentiment on MyUni.

## Current status (Milestone 2)

**Working today:**
- End-to-end **English text** sentiment via a shared pipeline
- Validated **activity contract** (`text` / `image` / `video`)
- **JSONL batch ingestion** with per-record errors and summary metrics

**Recognized but not analyzed yet:** image and video activities (reported as `unsupported`, no fake sentiment).

**Not implemented yet:** image/OCR/audio/video models, SQLite storage, daily aggregation, Streamlit demo.

## Requirements

- Python 3.10+ recommended (3.9+ should work)
- ~16 GB RAM machine is fine; **CPU is supported**
- GPU (CUDA) is optional — used automatically when available, never required
- First text analysis downloads Hugging Face model weights (network required once)

## Setup (Windows-friendly)

```powershell
cd D:\Work\myuni-sentiment-poc
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Model

Text classifier:

`cardiffnlp/twitter-roberta-base-sentiment-latest`

- Loaded **lazily** on first text analysis (not at import / process start)
- Labels: `positive` | `neutral` | `negative`
- POC score: `positive_probability - negative_probability` ≈ `[-1, +1]`
- This score is **not** the future client business scoring methodology

## Activity contract (batch)

Each JSONL line is one activity object:

| Field | Rule |
| --- | --- |
| `activity_id` | Required non-blank |
| `user_id` | Required non-blank |
| `activity_type` | `text` \| `image` \| `video` |
| `text` | Required non-blank for `text`; optional caption for `image`/`video` |
| `media_path` | Required for `image`/`video` |
| `created_at` | Required ISO-8601 timestamp |
| `metadata` | Optional object |
| `content_kind` | Optional reserved (`post`/`comment`/`story`/…); ignored by MVP routing |

Example:

```json
{"activity_id":"ACT001","user_id":"U001","activity_type":"text","text":"I loved today's event.","created_at":"2026-08-21T10:00:00+05:30"}
```

## CLI

### Single text (Milestone 1, still supported)

```powershell
python main.py "I really enjoyed today's workshop."

python main.py "This was terrible." --user-id U007 --activity-id ACT0042
```

Blank / non-string input exits with code `2`.

### JSONL batch (Milestone 2)

```powershell
python main.py --batch data/samples/activities.jsonl
```

Batch behavior:
1. Read each non-empty line independently
2. Validate against the activity contract
3. Process valid **text** activities with the real text analyzer
4. Mark valid **image**/**video** as `unsupported` (no fake scores)
5. Keep going after bad records
6. Emit per-record outcomes plus summary metrics:
   `total`, `valid`, `invalid`, `processed`, `unsupported`, `failed`

Exit code `1` if any `failed` records, or if nothing was processed while invalid/failed records exist.

## Tests

Fast tests (no model download for most cases):

```powershell
pytest tests/test_validation_unit.py tests/test_batch.py -q -m "not integration"
```

Full suite (includes real text inference):

```powershell
pytest -q
```

## Ambiguous social-media smoke script

```powershell
python scripts/smoke_ambiguous.py
```

## Project layout

```text
src/
  analyzers/text.py   # RoBERTa text analyzer
  batch.py            # JSONL ingestion + metrics
  pipeline.py         # MyUniSentimentPipeline
  schemas.py          # ActivityInput + result models
tests/
data/samples/activities.jsonl
main.py
```

## Current limitations

- Only text activities produce sentiment
- Image/video are validated and acknowledged, not analyzed
- English only (no Hindi / Hinglish)
- No SQLite persistence or user-level daily aggregation
- No Streamlit UI
- First text inference is slower due to model download + load
