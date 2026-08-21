# MyUni Multimodal Sentiment Analysis POC

English-first proof of concept for analyzing social activity sentiment on MyUni.

## Current status (Milestone 1)

**Working today:** end-to-end **English text** sentiment via a shared pipeline.

**Not implemented yet:** image, OCR, audio, video, SQLite storage, daily aggregation, Streamlit demo.

## Requirements

- Python 3.10+ recommended (3.9+ should work)
- ~16 GB RAM machine is fine; **CPU is supported**
- GPU (CUDA) is optional — used automatically when available, never required
- First run downloads the Hugging Face model weights (network required once)

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

- Loaded **lazily** on first analysis (not at import / process start)
- Labels: `positive` | `neutral` | `negative`
- POC score: `positive_probability - negative_probability` ≈ `[-1, +1]`
- This score is **not** the future client business scoring methodology

Cached by Hugging Face under the usual transformers cache directory after the first download.

## CLI

```powershell
python main.py "I really enjoyed today's workshop."

python main.py "This was terrible." --user-id U007 --activity-id ACT0042

python main.py "Campus wifi is fine." --log-level INFO
```

Blank / non-string input exits with code `2` and an error message on stderr.

## Tests

Fast validation-only tests (no model download):

```powershell
pytest tests/test_validation_unit.py -q
```

Full text inference smoke/unit suite (downloads model on first run):

```powershell
pytest tests/test_text_sentiment.py -q
```

Or everything:

```powershell
pytest -q
```

## Ambiguous social-media smoke script

Prints structured JSON for sarcastic / emoji-heavy examples. **No expected label is hardcoded.**

```powershell
python scripts/smoke_ambiguous.py
```

Examples:

- `Fantastic. Another surprise assignment.`
- `Amazing result again 💀`

## Output shape (text)

Results include activity metadata, overall sentiment, and `modalities.text` evidence (probabilities, model name, confidence, score).

## Project layout

```text
src/
  analyzers/text.py   # RoBERTa text analyzer
  pipeline.py         # MyUniSentimentPipeline
  schemas.py          # Pydantic result models
tests/
scripts/smoke_ambiguous.py
data/samples/
outputs/
main.py
```

## Current limitations

- Text modality only
- English only (no Hindi / Hinglish)
- No image / OCR / video / ASR
- No persistence (SQLite) or user-level daily aggregation
- No Streamlit UI
- First inference is slower due to model download + load
- Neutral vs mild polarity can be close on short, factual sentences
