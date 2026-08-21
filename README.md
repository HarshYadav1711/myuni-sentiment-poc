# MyUni Multimodal Sentiment Analysis POC

English-first proof of concept for analyzing social activity sentiment on MyUni.

## Current status (Milestone 4)

**Working today:**
- End-to-end **English text** sentiment
- Validated activity contract + **JSONL batch ingestion**
- **Image** activities (SigLIP 2 visual + OCR + caption + POC fusion)
- **Speech/audio branch** for future video: FFmpeg extraction + faster-whisper ASR + transcript sentiment (`AudioAnalyzer` / `pipeline.analyze_speech`)

**Not wired yet:** full **video** activities (frame sampling + multimodal fusion). Batch still reports video as `unsupported`.

**Not implemented yet:** SQLite storage, daily aggregation, Streamlit demo.

## Requirements

- Python 3.10+ recommended
- ~16 GB RAM; **CPU is supported**; GPU optional
- First text/visual/ASR analysis downloads model weights (network once)
- Optional: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for image OCR
- **Required for speech branch:** [FFmpeg](https://ffmpeg.org/) on `PATH`

## Setup (Windows-friendly)

```powershell
cd D:\Work\myuni-sentiment-poc
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/generate_sample_image.py
```

### FFmpeg (required for speech/audio extraction)

1. Install via winget: `winget install Gyan.FFmpeg`
   - Or download a release build: https://www.gyan.dev/ffmpeg/builds/
2. Ensure `ffmpeg.exe` is on `PATH`, then **open a new terminal**
3. Verify: `ffmpeg -version`
4. If FFmpeg is missing, `AudioAnalyzer` / `analyze_speech` raises an actionable `FFmpegNotFoundError` (does not invent transcripts)

### Tesseract OCR (optional, Windows)

1. Install the UB Mannheim Windows build: https://github.com/UB-Mannheim/tesseract/wiki
2. Ensure `tesseract.exe` is on `PATH`
3. Verify: `tesseract --version`
4. If missing, image analysis still returns visual (+ caption) evidence with an `OCR unavailable` warning

## Models

### Text

`cardiffnlp/twitter-roberta-base-sentiment-latest` — lazy-loaded; score = `P(positive) - P(negative)`

### Visual

`google/siglip2-base-patch16-224` — zero-shot concept scoring (not a trained sentiment classifier)

### Speech / ASR

Default: **`base.en`** via **faster-whisper** (CPU, `int8` compute type). Configurable in `src/config.py` / `AudioAnalyzer(whisper_model=...)`.

- Lazy model load
- CUDA not required
- Temporary extracted WAV files are cleaned up after each run
- Empty / no-speech media → warning, **no fabricated transcript**, no speech sentiment

## Speech branch API (Milestone 4)

```python
from src.pipeline import MyUniSentimentPipeline

pipeline = MyUniSentimentPipeline()
speech = pipeline.analyze_speech("path/to/clip.mp4")  # or .wav/.mp3/...
print(speech.transcript, speech.sentiment, speech.warnings)
```

Returned fields include: transcript, language, segments (timestamps), transcription duration, audio duration when available, speech sentiment, ASR model id, warnings.

## CLI

```powershell
# Single text
python main.py "I really enjoyed today's workshop."

# Batch (text + image processed; video still unsupported for full fusion)
python main.py --batch data/samples/activities.jsonl
```

## Tests

```powershell
# Fast unit tests (no large model downloads for ASR)
pytest -q -m "not integration"

# Includes text + SigLIP smoke (still skips optional ASR integration)
pytest -q

# Optional ASR integration (downloads Whisper; needs FFmpeg)
$env:MYUNI_RUN_ASR_INTEGRATION = "1"
pytest -q -m asr_integration
```

Speech unit tests cover: path validation, FFmpeg missing/failure handling, empty transcript behavior, standardized result shape, temp cleanup.

## Project layout

```text
src/
  analyzers/text.py
  analyzers/visual.py
  analyzers/ocr.py
  analyzers/image.py
  analyzers/audio.py      # FFmpeg + faster-whisper speech branch
  media/ffmpeg_utils.py
  batch.py
  config.py
  fusion.py
  pipeline.py
  schemas.py
```

## Current limitations

- Full video frame sampling + multimodal fusion not implemented (speech branch only)
- SigLIP prompts are heuristic zero-shot concepts
- OCR quality depends on Tesseract + image clarity
- ASR quality depends on audio clarity; English-only MVP (`base.en`)
- No SQLite / aggregation / Streamlit yet
