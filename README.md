# MyUni Multimodal Sentiment Analysis POC

English-first proof of concept for analyzing social activity sentiment on MyUni.

## Current status (Milestone 7)

**Working today:**
- Text / image / video multimodal analysis + explainable late fusion
- JSONL batch ingestion
- **SQLite persistence** (`batch_runs`, `activities`, `analysis_results`, `daily_user_scores`)
- **POC daily user aggregates** (not the client business score)

**Not implemented yet:** scene detection, native video VLMs, PostgreSQL, Streamlit demo.

## Requirements

- Python 3.10+
- ~16 GB RAM; CPU supported; GPU optional
- **FFmpeg + ffprobe** on `PATH` (required for video/speech)
- Optional: Tesseract OCR for embedded text

## Setup (Windows)

```powershell
cd D:\Work\myuni-sentiment-poc
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
winget install Gyan.FFmpeg   # then reopen terminal; ffmpeg -version / ffprobe -version
python scripts/generate_sample_image.py
python scripts/generate_sample_video.py
```

## Analyze a video

```powershell
python main.py --video data/samples/synthetic_sample.mp4 --caption "Quiet campus clip" --user-id U005 --activity-id ACT005

# Include compact per-frame debug rows:
python main.py --video data/samples/synthetic_sample.mp4 --video-debug
```

Or via batch JSONL (`activity_type: "video"`).

## Video strategy (MVP v1)

- Fixed sampling at **~1 FPS** (configurable; auto-reduced to respect `max_frames=60`)
- Per-frame visual scoring via existing SigLIP 2 zero-shot path
- OCR on an evenly spaced subset of frames (`max_ocr_frames=8`)
- Audio via existing `AudioAnalyzer` (graceful if no speech / no audio stream)
- Visual summary = confidence-weighted average of frame scores (**not** a temporal neural model)
- Overall = confidence-weighted late fusion of caption / visual / OCR / speech
- Temp frames cleaned unless `preserve_temp=True` on `VideoAnalyzer`
- Default JSON stays compact; `--video-debug` adds `analysis.video.frame_debug`

Serious decode/probe failures still error; OCR/speech/single-frame failures become warnings with partial evidence.

## SQLite persistence & daily aggregates (Milestone 7)

Local-only SQLite (stdlib `sqlite3`). No PostgreSQL in this POC.

```powershell
# Process JSONL into SQLite and print a concise batch summary
python main.py --batch data/samples/activities.jsonl --db data/myuni_poc.db

# Query POC daily user scores (NOT client business scores)
python main.py --daily-scores --db data/myuni_poc.db
python main.py --daily-scores --db data/myuni_poc.db --date 2026-08-21
python main.py --daily-scores --db data/myuni_poc.db --user-id U001
```

Tables: `batch_runs`, `activities`, `analysis_results`, `daily_user_scores`.

- `activity_id` is unique — reruns **skip** duplicates (no silent overwrite)
- Failed records are stored with `status=failed` and do not erase successes
- Daily fields: `activity_count`, `valid_analysis_count`, `mean_sentiment_score`, pos/neu/neg counts, `daily_sentiment_label`
- Documented as **POC daily aggregate**, not future client scoring

Without `--db`, `--batch` still runs analysis-only (in-memory summary JSON).

## Multimodal fusion (Milestone 6)

Transparent **late fusion** (no LLM explanations). Editable POC defaults:

`config/fusion.yaml`

```text
effective_weight_i = modality_weight_i * confidence_i
fused_score = sum(score_i * effective_weight_i) / sum(effective_weight_i)
```

- Label thresholds and modality weights are **POC evaluation defaults only** — not client scoring rules and not scientifically validated.
- Strong opposing modalities set `analysis.fusion.modality_conflict` and reduce overall confidence.
- Per-modality evidence remains intact under `analysis.modalities.*`.

## Models

| Modality | Model / tool |
| --- | --- |
| Text / caption / OCR / transcript | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Visual | `google/siglip2-base-patch16-224` (zero-shot concepts) |
| ASR | faster-whisper `base.en` (CPU `int8`) |
| Media | FFmpeg / ffprobe |

## Tests

```powershell
# Fast unit tests
pytest -q -m "not integration"

# Default suite (skips optional ASR/video integration)
pytest -q

# Optional video smoke (downloads models; needs FFmpeg)
$env:MYUNI_RUN_VIDEO_INTEGRATION = "1"
pytest -q -m video_integration
```

## Project layout

```text
src/analyzers/{text,visual,ocr,image,audio,video}.py
src/media/ffmpeg_utils.py
src/storage/{schema,repository,aggregation,service}.py
src/{pipeline,batch,fusion,config,schemas}.py
config/fusion.yaml
```

## Current limitations

- No scene/keyframe detection yet (fixed FPS only)
- No large native video VLM
- Daily aggregates are POC means/counts — not client business scores
- OCR/ASR quality depend on media clarity and installed binaries
- English only
- No Streamlit UI yet
