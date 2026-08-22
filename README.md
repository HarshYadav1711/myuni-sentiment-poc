# MyUni Multimodal Sentiment Analysis POC

English-first proof of concept for analyzing social activity sentiment on MyUni.

## What this POC proves

- End-to-end **text, image, and video** sentiment with a shared pipeline contract
- **Explainable late fusion** (per-modality evidence + conflict flags, no LLM layer)
- **JSONL batch** ingestion with per-record error isolation
- **SQLite persistence** and POC daily user aggregates (not client business scores)
- **Pluggable video sampling** (fixed FPS baseline + experimental scene/keyframe)
- **Streamlit demo** as a thin client over the same backend
- **Evaluation adapters** for TweetEval / MVSA / CMU-MOSI (bring your own data)

**Important:** fusion weights, label thresholds, and daily aggregates are **POC evaluation defaults only** — not the final client business scoring methodology.

Architecture details: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

## Requirements

- Python 3.10+ (3.13 tested)
- ~16 GB RAM recommended; CPU-first (~16 GB); GPU optional (CUDA used when available)
- **FFmpeg + ffprobe** on `PATH` (video + speech)
- **Tesseract OCR** on `PATH` (optional but recommended for embedded text)
- **`transformers>=4.45,<5`** — required for SigLIP 2 visual loading (5.x breaks AutoProcessor)
- Windows-friendly; tested paths use PowerShell examples

## Installation (Windows)

```powershell
cd D:\Work\myuni-sentiment-poc
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# External tools (reopen terminal after install)
winget install Gyan.FFmpeg
# Tesseract: https://github.com/UB-Mannheim/tesseract/wiki

python main.py --health
python scripts/generate_sample_image.py
python scripts/generate_sample_video.py
```

## Supported inputs

| Modality | Input | Notes |
| --- | --- | --- |
| Text | Non-blank English string | RoBERTa sentiment |
| Image | Local image path + optional caption | SigLIP visual + OCR + caption fusion |
| Video | Local video path + optional caption | Fixed FPS or scene/keyframe sampling; speech + visual + OCR fusion |

Activity contract (batch): `activity_id`, `user_id`, `activity_type`, `created_at`, plus modality-specific fields. See `data/samples/activities.jsonl`.

## Example commands

```powershell
# Environment / dependency report
python main.py --health
python main.py --health --db data/myuni_poc.db

# Text
python main.py "Great lecture today, really enjoyed it."

# Video
python main.py --video data/samples/synthetic_sample.mp4 --caption "Quiet campus clip"
python main.py --video data/samples/synthetic_sample.mp4 --sampling-strategy scene_keyframe

# Batch (analysis only)
python main.py --batch data/samples/activities.jsonl

# Batch + SQLite persistence
python main.py --batch data/samples/activities.jsonl --db data/myuni_poc.db

# Daily POC aggregates (NOT client business scores)
python main.py --daily-scores --db data/myuni_poc.db

# Experimental sampling comparison (same video, both strategies)
python scripts/compare_video_sampling.py data/samples/synthetic_sample.mp4 --caption "Quiet campus clip"
```

## Streamlit demo

Thin UI over `MyUniSentimentPipeline` — **no duplicated inference**.

```powershell
streamlit run app.py
```

- Tabs: **Text** · **Image** · **Video**
- Models load once per session; uploads are deleted after each analysis
- Sidebar shows FFmpeg / Tesseract / scene-sampling availability

## Models & configuration visibility

Configured identifiers (also in `python main.py --health`):

| Modality | Model / tool |
| --- | --- |
| Text / caption / OCR / transcript | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Visual | `google/siglip2-base-patch16-224` |
| ASR | faster-whisper `base.en` (CPU `int8`) |
| Fusion | `poc-fusion` via `config/fusion.yaml` |

Analysis JSON includes `analysis.runtime` with model IDs, video sampling settings, and fusion source path.

## Video sampling

| Strategy | Behavior |
| --- | --- |
| `fixed_fps` (baseline) | FFmpeg ~1 FPS; auto-reduced for `max_frames=60` |
| `scene_keyframe` | PySceneDetect cuts → representative stills via FFmpeg; caps + fallback |

## Datasets & evaluation

See **[docs/DATASETS.md](docs/DATASETS.md)**. This repo does not redistribute TweetEval / MVSA / MOSI files.

```powershell
python -m evaluation.run text --data evaluation/fixtures/text_samples.jsonl --stub --limit 5 --out outputs/eval_text_stub
```

## Testing

```powershell
# Fast unit tests (no model downloads)
pytest -q -m "not integration"

# Full suite (may download HF / ASR weights)
pytest -q

# POC smoke (health + eval stub; no model download by default)
python scripts/smoke_poc.py

# Optional: run representative text/image/video with real models
$env:MYUNI_RUN_SMOKE_MODELS=1; python scripts/smoke_poc.py

# Optional heavy integration markers
$env:MYUNI_RUN_ASR_INTEGRATION=1; pytest -q -m asr_integration
$env:MYUNI_RUN_VIDEO_INTEGRATION=1; pytest -q -m video_integration
```

## Project layout

```text
app.py                          Streamlit demo
main.py                         CLI entrypoint
config/fusion.yaml              POC fusion weights (not client scoring)
docs/{ARCHITECTURE,DATASETS}.md
evaluation/                     Benchmark adapters
scripts/{smoke_poc,compare_video_sampling,...}.py
src/{pipeline,config,schemas,fusion,batch,runtime_info,env_check}.py
src/analyzers/{text,visual,ocr,image,audio,video}.py
src/media/{ffmpeg_utils,samplers}.py
src/storage/                    SQLite persistence
src/ui/                         Demo presentation helpers
tests/
```

## Current POC limitations

- English only; no Hindi/Hinglish
- No native video VLM; fixed/scene frame sampling only
- Streamlit demo is local-only (no auth / deployment)
- First model run downloads weights (~GB) and is slow on CPU
- OCR/ASR quality depends on media clarity and installed binaries
- Daily SQLite aggregates are experimental means/counts — not client business scores
- Scene/keyframe sampling is experimental; fixed FPS remains the default baseline

## Next technical experiments

- Compare sampling strategies on real campus clips (`scripts/compare_video_sampling.py`)
- Tune `config/fusion.yaml` against local evaluation indexes
- Profile RAM/latency on target demo hardware (16 GB Windows laptops)
- Assess OCR impact on meme / text-overlay content with Tesseract installed vs missing
- Expand evaluation coverage (MOSI video, MVSA image) once local indexes are prepared

## Not implemented yet

Native video VLMs, PostgreSQL, authentication, cloud deployment, full CMU-MOSEI adapter.
