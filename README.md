# MyUni Multimodal Sentiment Analysis POC

English-first proof of concept for analyzing social activity sentiment on MyUni.

## Current status (Milestone 9)

**Working today:**
- Text / image / video multimodal analysis + explainable late fusion
- JSONL batch + SQLite persistence + POC daily aggregates
- Reproducible evaluation framework (TweetEval / MVSA / CMU-MOSI adapters)
- **Pluggable video sampling:** fixed FPS (baseline) + scene/keyframe (PySceneDetect)

**Not implemented yet:** native video VLMs, PostgreSQL, Streamlit demo, full CMU-MOSEI adapter (documented as future).

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

# Alternative sampling strategy (experimental; baseline remains fixed_fps):
python main.py --video data/samples/synthetic_sample.mp4 --sampling-strategy scene_keyframe
```

Or via batch JSONL (`activity_type: "video"`). Batch still uses the pipeline default (`fixed_fps`) unless you construct `MyUniSentimentPipeline(video_sampling_strategy=...)`.

## Video sampling strategies (Milestone 9)

Two strategies share the same `FrameSample` / `SampledFrames` structure consumed by `VideoAnalyzer`:

| Strategy | Class | Behavior |
| --- | --- | --- |
| `fixed_fps` (baseline) | `FixedFPSSampler` | FFmpeg `fps=` filter; default ~1 FPS; auto-reduced for `max_frames` |
| `scene_keyframe` | `SceneKeyframeSampler` | PySceneDetect `ContentDetector` → mid-scene (or N) keyframes → FFmpeg stills |

- FFmpeg is **not** removed; scene mode still extracts frames with FFmpeg.
- Scene mode caps frames (`max_frames`), supports `frames_per_scene`, and by default **falls back** to fixed FPS with a clear warning if detection fails (disable via `SceneSamplingConfig.fallback_to_fixed_fps=False`).
- Diagnostics include `sampling_strategy`, `extraction_seconds`, optional `scene_count`.

### Experimental comparison (same video, both strategies)

```powershell
python scripts/compare_video_sampling.py data/samples/synthetic_sample.mp4 --caption "Quiet campus clip" --out outputs/sampling_compare.json
```

The report lists frame counts, extraction / total analysis times, visual & overall sentiment, modality evidence, and whether the final prediction differs. **It does not declare either strategy better.**

## Video analysis pipeline

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

## Benchmark evaluation (Milestone 8)

Dataset acquisition, licenses, and JSONL formats: **[docs/DATASETS.md](docs/DATASETS.md)**.

This repo does **not** redistribute TweetEval / MVSA / MOSI files.

```powershell
# Stub predictor on bundled synthetic fixtures (no model download)
python -m evaluation.run text --data evaluation/fixtures/text_samples.jsonl --limit 5 --stub --out outputs/eval_text_stub

# After you prepare local indexes (see docs/DATASETS.md):
python -m evaluation.run text --data data/eval/tweeteval_index.jsonl --limit 100 --out outputs/eval_tweeteval
python -m evaluation.run image --data data/eval/mvsa_index.jsonl --limit 20 --out outputs/eval_mvsa
python -m evaluation.run video --data data/eval/mosi_index.jsonl --limit 10 --out outputs/eval_mosi

# Optional TweetEval via Hugging Face datasets (network; not used by unit tests)
pip install datasets
python -m evaluation.run text --tweeteval-hf --split test --limit 50 --stub
```

Outputs: `metrics.json` + `predictions.csv` + console summary (accuracy, P/R, macro/weighted F1, confusion matrix; MOSI also MAE/Pearson with explicit 3-way mapping documented).

## Tests (keep these separate)

```powershell
# 1) Unit tests — fast, no model downloads (includes evaluation fixtures)
pytest -q -m "not integration"

# 2) Integration / model tests — downloads HF/ASR weights as needed
pytest -q
# Optional heavy smoke:
#   $env:MYUNI_RUN_ASR_INTEGRATION=1; pytest -q -m asr_integration
#   $env:MYUNI_RUN_VIDEO_INTEGRATION=1; pytest -q -m video_integration

# 3) Benchmark evaluation — run via evaluation CLI against local/HF data (not pytest)
python -m evaluation.run text --data evaluation/fixtures/text_samples.jsonl --stub --limit 5
```

## Project layout

```text
evaluation/{metrics,common,run}.py
evaluation/{text,image,video}/
docs/DATASETS.md
src/analyzers/{text,visual,ocr,image,audio,video}.py
src/media/{ffmpeg_utils,samplers}.py
src/storage/{schema,repository,aggregation,service}.py
src/{pipeline,batch,fusion,config,schemas}.py
scripts/compare_video_sampling.py
config/fusion.yaml
```

## Current limitations

- Scene sampling is experimental; fixed FPS remains the default baseline
- No large native video VLM
- Daily aggregates are POC means/counts — not client business scores
- OCR/ASR quality depend on media clarity and installed binaries
- English only
- No Streamlit UI yet
