# MyUni Sentiment POC — Architecture

English-only proof of concept for multimodal activity sentiment (`positive` / `neutral` / `negative`). This document describes how the current codebase is structured. It is not a production design.

## What the POC proves

- A **single pipeline** can route text, image, and video activities through modality-specific analyzers and fuse evidence deterministically.
- **Late fusion** can expose per-modality scores, conflict flags, and warnings without an LLM explanation layer.
- **Batch ingestion + SQLite** can persist results and compute simple daily user aggregates for experimentation.
- **Video sampling strategies** (fixed FPS vs scene/keyframe) can be compared on the same asset without changing downstream analysis.
- A **Streamlit demo** can sit on top of the same pipeline without duplicating inference.

Fusion weights, thresholds, and daily aggregates are **POC evaluation defaults only** — not the final client business scoring methodology.

## High-level flow

```text
ActivityInput (CLI / batch / Streamlit)
        │
        ▼
MyUniSentimentPipeline
        │
        ├── text  → TextSentimentAnalyzer (RoBERTa)
        ├── image → ImageAnalyzer → Visual + OCR (+ optional caption text)
        └── video → VideoAnalyzer → FrameSampler → Visual/OCR subset + AudioAnalyzer (ASR) + caption
        │
        ▼
fuse_modalities()  ← config/fusion.yaml
        │
        ▼
ActivityAnalysisResult (+ PocRuntimeInfo: models, sampling, fusion source)
```

## Core modules

| Area | Location | Role |
| --- | --- | --- |
| Pipeline router | `src/pipeline.py` | Validates inputs, calls analyzers, attaches runtime metadata |
| Text | `src/analyzers/text.py` | Lazy RoBERTa load; score = P(pos) − P(neg) |
| Visual | `src/analyzers/visual.py` | Lazy SigLIP 2 zero-shot concept scoring |
| OCR | `src/analyzers/ocr.py` | Tesseract via pytesseract; graceful degradation |
| Image | `src/analyzers/image.py` | Visual + OCR + optional caption scoring |
| Audio / ASR | `src/analyzers/audio.py` | FFmpeg extract → faster-whisper → text sentiment |
| Video | `src/analyzers/video.py` | Probe → sampler → per-frame visual/OCR → speech → fusion inputs |
| Frame sampling | `src/media/samplers.py` | `FixedFPSSampler`, `SceneKeyframeSampler` |
| Media I/O | `src/media/ffmpeg_utils.py` | ffprobe, fixed-FPS extract, timestamp stills |
| Fusion | `src/fusion.py` + `config/fusion.yaml` | Confidence-weighted late fusion + conflict detection |
| Schemas | `src/schemas.py` | Pydantic contracts for input/output/batch |
| Config | `src/config.py` | Model IDs, video sampling defaults, fusion loader |
| Batch (memory) | `src/batch.py` | JSONL ingest without DB |
| Storage | `src/storage/` | SQLite schema, repository, daily aggregates, persistent batch |
| Evaluation | `evaluation/` | TweetEval / MVSA / MOSI adapters (bring your own data) |
| Demo UI | `app.py` + `src/ui/` | Streamlit presentation only |
| Health | `src/env_check.py` | Dependency and configuration report |

## Lazy model loading

Analyzers are constructed at pipeline startup but **weights load on first use**:

- Text: first `analyze()` call
- Visual: first image/frame scored
- ASR: first audio/video speech branch

The Streamlit app caches one `MyUniSentimentPipeline` per session (`st.cache_resource`) so models are not reloaded on every widget rerun. Analysis results are **not** cached.

## Video sampling

| Strategy | Implementation | Notes |
| --- | --- | --- |
| `fixed_fps` (baseline) | FFmpeg `fps=` filter | Default ~1 FPS; auto-reduced to respect `max_frames` |
| `scene_keyframe` | PySceneDetect + FFmpeg stills | Mid-scene representatives; caps + optional fallback to fixed FPS |

Diagnostics land in `analysis.video` (`sampling_strategy`, `extraction_seconds`, `scene_count`, etc.).

## Persistence semantics

SQLite tables: `batch_runs`, `activities`, `analysis_results`, `daily_user_scores`.

- **Processed** activities are skipped on re-ingest (idempotent success).
- **Failed** activities can be **retried** on a subsequent batch run (analysis row upserted).
- Invalid JSONL records are counted separately and do not stop the batch.

Daily scores are means/counts over stored activity results — documented as POC aggregates, not client business scores.

## External dependencies

| Tool | Required for | Failure mode |
| --- | --- | --- |
| FFmpeg + ffprobe | Video frame extract, audio extract | Actionable error via `FFmpegNotFoundError` |
| Tesseract | OCR on images / video frames | Warning + visual-only path continues |
| PySceneDetect | Scene/keyframe sampling | Fallback to fixed FPS (default) or clear error |

Check locally:

```powershell
python main.py --health
```

## Configuration visibility

Each analysis result may include `analysis.runtime`:

- `models` — configured HF / ASR identifiers
- `video_sampling` — strategy, fps, caps (video activities)
- `fusion_source` — path to loaded `fusion.yaml`
- `note` — POC disclaimer

Per-modality `SentimentEvidence.model` fields remain the primary modality-level identifiers.

## Testing layers

1. **Unit** — `pytest -q -m "not integration"` (no model downloads)
2. **Integration** — full pytest or opt-in ASR/video markers (downloads weights)
3. **Smoke** — `python scripts/smoke_poc.py` (health + eval stub; optional model path via env var)
4. **Benchmark** — `python -m evaluation.run ...` against local indexes

## Known non-goals (current POC)

- Native video VLM
- Authentication / multi-tenant deployment
- PostgreSQL / cloud infra
- Hindi/Hinglish or non-English ASR beyond `base.en`
- Scientifically validated fusion weights or client business scoring rules

## Suggested next technical experiments

- Compare fixed FPS vs scene/keyframe on real campus clips (`scripts/compare_video_sampling.py`)
- Tune fusion weights/thresholds against TweetEval / MVSA / MOSI fixtures
- Measure latency and RAM on target Windows laptops (16 GB baseline)
- Evaluate OCR quality with and without Tesseract on meme/text-overlay content
- Prototype caption-weight sensitivity in late fusion (config-only, no architecture change)
