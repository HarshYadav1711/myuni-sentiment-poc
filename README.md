# MyUni Multimodal Sentiment Analysis POC

English-first proof of concept for analyzing social activity sentiment on MyUni.

## Current status (Milestone 5)

**Working today:**
- **Text** sentiment (RoBERTa)
- **Image** (SigLIP 2 visual + OCR + caption + POC fusion)
- **Speech/audio** branch (FFmpeg + faster-whisper)
- **Video** end-to-end: ~1 FPS frame sampling → visual aggregate + practical OCR + ASR/speech + optional caption → explainable late fusion
- JSONL **batch** ingestion for text / image / video

**Not implemented yet:** scene detection, native video VLMs, SQLite storage, daily aggregation, Streamlit demo.

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
src/{pipeline,batch,fusion,config,schemas}.py
```

## Current limitations

- No scene/keyframe detection yet (fixed FPS only)
- No large native video VLM
- OCR/ASR quality depend on media clarity and installed binaries
- English only
- No SQLite / Streamlit yet
