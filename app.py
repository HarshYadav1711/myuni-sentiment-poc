#!/usr/bin/env python3
"""MyUni Sentiment POC — Streamlit demonstration UI.

The backend pipeline remains the source of truth. This module only collects
inputs, calls ``MyUniSentimentPipeline``, and renders results.
"""

from __future__ import annotations

import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.media.ffmpeg_utils import FFmpegNotFoundError, find_ffmpeg, find_ffprobe
from src.pipeline import MyUniSentimentPipeline
from src.schemas import ActivityAnalysisResult, ActivityInput, SentimentEvidence
from src.ui.display import (
    format_confidence,
    format_score,
    fusion_summary,
    humanize_dependency_hint,
    humanize_ffmpeg_error,
    label_color,
    overall_headline,
)

_CSS = """
<style>
  .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1100px; }
  h1.app-title {
    font-family: "Source Serif 4", "Georgia", serif;
    font-weight: 600;
    font-size: 1.85rem;
    letter-spacing: -0.02em;
    color: #1a2332;
    margin-bottom: 0.15rem;
  }
  p.app-sub {
    color: #5c6570;
    font-size: 0.95rem;
    margin-top: 0;
    margin-bottom: 1.25rem;
  }
  .result-panel {
    border: 1px solid #d8dee6;
    background: linear-gradient(180deg, #f7f9fb 0%, #eef2f6 100%);
    border-radius: 6px;
    padding: 1rem 1.15rem;
    margin: 0.75rem 0 1rem 0;
  }
  .result-label {
    font-size: 1.45rem;
    font-weight: 650;
    letter-spacing: 0.02em;
    text-transform: uppercase;
  }
  .muted { color: #5c6570; font-size: 0.9rem; }
  .warn-box {
    border-left: 3px solid #b54708;
    background: #fff8f1;
    padding: 0.65rem 0.85rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
  }
  div[data-testid="stSidebar"] { background: #f4f6f8; }
</style>
"""


@st.cache_resource(show_spinner="Loading analysis models (first run only)…")
def get_pipeline() -> MyUniSentimentPipeline:
    """Cache the pipeline/analyzers across Streamlit reruns (not analysis results)."""
    return MyUniSentimentPipeline()


@st.cache_data(ttl=300, show_spinner=False)
def check_ffmpeg_available() -> tuple[bool, str]:
    try:
        find_ffmpeg()
        find_ffprobe()
        return True, "FFmpeg / ffprobe found on PATH."
    except FFmpegNotFoundError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


@st.cache_data(ttl=300, show_spinner=False)
def check_tesseract_available() -> tuple[bool, str]:
    try:
        import pytesseract
        from pytesseract import TesseractNotFoundError
    except ImportError:
        return False, "pytesseract is not installed."
    try:
        ver = pytesseract.get_tesseract_version()
        return True, f"Tesseract available (version {ver})."
    except TesseractNotFoundError:
        return False, (
            "Tesseract executable not found. Install Tesseract OCR and add it to PATH "
            "(Windows: https://github.com/UB-Mannheim/tesseract/wiki)."
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"Tesseract check failed: {exc}"


@st.cache_data(ttl=600, show_spinner=False)
def scene_sampling_available() -> bool:
    try:
        import scenedetect  # noqa: F401
        return True
    except ImportError:
        return False


def _new_ids(prefix: str) -> tuple[str, str]:
    suffix = uuid.uuid4().hex[:8].upper()
    return f"{prefix}-{suffix}", "DEMO-USER"


def _save_upload(upload, dest_dir: Path) -> Path:
    suffix = Path(upload.name).suffix or ".bin"
    path = dest_dir / f"upload_{uuid.uuid4().hex[:10]}{suffix}"
    path.write_bytes(upload.getvalue())
    return path


def _render_overall(result: ActivityAnalysisResult) -> None:
    headline = overall_headline(result)
    color = label_color(headline["label"])
    st.markdown(
        f"""
        <div class="result-panel">
          <div class="muted">Overall sentiment</div>
          <div class="result-label" style="color:{color}">{headline["label"]}</div>
          <div style="margin-top:0.45rem">
            Score <strong>{format_score(headline["score"])}</strong>
            &nbsp;·&nbsp;
            Confidence <strong>{format_confidence(headline["confidence"])}</strong>
          </div>
          <div class="muted" style="margin-top:0.35rem">
            Source: pipeline fusion ({headline.get("model") or "poc-fusion"})
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_evidence_block(title: str, evidence: Optional[SentimentEvidence]) -> None:
    if evidence is None:
        st.caption(f"{title}: not available")
        return
    st.markdown(f"**{title}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Label", evidence.label)
    c2.metric("Score", format_score(evidence.score))
    c3.metric("Confidence", format_confidence(evidence.confidence))
    if evidence.probabilities:
        st.write("Probabilities")
        st.json({k: round(float(v), 4) for k, v in evidence.probabilities.items()})


def _render_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    hint = humanize_dependency_hint(warnings)
    with st.expander(f"Warnings ({len(warnings)})", expanded=bool(hint)):
        for w in warnings:
            st.markdown(f'<div class="warn-box">{w}</div>', unsafe_allow_html=True)
        if hint:
            st.info(hint)


def _render_fusion(result: ActivityAnalysisResult) -> None:
    summary = fusion_summary(result.analysis.fusion)
    if not summary:
        return
    with st.expander("Fusion details", expanded=False):
        if summary["modality_conflict"]:
            st.warning(
                f"Modality conflict detected (disagreement={summary['disagreement_score']:.3f}). "
                "Overall confidence may be reduced.",
            )
        else:
            st.caption("No strong modality conflict flagged.")
        st.write(
            {
                "contributing_modalities": summary["contributing_modalities"],
                "disagreement_score": round(float(summary["disagreement_score"]), 4),
                "explanation": summary["explanation"],
                "note": summary["note"],
            },
        )


def _analyze_activity(activity: ActivityInput) -> ActivityAnalysisResult:
    pipeline = get_pipeline()
    return pipeline.analyze_activity(activity)


def tab_text() -> None:
    st.subheader("Text")
    text = st.text_area(
        "English text",
        height=140,
        placeholder="Paste a post, comment, or caption…",
        key="text_input",
    )
    if st.button("Analyze text", type="primary", key="btn_text"):
        if not text or not text.strip():
            st.error("Enter non-empty text to analyze.")
            return
        try:
            with st.spinner("Analyzing text…"):
                result = get_pipeline().analyze_text(text.strip(), user_id="DEMO-USER")
            _render_overall(result)
            text_ev = result.analysis.modalities.text
            if text_ev is not None:
                _render_evidence_block("Text evidence", text_ev)
            _render_fusion(result)
            _render_warnings(result.analysis.warnings)
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Analysis failed: {exc}")


def tab_image() -> None:
    st.subheader("Image")
    upload = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        key="image_upload",
    )
    caption = st.text_input("Caption (optional)", key="image_caption")
    if upload is not None:
        st.image(upload, caption=upload.name, use_container_width=True)

    if st.button("Analyze image", type="primary", key="btn_image"):
        if upload is None:
            st.error("Upload an image first.")
            return
        tmp_root = Path(tempfile.mkdtemp(prefix="myuni_st_img_"))
        try:
            media_path = _save_upload(upload, tmp_root)
            activity_id, user_id = _new_ids("IMG")
            activity = ActivityInput(
                activity_id=activity_id,
                user_id=user_id,
                activity_type="image",
                text=caption.strip() or None,
                media_path=str(media_path),
                created_at=datetime.now(timezone.utc),
            )
            with st.spinner("Analyzing image…"):
                result = _analyze_activity(activity)
            _render_overall(result)
            mods = result.analysis.modalities
            _render_evidence_block("Visual evidence", mods.visual)
            if result.analysis.ocr_text:
                st.markdown("**OCR text**")
                st.code(result.analysis.ocr_text, language=None)
            else:
                st.caption("OCR text: none extracted")
            _render_evidence_block("OCR sentiment", mods.ocr)
            if mods.text is not None:
                _render_evidence_block("Caption sentiment", mods.text)
            _render_fusion(result)
            _render_warnings(result.analysis.warnings)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Analysis failed: {exc}")
        finally:
            import shutil

            shutil.rmtree(tmp_root, ignore_errors=True)


def tab_video() -> None:
    st.subheader("Video")
    ffmpeg_ok, ffmpeg_msg = check_ffmpeg_available()
    if not ffmpeg_ok:
        st.error(humanize_ffmpeg_error(RuntimeError(ffmpeg_msg)))

    upload = st.file_uploader(
        "Upload a short video",
        type=["mp4", "mov", "mkv", "webm", "avi"],
        key="video_upload",
    )
    caption = st.text_input("Caption (optional)", key="video_caption")

    scene_ok = scene_sampling_available()
    strategy_labels = {
        "Fixed 1 FPS (baseline)": "fixed_fps",
        "Scene / keyframes": "scene_keyframe",
    }
    options = list(strategy_labels.keys())
    if not scene_ok:
        options = [options[0]]
        st.caption("Scene/keyframe sampling unavailable (install `scenedetect[opencv]`).")

    choice = st.radio(
        "Frame sampling",
        options=options,
        horizontal=True,
        key="video_strategy",
    )
    strategy = strategy_labels[choice]

    if upload is not None:
        st.video(upload)

    if st.button("Analyze video", type="primary", key="btn_video", disabled=not ffmpeg_ok):
        if upload is None:
            st.error("Upload a video first.")
            return
        tmp_root = Path(tempfile.mkdtemp(prefix="myuni_st_vid_"))
        try:
            media_path = _save_upload(upload, tmp_root)
            pipeline = get_pipeline()
            pipeline.video_analyzer.set_sampling_strategy(strategy)

            activity_id, user_id = _new_ids("VID")
            activity = ActivityInput(
                activity_id=activity_id,
                user_id=user_id,
                activity_type="video",
                text=caption.strip() or None,
                media_path=str(media_path),
                created_at=datetime.now(timezone.utc),
            )
            with st.spinner("Analyzing video (may take a minute on first run)…"):
                result = pipeline.analyze_activity(activity)

            _render_overall(result)
            mods = result.analysis.modalities
            video = result.analysis.video

            m1, m2, m3, m4 = st.columns(4)
            if video is not None:
                m1.metric("Frames analyzed", f"{video.frames_analyzed}/{video.frames_extracted}")
                m2.metric(
                    "Processing (s)",
                    f"{video.processing_seconds:.2f}" if video.processing_seconds is not None else "—",
                )
                m3.metric(
                    "Duration (s)",
                    f"{video.duration_seconds:.2f}" if video.duration_seconds is not None else "—",
                )
                m4.metric("Strategy", video.sampling_strategy or strategy)
            else:
                st.caption("Video diagnostics unavailable.")

            with st.expander("Video metadata", expanded=False):
                if video is None:
                    st.caption("No video diagnostics.")
                else:
                    st.write(
                        {
                            "sampling_strategy": video.sampling_strategy,
                            "sampling_fps": video.sampling_fps,
                            "extraction_seconds": video.extraction_seconds,
                            "processing_seconds": video.processing_seconds,
                            "duration_seconds": video.duration_seconds,
                            "frames_extracted": video.frames_extracted,
                            "frames_analyzed": video.frames_analyzed,
                            "scene_count": video.scene_count,
                            "has_audio": video.has_audio,
                            "frame_timestamps": video.frame_timestamps,
                        },
                    )

            if result.analysis.transcript:
                st.markdown("**Transcript**")
                st.write(result.analysis.transcript)
            else:
                st.caption("Transcript: none")

            _render_evidence_block("Speech sentiment", mods.speech)
            _render_evidence_block("Visual sentiment", mods.visual)
            if result.analysis.ocr_text:
                with st.expander("OCR evidence summary", expanded=False):
                    st.code(result.analysis.ocr_text, language=None)
                    _render_evidence_block("OCR sentiment", mods.ocr)
            else:
                st.caption("OCR evidence: none")
            if mods.text is not None:
                _render_evidence_block("Caption sentiment", mods.text)

            _render_fusion(result)
            _render_warnings(result.analysis.warnings)
        except FFmpegNotFoundError as exc:
            st.error(humanize_ffmpeg_error(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Analysis failed: {humanize_ffmpeg_error(exc)}")
        finally:
            import shutil

            shutil.rmtree(tmp_root, ignore_errors=True)


def main() -> None:
    st.set_page_config(
        page_title="MyUni Sentiment POC",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown('<h1 class="app-title">MyUni Sentiment POC</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-sub">Multimodal sentiment demonstration — pipeline-backed, English-only MVP.</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Environment")
        ffmpeg_ok, ffmpeg_msg = check_ffmpeg_available()
        tess_ok, tess_msg = check_tesseract_available()
        st.write("FFmpeg:", "OK" if ffmpeg_ok else "Missing")
        if not ffmpeg_ok:
            st.caption(ffmpeg_msg)
        st.write("Tesseract:", "OK" if tess_ok else "Missing (OCR degraded)")
        if not tess_ok:
            st.caption(tess_msg)
        st.write("Scene sampling:", "OK" if scene_sampling_available() else "Not installed")

        st.markdown("### Privacy")
        st.caption(
            "Uploaded media is written to a temporary folder for analysis and deleted "
            "after the run. Models are cached in memory for this Streamlit session; "
            "analysis results are not cached across inputs.",
        )
        st.markdown("### Note")
        st.caption(
            "Scores and fusion weights are POC evaluation defaults — "
            "not client business scoring rules.",
        )

    text_tab, image_tab, video_tab = st.tabs(["Text", "Image", "Video"])
    with text_tab:
        tab_text()
    with image_tab:
        tab_image()
    with video_tab:
        tab_video()


if __name__ == "__main__":
    main()
