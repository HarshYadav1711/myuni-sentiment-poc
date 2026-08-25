#!/usr/bin/env python3
"""MyUni Sentiment Intelligence — client demo UI (previous visual shell).

Keeps the established two-column comparison look. Behavior is unified:
automatic modality detection, Twitter-RoBERTa for text, image/video preview
with not-implemented informational states (no fake scores).
"""

from __future__ import annotations

import logging
import sys
import tempfile
import uuid
from pathlib import Path
from textwrap import dedent
from typing import Any, Optional

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import MyUniSentimentPipeline
from src.routing.input_router import CapabilityStatus, InputRouter
from src.ui.display import (
    format_confidence_pct,
    format_probability_pct,
    format_score,
    label_color,
)

logger = logging.getLogger(__name__)

MAX_CHARS = 5000
MEDIA_TYPES = [
    "jpg", "jpeg", "png", "webp",
    "wav", "mp3", "m4a", "ogg", "flac",
    "mp4", "mov", "webm", "avi", "mkv",
]


def css(markup: str) -> None:
    st.markdown(dedent(markup).strip(), unsafe_allow_html=True)


def html(markup: str) -> None:
    st.html(dedent(markup).strip())


@st.cache_resource(show_spinner="Loading analysis models (first run only)…")
def get_pipeline() -> MyUniSentimentPipeline:
    return MyUniSentimentPipeline()


def _save_upload(upload, dest_dir: Path) -> Path:
    suffix = Path(upload.name).suffix or ".bin"
    path = dest_dir / f"upload_{uuid.uuid4().hex[:10]}{suffix}"
    path.write_bytes(upload.getvalue())
    return path


def _cleanup_temp_dir(path: Path) -> None:
    import shutil

    try:
        shutil.rmtree(path, ignore_errors=False)
    except OSError as exc:
        logger.warning("Failed to remove temporary upload dir %s: %s", path, exc)


def _pill_style(label: str) -> str:
    color = label_color(label)
    return f"background:{color}22;color:{color};border:1px solid {color}55;"


def _render_idle_preview() -> None:
    html(
        """
        <div class="section-card">
            <div class="section-title">Analysis Preview</div>
            <div class="section-description">See how MyUni understands the content.</div>
            <div class="preview-item">
                <div class="preview-icon preview-green">😊</div>
                <div>
                    <div class="preview-name">Overall Sentiment</div>
                    <div class="preview-desc">Positive, Neutral, or Negative — live for text, image, audio & video</div>
                </div>
            </div>
            <div class="preview-item">
                <div class="preview-icon preview-purple">🎭</div>
                <div>
                    <div class="preview-name">Emotion Detection</div>
                    <div class="preview-desc">Joy, sadness, anger, fear, etc. — roadmap</div>
                </div>
            </div>
            <div class="preview-item">
                <div class="preview-icon preview-blue">💙</div>
                <div>
                    <div class="preview-name">Well-being Indicators</div>
                    <div class="preview-desc">Mental well-being signals — roadmap</div>
                </div>
            </div>
            <div class="preview-item">
                <div class="preview-icon preview-red">⚠️</div>
                <div>
                    <div class="preview-name">Risk Assessment</div>
                    <div class="preview-desc">Identify potential concerns — roadmap</div>
                </div>
            </div>
            <div class="preview-item">
                <div class="preview-icon preview-yellow">💡</div>
                <div>
                    <div class="preview-name">Contextual Insights</div>
                    <div class="preview-desc">Detailed analysis and insights — roadmap</div>
                </div>
            </div>
            <div class="privacy">
                <div class="privacy-title">🛡️ Privacy First</div>
                <div class="privacy-text">
                    Analysis is confidential and follows strict privacy guidelines.
                </div>
            </div>
        </div>
        """
    )
    st.caption("Live in this build: text (RoBERTa), image (SigLIP 2), audio (Whisper→RoBERTa), video (hybrid).")


def _show_media_preview(preview: Optional[dict[str, Any]]) -> None:
    if not preview:
        return
    try:
        kind = preview.get("kind")
        if kind == "image":
            data = preview.get("bytes")
            if data is None and preview.get("path"):
                data = Path(preview["path"]).read_bytes()
            if data:
                st.image(data, use_container_width=True)
            st.caption(f'{preview.get("name", "")} · {preview.get("mime") or "image"}')
        elif kind == "audio":
            if preview.get("bytes") is not None:
                st.audio(preview["bytes"])
            elif preview.get("path"):
                st.audio(str(preview["path"]))
            st.caption(f'{preview.get("name", "")} · {preview.get("mime") or "audio"}')
        elif kind == "video":
            if preview.get("bytes") is not None:
                st.video(preview["bytes"])
            elif preview.get("path"):
                st.video(str(preview["path"]))
            st.caption(f'{preview.get("name", "")} · {preview.get("mime") or "video"}')
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not preview media: {exc}")


def _dist_html(probs: dict[str, Any]) -> str:
    return f"""
    <div class="dist-grid">
        <div class="dist-card">
            <div class="dist-label">Positive</div>
            <div class="dist-value">{format_probability_pct(float(probs.get("positive", 0.0)))}</div>
        </div>
        <div class="dist-card">
            <div class="dist-label">Neutral</div>
            <div class="dist-value">{format_probability_pct(float(probs.get("neutral", 0.0)))}</div>
        </div>
        <div class="dist-card">
            <div class="dist-label">Negative</div>
            <div class="dist-value">{format_probability_pct(float(probs.get("negative", 0.0)))}</div>
        </div>
    </div>
    """


def _render_ok_result(routed: Any) -> None:
    analysis = routed.analysis
    block = analysis.analysis
    modalities = block.modalities
    detected = (routed.detected_input.value.upper() if routed.detected_input else "UNKNOWN")
    model_name = routed.model_display_name or "—"

    if detected == "IMAGE" and modalities.visual is not None:
        evidence = modalities.visual
        title = "Visual Sentiment"
    elif detected == "AUDIO" and modalities.speech is not None:
        evidence = modalities.speech
        title = "Transcript Sentiment"
    else:
        evidence = modalities.text or modalities.visual or modalities.speech or block.overall
        title = "Overall Sentiment"

    probs = evidence.probabilities or {}
    pill = _pill_style(evidence.label)

    html(
        f"""
        <div class="section-card">
            <div class="section-title">Analysis Results</div>
            <div class="section-description">Detected Input: {detected}</div>
            <div class="result-inner">
                <div class="metric-label">{title}</div>
                <span class="sentiment-pill" style="{pill}">● {evidence.label.upper()}</span>
                <div style="margin-top:12px;font-size:14px;color:#334155;">
                    Confidence <strong>{format_confidence_pct(evidence.confidence)}</strong>
                    &nbsp;·&nbsp;
                    Score <strong>{format_score(evidence.score)}</strong>
                </div>
                <div class="metric-label" style="margin-top:8px;">
                    Model · {model_name}
                </div>
            </div>
            <div class="section-title" style="margin-top:14px;font-size:15px;">
                Probability distribution
            </div>
            {_dist_html(probs)}
        </div>
        """
    )

    if detected == "AUDIO" and block.transcript:
        st.markdown("**Transcript**")
        st.write(block.transcript)

    if detected == "VIDEO":
        if modalities.visual is not None:
            vprobs = modalities.visual.probabilities or {}
            frames = block.video.frames_analyzed if block.video else 0
            st.markdown("**Visual evidence**")
            st.caption(
                f"Frames analyzed: {frames} · Visual sentiment: {modalities.visual.label.upper()} "
                f"({format_confidence_pct(modalities.visual.confidence)})",
            )
            html(_dist_html(vprobs))
        if block.transcript or modalities.speech is not None:
            st.markdown("**Speech evidence**")
            if block.transcript:
                st.write(block.transcript)
            if modalities.speech is not None:
                st.caption(
                    f"Transcript sentiment: {modalities.speech.label.upper()} "
                    f"({format_confidence_pct(modalities.speech.confidence)})",
                )
                html(_dist_html(modalities.speech.probabilities or {}))
        if block.fusion is not None:
            used = ", ".join(block.fusion.contributing_modalities) or "none"
            st.caption(f"Fusion modalities: {used}")
            st.caption(block.fusion.explanation)

    html(
        """
        <div class="privacy">
            <div class="privacy-title">🛡️ Privacy First</div>
            <div class="privacy-text">
                Analysis is confidential and follows strict privacy guidelines.
            </div>
        </div>
        """
    )

    for warning in list(block.warnings or []):
        st.caption(warning)

    with st.expander("Technical details", expanded=False):
        st.code(routed.model_id or "")
        if detected == "TEXT":
            st.caption(
                "Twitter-RoBERTa. Label = highest probability. "
                "POC score = P(positive) − P(negative). Evaluation default only.",
            )
        elif detected == "IMAGE":
            st.caption(
                "SigLIP 2 zero-shot visual sentiment (not a dedicated sentiment classifier "
                "and not mental-health detection).",
            )
            st.code("google/siglip2-base-patch16-224")
        elif detected == "AUDIO":
            st.caption("faster-whisper base.en (CPU int8) → Twitter-RoBERTa on transcript.")
        elif detected == "VIDEO":
            st.caption(
                "Hybrid video: ~1 FPS frame sampling (max 12) with SigLIP 2 + "
                "faster-whisper speech → RoBERTa. Late fusion is a POC baseline only.",
            )


def _render_status(payload: dict[str, Any]) -> None:
    status = payload.get("status")
    if status == CapabilityStatus.OK and payload.get("routed") is not None:
        routed = payload["routed"]
        detected = routed.detected_input
        if detected and detected.value in {"image", "audio", "video"}:
            _show_media_preview(payload.get("preview"))
        _render_ok_result(routed)
        return

    if status in {
        CapabilityStatus.NOT_IMPLEMENTED,
        CapabilityStatus.INSUFFICIENT_EVIDENCE,
    }:
        detected = payload.get("detected_input")
        label = detected.value.upper() if detected else "MEDIA"
        message = payload.get("message") or ""
        html(
            f"""
            <div class="section-card">
                <div class="section-title">Analysis Results</div>
                <div class="section-description">Detected Input: {label}</div>
            </div>
            """
        )
        _show_media_preview(payload.get("preview"))
        if payload.get("routed") is not None and getattr(payload["routed"], "analysis", None):
            transcript = payload["routed"].analysis.analysis.transcript
            if transcript:
                st.markdown("**Transcript**")
                st.write(transcript)
        html(f'<div class="info-panel">{message}</div>')
        html(
            """
            <div class="privacy">
                <div class="privacy-title">🛡️ Privacy First</div>
                <div class="privacy-text">
                    Uploaded media is handled for this demo session only.
                </div>
            </div>
            """
        )
        return

    st.warning(payload.get("message") or "Unable to complete analysis.")


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="MyUni | Sentiment Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "draft_text" not in st.session_state:
    st.session_state.draft_text = ""
if "pending_preview" not in st.session_state:
    st.session_state.pending_preview = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

css(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 5% 20%, rgba(99, 102, 241, 0.07), transparent 25%),
            radial-gradient(circle at 95% 15%, rgba(168, 85, 247, 0.07), transparent 25%),
            #f8faff;
        color: #172554;
    }
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { background: transparent !important; }
    div[data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }

    .brand {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        margin-bottom: 5px;
    }
    .brand-icon {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: linear-gradient(135deg, #2563eb, #6366f1);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 25px;
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.22);
    }
    .brand-name {
        font-size: 31px;
        font-weight: 750;
        color: #1d4ed8;
        letter-spacing: -1px;
    }
    .hero-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #172554;
        margin-top: 12px;
        margin-bottom: 5px;
        letter-spacing: -1.5px;
    }
    .hero-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 16px;
        margin-bottom: 22px;
    }
    .section-card {
        background: #ffffff;
        border: 1px solid #e5eaf3;
        border-radius: 18px;
        padding: 22px;
        min-height: 100%;
    }
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #172554;
        margin-bottom: 4px;
    }
    .section-description {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 17px;
    }
    textarea {
        border-radius: 14px !important;
        border: 1px solid #dbe3ef !important;
        background: #fbfdff !important;
        color: #172554 !important;
    }
    textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.10) !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #fbfdff !important;
        border: 1.5px dashed #cbd5e1 !important;
        border-radius: 14px !important;
    }
    [data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"] {
        display: none !important;
    }
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stBaseButton-secondary"] {
        background: linear-gradient(90deg, #2563eb, #6366f1) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 650 !important;
    }
    [data-testid="stFileUploaderDropzone"] button p,
    [data-testid="stFileUploaderDropzone"] button span,
    [data-testid="stBaseButton-secondary"] p,
    [data-testid="stBaseButton-secondary"] span {
        color: #ffffff !important;
    }
    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 13px;
        min-height: 48px;
        background: linear-gradient(90deg, #2563eb, #6366f1, #8b5cf6);
        color: white;
        font-size: 16px;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(79,70,229,0.22);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 30px rgba(79,70,229,0.28);
    }
    .preview-item {
        display: flex;
        gap: 12px;
        align-items: center;
        padding: 10px 0;
    }
    .preview-icon {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        font-size: 17px;
    }
    .preview-green { background: #ecfdf5; }
    .preview-purple { background: #faf5ff; }
    .preview-blue { background: #eff6ff; }
    .preview-red { background: #fef2f2; }
    .preview-yellow { background: #fffbeb; }
    .preview-name { font-weight: 650; font-size: 14px; color: #1e293b; }
    .preview-desc { font-size: 12px; color: #64748b; }
    .privacy {
        margin-top: 20px;
        padding: 16px;
        border-radius: 14px;
        border: 1px solid #bbf7d0;
        background: linear-gradient(135deg, #f0fdf4, #f8fff9);
    }
    .privacy-title { color: #15803d; font-weight: 700; font-size: 14px; }
    .privacy-text { color: #64748b; font-size: 12px; margin-top: 3px; }
    .result-inner {
        background: #f8fafc;
        border: 1px solid #edf2f7;
        border-radius: 14px;
        padding: 16px;
    }
    .sentiment-pill {
        display: inline-block;
        margin-top: 8px;
        padding: 7px 14px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 13px;
        text-transform: uppercase;
    }
    .metric-label { color: #64748b; font-size: 12px; }
    .dist-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.55rem;
        margin: 0.45rem 0 0.5rem 0;
    }
    .dist-card {
        background: #f8fafc;
        border: 1px solid #edf2f7;
        border-radius: 12px;
        padding: 0.7rem 0.55rem;
        text-align: center;
    }
    .dist-label { color: #64748b; font-size: 0.78rem; font-weight: 600; margin-bottom: 0.2rem; }
    .dist-value { color: #172554; font-size: 1.2rem; font-weight: 750; }
    .info-panel {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.45;
        margin-top: 0.75rem;
    }
    .footer {
        text-align: center;
        margin-top: 25px;
        color: #94a3b8;
        font-size: 12px;
    }
    .beta {
        display: inline-block;
        margin-left: 5px;
        padding: 2px 7px;
        border-radius: 20px;
        background: #eef2ff;
        color: #4f46e5;
        font-weight: 700;
        font-size: 10px;
    }
    @media (max-width: 768px) {
        .hero-title { font-size: 31px; }
        .hero-subtitle { font-size: 14px; }
        .dist-grid { grid-template-columns: 1fr; }
    }
    </style>
    """
)

html(
    """
    <div class="brand">
        <div class="brand-icon">🎓</div>
        <div class="brand-name">MyUni</div>
    </div>
    <div class="hero-title">Sentiment Intelligence</div>
    <div class="hero-subtitle">
        AI-powered sentiment and well-being analysis for a healthier campus community.
    </div>
    """
)

left, right = st.columns([1.65, 0.95], gap="large")

with left:
    html(
        """
        <div class="section-card">
            <div class="section-title">Enter your content</div>
            <div class="section-description">
                Share a post, comment, image, audio, or video. Content type is detected automatically.
            </div>
        </div>
        """
    )

    text = st.text_area(
        "Content",
        placeholder="Share your thoughts...",
        height=180,
        max_chars=MAX_CHARS,
        label_visibility="collapsed",
        key="draft_text",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Supports English text")
    with c2:
        st.caption(f"Maximum {MAX_CHARS:,} characters")

    st.markdown("**Or upload an image, audio clip or video**")
    upload = st.file_uploader(
        "Upload media",
        type=MEDIA_TYPES,
        label_visibility="collapsed",
        key="media_upload",
    )
    st.caption("Content type is detected automatically.")

    if upload is not None:
        mime = (upload.type or "").lower()
        name = (upload.name or "").lower()
        if mime.startswith("image/") or name.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            kind = "image"
        elif mime.startswith("audio/") or name.endswith((".wav", ".mp3", ".m4a", ".ogg", ".flac")):
            kind = "audio"
        elif mime.startswith("video/") or name.endswith((".mp4", ".mov", ".webm", ".avi", ".mkv")):
            kind = "video"
        else:
            kind = "file"
        st.session_state.pending_preview = {
            "kind": kind,
            "bytes": upload.getvalue(),
            "name": upload.name,
            "mime": upload.type or "",
        }

    if st.session_state.pending_preview and not (text and text.strip()):
        _show_media_preview(st.session_state.pending_preview)

    html(
        """
        <div class="section-card" style="margin-top:1rem;">
            <div class="section-title">Analyze</div>
            <div class="section-description">
                Our AI will analyze the content and generate sentiment insights.
            </div>
        </div>
        """
    )
    analyze_clicked = st.button("✨  Analyze Now", type="primary", use_container_width=True)

    if analyze_clicked:
        has_text = bool(text and text.strip())
        has_upload = upload is not None

        if has_text and has_upload:
            st.session_state.last_result = {
                "status": CapabilityStatus.VALIDATION_ERROR,
                "message": (
                    "Please analyze one content item at a time. "
                    "Clear either the text field or the media selection, then try again."
                ),
            }
        elif not has_text and not has_upload:
            st.session_state.last_result = {
                "status": CapabilityStatus.VALIDATION_ERROR,
                "message": "Enter text or upload a supported image, audio, or video to analyze.",
            }
        else:
            tmp_root: Optional[Path] = None
            try:
                media_path = None
                mime_type = None
                filename = None
                text_arg = text if has_text else None

                if has_upload:
                    tmp_root = Path(tempfile.mkdtemp(prefix="myuni_upload_"))
                    try:
                        media_path = str(_save_upload(upload, tmp_root))
                        mime_type = upload.type
                        filename = upload.name
                    except Exception as exc:  # noqa: BLE001
                        st.session_state.last_result = {
                            "status": CapabilityStatus.VALIDATION_ERROR,
                            "message": f"Could not read the uploaded file: {exc}",
                        }
                        media_path = None

                if media_path is not None or text_arg:
                    if media_path and not text_arg:
                        try:
                            InputRouter.classify_media(
                                mime_type=mime_type,
                                filename=filename,
                                media_path=media_path,
                            )
                        except ValueError as exc:
                            st.session_state.last_result = {
                                "status": CapabilityStatus.VALIDATION_ERROR,
                                "message": str(exc),
                            }
                            media_path = None

                    if media_path is not None or text_arg:
                        with st.spinner("Analyzing…"):
                            routed = get_pipeline().analyze(
                                text=text_arg,
                                media_path=media_path,
                                mime_type=mime_type,
                                filename=filename,
                                user_id="DEMO-USER",
                            )
                        st.session_state.last_result = {
                            "status": routed.status,
                            "message": routed.message,
                            "detected_input": routed.detected_input,
                            "routed": (
                                routed
                                if routed.status
                                in {
                                    CapabilityStatus.OK,
                                    CapabilityStatus.INSUFFICIENT_EVIDENCE,
                                }
                                else None
                            ),
                            "preview": st.session_state.pending_preview,
                        }
            except Exception as exc:  # noqa: BLE001
                st.session_state.last_result = {
                    "status": CapabilityStatus.VALIDATION_ERROR,
                    "message": f"Analysis failed: {exc}",
                }
            finally:
                if tmp_root is not None:
                    _cleanup_temp_dir(tmp_root)

with right:
    if st.session_state.last_result is None:
        _render_idle_preview()
    else:
        _render_status(st.session_state.last_result)

html(
    """
    <div class="footer">
        MyUni Sentiment Intelligence <span class="beta">Beta</span><br>
        © 2026 MyUni. All rights reserved.
    </div>
    """
)
