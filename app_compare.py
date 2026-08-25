#!/usr/bin/env python3
"""MyUni Sentiment Intelligence — client demo UI (unified input).

Uses ``MyUniSentimentPipeline.analyze`` for routing:
- Text → Twitter-RoBERTa
- Image / Video → detection + preview + not_implemented (no fake scores)
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
from src.routing.input_router import CapabilityStatus, InputRouter, InputType
from src.ui.display import (
    format_confidence_pct,
    format_probability_pct,
    format_score,
    label_color,
)

logger = logging.getLogger(__name__)

MAX_CHARS = 5000
DEMO_DIR = ROOT / "demo_assets"
DEMO_TEXT = "The campus event today was genuinely amazing."
DEMO_POSITIVE_IMAGE_NAMES = ("positive_image.jpg", "positive_image.jpeg", "positive_image.png")
DEMO_NEGATIVE_IMAGE_NAMES = ("negative_image.jpg", "negative_image.jpeg", "negative_image.png")
DEMO_VIDEO_NAMES = ("demo_video.mp4", "demo_video.mov", "demo_video.webm")
MEDIA_TYPES = ["jpg", "jpeg", "png", "webp", "mp4", "mov", "webm", "avi", "mkv"]


def css(markup: str) -> None:
    st.markdown(dedent(markup).strip(), unsafe_allow_html=True)


def html(markup: str) -> None:
    st.html(dedent(markup).strip())


@st.cache_resource(show_spinner="Loading text analysis model (first run only)…")
def get_pipeline() -> MyUniSentimentPipeline:
    return MyUniSentimentPipeline()


def _find_demo_file(names: tuple[str, ...]) -> Optional[Path]:
    for name in names:
        path = DEMO_DIR / name
        if path.is_file():
            return path
    return None


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


def _mime_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }.get(ext, "application/octet-stream")


def _load_bytes(path: Path) -> bytes:
    return path.read_bytes()


st.set_page_config(
    page_title="MyUni Sentiment Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "draft_text" not in st.session_state:
    st.session_state.draft_text = ""
if "demo_media_path" not in st.session_state:
    st.session_state.demo_media_path = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "pending_preview" not in st.session_state:
    st.session_state.pending_preview = None

css(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 8% 12%, rgba(99,102,241,0.08), transparent 28%),
            radial-gradient(circle at 92% 10%, rgba(59,130,246,0.07), transparent 26%),
            #f8faff;
        color: #172554;
    }
    .block-container {
        max-width: 1080px;
        padding-top: 2.4rem !important;
        padding-bottom: 2rem !important;
    }
    #MainMenu, footer { visibility: hidden; }
    header { background: transparent !important; }
    div[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }

    .brand {
        display: flex; align-items: center; justify-content: center;
        gap: 10px; margin-bottom: 0.35rem;
    }
    .brand-mark {
        width: 40px; height: 40px; border-radius: 12px;
        background: linear-gradient(135deg, #2563eb, #6366f1);
        color: #fff; font-weight: 750; font-size: 1.05rem;
        display: inline-flex; align-items: center; justify-content: center;
        box-shadow: 0 8px 22px rgba(79,70,229,0.22);
    }
    .brand-name {
        font-size: 1.35rem; font-weight: 750; color: #1d4ed8; letter-spacing: -0.03em;
    }
    .hero-title {
        text-align: center; font-size: 2.15rem; font-weight: 800;
        color: #172554; letter-spacing: -0.035em; margin: 0.2rem 0 0.35rem 0;
    }
    .hero-sub {
        text-align: center; color: #64748b; font-size: 0.98rem;
        margin: 0 auto 1.1rem auto; max-width: 560px; line-height: 1.45;
    }
    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #172554; margin: 0 0 0.35rem 0;
    }
    .helper { color: #64748b; font-size: 0.88rem; margin: 0.2rem 0 0.75rem 0; }
    .card {
        background: #fff; border: 1px solid #e5eaf3; border-radius: 18px;
        padding: 1.1rem 1.15rem; box-shadow: 0 14px 40px rgba(30,41,59,0.06);
    }
    .result-panel {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 0.95rem 1rem; margin: 0.55rem 0;
    }
    .result-label {
        font-size: 1.35rem; font-weight: 750; letter-spacing: 0.04em;
        text-transform: uppercase; margin-top: 0.2rem;
    }
    .muted { color: #64748b; font-size: 0.86rem; }
    .info-panel {
        background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 0.85rem 0.95rem; color: #334155; font-size: 0.92rem; line-height: 1.45;
    }
    .dist-grid {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-top: 0.45rem;
    }
    .dist-card {
        background: #fff; border: 1px solid #e5eaf3; border-radius: 12px;
        padding: 0.65rem 0.5rem; text-align: center;
    }
    .dist-label { color: #64748b; font-size: 0.76rem; font-weight: 600; }
    .dist-value { color: #172554; font-size: 1.15rem; font-weight: 750; margin-top: 0.15rem; }
    .meta-line { color: #64748b; font-size: 0.8rem; margin-top: 0.35rem; }
    .footer {
        text-align: center; color: #94a3b8; font-size: 0.78rem; margin-top: 1.1rem; line-height: 1.5;
    }
    .stButton > button[kind="primary"] {
        width: 100%; border: none; border-radius: 12px; min-height: 2.85rem;
        background: linear-gradient(90deg, #2563eb, #6366f1, #8b5cf6) !important;
        color: #fff !important; font-weight: 700 !important;
    }
    [data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"] {
        display: none !important;
    }
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stBaseButton-secondary"] {
        background: linear-gradient(90deg, #2563eb, #6366f1) !important;
        color: #fff !important; border: none !important; border-radius: 10px !important;
    }
    [data-testid="stFileUploaderDropzone"] button p,
    [data-testid="stFileUploaderDropzone"] button span,
    [data-testid="stBaseButton-secondary"] p,
    [data-testid="stBaseButton-secondary"] span { color: #fff !important; }
    </style>
    """
)

html(
    """
    <div class="brand">
      <div class="brand-mark">U</div>
      <div class="brand-name">MyUni</div>
    </div>
    <div class="hero-title">MyUni Sentiment Intelligence</div>
    <div class="hero-sub">Multimodal social-content analysis with automatic input detection.</div>
    """
)

# ---- Demo examples (optional assets) ----
pos_img = _find_demo_file(DEMO_POSITIVE_IMAGE_NAMES)
neg_img = _find_demo_file(DEMO_NEGATIVE_IMAGE_NAMES)
demo_vid = _find_demo_file(DEMO_VIDEO_NAMES)

with st.expander("Demo examples", expanded=False):
    st.caption("Load a sample input for the client walkthrough. Media samples appear when present in demo_assets/.")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        if st.button("Sample text", use_container_width=True):
            st.session_state.draft_text = DEMO_TEXT
            st.session_state.demo_media_path = None
            st.session_state.pending_preview = None
            st.session_state.last_result = None
            st.rerun()
    with d2:
        if pos_img is None:
            st.caption("Positive image missing")
        elif st.button("Positive image", use_container_width=True):
            st.session_state.draft_text = ""
            st.session_state.demo_media_path = str(pos_img)
            st.session_state.pending_preview = {
                "kind": "image",
                "path": str(pos_img),
                "name": pos_img.name,
                "mime": _mime_for_path(pos_img),
            }
            st.session_state.last_result = None
            st.rerun()
    with d3:
        if neg_img is None:
            st.caption("Negative image missing")
        elif st.button("Negative image", use_container_width=True):
            st.session_state.draft_text = ""
            st.session_state.demo_media_path = str(neg_img)
            st.session_state.pending_preview = {
                "kind": "image",
                "path": str(neg_img),
                "name": neg_img.name,
                "mime": _mime_for_path(neg_img),
            }
            st.session_state.last_result = None
            st.rerun()
    with d4:
        if demo_vid is None:
            st.caption("Demo video missing")
        elif st.button("Demo video", use_container_width=True):
            st.session_state.draft_text = ""
            st.session_state.demo_media_path = str(demo_vid)
            st.session_state.pending_preview = {
                "kind": "video",
                "path": str(demo_vid),
                "name": demo_vid.name,
                "mime": _mime_for_path(demo_vid),
            }
            st.session_state.last_result = None
            st.rerun()

html('<div class="card">')
st.markdown('<div class="section-title">Analyze Content</div>', unsafe_allow_html=True)

text = st.text_area(
    "Type or paste a social post",
    height=140,
    max_chars=MAX_CHARS,
    key="draft_text",
    placeholder="Type or paste a social post",
)

upload = st.file_uploader(
    "Or upload an image or video",
    type=MEDIA_TYPES,
    key="media_upload",
)
st.caption("Content type is detected automatically.")

# Fresh upload clears demo media selection
if upload is not None:
    st.session_state.demo_media_path = None
    mime = (upload.type or "").lower()
    kind = "image" if mime.startswith("image/") else ("video" if mime.startswith("video/") else "file")
    st.session_state.pending_preview = {
        "kind": kind,
        "bytes": upload.getvalue(),
        "name": upload.name,
        "mime": upload.type or "",
    }

# Live media preview (upload or demo asset)
preview = st.session_state.pending_preview
if preview is not None and not (text and text.strip()):
    try:
        if preview["kind"] == "image":
            data = preview.get("bytes")
            if data is None and preview.get("path"):
                data = _load_bytes(Path(preview["path"]))
            if data:
                st.image(data, use_container_width=True)
            st.markdown(
                f'<div class="meta-line">{preview.get("name", "")}'
                f' · {preview.get("mime") or "image"}</div>',
                unsafe_allow_html=True,
            )
        elif preview["kind"] == "video":
            data = preview.get("bytes")
            path = preview.get("path")
            if data is not None:
                st.video(data)
            elif path:
                st.video(str(path))
            st.markdown(
                f'<div class="meta-line">{preview.get("name", "")}'
                f' · {preview.get("mime") or "video"}</div>',
                unsafe_allow_html=True,
            )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not preview media: {exc}")

analyze = st.button("Analyze", type="primary", use_container_width=True)
html("</div>")

if analyze:
    has_text = bool(text and text.strip())
    has_upload = upload is not None
    has_demo = bool(st.session_state.demo_media_path)

    if has_text and (has_upload or has_demo):
        st.session_state.last_result = {
            "status": CapabilityStatus.VALIDATION_ERROR,
            "message": (
                "Please analyze one content item at a time. "
                "Clear either the text field or the media selection, then try again."
            ),
        }
    elif not has_text and not has_upload and not has_demo:
        st.session_state.last_result = {
            "status": CapabilityStatus.VALIDATION_ERROR,
            "message": "Enter text or upload a supported image/video to analyze.",
        }
    else:
        tmp_root: Optional[Path] = None
        try:
            media_path = None
            mime_type = None
            filename = None
            text_arg = text if has_text else None

            if has_upload:
                tmp_root = Path(tempfile.mkdtemp(prefix="myuni_demo_"))
                try:
                    media_path = str(_save_upload(upload, tmp_root))
                except Exception as exc:  # noqa: BLE001
                    st.session_state.last_result = {
                        "status": CapabilityStatus.VALIDATION_ERROR,
                        "message": f"Could not read the uploaded file: {exc}",
                    }
                    media_path = None
                else:
                    mime_type = upload.type
                    filename = upload.name
            elif has_demo:
                demo_path = Path(st.session_state.demo_media_path)
                if not demo_path.is_file():
                    st.session_state.last_result = {
                        "status": CapabilityStatus.VALIDATION_ERROR,
                        "message": "Demo media file is missing. Re-select a sample or upload a file.",
                    }
                else:
                    media_path = str(demo_path)
                    mime_type = _mime_for_path(demo_path)
                    filename = demo_path.name

            if media_path is not None or text_arg:
                # Validate media kind early for corrupt / unsupported cases
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
                    preview_payload = st.session_state.pending_preview
                    st.session_state.last_result = {
                        "status": routed.status,
                        "message": routed.message,
                        "detected_input": routed.detected_input,
                        "routed": routed if routed.status == CapabilityStatus.OK else None,
                        "preview": preview_payload,
                    }
        except Exception as exc:  # noqa: BLE001
            st.session_state.last_result = {
                "status": CapabilityStatus.VALIDATION_ERROR,
                "message": f"Analysis failed: {exc}",
            }
        finally:
            if tmp_root is not None:
                _cleanup_temp_dir(tmp_root)

# ---- Results ----
result = st.session_state.last_result
if result is not None:
    status = result.get("status")
    detected = result.get("detected_input")
    preview = result.get("preview") or st.session_state.pending_preview

    if status == CapabilityStatus.VALIDATION_ERROR:
        st.warning(result.get("message") or "Invalid input.")

    elif status == CapabilityStatus.NOT_IMPLEMENTED:
        label = detected.value.upper() if detected else "MEDIA"
        st.markdown('<div class="section-title">Detected Input</div>', unsafe_allow_html=True)
        st.write(label)

        # Keep media visible with the result
        if preview:
            try:
                if preview.get("kind") == "image":
                    data = preview.get("bytes")
                    if data is None and preview.get("path"):
                        data = _load_bytes(Path(preview["path"]))
                    if data:
                        st.image(data, use_container_width=True)
                    st.caption(f'{preview.get("name", "")} · {preview.get("mime") or "image"}')
                elif preview.get("kind") == "video":
                    if preview.get("bytes") is not None:
                        st.video(preview["bytes"])
                    elif preview.get("path"):
                        st.video(str(preview["path"]))
                    st.caption(f'{preview.get("name", "")} · {preview.get("mime") or "video"}')
            except Exception as exc:  # noqa: BLE001
                st.info(f"Media preview unavailable: {exc}")

        html(f'<div class="info-panel">{result.get("message") or ""}</div>')

    elif status == CapabilityStatus.OK and result.get("routed") is not None:
        routed = result["routed"]
        analysis = routed.analysis
        text_ev = analysis.analysis.modalities.text
        evidence = text_ev or analysis.analysis.overall
        probs = evidence.probabilities or {}
        color = label_color(evidence.label)

        st.markdown('<div class="section-title">Detected Input</div>', unsafe_allow_html=True)
        st.write("TEXT")
        html(
            f"""
            <div class="result-panel">
              <div class="muted">Overall Sentiment</div>
              <div class="result-label" style="color:{color}">{evidence.label}</div>
              <div style="margin-top:0.55rem">
                Confidence <strong>{format_confidence_pct(evidence.confidence)}</strong>
                &nbsp;·&nbsp;
                Score <strong>{format_score(evidence.score)}</strong>
              </div>
              <div class="muted" style="margin-top:0.4rem">
                Model · {routed.model_display_name or "Twitter-RoBERTa"}
              </div>
            </div>
            <div class="muted" style="font-weight:650;margin-top:0.35rem">Probability distribution</div>
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
        )
        with st.expander("Technical details", expanded=False):
            st.code(routed.model_id or "cardiffnlp/twitter-roberta-base-sentiment-latest")
            st.caption(
                "Label = highest model probability. "
                "POC score = P(positive) − P(negative). Evaluation default only.",
            )

html(
    """
    <div class="footer">
      MyUni Sentiment Intelligence · Beta<br>
      Current text analysis measures general social-media sentiment.
    </div>
    """
)
