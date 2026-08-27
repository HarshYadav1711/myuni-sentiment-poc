#!/usr/bin/env python3
"""MyUni Sentiment Intelligence — Hugging Face Gradio client.

Presentation layer only. All inference goes through MyUniSentimentPipeline.analyze.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any, Optional

import gradio as gr

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import MyUniSentimentPipeline
from src.routing.input_router import CapabilityStatus
from src.ui.gradio_view import (
    BOTH_INPUTS_MESSAGE,
    EMPTY_INPUT_MESSAGE,
    render_idle,
    render_routed_result,
    render_technical_details,
    render_validation,
)

logger = logging.getLogger(__name__)

MAX_CHARS = 5000
MEDIA_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".flac",
    ".mp4",
    ".mov",
    ".webm",
    ".avi",
    ".mkv",
]
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
_VIDEO_EXT = {".mp4", ".mov", ".webm", ".avi", ".mkv"}

_pipeline: Optional[MyUniSentimentPipeline] = None

CSS = """
.gradio-container { max-width: 1120px !important; }
.mu-hero { text-align: center; padding: 0.4rem 0 0.2rem; }
.mu-hero h1 { font-size: 2rem; font-weight: 750; letter-spacing: -0.03em; margin: 0; color: #172554; }
.mu-hero p { color: #64748b; margin: 0.35rem 0 1rem; font-size: 1.02rem; }
.mu-card { background: #fff; border: 1px solid #e5eaf3; border-radius: 16px; padding: 1.05rem 1.1rem; }
.mu-kicker { color: #64748b; font-size: 0.78rem; font-weight: 650; text-transform: uppercase; letter-spacing: 0.04em; }
.mu-detected { font-size: 1.15rem; font-weight: 750; color: #1d4ed8; margin: 0.15rem 0 0.85rem; }
.mu-title { font-size: 1.05rem; font-weight: 700; color: #172554; }
.mu-copy { color: #334155; font-size: 0.95rem; line-height: 1.45; margin: 0.45rem 0; }
.mu-note { color: #64748b; font-size: 0.88rem; margin: 0.2rem 0 0.6rem; }
.mu-list { color: #334155; font-size: 0.92rem; line-height: 1.55; }
.mu-metric { background: #f8fafc; border: 1px solid #edf2f7; border-radius: 12px; padding: 0.9rem; }
.mu-metric-label { color: #64748b; font-size: 0.8rem; font-weight: 650; margin-bottom: 0.35rem; }
.mu-pill { display: inline-block; padding: 0.35rem 0.75rem; border-radius: 999px; font-weight: 750; font-size: 0.82rem; }
.mu-conf { margin-top: 0.55rem; color: #334155; font-size: 0.92rem; }
.mu-model { margin-top: 0.25rem; color: #64748b; font-size: 0.82rem; }
.mu-dist { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin: 0.7rem 0 0.4rem; }
.mu-dist-card { background: #f8fafc; border: 1px solid #edf2f7; border-radius: 12px; padding: 0.65rem 0.4rem; text-align: center; }
.mu-dist-label { color: #64748b; font-size: 0.75rem; font-weight: 650; }
.mu-dist-value { color: #172554; font-size: 1.15rem; font-weight: 750; }
.mu-sub { margin-top: 1rem; font-weight: 720; color: #172554; font-size: 0.95rem; }
.mu-quote { margin: 0.35rem 0 0.7rem; padding: 0.7rem 0.85rem; background: #f8fafc; border-left: 3px solid #6366f1; color: #1e293b; }
.mu-helper { color: #64748b !important; font-size: 0.88rem !important; }
@media (max-width: 720px) { .mu-dist { grid-template-columns: 1fr; } }
"""


def get_pipeline() -> MyUniSentimentPipeline:
    """Shared pipeline singleton. Same ML core as Streamlit/CLI."""
    global _pipeline
    if _pipeline is None:
        _pipeline = MyUniSentimentPipeline()
        # On Hugging Face, place SigLIP 2 on CUDA at startup (ZeroGPU packing).
        # RoBERTa and faster-whisper stay on CPU and are not warmed here.
        if os.environ.get("SPACE_ID"):
            _pipeline.image_analyzer._visual.load()
    return _pipeline


def _media_meta(media: Any) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (path, filename, mime_type) from a Gradio File payload."""
    if media is None:
        return None, None, None
    path: Optional[str] = None
    filename: Optional[str] = None
    if isinstance(media, dict):
        path = media.get("path") or media.get("name")
        filename = media.get("orig_name") or (media.get("meta") or {}).get("name")
        if not filename and path:
            filename = Path(str(path)).name
    elif isinstance(media, (str, Path)):
        path = str(media)
        filename = Path(path).name
    else:
        path = getattr(media, "name", None) or str(media)
        filename = getattr(media, "orig_name", None) or (Path(str(path)).name if path else None)
    if not path:
        return None, None, None
    name = str(filename or Path(path).name)
    mime, _ = mimetypes.guess_type(name)
    return str(path), name, mime


def _kind_from_name(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    ext = Path(filename).suffix.lower()
    if ext in _IMAGE_EXT:
        return "image"
    if ext in _AUDIO_EXT:
        return "audio"
    if ext in _VIDEO_EXT:
        return "video"
    return None


def preview_media(media: Any) -> tuple[Any, Any, Any]:
    """Show the actual uploaded image, audio, or video."""
    hidden_img = gr.update(visible=False, value=None)
    hidden_aud = gr.update(visible=False, value=None)
    hidden_vid = gr.update(visible=False, value=None)
    path, filename, _mime = _media_meta(media)
    if not path:
        return hidden_img, hidden_aud, hidden_vid
    kind = _kind_from_name(filename)
    if kind == "image":
        return gr.update(visible=True, value=path), hidden_aud, hidden_vid
    if kind == "audio":
        return hidden_img, gr.update(visible=True, value=path), hidden_vid
    if kind == "video":
        return hidden_img, hidden_aud, gr.update(visible=True, value=path)
    return hidden_img, hidden_aud, hidden_vid


def analyze_content(text: Optional[str], media: Any) -> tuple[Any, Any, Any, str, str]:
    """Validate one item, then call the existing pipeline entrypoint."""
    previews = preview_media(media)
    has_text = bool(text and str(text).strip())
    has_media = media is not None

    if has_text and has_media:
        return (*previews, render_validation(BOTH_INPUTS_MESSAGE), "")
    if not has_text and not has_media:
        return (*previews, render_validation(EMPTY_INPUT_MESSAGE), "")

    media_path = None
    mime_type = None
    filename = None
    text_arg = str(text).strip() if has_text else None
    if has_media:
        media_path, filename, mime_type = _media_meta(media)
        if not media_path:
            return (*previews, render_validation("Upload could not be saved for analysis. Please try again."), "")

    try:
        routed = get_pipeline().analyze(
            text=text_arg,
            media_path=media_path,
            mime_type=mime_type,
            filename=filename,
            user_id="DEMO-USER",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Gradio analysis failed")
        return (*previews, render_validation(f"Analysis failed: {exc}"), "")

    html = render_routed_result(routed)
    tech = ""
    if routed.status in {CapabilityStatus.OK, CapabilityStatus.INSUFFICIENT_EVIDENCE}:
        tech = render_technical_details(routed)
    return (*previews, html, tech)


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="MyUni Sentiment Intelligence",
        css=CSS,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    ) as blocks:
        gr.HTML(
            """
            <div class="mu-hero">
              <h1>MyUni Sentiment Intelligence</h1>
              <p>Multimodal social-content sentiment analysis with automatic input detection.</p>
            </div>
            """
        )
        with gr.Row():
            with gr.Column(scale=11):
                text = gr.Textbox(
                    label="Text",
                    placeholder="Type or paste a social post...",
                    lines=7,
                    max_lines=12,
                    max_length=MAX_CHARS,
                )
                media = gr.File(
                    label="Or upload an image, audio clip or video",
                    file_count="single",
                    file_types=MEDIA_EXTENSIONS,
                )
                gr.Markdown("Content type is detected automatically.", elem_classes=["mu-helper"])
                analyze_btn = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=10):
                image_out = gr.Image(label="Uploaded image", visible=False, interactive=False)
                audio_out = gr.Audio(label="Uploaded audio", visible=False, interactive=True)
                video_out = gr.Video(label="Uploaded video", visible=False, interactive=True)
                result = gr.HTML(value=render_idle())
                with gr.Accordion("Technical details", open=False):
                    tech = gr.Markdown(value="")

        media.change(
            fn=preview_media,
            inputs=[media],
            outputs=[image_out, audio_out, video_out],
        )
        analyze_btn.click(
            fn=analyze_content,
            inputs=[text, media],
            outputs=[image_out, audio_out, video_out, result, tech],
        )
    return blocks


demo = build_demo()
demo.queue(max_size=8, default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch()
