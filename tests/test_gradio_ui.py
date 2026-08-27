"""Gradio client presentation tests (no model load)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_TEXT_MODEL, DEFAULT_VISUAL_MODEL
from src.routing.input_router import CapabilityStatus, InputType, RoutedAnalysisResult
from src.schemas import (
    ActivityAnalysisResult,
    AnalysisBlock,
    FusionDiagnostics,
    InputMetadata,
    ModalityBundle,
    SentimentEvidence,
    VideoDiagnostics,
)
from src.ui.gradio_view import (
    BOTH_INPUTS_MESSAGE,
    NO_SPEECH_MESSAGE,
    render_idle,
    render_routed_result,
    render_technical_details,
    render_validation,
)


def _ev(label: str, *, model: str = "stub") -> SentimentEvidence:
    probs = {"positive": 0.1, "neutral": 0.1, "negative": 0.1}
    probs[label] = 0.8
    rem = 0.1
    for key in probs:
        if key != label:
            probs[key] = rem
    return SentimentEvidence(
        label=label,  # type: ignore[arg-type]
        score=probs["positive"] - probs["negative"],
        confidence=0.8,
        probabilities=probs,
        model=model,
    )


def test_gradio_entrypoint_exists_and_uses_pipeline() -> None:
    source = (ROOT / "app_gradio.py").read_text(encoding="utf-8")
    assert "MyUni Sentiment Intelligence" in source
    assert "Multimodal social-content sentiment analysis with automatic input detection." in source
    assert "Type or paste a social post..." in source
    assert "Or upload an image, audio clip or video" in source
    assert "Content type is detected automatically." in source
    assert 'gr.Button("Analyze"' in source
    assert "get_pipeline().analyze(" in source
    assert "MyUniSentimentPipeline" in source
    assert "@spaces.GPU" not in source
    assert "st.tabs" not in source


def test_streamlit_app_still_present() -> None:
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "def get_pipeline" in source
    assert "MyUniSentimentPipeline" in source
    assert "streamlit" in source


def test_both_inputs_message() -> None:
    html = render_validation(BOTH_INPUTS_MESSAGE)
    assert BOTH_INPUTS_MESSAGE in html
    assert "depressed" not in html.lower()
    assert "mental illness" not in html.lower()


def test_render_text_result() -> None:
    result = ActivityAnalysisResult(
        activity_id="A1",
        user_id="DEMO-USER",
        activity_type="text",
        input=InputMetadata(text_preview="hello"),
        analysis=AnalysisBlock(
            overall=_ev("positive", model="poc-fusion"),
            modalities=ModalityBundle(text=_ev("positive", model=DEFAULT_TEXT_MODEL)),
        ),
    )
    routed = RoutedAnalysisResult(
        status=CapabilityStatus.OK,
        detected_input=InputType.TEXT,
        analysis=result,
        model_display_name="Twitter-RoBERTa",
        model_id=DEFAULT_TEXT_MODEL,
    )
    html = render_routed_result(routed)
    assert "TEXT" in html
    assert "Overall Sentiment" in html
    assert "Twitter-RoBERTa" in html
    assert "POSITIVE" in html


def test_render_audio_no_speech() -> None:
    routed = RoutedAnalysisResult(
        status=CapabilityStatus.INSUFFICIENT_EVIDENCE,
        detected_input=InputType.AUDIO,
        message="unused",
    )
    html = render_routed_result(routed)
    assert NO_SPEECH_MESSAGE in html
    assert "NEUTRAL" not in html


def test_render_video_visual_only() -> None:
    result = ActivityAnalysisResult(
        activity_id="A2",
        user_id="DEMO-USER",
        activity_type="video",
        input=InputMetadata(media_path="clip.mp4"),
        analysis=AnalysisBlock(
            overall=_ev("negative", model="poc-fusion"),
            modalities=ModalityBundle(visual=_ev("negative", model=DEFAULT_VISUAL_MODEL)),
            fusion=FusionDiagnostics(
                contributing_modalities=["visual"],
                explanation="Only visual evidence was usable.",
            ),
            video=VideoDiagnostics(
                sampling_strategy="fixed_fps",
                frames_extracted=2,
                frames_analyzed=2,
            ),
        ),
    )
    routed = RoutedAnalysisResult(
        status=CapabilityStatus.OK,
        detected_input=InputType.VIDEO,
        analysis=result,
        model_display_name="SigLIP 2 + Faster-Whisper + Twitter-RoBERTa",
        model_id=DEFAULT_VISUAL_MODEL,
    )
    html = render_routed_result(routed)
    assert "VIDEO" in html
    assert "Visual Evidence" in html
    assert "Speech Evidence" in html
    assert NO_SPEECH_MESSAGE in html
    assert "visual evidence only" in html.lower()
    tech = render_technical_details(routed)
    assert "google/siglip2-base-patch16-224" in tech or DEFAULT_VISUAL_MODEL in tech


def test_ocr_unavailable_stays_in_technical_details() -> None:
    result = ActivityAnalysisResult(
        activity_id="A3",
        activity_type="image",
        input=InputMetadata(),
        analysis=AnalysisBlock(
            overall=_ev("negative", model="poc-fusion"),
            modalities=ModalityBundle(visual=_ev("negative", model=DEFAULT_VISUAL_MODEL)),
            warnings=["OCR unavailable: Tesseract executable not found"],
        ),
    )
    routed = RoutedAnalysisResult(
        status=CapabilityStatus.OK,
        detected_input=InputType.IMAGE,
        analysis=result,
        model_display_name="SigLIP 2",
        model_id=DEFAULT_VISUAL_MODEL,
    )
    html = render_routed_result(routed)
    assert "OCR unavailable" not in html
    assert "Visual Sentiment" in html
    tech = render_technical_details(routed)
    assert "OCR" in tech


def test_idle_and_client_safe_language() -> None:
    html = render_idle()
    lowered = html.lower()
    for banned in ("depressed", "mentally disturbed", "mental disorder", "mental illness", "suicide", "clinical diagnosis"):
        assert banned not in lowered


def test_gradio_module_imports_without_loading_models() -> None:
    import app_gradio

    assert app_gradio.demo is not None
    assert app_gradio._pipeline is None
    assert "analyze" in app_gradio.analyze_content.__name__
