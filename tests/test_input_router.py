"""Tests for automatic input detection and capability-aware routing."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_TEXT_MODEL
from src.pipeline import MyUniSentimentPipeline
from src.routing.input_router import (
    CLIENT_ENABLED_MODALITIES,
    CapabilityStatus,
    InputRouter,
    InputType,
)
from src.schemas import (
    ActivityAnalysisResult,
    AnalysisBlock,
    InputMetadata,
    ModalityBundle,
    SentimentEvidence,
)


def test_detect_text_priority_over_media() -> None:
    detected = InputRouter.detect(
        text="  Hello campus  ",
        media_path="ignored.png",
        mime_type="image/png",
        filename="ignored.png",
    )
    assert detected.input_type == InputType.TEXT
    assert detected.text == "Hello campus"


def test_detect_image_mime() -> None:
    detected = InputRouter.detect(
        text="   ",
        mime_type="image/png",
        filename="photo.png",
        media_path="/tmp/photo.png",
    )
    assert detected.input_type == InputType.IMAGE


def test_detect_image_jpeg_mime() -> None:
    detected = InputRouter.detect(mime_type="image/jpeg", filename="a.jpg")
    assert detected.input_type == InputType.IMAGE


def test_detect_video_mime() -> None:
    detected = InputRouter.detect(
        mime_type="video/mp4",
        filename="clip.mp4",
        media_path="/tmp/clip.mp4",
    )
    assert detected.input_type == InputType.VIDEO


def test_detect_video_by_extension_when_mime_missing() -> None:
    detected = InputRouter.detect(filename="talk.webm", media_path="talk.webm")
    assert detected.input_type == InputType.VIDEO


def test_unsupported_file_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported file type"):
        InputRouter.detect(filename="notes.pdf", mime_type="application/pdf")


def test_blank_input_raises() -> None:
    with pytest.raises(ValueError, match="Enter text or upload"):
        InputRouter.detect(text="  ")


def test_mime_extension_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        InputRouter.detect(mime_type="image/png", filename="clip.mp4")


def test_client_capabilities_text_only() -> None:
    assert InputType.TEXT in CLIENT_ENABLED_MODALITIES
    assert InputType.IMAGE not in CLIENT_ENABLED_MODALITIES
    assert InputType.VIDEO not in CLIENT_ENABLED_MODALITIES


def _fake_text_result() -> ActivityAnalysisResult:
    evidence = SentimentEvidence(
        label="positive",
        score=0.8,
        confidence=0.9,
        probabilities={"positive": 0.9, "neutral": 0.05, "negative": 0.05},
        model=DEFAULT_TEXT_MODEL,
    )
    return ActivityAnalysisResult(
        activity_id="ACT-TEST",
        user_id="DEMO-USER",
        activity_type="text",
        input=InputMetadata(text_length=10, text_preview="hello"),
        analysis=AnalysisBlock(
            overall=evidence,
            modalities=ModalityBundle(text=evidence),
        ),
    )


def test_pipeline_analyze_blank_validation() -> None:
    pipeline = MyUniSentimentPipeline.__new__(MyUniSentimentPipeline)
    routed = MyUniSentimentPipeline.analyze(
        pipeline,
        text="   ",
    )
    assert routed.status == CapabilityStatus.VALIDATION_ERROR
    assert routed.analysis is None


def test_pipeline_analyze_image_not_implemented_no_fake_scores() -> None:
    pipeline = MyUniSentimentPipeline.__new__(MyUniSentimentPipeline)
    routed = MyUniSentimentPipeline.analyze(
        pipeline,
        mime_type="image/png",
        filename="campus.png",
        media_path="/tmp/campus.png",
    )
    assert routed.status == CapabilityStatus.NOT_IMPLEMENTED
    assert routed.detected_input == InputType.IMAGE
    assert routed.analysis is None
    assert "not enabled" in (routed.message or "").lower()
    assert "detected and routed successfully" in (routed.message or "").lower()


def test_pipeline_analyze_video_not_implemented_no_fake_scores() -> None:
    pipeline = MyUniSentimentPipeline.__new__(MyUniSentimentPipeline)
    routed = MyUniSentimentPipeline.analyze(
        pipeline,
        mime_type="video/mp4",
        filename="event.mp4",
        media_path="/tmp/event.mp4",
    )
    assert routed.status == CapabilityStatus.NOT_IMPLEMENTED
    assert routed.detected_input == InputType.VIDEO
    assert routed.analysis is None
    assert "not enabled" in (routed.message or "").lower()
    assert "detected and routed successfully" in (routed.message or "").lower()


def test_pipeline_analyze_text_uses_same_analyze_text_path() -> None:
    pipeline = MyUniSentimentPipeline.__new__(MyUniSentimentPipeline)
    expected = _fake_text_result()
    pipeline.analyze_text = MagicMock(return_value=expected)  # type: ignore[method-assign]

    routed = MyUniSentimentPipeline.analyze(
        pipeline,
        text="The campus event today was genuinely amazing.",
        user_id="DEMO-USER",
    )

    assert routed.status == CapabilityStatus.OK
    assert routed.detected_input == InputType.TEXT
    assert routed.analysis is expected
    assert routed.model_display_name == "Twitter-RoBERTa"
    assert routed.model_id == DEFAULT_TEXT_MODEL
    pipeline.analyze_text.assert_called_once()
    args, kwargs = pipeline.analyze_text.call_args
    assert args[0] == "The campus event today was genuinely amazing."
    assert kwargs.get("user_id") == "DEMO-USER"


def test_unsupported_via_pipeline_analyze() -> None:
    pipeline = MyUniSentimentPipeline.__new__(MyUniSentimentPipeline)
    routed = MyUniSentimentPipeline.analyze(
        pipeline,
        filename="doc.docx",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert routed.status == CapabilityStatus.VALIDATION_ERROR
    assert routed.analysis is None
