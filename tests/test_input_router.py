"""Tests for automatic input detection and capability-aware routing."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_FUSION, DEFAULT_TEXT_MODEL, DEFAULT_VISUAL_MODEL
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


def test_detect_audio_mime() -> None:
    detected = InputRouter.detect(
        mime_type="audio/wav",
        filename="clip.wav",
        media_path="/tmp/clip.wav",
    )
    assert detected.input_type == InputType.AUDIO


def test_detect_audio_by_extension() -> None:
    detected = InputRouter.detect(filename="voice.m4a", media_path="voice.m4a")
    assert detected.input_type == InputType.AUDIO


def test_detect_video_mime() -> None:
    detected = InputRouter.detect(
        mime_type="video/mp4",
        filename="clip.mp4",
        media_path="/tmp/clip.mp4",
    )
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


def test_client_capabilities_multimodal() -> None:
    assert InputType.TEXT in CLIENT_ENABLED_MODALITIES
    assert InputType.IMAGE in CLIENT_ENABLED_MODALITIES
    assert InputType.AUDIO in CLIENT_ENABLED_MODALITIES
    assert InputType.VIDEO in CLIENT_ENABLED_MODALITIES


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


def _visual_evidence() -> SentimentEvidence:
    return SentimentEvidence(
        label="positive",
        score=0.5,
        confidence=0.6,
        probabilities={"positive": 0.6, "neutral": 0.3, "negative": 0.1},
        model=DEFAULT_VISUAL_MODEL,
        details={"raw_similarities": {"positive": 0.7, "neutral": 0.4, "negative": 0.2}},
    )


def test_pipeline_analyze_blank_validation() -> None:
    pipeline = MyUniSentimentPipeline.__new__(MyUniSentimentPipeline)
    routed = MyUniSentimentPipeline.analyze(pipeline, text="   ")
    assert routed.status == CapabilityStatus.VALIDATION_ERROR
    assert routed.analysis is None


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


def test_pipeline_analyze_image_ok(tmp_path: Path) -> None:
    img = tmp_path / "campus.png"
    img.write_bytes(b"fake")
    visual = _visual_evidence()
    result = ActivityAnalysisResult(
        activity_id="ACT-IMG",
        user_id="DEMO-USER",
        activity_type="image",
        input=InputMetadata(media_path=str(img)),
        analysis=AnalysisBlock(
            overall=visual,
            modalities=ModalityBundle(visual=visual),
        ),
    )
    pipeline = MyUniSentimentPipeline.__new__(MyUniSentimentPipeline)
    pipeline._client_analyze_image = MagicMock(return_value=result)  # type: ignore[method-assign]

    routed = MyUniSentimentPipeline.analyze(
        pipeline,
        mime_type="image/png",
        filename="campus.png",
        media_path=str(img),
    )
    assert routed.status == CapabilityStatus.OK
    assert routed.detected_input == InputType.IMAGE
    assert routed.analysis is result
    assert routed.model_display_name == "SigLIP 2"
    assert routed.model_id == DEFAULT_VISUAL_MODEL


def test_pipeline_analyze_audio_no_speech(tmp_path: Path) -> None:
    wav = tmp_path / "silent.wav"
    wav.write_bytes(b"RIFF")
    empty = ActivityAnalysisResult(
        activity_id="ACT-AUD",
        user_id="DEMO-USER",
        activity_type="audio",
        input=InputMetadata(media_path=str(wav)),
        analysis=AnalysisBlock(
            overall=SentimentEvidence(
                label="neutral",
                score=0.0,
                confidence=0.0,
                probabilities={"positive": 0.0, "neutral": 1.0, "negative": 0.0},
                model="poc-fusion",
            ),
            modalities=ModalityBundle(speech=None),
            warnings=["No speech detected (empty transcript)"],
            transcript=None,
        ),
    )
    pipeline = MyUniSentimentPipeline.__new__(MyUniSentimentPipeline)
    pipeline._client_analyze_audio = MagicMock(return_value=empty)  # type: ignore[method-assign]

    routed = MyUniSentimentPipeline.analyze(
        pipeline,
        mime_type="audio/wav",
        filename="silent.wav",
        media_path=str(wav),
    )
    assert routed.status == CapabilityStatus.INSUFFICIENT_EVIDENCE
    assert routed.detected_input == InputType.AUDIO
    assert "no usable speech" in (routed.message or "").lower()


def test_pipeline_analyze_audio_ok(tmp_path: Path) -> None:
    wav = tmp_path / "speech.wav"
    wav.write_bytes(b"RIFF")
    speech = SentimentEvidence(
        label="positive",
        score=0.7,
        confidence=0.8,
        probabilities={"positive": 0.8, "neutral": 0.1, "negative": 0.1},
        model=DEFAULT_TEXT_MODEL,
    )
    result = ActivityAnalysisResult(
        activity_id="ACT-AUD",
        user_id="DEMO-USER",
        activity_type="audio",
        input=InputMetadata(media_path=str(wav)),
        analysis=AnalysisBlock(
            overall=speech,
            modalities=ModalityBundle(speech=speech),
            transcript="Campus day was great.",
        ),
    )
    pipeline = MyUniSentimentPipeline.__new__(MyUniSentimentPipeline)
    pipeline._client_analyze_audio = MagicMock(return_value=result)  # type: ignore[method-assign]

    routed = MyUniSentimentPipeline.analyze(
        pipeline,
        mime_type="audio/mpeg",
        filename="speech.mp3",
        media_path=str(wav),
    )
    assert routed.status == CapabilityStatus.OK
    assert routed.detected_input == InputType.AUDIO
    assert routed.analysis.analysis.transcript == "Campus day was great."


def test_pipeline_analyze_video_insufficient(tmp_path: Path) -> None:
    vid = tmp_path / "empty.mp4"
    vid.write_bytes(b"fake")
    from src.fusion import fuse_modalities

    fusion = fuse_modalities({}, config=DEFAULT_FUSION)
    result = ActivityAnalysisResult(
        activity_id="ACT-VID",
        user_id="DEMO-USER",
        activity_type="video",
        input=InputMetadata(media_path=str(vid)),
        analysis=AnalysisBlock(
            overall=fusion.overall,
            modalities=ModalityBundle(),
            fusion=fusion.diagnostics,
        ),
    )
    pipeline = MyUniSentimentPipeline.__new__(MyUniSentimentPipeline)
    pipeline._client_analyze_video = MagicMock(return_value=result)  # type: ignore[method-assign]

    routed = MyUniSentimentPipeline.analyze(
        pipeline,
        mime_type="video/mp4",
        filename="empty.mp4",
        media_path=str(vid),
    )
    assert routed.status == CapabilityStatus.INSUFFICIENT_EVIDENCE
    assert routed.detected_input == InputType.VIDEO


def test_unsupported_via_pipeline_analyze() -> None:
    pipeline = MyUniSentimentPipeline.__new__(MyUniSentimentPipeline)
    routed = MyUniSentimentPipeline.analyze(
        pipeline,
        filename="doc.docx",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert routed.status == CapabilityStatus.VALIDATION_ERROR
    assert routed.analysis is None
