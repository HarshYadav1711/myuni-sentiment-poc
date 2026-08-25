"""Multimodal client path unit tests (mocked analyzers; no model download)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyzers.image import ImageAnalyzer, ImageModalityEvidence
from src.config import DEFAULT_FUSION, DEFAULT_VISUAL_MODEL, MAX_VIDEO_FRAMES
from src.fusion import aggregate_frame_visual_scores, fuse_modalities
from src.pipeline import MyUniSentimentPipeline
from src.routing.input_router import CapabilityStatus, InputType
from src.schemas import SentimentEvidence, VideoDiagnostics


def _ev(label: str, *, p: float = 0.7) -> SentimentEvidence:
    probs = {"positive": 0.1, "neutral": 0.1, "negative": 0.1}
    probs[label] = p
    rem = (1.0 - p) / 2.0
    for k in probs:
        if k != label:
            probs[k] = rem
    return SentimentEvidence(
        label=label,  # type: ignore[arg-type]
        score=probs["positive"] - probs["negative"],
        confidence=p,
        probabilities=probs,
        model=DEFAULT_VISUAL_MODEL,
        details={"raw_similarities": dict(probs)},
    )


def test_image_analyzer_returns_visual_distribution(tmp_path: Path) -> None:
    path = tmp_path / "pos.png"
    img = Image.new("RGB", (64, 64), color=(255, 220, 80))
    draw = ImageDraw.Draw(img)
    draw.ellipse((16, 16, 48, 48), fill=(255, 200, 0))
    img.save(path)

    visual = MagicMock()
    visual.analyze_image.return_value = _ev("positive")
    ocr = MagicMock()
    ocr.extract.return_value = MagicMock(available=False, text=None, warning=None)
    analyzer = ImageAnalyzer(visual_analyzer=visual, ocr_extractor=ocr)
    evidence = analyzer.analyze_path(path)
    assert isinstance(evidence, ImageModalityEvidence)
    assert evidence.visual.probabilities is not None
    assert set(evidence.visual.probabilities) == {"positive", "neutral", "negative"}
    assert evidence.visual.label == "positive"


def test_visual_frame_aggregation_and_cap_constant() -> None:
    assert MAX_VIDEO_FRAMES == 12
    frames = [_ev("positive"), _ev("neutral", p=0.5), _ev("negative")]
    agg = aggregate_frame_visual_scores(frames, config=DEFAULT_FUSION)
    assert agg is not None
    assert agg.probabilities is not None


def test_video_fusion_equal_weight_when_both_present() -> None:
    visual = _ev("positive")
    speech = _ev("negative")
    outcome = fuse_modalities({"visual": visual, "speech": speech}, config=DEFAULT_FUSION)
    assert set(outcome.diagnostics.contributing_modalities) == {"visual", "speech"}
    assert outcome.overall.probabilities is not None


def test_video_fusion_single_modality_explicit() -> None:
    visual = _ev("positive")
    outcome = fuse_modalities({"visual": visual, "speech": None}, config=DEFAULT_FUSION)
    assert outcome.diagnostics.contributing_modalities == ["visual"]


def test_client_video_wrap_uses_visual_and_speech(tmp_path: Path) -> None:
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"fake")
    bundle = MagicMock()
    bundle.visual = _ev("positive")
    bundle.speech = _ev("neutral", p=0.55)
    bundle.warnings = []
    bundle.transcript = "hello campus"
    bundle.diagnostics = VideoDiagnostics(
        frames_extracted=5,
        frames_analyzed=5,
        sampling_fps=1.0,
        sampling_strategy="fixed_fps",
    )
    pipeline = MyUniSentimentPipeline.__new__(MyUniSentimentPipeline)
    pipeline._fusion_config = DEFAULT_FUSION
    pipeline._video_analyzer = MagicMock()
    pipeline._video_analyzer.analyze.return_value = bundle
    pipeline._text_analyzer = MagicMock(model_name="text")
    pipeline._audio_analyzer = MagicMock(whisper_model_name="asr")
    pipeline._video_analyzer.frame_sampler = MagicMock(name="fixed_fps")
    pipeline._video_analyzer.sampling = MagicMock(fps=1.0, max_frames=12, max_ocr_frames=8)

    result = MyUniSentimentPipeline._client_analyze_video(
        pipeline,
        str(vid),
        user_id="DEMO-USER",
        activity_id="ACT-V1",
    )
    assert result.analysis.modalities.visual is not None
    assert result.analysis.modalities.speech is not None
    assert result.analysis.transcript == "hello campus"
    assert result.analysis.fusion is not None
    assert "visual" in result.analysis.fusion.contributing_modalities
    assert "speech" in result.analysis.fusion.contributing_modalities

    routed = MyUniSentimentPipeline.analyze(
        pipeline,
        mime_type="video/mp4",
        filename="clip.mp4",
        media_path=str(vid),
    )
    # analyze will call _client_analyze_video again — stub it
    pipeline._client_analyze_video = MagicMock(return_value=result)  # type: ignore[method-assign]
    routed = MyUniSentimentPipeline.analyze(
        pipeline,
        mime_type="video/mp4",
        filename="clip.mp4",
        media_path=str(vid),
    )
    assert routed.status == CapabilityStatus.OK
    assert routed.detected_input == InputType.VIDEO
