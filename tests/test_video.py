"""Fast unit tests for Milestone 5 video analysis (mocked FFmpeg / modality analyzers)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyzers.video import VideoAnalyzer, _ocr_frame_indices
from src.config import VideoSamplingConfig
from src.media.ffmpeg_utils import VideoProbeInfo
from src.pipeline import MyUniSentimentPipeline
from src.schemas import ActivityInput, SentimentEvidence, SpeechAnalysisResult


def _ev(label: str = "neutral", score: float = 0.0, conf: float = 0.5) -> SentimentEvidence:
    return SentimentEvidence(
        label=label,  # type: ignore[arg-type]
        score=score,
        confidence=conf,
        probabilities={"negative": 0.2, "neutral": 0.6, "positive": 0.2},
        model="stub",
    )


def test_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        VideoAnalyzer.validate_media_path(tmp_path / "missing.mp4")


def test_sampling_config_reduces_fps_for_long_videos() -> None:
    cfg = VideoSamplingConfig(fps=1.0, max_frames=10)
    assert cfg.effective_fps(5) == 1.0
    assert abs(cfg.effective_fps(100) - 0.1) < 1e-9


def test_ocr_frame_index_selection() -> None:
    assert _ocr_frame_indices(3, 8) == {0, 1, 2}
    idxs = _ocr_frame_indices(10, 3)
    assert 0 in idxs and 9 in idxs
    assert len(idxs) <= 3


def test_partial_modality_failure_still_returns_structure(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    image = MagicMock()
    # First frame OK, second fails inside analyze_path path — we stub extract + analyze.
    from src.analyzers.image import ImageModalityEvidence

    image.analyze_path.side_effect = [
        ImageModalityEvidence(visual=_ev("positive", 0.4, 0.7), warnings=["OCR unavailable: x"]),
        RuntimeError("frame decode glitch"),
    ]
    image._visual = MagicMock()
    image._visual.analyze_image.return_value = _ev("neutral", 0.0, 0.5)

    audio = MagicMock()
    audio.analyze.side_effect = RuntimeError("no speech backend")

    analyzer = VideoAnalyzer(
        image_analyzer=image,
        audio_analyzer=audio,
        sampling=VideoSamplingConfig(fps=1.0, max_frames=5, max_ocr_frames=5),
        debug=True,
    )

    frame1 = tmp_path / "frame_00001.jpg"
    frame2 = tmp_path / "frame_00002.jpg"
    frame1.write_bytes(b"x")
    frame2.write_bytes(b"y")

    probe = VideoProbeInfo(duration_seconds=2.0, has_video=True, has_audio=True)

    with patch("src.analyzers.video.probe_video", return_value=probe), patch(
        "src.media.samplers.extract_frames_at_fps",
        return_value=[frame1, frame2],
    ), patch(
        "src.analyzers.visual.VisualSentimentAnalyzer.load_image",
        return_value=MagicMock(),
    ):
        # Force both frames through OCR path so analyze_path is used.
        with patch("src.analyzers.video._ocr_frame_indices", return_value={0, 1}):
            bundle = analyzer.analyze(video)

    assert bundle.visual is not None  # at least one frame succeeded
    assert bundle.speech is None
    assert any("Speech analysis failed" in w for w in bundle.warnings)
    assert any("frame[1] analysis failed" in w for w in bundle.warnings)
    assert bundle.diagnostics.frames_extracted == 2
    assert bundle.diagnostics.frames_analyzed == 1
    assert bundle.diagnostics.frame_debug is not None
    assert bundle.overall is not None


def test_output_structure_via_pipeline(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    pipeline = MyUniSentimentPipeline()
    fake_bundle = SimpleNamespace(
        visual=_ev("positive", 0.3, 0.6),
        ocr=None,
        ocr_text=None,
        speech=_ev("negative", -0.2, 0.55),
        transcript="this was bad",
        speech_result=SpeechAnalysisResult(
            transcript="this was bad",
            language="en",
            asr_model="base.en",
            sentiment=_ev("negative", -0.2, 0.55),
        ),
        diagnostics=SimpleNamespace(),
        warnings=["No speech detected (empty transcript)".replace("empty", "partial")],
        overall=_ev("neutral", 0.0, 0.5),
    )
    # Proper diagnostics object
    from src.schemas import VideoDiagnostics

    fake_bundle.diagnostics = VideoDiagnostics(
        duration_seconds=2.0,
        sampling_strategy="fixed_fps",
        sampling_fps=1.0,
        frames_extracted=2,
        frames_analyzed=2,
        frame_timestamps=[0.0, 1.0],
        extraction_seconds=0.1,
        processing_seconds=0.5,
        has_audio=True,
    )

    pipeline._video_analyzer.analyze = MagicMock(return_value=fake_bundle)  # type: ignore[method-assign]
    pipeline._text_analyzer.analyze = MagicMock(  # type: ignore[method-assign]
        return_value=_ev("positive", 0.8, 0.9),
    )
    pipeline._text_analyzer.validate_text = lambda t: str(t).strip()  # type: ignore[method-assign]

    activity = ActivityInput(
        activity_id="ACT-V1",
        user_id="U1",
        activity_type="video",
        text="Campus clip",
        media_path=str(video),
        created_at=datetime.now(timezone.utc),
    )
    result = pipeline.analyze_activity(activity)
    payload = result.model_dump_json_compatible()

    assert payload["activity_type"] == "video"
    assert payload["analysis"]["modalities"]["text"] is not None
    assert payload["analysis"]["modalities"]["visual"] is not None
    assert payload["analysis"]["modalities"]["speech"] is not None
    assert payload["analysis"]["transcript"] == "this was bad"
    assert payload["analysis"]["video"]["sampling_fps"] == 1.0
    assert payload["analysis"]["video"]["sampling_strategy"] == "fixed_fps"
    assert payload["analysis"]["video"]["frames_extracted"] == 2
    assert "frame_debug" not in payload["analysis"]["video"] or payload["analysis"]["video"]["frame_debug"] is None
    assert payload["analysis"]["overall"]["model"] == "poc-fusion"


def test_temp_cleanup(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    created: list[Path] = []

    def fake_mkdtemp(prefix=""):  # type: ignore[no-untyped-def]
        d = tmp_path / f"{prefix}dir"
        d.mkdir()
        created.append(d)
        return str(d)

    image = MagicMock()
    from src.analyzers.image import ImageModalityEvidence

    image.analyze_path.return_value = ImageModalityEvidence(visual=_ev(), warnings=[])
    image._visual = MagicMock()
    audio = MagicMock()
    audio.analyze.return_value = SpeechAnalysisResult(
        transcript=None,
        language="en",
        asr_model="base.en",
        warnings=["No speech detected (empty transcript)"],
    )

    analyzer = VideoAnalyzer(
        image_analyzer=image,
        audio_analyzer=audio,
        sampling=VideoSamplingConfig(fps=1.0, max_frames=2, max_ocr_frames=2),
        preserve_temp=False,
    )
    probe = VideoProbeInfo(duration_seconds=1.0, has_video=True, has_audio=True)
    frame = tmp_path / "f.jpg"
    frame.write_bytes(b"x")

    with patch("src.analyzers.video.tempfile.mkdtemp", side_effect=fake_mkdtemp), patch(
        "src.analyzers.video.probe_video",
        return_value=probe,
    ), patch(
        "src.media.samplers.extract_frames_at_fps",
        return_value=[frame],
    ), patch("src.analyzers.video._ocr_frame_indices", return_value={0}):
        analyzer.analyze(video)

    assert created
    assert not created[0].exists()
