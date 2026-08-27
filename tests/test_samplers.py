"""Unit tests for video frame sampling strategies (Milestone 9)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import VideoSamplingConfig
from src.media.samplers import (
    FixedFPSSampler,
    FrameSample,
    SampledFrames,
    SceneKeyframeSampler,
    SceneSamplingConfig,
    build_frame_sampler,
    keyframe_timestamps_from_scenes,
)


def test_build_frame_sampler_fixed_and_scene() -> None:
    fixed = build_frame_sampler("fixed_fps")
    scene = build_frame_sampler("scene_keyframe")
    assert isinstance(fixed, FixedFPSSampler)
    assert isinstance(scene, SceneKeyframeSampler)
    assert fixed.name == "fixed_fps"
    assert scene.name == "scene_keyframe"


def test_build_frame_sampler_aliases() -> None:
    assert build_frame_sampler("fps").name == "fixed_fps"
    assert build_frame_sampler("scene").name == "scene_keyframe"
    assert build_frame_sampler("keyframe").name == "scene_keyframe"


def test_build_frame_sampler_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown video sampling strategy"):
        build_frame_sampler("magic_frames")


def test_fixed_fps_sampler_uses_ffmpeg_and_timestamps(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    out = tmp_path / "frames"
    f1 = out / "frame_00001.jpg"
    f2 = out / "frame_00002.jpg"

    def fake_extract(media, output_dir, *, fps, ffmpeg_path=None, **_kw):  # type: ignore[no-untyped-def]
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        f1.parent.mkdir(parents=True, exist_ok=True)
        f1.write_bytes(b"a")
        f2.write_bytes(b"b")
        return [f1, f2]

    sampler = FixedFPSSampler(VideoSamplingConfig(fps=1.0, max_frames=10))
    with patch("src.media.samplers.extract_frames_at_fps", side_effect=fake_extract):
        result = sampler.sample(video, out, duration_seconds=2.0)

    assert result.strategy == "fixed_fps"
    assert result.sampling_fps == 1.0
    assert len(result.frames) == 2
    assert result.timestamps == [0.0, 1.0]
    assert result.extraction_seconds >= 0.0


def test_fixed_fps_truncates_to_max_frames(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    out = tmp_path / "frames"
    paths = []
    for i in range(5):
        p = out / f"frame_{i:05d}.jpg"
        paths.append(p)

    sampler = FixedFPSSampler(VideoSamplingConfig(fps=1.0, max_frames=2))
    with patch("src.media.samplers.extract_frames_at_fps", return_value=paths):
        # Pretend files exist for FrameSample paths (no I/O required beyond list).
        result = sampler.sample(video, out, duration_seconds=5.0)

    assert len(result.frames) == 2
    assert any("Truncated" in w for w in result.warnings)


def test_scene_sampler_selects_mid_timestamps_and_caps(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    out = tmp_path / "frames"
    extracted: list[float] = []

    def fake_extract(media, output_path, *, timestamp_seconds, ffmpeg_path=None):  # type: ignore[no-untyped-def]
        extracted.append(timestamp_seconds)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"f")
        return Path(output_path)

    sampler = SceneKeyframeSampler(
        SceneSamplingConfig(max_frames=2, frames_per_scene=1, fallback_to_fixed_fps=False),
    )
    with patch.object(
        sampler,
        "_detect_keyframe_timestamps",
        return_value=[1.0, 3.0, 5.0],
    ), patch(
        "src.media.samplers.extract_frame_at_timestamp",
        side_effect=fake_extract,
    ):
        sampler._last_scene_count = 3
        result = sampler.sample(video, out, duration_seconds=6.0)

    assert result.strategy == "scene_keyframe"
    assert len(result.frames) == 2
    assert result.scene_count == 3
    assert any("Truncated" in w for w in result.warnings)
    assert result.sampling_fps is None
    assert len(extracted) == 2


def test_scene_sampler_fallback_when_detection_fails(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    out = tmp_path / "frames"
    f1 = out / "frame_00001.jpg"

    def fake_fps_extract(media, output_dir, *, fps, ffmpeg_path=None, **_kw):  # type: ignore[no-untyped-def]
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        f1.write_bytes(b"a")
        return [f1]

    sampler = SceneKeyframeSampler(
        SceneSamplingConfig(max_frames=5, fallback_to_fixed_fps=True),
    )
    with patch.object(
        sampler,
        "_detect_keyframe_timestamps",
        side_effect=RuntimeError("boom"),
    ), patch(
        "src.media.samplers.extract_frames_at_fps",
        side_effect=fake_fps_extract,
    ):
        result = sampler.sample(video, out, duration_seconds=1.0)

    assert result.strategy == "scene_keyframe_fallback_fixed_fps"
    assert len(result.frames) == 1
    assert any("Scene detection failed" in w for w in result.warnings)
    assert any("Falling back" in w for w in result.warnings)


def test_scene_sampler_fails_clearly_without_fallback(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    out = tmp_path / "frames"
    sampler = SceneKeyframeSampler(
        SceneSamplingConfig(fallback_to_fixed_fps=False),
    )
    with patch.object(
        sampler,
        "_detect_keyframe_timestamps",
        side_effect=ImportError("no scenedetect"),
    ):
        with pytest.raises(RuntimeError, match="Scene detection failed"):
            sampler.sample(video, out, duration_seconds=1.0)


def test_scene_midpoint_from_scene_bounds() -> None:
    """Representative frame = mid of each (start, end) when frames_per_scene=1."""

    class _TC:
        def __init__(self, seconds: float) -> None:
            self._s = seconds

        def get_seconds(self) -> float:
            return self._s

    scenes = [(_TC(0.0), _TC(2.0)), (_TC(2.0), _TC(6.0))]
    assert keyframe_timestamps_from_scenes(scenes, frames_per_scene=1) == [1.0, 4.0]
    spaced = keyframe_timestamps_from_scenes(scenes[:1], frames_per_scene=3)
    assert spaced == [0.0, 1.0, 2.0]


def test_video_analyzer_uses_injected_sampler(tmp_path: Path) -> None:
    from src.analyzers.image import ImageModalityEvidence
    from src.analyzers.video import VideoAnalyzer
    from src.media.ffmpeg_utils import VideoProbeInfo
    from src.schemas import SentimentEvidence, SpeechAnalysisResult

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    frame = tmp_path / "f.jpg"
    frame.write_bytes(b"x")

    class StubSampler:
        name = "stub"

        def sample(self, media_path, output_dir, *, duration_seconds, ffmpeg_path=None):
            return SampledFrames(
                frames=[FrameSample(path=frame, timestamp_seconds=0.5, index=0)],
                strategy="stub",
                extraction_seconds=0.01,
                sampling_fps=None,
                scene_count=1,
            )

    image = MagicMock()
    image.analyze_path.return_value = ImageModalityEvidence(
        visual=SentimentEvidence(
            label="neutral",
            score=0.0,
            confidence=0.5,
            probabilities={"negative": 0.2, "neutral": 0.6, "positive": 0.2},
            model="stub",
        ),
        warnings=[],
    )
    image._visual = MagicMock()
    image._visual.analyze_images.return_value = [
        SentimentEvidence(
            label="neutral",
            score=0.0,
            confidence=0.5,
            probabilities={"negative": 0.2, "neutral": 0.6, "positive": 0.2},
            model="stub",
        ),
    ]
    image.extract_ocr_evidence.return_value = (None, None, [])
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
        frame_sampler=StubSampler(),  # type: ignore[arg-type]
    )
    probe = VideoProbeInfo(duration_seconds=2.0, has_video=True, has_audio=True)
    with patch("src.analyzers.video.probe_video", return_value=probe), patch(
        "src.analyzers.video._ocr_frame_indices",
        return_value={0},
    ), patch(
        "src.analyzers.visual.VisualSentimentAnalyzer.load_image",
        return_value=MagicMock(),
    ):
        bundle = analyzer.analyze(video)

    assert bundle.diagnostics.sampling_strategy == "stub"
    assert bundle.diagnostics.frames_extracted == 1
    assert bundle.diagnostics.frame_timestamps == [0.5]
    assert bundle.diagnostics.extraction_seconds == 0.01
