"""Fast unit tests for Milestone 4 speech/audio branch (no Whisper download)."""

from __future__ import annotations

import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyzers.audio import AudioAnalyzer
from src.media.ffmpeg_utils import (
    FFmpegError,
    FFmpegNotFoundError,
    extract_audio_wav,
    find_ffmpeg,
)
from src.schemas import SentimentEvidence, SpeechAnalysisResult


def _write_silent_wav(path: Path, *, duration_s: float = 0.2, rate: int = 16000) -> Path:
    n_frames = int(duration_s * rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return path


def test_validate_media_path_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Media file not found"):
        AudioAnalyzer.validate_media_path(tmp_path / "missing.mp4")


def test_find_ffmpeg_missing_raises_actionable_error() -> None:
    with patch("src.media.ffmpeg_utils.shutil.which", return_value=None):
        with pytest.raises(FFmpegNotFoundError, match="FFmpeg executable not found"):
            find_ffmpeg()


def test_find_ffmpeg_invalid_configured_path(tmp_path: Path) -> None:
    with pytest.raises(FFmpegNotFoundError, match="Configured FFmpeg path"):
        find_ffmpeg(str(tmp_path / "no_ffmpeg.exe"))


def test_extract_audio_missing_source(tmp_path: Path) -> None:
    with patch("src.media.ffmpeg_utils.find_ffmpeg", return_value="ffmpeg"):
        with pytest.raises(FileNotFoundError):
            extract_audio_wav(tmp_path / "gone.mp4", tmp_path / "out.wav")


def test_extract_audio_ffmpeg_nonzero_exit(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")
    out = tmp_path / "out.wav"

    completed = MagicMock()
    completed.returncode = 1
    completed.stderr = "Error: Invalid data found when processing input"

    with patch("src.media.ffmpeg_utils.find_ffmpeg", return_value="ffmpeg"), patch(
        "src.media.ffmpeg_utils.subprocess.run",
        return_value=completed,
    ):
        with pytest.raises(FFmpegError, match="FFmpeg failed"):
            extract_audio_wav(media, out)


def test_extract_audio_empty_output(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")
    out = tmp_path / "out.wav"

    completed = MagicMock()
    completed.returncode = 0
    completed.stderr = ""

    def _fake_run(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        out.write_bytes(b"")  # empty file
        return completed

    with patch("src.media.ffmpeg_utils.find_ffmpeg", return_value="ffmpeg"), patch(
        "src.media.ffmpeg_utils.subprocess.run",
        side_effect=_fake_run,
    ):
        with pytest.raises(FFmpegError, match="no usable audio"):
            extract_audio_wav(media, out)


def test_empty_transcript_behavior(tmp_path: Path) -> None:
    media = tmp_path / "silent.wav"
    _write_silent_wav(media)
    out_wav = tmp_path / "norm.wav"

    text = MagicMock()
    analyzer = AudioAnalyzer(text_analyzer=text)
    analyzer._whisper_model = MagicMock()

    empty_segments: list[object] = []
    info = SimpleNamespace(language="en", duration=0.2)
    analyzer._whisper_model.transcribe.return_value = (iter(empty_segments), info)

    def _fake_extract(src, dest, **_kwargs):  # type: ignore[no-untyped-def]
        _write_silent_wav(Path(dest))
        return Path(dest)

    with patch("src.analyzers.audio.extract_audio_wav", side_effect=_fake_extract):
        result = analyzer.analyze(media)

    assert isinstance(result, SpeechAnalysisResult)
    assert result.transcript is None
    assert result.sentiment is None
    assert result.segments == []
    assert any("No speech detected" in w for w in result.warnings)
    assert result.asr_model == "base.en"
    assert result.language == "en"
    assert result.transcription_seconds is not None
    text.analyze.assert_not_called()


def test_standardized_speech_result_shape_with_transcript(tmp_path: Path) -> None:
    media = tmp_path / "talk.wav"
    _write_silent_wav(media)

    text = MagicMock()
    text.analyze.return_value = SentimentEvidence(
        label="positive",
        score=0.7,
        confidence=0.85,
        probabilities={"negative": 0.05, "neutral": 0.1, "positive": 0.85},
        model="stub-text",
        details={"device": "cpu"},
    )
    analyzer = AudioAnalyzer(whisper_model="base.en", text_analyzer=text)
    analyzer._whisper_model = MagicMock()

    seg = SimpleNamespace(start=0.0, end=1.2, text="  I loved the workshop.  ")
    info = SimpleNamespace(language="en", duration=1.5)
    analyzer._whisper_model.transcribe.return_value = (iter([seg]), info)

    def _fake_extract(src, dest, **_kwargs):  # type: ignore[no-untyped-def]
        _write_silent_wav(Path(dest))
        return Path(dest)

    with patch("src.analyzers.audio.extract_audio_wav", side_effect=_fake_extract):
        result = analyzer.analyze(media)

    payload = result.model_dump_json_compatible()
    assert payload["transcript"] == "I loved the workshop."
    assert payload["language"] == "en"
    assert payload["asr_model"] == "base.en"
    assert len(payload["segments"]) == 1
    assert payload["segments"][0]["start"] == 0.0
    assert payload["segments"][0]["end"] == 1.2
    assert payload["sentiment"]["label"] == "positive"
    assert payload["sentiment"]["details"]["source"] == "speech"
    assert payload["transcription_seconds"] is not None
    assert payload["audio_duration_seconds"] == 1.5
    assert isinstance(payload["warnings"], list)


def test_ffmpeg_not_found_surfaces_from_analyze(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake-video")
    analyzer = AudioAnalyzer()

    with patch(
        "src.analyzers.audio.extract_audio_wav",
        side_effect=FFmpegNotFoundError("FFmpeg executable not found on PATH"),
    ):
        with pytest.raises(FFmpegNotFoundError, match="FFmpeg executable not found"):
            analyzer.analyze(media)


def test_temp_files_cleaned_after_analyze(tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    _write_silent_wav(media)
    created_dirs: list[Path] = []

    class TrackingTempDir:
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            self._dir = tmp_path / f"asr_tmp_{len(created_dirs)}"
            self._dir.mkdir()
            created_dirs.append(self._dir)
            self.name = str(self._dir)

        def cleanup(self) -> None:
            for child in self._dir.iterdir():
                child.unlink(missing_ok=True)
            self._dir.rmdir()

    analyzer = AudioAnalyzer(text_analyzer=MagicMock())
    analyzer._whisper_model = MagicMock()
    analyzer._whisper_model.transcribe.return_value = (
        iter([]),
        SimpleNamespace(language="en", duration=0.1),
    )

    def _fake_extract(src, dest, **_kwargs):  # type: ignore[no-untyped-def]
        _write_silent_wav(Path(dest))
        return Path(dest)

    with patch("src.analyzers.audio.tempfile.TemporaryDirectory", TrackingTempDir), patch(
        "src.analyzers.audio.extract_audio_wav",
        side_effect=_fake_extract,
    ):
        analyzer.analyze(media)

    assert created_dirs
    assert not created_dirs[0].exists()
