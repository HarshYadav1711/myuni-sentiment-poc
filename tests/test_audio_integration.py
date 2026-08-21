"""Optional ASR integration tests (downloads Whisper + needs FFmpeg).

Skipped by default. Enable with:

    set MYUNI_RUN_ASR_INTEGRATION=1
    pytest -m asr_integration
"""

from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyzers.audio import AudioAnalyzer
from src.media.ffmpeg_utils import FFmpegNotFoundError, find_ffmpeg


def _ffmpeg_available() -> bool:
    try:
        find_ffmpeg()
        return True
    except FFmpegNotFoundError:
        return False


pytestmark = [
    pytest.mark.asr_integration,
    pytest.mark.skipif(
        os.environ.get("MYUNI_RUN_ASR_INTEGRATION", "").strip() not in {"1", "true", "yes"},
        reason="Set MYUNI_RUN_ASR_INTEGRATION=1 to run optional ASR integration tests",
    ),
    pytest.mark.skipif(not _ffmpeg_available(), reason="FFmpeg not installed"),
]


def _write_silent_wav(path: Path) -> None:
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)


@pytest.mark.integration
def test_real_whisper_on_silent_wav(tmp_path: Path) -> None:
    media = tmp_path / "silent.wav"
    _write_silent_wav(media)

    result = AudioAnalyzer(whisper_model="base.en", device="cpu").analyze(media)
    assert result.asr_model == "base.en"
    # Silent audio should not fabricate speech.
    assert result.transcript in (None, "")
    assert result.sentiment is None
    assert isinstance(result.warnings, list)
