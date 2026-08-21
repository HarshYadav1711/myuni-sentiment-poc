"""FFmpeg helpers for audio extraction / normalization."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

FFMPEG_MISSING_MESSAGE = (
    "FFmpeg executable not found on PATH. "
    "Install FFmpeg for Windows (e.g. https://www.gyan.dev/ffmpeg/builds/ "
    "or `winget install Gyan.FFmpeg`) and reopen the terminal so `ffmpeg` is available. "
    "Verify with: ffmpeg -version"
)


class FFmpegNotFoundError(RuntimeError):
    """Raised when the FFmpeg binary cannot be located."""


class FFmpegError(RuntimeError):
    """Raised when an FFmpeg invocation fails."""


def find_ffmpeg(ffmpeg_path: Optional[str] = None) -> str:
    """Return the FFmpeg executable path or raise an actionable error."""
    if ffmpeg_path:
        candidate = Path(ffmpeg_path)
        if candidate.is_file():
            return str(candidate)
        raise FFmpegNotFoundError(
            f"Configured FFmpeg path does not exist: {ffmpeg_path}. {FFMPEG_MISSING_MESSAGE}",
        )

    found = shutil.which("ffmpeg")
    if not found:
        raise FFmpegNotFoundError(FFMPEG_MISSING_MESSAGE)
    return found


def extract_audio_wav(
    media_path: PathLike,
    output_wav: PathLike,
    *,
    ffmpeg_path: Optional[str] = None,
    sample_rate: int = 16000,
    timeout_seconds: int = 300,
) -> Path:
    """Extract / normalize media audio to mono 16 kHz PCM WAV via FFmpeg.

    Works for video containers and common audio files. Overwrites ``output_wav``.
    """
    source = Path(media_path)
    if not source.is_file():
        raise FileNotFoundError(f"Media file not found: {source}")

    dest = Path(output_wav)
    dest.parent.mkdir(parents=True, exist_ok=True)

    exe = find_ffmpeg(ffmpeg_path)
    cmd = [
        exe,
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    logger.info("Extracting audio with FFmpeg: %s -> %s", source, dest)
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(
            f"FFmpeg timed out after {timeout_seconds}s while processing {source}",
        ) from exc
    except OSError as exc:
        raise FFmpegError(f"Failed to launch FFmpeg: {exc}") from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        # Keep the message actionable but bounded.
        detail = stderr[-800:] if stderr else "no stderr"
        raise FFmpegError(
            f"FFmpeg failed (exit {completed.returncode}) for {source}: {detail}",
        )

    if not dest.is_file() or dest.stat().st_size == 0:
        raise FFmpegError(
            f"FFmpeg produced no usable audio output for {source} "
            "(file may have no audio stream)",
        )

    return dest
