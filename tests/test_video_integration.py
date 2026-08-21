"""Optional video integration smoke (needs FFmpeg; downloads ML models).

Enable with:

    set MYUNI_RUN_VIDEO_INTEGRATION=1
    pytest -m video_integration
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.media.ffmpeg_utils import FFmpegNotFoundError, find_ffmpeg
from src.pipeline import MyUniSentimentPipeline
from src.schemas import ActivityInput


def _ffmpeg_ok() -> bool:
    try:
        find_ffmpeg()
        return True
    except FFmpegNotFoundError:
        return False


pytestmark = [
    pytest.mark.video_integration,
    pytest.mark.skipif(
        os.environ.get("MYUNI_RUN_VIDEO_INTEGRATION", "").strip() not in {"1", "true", "yes"},
        reason="Set MYUNI_RUN_VIDEO_INTEGRATION=1 to run optional video integration",
    ),
    pytest.mark.skipif(not _ffmpeg_ok(), reason="FFmpeg not installed"),
]


def _generate_short_video(path: Path) -> None:
    exe = find_ffmpeg()
    cmd = [
        exe,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=320x240:d=2",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=16000:cl=mono",
        "-t",
        "2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        pytest.skip(f"Could not generate sample video: {completed.stderr[-400:]}")


@pytest.mark.integration
def test_short_synthetic_video_smoke(tmp_path: Path) -> None:
    video = tmp_path / "synthetic_2s.mp4"
    _generate_short_video(video)

    pipeline = MyUniSentimentPipeline()
    activity = ActivityInput(
        activity_id="ACT-VID-SMOKE",
        user_id="U-SMOKE",
        activity_type="video",
        text="Quiet campus clip",
        media_path=str(video),
        created_at=datetime.now(timezone.utc),
    )
    result = pipeline.analyze_activity(activity)
    assert result.activity_type == "video"
    assert result.analysis.video is not None
    assert result.analysis.video.frames_extracted >= 1
    assert result.analysis.modalities.visual is not None or result.analysis.warnings
    assert result.analysis.overall.model == "poc-fusion"
