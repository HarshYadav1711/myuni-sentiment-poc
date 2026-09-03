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

    from src.config import TemporalReasonerConfig

    pipeline = MyUniSentimentPipeline(
        temporal_reasoner_config=TemporalReasonerConfig(enabled=False),
    )
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
    # Temporal context is parallel; must not replace fusion fields.
    assert result.analysis.temporal_context is not None
    assert result.analysis.temporal_context.window_seconds == 5.0
    assert len(result.analysis.temporal_context.windows) >= 1
    assert result.analysis.temporal_context.features.trajectory is not None
    # Synthetic clip uses anullsrc — expect no speech evidence.
    cov = result.analysis.temporal_context.features.evidence_coverage
    assert cov.speech_coverage == 0.0 or not any(
        w.speech_segments for w in result.analysis.temporal_context.windows
    )


@pytest.mark.integration
def test_demo_video_temporal_context_report() -> None:
    """Optional: report actual temporal_context from demo_assets/demo_video.mp4."""
    demo = ROOT / "demo_assets" / "demo_video.mp4"
    if not demo.is_file():
        pytest.skip("demo_assets/demo_video.mp4 not present")

    from src.config import TemporalReasonerConfig

    pipeline = MyUniSentimentPipeline(
        temporal_reasoner_config=TemporalReasonerConfig(enabled=False),
    )
    activity = ActivityInput(
        activity_id="ACT-DEMO-TEMPORAL",
        user_id="U-DEMO",
        activity_type="video",
        media_path=str(demo),
        created_at=datetime.now(timezone.utc),
    )
    result = pipeline.analyze_activity(activity)
    assert result.analysis.overall is not None
    assert result.analysis.temporal_context is not None

    tc = result.analysis.temporal_context
    feats = tc.features
    # Honest reporting — do not assert interesting trajectory.
    print("\n=== DEMO VIDEO TEMPORAL CONTEXT ===")
    print(f"duration_seconds={tc.duration_seconds}")
    print(f"window_seconds={tc.window_seconds}")
    print(f"events_total={tc.events_total} windows={len(tc.windows)}")
    print(f"trajectory={feats.trajectory}")
    print(f"negative_persistence={feats.negative_persistence}")
    print(f"longest_negative_run={feats.longest_negative_run}")
    print(f"strongest_negative_window={feats.strongest_negative_window}")
    print(f"sudden_negative_change={feats.sudden_negative_change}")
    print(f"cross_modal_agreement={feats.cross_modal_agreement}")
    print(f"cross_modal_conflicts={len(feats.cross_modal_conflicts)}")
    print(f"evidence_coverage={feats.evidence_coverage}")
    print(f"overall_sentiment={result.analysis.overall.label} score={result.analysis.overall.score}")
    print(f"transcript={result.analysis.transcript!r}")
    print(f"speech_coverage={feats.evidence_coverage.speech_coverage}")
    print("=== END DEMO TEMPORAL ===\n")

    # Existing sentiment fields remain.
    assert result.analysis.modalities is not None
    assert result.analysis.fusion is not None
