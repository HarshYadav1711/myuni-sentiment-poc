#!/usr/bin/env python3
"""Compare fixed-FPS vs scene-keyframe sampling on the SAME video (experimental).

Does not claim either strategy is better. Reports frame counts, timings,
visual/overall labels, modality evidence, and whether the final prediction differs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyzers.video import VideoAnalyzer
from src.config import VideoSamplingConfig
from src.media.samplers import SceneSamplingConfig, build_frame_sampler
from src.pipeline import MyUniSentimentPipeline
from src.schemas import SentimentEvidence


def _evidence_summary(ev: Optional[SentimentEvidence]) -> Optional[dict[str, Any]]:
    if ev is None:
        return None
    return {
        "label": ev.label,
        "score": round(float(ev.score), 4),
        "confidence": round(float(ev.confidence), 4),
        "model": ev.model,
    }


def _run_strategy(
    pipeline: MyUniSentimentPipeline,
    video_path: Path,
    strategy: str,
    *,
    caption: Optional[str],
) -> dict[str, Any]:
    pipeline.video_analyzer.set_sampling_strategy(strategy)
    started = time.perf_counter()
    if caption:
        from datetime import datetime, timezone

        from src.schemas import ActivityInput

        activity = ActivityInput(
            activity_id=f"CMP-{strategy}",
            user_id="COMPARE",
            activity_type="video",
            text=caption,
            media_path=str(video_path),
            created_at=datetime.now(timezone.utc),
        )
        result = pipeline.analyze_activity(activity)
        analysis = result.analysis
        total = time.perf_counter() - started
        video_diag = analysis.video
        return {
            "strategy": strategy,
            "resolved_strategy": getattr(video_diag, "sampling_strategy", strategy) if video_diag else strategy,
            "frames_extracted": video_diag.frames_extracted if video_diag else None,
            "frames_analyzed": video_diag.frames_analyzed if video_diag else None,
            "sampling_fps": video_diag.sampling_fps if video_diag else None,
            "scene_count": video_diag.scene_count if video_diag else None,
            "extraction_seconds": video_diag.extraction_seconds if video_diag else None,
            "video_processing_seconds": video_diag.processing_seconds if video_diag else None,
            "total_analysis_seconds": round(total, 4),
            "visual_sentiment": _evidence_summary(analysis.modalities.visual),
            "overall_sentiment": _evidence_summary(analysis.overall),
            "modality_evidence": {
                "text": _evidence_summary(analysis.modalities.text),
                "visual": _evidence_summary(analysis.modalities.visual),
                "ocr": _evidence_summary(analysis.modalities.ocr),
                "speech": _evidence_summary(analysis.modalities.speech),
            },
            "warnings": list(analysis.warnings),
            "frame_timestamps": list(video_diag.frame_timestamps) if video_diag else [],
        }

    # Direct VideoAnalyzer path (no caption).
    analyzer: VideoAnalyzer = pipeline.video_analyzer
    analyzer.set_sampling_strategy(strategy)
    bundle = analyzer.analyze(video_path)
    total = time.perf_counter() - started
    return {
        "strategy": strategy,
        "resolved_strategy": bundle.diagnostics.sampling_strategy,
        "frames_extracted": bundle.diagnostics.frames_extracted,
        "frames_analyzed": bundle.diagnostics.frames_analyzed,
        "sampling_fps": bundle.diagnostics.sampling_fps,
        "scene_count": bundle.diagnostics.scene_count,
        "extraction_seconds": bundle.diagnostics.extraction_seconds,
        "video_processing_seconds": bundle.diagnostics.processing_seconds,
        "total_analysis_seconds": round(total, 4),
        "visual_sentiment": _evidence_summary(bundle.visual),
        "overall_sentiment": _evidence_summary(bundle.overall),
        "modality_evidence": {
            "text": None,
            "visual": _evidence_summary(bundle.visual),
            "ocr": _evidence_summary(bundle.ocr),
            "speech": _evidence_summary(bundle.speech),
        },
        "warnings": list(bundle.warnings),
        "frame_timestamps": list(bundle.diagnostics.frame_timestamps),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Experimental comparison of fixed_fps vs scene_keyframe sampling "
            "on one video. Does not declare a winner."
        ),
    )
    p.add_argument("video", type=Path, help="Path to the video file")
    p.add_argument("--caption", default=None, help="Optional caption (enables full pipeline fusion)")
    p.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Max frames for both strategies (default: 60)",
    )
    p.add_argument(
        "--frames-per-scene",
        type=int,
        default=1,
        help="Representative frames per detected scene (default: 1)",
    )
    p.add_argument(
        "--no-scene-fallback",
        action="store_true",
        help="Fail clearly if scene detection fails (default: fall back to fixed_fps)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON report path (default: stdout only)",
    )
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    video = args.video
    if not video.is_file():
        print(f"Video not found: {video}", file=sys.stderr)
        return 1

    sampling = VideoSamplingConfig(fps=1.0, max_frames=args.max_frames, max_ocr_frames=8)
    scene = SceneSamplingConfig(
        max_frames=args.max_frames,
        frames_per_scene=args.frames_per_scene,
        fallback_to_fixed_fps=not args.no_scene_fallback,
    )

    # Ensure factory configs are wired into a shared VideoAnalyzer.
    pipeline = MyUniSentimentPipeline(
        video_sampling=sampling,
        video_sampling_strategy="fixed_fps",
    )
    pipeline.video_analyzer.scene_sampling = scene

    # Validate both strategies resolve.
    build_frame_sampler("fixed_fps", sampling=sampling)
    build_frame_sampler("scene_keyframe", scene=scene)

    fixed = _run_strategy(pipeline, video, "fixed_fps", caption=args.caption)
    scene_run = _run_strategy(pipeline, video, "scene_keyframe", caption=args.caption)

    fixed_label = (fixed.get("overall_sentiment") or {}).get("label")
    scene_label = (scene_run.get("overall_sentiment") or {}).get("label")
    prediction_differs = fixed_label != scene_label

    report = {
        "video": str(video.resolve()),
        "purpose": (
            "Experimental comparison of sampling strategies on the same video. "
            "Neither strategy is declared better."
        ),
        "runs": {
            "fixed_fps": fixed,
            "scene_keyframe": scene_run,
        },
        "comparison": {
            "frame_count_fixed_fps": fixed.get("frames_extracted"),
            "frame_count_scene_keyframe": scene_run.get("frames_extracted"),
            "extraction_seconds_fixed_fps": fixed.get("extraction_seconds"),
            "extraction_seconds_scene_keyframe": scene_run.get("extraction_seconds"),
            "total_analysis_seconds_fixed_fps": fixed.get("total_analysis_seconds"),
            "total_analysis_seconds_scene_keyframe": scene_run.get("total_analysis_seconds"),
            "visual_sentiment_fixed_fps": fixed.get("visual_sentiment"),
            "visual_sentiment_scene_keyframe": scene_run.get("visual_sentiment"),
            "overall_sentiment_fixed_fps": fixed.get("overall_sentiment"),
            "overall_sentiment_scene_keyframe": scene_run.get("overall_sentiment"),
            "final_prediction_differs": prediction_differs,
        },
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\nWrote report to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
