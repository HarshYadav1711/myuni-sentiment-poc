#!/usr/bin/env python3
"""CLI entrypoint for MyUni Multimodal Sentiment Analysis POC."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `python main.py ...` without installing the package.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.logging_config import configure_logging
from src.env_check import build_health_report
from src.pipeline import MyUniSentimentPipeline
from src.schemas import ActivityInput
from src.storage.repository import SentimentRepository
from src.storage.service import PersistentBatchRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MyUni sentiment POC — analysis, batch, SQLite persistence",
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="English text to analyze (single-activity CLI mode)",
    )
    parser.add_argument(
        "--batch",
        dest="batch_path",
        default=None,
        help="Path to a JSONL file of MyUni activities",
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        default=None,
        help="SQLite database path (with --batch persists results; with --daily-scores queries)",
    )
    parser.add_argument(
        "--daily-scores",
        action="store_true",
        help="Print POC daily user sentiment aggregates from SQLite (--db required)",
    )
    parser.add_argument(
        "--date",
        dest="score_date",
        default=None,
        help="Filter daily scores by UTC date YYYY-MM-DD",
    )
    parser.add_argument(
        "--video",
        dest="video_path",
        default=None,
        help="Path to a local video file to analyze end-to-end",
    )
    parser.add_argument(
        "--caption",
        dest="caption",
        default=None,
        help="Optional caption text for --video mode",
    )
    parser.add_argument(
        "--video-debug",
        action="store_true",
        help="Include per-frame debug rows in video diagnostics",
    )
    parser.add_argument(
        "--sampling-strategy",
        dest="sampling_strategy",
        default="fixed_fps",
        choices=["fixed_fps", "scene_keyframe"],
        help="Video frame sampling strategy (default: fixed_fps baseline)",
    )
    parser.add_argument(
        "--user-id",
        dest="user_id",
        default=None,
        help="Optional user identifier (e.g. U007)",
    )
    parser.add_argument(
        "--activity-id",
        dest="activity_id",
        default=None,
        help="Optional activity identifier (e.g. ACT0042)",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Print environment / dependency health report (JSON) and exit",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING)",
    )
    return parser


def _configure_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()

    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    if args.health:
        print(
            json.dumps(
                build_health_report(db_path=args.db_path),
                indent=2,
                ensure_ascii=False,
            ),
        )
        return 0

    if args.daily_scores:
        if not args.db_path:
            parser.error("--daily-scores requires --db PATH")
        return _run_daily_scores(
            args.db_path,
            score_date=args.score_date,
            user_id=args.user_id,
        )

    modes = [bool(args.batch_path), bool(args.video_path), bool(args.text)]
    if sum(modes) > 1:
        parser.error("Provide only one of: text argument, --batch, or --video")

    if args.batch_path:
        return _run_batch(args.batch_path, db_path=args.db_path)

    if args.video_path:
        return _run_video(
            args.video_path,
            caption=args.caption,
            user_id=args.user_id,
            activity_id=args.activity_id,
            video_debug=args.video_debug,
            sampling_strategy=args.sampling_strategy,
        )

    if args.text is None and not args.health:
        parser.error(
            "Provide text, --batch PATH, --video PATH, --health, or --daily-scores --db PATH",
        )

    return _run_single_text(args.text, args.user_id, args.activity_id)


def _run_single_text(
    text: str,
    user_id: str | None,
    activity_id: str | None,
) -> int:
    pipeline = MyUniSentimentPipeline()
    try:
        result = pipeline.analyze_text(
            text,
            user_id=user_id,
            activity_id=activity_id,
        )
    except ValueError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result.model_dump_json_compatible(), indent=2, ensure_ascii=False))
    return 0


def _run_video(
    video_path: str,
    *,
    caption: str | None,
    user_id: str | None,
    activity_id: str | None,
    video_debug: bool,
    sampling_strategy: str = "fixed_fps",
) -> int:
    pipeline = MyUniSentimentPipeline(
        video_debug=video_debug,
        video_sampling_strategy=sampling_strategy,
    )
    activity = ActivityInput(
        activity_id=activity_id or "ACT-VIDEO",
        user_id=user_id or "U-VIDEO",
        activity_type="video",
        text=caption,
        media_path=video_path,
        created_at=datetime.now(timezone.utc),
    )
    try:
        result = pipeline.analyze_activity(activity)
    except FileNotFoundError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"Video analysis error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.model_dump_json_compatible(), indent=2, ensure_ascii=False))
    return 0


def _run_batch(batch_path: str, *, db_path: str | None) -> int:
    try:
        if db_path:
            result = PersistentBatchRunner(db_path).process_file(batch_path)
        else:
            result = BatchIngestor().process_file(batch_path)
    except FileNotFoundError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2

    summary = result.summary
    concise = {
        "batch_id": result.batch_id,
        "source": result.source,
        "summary": summary.model_dump(),
        "db": db_path,
    }
    if db_path:
        print(json.dumps(concise, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result.model_dump_json_compatible(), indent=2, ensure_ascii=False))

    if summary.processed == 0 and (summary.invalid > 0 or summary.failed > 0):
        return 1
    if summary.failed > 0:
        return 1
    return 0


def _run_daily_scores(
    db_path: str,
    *,
    score_date: str | None,
    user_id: str | None,
) -> int:
    repo = SentimentRepository(db_path)
    rows = repo.get_daily_scores(score_date=score_date, user_id=user_id)
    payload = {
        "note": "POC daily aggregates — NOT the client business score",
        "filter": {"date": score_date, "user_id": user_id},
        "count": len(rows),
        "scores": [r.model_dump_json_compatible() for r in rows],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
