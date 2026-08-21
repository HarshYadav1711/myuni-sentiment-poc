#!/usr/bin/env python3
"""CLI entrypoint for MyUni Multimodal Sentiment Analysis POC."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow `python main.py ...` without installing the package.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.batch import BatchIngestor
from src.pipeline import MyUniSentimentPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MyUni sentiment POC — English text analysis and JSONL batch ingestion",
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
        "--user-id",
        dest="user_id",
        default=None,
        help="Optional user identifier for single-text mode (e.g. U007)",
    )
    parser.add_argument(
        "--activity-id",
        dest="activity_id",
        default=None,
        help="Optional activity identifier for single-text mode (e.g. ACT0042)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING)",
    )
    return parser


def _configure_stdio() -> None:
    # Prefer UTF-8 on Windows so social-media emoji text prints cleanly.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()

    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.batch_path and args.text:
        parser.error("Provide either a text argument or --batch, not both")

    if args.batch_path:
        return _run_batch(args.batch_path)

    if args.text is None:
        parser.error("Provide text to analyze, or use --batch PATH")

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


def _run_batch(batch_path: str) -> int:
    ingestor = BatchIngestor()
    try:
        result = ingestor.process_file(batch_path)
    except FileNotFoundError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result.model_dump_json_compatible(), indent=2, ensure_ascii=False))

    summary = result.summary
    # Non-zero when nothing useful was processed and there were problems.
    if summary.processed == 0 and (summary.invalid > 0 or summary.failed > 0):
        return 1
    if summary.failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
