#!/usr/bin/env python3
"""CLI for MyUni POC benchmark evaluation (no dataset files bundled)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.common import format_console_summary, write_results
from evaluation.image.mvsa import evaluate_mvsa_examples, load_mvsa_jsonl
from evaluation.text.tweeteval import (
    evaluate_text_examples,
    load_tweeteval_hf,
    load_tweeteval_jsonl,
)
from evaluation.video.mosi import evaluate_mosi_examples, load_mosi_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate MyUni POC predictors against prepared benchmark indexes",
    )
    parser.add_argument(
        "modality",
        choices=["text", "image", "video"],
        help="Which evaluation adapter to run",
    )
    parser.add_argument(
        "--data",
        help="Path to prepared local JSONL index (required unless --tweeteval-hf)",
    )
    parser.add_argument(
        "--tweeteval-hf",
        action="store_true",
        help="Load TweetEval sentiment from Hugging Face datasets (text only; network)",
    )
    parser.add_argument("--split", default="test", help="Split name for reporting / HF load")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N examples")
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for metrics.json + predictions.csv",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use deterministic stub predictor (no model download)",
    )
    return parser


def _stub_predict_text(payload: dict) -> dict:
    text = (payload.get("text") or "").lower()
    if any(w in text for w in ("love", "great", "good", "amazing")):
        return {"pred_label": "positive", "pred_score": 0.8}
    if any(w in text for w in ("hate", "terrible", "awful", "bad")):
        return {"pred_label": "negative", "pred_score": -0.8}
    return {"pred_label": "neutral", "pred_score": 0.0}


def _stub_predict_image(payload: dict) -> dict:
    return _stub_predict_text(payload)


def _stub_predict_video(payload: dict) -> dict:
    text = (payload.get("text") or "").lower()
    base = _stub_predict_text({"text": text})
    return base


def _pipeline_predict_text(payload: dict) -> dict:
    from src.pipeline import MyUniSentimentPipeline

    pipeline = MyUniSentimentPipeline()
    result = pipeline.analyze_text(payload["text"])
    overall = result.analysis.overall
    return {
        "pred_label": overall.label,
        "pred_score": overall.score,
        "raw": result.model_dump_json_compatible(),
    }


def _pipeline_predict_image(payload: dict) -> dict:
    from src.pipeline import MyUniSentimentPipeline
    from src.schemas import ActivityInput

    pipeline = MyUniSentimentPipeline()
    activity = ActivityInput(
        activity_id="EVAL-IMG",
        user_id="EVAL",
        activity_type="image",
        text=payload.get("text"),
        media_path=payload["image_path"],
        created_at=datetime.now(timezone.utc),
    )
    result = pipeline.analyze_activity(activity)
    overall = result.analysis.overall
    return {
        "pred_label": overall.label,
        "pred_score": overall.score,
        "raw": result.model_dump_json_compatible(),
    }


def _pipeline_predict_video(payload: dict) -> dict:
    from src.pipeline import MyUniSentimentPipeline
    from src.schemas import ActivityInput

    pipeline = MyUniSentimentPipeline()
    activity = ActivityInput(
        activity_id="EVAL-VID",
        user_id="EVAL",
        activity_type="video",
        text=payload.get("text"),
        media_path=payload["media_path"],
        created_at=datetime.now(timezone.utc),
    )
    result = pipeline.analyze_activity(activity)
    overall = result.analysis.overall
    return {
        "pred_label": overall.label,
        "pred_score": overall.score,
        "raw": result.model_dump_json_compatible(),
    }


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.modality == "text":
        if args.tweeteval_hf:
            examples = load_tweeteval_hf(split=args.split)
        else:
            if not args.data:
                parser.error("--data is required unless --tweeteval-hf is set")
            examples = load_tweeteval_jsonl(args.data)
        predict = _stub_predict_text if args.stub else _pipeline_predict_text
        result = evaluate_text_examples(
            examples,
            predict,
            limit=args.limit,
            split=args.split,
        )
    elif args.modality == "image":
        if not args.data:
            parser.error("--data is required for image/MVSA evaluation")
        examples = load_mvsa_jsonl(args.data)
        predict = _stub_predict_image if args.stub else _pipeline_predict_image
        result = evaluate_mvsa_examples(
            examples,
            predict,
            limit=args.limit,
            split=args.split,
        )
    else:
        if not args.data:
            parser.error("--data is required for video/MOSI evaluation")
        examples = load_mosi_jsonl(args.data, apply_3way=True)
        predict = _stub_predict_video if args.stub else _pipeline_predict_video
        result = evaluate_mosi_examples(
            examples,
            predict,
            limit=args.limit,
            split=args.split,
        )

    print(format_console_summary(result))
    out_dir = args.out or str(ROOT / "outputs" / f"eval_{args.modality}")
    paths = write_results(result, out_dir)
    print(f"Wrote {paths['metrics_json']}")
    print(f"Wrote {paths['predictions_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
