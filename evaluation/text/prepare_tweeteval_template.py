#!/usr/bin/env python3
"""Write a TweetEval-style JSONL template (does not download the corpus)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="data/eval/tweeteval_index.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": "demo-pos", "text": "I love this lecture", "label": "positive"},
        {"id": "demo-neu", "text": "The class is at 3pm", "label": "neutral"},
        {"id": "demo-neg", "text": "This assignment is terrible", "label": "negative"},
    ]
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"Wrote template {out} (replace with real TweetEval rows; see docs/DATASETS.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
