#!/usr/bin/env python3
"""Developer smoke script for ambiguous social-media text examples.

Does NOT assert expected labels — prints structured JSON for manual review.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import MyUniSentimentPipeline

AMBIGUOUS_EXAMPLES = [
    "Fantastic. Another surprise assignment.",
    "Amazing result again 💀",
]


def main() -> int:
    # Windows consoles often default to cp1252; force UTF-8 for emoji examples.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    pipeline = MyUniSentimentPipeline()

    for i, text in enumerate(AMBIGUOUS_EXAMPLES, start=1):
        print(f"\n=== Example {i}: {text!r} ===")
        result = pipeline.analyze_text(
            text,
            user_id="U-SMOKE",
            activity_id=f"ACT-SMOKE-{i:02d}",
        )
        print(json.dumps(result.model_dump_json_compatible(), indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
