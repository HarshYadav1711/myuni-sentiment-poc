#!/usr/bin/env python3
"""Write a MOSI-style JSONL template (does not download CMU-MOSI media)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/eval/mosi_index.jsonl")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": "mosi-demo-1",
            "video_path": "clips/demo1.mp4",
            "score": 2.0,
            "text": "I really enjoyed this movie",
        },
        {
            "id": "mosi-demo-2",
            "video_path": "clips/demo2.mp4",
            "score": -1.5,
            "text": "This was a terrible film",
        },
        {
            "id": "mosi-demo-3",
            "video_path": "clips/demo3.mp4",
            "score": 0.0,
            "text": "It was okay I guess",
        },
    ]
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"Wrote template {out} (replace paths/scores with local MOSI exports; see docs/DATASETS.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
