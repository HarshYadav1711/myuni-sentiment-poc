#!/usr/bin/env python3
"""Write an MVSA-style JSONL template (does not download MVSA media)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/eval/mvsa_index.jsonl")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": "mvsa-demo-1",
            "text": "great campus vibes",
            "image_path": "images/demo1.jpg",
            "label": "positive",
            "text_label": "positive",
            "image_label": "positive",
        },
    ]
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"Wrote template {out} (point image_path at your local MVSA files; see docs/DATASETS.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
