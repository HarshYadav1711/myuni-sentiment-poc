#!/usr/bin/env python3
"""Generate a tiny synthetic sample image (no third-party assets / no copyrighted media)."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples" / "synthetic_sample.png"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (320, 180), color=(70, 130, 180))
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 300, 160), outline=(255, 255, 255), width=3)
    draw.ellipse((120, 50, 200, 130), fill=(255, 215, 0))
    # Embedded text for optional OCR smoke checks (synthetic, not a copyrighted asset).
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None
    draw.text((40, 150), "CAMPUS EVENT", fill=(255, 255, 255), font=font)
    img.save(OUT)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
