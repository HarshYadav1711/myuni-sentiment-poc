#!/usr/bin/env python3
"""Generate a short synthetic sample video via FFmpeg (no copyrighted assets)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples" / "synthetic_sample.mp4"


def main() -> int:
    from src.media.ffmpeg_utils import FFmpegNotFoundError, find_ffmpeg

    try:
        exe = find_ffmpeg()
    except FFmpegNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=steelblue:s=320x240:d=2",
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
        str(OUT),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        print(completed.stderr[-800:], file=sys.stderr)
        return 1
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
