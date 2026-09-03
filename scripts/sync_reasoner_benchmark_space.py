"""Sync a minimal ZeroGPU reasoner-benchmark Space bundle.

Does NOT touch D:\\Work\\hf-deploy\\My-Space.
Does NOT download models.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(r"D:\Work\hf-deploy\myuni-temporal-reasoner-benchmark")

# Paths relative to repo root that the frozen-fixture reasoner path needs.
COPY_PATHS = [
    "src/__init__.py",
    "src/config.py",
    "src/schemas.py",
    "src/temporal",
    "evaluation/temporal_reasoner",
    "tests/benchmark/fixtures",
]


def _copy_tree(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for rel in COPY_PATHS:
        src = ROOT / rel
        if not src.exists():
            print(f"MISSING: {src}", file=sys.stderr)
            return 1
        _copy_tree(src, DEST / rel)
        print(f"copied {rel}")

    # Ensure package markers for evaluation / tests parents.
    for pkg in ("evaluation", "tests", "tests/benchmark"):
        marker = DEST / pkg / "__init__.py"
        marker.parent.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text('"""Package marker for Space imports."""\n', encoding="utf-8")

    print(f"Destination ready: {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
