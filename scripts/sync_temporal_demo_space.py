"""Sync a minimal Hugging Face Gradio bundle for the MyUni temporal demo.

Destination: D:\\Work\\hf-deploy\\myuni-temporal-demo
Target Space (manual push later): Norman22194/myuni-temporal-demo

Does NOT touch:
- D:\\Work\\hf-deploy\\My-Space
- D:\\Work\\hf-deploy\\myuni-temporal-reasoner-benchmark

Does NOT push.
Does NOT include secrets, tests, model caches, or .env files.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(r"D:\Work\hf-deploy\myuni-temporal-demo")

# Runtime paths required by app_gradio → MyUniSentimentPipeline (video temporal demo).
COPY_PATHS = [
    "app_gradio.py",
    "packages.txt",
    "config/fusion.yaml",
    "src/__init__.py",
    "src/config.py",
    "src/schemas.py",
    "src/pipeline.py",
    "src/fusion.py",
    "src/runtime_info.py",
    "src/env_check.py",
    "src/logging_config.py",
    "src/batch.py",
    "src/analyzers",
    "src/media",
    "src/routing",
    "src/storage",
    "src/ui",
    "src/temporal",
]

IGNORE = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.ckpt",
    ".env",
    ".env.*",
)


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=IGNORE)


def _write_requirements(dest: Path) -> None:
    src = ROOT / "requirements-hf.txt"
    text = src.read_text(encoding="utf-8")
    (dest / "requirements.txt").write_text(text, encoding="utf-8")


def _write_readme(dest: Path) -> None:
    readme = """---
title: MyUni Temporal Intelligence
sdk: gradio
sdk_version: 5.50.0
app_file: app_gradio.py
python_version: 3.12.12
---

# MyUni Temporal Intelligence

Client-facing end-to-end demo for multimodal social-content sentiment with
**temporal video context** and OpenRouter contextual explanation.

## What this Space does

- Text / image / audio / video analysis with automatic input detection
- Video path: frame sampling → SigLIP 2 (ZeroGPU) → Whisper + RoBERTa (CPU)
  → deterministic temporal windows/features → OpenRouter contextual reasoner
- Clean Gradio UI for client testing (not an internal benchmark UI)

## Overall Wellbeing Indicator

Categorical POC content-level indicator only:

- Low concern
- Moderate concern
- High concern
- Insufficient evidence

**POC content-level indicator — not a clinical assessment.**

No numerical wellbeing score (no X/10 or X/100). Not a psychiatric diagnosis.

## Hardware / models

| Stage | Where it runs |
| --- | --- |
| SigLIP 2 visual | ZeroGPU |
| Faster-Whisper `base.en` | CPU int8 |
| Twitter-RoBERTa | CPU |
| Temporal features | CPU |
| Contextual reasoner | OpenRouter HTTP (`openai/gpt-oss-20b:free`) |

Qwen is **not** invoked by default and is not on the ZeroGPU path.

## Required Space Secret

Configure in Space settings → Secrets:

- `OPENROUTER_API_KEY`

Never commit this value. The app only reports configured/unconfigured.

## Optional Space Variables

| Variable | Default |
| --- | --- |
| `TEMPORAL_REASONER_PROVIDER` | `openrouter` |
| `OPENROUTER_REASONER_MODEL` | `openai/gpt-oss-20b:free` |
| `TEMPORAL_REASONER_FALLBACK` | `none` |

Keep fallback at `none` so OpenRouter failures do not consume ZeroGPU quota via Qwen.

## Fail-soft behavior

If OpenRouter is unavailable or rate-limited:

- overall video sentiment still returns
- deterministic temporal context still returns
- UI shows: **Context explanation temporarily unavailable.**

## Notes

- Fusion weights and thresholds are POC evaluation defaults only
- OCR uses Tesseract when available (`packages.txt`); visual path still works without it
- FFmpeg is expected from the Space image
"""
    (dest / "README.md").write_text(readme, encoding="utf-8")


def _write_gitignore(dest: Path) -> None:
    gitignore = """# Hugging Face Space artifacts / caches / secrets
.cache/
__pycache__/
*.pyc
.pytest_cache/
outputs/
*.safetensors
*.bin
*.pt
*.pth
*.onnx
*.ckpt
.venv/
venv/
.env
.env.*
*.db
*.sqlite
*.sqlite3
.DS_Store
Thumbs.db
"""
    (dest / ".gitignore").write_text(gitignore, encoding="utf-8")


def _write_gitattributes(dest: Path) -> None:
    src = Path(r"D:\Work\hf-deploy\My-Space\.gitattributes")
    if src.is_file():
        shutil.copy2(src, dest / ".gitattributes")


def _write_push_instructions(dest: Path) -> None:
    text = """# How to create & push this Space (manual — do not auto-push)

1. On Hugging Face, create a **new** Space:
   - Owner/name: `Norman22194/myuni-temporal-demo`
   - SDK: Gradio
   - Hardware: **ZeroGPU**
   - Do **not** overwrite `My-Space` or the reasoner-benchmark Space

2. Clone the empty Space locally:
   ```bash
   git clone https://huggingface.co/spaces/Norman22194/myuni-temporal-demo
   ```

3. Copy the contents of this directory (`D:\\Work\\hf-deploy\\myuni-temporal-demo`)
   into the clone (overwrite README/requirements as needed).

4. In the Space settings, add Secret:
   - `OPENROUTER_API_KEY` = (your OpenRouter key)
   Optional Variables:
   - `TEMPORAL_REASONER_PROVIDER=openrouter`
   - `OPENROUTER_REASONER_MODEL=openai/gpt-oss-20b:free`
   - `TEMPORAL_REASONER_FALLBACK=none`

5. Commit and push from the Space clone (user action only):
   ```bash
   git add -A
   git commit -m "Add MyUni temporal demo Space (OpenRouter + ZeroGPU SigLIP)"
   git push
   ```

6. Open the Space, upload a short video, click Analyze.
   Confirm:
   - Overall Sentiment
   - Overall Wellbeing Indicator
   - Key Temporal Highlights
   - Summary Explanation (or fail-soft unavailable message)

Notes:
- SigLIP 2 is the only ZeroGPU model.
- OpenRouter is external HTTP; no live call is made on Space import.
- Qwen is not loaded by default.
"""
    (dest / "PUSH_INSTRUCTIONS.md").write_text(text, encoding="utf-8")


def main() -> int:
    # Hard safety: refuse if DEST accidentally points at protected trees.
    protected = {
        Path(r"D:\Work\hf-deploy\My-Space").resolve(),
        Path(r"D:\Work\hf-deploy\myuni-temporal-reasoner-benchmark").resolve(),
    }
    dest_resolved = DEST.resolve()
    if dest_resolved in protected:
        print(f"REFUSING to write into protected path: {dest_resolved}", file=sys.stderr)
        return 1
    for p in protected:
        try:
            dest_resolved.relative_to(p)
            print(f"REFUSING nested write under protected path: {p}", file=sys.stderr)
            return 1
        except ValueError:
            pass

    DEST.mkdir(parents=True, exist_ok=True)

    # Clear previous Python sources to avoid stale files, keep .git if present.
    for child in list(DEST.iterdir()):
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    for rel in COPY_PATHS:
        src = ROOT / rel
        if not src.exists():
            print(f"MISSING: {src}", file=sys.stderr)
            return 1
        _copy_path(src, DEST / rel)
        print(f"copied {rel}")

    _write_requirements(DEST)
    print("wrote requirements.txt")
    _write_readme(DEST)
    print("wrote README.md")
    _write_gitignore(DEST)
    print("wrote .gitignore")
    _write_gitattributes(DEST)
    print("wrote .gitattributes")
    _write_push_instructions(DEST)
    print("wrote PUSH_INSTRUCTIONS.md")

    # Ensure no accidental secrets / caches slipped in.
    banned_names = {".env", ".venv"}
    for path in DEST.rglob("*"):
        if path.name in banned_names or path.name.startswith(".env."):
            print(f"UNEXPECTED banned path: {path}", file=sys.stderr)
            return 1
        if path.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth", ".onnx", ".ckpt"}:
            print(f"UNEXPECTED model cache file: {path}", file=sys.stderr)
            return 1

    print(f"Destination ready: {DEST}")
    # Strip any local bytecode left from prior validation imports.
    for cache in DEST.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
