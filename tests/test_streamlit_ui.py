"""Unit tests for Streamlit demo presentation helpers (no model load)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui.display import (
    format_confidence,
    format_score,
    humanize_dependency_hint,
    humanize_ffmpeg_error,
    label_color,
)


def test_formatters() -> None:
    assert format_score(0.25) == "+0.250"
    assert format_score(-0.5) == "-0.500"
    assert format_confidence(0.8123) == "0.812"
    assert label_color("positive") == "#1B7F4E"
    assert label_color("negative") == "#B42318"


def test_humanize_ffmpeg_error_adds_action() -> None:
    msg = humanize_ffmpeg_error(RuntimeError("FFmpeg executable not found on PATH."))
    assert "winget install Gyan.FFmpeg" in msg


def test_dependency_hint_from_warnings() -> None:
    hint = humanize_dependency_hint(["OCR unavailable: Tesseract executable not found."])
    assert hint is not None
    assert "Tesseract" in hint
    assert humanize_dependency_hint(["frame[0] analysis failed: x"]) is None


def test_app_module_imports() -> None:
    """Smoke: app module loads without starting the Streamlit server."""
    import importlib

    # Avoid executing Streamlit page config side effects twice in the same process
    # by importing after sys.path is set (app uses streamlit at import time).
    mod = importlib.import_module("app")
    assert hasattr(mod, "get_pipeline")
    assert hasattr(mod, "main")
    assert hasattr(mod, "tab_text")
