"""Unit tests for Streamlit demo presentation helpers (no model load)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
    """Smoke: app module defines entrypoints without requiring Streamlit runtime."""
    app_path = ROOT / "app.py"
    source = app_path.read_text(encoding="utf-8")
    assert "def get_pipeline" in source
    assert "def main" in source
    assert "def tab_text" in source
    assert "main()" in source
