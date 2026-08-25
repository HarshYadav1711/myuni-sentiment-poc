"""Unit tests for Streamlit demo presentation helpers (no model load)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui.display import (
    format_confidence,
    format_confidence_pct,
    format_score,
    humanize_dependency_hint,
    humanize_ffmpeg_error,
    label_color,
)


def test_formatters() -> None:
    assert format_score(0.25) == "+0.250"
    assert format_score(-0.5) == "-0.500"
    assert format_confidence(0.8123) == "0.812"
    assert format_confidence_pct(0.8123) == "81.2%"
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


def test_app_module_unified_workspace() -> None:
    """Smoke: mockup-aligned workspace without modality tabs or sample chips."""
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "def get_pipeline" in source
    assert "def main" in source
    assert "main()" in source
    assert "Sentiment Intelligence" in source
    assert "Analyze Now" in source
    assert "Enter your content" in source
    assert "float-card" in source
    assert "theme_night_toggle" in source
    assert "def tab_text" not in source
    assert 'st.tabs(["Text", "Image", "Video"])' not in source
    assert "SAMPLE_TEXTS" not in source
    assert "pipeline().analyze(" in source or ".analyze(" in source
    # Preview may list roadmap items, but live capability is still text sentiment.
    assert "Overall Sentiment" in source
    assert "roadmap" in source.lower() or "Live in this build" in source
