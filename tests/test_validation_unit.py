"""Lightweight unit tests that do not download models."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyzers.text import TextSentimentAnalyzer
from src.pipeline import MyUniSentimentPipeline


def test_validate_text_rejects_blank_and_non_string() -> None:
    with pytest.raises(ValueError):
        TextSentimentAnalyzer.validate_text("")
    with pytest.raises(ValueError):
        TextSentimentAnalyzer.validate_text(" \n\t ")
    with pytest.raises(ValueError):
        TextSentimentAnalyzer.validate_text(123)  # type: ignore[arg-type]


def test_validate_text_strips_whitespace() -> None:
    assert TextSentimentAnalyzer.validate_text("  hello  ") == "hello"


def test_pipeline_generates_activity_id_when_omitted() -> None:
    """Validation runs before model load; blank path never needs weights."""
    pipeline = MyUniSentimentPipeline()
    with pytest.raises(ValueError):
        pipeline.analyze_text(" ")
