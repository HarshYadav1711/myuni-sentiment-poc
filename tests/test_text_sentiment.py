"""Unit/smoke tests for Milestone 1 text sentiment path.

Heavy model download/inference tests are marked and may be skipped with:
    pytest -m "not integration"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyzers.text import TextSentimentAnalyzer
from src.pipeline import MyUniSentimentPipeline
from src.schemas import ActivityAnalysisResult


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pipeline() -> MyUniSentimentPipeline:
    return MyUniSentimentPipeline()


def test_clearly_positive_sentence(pipeline: MyUniSentimentPipeline) -> None:
    result = pipeline.analyze_text("I really enjoyed today's workshop.")
    assert result.analysis.overall.label == "positive"
    assert result.analysis.overall.score > 0
    assert result.analysis.modalities.text is not None
    assert result.analysis.modalities.text.model == (
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )


def test_clearly_neutral_sentence(pipeline: MyUniSentimentPipeline) -> None:
    result = pipeline.analyze_text("The lecture is scheduled for Thursday at 3pm.")
    assert result.analysis.overall.label == "neutral"
    assert abs(result.analysis.overall.score) < 0.5


def test_clearly_negative_sentence(pipeline: MyUniSentimentPipeline) -> None:
    result = pipeline.analyze_text("This was terrible.")
    assert result.analysis.overall.label == "negative"
    assert result.analysis.overall.score < 0


def test_blank_input_validation(pipeline: MyUniSentimentPipeline) -> None:
    with pytest.raises(ValueError, match="non-blank"):
        pipeline.analyze_text("   ")
    with pytest.raises(ValueError, match="string"):
        pipeline.analyze_text(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-blank"):
        TextSentimentAnalyzer.validate_text("")


def test_activity_and_user_id_preservation(pipeline: MyUniSentimentPipeline) -> None:
    result = pipeline.analyze_text(
        "Campus wifi is working fine today.",
        user_id="U007",
        activity_id="ACT0042",
    )
    assert result.user_id == "U007"
    assert result.activity_id == "ACT0042"
    assert result.activity_type == "text"
    assert result.input.text_length is not None
    assert result.input.text_length > 0
    assert 0.0 <= result.analysis.overall.confidence <= 1.0
    assert -1.0 <= result.analysis.overall.score <= 1.0

    # Structured JSON round-trip
    payload = result.model_dump_json_compatible()
    assert payload["user_id"] == "U007"
    assert payload["activity_id"] == "ACT0042"
    reloaded = ActivityAnalysisResult.model_validate(payload)
    assert reloaded.activity_id == "ACT0042"
    # Ensure CLI-style serialization is valid JSON
    json.dumps(payload)
