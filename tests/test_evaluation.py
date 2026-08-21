"""Unit tests for evaluation metrics and adapters (no model downloads)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.common import format_console_summary, write_results
from evaluation.image.mvsa import evaluate_mvsa_examples, load_mvsa_jsonl, map_mvsa_label
from evaluation.metrics import classification_metrics, continuous_metrics
from evaluation.text.tweeteval import (
    evaluate_text_examples,
    load_tweeteval_jsonl,
    map_tweeteval_label,
)
from evaluation.video.mosi import (
    evaluate_mosi_examples,
    load_mosi_jsonl,
    mosi_score_to_3way,
    scale_mosi_to_unit,
)

FIXTURES = ROOT / "evaluation" / "fixtures"


def test_classification_metrics_perfect() -> None:
    y = ["positive", "neutral", "negative", "positive"]
    m = classification_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["f1_weighted"] == 1.0
    assert m["confusion_matrix"]["positive"]["positive"] == 2


def test_classification_metrics_errors() -> None:
    y_true = ["positive", "negative", "neutral"]
    y_pred = ["negative", "negative", "neutral"]
    m = classification_metrics(y_true, y_pred)
    assert m["n_examples"] == 3
    assert 0.0 <= m["accuracy"] <= 1.0
    assert "precision_macro" in m and "recall_macro" in m
    assert m["confusion_matrix"]["positive"]["negative"] == 1


def test_continuous_metrics() -> None:
    m = continuous_metrics([0.0, 1.0, -1.0], [0.0, 0.5, -0.5])
    assert m["mae"] == pytest.approx(1.0 / 3.0)
    assert m["pearson_correlation"] == pytest.approx(1.0)


def test_tweeteval_fixture_eval_stub() -> None:
    examples = load_tweeteval_jsonl(FIXTURES / "text_samples.jsonl")
    assert map_tweeteval_label(2) == "positive"

    def predict(payload):
        text = payload["text"].lower()
        if "love" in text or "amazing" in text:
            return {"pred_label": "positive", "pred_score": 0.9}
        if "terrible" in text or "hate" in text:
            return {"pred_label": "negative", "pred_score": -0.9}
        return {"pred_label": "neutral", "pred_score": 0.0}

    result = evaluate_text_examples(examples, predict, limit=5, split="fixture")
    assert result.n_evaluated == 5
    assert result.classification is not None
    assert result.classification["n_examples"] == 5
    summary = format_console_summary(result)
    assert "TweetEval" in summary


def test_mvsa_fixture_eval_stub() -> None:
    examples = load_mvsa_jsonl(FIXTURES / "image_samples.jsonl", require_images=False)
    assert map_mvsa_label("POS") == "positive"

    def predict(payload):
        text = payload["text"].lower()
        if "love" in text or "great" in text:
            return {"pred_label": "positive", "pred_score": 0.7}
        if "awful" in text:
            return {"pred_label": "negative", "pred_score": -0.7}
        return {"pred_label": "neutral", "pred_score": 0.0}

    result = evaluate_mvsa_examples(examples, predict, limit=2, split="fixture")
    assert result.n_requested == 2
    assert result.classification is not None


def test_mosi_3way_is_explicit_and_continuous_reported() -> None:
    assert mosi_score_to_3way(-0.1) == "negative"
    assert mosi_score_to_3way(0.0) == "neutral"
    assert mosi_score_to_3way(1.2) == "positive"
    assert scale_mosi_to_unit(3.0) == 1.0

    examples = load_mosi_jsonl(FIXTURES / "video_samples.jsonl", apply_3way=True)

    def predict(payload):
        text = (payload.get("text") or "").lower()
        if "love" in text:
            return {"pred_label": "positive", "pred_score": 0.8}
        if "terrible" in text:
            return {"pred_label": "negative", "pred_score": -0.7}
        return {"pred_label": "neutral", "pred_score": 0.0}

    result = evaluate_mosi_examples(examples, predict, split="fixture")
    assert result.label_mapping is not None
    assert "description" in result.label_mapping
    assert result.classification is not None
    assert result.continuous is not None
    assert "mae" in result.continuous


def test_write_results_json_csv(tmp_path: Path) -> None:
    examples = load_tweeteval_jsonl(FIXTURES / "text_samples.jsonl")

    def predict(payload):
        return {"pred_label": "neutral", "pred_score": 0.0}

    result = evaluate_text_examples(examples, predict, limit=3)
    paths = write_results(result, tmp_path / "out")
    metrics = json.loads(paths["metrics_json"].read_text(encoding="utf-8"))
    assert metrics["dataset"] == "TweetEval-sentiment"
    assert paths["predictions_csv"].is_file()


def test_unknown_label_raises() -> None:
    with pytest.raises(ValueError):
        map_tweeteval_label(9)
    with pytest.raises(ValueError):
        map_mvsa_label("happy")
