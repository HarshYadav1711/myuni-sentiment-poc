"""Deterministic unit tests for explainable multimodal fusion (Milestone 6)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    FusionConfig,
    FusionConflictConfig,
    FusionThresholds,
    load_fusion_config,
)
from src.fusion import detect_modality_conflict, fuse_modalities, score_to_label
from src.schemas import SentimentEvidence


def _ev(
    score: float,
    confidence: float = 0.9,
    *,
    label: str | None = None,
) -> SentimentEvidence:
    if label is None:
        if score > 0.15:
            label = "positive"
        elif score < -0.15:
            label = "negative"
        else:
            label = "neutral"
    return SentimentEvidence(
        label=label,  # type: ignore[arg-type]
        score=score,
        confidence=confidence,
        model="stub",
    )


@pytest.fixture
def cfg() -> FusionConfig:
    return FusionConfig(
        modality_weights={"text": 1.0, "visual": 1.0, "ocr": 0.8, "speech": 1.0},
        thresholds=FusionThresholds(positive_above=0.15, negative_below=-0.15),
        conflict=FusionConflictConfig(
            min_confidence=0.4,
            min_polarity=0.35,
            disagreement_threshold=0.9,
            confidence_penalty=0.5,
        ),
        note="test POC defaults",
        source_path="test",
    )


def test_load_fusion_yaml() -> None:
    loaded = load_fusion_config(ROOT / "config" / "fusion.yaml")
    assert "text" in loaded.modality_weights
    assert loaded.thresholds.positive_above == 0.15
    assert loaded.conflict.disagreement_threshold == 0.9
    assert loaded.source_path is not None


def test_one_modality(cfg: FusionConfig) -> None:
    outcome = fuse_modalities({"text": _ev(0.8), "visual": None}, config=cfg)
    assert outcome.overall.label == "positive"
    assert abs(outcome.overall.score - 0.8) < 1e-9
    assert abs(outcome.overall.confidence - 0.9) < 1e-9
    assert outcome.diagnostics.contributing_modalities == ["text"]
    assert outcome.diagnostics.modality_conflict is False
    assert "text" in outcome.diagnostics.effective_weights


def test_all_positive(cfg: FusionConfig) -> None:
    outcome = fuse_modalities(
        {
            "text": _ev(0.9),
            "visual": _ev(0.7),
            "ocr": _ev(0.6),
            "speech": _ev(0.8),
        },
        config=cfg,
    )
    assert outcome.overall.label == "positive"
    assert outcome.overall.score > 0.15
    assert set(outcome.diagnostics.contributing_modalities) == {
        "text",
        "visual",
        "ocr",
        "speech",
    }
    assert outcome.diagnostics.modality_conflict is False


def test_all_negative(cfg: FusionConfig) -> None:
    outcome = fuse_modalities(
        {
            "text": _ev(-0.9),
            "visual": _ev(-0.7),
            "speech": _ev(-0.8),
        },
        config=cfg,
    )
    assert outcome.overall.label == "negative"
    assert outcome.overall.score < -0.15
    assert outcome.diagnostics.modality_conflict is False


def test_neutral(cfg: FusionConfig) -> None:
    outcome = fuse_modalities(
        {
            "text": _ev(0.05, confidence=0.8),
            "visual": _ev(-0.04, confidence=0.8),
        },
        config=cfg,
    )
    assert outcome.overall.label == "neutral"
    assert abs(outcome.overall.score) <= 0.15


def test_missing_modalities(cfg: FusionConfig) -> None:
    outcome = fuse_modalities(
        {
            "text": None,
            "visual": _ev(0.5),
            "ocr": None,
            "speech": None,
        },
        config=cfg,
    )
    assert outcome.diagnostics.contributing_modalities == ["visual"]
    assert outcome.overall.label == "positive"


def test_strongly_conflicting_modalities(cfg: FusionConfig) -> None:
    text = _ev(-0.9, confidence=0.95)
    visual = _ev(0.85, confidence=0.9)
    conflict, disagreement, strong = detect_modality_conflict(
        {"text": text, "visual": visual},
        config=cfg,
    )
    assert conflict is True
    assert disagreement >= 0.9
    assert set(strong) == {"text", "visual"}

    outcome = fuse_modalities({"text": text, "visual": visual}, config=cfg)
    assert outcome.diagnostics.modality_conflict is True
    assert outcome.diagnostics.disagreement_score >= 0.9
    # Confidence lowered vs simple weighted mean of confidences.
    naive_conf = (0.95 * 0.95 + 0.9 * 0.9) / (0.95 + 0.9)
    assert outcome.overall.confidence < naive_conf
    assert "modality_conflict=true" in outcome.diagnostics.explanation


def test_low_confidence_does_not_force_conflict(cfg: FusionConfig) -> None:
    outcome = fuse_modalities(
        {
            "text": _ev(-0.9, confidence=0.2),
            "visual": _ev(0.9, confidence=0.2),
        },
        config=cfg,
    )
    # Both below min_confidence for conflict detection.
    assert outcome.diagnostics.modality_conflict is False


def test_zero_usable_evidence(cfg: FusionConfig) -> None:
    outcome = fuse_modalities(
        {
            "text": None,
            "visual": _ev(0.5, confidence=0.0),
            "ocr": None,
        },
        config=cfg,
    )
    assert outcome.overall.label == "neutral"
    assert outcome.overall.score == 0.0
    assert outcome.overall.confidence == 0.0
    assert outcome.diagnostics.contributing_modalities == []
    assert outcome.diagnostics.modality_conflict is False


def test_score_to_label_thresholds(cfg: FusionConfig) -> None:
    assert score_to_label(0.2, cfg) == "positive"
    assert score_to_label(-0.2, cfg) == "negative"
    assert score_to_label(0.0, cfg) == "neutral"


def test_effective_weights_include_confidence(cfg: FusionConfig) -> None:
    outcome = fuse_modalities(
        {
            "text": _ev(1.0, confidence=1.0),
            "ocr": _ev(1.0, confidence=0.5),  # configured weight 0.8
        },
        config=cfg,
    )
    assert abs(outcome.diagnostics.effective_weights["text"] - 1.0) < 1e-9
    assert abs(outcome.diagnostics.effective_weights["ocr"] - 0.4) < 1e-9


def test_diagnostics_are_deterministic(cfg: FusionConfig) -> None:
    mods = {"text": _ev(-0.8), "speech": _ev(0.75)}
    a = fuse_modalities(mods, config=cfg)
    b = fuse_modalities(mods, config=cfg)
    assert a.diagnostics.model_dump() == b.diagnostics.model_dump()
    assert a.overall.score == b.overall.score
    assert a.overall.confidence == b.overall.confidence
