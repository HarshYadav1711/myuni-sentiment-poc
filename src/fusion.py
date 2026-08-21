"""Simple explainable late fusion for POC overall sentiment (not client scoring)."""

from __future__ import annotations

from typing import Optional

from src.config import DEFAULT_FUSION, FusionConfig
from src.schemas import SentimentEvidence, SentimentLabel


def fuse_modality_scores(
    modalities: dict[str, Optional[SentimentEvidence]],
    config: FusionConfig = DEFAULT_FUSION,
) -> SentimentEvidence:
    """Confidence-weighted average of modality scores.

    POC-only rule:
    - ``fused_score = sum(w_i * conf_i * score_i) / sum(w_i * conf_i)``
    - label from fused_score vs ``neutral_band``
    - confidence = weighted mean of participating confidences
    """
    weighted_score = 0.0
    weight_sum = 0.0
    conf_acc = 0.0
    used: list[str] = []

    for name, evidence in modalities.items():
        if evidence is None:
            continue
        w = float(config.modality_weights.get(name, 1.0))
        if w <= 0:
            continue
        contrib = w * evidence.confidence
        if contrib <= 0:
            continue
        weighted_score += contrib * evidence.score
        weight_sum += contrib
        conf_acc += contrib * evidence.confidence
        used.append(name)

    if weight_sum <= 0:
        return SentimentEvidence(
            label="neutral",
            score=0.0,
            confidence=0.0,
            probabilities={"negative": 0.0, "neutral": 1.0, "positive": 0.0},
            model="poc-fusion",
            details={"rule": "no modality evidence available", "modalities_used": []},
        )

    fused_score = max(-1.0, min(1.0, weighted_score / weight_sum))
    fused_confidence = max(0.0, min(1.0, conf_acc / weight_sum))

    if fused_score > config.neutral_band:
        label: SentimentLabel = "positive"
    elif fused_score < -config.neutral_band:
        label = "negative"
    else:
        label = "neutral"

    return SentimentEvidence(
        label=label,
        score=float(fused_score),
        confidence=float(fused_confidence),
        model="poc-fusion",
        details={
            "rule": "confidence-weighted average of modality scores",
            "neutral_band": config.neutral_band,
            "modality_weights": dict(config.modality_weights),
            "modalities_used": used,
            "note": "POC-only fusion; not the client business scoring methodology",
        },
    )
