"""Transparent probability aggregation helpers for temporal windows.

Missing evidence is never treated as neutral or zero — callers must omit it.
"""

from __future__ import annotations

from typing import Optional, Sequence

from src.schemas import SentimentEvidence, SentimentLabel

_LABELS: tuple[SentimentLabel, ...] = ("negative", "neutral", "positive")


def extract_probabilities(evidence: SentimentEvidence) -> Optional[dict[str, float]]:
    """Return a normalized {negative, neutral, positive} map, or None if unavailable."""
    probs = evidence.probabilities
    if not probs:
        return None
    out: dict[str, float] = {}
    for key in _LABELS:
        try:
            out[key] = float(probs.get(key, 0.0))
        except (TypeError, ValueError):
            out[key] = 0.0
    total = sum(out.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in out.items()}


def mean_probability_distribution(
    evidence_list: Sequence[SentimentEvidence],
) -> Optional[dict[str, float]]:
    """Unweighted mean of probability distributions (baseline visual aggregation).

    Evidence items without usable probabilities are skipped. If none remain,
    returns None (not a neutral distribution).
    """
    collected: list[dict[str, float]] = []
    for ev in evidence_list:
        probs = extract_probabilities(ev)
        if probs is not None:
            collected.append(probs)
    if not collected:
        return None
    n = float(len(collected))
    return {key: sum(p[key] for p in collected) / n for key in _LABELS}


def length_weighted_probability_distribution(
    evidence_list: Sequence[SentimentEvidence],
    lengths: Sequence[float],
) -> Optional[dict[str, float]]:
    """Length-weighted mean of probability distributions (speech/OCR text).

    ``lengths`` must align with ``evidence_list``. Non-positive lengths and
    evidence without probabilities are skipped. Returns None if nothing usable.
    """
    if len(evidence_list) != len(lengths):
        raise ValueError("evidence_list and lengths must have the same length")

    num = {key: 0.0 for key in _LABELS}
    weight_sum = 0.0
    for ev, length in zip(evidence_list, lengths):
        probs = extract_probabilities(ev)
        w = float(length)
        if probs is None or w <= 0:
            continue
        weight_sum += w
        for key in _LABELS:
            num[key] += w * probs[key]

    if weight_sum <= 0:
        return None
    return {key: num[key] / weight_sum for key in _LABELS}


def label_from_probabilities(
    probs: dict[str, float],
    *,
    negative_threshold: float,
    positive_threshold: float,
) -> SentimentLabel:
    """Map a probability distribution to a label using explicit thresholds.

    Prefer thresholded polarity when P(neg) or P(pos) clears its threshold;
    otherwise fall back to argmax (may still be neutral).
    """
    neg = float(probs.get("negative", 0.0))
    pos = float(probs.get("positive", 0.0))
    if neg >= negative_threshold and neg >= pos:
        return "negative"
    if pos >= positive_threshold and pos >= neg:
        return "positive"
    # Argmax among the three; ties prefer neutral then positive.
    return max(_LABELS, key=lambda k: (probs.get(k, 0.0), k == "neutral", k == "positive"))


def score_from_probabilities(probs: dict[str, float]) -> float:
    """POC score = P(positive) - P(negative), clipped to [-1, 1]."""
    return max(-1.0, min(1.0, float(probs.get("positive", 0.0)) - float(probs.get("negative", 0.0))))


def max_class_probability(probs: dict[str, float]) -> float:
    return max(float(probs.get(k, 0.0)) for k in _LABELS)


def combine_modality_probabilities(
    modality_probs: dict[str, dict[str, float]],
) -> Optional[dict[str, float]]:
    """Unweighted mean across available modality distributions (missing omitted)."""
    if not modality_probs:
        return None
    values = list(modality_probs.values())
    n = float(len(values))
    return {key: sum(p[key] for p in values) / n for key in _LABELS}
