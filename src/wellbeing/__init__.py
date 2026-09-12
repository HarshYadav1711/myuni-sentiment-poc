"""Independent wellbeing / context classification (Phase 4 foundation).

Sentiment and wellbeing are separate tasks. This package classifies
content-level wellbeing relevance, target, and expressed language signals.
It is not wired into the temporal wellbeing gate or FinalTemporalAssessment yet.
"""

from __future__ import annotations

from src.wellbeing.classifier import (
    WellbeingClassifier,
    classify_wellbeing,
    get_wellbeing_classifier,
)
from src.wellbeing.schemas import WellbeingClassificationResult, WellbeingShadowAnalysis
from src.wellbeing.shadow import build_wellbeing_shadow

__all__ = [
    "WellbeingClassifier",
    "WellbeingClassificationResult",
    "WellbeingShadowAnalysis",
    "build_wellbeing_shadow",
    "classify_wellbeing",
    "get_wellbeing_classifier",
]
