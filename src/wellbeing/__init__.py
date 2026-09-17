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
from src.wellbeing.evidence import build_wellbeing_evidence_context
from src.wellbeing.authority_readiness import (
    current_authority_readiness,
    evaluate_candidate_authority_guard,
    evaluate_wellbeing_authority_readiness,
)
from src.wellbeing.migration import (
    attach_authority_comparison,
    classify_candidate_outcome,
    compare_wellbeing_policies,
)
from src.wellbeing.policy_candidate import build_wellbeing_policy_candidate
from src.wellbeing.schemas import (
    WellbeingAuthorityComparison,
    WellbeingAuthorityDecision,
    WellbeingAuthorityReadiness,
    WellbeingClassificationResult,
    WellbeingEvidenceContext,
    WellbeingPolicyCandidate,
    WellbeingShadowAnalysis,
)
from src.wellbeing.shadow import build_wellbeing_shadow

__all__ = [
    "WellbeingAuthorityComparison",
    "WellbeingAuthorityDecision",
    "WellbeingAuthorityReadiness",
    "WellbeingClassifier",
    "WellbeingClassificationResult",
    "WellbeingEvidenceContext",
    "WellbeingPolicyCandidate",
    "WellbeingShadowAnalysis",
    "attach_authority_comparison",
    "build_wellbeing_evidence_context",
    "build_wellbeing_policy_candidate",
    "build_wellbeing_shadow",
    "classify_candidate_outcome",
    "classify_wellbeing",
    "compare_wellbeing_policies",
    "current_authority_readiness",
    "evaluate_candidate_authority_guard",
    "evaluate_wellbeing_authority_readiness",
    "get_wellbeing_classifier",
]
