"""POC wellbeing-indicator gating over deterministic temporal context.

These are transparent content-level rules for the client demo.
They are NOT clinically validated thresholds and must not be described as such.

No numerical wellbeing score is produced — only a conservative categorical enum.
"""

from __future__ import annotations

from typing import Optional

from src.config import (
    WELLBEING_HIGH_NEG_RUN,
    WELLBEING_HIGH_PERSISTENCE,
    WELLBEING_HIGH_STRONGEST_NEG,
    WELLBEING_MIN_USABLE_COVERAGE,
    WELLBEING_MODERATE_PERSISTENCE,
)
from src.schemas import ContextType, TemporalContext, WellbeingIndicator

# Context types that are normally insufficient for a personal wellbeing indicator.
_CONSERVATIVE_INSUFFICIENT_CONTEXTS: frozenset[str] = frozenset(
    {
        "quoted_or_reposted_content",
        "informational",
        "narrative_or_entertainment",
        "general_commentary",
        "uncertain",
    },
)

WELLBEING_RULE_DOC = """
POC wellbeing indicator rules (content-level, not clinical):

1. insufficient_evidence when usable evidence coverage is below the POC floor,
   or when no usable windows exist.
2. insufficient_evidence when context_type is not personal_expression
   (quoted/reposted, informational, narrative, general commentary, uncertain).
   humor_or_sarcasm is also treated as insufficient_evidence for the POC.
3. For personal_expression only:
   - high_concern: strong negative persistence with worsening/long run, or
     very strong negative window with elevated persistence.
   - moderate_concern: moderate persistence, sudden negative change, or
     increasing_negative trajectory.
   - low_concern: otherwise (including stable/low-negative personal expression).

The LLM may explain the indicator but must not invent it.
No 0–10 or 0–100 numerical score is used — there is no calibrated formula.
""".strip()

# Client-facing display labels (internal enum unchanged).
WELLBEING_CLIENT_LABELS: dict[str, str] = {
    "low_concern": "Low Stress",
    "moderate_concern": "Moderate Stress",
    "high_concern": "High Stress",
    "insufficient_evidence": "Insufficient Evidence",
}


def _coverage_insufficient(temporal: TemporalContext) -> bool:
    feats = temporal.features
    coverage = float(feats.evidence_coverage.overall_usable_coverage or 0.0)
    usable = [w for w in temporal.windows if w.usable]
    return coverage < WELLBEING_MIN_USABLE_COVERAGE or not usable


def _personal_expression_indicator(temporal: TemporalContext) -> WellbeingIndicator:
    """Gated indicator assuming context_type == personal_expression (coverage already ok)."""
    feats = temporal.features
    persistence = float(feats.negative_persistence or 0.0)
    trajectory = str(feats.trajectory or "")
    longest_run = int(feats.longest_negative_run or 0)
    strongest = feats.strongest_negative_window
    strongest_score = float(strongest.score) if strongest is not None else 0.0
    sudden = bool(feats.sudden_negative_change.detected)

    if (
        persistence >= WELLBEING_HIGH_PERSISTENCE
        and (
            trajectory == "increasing_negative"
            or longest_run >= WELLBEING_HIGH_NEG_RUN
        )
    ) or (
        strongest_score >= WELLBEING_HIGH_STRONGEST_NEG
        and persistence >= WELLBEING_MODERATE_PERSISTENCE
    ):
        return "high_concern"

    if (
        persistence >= WELLBEING_MODERATE_PERSISTENCE
        or sudden
        or trajectory == "increasing_negative"
    ):
        return "moderate_concern"

    return "low_concern"


def compute_wellbeing_indicator(
    temporal: TemporalContext,
    *,
    context_type: ContextType | str = "uncertain",
) -> WellbeingIndicator:
    """Return a conservative categorical indicator from deterministic facts + context."""
    if _coverage_insufficient(temporal):
        return "insufficient_evidence"

    ctx = str(context_type or "uncertain").strip().lower()
    if ctx != "personal_expression":
        # Conservative POC: only personal_expression may leave insufficient_evidence.
        # humor_or_sarcasm / quoted / informational / etc. → insufficient_evidence.
        return "insufficient_evidence"

    return _personal_expression_indicator(temporal)


def personal_expression_wellbeing_candidate(
    temporal: TemporalContext,
) -> WellbeingIndicator:
    """Deterministic candidate used when context_type resolves to personal_expression.

    If coverage is insufficient, returns insufficient_evidence. The LLM must not
    override this gated result.
    """
    if _coverage_insufficient(temporal):
        return "insufficient_evidence"
    return _personal_expression_indicator(temporal)


def wellbeing_indicator_label(indicator: WellbeingIndicator | str) -> str:
    """Human-readable client label for UI display (Stress categories)."""
    key = str(indicator)
    if key in WELLBEING_CLIENT_LABELS:
        return WELLBEING_CLIENT_LABELS[key]
    return key.replace("_", " ").title()
