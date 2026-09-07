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


def compute_wellbeing_indicator(
    temporal: TemporalContext,
    *,
    context_type: ContextType | str = "uncertain",
) -> WellbeingIndicator:
    """Return a conservative categorical indicator from deterministic facts + context."""
    feats = temporal.features
    coverage = float(feats.evidence_coverage.overall_usable_coverage or 0.0)
    usable = [w for w in temporal.windows if w.usable]
    if coverage < WELLBEING_MIN_USABLE_COVERAGE or not usable:
        return "insufficient_evidence"

    ctx = str(context_type or "uncertain").strip().lower()
    if ctx != "personal_expression":
        # Conservative POC: only personal_expression may leave insufficient_evidence.
        # humor_or_sarcasm / quoted / informational / etc. → insufficient_evidence.
        return "insufficient_evidence"

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


def wellbeing_indicator_label(indicator: WellbeingIndicator | str) -> str:
    """Human-readable label for UI display."""
    mapping = {
        "low_concern": "Low concern",
        "moderate_concern": "Moderate concern",
        "high_concern": "High concern",
        "insufficient_evidence": "Insufficient evidence",
    }
    return mapping.get(str(indicator), str(indicator).replace("_", " ").title())
