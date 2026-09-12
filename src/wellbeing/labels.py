"""Central taxonomy and zero-shot hypotheses for the wellbeing classifier.

All candidate verbalizations and hypothesis templates live here.
Do not scatter hidden prompts across other modules.

Labels describe LANGUAGE / EVIDENCE IN CONTENT — not diagnoses,
not clinical conditions, and not inferred internal mental states.
"""

from __future__ import annotations

from typing import Final, Mapping

# ---------------------------------------------------------------------------
# Relevance — exclusive (multi_label=False)
# ---------------------------------------------------------------------------

RELEVANCE_LABELS: Final[tuple[str, ...]] = (
    "personal_wellbeing",
    "wellbeing_topic_only",
    "not_wellbeing_related",
    "ambiguous",
)

RELEVANCE_DEFINITIONS: Final[Mapping[str, str]] = {
    "personal_wellbeing": (
        "The speaker/author is expressing information about their own current "
        "or recent wellbeing, stress, emotional difficulty, coping or recovery."
    ),
    "wellbeing_topic_only": (
        "The content discusses wellbeing, mental health, stress, burnout, "
        "coping or related issues but does not clearly express the author's "
        "own wellbeing state."
    ),
    "not_wellbeing_related": "No meaningful personal-wellbeing content.",
    "ambiguous": (
        "The available text is insufficient or unclear regarding wellbeing "
        "relevance."
    ),
}

# Human-readable NLI candidates (NOT raw label IDs).
RELEVANCE_CANDIDATES: Final[Mapping[str, str]] = {
    "personal_wellbeing": (
        "the author expressing their own current or recent wellbeing, stress, "
        "emotional difficulty, coping, or recovery"
    ),
    "wellbeing_topic_only": (
        "a discussion of wellbeing, mental health, stress, burnout, or coping "
        "as a topic without the author clearly describing their own wellbeing"
    ),
    "not_wellbeing_related": (
        "content with no meaningful personal-wellbeing relevance"
    ),
    "ambiguous": (
        "text that is unclear or insufficient to judge wellbeing relevance"
    ),
}

RELEVANCE_HYPOTHESIS_TEMPLATE: Final[str] = "This text is about {}."

# ---------------------------------------------------------------------------
# Target — exclusive (multi_label=False)
# ---------------------------------------------------------------------------

TARGET_LABELS: Final[tuple[str, ...]] = (
    "self",
    "other_person",
    "group_or_community",
    "institution_or_event",
    "general_or_unknown",
)

TARGET_DEFINITIONS: Final[Mapping[str, str]] = {
    "self": "Expression is substantially about the speaker/author.",
    "other_person": (
        "Expression describes another identifiable or referenced person."
    ),
    "group_or_community": (
        "Expression concerns a group/community rather than specifically the "
        "speaker."
    ),
    "institution_or_event": (
        "Content is mainly about an institution, situation, event, news item, "
        "media content, class, university, etc."
    ),
    "general_or_unknown": "Target cannot be reliably determined.",
}

TARGET_CANDIDATES: Final[Mapping[str, str]] = {
    "self": "the speaker or author themselves",
    "other_person": "another specific person other than the speaker",
    "group_or_community": "a group or community rather than the speaker alone",
    "institution_or_event": (
        "an institution, event, news item, class, university, or similar "
        "situation"
    ),
    "general_or_unknown": "an unclear or undetermined target",
}

TARGET_HYPOTHESIS_TEMPLATE: Final[str] = (
    "The main target of this expression is {}."
)

# ---------------------------------------------------------------------------
# Expressed signals — multi-label (multi_label=True)
# ---------------------------------------------------------------------------

SIGNAL_LABELS: Final[tuple[str, ...]] = (
    "stress_or_overwhelm",
    "anxiety_or_fear_language",
    "loneliness_or_isolation",
    "hopelessness_like_language",
    "exhaustion_or_burnout_like_language",
    "self_directed_negativity",
    "interpersonal_distress",
    "academic_pressure",
    "positive_wellbeing_or_recovery",
)

SIGNAL_DEFINITIONS: Final[Mapping[str, str]] = {
    "stress_or_overwhelm": (
        "Language expressing stress or feeling overwhelmed (content evidence, "
        "not a diagnosis)."
    ),
    "anxiety_or_fear_language": (
        "Language using anxiety- or fear-related wording (content evidence, "
        "not an anxiety disorder)."
    ),
    "loneliness_or_isolation": (
        "Language expressing loneliness or social isolation (content evidence)."
    ),
    "hopelessness_like_language": (
        "Language resembling hopelessness or giving-up wording (content "
        "evidence, not a clinical judgment)."
    ),
    "exhaustion_or_burnout_like_language": (
        "Language resembling exhaustion or burnout (content evidence, not a "
        "diagnosis)."
    ),
    "self_directed_negativity": (
        "Language expressing negative self-evaluation by the author "
        "(content evidence)."
    ),
    "interpersonal_distress": (
        "Language about interpersonal conflict or relationship distress "
        "(content evidence)."
    ),
    "academic_pressure": (
        "Language about academic pressure, exams, coursework, or study load "
        "(content evidence)."
    ),
    "positive_wellbeing_or_recovery": (
        "Language expressing improving wellbeing, coping success, or recovery "
        "(content evidence)."
    ),
}

SIGNAL_CANDIDATES: Final[Mapping[str, str]] = {
    "stress_or_overwhelm": (
        "language expressing stress or feeling overwhelmed"
    ),
    "anxiety_or_fear_language": (
        "language using anxiety-related or fear-related wording"
    ),
    "loneliness_or_isolation": (
        "language expressing loneliness or social isolation"
    ),
    "hopelessness_like_language": (
        "language resembling hopelessness or giving up"
    ),
    "exhaustion_or_burnout_like_language": (
        "language resembling exhaustion or burnout"
    ),
    "self_directed_negativity": (
        "language expressing negative self-evaluation by the author"
    ),
    "interpersonal_distress": (
        "language about interpersonal conflict or relationship distress"
    ),
    "academic_pressure": (
        "language about academic pressure, exams, coursework, or study load"
    ),
    "positive_wellbeing_or_recovery": (
        "language expressing improving wellbeing, coping success, or recovery"
    ),
}

SIGNAL_HYPOTHESIS_TEMPLATE: Final[str] = "This text contains {}."

# Forbidden diagnosis-style identifiers (must never appear as classifier labels).
FORBIDDEN_DIAGNOSIS_LABELS: Final[frozenset[str]] = frozenset(
    {
        "depressed",
        "depression",
        "mentally_ill",
        "suicidal_person",
        "anxiety_disorder",
        "bipolar",
        "PTSD",
        "ptsd",
    },
)


def candidate_list(candidates: Mapping[str, str], labels: tuple[str, ...]) -> list[str]:
    """Ordered list of human-readable candidate strings for a taxonomy."""
    return [candidates[label] for label in labels]


def verbalization_to_label(
    verbalization: str,
    candidates: Mapping[str, str],
) -> str:
    """Map a pipeline candidate string back to its taxonomy label id."""
    normalized = verbalization.strip()
    for label, text in candidates.items():
        if text == normalized:
            return label
    raise KeyError(f"Unknown candidate verbalization: {verbalization!r}")
