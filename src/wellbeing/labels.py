"""Central taxonomy and zero-shot hypotheses for the wellbeing classifier.

All candidate verbalizations and hypothesis templates live here.
Do not scatter hidden prompts across other modules.

Labels describe LANGUAGE / EVIDENCE IN CONTENT — not diagnoses,
not clinical conditions, and not inferred internal mental states.

Phase 4A.2: verbalizations rewritten for clearer personal vs topic,
self vs other, and explicit-evidence signal semantics.
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
        "The author is explicitly describing their own current or recent "
        "wellbeing, stress, emotional difficulty, coping, or recovery."
    ),
    "wellbeing_topic_only": (
        "The text discusses wellbeing, stress, mental health, coping, or "
        "another person's wellbeing without clearly describing the author's "
        "own wellbeing."
    ),
    "not_wellbeing_related": (
        "The text is not meaningfully about a person's wellbeing, stress, "
        "coping, recovery, or emotional difficulty."
    ),
    "ambiguous": (
        "It is unclear whether the text describes the author's personal "
        "wellbeing rather than figurative, casual, quoted, or general "
        "discussion."
    ),
}

# Human-readable NLI candidates (NOT raw label IDs).
RELEVANCE_CANDIDATES: Final[Mapping[str, str]] = {
    "personal_wellbeing": (
        "the author explicitly describing their own current or recent "
        "wellbeing, stress, emotional difficulty, coping, or recovery"
    ),
    "wellbeing_topic_only": (
        "a discussion of wellbeing, stress, mental health, coping, or "
        "another person's wellbeing without clearly describing the author's "
        "own wellbeing"
    ),
    "not_wellbeing_related": (
        "text that is not meaningfully about a person's wellbeing, stress, "
        "coping, recovery, or emotional difficulty"
    ),
    "ambiguous": (
        "unclear whether the text describes the author's personal wellbeing "
        "rather than figurative, casual, quoted, or general discussion"
    ),
}

RELEVANCE_HYPOTHESIS_TEMPLATE: Final[str] = "This text is best described as: {}."

# ---------------------------------------------------------------------------
# Target — exclusive (multi_label=False)
# Whose wellbeing experience is being described?
# ---------------------------------------------------------------------------

TARGET_LABELS: Final[tuple[str, ...]] = (
    "self",
    "other_person",
    "group_or_community",
    "institution_or_event",
    "general_or_unknown",
)

TARGET_DEFINITIONS: Final[Mapping[str, str]] = {
    "self": (
        "The wellbeing experience being described belongs primarily to the "
        "author or speaker."
    ),
    "other_person": (
        "The wellbeing experience being described belongs primarily to "
        "another person, not the author or speaker."
    ),
    "group_or_community": (
        "The wellbeing experience being described belongs primarily to a "
        "group or community."
    ),
    "institution_or_event": (
        "The text mainly discusses an institution, event, situation, class, "
        "exam, or topic and does not primarily describe a person's wellbeing "
        "experience."
    ),
    "general_or_unknown": (
        "No clear person or group can reliably be identified as the subject "
        "of the wellbeing experience."
    ),
}

TARGET_CANDIDATES: Final[Mapping[str, str]] = {
    "self": (
        "a wellbeing experience that belongs primarily to the author or "
        "speaker themselves"
    ),
    "other_person": (
        "a wellbeing experience that belongs primarily to another person, "
        "not the author or speaker"
    ),
    "group_or_community": (
        "a wellbeing experience that belongs primarily to a group or "
        "community"
    ),
    "institution_or_event": (
        "mainly an institution, event, situation, class, exam, or topic "
        "without primarily describing a person's wellbeing experience"
    ),
    "general_or_unknown": (
        "no clear person or group as the subject of a wellbeing experience"
    ),
}

TARGET_HYPOTHESIS_TEMPLATE: Final[str] = (
    "Regarding whose wellbeing experience is being described, this text is "
    "best described as: {}."
)

# ---------------------------------------------------------------------------
# Expressed signals — multi-label (multi_label=True)
# Require EXPLICIT expressed evidence; not generic negativity.
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
        "The author explicitly describes feeling stressed, overwhelmed, "
        "under pressure, or unable to cope (content evidence, not a diagnosis)."
    ),
    "anxiety_or_fear_language": (
        "The author explicitly describes feeling afraid, worried, anxious, "
        "panicked, or fearful (content evidence, not an anxiety disorder)."
    ),
    "loneliness_or_isolation": (
        "The author explicitly describes feeling lonely, isolated, excluded, "
        "or without support (content evidence)."
    ),
    "hopelessness_like_language": (
        "The author explicitly expresses hopelessness, giving up, or a belief "
        "that things will not improve about their own experience (content "
        "evidence, not a clinical judgment)."
    ),
    "exhaustion_or_burnout_like_language": (
        "The author explicitly describes feeling exhausted, burned out, "
        "drained, or unable to continue because of fatigue (content evidence)."
    ),
    "self_directed_negativity": (
        "The author explicitly criticizes or devalues themself, their worth, "
        "their abilities, or their identity (not generic dislike of media or "
        "objects)."
    ),
    "interpersonal_distress": (
        "The author explicitly describes personal distress caused by "
        "conflict, rejection, breakup, bullying, or relationship problems."
    ),
    "academic_pressure": (
        "The author explicitly describes their own stress or pressure related "
        "to exams, grades, assignments, deadlines, studies, or academic "
        "workload."
    ),
    "positive_wellbeing_or_recovery": (
        "The author explicitly describes their wellbeing improving, coping "
        "better, recovering, resting, or feeling more stable."
    ),
}

SIGNAL_CANDIDATES: Final[Mapping[str, str]] = {
    "stress_or_overwhelm": (
        "the author explicitly describing feeling stressed, overwhelmed, "
        "under pressure, or unable to cope"
    ),
    "anxiety_or_fear_language": (
        "the author explicitly describing feeling afraid, worried, anxious, "
        "panicked, or fearful"
    ),
    "loneliness_or_isolation": (
        "the author explicitly describing feeling lonely, isolated, "
        "excluded, or without support"
    ),
    "hopelessness_like_language": (
        "the author explicitly expressing hopelessness, giving up, or a "
        "belief that things will not improve about their own experience"
    ),
    "exhaustion_or_burnout_like_language": (
        "the author explicitly describing feeling exhausted, burned out, "
        "drained, or unable to continue because of fatigue"
    ),
    "self_directed_negativity": (
        "the author explicitly criticizing or devaluing themself, their "
        "worth, their abilities, or their identity"
    ),
    "interpersonal_distress": (
        "the author explicitly describing personal distress caused by "
        "conflict, rejection, breakup, bullying, or relationship problems"
    ),
    "academic_pressure": (
        "the author explicitly describing their own stress or pressure "
        "related to exams, grades, assignments, deadlines, studies, or "
        "academic workload"
    ),
    "positive_wellbeing_or_recovery": (
        "the author explicitly describing their wellbeing improving, coping "
        "better, recovering, resting, or feeling more stable"
    ),
}

SIGNAL_HYPOTHESIS_TEMPLATE: Final[str] = "This text contains evidence of {}."

# ---------------------------------------------------------------------------
# Dual-head attribution evidence (Phase 4A.5) — independent multi_label=True
# Run ONLY when relevance==personal_wellbeing AND target==self.
#
# Two independent semantic questions (not forced exclusive):
# A) direct_self_experience
# B) reported_other_experience
#
# Policy derives final_label: self_experience | not_self_experience | unclear.
# ---------------------------------------------------------------------------

DUAL_ATTRIBUTION_LABELS: Final[tuple[str, ...]] = (
    "direct_self_experience",
    "reported_other_experience",
)

DUAL_ATTRIBUTION_DEFINITIONS: Final[Mapping[str, str]] = {
    "direct_self_experience": (
        "The author or speaker is directly describing their own wellbeing, "
        "stress, coping, emotional difficulty, or recovery."
    ),
    "reported_other_experience": (
        "The text reports, quotes, paraphrases, or describes another person's "
        "wellbeing experience rather than the author's own experience."
    ),
}

DUAL_ATTRIBUTION_CANDIDATES: Final[Mapping[str, str]] = {
    "direct_self_experience": (
        "the author or speaker directly describing their own wellbeing, "
        "stress, coping, emotional difficulty, or recovery"
    ),
    "reported_other_experience": (
        "the text reporting, quoting, paraphrasing, or describing another "
        "person's wellbeing experience rather than the author's own experience"
    ),
}

DUAL_ATTRIBUTION_HYPOTHESIS_TEMPLATE: Final[str] = "This text contains evidence of {}."

# Independent entailment-style evidence (not forced self vs not-self).
DUAL_ATTRIBUTION_MULTI_LABEL: Final[bool] = True

# Final/policy attribution outcomes (may include policy-derived unclear).
FINAL_ATTRIBUTION_LABELS: Final[tuple[str, ...]] = (
    "self_experience",
    "not_self_experience",
    "unclear",
)

# ---------------------------------------------------------------------------
# LEGACY binary attribution (Phase 4A.4) — evaluation/comparison only
# Kept for latency / quality comparison. Must NOT drive production eligibility.
# ---------------------------------------------------------------------------

RAW_ATTRIBUTION_LABELS: Final[tuple[str, ...]] = (
    "self_experience",
    "not_self_experience",
)

# Back-compat alias.
ATTRIBUTION_LABELS: Final[tuple[str, ...]] = RAW_ATTRIBUTION_LABELS

ATTRIBUTION_DEFINITIONS: Final[Mapping[str, str]] = {
    "self_experience": (
        "The person experiencing the wellbeing state described in the text "
        "is the author or speaker themself."
    ),
    "not_self_experience": (
        "The person experiencing the wellbeing state described in the text "
        "is someone other than the author or speaker."
    ),
}

ATTRIBUTION_CANDIDATES: Final[Mapping[str, str]] = {
    "self_experience": (
        "the person experiencing the wellbeing state described in the text "
        "is the author or speaker themself"
    ),
    "not_self_experience": (
        "the person experiencing the wellbeing state described in the text "
        "is someone other than the author or speaker"
    ),
}

ATTRIBUTION_HYPOTHESIS_TEMPLATE: Final[str] = (
    "Regarding who is experiencing the described wellbeing state, {}."
)

# Legacy exclusive binary mode (comparison only).
ATTRIBUTION_MULTI_LABEL: Final[bool] = False

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
