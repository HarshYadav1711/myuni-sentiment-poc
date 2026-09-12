"""Human-readable semantic fixtures for later real-model evaluation.

These capture HUMAN EXPECTATIONS for Phase 4 calibration / review.
Ordinary mocked unit tests must NOT assert that the zero-shot POC model
already classifies every example correctly.
"""

from __future__ import annotations

from typing import Any

# Expected annotations are human judgment for later evaluation only.
WELLBEING_SEMANTIC_FIXTURES: list[dict[str, Any]] = [
    {
        "id": "A_exams_overwhelmed",
        "text": (
            "I have three exams this week and I feel completely overwhelmed."
        ),
        "expected_human": {
            "relevance": "personal_wellbeing",
            "target": "self",
            "signals_include": ["stress_or_overwhelm", "academic_pressure"],
        },
        "notes": "Clear first-person academic stress / overwhelm.",
    },
    {
        "id": "B_friend_stressed",
        "text": "My friend has been really stressed lately.",
        "expected_human": {
            "relevance": "wellbeing_topic_only",
            "target": "other_person",
            "signals_include": ["stress_or_overwhelm"],
            "target_not": ["self"],
        },
        "notes": "Wellbeing-related but about another person — not self.",
    },
    {
        "id": "C_university_seminar",
        "text": (
            "Our university is hosting a mental health awareness seminar."
        ),
        "expected_human": {
            "relevance": "wellbeing_topic_only",
            "target": "institution_or_event",
        },
        "notes": "Topic/institutional announcement, not personal state.",
    },
    {
        "id": "D_movie_depressing",
        "text": "This movie was depressing.",
        "expected_human": {
            "relevance": "not_wellbeing_related",
            "notes_detail": (
                "Negative media commentary is not necessarily personal "
                "wellbeing."
            ),
        },
        "notes": "Media evaluation; do not treat as personal wellbeing.",
    },
    {
        "id": "E_recovery_on_track",
        "text": (
            "I've been doing better lately and finally feel like I'm getting "
            "back on track."
        ),
        "expected_human": {
            "relevance": "personal_wellbeing",
            "target": "self",
            "signals_include": ["positive_wellbeing_or_recovery"],
        },
        "notes": "First-person recovery / improving wellbeing language.",
    },
    {
        "id": "F_exams_stressful_group",
        "text": "Everyone keeps saying exams are stressful lol.",
        "expected_human": {
            "relevance_candidates": ["ambiguous", "wellbeing_topic_only"],
            "target_candidates": ["group_or_community", "general_or_unknown"],
            "do_not_auto_treat_as": "self distress",
        },
        "notes": (
            "Potential ambiguity / group context — do not automatically "
            "treat as self distress."
        ),
    },
]
