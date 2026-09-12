"""Attribution-focused DEVELOPMENT cases (Phase 4A.5).

ENGINEERING EXAMPLES ONLY — NOT real student data. NOT clinical annotations.

DATASET ROLE:
- DEVELOPMENT / CALIBRATION coverage for dual-head attribution.
- Used with calibration_cases + A–K + previous FH40 for policy search.
- NOT a holdout. Do not call this set holdout.

Covers: direct self, direct other, quoted other, quoted self, indirect
reported speech, mixed self+other, topic discussion, figurative content.
"""

from __future__ import annotations

from typing import Any, Literal

ExpectedEligibility = Literal["eligible", "not_eligible", "uncertain"]

WELLBEING_ATTRIBUTION_DEV_CASES: list[dict[str, Any]] = [
    # ----- direct self -----
    {
        "id": "attr_self_01",
        "category": "direct_self",
        "text": "I am struggling to manage my stress before this week's viva.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "attr_self_02",
        "category": "direct_self",
        "text": "Honestly I feel worn down and anxious about catching up on labs.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "attr_self_03",
        "category": "direct_self",
        "text": "I told myself I would rest, and today I finally feel calmer.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "attr_self_04",
        "category": "direct_self",
        "text": "I'm finding it hard to cope with the constant assessment load.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    # ----- quoted self (author quoting their own words) -----
    {
        "id": "attr_quoted_self_01",
        "category": "quoted_self",
        "text": "I told my friend, 'I'm completely overwhelmed.'",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "attr_quoted_self_02",
        "category": "quoted_self",
        "text": "I texted my tutor: \"I can't cope with the deadlines this week.\"",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "attr_quoted_self_03",
        "category": "quoted_self",
        "text": "I posted, 'I feel much steadier after taking a break.'",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "attr_quoted_self_04",
        "category": "quoted_self",
        "text": "I admitted to my mentor, \"I've been lonely since moving halls.\"",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    # ----- quoted other / reported speech -----
    {
        "id": "attr_quoted_other_01",
        "category": "quoted_other",
        "text": "My roommate said, 'I'm completely exhausted.'",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "attr_quoted_other_02",
        "category": "quoted_other",
        "text": "My friend told me, 'I can't cope with exams anymore.'",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "attr_quoted_other_03",
        "category": "quoted_other",
        "text": "She said, 'I feel much better now.'",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "attr_quoted_other_04",
        "category": "quoted_other",
        "text": "He posted, 'I'm overwhelmed.'",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "attr_quoted_other_05",
        "category": "quoted_other",
        "text": "Jordan messaged me: \"I feel hopeless about resits.\"",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "attr_quoted_other_06",
        "category": "quoted_other",
        "text": "In the group chat Sam wrote, 'I'm burned out and need a week off.'",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    # ----- indirect reported / direct other -----
    {
        "id": "attr_other_01",
        "category": "direct_other",
        "text": "My roommate told me she feels completely overwhelmed by exams lately.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "attr_other_02",
        "category": "direct_other",
        "text": "My friend said the labs have left him drained and anxious.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "attr_other_03",
        "category": "indirect_reported",
        "text": "He told me he can't sleep because of the dissertation pressure.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "attr_other_04",
        "category": "indirect_reported",
        "text": "Apparently she feels isolated after switching courses mid-year.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "attr_other_05",
        "category": "indirect_reported",
        "text": "They mentioned feeling rejected after the group project reshuffle.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    # ----- mixed self + other -----
    {
        "id": "attr_mixed_01",
        "category": "mixed_self_other",
        "text": (
            "My roommate has been overwhelmed lately and honestly I'm starting "
            "to feel stressed too."
        ),
        "expected_personal_eligibility": "uncertain",
        "notes": (
            "Contains legitimate self-expression AND other evidence; "
            "unclear or self may be acceptable; do not auto-block solely "
            "because other evidence is present."
        ),
    },
    {
        "id": "attr_mixed_02",
        "category": "mixed_self_other",
        "text": (
            "My flatmate keeps panicking about grades, and watching that is "
            "making me anxious about my own progress."
        ),
        "expected_personal_eligibility": "uncertain",
    },
    {
        "id": "attr_mixed_03",
        "category": "mixed_self_other",
        "text": (
            "She said she's exhausted, and I realised I've been feeling "
            "burned out myself."
        ),
        "expected_personal_eligibility": "uncertain",
    },
    {
        "id": "attr_mixed_04",
        "category": "mixed_self_other",
        "text": (
            "My friend is lonely in halls; I feel lonely too after moving cities."
        ),
        "expected_personal_eligibility": "uncertain",
    },
    # ----- topic-only -----
    {
        "id": "attr_topic_01",
        "category": "topic_discussion",
        "text": "Campus radio covered student burnout statistics for the semester.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "attr_topic_02",
        "category": "topic_discussion",
        "text": "The handbook explains common signs of academic stress for first-years.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "attr_topic_03",
        "category": "topic_discussion",
        "text": "Tomorrow's workshop is about coping strategies and peer support.",
        "expected_personal_eligibility": "not_eligible",
    },
    # ----- figurative -----
    {
        "id": "attr_fig_01",
        "category": "figurative",
        "text": "This spreadsheet is destroying my will to live lol.",
        "expected_personal_eligibility": "uncertain",
    },
    {
        "id": "attr_fig_02",
        "category": "figurative",
        "text": "That TV finale was emotionally exhausting to watch.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "attr_fig_03",
        "category": "figurative",
        "text": "I hate how freezing the library AC is.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "attr_fig_04",
        "category": "figurative",
        "text": "My brain melted during that three-hour seminar meme session.",
        "expected_personal_eligibility": "uncertain",
    },
]


def attribution_dev_eligibility_distribution() -> dict[str, int]:
    counts: dict[str, int] = {
        "eligible": 0,
        "not_eligible": 0,
        "uncertain": 0,
    }
    for case in WELLBEING_ATTRIBUTION_DEV_CASES:
        key = str(case["expected_personal_eligibility"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def attribution_dev_category_distribution() -> dict[str, int]:
    out: dict[str, int] = {}
    for case in WELLBEING_ATTRIBUTION_DEV_CASES:
        cat = str(case.get("category", "other"))
        out[cat] = out.get(cat, 0) + 1
    return out
