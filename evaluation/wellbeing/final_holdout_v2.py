"""Fresh final evaluation set V2 for Phase 4A.5 (UNTOUCHED HOLDOUT).

Created and human-annotated BEFORE frozen dual-head policy evaluation.

ENGINEERING EXAMPLES ONLY — NOT real student data, NOT clinical labels.

DO NOT reuse:
- A–K fixture wording
- calibration_cases.py texts
- previous final_holdout_cases.py (FH40) texts
- attribution_dev_cases.py texts

DO NOT use this file to choose thresholds.
Evaluate EXACTLY ONCE after policy freeze.
"""

from __future__ import annotations

from typing import Any

WELLBEING_FINAL_HOLDOUT_V2_CASES: list[dict[str, Any]] = [
    # ----- ~15 clear first-person -----
    {
        "id": "fh2_fp_01",
        "bucket": "clear_first_person",
        "text": "I feel stretched thin by my coursework and it's affecting my sleep.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fh2_fp_02",
        "bucket": "clear_first_person",
        "text": "Right now I am panicking about tomorrow's presentation.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fh2_fp_03",
        "bucket": "clear_first_person",
        "text": "I feel empty after weeks of late-night revising without a break.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fh2_fp_04",
        "bucket": "clear_first_person",
        "text": "My GPA targets are making me feel constantly on edge this term.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fh2_fp_05",
        "bucket": "clear_first_person",
        "text": "I'm afraid I won't finish the portfolio and that fear won't leave.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fh2_fp_06",
        "bucket": "clear_first_person",
        "text": "After the argument I feel rejected and can't focus in seminars.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fh2_fp_07",
        "bucket": "clear_first_person",
        "text": "I keep calling myself a failure and it is wearing me down.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fh2_fp_08",
        "bucket": "clear_first_person",
        "text": "I feel cut off in my accommodation and have barely spoken to anyone.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fh2_fp_09",
        "bucket": "clear_first_person",
        "text": "I feel hopeless about catching up after missing two weeks of labs.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fh2_fp_10",
        "bucket": "clear_first_person",
        "text": "I am burned out from juggling a part-time job and three modules.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fh2_fp_11",
        "bucket": "clear_first_person",
        "text": "I started journaling each evening and I feel more grounded now.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fh2_fp_12",
        "bucket": "clear_first_person",
        "text": "Counselling sessions helped and I feel more hopeful about exams.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fh2_fp_13",
        "bucket": "clear_first_person",
        "text": "I slept through the weekend and I feel restored for Monday.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fh2_fp_14",
        "bucket": "clear_first_person",
        "text": "I asked for help early and I feel less cornered than last week.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fh2_fp_15",
        "bucket": "clear_first_person",
        "text": "I am coping better with deadlines and feel proud of that progress.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    # ----- ~10 clear third-person -----
    {
        "id": "fh2_tp_01",
        "bucket": "clear_third_person",
        "text": "My housemate says continuous assessment has left her wiped out.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "fh2_tp_02",
        "bucket": "clear_third_person",
        "text": "Priya told me she feels hopeless about securing a placement.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "fh2_tp_03",
        "bucket": "clear_third_person",
        "text": "My cousin has been anxious about starting a new degree abroad.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "fh2_tp_04",
        "bucket": "clear_third_person",
        "text": "His partner keeps saying he feels crushed by rent and tuition.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "fh2_tp_05",
        "bucket": "clear_third_person",
        "text": "The demonstrator mentioned her cohort looks exhausted before viva.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "fh2_tp_06",
        "bucket": "clear_third_person",
        "text": "My uncle said he feels lonely now that the house is empty.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "fh2_tp_07",
        "bucket": "clear_third_person",
        "text": "A project partner admitted she feels left out of design meetings.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "fh2_tp_08",
        "bucket": "clear_third_person",
        "text": "My neighbour keeps describing how stressed her daughter is at uni.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "fh2_tp_09",
        "bucket": "clear_third_person",
        "text": "The coach said several athletes feel drained after double training.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "fh2_tp_10",
        "bucket": "clear_third_person",
        "text": "My sibling is struggling with homesickness after moving halls.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    # ----- ~8 quoted / reported other -----
    {
        "id": "fh2_qo_01",
        "bucket": "quoted_reported_other",
        "text": 'He messaged me: "I feel completely lost and can\'t handle seminars."',
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "fh2_qo_02",
        "bucket": "quoted_reported_other",
        "text": "Someone forwarded a note where they wrote they feel worthless lately.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "fh2_qo_03",
        "bucket": "quoted_reported_other",
        "text": 'In the memoir the narrator writes, "I am exhausted beyond words."',
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "fh2_qo_04",
        "bucket": "quoted_reported_other",
        "text": "My classmate said, 'I can't cope with the resit timetable.'",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "fh2_qo_05",
        "bucket": "quoted_reported_other",
        "text": "She posted a story: \"I'm overwhelmed by every deadline.\"",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "fh2_qo_06",
        "bucket": "quoted_reported_other",
        "text": "He kept repeating, 'I feel burned out,' during the call.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "fh2_qo_07",
        "bucket": "quoted_reported_other",
        "text": "According to her email, she feels anxious opening assignment feedback.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "fh2_qo_08",
        "bucket": "quoted_reported_other",
        "text": "They shared a screenshot saying 'I feel hopeless about next term.'",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    # ----- ~4 quoted self -----
    {
        "id": "fh2_qs_01",
        "bucket": "quoted_self",
        "text": "I told my advisor, 'I'm completely overwhelmed by the timetable.'",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fh2_qs_02",
        "bucket": "quoted_self",
        "text": "I emailed support: \"I can't cope with the workload this month.\"",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fh2_qs_03",
        "bucket": "quoted_self",
        "text": "I posted, 'I feel steadier after taking two rest days.'",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fh2_qs_04",
        "bucket": "quoted_self",
        "text": "I admitted to a friend, \"I've been lonely since changing courses.\"",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    # ----- ~5 topic-only -----
    {
        "id": "fh2_topic_01",
        "bucket": "topic_only",
        "text": "Student wellbeing services released a checklist for recognising burnout.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh2_topic_02",
        "bucket": "topic_only",
        "text": "The faculty is piloting a peer-listening scheme for emotional support.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh2_topic_03",
        "bucket": "topic_only",
        "text": "Researchers compared anxiety inventories across four campus sites.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh2_topic_04",
        "bucket": "topic_only",
        "text": "Tonight's briefing covers how departments can improve student resilience.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh2_topic_05",
        "bucket": "topic_only",
        "text": "A newsletter summarised myths about university counselling wait times.",
        "expected_personal_eligibility": "not_eligible",
    },
    # ----- ~4 figurative -----
    {
        "id": "fh2_fig_01",
        "bucket": "figurative",
        "text": "This lab report is absolute chaos and I'm done with the formatting.",
        "expected_personal_eligibility": "uncertain",
    },
    {
        "id": "fh2_fig_02",
        "bucket": "figurative",
        "text": "That documentary was bleak from the opening scene.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh2_fig_03",
        "bucket": "figurative",
        "text": "I hate how unreliable this campus Wi-Fi is.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh2_fig_04",
        "bucket": "figurative",
        "text": "We got crushed in the quiz bowl final lol.",
        "expected_personal_eligibility": "not_eligible",
    },
    # ----- ~4 mixed / ambiguous -----
    {
        "id": "fh2_mix_01",
        "bucket": "mixed_ambiguous",
        "text": "My brother is panicked about fees and it's starting to stress me out too.",
        "expected_personal_eligibility": "uncertain",
    },
    {
        "id": "fh2_mix_02",
        "bucket": "mixed_ambiguous",
        "text": "The whole studio seems exhausted; I might be picking up that mood.",
        "expected_personal_eligibility": "uncertain",
    },
    {
        "id": "fh2_mix_03",
        "bucket": "mixed_ambiguous",
        "text": "Someone said assessments ruin everyone's mood, including mine maybe.",
        "expected_personal_eligibility": "uncertain",
    },
    {
        "id": "fh2_mix_04",
        "bucket": "mixed_ambiguous",
        "text": "Not sure if this is burnout talking or just a rough fortnight.",
        "expected_personal_eligibility": "uncertain",
    },
]


def final_holdout_v2_distribution() -> dict[str, Any]:
    elig: dict[str, int] = {"eligible": 0, "not_eligible": 0, "uncertain": 0}
    buckets: dict[str, int] = {}
    for case in WELLBEING_FINAL_HOLDOUT_V2_CASES:
        elig[str(case["expected_personal_eligibility"])] += 1
        b = str(case.get("bucket", "other"))
        buckets[b] = buckets.get(b, 0) + 1
    return {
        "eligibility": elig,
        "buckets": buckets,
        "n": len(WELLBEING_FINAL_HOLDOUT_V2_CASES),
    }
