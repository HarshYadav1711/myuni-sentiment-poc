"""Previous 40-case evaluation set — now DEVELOPMENT / REGRESSION only (Phase 4A.5).

Originally created as an untouched holdout for Phase 4A.4. Its failures
(especially quoted first-person speech false eligibility) influenced the
dual-head attribution redesign, so it is NO LONGER an unbiased holdout.

DATASET ROLE NOW:
- REGRESSION / DEVELOPMENT diagnostics
- Verify fh_quote_01 / fh_quote_03 are no longer falsely eligible
- Do NOT use for threshold fitting after Phase 4A.5 starts

Fresh untouched evaluation: evaluation/wellbeing/final_holdout_v2.py
"""

from __future__ import annotations

from typing import Any

# Alias retained for import compatibility; role is regression, not holdout.
WELLBEING_FINAL_HOLDOUT_CASES: list[dict[str, Any]] = [
    # ----- clear first-person stress / academic / exhaustion -----
    {
        "id": "fh_fp_01",
        "bucket": "clear_first_person",
        "text": "I can't stop worrying about failing this module and it's wrecking my sleep.",
        "expected_personal_eligibility": "eligible",
    },
    {
        "id": "fh_fp_02",
        "bucket": "clear_first_person",
        "text": "Right now I feel crushed by my dissertation deadlines.",
        "expected_personal_eligibility": "eligible",
    },
    {
        "id": "fh_fp_03",
        "bucket": "clear_first_person",
        "text": "I am so tired from night shifts and studying that I feel empty.",
        "expected_personal_eligibility": "eligible",
    },
    {
        "id": "fh_fp_04",
        "bucket": "clear_first_person",
        "text": "I feel under constant academic pressure from my GPA targets.",
        "expected_personal_eligibility": "eligible",
    },
    {
        "id": "fh_fp_05",
        "bucket": "clear_first_person",
        "text": "I'm scared I'll never catch up with the reading list this term.",
        "expected_personal_eligibility": "eligible",
    },
    {
        "id": "fh_fp_06",
        "bucket": "clear_first_person",
        "text": "After the breakup I feel rejected and can't concentrate in lectures.",
        "expected_personal_eligibility": "eligible",
    },
    {
        "id": "fh_fp_07",
        "bucket": "clear_first_person",
        "text": "I keep blaming myself for every mistake and it hurts.",
        "expected_personal_eligibility": "eligible",
    },
    {
        "id": "fh_fp_08",
        "bucket": "clear_first_person",
        "text": "I feel isolated in halls and like nobody notices me.",
        "expected_personal_eligibility": "eligible",
    },
    # ----- clear first-person recovery -----
    {
        "id": "fh_rec_01",
        "bucket": "clear_first_person",
        "text": "I started walking each morning and I feel steadier emotionally.",
        "expected_personal_eligibility": "eligible",
    },
    {
        "id": "fh_rec_02",
        "bucket": "clear_first_person",
        "text": "Therapy has helped and I feel more hopeful about next semester.",
        "expected_personal_eligibility": "eligible",
    },
    {
        "id": "fh_rec_03",
        "bucket": "clear_first_person",
        "text": "I rested properly over the weekend and I feel restored.",
        "expected_personal_eligibility": "eligible",
    },
    # ----- clear third-person -----
    {
        "id": "fh_tp_01",
        "bucket": "clear_third_person",
        "text": "My flatmate says the labs have left her completely drained.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh_tp_02",
        "bucket": "clear_third_person",
        "text": "Alex told me he feels hopeless about his internship interviews.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh_tp_03",
        "bucket": "clear_third_person",
        "text": "My cousin has been anxious about moving abroad alone.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh_tp_04",
        "bucket": "clear_third_person",
        "text": "Her boyfriend keeps saying he feels overwhelmed by rent and fees.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh_tp_05",
        "bucket": "clear_third_person",
        "text": "The TA mentioned her students seem burned out before viva week.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh_tp_06",
        "bucket": "clear_third_person",
        "text": "My dad said he feels lonely now that I've left for university.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh_tp_07",
        "bucket": "clear_third_person",
        "text": "A teammate admitted she feels excluded from the project chats.",
        "expected_personal_eligibility": "not_eligible",
    },
    # ----- quoted / reported speech -----
    {
        "id": "fh_quote_01",
        "bucket": "clear_third_person",
        "text": 'He texted me: "I feel completely lost and can\'t cope with uni."',
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh_quote_02",
        "bucket": "clear_third_person",
        "text": "Someone shared a screenshot where they wrote they feel worthless.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh_quote_03",
        "bucket": "clear_third_person",
        "text": 'In the novel the narrator says, "I am exhausted beyond belief."',
        "expected_personal_eligibility": "not_eligible",
    },
    # ----- topic-only / institutional -----
    {
        "id": "fh_topic_01",
        "bucket": "topic_only",
        "text": "Student services published a guide on recognising academic burnout.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh_topic_02",
        "bucket": "topic_only",
        "text": "The faculty is launching a peer-support scheme for emotional wellbeing.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh_topic_03",
        "bucket": "topic_only",
        "text": "Researchers compared stress questionnaires across three universities.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh_topic_04",
        "bucket": "topic_only",
        "text": "Tonight's panel discusses how institutions can support student resilience.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh_topic_05",
        "bucket": "topic_only",
        "text": "A podcast episode covered common myths about campus counselling.",
        "expected_personal_eligibility": "not_eligible",
    },
    # ----- figurative / general negativity -----
    {
        "id": "fh_fig_01",
        "bucket": "figurative_ambiguous",
        "text": "This group project is absolute chaos and I'm done with it.",
        "expected_personal_eligibility": "uncertain",
    },
    {
        "id": "fh_fig_02",
        "bucket": "figurative_ambiguous",
        "text": "That thriller was bleak from start to finish.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh_fig_03",
        "bucket": "figurative_ambiguous",
        "text": "I hate how slow this printer is.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh_fig_04",
        "bucket": "figurative_ambiguous",
        "text": "We got demolished in the debate tournament lol.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh_fig_05",
        "bucket": "figurative_ambiguous",
        "text": "Brain.exe has stopped working after that 9am lecture.",
        "expected_personal_eligibility": "uncertain",
    },
    # ----- mixed / ambiguous authorship -----
    {
        "id": "fh_mix_01",
        "bucket": "figurative_ambiguous",
        "text": "My sister is panicked about fees and it's starting to stress me out too.",
        "expected_personal_eligibility": "uncertain",
    },
    {
        "id": "fh_mix_02",
        "bucket": "figurative_ambiguous",
        "text": "The whole lab seems exhausted; I might be absorbing that vibe.",
        "expected_personal_eligibility": "uncertain",
    },
    {
        "id": "fh_mix_03",
        "bucket": "figurative_ambiguous",
        "text": "Someone said exams ruin everyone's mood, including mine maybe.",
        "expected_personal_eligibility": "uncertain",
    },
    {
        "id": "fh_amb_01",
        "bucket": "figurative_ambiguous",
        "text": "Feeling the weight of deadlines again.",
        "expected_personal_eligibility": "uncertain",
    },
    {
        "id": "fh_amb_02",
        "bucket": "figurative_ambiguous",
        "text": "Not sure if this is burnout or just a rough week.",
        "expected_personal_eligibility": "uncertain",
    },
    # ----- group / community -----
    {
        "id": "fh_group_01",
        "bucket": "topic_only",
        "text": "Our cohort collectively feels the strain of continuous assessment.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh_group_02",
        "bucket": "topic_only",
        "text": "International students often discuss homesickness in the society meetings.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh_group_03",
        "bucket": "topic_only",
        "text": "The union survey found many members report sleep problems during exams.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fh_fp_09",
        "bucket": "clear_first_person",
        "text": "I finally asked for an extension and I feel less cornered than yesterday.",
        "expected_personal_eligibility": "eligible",
    },
]


def final_holdout_distribution() -> dict[str, Any]:
    elig: dict[str, int] = {"eligible": 0, "not_eligible": 0, "uncertain": 0}
    buckets: dict[str, int] = {}
    for case in WELLBEING_FINAL_HOLDOUT_CASES:
        elig[str(case["expected_personal_eligibility"])] += 1
        b = str(case.get("bucket", "other"))
        buckets[b] = buckets.get(b, 0) + 1
    return {"eligibility": elig, "buckets": buckets, "n": len(WELLBEING_FINAL_HOLDOUT_CASES)}
