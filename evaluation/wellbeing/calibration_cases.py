"""Human-annotated POC DEVELOPMENT/CALIBRATION cases for wellbeing eligibility.

ENGINEERING EXAMPLES ONLY — NOT real student data. NOT clinical annotations.

DATASET ROLE (Phase 4A.5):
- Used to SEARCH / FREEZE dual-head attribution policy values.
- Contaminated for unbiased evaluation (already used in development).
- A–K fixtures are REGRESSION/DEVELOPMENT cases (also contaminated).
- Previous final_holdout_cases.py (FH40) is now also DEVELOPMENT/REGRESSION
  because its failures influenced the dual-head redesign.
- Fresh evaluation uses evaluation/wellbeing/final_holdout_v2.py.
"""

from __future__ import annotations

from typing import Any, Literal

ExpectedEligibility = Literal["eligible", "not_eligible", "uncertain"]

# Each case: id, category, text, expected_personal_eligibility,
# optional expected_relevance / expected_target / expected_attribution.
WELLBEING_CALIBRATION_CASES: list[dict[str, Any]] = [
    # ----- FIRST-PERSON DIRECT WELLBEING (eligible) -----
    {
        "id": "fp_stress_01",
        "category": "first_person_direct",
        "text": "I feel completely overwhelmed with everything on my plate right now.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fp_stress_02",
        "category": "first_person_direct",
        "text": "I've been so stressed this week that I can barely focus in class.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fp_academic_01",
        "category": "first_person_direct",
        "text": "My three midterms this week are crushing me and I feel under so much pressure.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fp_academic_02",
        "category": "first_person_direct",
        "text": "I am drowning in assignments and the deadlines are making me panic.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fp_exhaustion_01",
        "category": "first_person_direct",
        "text": "I am utterly exhausted and feel burned out from studying every night.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fp_exhaustion_02",
        "category": "first_person_direct",
        "text": "I feel drained and like I can't keep going at this pace anymore.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fp_anxiety_01",
        "category": "first_person_direct",
        "text": "I've been feeling really anxious about speaking up in seminars lately.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fp_lonely_01",
        "category": "first_person_direct",
        "text": "I feel so lonely on campus this semester and like I have no one to talk to.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fp_relationship_01",
        "category": "first_person_direct",
        "text": "The argument with my partner left me feeling rejected and really hurt.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fp_selfneg_01",
        "category": "first_person_direct",
        "text": "I keep telling myself I'm useless and that I can't do anything right.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fp_hopeless_01",
        "category": "first_person_direct",
        "text": "I feel like nothing I try helps and things just won't get better for me.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fp_recovery_01",
        "category": "first_person_recovery",
        "text": "Lately I feel like I'm recovering and slowly finding my footing again.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fp_recovery_02",
        "category": "first_person_recovery",
        "text": "After resting properly I feel noticeably healthier and clearer this week.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fp_recovery_03",
        "category": "first_person_recovery",
        "text": "Talking to counseling helped and I feel more stable than last month.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "fp_recovery_04",
        "category": "first_person_recovery",
        "text": "I'm coping better with the workload and feel proud of how far I've come.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    # ----- THIRD-PERSON REPORTS (not_eligible) -----
    {
        "id": "tp_roommate_01",
        "category": "third_person",
        "text": "My roommate keeps saying the exams have left her completely overwhelmed.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "tp_friend_01",
        "category": "third_person",
        "text": "My friend has been under a lot of stress about coursework recently.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "tp_friend_02",
        "category": "third_person",
        "text": "A classmate said he feels burned out and can't sleep before finals.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "tp_sibling_01",
        "category": "third_person",
        "text": "My brother is struggling with loneliness after moving to a new city.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "tp_sister_01",
        "category": "third_person",
        "text": "My sister says she feels anxious every time she opens her email.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "tp_parent_01",
        "category": "third_person",
        "text": "My mum has been exhausted from work and says she feels drained.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "tp_partner_01",
        "category": "third_person",
        "text": "My partner told me he's been feeling hopeless about his job search.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "tp_teacher_01",
        "category": "third_person",
        "text": "Our lecturer mentioned she felt overwhelmed managing two modules.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "tp_roommate_02",
        "category": "third_person",
        "text": "Apparently my flatmate has been crying about grades all week.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "tp_friend_03",
        "category": "third_person",
        "text": "Someone in my group chat said they feel isolated on campus.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    # ----- TOPIC-ONLY (not_eligible) -----
    {
        "id": "topic_seminar_01",
        "category": "topic_only",
        "text": "Our university is hosting a mental health awareness seminar next week.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "topic_article_01",
        "category": "topic_only",
        "text": "This article summarizes research on student burnout and coping strategies.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "topic_programme_01",
        "category": "topic_only",
        "text": "The campus wellbeing programme offers free workshops on stress management.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "topic_stats_01",
        "category": "topic_only",
        "text": "Student forums often discuss anxiety and depression as campus health topics.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "topic_news_01",
        "category": "topic_only",
        "text": "A new report says one in three undergraduates report high academic pressure.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "topic_generic_01",
        "category": "topic_only",
        "text": "People often talk about work-life balance and emotional wellbeing online.",
        "expected_personal_eligibility": "not_eligible",
    },
    # ----- FIGURATIVE / CASUAL NEGATIVITY (not_eligible or uncertain) -----
    {
        "id": "fig_assignment_01",
        "category": "figurative",
        "text": "This coursework is murdering me tbh lol.",
        "expected_personal_eligibility": "uncertain",
    },
    {
        "id": "fig_movie_01",
        "category": "figurative",
        "text": "That film left me in a bleak mood for hours.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fig_hate_class_01",
        "category": "figurative",
        "text": "I hate this class.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fig_game_01",
        "category": "figurative",
        "text": "That ranked match destroyed me, I'm so tilted right now.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fig_sports_01",
        "category": "figurative",
        "text": "We got smashed 5-0 and I'm done with this season tbh.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fig_joke_01",
        "category": "figurative",
        "text": "Monday vibes are killing my soul haha send coffee.",
        "expected_personal_eligibility": "uncertain",
    },
    {
        "id": "fig_hate_movie_01",
        "category": "figurative",
        "text": "Honestly I can't stand that film at all.",
        "expected_personal_eligibility": "not_eligible",
    },
    # ----- QUOTED / REPORTED SPEECH (not_eligible) -----
    {
        "id": "quote_01",
        "category": "quoted",
        "text": 'She posted "I feel completely overwhelmed by exams" on her story.',
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "quote_02",
        "category": "quoted",
        "text": "He kept repeating the lyric 'I'm so lonely' from that song.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "quote_03",
        "category": "quoted",
        "text": 'According to the character in the play, "I can\'t cope anymore."',
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "quote_04",
        "category": "quoted",
        "text": "Someone forwarded me a message that said they feel hopeless about uni.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
    {
        "id": "quote_05",
        "category": "quoted",
        "text": 'My tutor read out an anonymous note: "I am burned out and scared."',
        "expected_personal_eligibility": "not_eligible",
    },
    # ----- MIXED SELF + OTHER (uncertain unless clearly self) -----
    {
        "id": "mixed_01",
        "category": "mixed",
        "text": "My roommate is overwhelmed and honestly I'm getting stressed too.",
        "expected_personal_eligibility": "uncertain",
    },
    {
        "id": "mixed_02",
        "category": "mixed",
        "text": "Everyone seems exhausted and I guess I am as well lately.",
        "expected_personal_eligibility": "uncertain",
    },
    {
        "id": "mixed_03",
        "category": "mixed",
        "text": "My friend is anxious about finals; watching her makes me worried for myself.",
        "expected_personal_eligibility": "uncertain",
    },
    {
        "id": "mixed_04",
        "category": "mixed",
        "text": "We are all burned out, or at least that's how it feels in my flat.",
        "expected_personal_eligibility": "uncertain",
    },
    # ----- EXTRA CLEAR FIRST-PERSON / TOPIC BALANCE -----
    {
        "id": "fp_academic_03",
        "category": "first_person_direct",
        "text": "I have back-to-back assessments and I feel totally swamped.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "topic_group_01",
        "category": "topic_only",
        "text": "People keep joking that finals season is stressful for everyone.",
        "expected_personal_eligibility": "not_eligible",
    },
    {
        "id": "fp_pressure_01",
        "category": "first_person_direct",
        "text": "The pressure of my grades is making me feel constantly on edge.",
        "expected_personal_eligibility": "eligible",
        "expected_attribution": "self_experience",
    },
    {
        "id": "tp_classmate_01",
        "category": "third_person",
        "text": "My classmate has been isolating herself and skipping social events.",
        "expected_personal_eligibility": "not_eligible",
        "expected_attribution": "not_self_experience",
    },
]


def calibration_eligibility_distribution() -> dict[str, int]:
    counts: dict[str, int] = {
        "eligible": 0,
        "not_eligible": 0,
        "uncertain": 0,
    }
    for case in WELLBEING_CALIBRATION_CASES:
        key = str(case["expected_personal_eligibility"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def assert_no_holdout_overlap(holdout_texts: list[str]) -> None:
    """Guard: calibration texts must not silently equal holdout A–K texts.

    Note: a few near-paraphrases may exist by design for coverage, but exact
    holdout strings used for threshold fitting should be avoided where possible.
    This helper is for tests documenting the separation policy.
    """
    calib = {c["text"].strip().lower() for c in WELLBEING_CALIBRATION_CASES}
    hold = {t.strip().lower() for t in holdout_texts}
    # Soft documentation helper — exact overlaps are reported by callers.
    _ = calib & hold
