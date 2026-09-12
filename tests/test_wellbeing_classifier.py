"""Unit tests for Phase 4A.5 dual-head attribution + reported-speech guard.

Ordinary tests mock the Hugging Face zero-shot pipeline.
They must NOT download DeBERTa or call OpenRouter / HF Inference API.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import pytest

from evaluation.wellbeing.attribution_dev_cases import (
    WELLBEING_ATTRIBUTION_DEV_CASES,
)
from evaluation.wellbeing.calibration_cases import WELLBEING_CALIBRATION_CASES
from evaluation.wellbeing.calibrate_margins import search_thresholds
from evaluation.wellbeing.final_holdout_cases import (
    WELLBEING_FINAL_HOLDOUT_CASES,
    final_holdout_distribution,
)
from evaluation.wellbeing.final_holdout_v2 import (
    WELLBEING_FINAL_HOLDOUT_V2_CASES,
    final_holdout_v2_distribution,
)
from src.config import (
    DEFAULT_WELLBEING_CLASSIFIER_MODEL,
    WELLBEING_SIGNAL_POC_THRESHOLD,
)
from src.wellbeing import WellbeingClassifier, classify_wellbeing
from src.wellbeing.classifier import reset_wellbeing_classifier_for_tests
from src.wellbeing.labels import (
    ATTRIBUTION_CANDIDATES,
    DUAL_ATTRIBUTION_CANDIDATES,
    DUAL_ATTRIBUTION_HYPOTHESIS_TEMPLATE,
    DUAL_ATTRIBUTION_LABELS,
    DUAL_ATTRIBUTION_MULTI_LABEL,
    FINAL_ATTRIBUTION_LABELS,
    FORBIDDEN_DIAGNOSIS_LABELS,
    RAW_ATTRIBUTION_LABELS,
    RELEVANCE_CANDIDATES,
    RELEVANCE_LABELS,
    SIGNAL_CANDIDATES,
    SIGNAL_LABELS,
    TARGET_CANDIDATES,
    TARGET_LABELS,
    candidate_list,
)
from src.wellbeing.policy import decide_eligibility, finalize_self_attribution
from src.wellbeing.schemas import AttributionEvidence, SelfAttributionResult
from wellbeing_fixtures import WELLBEING_SEMANTIC_FIXTURES


def _ordered_scores(
    labels: tuple[str, ...],
    candidates: Mapping[str, str],
    ranking: Sequence[str],
    values: Sequence[float],
) -> dict[str, Any]:
    assert set(ranking) == set(labels)
    return {
        "labels": [candidates[label] for label in ranking],
        "scores": list(values),
    }


class MockZeroShotPipeline:
    def __init__(
        self,
        *,
        relevance: Mapping[str, Any] | None = None,
        target: Mapping[str, Any] | None = None,
        signals: Mapping[str, Any] | None = None,
        dual_attribution: Mapping[str, Any] | None = None,
        legacy_attribution: Mapping[str, Any] | None = None,
        fail: bool = False,
        fail_attribution: bool = False,
        per_text: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = fail
        self.fail_attribution = fail_attribution
        self.per_text = per_text or {}
        self.relevance = relevance or _ordered_scores(
            RELEVANCE_LABELS,
            RELEVANCE_CANDIDATES,
            (
                "personal_wellbeing",
                "wellbeing_topic_only",
                "ambiguous",
                "not_wellbeing_related",
            ),
            (0.72, 0.18, 0.07, 0.03),
        )
        self.target = target or _ordered_scores(
            TARGET_LABELS,
            TARGET_CANDIDATES,
            (
                "self",
                "other_person",
                "group_or_community",
                "institution_or_event",
                "general_or_unknown",
            ),
            (0.81, 0.09, 0.05, 0.03, 0.02),
        )
        self.signals = signals or _ordered_scores(
            SIGNAL_LABELS,
            SIGNAL_CANDIDATES,
            (
                "stress_or_overwhelm",
                "academic_pressure",
                "anxiety_or_fear_language",
                "exhaustion_or_burnout_like_language",
                "self_directed_negativity",
                "interpersonal_distress",
                "loneliness_or_isolation",
                "hopelessness_like_language",
                "positive_wellbeing_or_recovery",
            ),
            (0.91, 0.84, 0.40, 0.22, 0.18, 0.12, 0.08, 0.05, 0.03),
        )
        self.dual_attribution = dual_attribution or _ordered_scores(
            DUAL_ATTRIBUTION_LABELS,
            DUAL_ATTRIBUTION_CANDIDATES,
            ("direct_self_experience", "reported_other_experience"),
            (0.88, 0.12),
        )
        self.legacy_attribution = legacy_attribution or _ordered_scores(
            RAW_ATTRIBUTION_LABELS,
            ATTRIBUTION_CANDIDATES,
            ("self_experience", "not_self_experience"),
            (0.85, 0.15),
        )

    def _payload_for(
        self,
        text: str,
        *,
        kind: str,
        default: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self.per_text.get(text, {}).get(kind) or default

    def _one(
        self,
        text: str,
        candidate_labels: Sequence[str],
        multi_label: bool,
    ) -> Mapping[str, Any]:
        relevance_cands = candidate_list(RELEVANCE_CANDIDATES, RELEVANCE_LABELS)
        target_cands = candidate_list(TARGET_CANDIDATES, TARGET_LABELS)
        signal_cands = candidate_list(SIGNAL_CANDIDATES, SIGNAL_LABELS)
        dual_cands = candidate_list(
            DUAL_ATTRIBUTION_CANDIDATES,
            DUAL_ATTRIBUTION_LABELS,
        )
        legacy_cands = candidate_list(
            ATTRIBUTION_CANDIDATES,
            RAW_ATTRIBUTION_LABELS,
        )

        if list(candidate_labels) == relevance_cands:
            assert multi_label is False
            return self._payload_for(text, kind="relevance", default=self.relevance)
        if list(candidate_labels) == target_cands:
            assert multi_label is False
            return self._payload_for(text, kind="target", default=self.target)
        if list(candidate_labels) == signal_cands:
            assert multi_label is True
            return self._payload_for(text, kind="signals", default=self.signals)
        if list(candidate_labels) == dual_cands:
            assert multi_label is True
            if self.fail_attribution:
                raise RuntimeError("mock_attribution_failure")
            return self._payload_for(
                text,
                kind="dual_attribution",
                default=self.dual_attribution,
            )
        if list(candidate_labels) == legacy_cands:
            assert multi_label is False
            return self._payload_for(
                text,
                kind="legacy_attribution",
                default=self.legacy_attribution,
            )
        raise AssertionError(f"Unexpected candidates: {candidate_labels!r}")

    def __call__(
        self,
        sequences: str | Sequence[str],
        candidate_labels: Sequence[str],
        hypothesis_template: str,
        multi_label: bool = False,
        batch_size: int | None = None,
    ) -> Mapping[str, Any] | list[Mapping[str, Any]]:
        self.calls.append(
            {
                "sequences": sequences,
                "candidate_labels": list(candidate_labels),
                "hypothesis_template": hypothesis_template,
                "multi_label": multi_label,
                "batch_size": batch_size,
            },
        )
        if self.fail:
            raise RuntimeError("mock_inference_failure")
        if isinstance(sequences, str):
            return self._one(sequences, candidate_labels, multi_label)
        return [self._one(t, candidate_labels, multi_label) for t in sequences]


def _clf(**kwargs: Any) -> WellbeingClassifier:
    mock_kwargs = {
        k: kwargs.pop(k)
        for k in list(kwargs)
        if k
        in {
            "relevance",
            "target",
            "signals",
            "dual_attribution",
            "legacy_attribution",
            "fail",
            "fail_attribution",
            "per_text",
        }
    }
    mock = MockZeroShotPipeline(**mock_kwargs)
    clf = WellbeingClassifier(
        enabled=True,
        batch_size=4,
        relevance_min_margin=kwargs.pop("relevance_min_margin", 0.0),
        target_min_margin=kwargs.pop("target_min_margin", 0.0),
        direct_self_min_score=kwargs.pop("direct_self_min_score", 0.5),
        reported_other_block_score=kwargs.pop("reported_other_block_score", 0.5),
        include_legacy_binary_attribution=kwargs.pop(
            "include_legacy_binary_attribution",
            False,
        ),
        pipeline_factory=lambda _m: mock,
        **kwargs,
    )
    clf._mock = mock  # type: ignore[attr-defined]
    return clf


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    reset_wellbeing_classifier_for_tests()
    yield
    reset_wellbeing_classifier_for_tests()


def test_direct_self_hypothesis_centrally_defined() -> None:
    assert "direct_self_experience" in DUAL_ATTRIBUTION_CANDIDATES
    text = DUAL_ATTRIBUTION_CANDIDATES["direct_self_experience"]
    assert "own wellbeing" in text
    assert "author or speaker" in text


def test_reported_other_hypothesis_centrally_defined() -> None:
    assert "reported_other_experience" in DUAL_ATTRIBUTION_CANDIDATES
    text = DUAL_ATTRIBUTION_CANDIDATES["reported_other_experience"]
    assert "another person" in text or "other" in text
    assert "quoting" in text or "reporting" in text


def test_dual_attribution_uses_multi_label_true() -> None:
    assert DUAL_ATTRIBUTION_MULTI_LABEL is True
    clf = _clf()
    clf.classify("I feel completely overwhelmed by my exams today.")
    dual_calls = [
        c
        for c in clf._mock.calls  # type: ignore[attr-defined]
        if c["candidate_labels"]
        == candidate_list(DUAL_ATTRIBUTION_CANDIDATES, DUAL_ATTRIBUTION_LABELS)
    ]
    assert len(dual_calls) == 1
    assert dual_calls[0]["multi_label"] is True
    assert dual_calls[0]["hypothesis_template"] == DUAL_ATTRIBUTION_HYPOTHESIS_TEMPLATE


def test_direct_self_high_reported_other_low_is_self() -> None:
    clf = _clf(
        dual_attribution=_ordered_scores(
            DUAL_ATTRIBUTION_LABELS,
            DUAL_ATTRIBUTION_CANDIDATES,
            ("direct_self_experience", "reported_other_experience"),
            (0.90, 0.10),
        ),
        direct_self_min_score=0.5,
        reported_other_block_score=0.5,
    )
    result = clf.classify("I feel completely overwhelmed by my exams today.")
    assert result.self_attribution.final_label == "self_experience"
    assert result.eligibility_status == "eligible"


def test_direct_self_low_reported_other_high_is_not_self() -> None:
    clf = _clf(
        dual_attribution=_ordered_scores(
            DUAL_ATTRIBUTION_LABELS,
            DUAL_ATTRIBUTION_CANDIDATES,
            ("reported_other_experience", "direct_self_experience"),
            (0.85, 0.20),
        ),
        direct_self_min_score=0.5,
        reported_other_block_score=0.5,
    )
    result = clf.classify("My roommate said she feels overwhelmed enough.")
    assert result.self_attribution.final_label == "not_self_experience"
    assert result.eligibility_status == "not_eligible"


def test_both_high_is_unclear_conflict() -> None:
    clf = _clf(
        dual_attribution=_ordered_scores(
            DUAL_ATTRIBUTION_LABELS,
            DUAL_ATTRIBUTION_CANDIDATES,
            ("direct_self_experience", "reported_other_experience"),
            (0.80, 0.75),
        ),
        direct_self_min_score=0.5,
        reported_other_block_score=0.5,
    )
    result = clf.classify(
        "My roommate is overwhelmed and honestly I'm getting stressed too.",
    )
    assert result.self_attribution.final_label == "unclear"
    assert result.self_attribution.attribution_evidence.evidence_status == "conflict"
    assert result.eligibility_status == "uncertain"


def test_both_low_is_unclear() -> None:
    clf = _clf(
        dual_attribution=_ordered_scores(
            DUAL_ATTRIBUTION_LABELS,
            DUAL_ATTRIBUTION_CANDIDATES,
            ("direct_self_experience", "reported_other_experience"),
            (0.25, 0.20),
        ),
        direct_self_min_score=0.5,
        reported_other_block_score=0.5,
    )
    result = clf.classify("I feel completely overwhelmed by my exams today.")
    assert result.self_attribution.final_label == "unclear"
    assert result.eligibility_status == "uncertain"


def test_quoted_other_blocks_eligibility() -> None:
    text = "My roommate said, 'I'm completely exhausted.'"
    clf = _clf(
        dual_attribution=_ordered_scores(
            DUAL_ATTRIBUTION_LABELS,
            DUAL_ATTRIBUTION_CANDIDATES,
            ("reported_other_experience", "direct_self_experience"),
            (0.82, 0.18),
        ),
    )
    result = clf.classify(text)
    assert result.eligibility_status == "not_eligible"
    assert result.personal_wellbeing_eligible is False


def test_quoted_self_can_remain_eligible() -> None:
    text = "I told my friend, 'I'm completely overwhelmed.'"
    clf = _clf(
        dual_attribution=_ordered_scores(
            DUAL_ATTRIBUTION_LABELS,
            DUAL_ATTRIBUTION_CANDIDATES,
            ("direct_self_experience", "reported_other_experience"),
            (0.86, 0.22),
        ),
    )
    result = clf.classify(text)
    assert result.eligibility_status == "eligible"


def test_mixed_self_other_conservative() -> None:
    finalized = finalize_self_attribution(
        SelfAttributionResult(
            status="ok",
            scores={
                "direct_self_experience": 0.78,
                "reported_other_experience": 0.72,
            },
            attribution_evidence=AttributionEvidence(
                direct_self_score=0.78,
                reported_other_score=0.72,
                evidence_status="ok",
            ),
        ),
        direct_self_min_score=0.5,
        reported_other_block_score=0.5,
    )
    assert finalized.final_label == "unclear"


def test_roommate_case_blocked() -> None:
    clf = _clf(
        dual_attribution=_ordered_scores(
            DUAL_ATTRIBUTION_LABELS,
            DUAL_ATTRIBUTION_CANDIDATES,
            ("reported_other_experience", "direct_self_experience"),
            (0.88, 0.15),
        ),
    )
    result = clf.classify(
        "My roommate told me she has been completely overwhelmed by exams.",
    )
    assert result.personal_wellbeing_eligible is False


def test_recovery_case_eligible() -> None:
    clf = _clf(
        dual_attribution=_ordered_scores(
            DUAL_ATTRIBUTION_LABELS,
            DUAL_ATTRIBUTION_CANDIDATES,
            ("direct_self_experience", "reported_other_experience"),
            (0.91, 0.08),
        ),
        signals=_ordered_scores(
            SIGNAL_LABELS,
            SIGNAL_CANDIDATES,
            (
                "positive_wellbeing_or_recovery",
                "stress_or_overwhelm",
                "academic_pressure",
                "anxiety_or_fear_language",
                "exhaustion_or_burnout_like_language",
                "self_directed_negativity",
                "interpersonal_distress",
                "loneliness_or_isolation",
                "hopelessness_like_language",
            ),
            (0.95, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01),
        ),
    )
    result = clf.classify("I've been doing better lately and feel back on track.")
    assert result.eligibility_status == "eligible"
    assert {s.signal for s in result.signals if s.selected} == {
        "positive_wellbeing_or_recovery",
    }


def test_signal_selection_only_on_eligible() -> None:
    clf = _clf(
        dual_attribution=_ordered_scores(
            DUAL_ATTRIBUTION_LABELS,
            DUAL_ATTRIBUTION_CANDIDATES,
            ("reported_other_experience", "direct_self_experience"),
            (0.80, 0.20),
        ),
    )
    result = clf.classify("Someone else is stressed enough for classification.")
    assert result.eligibility_status != "eligible"
    assert all(not s.selected for s in result.signals)
    assert any(s.threshold_passed for s in result.signals)


def test_no_regex_person_name_authority() -> None:
    from src.wellbeing import policy as policy_mod
    from src.wellbeing import classifier as clf_mod
    import inspect

    policy_src = inspect.getsource(policy_mod)
    clf_src = inspect.getsource(clf_mod)
    for forbidden in ('"my roommate"', "'my roommate'", "person_names", "NAME_LIST"):
        assert forbidden not in policy_src
        assert forbidden not in clf_src
    assert "re.search" not in policy_src
    assert "re.match" not in policy_src


def test_no_raw_text_in_diagnostics(caplog: pytest.LogCaptureFixture) -> None:
    secret = "SECRET_PHRASE_XYZ"
    clf = _clf(fail=True)
    with caplog.at_level(logging.WARNING):
        result = clf.classify(f"I feel overwhelmed because {secret} happened.")
    assert secret not in str(result.model_dump())
    assert secret not in caplog.text


def test_legacy_binary_does_not_affect_production() -> None:
    clf = _clf(
        dual_attribution=_ordered_scores(
            DUAL_ATTRIBUTION_LABELS,
            DUAL_ATTRIBUTION_CANDIDATES,
            ("direct_self_experience", "reported_other_experience"),
            (0.90, 0.10),
        ),
        legacy_attribution=_ordered_scores(
            RAW_ATTRIBUTION_LABELS,
            ATTRIBUTION_CANDIDATES,
            ("not_self_experience", "self_experience"),
            (0.95, 0.05),
        ),
        include_legacy_binary_attribution=True,
    )
    result = clf.classify("I feel completely overwhelmed by my exams today.")
    assert result.self_attribution.legacy_binary is not None
    assert result.self_attribution.legacy_binary.raw_label == "not_self_experience"
    # Production decision follows dual-head, not legacy binary.
    assert result.self_attribution.final_label == "self_experience"
    assert result.eligibility_status == "eligible"


def test_previous_40_case_marked_regression() -> None:
    import evaluation.wellbeing.final_holdout_cases as fh

    doc = (fh.__doc__ or "").lower()
    assert "regression" in doc or "development" in doc
    assert "no longer" in doc or "not" in doc
    assert len(WELLBEING_FINAL_HOLDOUT_CASES) >= 30
    dist = final_holdout_distribution()
    assert dist["n"] == len(WELLBEING_FINAL_HOLDOUT_CASES)


def test_v2_holdout_separate_from_all_dev_sets() -> None:
    assert len(WELLBEING_FINAL_HOLDOUT_V2_CASES) >= 50
    dist = final_holdout_v2_distribution()
    assert dist["n"] == len(WELLBEING_FINAL_HOLDOUT_V2_CASES)
    v2 = {c["text"].strip().lower() for c in WELLBEING_FINAL_HOLDOUT_V2_CASES}
    calib = {c["text"].strip().lower() for c in WELLBEING_CALIBRATION_CASES}
    attr = {c["text"].strip().lower() for c in WELLBEING_ATTRIBUTION_DEV_CASES}
    fh40 = {c["text"].strip().lower() for c in WELLBEING_FINAL_HOLDOUT_CASES}
    ak = {f["text"].strip().lower() for f in WELLBEING_SEMANTIC_FIXTURES}
    assert v2.isdisjoint(calib)
    assert v2.isdisjoint(attr)
    assert v2.isdisjoint(fh40)
    assert v2.isdisjoint(ak)


def test_v2_not_used_for_calibration() -> None:
    import inspect

    from evaluation.wellbeing import calibrate_margins as mod

    src = inspect.getsource(mod.search_thresholds)
    assert "WELLBEING_FINAL_HOLDOUT_V2" not in src
    assert "final_holdout_v2" not in src
    dev_src = inspect.getsource(mod.development_cases)
    assert "FINAL_HOLDOUT_V2" not in dev_src
    assert "WELLBEING_FINAL_HOLDOUT_CASES" not in dev_src


def test_search_thresholds_does_not_mutate_config() -> None:
    from src import config as cfg

    before = (
        cfg.WELLBEING_DIRECT_SELF_MIN_SCORE,
        cfg.WELLBEING_REPORTED_OTHER_BLOCK_SCORE,
    )
    search_thresholds(
        [
            {
                "id": "x",
                "expected_personal_eligibility": "eligible",
                "prediction": {
                    "status": "ok",
                    "relevance_label": "personal_wellbeing",
                    "target_label": "self",
                    "relevance_margin": 0.5,
                    "target_margin": 0.5,
                    "attribution": {
                        "status": "ok",
                        "raw_label": None,
                        "scores": {
                            "direct_self_experience": 0.8,
                            "reported_other_experience": 0.1,
                        },
                        "top_score": 0.8,
                        "top1_top2_margin": 0.7,
                        "error_code": None,
                        "attribution_evidence": {
                            "direct_self_score": 0.8,
                            "reported_other_score": 0.1,
                            "evidence_status": "ok",
                            "error_code": None,
                        },
                    },
                },
            },
        ],
    )
    assert (
        cfg.WELLBEING_DIRECT_SELF_MIN_SCORE,
        cfg.WELLBEING_REPORTED_OTHER_BLOCK_SCORE,
    ) == before


def test_cpu_only_and_lazy_load() -> None:
    fresh = WellbeingClassifier(enabled=True, pipeline_factory=lambda _m: None)
    assert fresh.is_loaded is False
    assert WELLBEING_SIGNAL_POC_THRESHOLD == 0.5
    assert DEFAULT_WELLBEING_CLASSIFIER_MODEL == (
        "MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c"
    )


def test_attribution_only_runs_conditionally() -> None:
    clf = _clf(
        relevance=_ordered_scores(
            RELEVANCE_LABELS,
            RELEVANCE_CANDIDATES,
            (
                "wellbeing_topic_only",
                "personal_wellbeing",
                "ambiguous",
                "not_wellbeing_related",
            ),
            (0.70, 0.15, 0.10, 0.05),
        ),
        target=_ordered_scores(
            TARGET_LABELS,
            TARGET_CANDIDATES,
            (
                "other_person",
                "self",
                "general_or_unknown",
                "group_or_community",
                "institution_or_event",
            ),
            (0.75, 0.10, 0.08, 0.05, 0.02),
        ),
    )
    result = clf.classify("My friend has been really stressed lately enough.")
    assert result.self_attribution.status == "not_evaluated"
    assert clf.last_attribution_call_count == 0


def test_attribution_error_fails_closed() -> None:
    clf = _clf(fail_attribution=True)
    result = clf.classify("I feel completely overwhelmed by my exams today.")
    assert result.self_attribution.status == "error"
    assert result.personal_wellbeing_eligible is False


def test_no_diagnosis_labels() -> None:
    for forbidden in FORBIDDEN_DIAGNOSIS_LABELS:
        assert forbidden not in DUAL_ATTRIBUTION_LABELS
        assert forbidden not in FINAL_ATTRIBUTION_LABELS


def test_classify_wellbeing_convenience() -> None:
    from src.wellbeing import classifier as clf_mod

    clf_mod._default_classifier = _clf()
    result = classify_wellbeing(
        "I feel completely overwhelmed by my exams today.",
    )
    assert result.eligibility_status == "eligible"


def test_policy_decide_eligibility_uses_dual_scores() -> None:
    decision = decide_eligibility(
        classifier_status="ok",
        relevance_label="personal_wellbeing",
        target_label="self",
        relevance_margin=0.5,
        target_margin=0.5,
        attribution=SelfAttributionResult(
            status="ok",
            scores={
                "direct_self_experience": 0.2,
                "reported_other_experience": 0.15,
            },
            attribution_evidence=AttributionEvidence(
                direct_self_score=0.2,
                reported_other_score=0.15,
                evidence_status="ok",
            ),
        ),
        relevance_min_margin=0.0,
        target_min_margin=0.0,
        direct_self_min_score=0.5,
        reported_other_block_score=0.5,
    )
    assert decision.status == "uncertain"
    assert decision.finalized_attribution.final_label == "unclear"


def test_a_k_docs_mark_regression() -> None:
    from evaluation.wellbeing import calibrate_margins as mod

    assert "REGRESSION" in mod.__doc__ or "regression" in mod.__doc__.lower()
    assert "contaminated" in mod.__doc__.lower() or "development" in mod.__doc__.lower()
