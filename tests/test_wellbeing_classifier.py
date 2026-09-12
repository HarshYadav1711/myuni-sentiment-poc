"""Unit tests for the Phase 4 independent wellbeing classifier foundation.

Ordinary tests mock the Hugging Face zero-shot pipeline.
They must NOT download DeBERTa or call OpenRouter / HF Inference API.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import pytest

from src.config import (
    DEFAULT_WELLBEING_CLASSIFIER_MODEL,
    WELLBEING_SIGNAL_POC_THRESHOLD,
    resolve_wellbeing_classifier_config,
)
from src.wellbeing import WellbeingClassifier, classify_wellbeing
from src.wellbeing.classifier import reset_wellbeing_classifier_for_tests
from src.wellbeing.labels import (
    FORBIDDEN_DIAGNOSIS_LABELS,
    RELEVANCE_CANDIDATES,
    RELEVANCE_LABELS,
    SIGNAL_CANDIDATES,
    SIGNAL_LABELS,
    TARGET_CANDIDATES,
    TARGET_LABELS,
    candidate_list,
)
from wellbeing_fixtures import WELLBEING_SEMANTIC_FIXTURES


def _ordered_scores(
    labels: tuple[str, ...],
    candidates: Mapping[str, str],
    ranking: Sequence[str],
    values: Sequence[float],
) -> dict[str, Any]:
    """Build a HF-like zero-shot payload ordered by descending score."""
    assert set(ranking) == set(labels)
    assert len(ranking) == len(values)
    return {
        "labels": [candidates[label] for label in ranking],
        "scores": list(values),
    }


class MockZeroShotPipeline:
    """Deterministic fake for transformers zero-shot-classification."""

    def __init__(
        self,
        *,
        relevance: Mapping[str, Any] | None = None,
        target: Mapping[str, Any] | None = None,
        signals: Mapping[str, Any] | None = None,
        fail: bool = False,
        unknown_label: bool = False,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = fail
        self.unknown_label = unknown_label
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

    def __call__(
        self,
        text: str,
        candidate_labels: Sequence[str],
        hypothesis_template: str,
        multi_label: bool = False,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "text": text,
                "candidate_labels": list(candidate_labels),
                "hypothesis_template": hypothesis_template,
                "multi_label": multi_label,
            },
        )
        if self.fail:
            raise RuntimeError("mock_inference_failure")
        if self.unknown_label:
            return {
                "labels": ["totally unknown candidate"],
                "scores": [0.9],
            }

        relevance_cands = candidate_list(RELEVANCE_CANDIDATES, RELEVANCE_LABELS)
        target_cands = candidate_list(TARGET_CANDIDATES, TARGET_LABELS)
        signal_cands = candidate_list(SIGNAL_CANDIDATES, SIGNAL_LABELS)

        if list(candidate_labels) == relevance_cands:
            assert multi_label is False
            return self.relevance
        if list(candidate_labels) == target_cands:
            assert multi_label is False
            return self.target
        if list(candidate_labels) == signal_cands:
            assert multi_label is True
            return self.signals
        raise AssertionError(f"Unexpected candidates: {candidate_labels!r}")


def _classifier_with_mock(**kwargs: Any) -> WellbeingClassifier:
    mock = MockZeroShotPipeline(**kwargs)
    clf = WellbeingClassifier(
        enabled=True,
        pipeline_factory=lambda _model_id: mock,
    )
    clf._mock = mock  # type: ignore[attr-defined]
    return clf


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    reset_wellbeing_classifier_for_tests()
    yield
    reset_wellbeing_classifier_for_tests()


def test_package_import_does_not_load_model() -> None:
    import src.wellbeing as wb
    import src.wellbeing.classifier as clf_mod

    assert wb.WellbeingClassifier is not None
    fresh = WellbeingClassifier(enabled=True, pipeline_factory=lambda _m: None)
    assert fresh.is_loaded is False
    assert clf_mod.get_wellbeing_classifier().is_loaded is False


def test_relevance_exclusive_mapping() -> None:
    clf = _classifier_with_mock()
    result = clf.classify(
        "I have three exams this week and I feel completely overwhelmed.",
    )
    assert result.status == "ok"
    assert result.relevance.label == "personal_wellbeing"
    assert set(result.relevance.scores) == set(RELEVANCE_LABELS)
    assert result.relevance.confidence == pytest.approx(0.72)


def test_target_exclusive_mapping() -> None:
    clf = _classifier_with_mock()
    result = clf.classify("I feel overwhelmed by exams.")
    assert result.target.label == "self"
    assert set(result.target.scores) == set(TARGET_LABELS)
    assert result.target.confidence == pytest.approx(0.81)


def test_signals_multilabel_mapping_and_ordering() -> None:
    clf = _classifier_with_mock()
    result = clf.classify("I feel overwhelmed by exams.")
    assert [item.signal for item in result.signals] == [
        "stress_or_overwhelm",
        "academic_pressure",
        "anxiety_or_fear_language",
        "exhaustion_or_burnout_like_language",
        "self_directed_negativity",
        "interpersonal_distress",
        "loneliness_or_isolation",
        "hopelessness_like_language",
        "positive_wellbeing_or_recovery",
    ]
    scores = [item.score for item in result.signals]
    assert scores == sorted(scores, reverse=True)


def test_selected_signal_threshold() -> None:
    clf = _classifier_with_mock()
    clf.signal_threshold = 0.5
    result = clf.classify("I feel overwhelmed by exams.")
    selected = {item.signal for item in result.signals if item.selected}
    assert selected == {"stress_or_overwhelm", "academic_pressure"}
    for item in result.signals:
        assert item.selected == (item.score >= 0.5)


def test_relevance_top1_top2_margin() -> None:
    clf = _classifier_with_mock()
    result = clf.classify("I feel overwhelmed by exams.")
    assert result.uncertainty.relevance_top1_top2_margin == pytest.approx(
        0.72 - 0.18,
    )


def test_target_margin() -> None:
    clf = _classifier_with_mock()
    result = clf.classify("I feel overwhelmed by exams.")
    assert result.uncertainty.target_top1_top2_margin == pytest.approx(
        0.81 - 0.09,
    )


@pytest.mark.parametrize("bad", [None, "", "   ", "\n\t", "hi", "short"])
def test_empty_whitespace_and_short_input(bad: object) -> None:
    clf = _classifier_with_mock()
    result = clf.classify(bad)  # type: ignore[arg-type]
    assert result.status == "insufficient_text"
    assert result.relevance.label is None
    assert result.target.label is None
    assert all(not item.selected for item in result.signals)
    assert all(item.score == 0.0 for item in result.signals)
    assert clf._mock.calls == []  # type: ignore[attr-defined]


def test_no_fake_neutral_result_on_missing() -> None:
    clf = _classifier_with_mock()
    result = clf.classify(None)
    assert result.status == "insufficient_text"
    assert result.relevance.label != "not_wellbeing_related"
    assert result.error_code == "insufficient_text"


def test_disabled_classifier() -> None:
    mock = MockZeroShotPipeline()
    clf = WellbeingClassifier(
        enabled=False,
        pipeline_factory=lambda _m: mock,
    )
    result = clf.classify("I feel completely overwhelmed by my exams today.")
    assert result.status == "classifier_unavailable"
    assert result.error_code == "classifier_disabled"
    assert clf.is_loaded is False
    assert mock.calls == []


def test_model_load_failure() -> None:
    def _boom(_model_id: str) -> Any:
        raise OSError("mock_load_failure")

    clf = WellbeingClassifier(enabled=True, pipeline_factory=_boom)
    result = clf.classify("I feel completely overwhelmed by my exams today.")
    assert result.status == "classifier_unavailable"
    assert result.error_code == "OSError"
    assert clf.is_loaded is False


def test_inference_failure() -> None:
    clf = _classifier_with_mock(fail=True)
    result = clf.classify("I feel completely overwhelmed by my exams today.")
    assert result.status == "error"
    assert result.error_code == "RuntimeError"
    assert "overwhelmed" not in (result.error_message or "")


def test_unknown_hf_output_fails_safely() -> None:
    clf = _classifier_with_mock(unknown_label=True)
    result = clf.classify("I feel completely overwhelmed by my exams today.")
    assert result.status == "error"
    assert result.error_code == "ValueError"


def test_model_identifier_returned() -> None:
    clf = _classifier_with_mock()
    result = clf.classify("I feel completely overwhelmed by my exams today.")
    assert result.model_id == DEFAULT_WELLBEING_CLASSIFIER_MODEL


def test_raw_user_text_not_copied_into_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "UNIQUE_SECRET_PHRASE_XYZ_should_never_appear"
    clf = _classifier_with_mock(fail=True)
    with caplog.at_level(logging.WARNING):
        result = clf.classify(f"I feel overwhelmed because {secret} happened.")
    dumped = result.model_dump()
    assert secret not in str(dumped)
    assert secret not in caplog.text
    assert result.error_message is not None
    assert secret not in result.error_message


def test_signal_threshold_marked_poc_configurable() -> None:
    cfg = resolve_wellbeing_classifier_config()
    assert cfg.signal_poc_threshold == WELLBEING_SIGNAL_POC_THRESHOLD
    assert "POC" in "WELLBEING_SIGNAL_POC_THRESHOLD"
    # Config docstring / constant comment contract checked via name + docs.
    assert WELLBEING_SIGNAL_POC_THRESHOLD == 0.5


def test_no_diagnosis_labels() -> None:
    for forbidden in FORBIDDEN_DIAGNOSIS_LABELS:
        assert forbidden not in RELEVANCE_LABELS
        assert forbidden not in TARGET_LABELS
        assert forbidden not in SIGNAL_LABELS
        assert forbidden.lower() not in {
            label.lower() for label in SIGNAL_LABELS
        }


def test_no_visual_facial_input_dependency() -> None:
    clf = _classifier_with_mock()
    # classify accepts text + source_type only — no image / face args.
    result = clf.classify(
        "I feel completely overwhelmed by my exams today.",
        source_type="speech_transcript",
    )
    assert result.input_metadata.source_type == "speech_transcript"
    assert not hasattr(clf.classify, "image")
    assert "siglip" not in clf.classify.__code__.co_varnames
    assert "face" not in clf.classify.__code__.co_varnames


def test_multi_label_flag_usage() -> None:
    clf = _classifier_with_mock()
    clf.classify("I feel completely overwhelmed by my exams today.")
    flags = [call["multi_label"] for call in clf._mock.calls]  # type: ignore[attr-defined]
    assert flags == [False, False, True]


def test_semantic_fixtures_are_human_expectations_not_model_asserts() -> None:
    """Fixtures exist for later evaluation; do not assert live model labels."""
    assert len(WELLBEING_SEMANTIC_FIXTURES) >= 6
    ids = {item["id"] for item in WELLBEING_SEMANTIC_FIXTURES}
    assert "A_exams_overwhelmed" in ids
    assert "F_exams_stressful_group" in ids
    # Ensure unit tests did not bind fixture texts to mocked exclusive labels.
    for item in WELLBEING_SEMANTIC_FIXTURES:
        assert "expected_human" in item


def test_env_rejects_remote_api_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "WELLBEING_CLASSIFIER_MODEL",
        "https://openrouter.ai/api/v1",
    )
    with pytest.raises(ValueError, match="local Hugging Face"):
        resolve_wellbeing_classifier_config()


def test_classify_wellbeing_convenience(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_wellbeing_classifier_for_tests()
    mock = MockZeroShotPipeline()

    def _factory(model_id: str) -> MockZeroShotPipeline:
        assert model_id == DEFAULT_WELLBEING_CLASSIFIER_MODEL
        return mock

    monkeypatch.setattr(
        "src.wellbeing.classifier.WellbeingClassifier._create_pipeline",
        lambda self: _factory(self.model_id),
    )
    # Replace singleton with a mock-backed instance.
    from src.wellbeing import classifier as clf_mod

    clf_mod._default_classifier = WellbeingClassifier(
        enabled=True,
        pipeline_factory=_factory,
    )
    result = classify_wellbeing(
        "I feel completely overwhelmed by my exams today.",
    )
    assert result.status == "ok"
    assert result.relevance.label == "personal_wellbeing"
