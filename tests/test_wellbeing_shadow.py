"""Phase 4B wellbeing shadow-mode tests (mocked DeBERTa — no downloads)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence
from unittest.mock import MagicMock, patch

import pytest

from src.config import (
    DEFAULT_WELLBEING_CLASSIFIER_MODEL,
    WELLBEING_DIRECT_SELF_MIN_SCORE,
    WELLBEING_REPORTED_OTHER_BLOCK_SCORE,
    WELLBEING_SIGNAL_POC_THRESHOLD,
    resolve_wellbeing_shadow_enabled,
)
from src.pipeline import MyUniSentimentPipeline
from src.schemas import (
    ActivityInput,
    AnalysisBlock,
    FinalTemporalAssessment,
    ModalityBundle,
    SentimentEvidence,
    SpeechSegment,
    TemporalContext,
    TemporalFeatures,
    TemporalWindow,
)
from src.wellbeing.classifier import reset_wellbeing_classifier_for_tests
from src.wellbeing.schemas import (
    ExclusiveClassification,
    SelfAttributionResult,
    SignalScore,
    WellbeingClassificationResult,
    WellbeingShadowAnalysis,
)
from src.wellbeing.shadow import build_wellbeing_shadow, collect_shadow_items


def _ev(label: str = "neutral", score: float = 0.0, conf: float = 0.5) -> SentimentEvidence:
    return SentimentEvidence(
        label=label,  # type: ignore[arg-type]
        score=score,
        confidence=conf,
        probabilities={"negative": 0.2, "neutral": 0.6, "positive": 0.2},
        model="stub",
    )


def _fake_classification(
    *,
    eligibility: str = "eligible",
    selected: Optional[list[str]] = None,
    status: str = "ok",
) -> WellbeingClassificationResult:
    selected = selected or (["stress_or_overwhelm"] if eligibility == "eligible" else [])
    signals = [
        SignalScore(
            signal="stress_or_overwhelm",
            score=0.9 if "stress_or_overwhelm" in selected else 0.1,
            threshold_passed="stress_or_overwhelm" in selected,
            selected="stress_or_overwhelm" in selected,
        ),
        SignalScore(
            signal="academic_pressure",
            score=0.2,
            threshold_passed=False,
            selected=False,
        ),
    ]
    return WellbeingClassificationResult(
        model_id=DEFAULT_WELLBEING_CLASSIFIER_MODEL,
        status=status,  # type: ignore[arg-type]
        relevance=ExclusiveClassification(label="personal_wellbeing", scores={}),
        target=ExclusiveClassification(label="self", scores={}),
        signals=signals,
        self_attribution=SelfAttributionResult(
            status="ok",
            final_label="self_experience" if eligibility == "eligible" else "unclear",
            label="self_experience" if eligibility == "eligible" else "unclear",
        ),
        eligibility_status=eligibility,  # type: ignore[arg-type]
        personal_wellbeing_eligible=eligibility == "eligible",
    )


class RecordingClassifier:
    def __init__(self, results: Optional[list[WellbeingClassificationResult]] = None) -> None:
        self.model_id = DEFAULT_WELLBEING_CLASSIFIER_MODEL
        self.calls: list[list[Optional[str]]] = []
        self._results = results

    def classify_many(
        self,
        texts: Sequence[Optional[str]],
        *,
        source_type: str = "text",
    ) -> list[WellbeingClassificationResult]:
        self.calls.append(list(texts))
        if self._results is not None:
            assert len(self._results) == len(texts)
            return list(self._results)
        return [_fake_classification() for _ in texts]


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_wellbeing_classifier_for_tests()
    yield
    reset_wellbeing_classifier_for_tests()


def test_shadow_defaults_disabled() -> None:
    assert resolve_wellbeing_shadow_enabled() is False


def test_shadow_disabled_returns_none_and_no_classifier_call() -> None:
    clf = RecordingClassifier()
    out = build_wellbeing_shadow(
        primary_text="I feel overwhelmed by exams this week.",
        classifier=clf,
        enabled=False,
    )
    assert out is None
    assert clf.calls == []


def test_shadow_disabled_pipeline_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WELLBEING_SHADOW_ENABLED", "0")
    pipe = MyUniSentimentPipeline()
    text_mock = MagicMock()
    text_mock.validate_text.side_effect = lambda t: str(t).strip()
    text_mock.analyze.return_value = _ev("negative", -0.4, 0.8)
    pipe._text_analyzer = text_mock

    with patch("src.pipeline.build_wellbeing_shadow") as shadow_fn:
        result = pipe.analyze_text("I feel overwhelmed by exams this week.")
        shadow_fn.assert_not_called()
    assert result.analysis.wellbeing_shadow is None
    assert result.analysis.overall.label == "negative"


def test_text_shadow_classification() -> None:
    clf = RecordingClassifier()
    out = build_wellbeing_shadow(
        primary_text="I feel overwhelmed by exams this week.",
        classifier=clf,
        enabled=True,
    )
    assert out is not None
    assert out.mode == "shadow"
    assert out.affects_final_assessment is False
    assert out.status == "ok"
    assert len(out.source_results) == 1
    assert out.source_results[0].source_role == "primary_text"
    assert out.source_results[0].classification is not None
    assert "I feel" not in str(out.model_dump())


def test_audio_uses_existing_transcript_empty_insufficient() -> None:
    clf = RecordingClassifier()
    out = build_wellbeing_shadow(transcript="   ", classifier=clf, enabled=True)
    assert out is not None
    assert out.status == "insufficient_text"
    assert clf.calls == []


def test_audio_transcript_classified() -> None:
    clf = RecordingClassifier()
    out = build_wellbeing_shadow(
        transcript="I can't cope with the workload anymore.",
        classifier=clf,
        enabled=True,
    )
    assert out is not None
    assert out.source_results[0].source_role == "transcript"
    assert len(clf.calls[0]) == 1


def test_image_visual_only_does_not_run_classifier() -> None:
    clf = RecordingClassifier()
    out = build_wellbeing_shadow(classifier=clf, enabled=True)
    assert out is not None
    assert out.status == "insufficient_text"
    assert clf.calls == []


def test_image_caption_may_run_classifier_ocr_excluded() -> None:
    items = collect_shadow_items(
        caption="I feel lonely on campus this term.",
        # OCR must not appear even if somehow passed — API has no ocr arg.
    )
    assert [i.source_role for i in items] == ["caption"]
    clf = RecordingClassifier()
    out = build_wellbeing_shadow(
        caption="I feel lonely on campus this term.",
        classifier=clf,
        enabled=True,
    )
    assert out is not None
    assert out.source_results[0].source_role == "caption"


def test_video_batch_classify_many_preserves_order_and_timestamps() -> None:
    ctx = TemporalContext(
        window_seconds=5.0,
        duration_seconds=10.0,
        events=[],
        windows=[
            TemporalWindow(
                start=0.0,
                end=5.0,
                index=0,
                speech_segments=[SpeechSegment(start=0.1, end=1.0, text="I feel stressed")],
                usable=True,
            ),
            TemporalWindow(
                start=5.0,
                end=10.0,
                index=1,
                speech_segments=[SpeechSegment(start=5.1, end=6.0, text="I feel better now")],
                usable=True,
            ),
            TemporalWindow(
                start=10.0,
                end=15.0,
                index=2,
                speech_segments=[],
                usable=True,
            ),
        ],
        features=TemporalFeatures(),
    )
    results = [
        _fake_classification(eligibility="eligible", selected=["stress_or_overwhelm"]),
        _fake_classification(eligibility="not_eligible", selected=[]),
        _fake_classification(eligibility="uncertain", selected=[]),
        _fake_classification(eligibility="eligible", selected=["stress_or_overwhelm"]),
    ]
    # caption + transcript + 2 usable speech windows
    clf = RecordingClassifier(results=results)
    out = build_wellbeing_shadow(
        caption="Campus update from me",
        transcript="I feel stressed. I feel better now.",
        temporal_context=ctx,
        classifier=clf,
        enabled=True,
    )
    assert out is not None
    assert len(clf.calls) == 1
    assert len(clf.calls[0]) == 4  # classify_many once
    roles = [s.source_role for s in out.source_results]
    assert roles == ["caption", "transcript"]
    assert [w.window_index for w in out.window_results] == [0, 1]
    assert out.window_results[0].start == 0.0
    assert out.window_results[0].end == 5.0
    assert out.summary.windows_submitted == 2
    assert out.summary.items_submitted == 4
    assert out.summary.eligible_source_count == 1  # caption eligible, transcript not
    assert out.summary.eligible_window_count == 1
    assert out.summary.uncertain_window_count == 1
    assert out.summary.selected_signal_counts.get("stress_or_overwhelm") == 2
    dumped = out.model_dump()
    assert "I feel stressed" not in str(dumped)
    assert "I feel better" not in str(dumped)


def test_non_usable_windows_excluded() -> None:
    ctx = TemporalContext(
        window_seconds=5.0,
        duration_seconds=5.0,
        events=[],
        windows=[
            TemporalWindow(
                start=0.0,
                end=5.0,
                index=0,
                speech_segments=[SpeechSegment(start=0.1, end=1.0, text="I feel stressed")],
                usable=False,
            ),
        ],
        features=TemporalFeatures(),
    )
    items = collect_shadow_items(temporal_context=ctx)
    assert items == []


def test_shadow_failure_does_not_raise() -> None:
    class Boom:
        model_id = "x"

        def classify_many(self, texts: Sequence[Optional[str]], **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    out = build_wellbeing_shadow(
        primary_text="I feel overwhelmed by exams this week.",
        classifier=Boom(),
        enabled=True,
    )
    assert out is not None
    assert out.status == "error"
    assert out.error_code == "RuntimeError"


def test_pipeline_shadow_failure_keeps_main_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WELLBEING_SHADOW_ENABLED", "1")
    pipe = MyUniSentimentPipeline()
    text_mock = MagicMock()
    text_mock.validate_text.side_effect = lambda t: str(t).strip()
    text_mock.analyze.return_value = _ev("negative", -0.5, 0.9)
    pipe._text_analyzer = text_mock

    with patch(
        "src.pipeline.build_wellbeing_shadow",
        return_value=WellbeingShadowAnalysis(status="error", error_code="X"),
    ):
        result = pipe.analyze_text("I feel overwhelmed by exams this week.")
    assert result.analysis.overall.label == "negative"
    assert result.analysis.wellbeing_shadow is not None
    assert result.analysis.wellbeing_shadow.status == "error"


def test_final_assessment_sentiment_temporal_equal_on_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    from src.schemas import TemporalHighlight

    final = FinalTemporalAssessment(
        overall_wellbeing_indicator="moderate_concern",
        key_temporal_highlights=[
            TemporalHighlight(
                start=0.0,
                end=5.0,
                timestamp_label="0-5s",
                description="stress language",
            ),
        ],
        summary_explanation="unchanged",
        evidence_summary="ev",
        uncertainty_note="none",
        context_type="personal_expression",
        status="ok",
    )
    features = TemporalFeatures(usable_windows=2, overall_usable_coverage=0.5)
    ctx = TemporalContext(
        window_seconds=5.0,
        duration_seconds=10.0,
        events=[],
        windows=[
            TemporalWindow(
                start=0.0,
                end=5.0,
                index=0,
                speech_segments=[
                    SpeechSegment(start=0.0, end=1.0, text="I feel completely overwhelmed"),
                ],
                usable=True,
            ),
        ],
        features=features,
    )
    bundle = SimpleNamespace(
        visual=_ev("negative", -0.3, 0.7),
        ocr=None,
        ocr_text="IGNORE OCR FOR WELLBEING",
        speech=_ev("negative", -0.4, 0.8),
        transcript="I feel completely overwhelmed",
        speech_result=None,
        diagnostics=None,
        warnings=[],
        overall=_ev("negative", -0.35, 0.75),
        temporal_context=ctx,
        deterministic_context=None,
        temporal_reasoning=None,
        temporal_reasoner_diagnostics=None,
        final_temporal_assessment=final,
    )

    pipe = MyUniSentimentPipeline()
    pipe._video_analyzer = MagicMock()
    pipe._video_analyzer.analyze.return_value = bundle
    text_mock = MagicMock()
    text_mock.validate_text.side_effect = lambda t: str(t).strip()
    text_mock.analyze.return_value = _ev("neutral", 0.0, 0.5)
    pipe._text_analyzer = text_mock

    activity = ActivityInput(
        activity_id="ACT-1",
        user_id="U1",
        activity_type="video",
        media_path=str(video),
        text="Author caption about my stress",
        created_at=datetime.now(timezone.utc),
    )

    monkeypatch.setenv("WELLBEING_SHADOW_ENABLED", "0")
    off = pipe.analyze_activity(activity)

    monkeypatch.setenv("WELLBEING_SHADOW_ENABLED", "1")
    from src.wellbeing.schemas import WellbeingShadowSummary

    fake_shadow = WellbeingShadowAnalysis(
        status="ok",
        model_id=DEFAULT_WELLBEING_CLASSIFIER_MODEL,
        summary=WellbeingShadowSummary(items_submitted=3),
    )
    with patch("src.pipeline.build_wellbeing_shadow", return_value=fake_shadow):
        on = pipe.analyze_activity(activity)

    assert off.analysis.final_temporal_assessment == on.analysis.final_temporal_assessment
    assert off.analysis.overall.model_dump() == on.analysis.overall.model_dump()
    assert off.analysis.temporal_context == on.analysis.temporal_context
    assert off.analysis.wellbeing_shadow is None
    assert on.analysis.wellbeing_shadow is not None
    assert on.analysis.wellbeing_shadow.affects_final_assessment is False
    # Whisper / temporal not recomputed — same mocked analyze call path
    assert pipe._video_analyzer.analyze.call_count == 2


def test_no_diagnosis_or_score_fields() -> None:
    out = build_wellbeing_shadow(
        primary_text="I feel overwhelmed by exams this week.",
        classifier=RecordingClassifier(),
        enabled=True,
    )
    assert out is not None
    dumped = out.model_dump()
    blob = str(dumped).lower()
    for forbidden in (
        "mental_health_score",
        "overall_wellbeing_score",
        "overall_stress_score",
        "depression",
        "suicidal",
    ):
        assert forbidden not in blob
    assert "affects_final_assessment" in dumped
    assert dumped["affects_final_assessment"] is False


def test_thresholds_and_model_unchanged() -> None:
    assert WELLBEING_SIGNAL_POC_THRESHOLD == 0.5
    assert WELLBEING_DIRECT_SELF_MIN_SCORE == 0.35
    assert WELLBEING_REPORTED_OTHER_BLOCK_SCORE == 0.55
    assert DEFAULT_WELLBEING_CLASSIFIER_MODEL == (
        "MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c"
    )


def test_lazy_load_when_shadow_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WELLBEING_SHADOW_ENABLED", "0")
    with patch("src.wellbeing.classifier.get_wellbeing_classifier") as getter:
        out = build_wellbeing_shadow(primary_text="hello there friend", enabled=None)
        assert out is None
        getter.assert_not_called()


def test_no_raw_text_in_shadow_logs(caplog: pytest.LogCaptureFixture) -> None:
    secret = "SECRET_SHADOW_PHRASE_ZZZ"

    class Boom:
        model_id = "x"

        def classify_many(self, texts: Sequence[Optional[str]], **kwargs: Any) -> Any:
            raise RuntimeError("fail")

    with caplog.at_level(logging.WARNING):
        out = build_wellbeing_shadow(
            primary_text=f"I feel bad because {secret}",
            classifier=Boom(),
            enabled=True,
        )
    assert out is not None
    assert secret not in caplog.text
    assert secret not in str(out.model_dump())


def test_analysis_block_accepts_optional_shadow() -> None:
    block = AnalysisBlock(
        overall=_ev(),
        modalities=ModalityBundle(),
    )
    assert block.wellbeing_shadow is None
    with_shadow = block.model_copy(
        update={
            "wellbeing_shadow": WellbeingShadowAnalysis(status="insufficient_text"),
        },
    )
    assert with_shadow.wellbeing_shadow is not None
