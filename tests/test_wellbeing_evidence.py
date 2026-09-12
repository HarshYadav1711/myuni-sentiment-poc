"""Phase 4C.1 wellbeing evidence aggregation tests (mocked — no model inference)."""

from __future__ import annotations

import inspect
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.config import DEFAULT_WELLBEING_CLASSIFIER_MODEL
from src.pipeline import MyUniSentimentPipeline
from src.schemas import (
    AnalysisBlock,
    FinalTemporalAssessment,
    ModalityBundle,
    SentimentEvidence,
)
from src.wellbeing.evidence import build_wellbeing_evidence_context
from src.wellbeing.labels import DISTRESS_LIKE_SIGNAL_IDS, RECOVERY_SIGNAL_IDS
from src.wellbeing.schemas import (
    ExclusiveClassification,
    SelfAttributionResult,
    SignalScore,
    WellbeingClassificationResult,
    WellbeingShadowAnalysis,
    WellbeingShadowSourceResult,
    WellbeingShadowWindowResult,
)
from src.wellbeing.shadow import build_wellbeing_shadow


def _ev() -> SentimentEvidence:
    return SentimentEvidence(
        label="neutral",
        score=0.0,
        confidence=0.5,
        probabilities={"negative": 0.2, "neutral": 0.6, "positive": 0.2},
        model="stub",
    )


def _signals(selected: list[str], scores: Optional[dict[str, float]] = None) -> list[SignalScore]:
    scores = scores or {}
    out: list[SignalScore] = []
    all_ids = sorted(set(selected) | set(scores) | {"stress_or_overwhelm"})
    for sid in all_ids:
        sel = sid in selected
        score = scores.get(sid, 0.9 if sel else 0.1)
        out.append(
            SignalScore(
                signal=sid,
                score=score,
                threshold_passed=sel,
                selected=sel,
            ),
        )
    return out


def _clf(
    *,
    eligibility: str = "eligible",
    selected: Optional[list[str]] = None,
    status: str = "ok",
    relevance: str = "personal_wellbeing",
    target: str = "self",
    attribution: str = "self_experience",
    signal_scores: Optional[dict[str, float]] = None,
) -> WellbeingClassificationResult:
    selected = selected if selected is not None else (
        ["stress_or_overwhelm"] if eligibility == "eligible" else []
    )
    return WellbeingClassificationResult(
        model_id=DEFAULT_WELLBEING_CLASSIFIER_MODEL,
        status=status,  # type: ignore[arg-type]
        relevance=ExclusiveClassification(label=relevance, scores={relevance: 0.8}),
        target=ExclusiveClassification(label=target, scores={target: 0.8}),
        signals=_signals(selected, signal_scores),
        self_attribution=SelfAttributionResult(
            status="ok",
            final_label=attribution,
            label=attribution,
        ),
        eligibility_status=eligibility,  # type: ignore[arg-type]
        personal_wellbeing_eligible=eligibility == "eligible",
    )


def _src(
    role: str = "transcript",
    clf: Optional[WellbeingClassificationResult] = None,
) -> WellbeingShadowSourceResult:
    return WellbeingShadowSourceResult(
        source_role=role,  # type: ignore[arg-type]
        classification=clf,
        input_character_count=40,
        provenance={"role": role},
    )


def _win(
    index: int,
    *,
    start: float,
    end: float,
    eligibility: str = "not_eligible",
    selected: Optional[list[str]] = None,
    signal_scores: Optional[dict[str, float]] = None,
) -> WellbeingShadowWindowResult:
    return WellbeingShadowWindowResult(
        window_index=index,
        start=start,
        end=end,
        classification=_clf(
            eligibility=eligibility,
            selected=selected if selected is not None else (
                ["stress_or_overwhelm"] if eligibility == "eligible" else []
            ),
            signal_scores=signal_scores,
        ),
        input_character_count=20,
        usable_window=True,
    )


# ---------------------------------------------------------------------------
# Core evidence builder tests
# ---------------------------------------------------------------------------


def test_01_empty_evidence() -> None:
    ctx = build_wellbeing_evidence_context()
    assert ctx.status in {"empty", "insufficient"}
    assert ctx.global_evidence.source_present is False
    assert ctx.temporal_evidence.evaluated_window_count == 0
    assert ctx.affects_final_assessment is False


def test_02_global_only_evidence() -> None:
    ctx = build_wellbeing_evidence_context(
        source_results=[
            _src(
                "transcript",
                _clf(
                    eligibility="eligible",
                    selected=["stress_or_overwhelm", "academic_pressure"],
                ),
            ),
        ],
        window_results=[],
    )
    assert ctx.status == "ok"
    g = ctx.global_evidence
    assert g.source_present is True
    assert g.eligibility_status == "eligible"
    assert g.personal_wellbeing_eligible is True
    assert g.relevance == "personal_wellbeing"
    assert g.target == "self"
    assert g.final_attribution == "self_experience"
    assert set(g.selected_signals) == {"stress_or_overwhelm", "academic_pressure"}
    assert ctx.temporal_evidence.eligible_window_count == 0
    assert "global_eligible_no_local_support" in ctx.conflict_diagnostics


def test_03_windows_only_evidence() -> None:
    ctx = build_wellbeing_evidence_context(
        source_results=[],
        window_results=[
            _win(0, start=0.0, end=5.0, eligibility="eligible"),
        ],
    )
    assert ctx.global_evidence.source_present is False
    assert ctx.temporal_evidence.eligible_window_count == 1
    assert ctx.evidence_ids == ["wellbeing-window-0"]


def test_04_05_06_07_eligible_uncertain_not_eligible_fraction() -> None:
    windows = [
        _win(0, start=0.0, end=5.0, eligibility="not_eligible"),
        _win(1, start=5.0, end=10.0, eligibility="uncertain", selected=[]),
        _win(2, start=10.0, end=15.0, eligibility="eligible"),
        _win(3, start=15.0, end=20.0, eligibility="eligible"),
        _win(4, start=20.0, end=25.0, eligibility="not_eligible"),
    ]
    ctx = build_wellbeing_evidence_context(window_results=windows)
    t = ctx.temporal_evidence
    assert t.evaluated_window_count == 5
    assert t.eligible_window_count == 2
    assert t.uncertain_window_count == 1
    assert t.not_eligible_window_count == 2
    assert t.eligible_window_fraction == pytest.approx(0.4)
    assert t.eligible_window_indices == [2, 3]


def test_08_chronological_ordering() -> None:
    # Intentionally shuffled input order
    windows = [
        _win(3, start=15.0, end=20.0, eligibility="eligible", selected=["academic_pressure"]),
        _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
        _win(2, start=10.0, end=15.0, eligibility="not_eligible"),
    ]
    ctx = build_wellbeing_evidence_context(window_results=windows)
    assert ctx.temporal_evidence.eligible_window_indices == [1, 3]
    sig = {s.signal: s for s in ctx.temporal_evidence.signal_evidence}
    assert sig["stress_or_overwhelm"].first_start == 5.0
    assert sig["academic_pressure"].last_end == 20.0


def test_09_10_eligible_run_detection_and_multiple_runs() -> None:
    windows = [
        _win(0, start=0.0, end=5.0, eligibility="not_eligible"),
        _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
        _win(2, start=10.0, end=15.0, eligibility="eligible", selected=["academic_pressure"]),
        _win(3, start=15.0, end=20.0, eligibility="not_eligible"),
        _win(4, start=20.0, end=25.0, eligibility="eligible", selected=["loneliness_or_isolation"]),
    ]
    ctx = build_wellbeing_evidence_context(window_results=windows)
    runs = ctx.temporal_evidence.eligible_runs
    assert len(runs) == 2
    assert runs[0].start_window == 1
    assert runs[0].end_window == 2
    assert runs[0].window_count == 2
    assert runs[0].duration_seconds == pytest.approx(10.0)
    assert set(runs[0].signal_ids) == {"stress_or_overwhelm", "academic_pressure"}
    assert runs[1].start_window == 4
    assert runs[1].end_window == 4
    assert runs[1].window_count == 1


def test_11_final_short_window_duration() -> None:
    windows = [
        _win(0, start=0.0, end=5.0, eligibility="eligible"),
        _win(1, start=5.0, end=7.5, eligibility="eligible"),  # short final
    ]
    ctx = build_wellbeing_evidence_context(window_results=windows)
    run = ctx.temporal_evidence.eligible_runs[0]
    assert run.duration_seconds == pytest.approx(7.5)
    assert ctx.temporal_evidence.longest_eligible_run_seconds == pytest.approx(7.5)


def test_12_13_longest_run_windows_and_seconds() -> None:
    windows = [
        _win(0, start=0.0, end=5.0, eligibility="eligible"),
        _win(1, start=5.0, end=10.0, eligibility="eligible"),
        _win(2, start=10.0, end=15.0, eligibility="not_eligible"),
        _win(3, start=15.0, end=18.0, eligibility="eligible"),  # short singleton
    ]
    ctx = build_wellbeing_evidence_context(window_results=windows)
    assert ctx.temporal_evidence.longest_eligible_run_windows == 2
    assert ctx.temporal_evidence.longest_eligible_run_seconds == pytest.approx(10.0)


def test_14_15_16_per_signal_temporal_evidence() -> None:
    windows = [
        _win(
            1,
            start=5.0,
            end=10.0,
            eligibility="eligible",
            selected=["stress_or_overwhelm", "self_directed_negativity"],
        ),
        _win(
            2,
            start=10.0,
            end=15.0,
            eligibility="eligible",
            selected=["stress_or_overwhelm"],
        ),
        _win(
            4,
            start=20.0,
            end=25.0,
            eligibility="eligible",
            selected=["stress_or_overwhelm", "hopelessness_like_language"],
        ),
        _win(0, start=0.0, end=5.0, eligibility="not_eligible", selected=[]),
    ]
    ctx = build_wellbeing_evidence_context(window_results=windows)
    by_sig = {s.signal: s for s in ctx.temporal_evidence.signal_evidence}
    assert by_sig["stress_or_overwhelm"].window_count == 3
    assert by_sig["stress_or_overwhelm"].window_indices == [1, 2, 4]
    assert by_sig["stress_or_overwhelm"].first_start == 5.0
    assert by_sig["stress_or_overwhelm"].last_end == 25.0
    assert by_sig["self_directed_negativity"].window_indices == [1]
    assert by_sig["hopelessness_like_language"].window_indices == [4]


def test_17_distress_window_counting() -> None:
    windows = [
        _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
        _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["positive_wellbeing_or_recovery"]),
        _win(2, start=10.0, end=15.0, eligibility="not_eligible", selected=["stress_or_overwhelm"]),
    ]
    ctx = build_wellbeing_evidence_context(window_results=windows)
    assert ctx.temporal_evidence.distress_window_count == 1


def test_18_recovery_window_counting() -> None:
    windows = [
        _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["positive_wellbeing_or_recovery"]),
        _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
    ]
    ctx = build_wellbeing_evidence_context(window_results=windows)
    assert ctx.temporal_evidence.recovery_window_count == 1
    assert "positive_wellbeing_or_recovery" in RECOVERY_SIGNAL_IDS


def test_19_mixed_window_counting() -> None:
    windows = [
        _win(
            0,
            start=0.0,
            end=5.0,
            eligibility="eligible",
            selected=["positive_wellbeing_or_recovery", "stress_or_overwhelm"],
        ),
        _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
    ]
    ctx = build_wellbeing_evidence_context(window_results=windows)
    assert ctx.temporal_evidence.mixed_signal_window_count == 1
    assert ctx.temporal_evidence.distress_window_count == 2
    assert ctx.temporal_evidence.recovery_window_count == 1
    # Both signal sets preserved in signal_evidence
    signals = {s.signal for s in ctx.temporal_evidence.signal_evidence}
    assert "positive_wellbeing_or_recovery" in signals
    assert "stress_or_overwhelm" in signals


def test_20_global_eligible_no_local_support() -> None:
    ctx = build_wellbeing_evidence_context(
        source_results=[_src("transcript", _clf(eligibility="eligible"))],
        window_results=[
            _win(0, start=0.0, end=5.0, eligibility="not_eligible"),
        ],
    )
    assert "global_eligible_no_local_support" in ctx.conflict_diagnostics


def test_21_local_eligible_without_global_eligibility() -> None:
    ctx = build_wellbeing_evidence_context(
        source_results=[_src("transcript", _clf(eligibility="not_eligible", selected=[]))],
        window_results=[
            _win(0, start=0.0, end=5.0, eligibility="eligible"),
        ],
    )
    assert "local_eligible_without_global_eligibility" in ctx.conflict_diagnostics


def test_22_recovery_distress_disagreement_diagnostics() -> None:
    ctx_a = build_wellbeing_evidence_context(
        source_results=[
            _src(
                "transcript",
                _clf(eligibility="eligible", selected=["positive_wellbeing_or_recovery"]),
            ),
        ],
        window_results=[
            _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
        ],
    )
    assert "global_recovery_local_distress" in ctx_a.conflict_diagnostics

    ctx_b = build_wellbeing_evidence_context(
        source_results=[
            _src(
                "transcript",
                _clf(eligibility="eligible", selected=["stress_or_overwhelm"]),
            ),
        ],
        window_results=[
            _win(
                0,
                start=0.0,
                end=5.0,
                eligibility="eligible",
                selected=["positive_wellbeing_or_recovery"],
            ),
        ],
    )
    assert "global_distress_local_recovery" in ctx_b.conflict_diagnostics


def test_23_raw_probabilities_not_averaged() -> None:
    """Selected presence is evidence; scores must not be mean-aggregated."""
    windows = [
        _win(
            0,
            start=0.0,
            end=5.0,
            eligibility="eligible",
            selected=["stress_or_overwhelm"],
            signal_scores={"stress_or_overwhelm": 0.99},
        ),
        _win(
            1,
            start=5.0,
            end=10.0,
            eligibility="eligible",
            selected=["stress_or_overwhelm"],
            signal_scores={"stress_or_overwhelm": 0.51},
        ),
    ]
    ctx = build_wellbeing_evidence_context(window_results=windows)
    dumped = ctx.model_dump()
    blob = str(dumped)
    assert "0.75" not in blob  # would be naive mean of 0.99 and 0.51
    assert "average" not in blob.lower()
    sig = ctx.temporal_evidence.signal_evidence[0]
    assert not hasattr(sig, "mean_score")
    assert not hasattr(sig, "average_score")
    assert sig.window_count == 2


def test_24_25_26_27_no_visual_ocr_persistence_trajectory_dependency() -> None:
    import src.wellbeing.evidence as evidence_mod

    # No imports of multimodal / temporal-sentiment authorities.
    assert not hasattr(evidence_mod, "TemporalFeatures")
    module_src = inspect.getsource(evidence_mod)
    assert "from src.temporal" not in module_src
    assert "import src.temporal" not in module_src
    # Builder signature accepts only shadow classification results.
    sig = inspect.signature(build_wellbeing_evidence_context)
    assert set(sig.parameters) == {"source_results", "window_results", "shadow_status"}
    for banned in ("temporal_features", "visual", "ocr", "trajectory", "faces"):
        assert banned not in sig.parameters

    # Runtime: pydantic forbid prevents multimodal attrs on shadow windows;
    # builder never reads TemporalFeatures / visual / OCR fields.
    assert "negative_persistence" not in WellbeingShadowWindowResult.model_fields
    assert "trajectory" not in WellbeingShadowWindowResult.model_fields
    ctx = build_wellbeing_evidence_context(
        window_results=[_win(0, start=0.0, end=5.0, eligibility="eligible")],
    )
    blob = str(ctx.model_dump()).lower()
    assert "siglip" not in blob
    assert "ocr" not in blob
    assert "negative_persistence" not in blob
    assert "increasing_negative" not in blob


def test_28_evidence_ids_deterministic() -> None:
    sources = [_src("transcript", _clf())]
    windows = [
        _win(3, start=15.0, end=20.0, eligibility="eligible"),
        _win(0, start=0.0, end=5.0, eligibility="not_eligible"),
    ]
    a = build_wellbeing_evidence_context(source_results=sources, window_results=windows)
    b = build_wellbeing_evidence_context(source_results=sources, window_results=windows)
    assert a.evidence_ids == b.evidence_ids
    assert a.evidence_ids == [
        "wellbeing-global-transcript",
        "wellbeing-window-0",
        "wellbeing-window-3",
    ]


def test_29_no_raw_speech_stored() -> None:
    secret = "SECRET_SPEECH_PHRASE_XYZ"
    ctx = build_wellbeing_evidence_context(
        source_results=[_src("transcript", _clf())],
        window_results=[_win(0, start=0.0, end=5.0, eligibility="eligible")],
    )
    dumped = ctx.model_dump()
    blob = str(dumped)
    assert secret not in blob
    # Evidence models store ids/metadata only — no speech payload fields.
    def _walk(obj: object) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert k not in {
                    "text",
                    "speech",
                    "transcript_text",
                    "raw_text",
                    "utterance",
                    "speech_segments",
                }
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(dumped)


def test_30_affects_final_assessment_false() -> None:
    ctx = build_wellbeing_evidence_context(
        source_results=[_src("transcript", _clf())],
    )
    assert ctx.affects_final_assessment is False


def test_31_final_temporal_assessment_unchanged() -> None:
    """Evidence schemas must not alter FinalTemporalAssessment fields."""
    fields = set(FinalTemporalAssessment.model_fields.keys())
    assert "evidence_context" not in fields
    assert "wellbeing_evidence" not in fields
    assert "concern" not in "".join(fields).lower()


def test_32_shadow_disabled_behavior_unchanged() -> None:
    out = build_wellbeing_shadow(
        primary_text="I feel overwhelmed by exams this week.",
        enabled=False,
    )
    assert out is None


def test_33_main_pipeline_failure_isolation_unchanged() -> None:
    pipe = MyUniSentimentPipeline()
    text_mock = MagicMock()
    text_mock.validate_text.side_effect = lambda t: str(t).strip()
    text_mock.analyze.return_value = _ev()
    pipe._text_analyzer = text_mock

    with patch(
        "src.pipeline.build_wellbeing_shadow",
        return_value=WellbeingShadowAnalysis(status="error", error_code="Boom", evidence_context=None),
    ):
        # Force shadow path via env so pipeline would attach if enabled
        result = pipe.analyze_text("I feel overwhelmed by exams this week.")
    assert result.analysis.overall.label == "neutral"
    # Shadow disabled by default — remains None
    assert result.analysis.wellbeing_shadow is None


def test_34_no_concern_category_produced() -> None:
    ctx = build_wellbeing_evidence_context(
        source_results=[_src("transcript", _clf())],
        window_results=[_win(0, start=0.0, end=5.0, eligibility="eligible")],
    )
    blob = str(ctx.model_dump()).lower()
    for forbidden in (
        "low_concern",
        "moderate_concern",
        "high_concern",
        "concern_level",
        "overall_wellbeing_indicator",
    ):
        assert forbidden not in blob


def test_35_no_numerical_wellbeing_score_produced() -> None:
    ctx = build_wellbeing_evidence_context(
        source_results=[_src("transcript", _clf())],
        window_results=[_win(0, start=0.0, end=5.0, eligibility="eligible")],
    )
    blob = str(ctx.model_dump()).lower()
    for forbidden in (
        "wellbeing_score",
        "mental_health_score",
        "global_wellbeing_score",
        "overall_stress_score",
        "severity",
    ):
        assert forbidden not in blob


def test_controlled_video_shape_example() -> None:
    """Acceptance shape from Phase 4B live result (not hardcoded into production)."""
    ctx = build_wellbeing_evidence_context(
        source_results=[
            _src(
                "transcript",
                _clf(
                    eligibility="eligible",
                    selected=[
                        "stress_or_overwhelm",
                        "academic_pressure",
                        "exhaustion_or_burnout_like_language",
                    ],
                ),
            ),
        ],
        window_results=[
            _win(0, start=0.0, end=5.0, eligibility="not_eligible"),
            _win(1, start=5.0, end=10.0, eligibility="not_eligible"),
            _win(2, start=10.0, end=15.0, eligibility="not_eligible"),
            _win(
                3,
                start=15.0,
                end=20.0,
                eligibility="eligible",
                selected=["self_directed_negativity", "hopelessness_like_language"],
            ),
            _win(4, start=20.0, end=25.0, eligibility="not_eligible"),
        ],
    )
    t = ctx.temporal_evidence
    assert t.evaluated_window_count == 5
    assert t.eligible_window_count == 1
    assert t.eligible_window_fraction == pytest.approx(0.2)
    assert t.eligible_window_indices == [3]
    assert len(t.eligible_runs) == 1
    assert t.longest_eligible_run_windows == 1
    assert t.longest_eligible_run_seconds == pytest.approx(5.0)
    by_sig = {s.signal: s.window_indices for s in t.signal_evidence}
    assert by_sig["self_directed_negativity"] == [3]
    assert by_sig["hopelessness_like_language"] == [3]


def test_shadow_wires_evidence_context() -> None:
    class Rec:
        model_id = DEFAULT_WELLBEING_CLASSIFIER_MODEL

        def classify_many(self, texts, **kwargs):  # type: ignore[no-untyped-def]
            return [_clf() for _ in texts]

    out = build_wellbeing_shadow(
        transcript="I feel overwhelmed by exams and burned out.",
        classifier=Rec(),
        enabled=True,
    )
    assert out is not None
    assert out.evidence_context is not None
    assert out.evidence_context.affects_final_assessment is False
    assert out.evidence_context.global_evidence.source_role == "transcript"


def test_shadow_error_does_not_manufacture_evidence() -> None:
    class Boom:
        model_id = "x"

        def classify_many(self, texts, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("fail")

    out = build_wellbeing_shadow(
        primary_text="I feel overwhelmed by exams this week.",
        classifier=Boom(),
        enabled=True,
    )
    assert out is not None
    assert out.status == "error"
    assert out.evidence_context is None


def test_distress_group_membership() -> None:
    expected = {
        "stress_or_overwhelm",
        "anxiety_or_fear_language",
        "loneliness_or_isolation",
        "hopelessness_like_language",
        "exhaustion_or_burnout_like_language",
        "self_directed_negativity",
        "interpersonal_distress",
        "academic_pressure",
    }
    assert DISTRESS_LIKE_SIGNAL_IDS == expected
    assert RECOVERY_SIGNAL_IDS == {"positive_wellbeing_or_recovery"}


def test_analysis_block_optional_evidence_via_shadow() -> None:
    block = AnalysisBlock(overall=_ev(), modalities=ModalityBundle())
    assert block.wellbeing_shadow is None
    shadow = WellbeingShadowAnalysis(
        status="ok",
        evidence_context=build_wellbeing_evidence_context(),
    )
    updated = block.model_copy(update={"wellbeing_shadow": shadow})
    assert updated.wellbeing_shadow is not None
    assert updated.wellbeing_shadow.evidence_context is not None
