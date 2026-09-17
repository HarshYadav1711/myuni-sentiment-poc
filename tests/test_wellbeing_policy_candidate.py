"""Phase 4C.2 shadow wellbeing policy candidate tests (fixtures only — no ML)."""

from __future__ import annotations

import inspect
from typing import Optional, Sequence
from unittest.mock import MagicMock, patch

from src.pipeline import MyUniSentimentPipeline
from src.schemas import FinalTemporalAssessment, SentimentEvidence
from src.wellbeing.evidence import build_wellbeing_evidence_context
from src.wellbeing.policy_candidate import (
    POLICY_VERSION,
    build_distress_runs,
    build_recovery_runs,
    build_wellbeing_policy_candidate,
)
from src.wellbeing.schemas import (
    ExclusiveClassification,
    SelfAttributionResult,
    SignalScore,
    WellbeingClassificationResult,
    WellbeingEvidenceContext,
    WellbeingGlobalEvidence,
    WellbeingShadowAnalysis,
    WellbeingShadowSourceResult,
    WellbeingShadowWindowResult,
    WellbeingSignalTemporalEvidence,
    WellbeingTemporalEvidence,
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


def _signals(selected: Sequence[str]) -> list[SignalScore]:
    out: list[SignalScore] = []
    for sid in sorted(set(selected) | {"stress_or_overwhelm"}):
        sel = sid in selected
        out.append(
            SignalScore(
                signal=sid,
                score=0.9 if sel else 0.1,
                threshold_passed=sel,
                selected=sel,
            ),
        )
    return out


def _clf(
    *,
    eligibility: str = "eligible",
    selected: Optional[Sequence[str]] = None,
    status: str = "ok",
) -> WellbeingClassificationResult:
    selected = list(selected) if selected is not None else (
        ["stress_or_overwhelm"] if eligibility == "eligible" else []
    )
    return WellbeingClassificationResult(
        model_id="stub-model",
        status=status,  # type: ignore[arg-type]
        relevance=ExclusiveClassification(label="personal_wellbeing", scores={}),
        target=ExclusiveClassification(label="self", scores={}),
        signals=_signals(selected),
        self_attribution=SelfAttributionResult(
            status="ok",
            final_label="self_experience" if eligibility == "eligible" else "unclear",
            label="self_experience" if eligibility == "eligible" else "unclear",
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
    )


def _win(
    index: int,
    *,
    start: float,
    end: float,
    eligibility: str = "not_eligible",
    selected: Optional[Sequence[str]] = None,
) -> WellbeingShadowWindowResult:
    return WellbeingShadowWindowResult(
        window_index=index,
        start=start,
        end=end,
        classification=_clf(
            eligibility=eligibility,
            selected=list(selected) if selected is not None else (
                ["stress_or_overwhelm"] if eligibility == "eligible" else []
            ),
        ),
        input_character_count=20,
        usable_window=True,
    )


def _evidence(
    *,
    global_eligibility: Optional[str] = "eligible",
    global_selected: Optional[Sequence[str]] = None,
    global_status: str = "ok",
    global_present: bool = True,
    windows: Optional[list[WellbeingShadowWindowResult]] = None,
    evidence_status: str = "ok",
) -> WellbeingEvidenceContext:
    sources: list[WellbeingShadowSourceResult] = []
    if global_present:
        if global_eligibility is None:
            sources.append(_src("transcript", None))
        else:
            sources.append(
                _src(
                    "transcript",
                    _clf(
                        eligibility=global_eligibility,
                        selected=list(global_selected) if global_selected is not None else (
                            ["stress_or_overwhelm"]
                            if global_eligibility == "eligible"
                            else []
                        ),
                        status=global_status,
                    ),
                ),
            )
    return build_wellbeing_evidence_context(
        source_results=sources,
        window_results=windows or [],
        shadow_status=evidence_status,
    )


# ---------------------------------------------------------------------------
# Core decision-table tests
# ---------------------------------------------------------------------------


def test_01_no_evidence_insufficient() -> None:
    out = build_wellbeing_policy_candidate(None)
    assert out.indicator == "insufficient_evidence"
    assert out.status == "unavailable"


def test_02_global_missing_insufficient() -> None:
    ev = _evidence(global_present=False, windows=[])
    out = build_wellbeing_policy_candidate(ev)
    assert out.indicator == "insufficient_evidence"
    assert "global_unavailable" in out.reason_codes


def test_03_global_classifier_error_insufficient() -> None:
    ev = _evidence(global_eligibility="eligible", global_status="error")
    # Force classification_status error on global
    ev = ev.model_copy(
        update={
            "global_evidence": ev.global_evidence.model_copy(
                update={"classification_status": "error"},
            ),
        },
    )
    out = build_wellbeing_policy_candidate(ev)
    assert out.indicator == "insufficient_evidence"
    assert out.status == "unavailable"
    assert "global_unavailable" in out.reason_codes


def test_04_global_not_eligible_insufficient() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(global_eligibility="not_eligible", global_selected=[]),
    )
    assert out.indicator == "insufficient_evidence"
    assert "global_not_eligible" in out.reason_codes


def test_05_global_uncertain_insufficient() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(global_eligibility="uncertain", global_selected=[]),
    )
    assert out.indicator == "insufficient_evidence"
    assert "global_uncertain" in out.reason_codes


def test_06_local_only_eligible_insufficient() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(
            global_eligibility="not_eligible",
            global_selected=[],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="eligible"),
            ],
        ),
    )
    assert out.indicator == "insufficient_evidence"
    assert "local_eligible_without_global_eligibility" in out.reason_codes


def test_07_global_recovery_only_low() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(
            global_selected=["positive_wellbeing_or_recovery"],
            windows=[],
        ),
    )
    assert out.indicator == "low_concern"
    assert out.status == "ok"
    assert "global_personal_recovery" in out.reason_codes


def test_08_global_no_distress_eligible_low() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(global_selected=[]),
    )
    assert out.indicator == "low_concern"
    assert out.global_eligible is True


def test_09_global_distress_no_local_moderate() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(
            global_selected=["stress_or_overwhelm", "academic_pressure"],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="not_eligible"),
            ],
        ),
    )
    assert out.indicator == "moderate_concern"
    assert "global_eligible_no_local_support" in out.reason_codes
    assert "global_personal_distress" in out.reason_codes


def test_10_global_distress_one_local_moderate() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(
            global_selected=["stress_or_overwhelm"],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="not_eligible"),
                _win(
                    1,
                    start=5.0,
                    end=10.0,
                    eligibility="eligible",
                    selected=["stress_or_overwhelm"],
                ),
            ],
        ),
    )
    assert out.indicator == "moderate_concern"
    assert out.distress_pattern in {"isolated", "mixed_with_recovery"}


def test_11_global_distress_two_nonconsecutive_moderate() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(
            global_selected=["stress_or_overwhelm"],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                _win(1, start=5.0, end=10.0, eligibility="not_eligible"),
                _win(2, start=10.0, end=15.0, eligibility="eligible", selected=["academic_pressure"]),
            ],
        ),
    )
    assert out.indicator == "moderate_concern"
    runs = build_distress_runs(
        _evidence(
            global_selected=["stress_or_overwhelm"],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                _win(1, start=5.0, end=10.0, eligibility="not_eligible"),
                _win(2, start=10.0, end=15.0, eligibility="eligible", selected=["academic_pressure"]),
            ],
        ),
    )
    assert len(runs) == 2
    assert all(r.window_count == 1 for r in runs)


def test_12_global_distress_two_consecutive_high() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(
            global_selected=["stress_or_overwhelm"],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["academic_pressure"]),
            ],
        ),
    )
    assert out.indicator == "high_concern"
    assert out.status == "ok"


def test_13_three_consecutive_distress_high() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(
            global_selected=["exhaustion_or_burnout_like_language"],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["anxiety_or_fear_language"]),
                _win(2, start=10.0, end=15.0, eligibility="eligible", selected=["loneliness_or_isolation"]),
            ],
        ),
    )
    assert out.indicator == "high_concern"


def test_14_isolated_hopelessness_does_not_force_high() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(
            global_selected=["stress_or_overwhelm"],
            windows=[
                _win(
                    3,
                    start=15.0,
                    end=20.0,
                    eligibility="eligible",
                    selected=["hopelessness_like_language"],
                ),
            ],
        ),
    )
    assert out.indicator == "moderate_concern"
    assert out.indicator != "high_concern"


def test_15_isolated_self_directed_negativity_does_not_force_high() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(
            global_selected=["stress_or_overwhelm"],
            windows=[
                _win(
                    1,
                    start=5.0,
                    end=10.0,
                    eligibility="eligible",
                    selected=["self_directed_negativity"],
                ),
            ],
        ),
    )
    assert out.indicator == "moderate_concern"


def test_16_isolated_anxiety_does_not_force_high() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(
            global_selected=["anxiety_or_fear_language"],
            windows=[
                _win(
                    0,
                    start=0.0,
                    end=5.0,
                    eligibility="eligible",
                    selected=["anxiety_or_fear_language"],
                ),
            ],
        ),
    )
    assert out.indicator == "moderate_concern"


def test_17_different_distress_labels_consecutive_still_persistent() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(
            global_selected=["stress_or_overwhelm"],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["hopelessness_like_language"]),
                _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["self_directed_negativity"]),
            ],
        ),
    )
    assert out.indicator == "high_concern"


def test_18_recovery_only_repeated_windows_low() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(
            global_selected=["positive_wellbeing_or_recovery"],
            windows=[
                _win(
                    0,
                    start=0.0,
                    end=5.0,
                    eligibility="eligible",
                    selected=["positive_wellbeing_or_recovery"],
                ),
                _win(
                    1,
                    start=5.0,
                    end=10.0,
                    eligibility="eligible",
                    selected=["positive_wellbeing_or_recovery"],
                ),
            ],
        ),
    )
    assert out.indicator == "low_concern"


def test_19_global_recovery_persistent_local_distress_conflict() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(
            global_selected=["positive_wellbeing_or_recovery"],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["academic_pressure"]),
            ],
        ),
    )
    assert out.status == "conflict"
    assert out.indicator == "insufficient_evidence"
    assert "mixed_distress_recovery" in out.reason_codes


def test_20_global_distress_recurrent_local_recovery_conflict() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(
            global_selected=["stress_or_overwhelm"],
            windows=[
                _win(
                    0,
                    start=0.0,
                    end=5.0,
                    eligibility="eligible",
                    selected=["positive_wellbeing_or_recovery"],
                ),
                _win(1, start=5.0, end=10.0, eligibility="not_eligible"),
                _win(
                    2,
                    start=10.0,
                    end=15.0,
                    eligibility="eligible",
                    selected=["positive_wellbeing_or_recovery"],
                ),
            ],
        ),
    )
    assert out.status == "conflict"
    assert out.indicator == "insufficient_evidence"


def test_21_global_distress_and_recovery_conflict() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(
            global_selected=["stress_or_overwhelm", "positive_wellbeing_or_recovery"],
        ),
    )
    assert out.status == "conflict"
    assert out.indicator == "insufficient_evidence"
    assert "mixed_distress_recovery" in out.reason_codes


def test_22_mixed_window_preserved() -> None:
    ev = _evidence(
        global_selected=["stress_or_overwhelm"],
        windows=[
            _win(
                0,
                start=0.0,
                end=5.0,
                eligibility="eligible",
                selected=["stress_or_overwhelm", "positive_wellbeing_or_recovery"],
            ),
        ],
    )
    assert ev.temporal_evidence.mixed_signal_window_count == 1
    out = build_wellbeing_policy_candidate(ev)
    # Global distress + isolated local (mixed window) → moderate, not averaged away
    assert out.indicator == "moderate_concern"
    assert out.distress_pattern == "mixed_with_recovery"


def test_23_distress_run_adjacency() -> None:
    ev = _evidence(
        global_selected=["stress_or_overwhelm"],
        windows=[
            _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
            _win(2, start=10.0, end=15.0, eligibility="eligible", selected=["academic_pressure"]),
        ],
    )
    runs = build_distress_runs(ev)
    assert len(runs) == 1
    assert runs[0].window_indices == (1, 2)
    assert runs[0].window_count == 2


def test_24_nonconsecutive_indices_break_runs() -> None:
    ev = _evidence(
        global_selected=["stress_or_overwhelm"],
        windows=[
            _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
            _win(2, start=10.0, end=15.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
        ],
    )
    runs = build_distress_runs(ev)
    assert len(runs) == 2


def test_25_final_short_window_duration_preserved() -> None:
    ev = _evidence(
        global_selected=["stress_or_overwhelm"],
        windows=[
            _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
            _win(1, start=5.0, end=7.5, eligibility="eligible", selected=["academic_pressure"]),
        ],
    )
    runs = build_distress_runs(ev)
    assert len(runs) == 1
    assert runs[0].duration_seconds == 7.5


def test_26_recovery_run_adjacency() -> None:
    ev = _evidence(
        global_selected=["positive_wellbeing_or_recovery"],
        windows=[
            _win(
                0,
                start=0.0,
                end=5.0,
                eligibility="eligible",
                selected=["positive_wellbeing_or_recovery"],
            ),
            _win(
                1,
                start=5.0,
                end=10.0,
                eligibility="eligible",
                selected=["positive_wellbeing_or_recovery"],
            ),
        ],
    )
    runs = build_recovery_runs(ev)
    assert len(runs) == 1
    assert runs[0].window_indices == (0, 1)


def test_27_evidence_ids_preserved() -> None:
    ev = _evidence(
        global_selected=["stress_or_overwhelm"],
        windows=[
            _win(3, start=15.0, end=20.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
        ],
    )
    out = build_wellbeing_policy_candidate(ev)
    assert "wellbeing-global-transcript" in out.supporting_evidence_ids
    assert "wellbeing-window-3" in out.supporting_evidence_ids
    for eid in out.supporting_evidence_ids + out.conflict_evidence_ids:
        assert eid in ev.evidence_ids


def test_28_reason_codes_deterministic() -> None:
    ev = _evidence(global_selected=["stress_or_overwhelm"])
    a = build_wellbeing_policy_candidate(ev)
    b = build_wellbeing_policy_candidate(ev)
    assert a.reason_codes == b.reason_codes
    assert a.model_dump() == b.model_dump()


def test_29_policy_version_correct() -> None:
    out = build_wellbeing_policy_candidate(_evidence())
    assert out.policy_version == "phase4c2-v1"
    assert out.policy_version == POLICY_VERSION


def test_30_affects_final_assessment_false() -> None:
    out = build_wellbeing_policy_candidate(_evidence())
    assert out.affects_final_assessment is False


def test_31_no_numeric_wellbeing_score() -> None:
    out = build_wellbeing_policy_candidate(
        _evidence(global_selected=["stress_or_overwhelm"]),
    )
    blob = str(out.model_dump()).lower()
    for forbidden in (
        "wellbeing_score",
        "mental_health_score",
        "confidence",
        "0-100",
        "severity",
    ):
        assert forbidden not in blob


def test_32_33_34_35_36_37_no_forbidden_dependencies() -> None:
    import src.wellbeing.policy_candidate as mod

    src = inspect.getsource(mod)
    assert "from src.temporal" not in src
    assert "import src.temporal" not in src
    assert "from src.openrouter" not in src
    assert "import src.openrouter" not in src
    assert "classify_many" not in src
    assert "get_wellbeing_classifier" not in src
    sig = inspect.signature(build_wellbeing_policy_candidate)
    assert list(sig.parameters) == ["evidence"]
    for banned in (
        "temporal_features",
        "visual",
        "ocr",
        "trajectory",
        "negative_persistence",
        "transcript",
        "faces",
    ):
        assert banned not in sig.parameters
    # No numeric score fields on the candidate schema.
    fields = set(mod.WellbeingPolicyCandidate.model_fields.keys()) if hasattr(mod, "WellbeingPolicyCandidate") else set()
    from src.wellbeing.schemas import WellbeingPolicyCandidate

    fields = set(WellbeingPolicyCandidate.model_fields.keys())
    assert "score" not in fields
    assert "confidence" not in fields


def test_38_final_temporal_assessment_unchanged() -> None:
    fields = set(FinalTemporalAssessment.model_fields.keys())
    assert "policy_candidate" not in fields
    assert "wellbeing_policy" not in fields


def test_39_shadow_disabled_unchanged() -> None:
    assert build_wellbeing_shadow(primary_text="I feel overwhelmed by exams.", enabled=False) is None


def test_40_shadow_failure_isolation_unchanged() -> None:
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
    assert out.policy_candidate is None

    pipe = MyUniSentimentPipeline()
    text_mock = MagicMock()
    text_mock.validate_text.side_effect = lambda t: str(t).strip()
    text_mock.analyze.return_value = _ev()
    pipe._text_analyzer = text_mock
    result = pipe.analyze_text("I feel overwhelmed by exams this week.")
    assert result.analysis.overall.label == "neutral"
    assert result.analysis.wellbeing_shadow is None


def test_controlled_video_conceptual_moderate() -> None:
    """Phase 4B live shape → moderate (isolated local distress, not persistent)."""
    out = build_wellbeing_policy_candidate(
        _evidence(
            global_selected=[
                "stress_or_overwhelm",
                "academic_pressure",
                "exhaustion_or_burnout_like_language",
            ],
            windows=[
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
        ),
    )
    assert out.indicator == "moderate_concern"
    assert out.status == "ok"


def test_shadow_wires_policy_candidate() -> None:
    class Rec:
        model_id = "stub"

        def classify_many(self, texts, **kwargs):  # type: ignore[arg-type]
            return [_clf(selected=["stress_or_overwhelm"]) for _ in texts]

    out = build_wellbeing_shadow(
        transcript="I feel overwhelmed by exams and burned out.",
        classifier=Rec(),
        enabled=True,
    )
    assert out is not None
    assert out.policy_candidate is not None
    assert out.policy_candidate.affects_final_assessment is False
    assert out.policy_candidate.policy_version == "phase4c2-v1"
    assert out.policy_candidate.indicator in {
        "low_concern",
        "moderate_concern",
        "high_concern",
        "insufficient_evidence",
    }


def test_local_support_levels() -> None:
    none = build_wellbeing_policy_candidate(_evidence(windows=[]))
    assert none.local_support_level == "none"

    isolated = build_wellbeing_policy_candidate(
        _evidence(
            windows=[_win(0, start=0.0, end=5.0, eligibility="eligible")],
        ),
    )
    assert isolated.local_support_level == "isolated"

    recurrent = build_wellbeing_policy_candidate(
        _evidence(
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="eligible"),
                _win(1, start=5.0, end=10.0, eligibility="not_eligible"),
                _win(2, start=10.0, end=15.0, eligibility="eligible"),
            ],
        ),
    )
    assert recurrent.local_support_level == "recurrent"

    persistent = build_wellbeing_policy_candidate(
        _evidence(
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="eligible"),
                _win(1, start=5.0, end=10.0, eligibility="eligible"),
            ],
        ),
    )
    assert persistent.local_support_level == "persistent"


def test_direct_fixture_context_without_shadow_builder() -> None:
    """Policy accepts structured WellbeingEvidenceContext fixtures directly."""
    ctx = WellbeingEvidenceContext(
        status="ok",
        global_evidence=WellbeingGlobalEvidence(
            source_present=True,
            source_role="transcript",
            classification_status="ok",
            relevance="personal_wellbeing",
            target="self",
            eligibility_status="eligible",
            personal_wellbeing_eligible=True,
            final_attribution="self_experience",
            selected_signals=["stress_or_overwhelm"],
        ),
        temporal_evidence=WellbeingTemporalEvidence(
            evaluated_window_count=1,
            eligible_window_count=1,
            eligible_window_indices=[0],
            eligible_window_fraction=1.0,
            distress_window_count=1,
            signal_evidence=[
                WellbeingSignalTemporalEvidence(
                    signal="stress_or_overwhelm",
                    window_count=1,
                    window_indices=[0],
                    first_start=0.0,
                    last_end=5.0,
                ),
            ],
        ),
        evidence_ids=["wellbeing-global-transcript", "wellbeing-window-0"],
    )
    out = build_wellbeing_policy_candidate(ctx)
    assert out.indicator == "moderate_concern"
