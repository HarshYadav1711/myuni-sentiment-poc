"""Phase 4C.4 — shadow chain replay cases.

Structured Phase 4B-style classifier outputs for end-to-end replay of the
shadow source/window → evidence → policy composition path.

No raw text. No model inference. Expectations are HUMAN-WRITTEN and must
not be derived by calling production evidence/policy builders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from src.config import DEFAULT_WELLBEING_CLASSIFIER_MODEL
from src.wellbeing.schemas import (
    AttributionEvidence,
    ExclusiveClassification,
    SelfAttributionResult,
    SignalScore,
    WellbeingClassificationResult,
    WellbeingShadowSourceResult,
    WellbeingShadowWindowResult,
)

POLICY_VERSION = "phase4c2-v1"


@dataclass(frozen=True)
class ReplayExpectation:
    """Human-authored expected chain outputs (independent of production builders)."""

    evidence_status: str
    eligible_window_count: int
    eligible_window_indices: tuple[int, ...] = ()
    conflict_diagnostics: tuple[str, ...] = ()
    policy_status: str = "insufficient_evidence"
    policy_indicator: str = "insufficient_evidence"
    required_reason_codes: tuple[str, ...] = ()
    forbidden_reason_codes: tuple[str, ...] = ()
    affects_final_assessment: bool = False


@dataclass(frozen=True)
class ShadowReplayCase:
    """One structured classifier-output replay fixture."""

    case_id: str
    description: str
    modality: str  # text | audio | caption | video | failure | conflict | regression
    source_results: tuple[WellbeingShadowSourceResult, ...]
    window_results: tuple[WellbeingShadowWindowResult, ...]
    expected: ReplayExpectation
    tags: tuple[str, ...] = ()
    structural_only: bool = False  # True = schema-valid structural regression, not exact historical scores


def _signals(selected: Sequence[str]) -> list[SignalScore]:
    """Structural signal vector — scores are placeholders, not historical probabilities."""
    ids = sorted(set(selected) | {"stress_or_overwhelm", "positive_wellbeing_or_recovery"})
    out: list[SignalScore] = []
    for sid in ids:
        sel = sid in selected
        out.append(
            SignalScore(
                signal=sid,
                score=0.0,  # structural placeholder — do not invent exact probabilities
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
    relevance: str = "personal_wellbeing",
    target: str = "self",
    attribution: str = "self_experience",
    attribution_status: str = "ok",
) -> WellbeingClassificationResult:
    if selected is None:
        selected = ["stress_or_overwhelm"] if eligibility == "eligible" else []
    return WellbeingClassificationResult(
        model_id=DEFAULT_WELLBEING_CLASSIFIER_MODEL,
        status=status,  # type: ignore[arg-type]
        relevance=ExclusiveClassification(label=relevance, scores={}),
        target=ExclusiveClassification(label=target, scores={}),
        signals=_signals(selected),
        self_attribution=SelfAttributionResult(
            status=attribution_status,  # type: ignore[arg-type]
            final_label=attribution,
            label=attribution,
            attribution_evidence=AttributionEvidence(
                evidence_status="ok" if attribution_status == "ok" else "not_evaluated",
            ),
        ),
        eligibility_status=eligibility,  # type: ignore[arg-type]
        personal_wellbeing_eligible=eligibility == "eligible",
    )


def _src(
    role: str,
    clf: Optional[WellbeingClassificationResult],
) -> WellbeingShadowSourceResult:
    return WellbeingShadowSourceResult(
        source_role=role,  # type: ignore[arg-type]
        classification=clf,
        input_character_count=0,  # no raw text length claim beyond metadata
        provenance={"role": role},
    )


def _win(
    index: int,
    *,
    start: float,
    end: float,
    eligibility: str = "not_eligible",
    selected: Optional[Sequence[str]] = None,
) -> WellbeingShadowWindowResult:
    if selected is None:
        selected = ["stress_or_overwhelm"] if eligibility == "eligible" else []
    return WellbeingShadowWindowResult(
        window_index=index,
        start=start,
        end=end,
        classification=_clf(eligibility=eligibility, selected=selected),
        input_character_count=0,
        usable_window=True,
    )


def _case(
    case_id: str,
    description: str,
    *,
    modality: str,
    sources: Sequence[WellbeingShadowSourceResult],
    windows: Sequence[WellbeingShadowWindowResult] = (),
    expected: ReplayExpectation,
    tags: tuple[str, ...] = (),
    structural_only: bool = False,
) -> ShadowReplayCase:
    return ShadowReplayCase(
        case_id=case_id,
        description=description,
        modality=modality,
        source_results=tuple(sources),
        window_results=tuple(windows),
        expected=expected,
        tags=tags,
        structural_only=structural_only,
    )


def _build_cases() -> list[ShadowReplayCase]:
    c: list[ShadowReplayCase] = []

    # ------------------------------------------------------------------
    # TEXT-LIKE GLOBAL ONLY
    # ------------------------------------------------------------------
    c.append(
        _case(
            "text_personal_distress",
            "Primary text clear personal distress",
            modality="text",
            sources=[_src("primary_text", _clf(selected=["stress_or_overwhelm", "academic_pressure"]))],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                conflict_diagnostics=("global_eligible_no_local_support",),
                policy_status="ok",
                policy_indicator="moderate_concern",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("text", "moderate", "chain1"),
        ),
    )
    c.append(
        _case(
            "text_personal_recovery",
            "Primary text clear recovery",
            modality="text",
            sources=[_src("primary_text", _clf(selected=["positive_wellbeing_or_recovery"]))],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                conflict_diagnostics=("global_eligible_no_local_support",),
                policy_status="ok",
                policy_indicator="low_concern",
                required_reason_codes=("global_personal_recovery",),
            ),
            tags=("text", "low", "chain1"),
        ),
    )
    c.append(
        _case(
            "text_no_distress_signals",
            "Primary text eligible with no selected distress",
            modality="text",
            sources=[_src("primary_text", _clf(selected=[]))],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                conflict_diagnostics=("global_eligible_no_local_support",),
                policy_status="ok",
                policy_indicator="low_concern",
                required_reason_codes=("global_eligible_no_local_support",),
            ),
            tags=("text", "low", "chain1"),
        ),
    )
    c.append(
        _case(
            "text_global_uncertain",
            "Primary text uncertain eligibility",
            modality="text",
            sources=[_src("primary_text", _clf(eligibility="uncertain", selected=[], attribution="unclear"))],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                policy_status="insufficient_evidence",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("global_uncertain",),
            ),
            tags=("text", "insufficient", "chain2"),
        ),
    )
    c.append(
        _case(
            "text_global_not_eligible",
            "Primary text not personal",
            modality="text",
            sources=[
                _src(
                    "primary_text",
                    _clf(
                        eligibility="not_eligible",
                        selected=[],
                        relevance="not_wellbeing_related",
                        target="general_or_unknown",
                        attribution="not_self_experience",
                    ),
                ),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                policy_status="insufficient_evidence",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("global_not_eligible",),
            ),
            tags=("text", "insufficient", "chain2"),
        ),
    )
    c.append(
        _case(
            "text_classifier_unavailable",
            "Primary text classifier unavailable",
            modality="text",
            sources=[
                _src(
                    "primary_text",
                    _clf(eligibility="not_eligible", selected=[], status="classifier_unavailable"),
                ),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                policy_status="unavailable",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("global_unavailable",),
            ),
            tags=("text", "failure", "insufficient"),
        ),
    )

    # ------------------------------------------------------------------
    # AUDIO-LIKE GLOBAL ONLY
    # ------------------------------------------------------------------
    c.append(
        _case(
            "audio_transcript_eligible_distress",
            "Transcript eligible personal distress",
            modality="audio",
            sources=[_src("transcript", _clf(selected=["stress_or_overwhelm"]))],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                conflict_diagnostics=("global_eligible_no_local_support",),
                policy_status="ok",
                policy_indicator="moderate_concern",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("audio", "moderate", "chain4"),
        ),
    )
    c.append(
        _case(
            "audio_transcript_recovery",
            "Transcript eligible recovery",
            modality="audio",
            sources=[_src("transcript", _clf(selected=["positive_wellbeing_or_recovery"]))],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                conflict_diagnostics=("global_eligible_no_local_support",),
                policy_status="ok",
                policy_indicator="low_concern",
                required_reason_codes=("global_personal_recovery",),
            ),
            tags=("audio", "low"),
        ),
    )
    c.append(
        _case(
            "audio_attribution_uncertain",
            "Transcript attribution unclear → uncertain",
            modality="audio",
            sources=[
                _src(
                    "transcript",
                    _clf(eligibility="uncertain", selected=[], attribution="unclear"),
                ),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                policy_status="insufficient_evidence",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("global_uncertain",),
            ),
            tags=("audio", "insufficient"),
        ),
    )
    c.append(
        _case(
            "audio_no_sources",
            "Empty audio-like replay (insufficient structured sources)",
            modality="audio",
            sources=[],
            windows=[],
            expected=ReplayExpectation(
                evidence_status="empty",
                eligible_window_count=0,
                policy_status="insufficient_evidence",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("global_unavailable",),
            ),
            tags=("audio", "failure", "insufficient"),
        ),
    )

    # ------------------------------------------------------------------
    # CAPTION-LIKE GLOBAL ONLY
    # ------------------------------------------------------------------
    c.append(
        _case(
            "caption_eligible_distress",
            "Authored caption eligible distress",
            modality="caption",
            sources=[_src("caption", _clf(selected=["loneliness_or_isolation"]))],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                conflict_diagnostics=("global_eligible_no_local_support",),
                policy_status="ok",
                policy_indicator="moderate_concern",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("caption", "moderate"),
        ),
    )
    c.append(
        _case(
            "caption_eligible_recovery",
            "Authored caption recovery",
            modality="caption",
            sources=[_src("caption", _clf(selected=["positive_wellbeing_or_recovery"]))],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                conflict_diagnostics=("global_eligible_no_local_support",),
                policy_status="ok",
                policy_indicator="low_concern",
                required_reason_codes=("global_personal_recovery",),
            ),
            tags=("caption", "low"),
        ),
    )
    c.append(
        _case(
            "caption_not_personal",
            "Caption not personal wellbeing",
            modality="caption",
            sources=[
                _src(
                    "caption",
                    _clf(
                        eligibility="not_eligible",
                        selected=[],
                        relevance="wellbeing_topic_only",
                        target="institution_or_event",
                        attribution="not_self_experience",
                    ),
                ),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                policy_status="insufficient_evidence",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("global_not_eligible",),
            ),
            tags=("caption", "insufficient"),
        ),
    )

    # ------------------------------------------------------------------
    # VIDEO GLOBAL + WINDOWS
    # ------------------------------------------------------------------
    c.append(
        _case(
            "video_controlled_phase4b_replay",
            "Structured replay of previously observed Phase 4B live video evidence",
            modality="video",
            sources=[
                _src(
                    "transcript",
                    _clf(
                        selected=[
                            "stress_or_overwhelm",
                            "academic_pressure",
                            "exhaustion_or_burnout_like_language",
                        ],
                    ),
                ),
            ],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="not_eligible", selected=[]),
                _win(1, start=5.0, end=10.0, eligibility="not_eligible", selected=[]),
                _win(2, start=10.0, end=15.0, eligibility="not_eligible", selected=[]),
                _win(
                    3,
                    start=15.0,
                    end=20.0,
                    eligibility="eligible",
                    selected=["self_directed_negativity", "hopelessness_like_language"],
                ),
                _win(4, start=20.0, end=25.0, eligibility="not_eligible", selected=[]),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=1,
                eligible_window_indices=(3,),
                conflict_diagnostics=(),
                policy_status="ok",
                policy_indicator="moderate_concern",
                required_reason_codes=("global_personal_distress", "local_distress_present"),
            ),
            tags=("video", "replay", "moderate", "chain5", "chain8"),
        ),
    )
    c.append(
        _case(
            "video_zero_local_eligible",
            "Global distress + zero eligible windows",
            modality="video",
            sources=[_src("transcript", _clf(selected=["stress_or_overwhelm"]))],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="not_eligible", selected=[]),
                _win(1, start=5.0, end=10.0, eligibility="not_eligible", selected=[]),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                conflict_diagnostics=("global_eligible_no_local_support",),
                policy_status="ok",
                policy_indicator="moderate_concern",
                required_reason_codes=("global_personal_distress", "global_eligible_no_local_support"),
            ),
            tags=("video", "moderate", "chain4"),
        ),
    )
    c.append(
        _case(
            "video_isolated_local_distress",
            "Global distress + one local distress window",
            modality="video",
            sources=[_src("transcript", _clf(selected=["stress_or_overwhelm"]))],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="not_eligible", selected=[]),
                _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=1,
                eligible_window_indices=(1,),
                policy_status="ok",
                policy_indicator="moderate_concern",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("video", "moderate", "chain5"),
        ),
    )
    c.append(
        _case(
            "video_separated_local_distress",
            "Global distress + separated distress windows → not high",
            modality="video",
            sources=[_src("transcript", _clf(selected=["stress_or_overwhelm"]))],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                _win(1, start=5.0, end=10.0, eligibility="not_eligible", selected=[]),
                _win(2, start=10.0, end=15.0, eligibility="eligible", selected=["academic_pressure"]),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=2,
                eligible_window_indices=(0, 2),
                policy_status="ok",
                policy_indicator="moderate_concern",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("video", "moderate", "chain7"),
        ),
    )
    c.append(
        _case(
            "video_persistent_local_distress",
            "Global distress + two consecutive distress windows → high",
            modality="video",
            sources=[_src("transcript", _clf(selected=["stress_or_overwhelm"]))],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["academic_pressure"]),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=2,
                eligible_window_indices=(0, 1),
                policy_status="ok",
                policy_indicator="high_concern",
                required_reason_codes=("global_personal_distress", "persistent_local_support"),
            ),
            tags=("video", "high", "chain6"),
        ),
    )
    c.append(
        _case(
            "video_recovery_windows",
            "Global recovery + consecutive recovery windows → low",
            modality="video",
            sources=[_src("transcript", _clf(selected=["positive_wellbeing_or_recovery"]))],
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
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=2,
                eligible_window_indices=(0, 1),
                policy_status="ok",
                policy_indicator="low_concern",
                required_reason_codes=("global_personal_recovery",),
            ),
            tags=("video", "low", "chain1"),
        ),
    )
    c.append(
        _case(
            "video_mixed_window",
            "Global distress + mixed local window → moderate, not averaged",
            modality="video",
            sources=[_src("transcript", _clf(selected=["stress_or_overwhelm"]))],
            windows=[
                _win(
                    0,
                    start=0.0,
                    end=5.0,
                    eligibility="eligible",
                    selected=["stress_or_overwhelm", "positive_wellbeing_or_recovery"],
                ),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=1,
                eligible_window_indices=(0,),
                policy_status="ok",
                policy_indicator="moderate_concern",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("video", "moderate", "chain9"),
        ),
    )
    c.append(
        _case(
            "video_short_final_persistent",
            "Short final window completes consecutive distress pair → high",
            modality="video",
            sources=[_src("transcript", _clf(selected=["stress_or_overwhelm"]))],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                _win(1, start=5.0, end=7.5, eligibility="eligible", selected=["academic_pressure"]),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=2,
                eligible_window_indices=(0, 1),
                policy_status="ok",
                policy_indicator="high_concern",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("video", "high", "chain6"),
        ),
    )
    c.append(
        _case(
            "video_sparse_indices_nonconsecutive",
            "Sparse indices [1,4] → moderate",
            modality="video",
            sources=[_src("transcript", _clf(selected=["exhaustion_or_burnout_like_language"]))],
            windows=[
                _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                _win(4, start=20.0, end=25.0, eligibility="eligible", selected=["loneliness_or_isolation"]),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=2,
                eligible_window_indices=(1, 4),
                policy_status="ok",
                policy_indicator="moderate_concern",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("video", "moderate", "chain7"),
        ),
    )
    c.append(
        _case(
            "video_isolated_hopelessness_not_high",
            "Isolated hopelessness-like window → moderate (signal name alone)",
            modality="video",
            sources=[_src("transcript", _clf(selected=["stress_or_overwhelm"]))],
            windows=[
                _win(
                    3,
                    start=15.0,
                    end=20.0,
                    eligibility="eligible",
                    selected=["hopelessness_like_language"],
                ),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=1,
                eligible_window_indices=(3,),
                policy_status="ok",
                policy_indicator="moderate_concern",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("video", "moderate", "chain5", "chain8"),
        ),
    )
    c.append(
        _case(
            "video_mixed_labels_consecutive_high",
            "Different distress labels across consecutive windows → high",
            modality="video",
            sources=[_src("transcript", _clf(selected=["stress_or_overwhelm"]))],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["hopelessness_like_language"]),
                _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["self_directed_negativity"]),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=2,
                eligible_window_indices=(0, 1),
                policy_status="ok",
                policy_indicator="high_concern",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("video", "high", "chain6"),
        ),
    )

    # ------------------------------------------------------------------
    # LOCAL-ONLY / CONFLICT
    # ------------------------------------------------------------------
    c.append(
        _case(
            "conflict_global_not_eligible_local_eligible",
            "Global not eligible + local eligible → insufficient",
            modality="conflict",
            sources=[
                _src(
                    "transcript",
                    _clf(eligibility="not_eligible", selected=[], attribution="not_self_experience"),
                ),
            ],
            windows=[_win(0, start=0.0, end=5.0, eligibility="eligible")],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=1,
                eligible_window_indices=(0,),
                conflict_diagnostics=("local_eligible_without_global_eligibility",),
                policy_status="insufficient_evidence",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("local_eligible_without_global_eligibility",),
            ),
            tags=("conflict", "insufficient", "chain3"),
        ),
    )
    c.append(
        _case(
            "conflict_global_uncertain_local_eligible",
            "Global uncertain + local eligible → insufficient",
            modality="conflict",
            sources=[
                _src("transcript", _clf(eligibility="uncertain", selected=[], attribution="unclear")),
            ],
            windows=[_win(2, start=10.0, end=15.0, eligibility="eligible")],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=1,
                eligible_window_indices=(2,),
                # Evidence diagnostics currently encode not_eligible local disagreement;
                # uncertain+local is carried by policy reason codes.
                conflict_diagnostics=(),
                policy_status="insufficient_evidence",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("global_uncertain", "local_eligible_without_global_eligibility"),
            ),
            tags=("conflict", "insufficient", "chain3"),
        ),
    )
    c.append(
        _case(
            "conflict_global_recovery_persistent_local",
            "Global recovery + persistent local distress → conflict",
            modality="conflict",
            sources=[_src("transcript", _clf(selected=["positive_wellbeing_or_recovery"]))],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["academic_pressure"]),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=2,
                eligible_window_indices=(0, 1),
                conflict_diagnostics=("global_recovery_local_distress",),
                policy_status="conflict",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("mixed_distress_recovery",),
            ),
            tags=("conflict", "chain9"),
        ),
    )
    c.append(
        _case(
            "conflict_global_distress_recurrent_recovery",
            "Global distress + recurrent local recovery → conflict",
            modality="conflict",
            sources=[_src("transcript", _clf(selected=["stress_or_overwhelm"]))],
            windows=[
                _win(
                    0,
                    start=0.0,
                    end=5.0,
                    eligibility="eligible",
                    selected=["positive_wellbeing_or_recovery"],
                ),
                _win(1, start=5.0, end=10.0, eligibility="not_eligible", selected=[]),
                _win(
                    2,
                    start=10.0,
                    end=15.0,
                    eligibility="eligible",
                    selected=["positive_wellbeing_or_recovery"],
                ),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=2,
                eligible_window_indices=(0, 2),
                conflict_diagnostics=("global_distress_local_recovery",),
                policy_status="conflict",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("mixed_distress_recovery",),
            ),
            tags=("conflict", "chain9"),
        ),
    )
    c.append(
        _case(
            "conflict_global_mixed_distress_recovery",
            "Global distress+recovery unresolved → conflict",
            modality="conflict",
            sources=[
                _src(
                    "transcript",
                    _clf(selected=["stress_or_overwhelm", "positive_wellbeing_or_recovery"]),
                ),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                conflict_diagnostics=("global_eligible_no_local_support",),
                policy_status="conflict",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("mixed_distress_recovery",),
            ),
            tags=("conflict", "chain9"),
        ),
    )
    c.append(
        _case(
            "conflict_local_only_no_global_source",
            "Local eligible without any global source",
            modality="conflict",
            sources=[],
            windows=[_win(0, start=0.0, end=5.0, eligibility="eligible")],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=1,
                eligible_window_indices=(0,),
                # Evidence-layer diagnostics require a present global source;
                # local-only absence is expressed in policy reason codes.
                conflict_diagnostics=(),
                policy_status="insufficient_evidence",
                policy_indicator="insufficient_evidence",
                required_reason_codes=(
                    "global_unavailable",
                    "local_eligible_without_global_eligibility",
                ),
            ),
            tags=("conflict", "insufficient", "chain3"),
        ),
    )

    # ------------------------------------------------------------------
    # FAILURE
    # ------------------------------------------------------------------
    c.append(
        _case(
            "failure_classifier_error_global",
            "Global classifier error",
            modality="failure",
            sources=[
                _src(
                    "transcript",
                    _clf(eligibility="not_eligible", selected=[], status="error"),
                ),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                policy_status="unavailable",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("global_unavailable",),
            ),
            tags=("failure", "insufficient"),
        ),
    )
    c.append(
        _case(
            "failure_classifier_unavailable_with_windows",
            "Classifier unavailable ignores local persistence",
            modality="failure",
            sources=[
                _src(
                    "transcript",
                    _clf(
                        eligibility="eligible",
                        selected=["stress_or_overwhelm"],
                        status="classifier_unavailable",
                    ),
                ),
            ],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="eligible"),
                _win(1, start=5.0, end=10.0, eligibility="eligible"),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=2,
                eligible_window_indices=(0, 1),
                policy_status="unavailable",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("global_unavailable",),
            ),
            tags=("failure", "insufficient", "chain2"),
        ),
    )
    c.append(
        _case(
            "failure_source_classification_none",
            "Source present but classification None",
            modality="failure",
            sources=[_src("transcript", None)],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                policy_status="insufficient_evidence",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("global_not_eligible",),
            ),
            tags=("failure", "insufficient"),
        ),
    )

    # ------------------------------------------------------------------
    # STRUCTURAL REGRESSIONS (A/E/I/J + quoted/topic/figurative)
    # Labels/states only — not exact historical probabilities.
    # ------------------------------------------------------------------
    c.append(
        _case(
            "regression_A_personal_self_distress",
            "Structural regression A: personal/self distress",
            modality="regression",
            sources=[
                _src(
                    "primary_text",
                    _clf(
                        selected=["stress_or_overwhelm", "academic_pressure"],
                        relevance="personal_wellbeing",
                        target="self",
                        attribution="self_experience",
                    ),
                ),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                conflict_diagnostics=("global_eligible_no_local_support",),
                policy_status="ok",
                policy_indicator="moderate_concern",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("regression", "text", "moderate"),
            structural_only=True,
        ),
    )
    c.append(
        _case(
            "regression_E_personal_self_recovery",
            "Structural regression E: personal/self recovery",
            modality="regression",
            sources=[
                _src(
                    "primary_text",
                    _clf(
                        selected=["positive_wellbeing_or_recovery"],
                        relevance="personal_wellbeing",
                        target="self",
                        attribution="self_experience",
                    ),
                ),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                conflict_diagnostics=("global_eligible_no_local_support",),
                policy_status="ok",
                policy_indicator="low_concern",
                required_reason_codes=("global_personal_recovery",),
            ),
            tags=("regression", "text", "low"),
            structural_only=True,
        ),
    )
    c.append(
        _case(
            "regression_I_roommate_conflict_uncertain",
            "Structural regression I: roommate/conflicting attribution → uncertain",
            modality="regression",
            sources=[
                _src(
                    "primary_text",
                    _clf(
                        eligibility="uncertain",
                        selected=["stress_or_overwhelm"],
                        relevance="personal_wellbeing",
                        target="self",
                        attribution="unclear",
                    ),
                ),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                policy_status="insufficient_evidence",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("global_uncertain",),
            ),
            tags=("regression", "text", "insufficient"),
            structural_only=True,
        ),
    )
    c.append(
        _case(
            "regression_J_personal_recovery",
            "Structural regression J: personal recovery",
            modality="regression",
            sources=[
                _src(
                    "primary_text",
                    _clf(selected=["positive_wellbeing_or_recovery"]),
                ),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                conflict_diagnostics=("global_eligible_no_local_support",),
                policy_status="ok",
                policy_indicator="low_concern",
                required_reason_codes=("global_personal_recovery",),
            ),
            tags=("regression", "text", "low"),
            structural_only=True,
        ),
    )
    c.append(
        _case(
            "regression_quoted_other_blocked",
            "Structural regression: quoted-other blocked → not eligible",
            modality="regression",
            sources=[
                _src(
                    "primary_text",
                    _clf(
                        eligibility="not_eligible",
                        selected=["stress_or_overwhelm"],
                        relevance="personal_wellbeing",
                        target="self",
                        attribution="not_self_experience",
                    ),
                ),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                policy_status="insufficient_evidence",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("global_not_eligible",),
            ),
            tags=("regression", "text", "insufficient"),
            structural_only=True,
        ),
    )
    c.append(
        _case(
            "regression_topic_only",
            "Structural regression: wellbeing topic only → not eligible",
            modality="regression",
            sources=[
                _src(
                    "primary_text",
                    _clf(
                        eligibility="not_eligible",
                        selected=[],
                        relevance="wellbeing_topic_only",
                        target="general_or_unknown",
                        attribution="not_self_experience",
                    ),
                ),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                policy_status="insufficient_evidence",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("global_not_eligible",),
            ),
            tags=("regression", "text", "insufficient"),
            structural_only=True,
        ),
    )
    c.append(
        _case(
            "regression_figurative_non_personal",
            "Structural regression: figurative/non-personal → not eligible",
            modality="regression",
            sources=[
                _src(
                    "primary_text",
                    _clf(
                        eligibility="not_eligible",
                        selected=[],
                        relevance="not_wellbeing_related",
                        target="institution_or_event",
                        attribution="not_self_experience",
                    ),
                ),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                policy_status="insufficient_evidence",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("global_not_eligible",),
            ),
            tags=("regression", "text", "insufficient"),
            structural_only=True,
        ),
    )

    # ------------------------------------------------------------------
    # EXTRA VIDEO / TEXT COVERAGE
    # ------------------------------------------------------------------
    c.append(
        _case(
            "video_three_consecutive_high",
            "Three consecutive distress windows → high",
            modality="video",
            sources=[_src("transcript", _clf(selected=["stress_or_overwhelm"]))],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["anxiety_or_fear_language"]),
                _win(2, start=10.0, end=15.0, eligibility="eligible", selected=["loneliness_or_isolation"]),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=3,
                eligible_window_indices=(0, 1, 2),
                policy_status="ok",
                policy_indicator="high_concern",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("video", "high", "chain6"),
        ),
    )
    c.append(
        _case(
            "video_recovery_isolated_local_distress_low",
            "Global recovery + isolated local distress → low (phase4c2-v1)",
            modality="video",
            sources=[_src("transcript", _clf(selected=["positive_wellbeing_or_recovery"]))],
            windows=[
                _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=1,
                eligible_window_indices=(0,),
                conflict_diagnostics=("global_recovery_local_distress",),
                policy_status="ok",
                policy_indicator="low_concern",
                required_reason_codes=("global_personal_recovery", "local_distress_present"),
            ),
            tags=("video", "low", "chain1"),
        ),
    )
    c.append(
        _case(
            "text_global_hopelessness_only_moderate",
            "Global hopelessness-like alone does not auto-high",
            modality="text",
            sources=[_src("primary_text", _clf(selected=["hopelessness_like_language"]))],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                conflict_diagnostics=("global_eligible_no_local_support",),
                policy_status="ok",
                policy_indicator="moderate_concern",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("text", "moderate", "chain8"),
        ),
    )
    c.append(
        _case(
            "audio_transcript_not_eligible",
            "Transcript not eligible",
            modality="audio",
            sources=[
                _src(
                    "transcript",
                    _clf(
                        eligibility="not_eligible",
                        selected=[],
                        relevance="not_wellbeing_related",
                        target="other_person",
                        attribution="not_self_experience",
                    ),
                ),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                policy_status="insufficient_evidence",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("global_not_eligible",),
            ),
            tags=("audio", "insufficient", "chain2"),
        ),
    )
    c.append(
        _case(
            "video_indices_1_2_consecutive_high",
            "Window indices [1,2] consecutive → high",
            modality="video",
            sources=[_src("transcript", _clf(selected=["stress_or_overwhelm"]))],
            windows=[
                _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                _win(2, start=10.0, end=15.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=2,
                eligible_window_indices=(1, 2),
                policy_status="ok",
                policy_indicator="high_concern",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("video", "high", "chain6"),
        ),
    )
    c.append(
        _case(
            "video_indices_1_3_nonconsecutive_moderate",
            "Window indices [1,3] nonconsecutive → moderate",
            modality="video",
            sources=[_src("transcript", _clf(selected=["stress_or_overwhelm"]))],
            windows=[
                _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                _win(3, start=15.0, end=20.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=2,
                eligible_window_indices=(1, 3),
                policy_status="ok",
                policy_indicator="moderate_concern",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("video", "moderate", "chain7"),
        ),
    )
    c.append(
        _case(
            "caption_uncertain",
            "Caption uncertain eligibility",
            modality="caption",
            sources=[
                _src("caption", _clf(eligibility="uncertain", selected=[], attribution="unclear")),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                policy_status="insufficient_evidence",
                policy_indicator="insufficient_evidence",
                required_reason_codes=("global_uncertain",),
            ),
            tags=("caption", "insufficient"),
        ),
    )
    c.append(
        _case(
            "video_global_distress_single_recovery_window_moderate",
            "Global distress + single recovery window (not recurrent) → moderate",
            modality="video",
            sources=[_src("transcript", _clf(selected=["stress_or_overwhelm"]))],
            windows=[
                _win(
                    0,
                    start=0.0,
                    end=5.0,
                    eligibility="eligible",
                    selected=["positive_wellbeing_or_recovery"],
                ),
            ],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=1,
                eligible_window_indices=(0,),
                conflict_diagnostics=("global_distress_local_recovery",),
                policy_status="ok",
                policy_indicator="moderate_concern",
                required_reason_codes=("global_personal_distress", "local_recovery_present"),
            ),
            tags=("video", "moderate"),
        ),
    )
    c.append(
        _case(
            "text_eligible_exhaustion_only",
            "Primary text exhaustion-only distress → moderate",
            modality="text",
            sources=[_src("primary_text", _clf(selected=["exhaustion_or_burnout_like_language"]))],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                conflict_diagnostics=("global_eligible_no_local_support",),
                policy_status="ok",
                policy_indicator="moderate_concern",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("text", "moderate", "chain4"),
        ),
    )
    c.append(
        _case(
            "audio_transcript_eligible_no_signals",
            "Transcript eligible with empty selected signals → low",
            modality="audio",
            sources=[_src("transcript", _clf(selected=[]))],
            expected=ReplayExpectation(
                evidence_status="ok",
                eligible_window_count=0,
                conflict_diagnostics=("global_eligible_no_local_support",),
                policy_status="ok",
                policy_indicator="low_concern",
                required_reason_codes=("global_eligible_no_local_support",),
            ),
            tags=("audio", "low", "chain1"),
        ),
    )

    seen: set[str] = set()
    unique: list[ShadowReplayCase] = []
    for item in c:
        if item.case_id in seen:
            raise ValueError(f"Duplicate case_id: {item.case_id}")
        seen.add(item.case_id)
        unique.append(item)
    return unique


ALL_SHADOW_REPLAY_CASES: list[ShadowReplayCase] = _build_cases()


def get_replay_cases(
    *,
    modality: Optional[str] = None,
    tag: Optional[str] = None,
) -> list[ShadowReplayCase]:
    items = ALL_SHADOW_REPLAY_CASES
    if modality is not None:
        items = [c for c in items if c.modality == modality]
    if tag is not None:
        items = [c for c in items if tag in c.tags]
    return list(items)


def replay_counts_by_expected_indicator() -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in ALL_SHADOW_REPLAY_CASES:
        counts[c.expected.policy_indicator] = counts.get(c.expected.policy_indicator, 0) + 1
    return counts


def replay_counts_by_expected_status() -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in ALL_SHADOW_REPLAY_CASES:
        counts[c.expected.policy_status] = counts.get(c.expected.policy_status, 0) + 1
    return counts


def replay_counts_by_modality() -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in ALL_SHADOW_REPLAY_CASES:
        counts[c.modality] = counts.get(c.modality, 0) + 1
    return counts
