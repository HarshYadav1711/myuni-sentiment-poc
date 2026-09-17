"""Phase 4C.3 — structured wellbeing policy validation scenarios.

Synthetic engineering evidence states for validating phase4c2-v1.

These are NOT student examples, clinical cases, or live inference outputs.
Expected policy outcomes are HUMAN-WRITTEN and independent of the policy
implementation (fixtures must not call the policy builder when defining
expectations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from src.config import DEFAULT_WELLBEING_CLASSIFIER_MODEL
from src.wellbeing.evidence import build_wellbeing_evidence_context
from src.wellbeing.labels import DISTRESS_LIKE_SIGNAL_IDS
from src.wellbeing.schemas import (
    ExclusiveClassification,
    SelfAttributionResult,
    SignalScore,
    WellbeingClassificationResult,
    WellbeingEvidenceContext,
    WellbeingShadowSourceResult,
    WellbeingShadowWindowResult,
)

POLICY_VERSION_UNDER_TEST = "phase4c2-v1"


@dataclass(frozen=True)
class PolicyExpectation:
    """Human-authored expected policy outcome (independent of implementation)."""

    status: str
    indicator: str
    local_support_level: str
    distress_pattern: str
    recovery_pattern: str
    required_reason_codes: tuple[str, ...] = ()
    forbidden_reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyScenario:
    """One structured evidence state + human expectation."""

    scenario_id: str
    description: str
    evidence: Optional[WellbeingEvidenceContext]
    expected: PolicyExpectation
    tags: tuple[str, ...] = ()
    branch: str = ""  # insufficient | low | moderate | high | conflict | adversarial | replay


def _signals(selected: Sequence[str]) -> list[SignalScore]:
    ids = sorted(set(selected) | {"stress_or_overwhelm"})
    out: list[SignalScore] = []
    for sid in ids:
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
    if selected is None:
        selected = ["stress_or_overwhelm"] if eligibility == "eligible" else []
    return WellbeingClassificationResult(
        model_id=DEFAULT_WELLBEING_CLASSIFIER_MODEL,
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
    if selected is None:
        selected = ["stress_or_overwhelm"] if eligibility == "eligible" else []
    return WellbeingShadowWindowResult(
        window_index=index,
        start=start,
        end=end,
        classification=_clf(eligibility=eligibility, selected=selected),
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
    source_role: str = "transcript",
) -> WellbeingEvidenceContext:
    sources: list[WellbeingShadowSourceResult] = []
    if global_present:
        if global_eligibility is None:
            sources.append(_src(source_role, None))
        else:
            if global_selected is None:
                global_selected = (
                    ["stress_or_overwhelm"]
                    if global_eligibility == "eligible"
                    else []
                )
            sources.append(
                _src(
                    source_role,
                    _clf(
                        eligibility=global_eligibility,
                        selected=list(global_selected),
                        status=global_status,
                    ),
                ),
            )
    return build_wellbeing_evidence_context(
        source_results=sources,
        window_results=windows or [],
        shadow_status=evidence_status,
    )


def _scenario(
    scenario_id: str,
    description: str,
    *,
    evidence: Optional[WellbeingEvidenceContext],
    expected: PolicyExpectation,
    tags: tuple[str, ...] = (),
    branch: str = "",
) -> PolicyScenario:
    return PolicyScenario(
        scenario_id=scenario_id,
        description=description,
        evidence=evidence,
        expected=expected,
        tags=tags,
        branch=branch,
    )


def _build_all_scenarios() -> list[PolicyScenario]:
    """Human-authored structured scenarios (expectations not derived from policy)."""
    s: list[PolicyScenario] = []

    # ------------------------------------------------------------------
    # INSUFFICIENT / UNAVAILABLE / CONFLICT gates
    # ------------------------------------------------------------------
    s.append(
        _scenario(
            "insuf_no_evidence",
            "No evidence context object",
            evidence=None,
            expected=PolicyExpectation(
                status="unavailable",
                indicator="insufficient_evidence",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="none",
                required_reason_codes=("global_unavailable", "no_personal_wellbeing_evidence"),
            ),
            tags=("insufficient", "gate"),
            branch="insufficient",
        ),
    )
    s.append(
        _scenario(
            "insuf_global_absent",
            "Global source absent, no windows",
            evidence=_evidence(global_present=False, windows=[]),
            expected=PolicyExpectation(
                status="insufficient_evidence",
                indicator="insufficient_evidence",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="none",
                required_reason_codes=("global_unavailable",),
            ),
            tags=("insufficient", "gate"),
            branch="insufficient",
        ),
    )
    s.append(
        _scenario(
            "insuf_classifier_unavailable",
            "Global classifier unavailable",
            evidence=_evidence(global_status="classifier_unavailable", global_selected=[]),
            expected=PolicyExpectation(
                status="unavailable",
                indicator="insufficient_evidence",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="none",
                required_reason_codes=("global_unavailable",),
            ),
            tags=("insufficient", "gate"),
            branch="insufficient",
        ),
    )
    s.append(
        _scenario(
            "insuf_classifier_error",
            "Global classifier error",
            evidence=_evidence(global_status="error", global_selected=[]),
            expected=PolicyExpectation(
                status="unavailable",
                indicator="insufficient_evidence",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="none",
                required_reason_codes=("global_unavailable",),
            ),
            tags=("insufficient", "gate"),
            branch="insufficient",
        ),
    )
    s.append(
        _scenario(
            "insuf_global_uncertain",
            "Global uncertain eligibility",
            evidence=_evidence(global_eligibility="uncertain", global_selected=[]),
            expected=PolicyExpectation(
                status="insufficient_evidence",
                indicator="insufficient_evidence",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="none",
                required_reason_codes=("global_uncertain",),
            ),
            tags=("insufficient", "gate"),
            branch="insufficient",
        ),
    )
    s.append(
        _scenario(
            "insuf_global_not_eligible",
            "Global not eligible",
            evidence=_evidence(global_eligibility="not_eligible", global_selected=[]),
            expected=PolicyExpectation(
                status="insufficient_evidence",
                indicator="insufficient_evidence",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="none",
                required_reason_codes=("global_not_eligible",),
            ),
            tags=("insufficient", "gate"),
            branch="insufficient",
        ),
    )
    s.append(
        _scenario(
            "insuf_local_only",
            "Local eligible without global eligibility",
            evidence=_evidence(
                global_eligibility="not_eligible",
                global_selected=[],
                windows=[_win(0, start=0.0, end=5.0, eligibility="eligible")],
            ),
            expected=PolicyExpectation(
                status="insufficient_evidence",
                indicator="insufficient_evidence",
                local_support_level="isolated",
                distress_pattern="isolated",
                recovery_pattern="none",
                required_reason_codes=("local_eligible_without_global_eligibility",),
            ),
            tags=("insufficient", "invariant9"),
            branch="insufficient",
        ),
    )
    s.append(
        _scenario(
            "insuf_local_only_persistent",
            "Persistent local distress but global not eligible",
            evidence=_evidence(
                global_eligibility="not_eligible",
                global_selected=[],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible"),
                    _win(1, start=5.0, end=10.0, eligibility="eligible"),
                ],
            ),
            expected=PolicyExpectation(
                status="insufficient_evidence",
                indicator="insufficient_evidence",
                local_support_level="persistent",
                distress_pattern="persistent",
                recovery_pattern="none",
                required_reason_codes=("local_eligible_without_global_eligibility",),
                forbidden_reason_codes=(),
            ),
            tags=("insufficient", "adversarial", "invariant9"),
            branch="insufficient",
        ),
    )
    s.append(
        _scenario(
            "insuf_global_absent_local_eligible",
            "Missing global with local eligible windows",
            evidence=_evidence(
                global_present=False,
                windows=[_win(0, start=0.0, end=5.0, eligibility="eligible")],
            ),
            expected=PolicyExpectation(
                status="insufficient_evidence",
                indicator="insufficient_evidence",
                local_support_level="isolated",
                distress_pattern="isolated",
                recovery_pattern="none",
                required_reason_codes=(
                    "global_unavailable",
                    "local_eligible_without_global_eligibility",
                ),
            ),
            tags=("insufficient", "invariant9"),
            branch="insufficient",
        ),
    )
    s.append(
        _scenario(
            "insuf_uncertain_with_local",
            "Global uncertain with local eligible",
            evidence=_evidence(
                global_eligibility="uncertain",
                global_selected=[],
                windows=[_win(2, start=10.0, end=15.0, eligibility="eligible")],
            ),
            expected=PolicyExpectation(
                status="insufficient_evidence",
                indicator="insufficient_evidence",
                local_support_level="isolated",
                distress_pattern="isolated",
                recovery_pattern="none",
                required_reason_codes=(
                    "global_uncertain",
                    "local_eligible_without_global_eligibility",
                ),
            ),
            tags=("insufficient",),
            branch="insufficient",
        ),
    )
    s.append(
        _scenario(
            "conflict_global_mixed",
            "Global distress + recovery unresolved",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm", "positive_wellbeing_or_recovery"],
            ),
            expected=PolicyExpectation(
                status="conflict",
                indicator="insufficient_evidence",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="mixed_with_distress",
                required_reason_codes=("mixed_distress_recovery", "global_local_conflict"),
            ),
            tags=("conflict", "invariant8"),
            branch="conflict",
        ),
    )
    s.append(
        _scenario(
            "conflict_global_recovery_persistent_local",
            "Global recovery + persistent local distress",
            evidence=_evidence(
                global_selected=["positive_wellbeing_or_recovery"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                    _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["academic_pressure"]),
                ],
            ),
            expected=PolicyExpectation(
                status="conflict",
                indicator="insufficient_evidence",
                local_support_level="persistent",
                distress_pattern="mixed_with_recovery",
                recovery_pattern="mixed_with_distress",
                required_reason_codes=("mixed_distress_recovery",),
            ),
            tags=("conflict", "recovery", "invariant7", "invariant8"),
            branch="conflict",
        ),
    )
    s.append(
        _scenario(
            "conflict_global_distress_recurrent_recovery",
            "Global distress + recurrent local recovery",
            evidence=_evidence(
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
            expected=PolicyExpectation(
                status="conflict",
                indicator="insufficient_evidence",
                local_support_level="recurrent",
                distress_pattern="none",
                recovery_pattern="mixed_with_distress",
                required_reason_codes=("mixed_distress_recovery",),
            ),
            tags=("conflict", "invariant8"),
            branch="conflict",
        ),
    )
    s.append(
        _scenario(
            "conflict_global_mixed_with_local_distress",
            "Global distress+recovery with local distress",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm", "positive_wellbeing_or_recovery"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                ],
            ),
            expected=PolicyExpectation(
                status="conflict",
                indicator="insufficient_evidence",
                local_support_level="isolated",
                distress_pattern="mixed_with_recovery",
                recovery_pattern="mixed_with_distress",
                required_reason_codes=("mixed_distress_recovery",),
            ),
            tags=("conflict",),
            branch="conflict",
        ),
    )

    # ------------------------------------------------------------------
    # LOW
    # ------------------------------------------------------------------
    s.append(
        _scenario(
            "low_global_recovery_only",
            "Global eligible recovery only, zero local",
            evidence=_evidence(global_selected=["positive_wellbeing_or_recovery"]),
            expected=PolicyExpectation(
                status="ok",
                indicator="low_concern",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="global_only",
                required_reason_codes=("global_personal_recovery",),
            ),
            tags=("low",),
            branch="low",
        ),
    )
    s.append(
        _scenario(
            "low_global_no_distress",
            "Global eligible with empty selected signals",
            evidence=_evidence(global_selected=[]),
            expected=PolicyExpectation(
                status="ok",
                indicator="low_concern",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="none",
                required_reason_codes=("global_eligible_no_local_support",),
            ),
            tags=("low",),
            branch="low",
        ),
    )
    s.append(
        _scenario(
            "low_global_recovery_zero_windows_explicit",
            "Global recovery with evaluated but not-eligible windows",
            evidence=_evidence(
                global_selected=["positive_wellbeing_or_recovery"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="not_eligible"),
                    _win(1, start=5.0, end=10.0, eligibility="not_eligible"),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="low_concern",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="global_only",
                required_reason_codes=("global_personal_recovery",),
            ),
            tags=("low",),
            branch="low",
        ),
    )
    s.append(
        _scenario(
            "low_global_recovery_isolated_local_recovery",
            "Global recovery + one recovery window",
            evidence=_evidence(
                global_selected=["positive_wellbeing_or_recovery"],
                windows=[
                    _win(
                        0,
                        start=0.0,
                        end=5.0,
                        eligibility="eligible",
                        selected=["positive_wellbeing_or_recovery"],
                    ),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="low_concern",
                local_support_level="isolated",
                distress_pattern="none",
                recovery_pattern="global_only",
                required_reason_codes=("global_personal_recovery", "local_recovery_present"),
            ),
            tags=("low", "recovery"),
            branch="low",
        ),
    )
    s.append(
        _scenario(
            "low_repeated_local_recovery",
            "Global recovery + consecutive recovery windows",
            evidence=_evidence(
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
            expected=PolicyExpectation(
                status="ok",
                indicator="low_concern",
                local_support_level="persistent",
                distress_pattern="none",
                recovery_pattern="recurrent",
                required_reason_codes=("global_personal_recovery",),
            ),
            tags=("low", "recovery"),
            branch="low",
        ),
    )
    s.append(
        _scenario(
            "low_global_recovery_isolated_local_distress",
            "Global recovery + isolated local distress (phase4c2-v1: still low)",
            evidence=_evidence(
                global_selected=["positive_wellbeing_or_recovery"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="low_concern",
                local_support_level="isolated",
                distress_pattern="mixed_with_recovery",
                recovery_pattern="mixed_with_distress",
                required_reason_codes=("global_personal_recovery", "local_distress_present"),
            ),
            tags=("low", "recovery", "invariant7"),
            branch="low",
        ),
    )
    s.append(
        _scenario(
            "low_primary_text_recovery",
            "Global primary_text recovery role",
            evidence=_evidence(
                source_role="primary_text",
                global_selected=["positive_wellbeing_or_recovery"],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="low_concern",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="global_only",
                required_reason_codes=("global_personal_recovery",),
            ),
            tags=("low",),
            branch="low",
        ),
    )
    s.append(
        _scenario(
            "low_separated_recovery_windows",
            "Global recovery + nonconsecutive recovery windows",
            evidence=_evidence(
                global_selected=["positive_wellbeing_or_recovery"],
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
            expected=PolicyExpectation(
                status="ok",
                indicator="low_concern",
                local_support_level="recurrent",
                distress_pattern="none",
                recovery_pattern="recurrent",
                required_reason_codes=("global_personal_recovery",),
            ),
            tags=("low", "recovery"),
            branch="low",
        ),
    )

    # ------------------------------------------------------------------
    # MODERATE (+ monotonicity ladder)
    # ------------------------------------------------------------------
    s.append(
        _scenario(
            "mod_global_distress_zero_local",
            "Global distress + zero local support (invariant 10)",
            evidence=_evidence(global_selected=["stress_or_overwhelm", "academic_pressure"]),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="none",
                required_reason_codes=(
                    "global_personal_distress",
                    "global_eligible_no_local_support",
                ),
            ),
            tags=("moderate", "monotonicity", "invariant10"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "mod_global_distress_one_local",
            "Global distress + one local distress window",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="not_eligible"),
                    _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="isolated",
                distress_pattern="isolated",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress", "local_distress_present"),
            ),
            tags=("moderate", "monotonicity"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "mod_global_distress_separated",
            "Global distress + two nonconsecutive distress windows",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                    _win(1, start=5.0, end=10.0, eligibility="not_eligible"),
                    _win(2, start=10.0, end=15.0, eligibility="eligible", selected=["academic_pressure"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="recurrent",
                distress_pattern="recurrent",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate", "monotonicity", "window_structure"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "mod_global_distress_sparse_indices",
            "Global distress + sparse indices [1,4]",
            evidence=_evidence(
                global_selected=["exhaustion_or_burnout_like_language"],
                windows=[
                    _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                    _win(4, start=20.0, end=25.0, eligibility="eligible", selected=["loneliness_or_isolation"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="recurrent",
                distress_pattern="recurrent",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate", "window_structure"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "mod_isolated_hopelessness",
            "Isolated hopelessness must not force high (invariant 3)",
            evidence=_evidence(
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
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="isolated",
                distress_pattern="isolated",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
                forbidden_reason_codes=(),
            ),
            tags=("moderate", "invariant3", "adversarial", "safety"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "mod_isolated_self_negativity",
            "Isolated self-directed negativity must not force high (invariant 4)",
            evidence=_evidence(
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
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="isolated",
                distress_pattern="isolated",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate", "invariant4", "adversarial", "safety"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "mod_isolated_anxiety",
            "Isolated anxiety must not force high (invariant 5)",
            evidence=_evidence(
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
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="isolated",
                distress_pattern="isolated",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate", "invariant5", "adversarial", "safety"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "mod_all_eight_distress_one_window",
            "All eight distress IDs in one isolated window still moderate",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(
                        0,
                        start=0.0,
                        end=5.0,
                        eligibility="eligible",
                        selected=sorted(DISTRESS_LIKE_SIGNAL_IDS),
                    ),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="isolated",
                distress_pattern="isolated",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate", "adversarial", "invariant3"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "mod_mixed_window_preserved",
            "Global distress + mixed local window → moderate, not averaged",
            evidence=_evidence(
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
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="isolated",
                distress_pattern="mixed_with_recovery",
                recovery_pattern="mixed_with_distress",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate", "invariant7", "invariant8"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "mod_duplicate_signals_same_window",
            "Duplicate-like multi-signal presence in one window",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm", "academic_pressure"],
                windows=[
                    _win(
                        2,
                        start=10.0,
                        end=15.0,
                        eligibility="eligible",
                        selected=["stress_or_overwhelm", "academic_pressure", "interpersonal_distress"],
                    ),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="isolated",
                distress_pattern="isolated",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate",),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "mod_signal_name_change_no_escalate",
            "Same structure as one-local but different signal name → still moderate",
            evidence=_evidence(
                global_selected=["loneliness_or_isolation"],
                windows=[
                    _win(
                        1,
                        start=5.0,
                        end=10.0,
                        eligibility="eligible",
                        selected=["loneliness_or_isolation"],
                    ),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="isolated",
                distress_pattern="isolated",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate", "monotonicity", "safety"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "mod_twenty_nonconsecutive_distress",
            "Many nonconsecutive distress windows remain moderate (not high)",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(
                        i,
                        start=float(i * 5),
                        end=float(i * 5 + 5),
                        eligibility="eligible" if i % 2 == 0 else "not_eligible",
                        selected=["stress_or_overwhelm"] if i % 2 == 0 else [],
                    )
                    for i in range(20)
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="recurrent",
                distress_pattern="recurrent",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate", "adversarial", "window_structure"),
            branch="moderate",
        ),
    )

    # ------------------------------------------------------------------
    # HIGH
    # ------------------------------------------------------------------
    s.append(
        _scenario(
            "high_two_consecutive",
            "Global distress + two consecutive distress windows",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                    _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["academic_pressure"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="high_concern",
                local_support_level="persistent",
                distress_pattern="persistent",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress", "persistent_local_support"),
            ),
            tags=("high", "monotonicity", "invariant2", "invariant6"),
            branch="high",
        ),
    )
    s.append(
        _scenario(
            "high_three_consecutive",
            "Global distress + three consecutive distress windows",
            evidence=_evidence(
                global_selected=["exhaustion_or_burnout_like_language"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                    _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["anxiety_or_fear_language"]),
                    _win(2, start=10.0, end=15.0, eligibility="eligible", selected=["loneliness_or_isolation"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="high_concern",
                local_support_level="persistent",
                distress_pattern="persistent",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("high", "invariant6"),
            branch="high",
        ),
    )
    s.append(
        _scenario(
            "high_mixed_labels_consecutive",
            "Different distress labels across adjacent windows (invariant 6)",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["hopelessness_like_language"]),
                    _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["self_directed_negativity"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="high_concern",
                local_support_level="persistent",
                distress_pattern="persistent",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("high", "invariant6"),
            branch="high",
        ),
    )
    s.append(
        _scenario(
            "high_short_final_window",
            "Short final window forms second consecutive distress window",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                    _win(1, start=5.0, end=7.5, eligibility="eligible", selected=["academic_pressure"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="high_concern",
                local_support_level="persistent",
                distress_pattern="persistent",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("high", "window_structure"),
            branch="high",
        ),
    )
    s.append(
        _scenario(
            "high_indices_4_5",
            "Consecutive sparse-ish indices [4,5]",
            evidence=_evidence(
                global_selected=["interpersonal_distress"],
                windows=[
                    _win(4, start=20.0, end=25.0, eligibility="eligible", selected=["interpersonal_distress"]),
                    _win(5, start=25.0, end=30.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="high_concern",
                local_support_level="persistent",
                distress_pattern="persistent",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("high", "window_structure"),
            branch="high",
        ),
    )
    s.append(
        _scenario(
            "high_with_leading_not_eligible",
            "Consecutive distress after leading not-eligible windows",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="not_eligible"),
                    _win(1, start=5.0, end=10.0, eligibility="not_eligible"),
                    _win(2, start=10.0, end=15.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                    _win(3, start=15.0, end=20.0, eligibility="eligible", selected=["academic_pressure"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="high_concern",
                local_support_level="persistent",
                distress_pattern="persistent",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("high",),
            branch="high",
        ),
    )

    # ------------------------------------------------------------------
    # CONTROLLED VIDEO STRUCTURED REPLAY
    # ------------------------------------------------------------------
    s.append(
        _scenario(
            "replay_controlled_video_phase4b",
            "Structured replay of previously observed Phase 4B/4C evidence (not live)",
            evidence=_evidence(
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
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="isolated",
                distress_pattern="isolated",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress", "local_distress_present"),
            ),
            tags=("replay", "moderate", "safety"),
            branch="moderate",
        ),
    )

    # ------------------------------------------------------------------
    # ADDITIONAL BRANCH / STRUCTURE COVERAGE
    # ------------------------------------------------------------------
    s.append(
        _scenario(
            "mod_global_distress_not_eligible_windows_only",
            "Global distress with only not-eligible evaluated windows",
            evidence=_evidence(
                global_selected=["academic_pressure"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="not_eligible"),
                    _win(1, start=5.0, end=10.0, eligibility="uncertain", selected=[]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate", "invariant10"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "insuf_evidence_status_error",
            "Evidence context status=error",
            evidence=_evidence(evidence_status="error", global_selected=[]),
            expected=PolicyExpectation(
                status="unavailable",
                indicator="insufficient_evidence",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="none",
                required_reason_codes=("global_unavailable",),
            ),
            tags=("insufficient", "adversarial"),
            branch="insufficient",
        ),
    )
    s.append(
        _scenario(
            "insuf_classifier_error_populated_fields",
            "Classifier error with otherwise populated eligible-looking fields",
            evidence=_evidence(
                global_eligibility="eligible",
                global_status="error",
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible"),
                    _win(1, start=5.0, end=10.0, eligibility="eligible"),
                ],
            ),
            expected=PolicyExpectation(
                status="unavailable",
                indicator="insufficient_evidence",
                local_support_level="persistent",
                distress_pattern="persistent",
                recovery_pattern="none",
                required_reason_codes=("global_unavailable",),
            ),
            tags=("insufficient", "adversarial"),
            branch="insufficient",
        ),
    )
    s.append(
        _scenario(
            "mod_out_of_order_window_construction",
            "Windows constructed out of order still classified by index adjacency",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(2, start=10.0, end=15.0, eligibility="eligible", selected=["academic_pressure"]),
                    _win(0, start=0.0, end=5.0, eligibility="not_eligible"),
                    _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="high_concern",
                local_support_level="persistent",
                distress_pattern="persistent",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("high", "adversarial", "window_structure"),
            branch="high",
        ),
    )
    # Note: out-of-order windows [1,2] eligible consecutive → HIGH not moderate.
    # Fix expectation: windows 1 and 2 are consecutive → high. Good.

    s.append(
        _scenario(
            "mod_indices_1_3_nonconsecutive",
            "Indices [1,3] are nonconsecutive → moderate",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                    _win(3, start=15.0, end=20.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="recurrent",
                distress_pattern="recurrent",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate", "window_structure"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "high_indices_1_2_consecutive",
            "Indices [1,2] consecutive → high",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                    _win(2, start=10.0, end=15.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="high_concern",
                local_support_level="persistent",
                distress_pattern="persistent",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("high", "window_structure"),
            branch="high",
        ),
    )
    s.append(
        _scenario(
            "low_caption_role_no_distress",
            "Caption global eligible no distress",
            evidence=_evidence(source_role="caption", global_selected=[]),
            expected=PolicyExpectation(
                status="ok",
                indicator="low_concern",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="none",
                required_reason_codes=("global_eligible_no_local_support",),
            ),
            tags=("low",),
            branch="low",
        ),
    )
    s.append(
        _scenario(
            "mod_global_only_exhaustion",
            "Global exhaustion-only distress signal",
            evidence=_evidence(global_selected=["exhaustion_or_burnout_like_language"]),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate",),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "mod_global_interpersonal_only",
            "Global interpersonal distress only",
            evidence=_evidence(global_selected=["interpersonal_distress"]),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate",),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "conflict_global_recovery_recurrent_local_distress",
            "Global recovery + recurrent (nonconsecutive) local distress → conflict",
            evidence=_evidence(
                global_selected=["positive_wellbeing_or_recovery"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                    _win(1, start=5.0, end=10.0, eligibility="not_eligible"),
                    _win(2, start=10.0, end=15.0, eligibility="eligible", selected=["academic_pressure"]),
                ],
            ),
            expected=PolicyExpectation(
                status="conflict",
                indicator="insufficient_evidence",
                local_support_level="recurrent",
                distress_pattern="mixed_with_recovery",
                recovery_pattern="mixed_with_distress",
                required_reason_codes=("mixed_distress_recovery", "global_local_conflict"),
            ),
            tags=("conflict", "recovery", "invariant7"),
            branch="conflict",
        ),
    )
    s.append(
        _scenario(
            "insuf_empty_evidence_status",
            "Empty evidence status with no sources",
            evidence=build_wellbeing_evidence_context(
                source_results=[],
                window_results=[],
                shadow_status="insufficient_text",
            ),
            expected=PolicyExpectation(
                status="insufficient_evidence",
                indicator="insufficient_evidence",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="none",
                required_reason_codes=("global_unavailable",),
            ),
            tags=("insufficient",),
            branch="insufficient",
        ),
    )
    s.append(
        _scenario(
            "mod_global_distress_one_recovery_window_ok",
            "Global distress + single local recovery (not recurrent) → moderate",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(
                        0,
                        start=0.0,
                        end=5.0,
                        eligibility="eligible",
                        selected=["positive_wellbeing_or_recovery"],
                    ),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="isolated",
                distress_pattern="none",
                recovery_pattern="mixed_with_distress",
                required_reason_codes=("global_personal_distress", "local_recovery_present"),
            ),
            tags=("moderate", "recovery"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "high_four_consecutive",
            "Four consecutive distress windows",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(i, start=float(i * 5), end=float(i * 5 + 5), eligibility="eligible", selected=["stress_or_overwhelm"])
                    for i in range(4)
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="high_concern",
                local_support_level="persistent",
                distress_pattern="persistent",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("high",),
            branch="high",
        ),
    )
    s.append(
        _scenario(
            "mod_global_anxiety_zero_local",
            "Global anxiety only, zero local",
            evidence=_evidence(global_selected=["anxiety_or_fear_language"]),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate",),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "mod_global_hopelessness_zero_local",
            "Global hopelessness-like only does not auto-high without persistence",
            evidence=_evidence(global_selected=["hopelessness_like_language"]),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate", "safety", "invariant3"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "low_global_recovery_uncertain_local_ignored",
            "Global recovery; local uncertain windows do not count as support",
            evidence=_evidence(
                global_selected=["positive_wellbeing_or_recovery"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="uncertain", selected=[]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="low_concern",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="global_only",
                required_reason_codes=("global_personal_recovery",),
            ),
            tags=("low",),
            branch="low",
        ),
    )
    s.append(
        _scenario(
            "insuf_global_not_eligible_many_local",
            "Many local distress windows cannot override global not-eligible",
            evidence=_evidence(
                global_eligibility="not_eligible",
                global_selected=[],
                windows=[
                    _win(i, start=float(i * 5), end=float(i * 5 + 5), eligibility="eligible")
                    for i in range(5)
                ],
            ),
            expected=PolicyExpectation(
                status="insufficient_evidence",
                indicator="insufficient_evidence",
                local_support_level="persistent",
                distress_pattern="persistent",
                recovery_pattern="none",
                required_reason_codes=(
                    "global_not_eligible",
                    "local_eligible_without_global_eligibility",
                ),
            ),
            tags=("insufficient", "adversarial", "invariant1", "invariant9"),
            branch="insufficient",
        ),
    )
    s.append(
        _scenario(
            "mod_distress_broken_by_recovery_window",
            "Distress 0, recovery 1, distress 2 → nonconsecutive distress → moderate",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                    _win(
                        1,
                        start=5.0,
                        end=10.0,
                        eligibility="eligible",
                        selected=["positive_wellbeing_or_recovery"],
                    ),
                    _win(2, start=10.0, end=15.0, eligibility="eligible", selected=["academic_pressure"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="persistent",
                distress_pattern="mixed_with_recovery",
                recovery_pattern="mixed_with_distress",
                required_reason_codes=("global_personal_distress", "local_distress_present"),
            ),
            tags=("moderate", "window_structure", "recovery"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "mod_global_self_negativity_zero_local",
            "Global self-directed negativity alone → moderate (no auto-high)",
            evidence=_evidence(global_selected=["self_directed_negativity"]),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate", "safety"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "high_short_final_after_gap_not",
            "Nonconsecutive [0] and short [2] → moderate not high",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                    _win(1, start=5.0, end=10.0, eligibility="not_eligible"),
                    _win(2, start=10.0, end=12.0, eligibility="eligible", selected=["academic_pressure"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="recurrent",
                distress_pattern="recurrent",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate", "window_structure"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "insuf_global_uncertain_persistent_local",
            "Uncertain global + persistent local still insufficient",
            evidence=_evidence(
                global_eligibility="uncertain",
                global_selected=[],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible"),
                    _win(1, start=5.0, end=10.0, eligibility="eligible"),
                ],
            ),
            expected=PolicyExpectation(
                status="insufficient_evidence",
                indicator="insufficient_evidence",
                local_support_level="persistent",
                distress_pattern="persistent",
                recovery_pattern="none",
                required_reason_codes=(
                    "global_uncertain",
                    "local_eligible_without_global_eligibility",
                ),
            ),
            tags=("insufficient", "invariant1", "invariant9"),
            branch="insufficient",
        ),
    )
    s.append(
        _scenario(
            "low_global_eligible_no_signals_with_recovery_local",
            "Global eligible empty signals + local recovery → low",
            evidence=_evidence(
                global_selected=[],
                windows=[
                    _win(
                        0,
                        start=0.0,
                        end=5.0,
                        eligibility="eligible",
                        selected=["positive_wellbeing_or_recovery"],
                    ),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="low_concern",
                local_support_level="isolated",
                distress_pattern="none",
                recovery_pattern="local_only",
                required_reason_codes=("local_recovery_present",),
            ),
            tags=("low", "recovery"),
            branch="low",
        ),
    )
    s.append(
        _scenario(
            "mod_global_distress_isolated_recovery_and_distress",
            "Global distress + one distress + one separated recovery → conflict (2 recovery? no) moderate",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
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
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="recurrent",
                distress_pattern="mixed_with_recovery",
                recovery_pattern="mixed_with_distress",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate", "recovery"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "adversarial_missing_evidence_ids",
            "Evidence with empty evidence_ids still fails safe / deterministic",
            evidence=_evidence(global_selected=["stress_or_overwhelm"]).model_copy(
                update={"evidence_ids": []},
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("adversarial", "moderate"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "mod_global_loneliness_one_window",
            "Global loneliness + matching local window",
            evidence=_evidence(
                global_selected=["loneliness_or_isolation"],
                windows=[
                    _win(
                        0,
                        start=0.0,
                        end=5.0,
                        eligibility="eligible",
                        selected=["loneliness_or_isolation"],
                    ),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="isolated",
                distress_pattern="isolated",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate",),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "high_consecutive_with_duplicate_signal",
            "Consecutive windows sharing same distress signal → high",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                    _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="high_concern",
                local_support_level="persistent",
                distress_pattern="persistent",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("high",),
            branch="high",
        ),
    )
    s.append(
        _scenario(
            "insuf_classifier_unavailable_with_local",
            "Classifier unavailable ignores local persistent distress",
            evidence=_evidence(
                global_status="classifier_unavailable",
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible"),
                    _win(1, start=5.0, end=10.0, eligibility="eligible"),
                ],
            ),
            expected=PolicyExpectation(
                status="unavailable",
                indicator="insufficient_evidence",
                local_support_level="persistent",
                distress_pattern="persistent",
                recovery_pattern="none",
                required_reason_codes=("global_unavailable",),
            ),
            tags=("insufficient", "adversarial", "invariant1"),
            branch="insufficient",
        ),
    )
    s.append(
        _scenario(
            "mod_three_separated_distress",
            "Three nonconsecutive distress windows remain moderate",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                    _win(2, start=10.0, end=15.0, eligibility="eligible", selected=["anxiety_or_fear_language"]),
                    _win(5, start=25.0, end=30.0, eligibility="eligible", selected=["academic_pressure"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="recurrent",
                distress_pattern="recurrent",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate", "monotonicity", "window_structure"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "high_mixed_labels_stress_academic",
            "stress then academic_pressure consecutive → high (invariant 6)",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                    _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["academic_pressure"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="high_concern",
                local_support_level="persistent",
                distress_pattern="persistent",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("high", "invariant6", "monotonicity"),
            branch="high",
        ),
    )
    s.append(
        _scenario(
            "low_global_recovery_two_not_eligible",
            "Recovery global with only not-eligible locals",
            evidence=_evidence(
                global_selected=["positive_wellbeing_or_recovery"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="not_eligible"),
                    _win(1, start=5.0, end=10.0, eligibility="not_eligible"),
                    _win(2, start=10.0, end=15.0, eligibility="not_eligible"),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="low_concern",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="global_only",
                required_reason_codes=("global_personal_recovery",),
            ),
            tags=("low", "recovery"),
            branch="low",
        ),
    )
    s.append(
        _scenario(
            "conflict_global_distress_recovery_zero_local",
            "Global mixed distress+recovery, zero local",
            evidence=_evidence(
                global_selected=[
                    "stress_or_overwhelm",
                    "anxiety_or_fear_language",
                    "positive_wellbeing_or_recovery",
                ],
            ),
            expected=PolicyExpectation(
                status="conflict",
                indicator="insufficient_evidence",
                local_support_level="none",
                distress_pattern="none",
                recovery_pattern="mixed_with_distress",
                required_reason_codes=("mixed_distress_recovery",),
            ),
            tags=("conflict", "invariant8"),
            branch="conflict",
        ),
    )
    s.append(
        _scenario(
            "mod_short_final_isolated",
            "Single short final distress window stays moderate",
            evidence=_evidence(
                global_selected=["stress_or_overwhelm"],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="not_eligible"),
                    _win(1, start=5.0, end=7.0, eligibility="eligible", selected=["stress_or_overwhelm"]),
                ],
            ),
            expected=PolicyExpectation(
                status="ok",
                indicator="moderate_concern",
                local_support_level="isolated",
                distress_pattern="isolated",
                recovery_pattern="none",
                required_reason_codes=("global_personal_distress",),
            ),
            tags=("moderate", "window_structure"),
            branch="moderate",
        ),
    )
    s.append(
        _scenario(
            "insuf_primary_text_not_eligible_local_high_like",
            "Primary-text global not eligible cannot become high via local persistence",
            evidence=_evidence(
                source_role="primary_text",
                global_eligibility="not_eligible",
                global_selected=[],
                windows=[
                    _win(0, start=0.0, end=5.0, eligibility="eligible", selected=["hopelessness_like_language"]),
                    _win(1, start=5.0, end=10.0, eligibility="eligible", selected=["self_directed_negativity"]),
                ],
            ),
            expected=PolicyExpectation(
                status="insufficient_evidence",
                indicator="insufficient_evidence",
                local_support_level="persistent",
                distress_pattern="persistent",
                recovery_pattern="none",
                required_reason_codes=(
                    "global_not_eligible",
                    "local_eligible_without_global_eligibility",
                ),
            ),
            tags=("insufficient", "adversarial", "invariant1", "invariant9"),
            branch="insufficient",
        ),
    )

    # Deduplicate by scenario_id (defensive).
    seen: set[str] = set()
    unique: list[PolicyScenario] = []
    for item in s:
        if item.scenario_id in seen:
            raise ValueError(f"Duplicate scenario_id: {item.scenario_id}")
        seen.add(item.scenario_id)
        unique.append(item)
    return unique


ALL_POLICY_SCENARIOS: list[PolicyScenario] = _build_all_scenarios()


def get_policy_scenarios(
    *,
    tag: Optional[str] = None,
    branch: Optional[str] = None,
) -> list[PolicyScenario]:
    """Return scenarios, optionally filtered by tag or branch."""
    items = ALL_POLICY_SCENARIOS
    if tag is not None:
        items = [s for s in items if tag in s.tags]
    if branch is not None:
        items = [s for s in items if s.branch == branch]
    return list(items)


def scenario_counts_by_indicator() -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in ALL_POLICY_SCENARIOS:
        counts[s.expected.indicator] = counts.get(s.expected.indicator, 0) + 1
    return counts


def scenario_counts_by_status() -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in ALL_POLICY_SCENARIOS:
        counts[s.expected.status] = counts.get(s.expected.status, 0) + 1
    return counts


def scenario_counts_by_branch() -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in ALL_POLICY_SCENARIOS:
        counts[s.branch] = counts.get(s.branch, 0) + 1
    return counts
