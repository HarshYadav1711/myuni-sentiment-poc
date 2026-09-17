"""Phase 4C.4 — end-to-end shadow chain replay harness.

Replays:

  structured Phase 4B classifier outputs
  → build_wellbeing_evidence_context (production)
  → build_wellbeing_policy_candidate (production)

No DeBERTa / Whisper / SigLIP / OCR / OpenRouter / video inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from evaluation.wellbeing.shadow_replay_cases import (
    ALL_SHADOW_REPLAY_CASES,
    POLICY_VERSION,
    ShadowReplayCase,
    get_replay_cases,
    replay_counts_by_expected_indicator,
    replay_counts_by_expected_status,
    replay_counts_by_modality,
)
from src.wellbeing.evidence import build_wellbeing_evidence_context
from src.wellbeing.labels import DISTRESS_LIKE_SIGNAL_IDS, RECOVERY_SIGNAL_IDS
from src.wellbeing.policy_candidate import build_wellbeing_policy_candidate
from src.wellbeing.schemas import (
    WellbeingEvidenceContext,
    WellbeingPolicyCandidate,
    WellbeingShadowAnalysis,
)

FORBIDDEN_RAW_TEXT_KEYS = frozenset(
    {
        "text",
        "raw_text",
        "transcript_text",
        "caption_text",
        "speech_text",
        "utterance",
        "speech_segments",
        "content",
        "body",
        "message",
    },
)

FORBIDDEN_SOURCE_ROLES = frozenset({"ocr", "visual", "siglip", "face", "faces", "frame"})

CHAIN_INVARIANT_IDS = tuple(f"C{i}" for i in range(1, 13))


@dataclass
class ReplayCaseResult:
    case_id: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    actual_policy_indicator: str = ""
    actual_policy_status: str = ""
    expected_policy_indicator: str = ""
    expected_policy_status: str = ""


@dataclass
class ChainInvariantResult:
    invariant_id: str
    description: str
    passed: bool
    violations: list[str] = field(default_factory=list)


@dataclass
class ShadowReplayReport:
    case_total: int = 0
    exact_agreement: int = 0
    determinism_failures: int = 0
    input_mutation_failures: int = 0
    evidence_grounding_failures: int = 0
    privacy_key_failures: int = 0
    source_boundary_failures: int = 0
    authority_isolation_failures: int = 0
    chain_invariant_count: int = 0
    chain_invariant_violations: int = 0
    case_results: list[ReplayCaseResult] = field(default_factory=list)
    invariant_results: list[ChainInvariantResult] = field(default_factory=list)
    counts_by_expected_indicator: dict[str, int] = field(default_factory=dict)
    counts_by_expected_status: dict[str, int] = field(default_factory=dict)
    counts_by_modality: dict[str, int] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return (
            all(r.passed for r in self.case_results)
            and self.chain_invariant_violations == 0
            and self.determinism_failures == 0
            and self.input_mutation_failures == 0
            and self.evidence_grounding_failures == 0
            and self.privacy_key_failures == 0
            and self.source_boundary_failures == 0
            and self.authority_isolation_failures == 0
        )

    def summary_dict(self) -> dict[str, Any]:
        return {
            "case_total": self.case_total,
            "exact_agreement": self.exact_agreement,
            "determinism_failures": self.determinism_failures,
            "input_mutation_failures": self.input_mutation_failures,
            "evidence_grounding_failures": self.evidence_grounding_failures,
            "privacy_key_failures": self.privacy_key_failures,
            "source_boundary_failures": self.source_boundary_failures,
            "authority_isolation_failures": self.authority_isolation_failures,
            "chain_invariant_count": self.chain_invariant_count,
            "chain_invariant_violations": self.chain_invariant_violations,
            "counts_by_expected_indicator": dict(self.counts_by_expected_indicator),
            "counts_by_expected_status": dict(self.counts_by_expected_status),
            "counts_by_modality": dict(self.counts_by_modality),
            "all_passed": self.all_passed,
            "policy_version": POLICY_VERSION,
        }


def replay_shadow_chain(
    case: ShadowReplayCase,
) -> tuple[WellbeingEvidenceContext, WellbeingPolicyCandidate]:
    """Run production evidence + policy builders on structured classifier outputs."""
    evidence = build_wellbeing_evidence_context(
        source_results=list(case.source_results),
        window_results=list(case.window_results),
        shadow_status="ok" if (case.source_results or case.window_results) else None,
    )
    policy = build_wellbeing_policy_candidate(evidence)
    return evidence, policy


def _collect_keys(obj: Any, keys: Optional[set[str]] = None) -> set[str]:
    if keys is None:
        keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k))
            _collect_keys(v, keys)
    elif isinstance(obj, list):
        for item in obj:
            _collect_keys(item, keys)
    return keys


def assert_no_raw_text_keys(payload: dict[str, Any]) -> list[str]:
    found = _collect_keys(payload) & FORBIDDEN_RAW_TEXT_KEYS
    return sorted(found)


def assert_source_boundaries(case: ShadowReplayCase) -> list[str]:
    failures: list[str] = []
    roles = {src.source_role for src in case.source_results}
    for role in roles:
        if role in FORBIDDEN_SOURCE_ROLES or str(role).lower() in FORBIDDEN_SOURCE_ROLES:
            failures.append(f"forbidden_source_role={role}")
    if case.modality == "text":
        if roles and roles != {"primary_text"}:
            # allow empty for failure-like empties tagged text? text cases should be primary_text
            if any(r not in {"primary_text"} for r in roles):
                failures.append(f"text_modality_bad_roles={sorted(roles)}")
        if case.window_results:
            failures.append("text_modality_has_windows")
    if case.modality == "audio":
        if any(r not in {"transcript"} for r in roles):
            failures.append(f"audio_modality_bad_roles={sorted(roles)}")
        if case.window_results:
            failures.append("audio_modality_has_windows")
    if case.modality == "caption":
        if any(r not in {"caption"} for r in roles):
            failures.append(f"caption_modality_bad_roles={sorted(roles)}")
        if case.window_results:
            failures.append("caption_modality_has_windows")
    if case.modality == "video":
        if any(r not in {"transcript"} for r in roles):
            failures.append(f"video_modality_bad_roles={sorted(roles)}")
        # windows optional but when present they are speech_window classifications
    return failures


def evaluate_replay_case(case: ShadowReplayCase) -> ReplayCaseResult:
    expected = case.expected
    failures: list[str] = []

    # Snapshot inputs for immutability.
    src_before = [s.model_dump() for s in case.source_results]
    win_before = [w.model_dump() for w in case.window_results]

    evidence1, policy1 = replay_shadow_chain(case)
    evidence2, policy2 = replay_shadow_chain(case)

    src_after = [s.model_dump() for s in case.source_results]
    win_after = [w.model_dump() for w in case.window_results]
    if src_before != src_after or win_before != win_after:
        failures.append("input_mutated")

    if evidence1.model_dump() != evidence2.model_dump() or policy1.model_dump() != policy2.model_dump():
        failures.append("determinism_failure")

    # Evidence expectations
    if evidence1.status != expected.evidence_status:
        failures.append(
            f"evidence_status expected={expected.evidence_status} actual={evidence1.status}",
        )
    if evidence1.temporal_evidence.eligible_window_count != expected.eligible_window_count:
        failures.append(
            "eligible_window_count "
            f"expected={expected.eligible_window_count} "
            f"actual={evidence1.temporal_evidence.eligible_window_count}",
        )
    if tuple(evidence1.temporal_evidence.eligible_window_indices) != expected.eligible_window_indices:
        failures.append(
            "eligible_window_indices "
            f"expected={expected.eligible_window_indices} "
            f"actual={tuple(evidence1.temporal_evidence.eligible_window_indices)}",
        )
    for diag in expected.conflict_diagnostics:
        if diag not in evidence1.conflict_diagnostics:
            failures.append(f"missing_conflict_diagnostic={diag}")

    # Policy expectations
    if policy1.status != expected.policy_status:
        failures.append(
            f"policy_status expected={expected.policy_status} actual={policy1.status}",
        )
    if policy1.indicator != expected.policy_indicator:
        failures.append(
            f"policy_indicator expected={expected.policy_indicator} actual={policy1.indicator}",
        )
    for code in expected.required_reason_codes:
        if code not in policy1.reason_codes:
            failures.append(f"missing_reason={code}")
    for code in expected.forbidden_reason_codes:
        if code in policy1.reason_codes:
            failures.append(f"forbidden_reason={code}")

    if policy1.affects_final_assessment is not False:
        failures.append("policy_affects_final_assessment")
    if evidence1.affects_final_assessment is not False:
        failures.append("evidence_affects_final_assessment")
    if expected.affects_final_assessment is not False:
        failures.append("expected_affects_final_not_false")
    if policy1.policy_version != POLICY_VERSION:
        failures.append(f"policy_version={policy1.policy_version}")

    # Evidence grounding
    eid_set = set(evidence1.evidence_ids)
    for sid in policy1.supporting_evidence_ids:
        if sid not in eid_set:
            failures.append(f"ungrounded_supporting={sid}")
    for cid in policy1.conflict_evidence_ids:
        if cid not in eid_set:
            failures.append(f"ungrounded_conflict={cid}")

    # Privacy keys on serialized chain outputs
    shadow = WellbeingShadowAnalysis(
        status="ok",
        source_results=list(case.source_results),
        window_results=list(case.window_results),
        evidence_context=evidence1,
        policy_candidate=policy1,
        affects_final_assessment=False,
    )
    for name, payload in (
        ("shadow", shadow.model_dump()),
        ("evidence", evidence1.model_dump()),
        ("policy", policy1.model_dump()),
    ):
        bad = assert_no_raw_text_keys(payload)
        if bad:
            failures.append(f"privacy_keys_{name}={bad}")

    # Source boundaries
    for f in assert_source_boundaries(case):
        failures.append(f)

    return ReplayCaseResult(
        case_id=case.case_id,
        passed=not failures,
        failures=failures,
        actual_policy_indicator=policy1.indicator,
        actual_policy_status=policy1.status,
        expected_policy_indicator=expected.policy_indicator,
        expected_policy_status=expected.policy_status,
    )


def check_chain_invariants(
    cases: Sequence[ShadowReplayCase],
) -> list[ChainInvariantResult]:
    results: list[ChainInvariantResult] = []

    def _run(case: ShadowReplayCase) -> tuple[WellbeingEvidenceContext, WellbeingPolicyCandidate]:
        return replay_shadow_chain(case)

    # C1: eligible global recovery/no distress never moderate/high
    v1: list[str] = []
    for case in cases:
        ev, pol = _run(case)
        g = ev.global_evidence
        if not (g.source_present and g.eligibility_status == "eligible"):
            continue
        has_distress = any(s in DISTRESS_LIKE_SIGNAL_IDS for s in g.selected_signals)
        has_recovery = any(s in RECOVERY_SIGNAL_IDS for s in g.selected_signals)
        if (has_recovery and not has_distress) or (not has_distress):
            # recovery-only or no-distress eligible — unless conflict path already abstained
            if pol.indicator in {"moderate_concern", "high_concern"}:
                # exception: conflict/unavailable shouldn't happen as moderate/high anyway
                v1.append(case.case_id)
    results.append(
        ChainInvariantResult(
            "C1",
            "Eligible global recovery/no distress never becomes moderate/high",
            passed=not v1,
            violations=v1,
        ),
    )

    # C2: globally non-personal never low/moderate/high
    v2: list[str] = []
    for case in cases:
        ev, pol = _run(case)
        g = ev.global_evidence
        personal = bool(g.source_present and g.eligibility_status == "eligible")
        if not personal and pol.indicator in {
            "low_concern",
            "moderate_concern",
            "high_concern",
        }:
            v2.append(case.case_id)
    results.append(
        ChainInvariantResult(
            "C2",
            "Globally non-personal content never becomes low/moderate/high",
            passed=not v2,
            violations=v2,
        ),
    )

    # C3: local-only never low/moderate/high
    v3: list[str] = []
    for case in cases:
        if "chain3" not in case.tags and case.case_id not in {
            "conflict_global_not_eligible_local_eligible",
            "conflict_global_uncertain_local_eligible",
            "conflict_local_only_no_global_source",
        }:
            # still check structurally
            pass
        ev, pol = _run(case)
        g = ev.global_evidence
        local_only = (
            (not g.source_present or g.eligibility_status != "eligible")
            and ev.temporal_evidence.eligible_window_count > 0
        )
        if local_only and pol.indicator in {
            "low_concern",
            "moderate_concern",
            "high_concern",
        }:
            v3.append(case.case_id)
    results.append(
        ChainInvariantResult(
            "C3",
            "Local-only evidence never becomes low/moderate/high",
            passed=not v3,
            violations=v3,
        ),
    )

    # C4: global distress + no local support can remain moderate
    v4: list[str] = []
    for case in cases:
        if "chain4" not in case.tags:
            continue
        _, pol = _run(case)
        if pol.indicator != "moderate_concern":
            v4.append(case.case_id)
    results.append(
        ChainInvariantResult(
            "C4",
            "Global distress + no local support can remain moderate",
            passed=not v4,
            violations=v4,
        ),
    )

    # C5: isolated local distress never high
    v5: list[str] = []
    for case in cases:
        if "chain5" not in case.tags:
            continue
        _, pol = _run(case)
        if pol.indicator == "high_concern":
            v5.append(case.case_id)
    results.append(
        ChainInvariantResult(
            "C5",
            "Isolated local distress never produces high",
            passed=not v5,
            violations=v5,
        ),
    )

    # C6: two consecutive distress windows can produce high
    v6: list[str] = []
    for case in cases:
        if "chain6" not in case.tags:
            continue
        _, pol = _run(case)
        if pol.indicator != "high_concern":
            v6.append(case.case_id)
    results.append(
        ChainInvariantResult(
            "C6",
            "Two consecutive distress windows can produce high",
            passed=not v6,
            violations=v6,
        ),
    )

    # C7: separated distress windows do not produce high
    v7: list[str] = []
    for case in cases:
        if "chain7" not in case.tags:
            continue
        _, pol = _run(case)
        if pol.indicator == "high_concern":
            v7.append(case.case_id)
    results.append(
        ChainInvariantResult(
            "C7",
            "Separated distress windows do not produce high",
            passed=not v7,
            violations=v7,
        ),
    )

    # C8: signal NAME alone cannot produce high
    v8: list[str] = []
    for case in cases:
        if "chain8" not in case.tags:
            continue
        _, pol = _run(case)
        if pol.indicator == "high_concern":
            v8.append(case.case_id)
    results.append(
        ChainInvariantResult(
            "C8",
            "Signal name alone cannot produce high",
            passed=not v8,
            violations=v8,
        ),
    )

    # C9: mixed recovery/distress conflict never averaged into category
    v9: list[str] = []
    for case in cases:
        if "chain9" not in case.tags:
            continue
        _, pol = _run(case)
        if case.expected.policy_status == "conflict":
            if pol.status != "conflict" or pol.indicator != "insufficient_evidence":
                v9.append(case.case_id)
        # mixed window moderate is allowed when not strong conflict
        if pol.indicator in {"low_concern", "moderate_concern", "high_concern"} and pol.status == "conflict":
            v9.append(case.case_id)
    results.append(
        ChainInvariantResult(
            "C9",
            "Mixed recovery/distress conflict never gets averaged",
            passed=not v9,
            violations=v9,
        ),
    )

    # C10: visual/OCR cannot appear as personal wellbeing source
    v10: list[str] = []
    for case in cases:
        for src in case.source_results:
            if src.source_role in FORBIDDEN_SOURCE_ROLES:
                v10.append(case.case_id)
    results.append(
        ChainInvariantResult(
            "C10",
            "Visual/OCR cannot appear as personal wellbeing source",
            passed=not v10,
            violations=v10,
        ),
    )

    # C11: evidence IDs remain grounded
    v11: list[str] = []
    for case in cases:
        ev, pol = _run(case)
        eid = set(ev.evidence_ids)
        for sid in pol.supporting_evidence_ids + pol.conflict_evidence_ids:
            if sid not in eid:
                v11.append(case.case_id)
                break
    results.append(
        ChainInvariantResult(
            "C11",
            "Evidence IDs remain grounded",
            passed=not v11,
            violations=v11,
        ),
    )

    # C12: candidate remains non-authoritative
    v12: list[str] = []
    for case in cases:
        ev, pol = _run(case)
        if pol.affects_final_assessment is not False or ev.affects_final_assessment is not False:
            v12.append(case.case_id)
        if pol.policy_version != POLICY_VERSION:
            v12.append(case.case_id)
    results.append(
        ChainInvariantResult(
            "C12",
            "Candidate remains non-authoritative (affects_final_assessment=False)",
            passed=not v12,
            violations=v12,
        ),
    )

    return results


def run_shadow_replay(
    cases: Optional[Sequence[ShadowReplayCase]] = None,
) -> ShadowReplayReport:
    cases = list(cases) if cases is not None else list(ALL_SHADOW_REPLAY_CASES)
    report = ShadowReplayReport(
        case_total=len(cases),
        counts_by_expected_indicator=replay_counts_by_expected_indicator(),
        counts_by_expected_status=replay_counts_by_expected_status(),
        counts_by_modality=replay_counts_by_modality(),
    )

    for case in cases:
        result = evaluate_replay_case(case)
        report.case_results.append(result)
        if result.passed:
            report.exact_agreement += 1
        if any("determinism" in f for f in result.failures):
            report.determinism_failures += 1
        if any("input_mutated" in f for f in result.failures):
            report.input_mutation_failures += 1
        if any("ungrounded" in f for f in result.failures):
            report.evidence_grounding_failures += 1
        if any("privacy_keys" in f for f in result.failures):
            report.privacy_key_failures += 1
        if any(
            x in f
            for f in result.failures
            for x in ("forbidden_source", "modality_bad", "modality_has_windows")
        ):
            report.source_boundary_failures += 1
        if any("affects_final" in f for f in result.failures):
            report.authority_isolation_failures += 1

    inv = check_chain_invariants(cases)
    report.invariant_results = inv
    report.chain_invariant_count = len(inv)
    report.chain_invariant_violations = sum(1 for r in inv if not r.passed)
    return report


def write_report_artifact(
    report: ShadowReplayReport,
    path: Optional[Path] = None,
) -> Path:
    """Write gitignored machine-generated report (not for commit by default)."""
    target = path or Path("evaluation/wellbeing/_phase4c4_report.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": report.summary_dict(),
        "failures": [
            {"case_id": r.case_id, "failures": r.failures}
            for r in report.case_results
            if not r.passed
        ],
        "invariants": [
            {
                "id": i.invariant_id,
                "passed": i.passed,
                "violations": i.violations,
                "description": i.description,
            }
            for i in report.invariant_results
        ],
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def find_replay_case(case_id: str) -> ShadowReplayCase:
    for case in ALL_SHADOW_REPLAY_CASES:
        if case.case_id == case_id:
            return case
    raise KeyError(case_id)


__all__ = [
    "ShadowReplayReport",
    "assert_no_raw_text_keys",
    "check_chain_invariants",
    "evaluate_replay_case",
    "find_replay_case",
    "get_replay_cases",
    "replay_shadow_chain",
    "run_shadow_replay",
    "write_report_artifact",
]
