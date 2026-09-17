"""Phase 4C.2 — deterministic shadow wellbeing policy candidate.

Consumes ONLY ``WellbeingEvidenceContext``. No model calls, no network,
no visual/OCR/sentiment/OpenRouter inputs.

Hierarchy:
  GLOBAL = contextual eligibility authority (authorship / whole-content)
  WINDOWS = temporal / local support (localization / recurrence)

Categories are CONTENT-LEVEL wellbeing evidence labels.
They are NOT diagnoses, clinical risk levels, or mental-health scores.

``persistent`` distress = >= 2 consecutive eligible distress-supporting
windows — a transparent engineering POC recurrence rule, not clinically
validated.

Shadow-only: never replaces compute_wellbeing_indicator() /
FinalTemporalAssessment / Gradio client output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from src.wellbeing.labels import DISTRESS_LIKE_SIGNAL_IDS, RECOVERY_SIGNAL_IDS
from src.wellbeing.schemas import (
    WellbeingDistressPattern,
    WellbeingEvidenceContext,
    WellbeingGlobalEvidence,
    WellbeingLocalSupportLevel,
    WellbeingPolicyCandidate,
    WellbeingRecoveryPattern,
    WellbeingTemporalEvidence,
)

POLICY_VERSION = "phase4c2-v1"


@dataclass(frozen=True)
class _IndexRun:
    """Consecutive window-index run with provenance timestamps."""

    window_indices: tuple[int, ...]
    start: Optional[float]
    end: Optional[float]

    @property
    def window_count(self) -> int:
        return len(self.window_indices)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.start is None or self.end is None:
            return None
        return float(self.end) - float(self.start)


def _has_distress(signals: Sequence[str]) -> bool:
    return any(s in DISTRESS_LIKE_SIGNAL_IDS for s in signals)


def _has_recovery(signals: Sequence[str]) -> bool:
    return any(s in RECOVERY_SIGNAL_IDS for s in signals)


def _window_ids_for_indices(
    evidence_ids: Sequence[str],
    indices: Sequence[int],
) -> list[str]:
    wanted = {f"wellbeing-window-{i}" for i in indices}
    return [eid for eid in evidence_ids if eid in wanted]


def _global_evidence_id(evidence: WellbeingEvidenceContext) -> Optional[str]:
    role = evidence.global_evidence.source_role
    if role is None:
        return None
    candidate = f"wellbeing-global-{role}"
    if candidate in evidence.evidence_ids:
        return candidate
    return None


def _infer_window_bounds(
    temporal: WellbeingTemporalEvidence,
) -> tuple[dict[int, float], dict[int, float]]:
    """Best-effort per-window start/end from signal evidence + eligible runs."""
    starts: dict[int, float] = {}
    ends: dict[int, float] = {}

    for run in temporal.eligible_runs:
        starts[run.start_window] = float(run.start)
        ends[run.end_window] = float(run.end)
        # Singleton run: same window is both endpoints.
        if run.window_count == 1:
            starts[run.start_window] = float(run.start)
            ends[run.end_window] = float(run.end)

    for sig in temporal.signal_evidence:
        if not sig.window_indices:
            continue
        first_idx = sig.window_indices[0]
        last_idx = sig.window_indices[-1]
        if sig.first_start is not None:
            prev = starts.get(first_idx)
            starts[first_idx] = (
                float(sig.first_start)
                if prev is None
                else min(prev, float(sig.first_start))
            )
        if sig.last_end is not None:
            prev = ends.get(last_idx)
            ends[last_idx] = (
                float(sig.last_end)
                if prev is None
                else max(prev, float(sig.last_end))
            )
        # Single-window signal pins both bounds for that window.
        if len(sig.window_indices) == 1 and sig.first_start is not None and sig.last_end is not None:
            idx = sig.window_indices[0]
            starts[idx] = float(sig.first_start)
            ends[idx] = float(sig.last_end)

    return starts, ends


def _build_index_runs(
    indices: Sequence[int],
    starts: dict[int, float],
    ends: dict[int, float],
) -> list[_IndexRun]:
    """Consecutive runs by window_index adjacency (not fixed duration)."""
    ordered = sorted(set(indices))
    if not ordered:
        return []

    runs: list[_IndexRun] = []
    buf: list[int] = [ordered[0]]

    def _flush(group: list[int]) -> _IndexRun:
        first, last = group[0], group[-1]
        return _IndexRun(
            window_indices=tuple(group),
            start=starts.get(first),
            end=ends.get(last),
        )

    for idx in ordered[1:]:
        if idx == buf[-1] + 1:
            buf.append(idx)
        else:
            runs.append(_flush(buf))
            buf = [idx]
    runs.append(_flush(buf))
    return runs


def _distress_window_indices(temporal: WellbeingTemporalEvidence) -> list[int]:
    idxs: set[int] = set()
    for sig in temporal.signal_evidence:
        if sig.signal in DISTRESS_LIKE_SIGNAL_IDS:
            idxs.update(sig.window_indices)
    return sorted(idxs)


def _recovery_window_indices(temporal: WellbeingTemporalEvidence) -> list[int]:
    idxs: set[int] = set()
    for sig in temporal.signal_evidence:
        if sig.signal in RECOVERY_SIGNAL_IDS:
            idxs.update(sig.window_indices)
    return sorted(idxs)


def _local_support_level(temporal: WellbeingTemporalEvidence) -> WellbeingLocalSupportLevel:
    """Structural temporal support over eligible windows (not clinical persistence)."""
    n = temporal.eligible_window_count
    if n == 0:
        return "none"
    longest = temporal.longest_eligible_run_windows
    if longest >= 2:
        return "persistent"
    if n >= 2:
        return "recurrent"
    return "isolated"


def _distress_pattern(
    *,
    distress_indices: Sequence[int],
    distress_runs: Sequence[_IndexRun],
    recovery_indices: Sequence[int],
    global_has_recovery: bool,
    mixed_window_count: int,
) -> WellbeingDistressPattern:
    n = len(distress_indices)
    if n == 0:
        return "none"

    longest = max((r.window_count for r in distress_runs), default=0)
    has_local_recovery = len(recovery_indices) > 0
    mixed = (
        mixed_window_count > 0
        or (n > 0 and has_local_recovery)
        or (n > 0 and global_has_recovery)
    )

    if longest >= 2:
        base: WellbeingDistressPattern = "persistent"
    elif n >= 2:
        base = "recurrent"
    else:
        base = "isolated"

    if mixed and base != "none":
        # Coexistence flag — decision table may still abstain via conflict.
        return "mixed_with_recovery"
    return base


def _recovery_pattern(
    *,
    global_has_recovery: bool,
    recovery_indices: Sequence[int],
    recovery_runs: Sequence[_IndexRun],
    distress_indices: Sequence[int],
    global_has_distress: bool,
    mixed_window_count: int,
) -> WellbeingRecoveryPattern:
    local_n = len(recovery_indices)
    has_distress = len(distress_indices) > 0 or global_has_distress
    mixed = (
        mixed_window_count > 0
        or (local_n > 0 and has_distress)
        or (global_has_recovery and has_distress)
    )

    if mixed and (global_has_recovery or local_n > 0):
        return "mixed_with_distress"

    if local_n == 0 and not global_has_recovery:
        return "none"
    if global_has_recovery and local_n == 0:
        return "global_only"
    if local_n > 0 and not global_has_recovery:
        # Local-only may still be recurrent.
        longest = max((r.window_count for r in recovery_runs), default=0)
        if local_n >= 2 or longest >= 2:
            return "recurrent"
        return "local_only"
    # Global + local recovery without distress mix
    longest = max((r.window_count for r in recovery_runs), default=0)
    if local_n >= 2 or longest >= 2:
        return "recurrent"
    return "global_only"


def _structural_distress_level(
    distress_indices: Sequence[int],
    distress_runs: Sequence[_IndexRun],
) -> WellbeingLocalSupportLevel:
    """Underlying structural label ignoring mixed_with_recovery overlay."""
    n = len(distress_indices)
    if n == 0:
        return "none"
    longest = max((r.window_count for r in distress_runs), default=0)
    if longest >= 2:
        return "persistent"
    if n >= 2:
        return "recurrent"
    return "isolated"


def build_wellbeing_policy_candidate(
    evidence: Optional[WellbeingEvidenceContext],
) -> WellbeingPolicyCandidate:
    """Build deterministic shadow policy candidate from evidence context only."""
    if evidence is None:
        return WellbeingPolicyCandidate(
            status="unavailable",
            indicator="insufficient_evidence",
            reason_codes=["global_unavailable", "no_personal_wellbeing_evidence"],
            policy_version=POLICY_VERSION,
            affects_final_assessment=False,
        )

    global_ev: WellbeingGlobalEvidence = evidence.global_evidence
    temporal: WellbeingTemporalEvidence = evidence.temporal_evidence
    reasons: list[str] = []
    supporting: list[str] = []
    conflicting: list[str] = []

    starts, ends = _infer_window_bounds(temporal)
    distress_idxs = _distress_window_indices(temporal)
    recovery_idxs = _recovery_window_indices(temporal)
    distress_runs = _build_index_runs(distress_idxs, starts, ends)
    recovery_runs = _build_index_runs(recovery_idxs, starts, ends)

    global_signals = list(global_ev.selected_signals)
    global_has_distress = _has_distress(global_signals)
    global_has_recovery = _has_recovery(global_signals)
    global_has_mixed = global_has_distress and global_has_recovery

    local_support = _local_support_level(temporal)
    distress_struct = _structural_distress_level(distress_idxs, distress_runs)
    distress_pat = _distress_pattern(
        distress_indices=distress_idxs,
        distress_runs=distress_runs,
        recovery_indices=recovery_idxs,
        global_has_recovery=global_has_recovery,
        mixed_window_count=temporal.mixed_signal_window_count,
    )
    recovery_pat = _recovery_pattern(
        global_has_recovery=global_has_recovery,
        recovery_indices=recovery_idxs,
        recovery_runs=recovery_runs,
        distress_indices=distress_idxs,
        global_has_distress=global_has_distress,
        mixed_window_count=temporal.mixed_signal_window_count,
    )

    global_id = _global_evidence_id(evidence)
    local_eligible = temporal.eligible_window_count > 0
    global_eligible = bool(
        global_ev.source_present and global_ev.eligibility_status == "eligible",
    )

    # --- support-level reason codes (descriptive) ---
    if local_support == "none":
        reasons.append("no_local_support")
    elif local_support == "isolated":
        reasons.append("isolated_local_support")
    elif local_support == "recurrent":
        reasons.append("recurrent_local_support")
    elif local_support == "persistent":
        reasons.append("persistent_local_support")

    if distress_idxs:
        reasons.append("local_distress_present")
    if recovery_idxs:
        reasons.append("local_recovery_present")

    for diag in evidence.conflict_diagnostics:
        if diag not in reasons:
            reasons.append(diag)

    # --- Global gate / availability ---
    if evidence.status in {"empty", "insufficient"} and not global_ev.source_present:
        reasons.extend(["global_unavailable", "no_personal_wellbeing_evidence"])
        return WellbeingPolicyCandidate(
            status="insufficient_evidence",
            indicator="insufficient_evidence",
            reason_codes=_unique(reasons),
            supporting_evidence_ids=[],
            conflict_evidence_ids=[],
            global_eligible=False,
            local_support_level=local_support,
            distress_pattern=distress_pat,
            recovery_pattern=recovery_pat,
            policy_version=POLICY_VERSION,
            affects_final_assessment=False,
        )

    if evidence.status == "error":
        reasons.append("global_unavailable")
        return WellbeingPolicyCandidate(
            status="unavailable",
            indicator="insufficient_evidence",
            reason_codes=_unique(reasons),
            global_eligible=False,
            local_support_level=local_support,
            distress_pattern=distress_pat,
            recovery_pattern=recovery_pat,
            policy_version=POLICY_VERSION,
            affects_final_assessment=False,
        )

    if not global_ev.source_present:
        reasons.extend(["global_unavailable", "no_personal_wellbeing_evidence"])
        if local_eligible:
            reasons.append("local_eligible_without_global_eligibility")
            conflicting.extend(
                _window_ids_for_indices(evidence.evidence_ids, temporal.eligible_window_indices),
            )
        return WellbeingPolicyCandidate(
            status="insufficient_evidence",
            indicator="insufficient_evidence",
            reason_codes=_unique(reasons),
            supporting_evidence_ids=[],
            conflict_evidence_ids=_unique(conflicting),
            global_eligible=False,
            local_support_level=local_support,
            distress_pattern=distress_pat,
            recovery_pattern=recovery_pat,
            policy_version=POLICY_VERSION,
            affects_final_assessment=False,
        )

    clf_status = global_ev.classification_status
    if clf_status in {"error", "classifier_unavailable"}:
        reasons.append("global_unavailable")
        return WellbeingPolicyCandidate(
            status="unavailable",
            indicator="insufficient_evidence",
            reason_codes=_unique(reasons),
            supporting_evidence_ids=[global_id] if global_id else [],
            global_eligible=False,
            local_support_level=local_support,
            distress_pattern=distress_pat,
            recovery_pattern=recovery_pat,
            policy_version=POLICY_VERSION,
            affects_final_assessment=False,
        )

    if global_ev.eligibility_status == "uncertain":
        reasons.append("global_uncertain")
        if local_eligible:
            reasons.append("local_eligible_without_global_eligibility")
        return WellbeingPolicyCandidate(
            status="insufficient_evidence",
            indicator="insufficient_evidence",
            reason_codes=_unique(reasons),
            supporting_evidence_ids=[global_id] if global_id else [],
            conflict_evidence_ids=_window_ids_for_indices(
                evidence.evidence_ids,
                temporal.eligible_window_indices,
            )
            if local_eligible
            else [],
            global_eligible=False,
            local_support_level=local_support,
            distress_pattern=distress_pat,
            recovery_pattern=recovery_pat,
            policy_version=POLICY_VERSION,
            affects_final_assessment=False,
        )

    if global_ev.eligibility_status != "eligible":
        reasons.append("global_not_eligible")
        reasons.append("no_personal_wellbeing_evidence")
        if local_eligible:
            reasons.append("local_eligible_without_global_eligibility")
            conflicting.extend(
                _window_ids_for_indices(
                    evidence.evidence_ids,
                    temporal.eligible_window_indices,
                ),
            )
        return WellbeingPolicyCandidate(
            status="insufficient_evidence",
            indicator="insufficient_evidence",
            reason_codes=_unique(reasons),
            supporting_evidence_ids=[global_id] if global_id else [],
            conflict_evidence_ids=_unique(conflicting),
            global_eligible=False,
            local_support_level=local_support,
            distress_pattern=distress_pat,
            recovery_pattern=recovery_pat,
            policy_version=POLICY_VERSION,
            affects_final_assessment=False,
        )

    # --- Global is eligible ---
    assert global_eligible
    if global_id:
        supporting.append(global_id)

    if global_has_distress:
        reasons.append("global_personal_distress")
    if global_has_recovery:
        reasons.append("global_personal_recovery")

    if temporal.eligible_window_count == 0:
        if "global_eligible_no_local_support" not in reasons:
            reasons.append("global_eligible_no_local_support")

    # Strong unresolved mixed conflicts → abstain
    strong_conflict = False
    if global_has_mixed:
        strong_conflict = True
        reasons.append("mixed_distress_recovery")
        reasons.append("global_local_conflict")
    if global_has_recovery and distress_struct == "persistent":
        strong_conflict = True
        reasons.append("mixed_distress_recovery")
        reasons.append("global_local_conflict")
    if global_has_distress and (
        recovery_pat == "recurrent"
        or (len(recovery_idxs) >= 2)
    ):
        # global distress + recurrent local recovery
        strong_conflict = True
        reasons.append("mixed_distress_recovery")
        reasons.append("global_local_conflict")

    if strong_conflict:
        conflicting.extend(supporting)
        conflicting.extend(
            _window_ids_for_indices(evidence.evidence_ids, distress_idxs),
        )
        conflicting.extend(
            _window_ids_for_indices(evidence.evidence_ids, recovery_idxs),
        )
        return WellbeingPolicyCandidate(
            status="conflict",
            indicator="insufficient_evidence",
            reason_codes=_unique(reasons),
            supporting_evidence_ids=[],
            conflict_evidence_ids=_unique(conflicting),
            global_eligible=True,
            local_support_level=local_support,
            distress_pattern=distress_pat,
            recovery_pattern=recovery_pat,
            policy_version=POLICY_VERSION,
            affects_final_assessment=False,
        )

    # Consistent path reason
    if distress_idxs and global_has_distress:
        reasons.append("global_local_consistent")
    elif not distress_idxs and not recovery_idxs:
        reasons.append("global_local_consistent")

    supporting.extend(_window_ids_for_indices(evidence.evidence_ids, distress_idxs))
    supporting.extend(_window_ids_for_indices(evidence.evidence_ids, recovery_idxs))

    # --- HIGH: global distress + persistent local distress ---
    if global_has_distress and distress_struct == "persistent":
        if "persistent_local_support" not in reasons:
            reasons.append("persistent_local_support")
        return WellbeingPolicyCandidate(
            status="ok",
            indicator="high_concern",
            reason_codes=_unique(reasons),
            supporting_evidence_ids=_unique(supporting),
            conflict_evidence_ids=[],
            global_eligible=True,
            local_support_level=local_support,
            distress_pattern=distress_pat,
            recovery_pattern=recovery_pat,
            policy_version=POLICY_VERSION,
            affects_final_assessment=False,
        )

    # --- MODERATE: global distress + non-persistent local distress pattern ---
    if global_has_distress:
        # Local distress may be none / isolated / recurrent (including zero windows).
        return WellbeingPolicyCandidate(
            status="ok",
            indicator="moderate_concern",
            reason_codes=_unique(reasons),
            supporting_evidence_ids=_unique(supporting),
            conflict_evidence_ids=[],
            global_eligible=True,
            local_support_level=local_support,
            distress_pattern=distress_pat,
            recovery_pattern=recovery_pat,
            policy_version=POLICY_VERSION,
            affects_final_assessment=False,
        )

    # --- LOW: recovery-only or no-distress global, without recurrent/persistent local distress ---
    meaningful_local_distress = distress_struct in {"recurrent", "persistent"}
    if global_has_recovery and not global_has_distress and not meaningful_local_distress:
        return WellbeingPolicyCandidate(
            status="ok",
            indicator="low_concern",
            reason_codes=_unique(reasons),
            supporting_evidence_ids=_unique(supporting),
            conflict_evidence_ids=[],
            global_eligible=True,
            local_support_level=local_support,
            distress_pattern=distress_pat,
            recovery_pattern=recovery_pat,
            policy_version=POLICY_VERSION,
            affects_final_assessment=False,
        )

    if not global_has_distress and not meaningful_local_distress:
        # Eligible global with no selected distress (and no strong local distress).
        return WellbeingPolicyCandidate(
            status="ok",
            indicator="low_concern",
            reason_codes=_unique(reasons),
            supporting_evidence_ids=_unique(supporting),
            conflict_evidence_ids=[],
            global_eligible=True,
            local_support_level=local_support,
            distress_pattern=distress_pat,
            recovery_pattern=recovery_pat,
            policy_version=POLICY_VERSION,
            affects_final_assessment=False,
        )

    # Recovery global + recurrent/persistent local distress without earlier conflict
    # (persistent already handled as conflict when global recovery). Remaining:
    if meaningful_local_distress:
        reasons.append("mixed_distress_recovery")
        reasons.append("global_local_conflict")
        conflicting.extend(supporting)
        conflicting.extend(
            _window_ids_for_indices(evidence.evidence_ids, distress_idxs),
        )
        return WellbeingPolicyCandidate(
            status="conflict",
            indicator="insufficient_evidence",
            reason_codes=_unique(reasons),
            supporting_evidence_ids=[],
            conflict_evidence_ids=_unique(conflicting),
            global_eligible=True,
            local_support_level=local_support,
            distress_pattern=distress_pat,
            recovery_pattern=recovery_pat,
            policy_version=POLICY_VERSION,
            affects_final_assessment=False,
        )

    reasons.append("no_personal_wellbeing_evidence")
    return WellbeingPolicyCandidate(
        status="insufficient_evidence",
        indicator="insufficient_evidence",
        reason_codes=_unique(reasons),
        supporting_evidence_ids=_unique(supporting),
        conflict_evidence_ids=[],
        global_eligible=True,
        local_support_level=local_support,
        distress_pattern=distress_pat,
        recovery_pattern=recovery_pat,
        policy_version=POLICY_VERSION,
        affects_final_assessment=False,
    )


def _unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


# Expose helpers for tests / inspection (deterministic, no I/O).
def build_distress_runs(
    evidence: WellbeingEvidenceContext,
) -> list[_IndexRun]:
    starts, ends = _infer_window_bounds(evidence.temporal_evidence)
    return _build_index_runs(
        _distress_window_indices(evidence.temporal_evidence),
        starts,
        ends,
    )


def build_recovery_runs(
    evidence: WellbeingEvidenceContext,
) -> list[_IndexRun]:
    starts, ends = _infer_window_bounds(evidence.temporal_evidence)
    return _build_index_runs(
        _recovery_window_indices(evidence.temporal_evidence),
        starts,
        ends,
    )
