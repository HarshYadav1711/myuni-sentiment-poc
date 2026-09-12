"""Phase 4C.1 — deterministic wellbeing evidence aggregation.

Pure post-processing over Phase 4B shadow classification results.

Hierarchy (explicit):
  GLOBAL source (transcript / primary text / caption):
      authorship, context, whole-content semantics
  LOCAL windows:
      temporal localization, recurrence, transitions

Do NOT majority-vote global + windows as equivalent units.
Do NOT average signal probabilities or relevance scores.
Do NOT consume visual / SigLIP / OCR / multimodal temporal negativity.

This is an evidence FEATURE layer — not a concern policy.
"""

from __future__ import annotations

from typing import Optional, Sequence

from src.wellbeing.labels import DISTRESS_LIKE_SIGNAL_IDS, RECOVERY_SIGNAL_IDS
from src.wellbeing.schemas import (
    ShadowSourceRole,
    WellbeingConflictDiagnostic,
    WellbeingEligibleRun,
    WellbeingEvidenceContext,
    WellbeingEvidenceStatus,
    WellbeingGlobalEvidence,
    WellbeingShadowSourceResult,
    WellbeingShadowWindowResult,
    WellbeingSignalTemporalEvidence,
    WellbeingTemporalEvidence,
)

# Prefer full-content authored sources for global evidence.
_GLOBAL_SOURCE_PRIORITY: tuple[ShadowSourceRole, ...] = (
    "transcript",
    "primary_text",
    "caption",
)


def _selected_signal_ids(classification: object) -> list[str]:
    signals = getattr(classification, "signals", None) or []
    return [s.signal for s in signals if getattr(s, "selected", False)]


def _pick_global_source(
    source_results: Sequence[WellbeingShadowSourceResult],
) -> Optional[WellbeingShadowSourceResult]:
    by_role = {src.source_role: src for src in source_results}
    for role in _GLOBAL_SOURCE_PRIORITY:
        if role in by_role:
            return by_role[role]
    return None


def _build_global_evidence(
    source_results: Sequence[WellbeingShadowSourceResult],
) -> WellbeingGlobalEvidence:
    src = _pick_global_source(source_results)
    if src is None:
        return WellbeingGlobalEvidence(source_present=False)

    clf = src.classification
    if clf is None:
        return WellbeingGlobalEvidence(
            source_present=True,
            source_role=src.source_role,
        )

    attribution = clf.self_attribution
    final_attr = None
    if attribution is not None:
        final_attr = attribution.final_label or attribution.label

    return WellbeingGlobalEvidence(
        source_present=True,
        source_role=src.source_role,
        classification_status=clf.status,
        relevance=clf.relevance.label if clf.relevance else None,
        target=clf.target.label if clf.target else None,
        eligibility_status=clf.eligibility_status,
        personal_wellbeing_eligible=bool(clf.personal_wellbeing_eligible),
        final_attribution=final_attr,
        selected_signals=_selected_signal_ids(clf),
    )


def _eligible_runs(
    eligible_windows: Sequence[WellbeingShadowWindowResult],
) -> list[WellbeingEligibleRun]:
    """Consecutive eligible windows by chronological window_index order."""
    if not eligible_windows:
        return []

    runs: list[WellbeingEligibleRun] = []
    run_buf: list[WellbeingShadowWindowResult] = [eligible_windows[0]]

    def _flush(buf: list[WellbeingShadowWindowResult]) -> WellbeingEligibleRun:
        signal_set: set[str] = set()
        for win in buf:
            if win.classification is not None:
                signal_set.update(_selected_signal_ids(win.classification))
        start_w = buf[0]
        end_w = buf[-1]
        return WellbeingEligibleRun(
            start_window=start_w.window_index,
            end_window=end_w.window_index,
            start=start_w.start,
            end=end_w.end,
            window_count=len(buf),
            duration_seconds=float(end_w.end) - float(start_w.start),
            signal_ids=sorted(signal_set),
        )

    for win in eligible_windows[1:]:
        prev = run_buf[-1]
        # Consecutive by evaluated eligible sequence adjacency in index order:
        # a run breaks when window_index is not prev+1 among eligible set —
        # i.e. any gap in indices between eligible windows ends the run.
        if win.window_index == prev.window_index + 1:
            run_buf.append(win)
        else:
            runs.append(_flush(run_buf))
            run_buf = [win]
    runs.append(_flush(run_buf))
    return runs


def _signal_temporal_evidence(
    eligible_windows: Sequence[WellbeingShadowWindowResult],
) -> list[WellbeingSignalTemporalEvidence]:
    """Selected signals in eligible windows only — counts/indices, no score avg."""
    by_signal: dict[str, list[WellbeingShadowWindowResult]] = {}
    for win in eligible_windows:
        if win.classification is None:
            continue
        for sig_id in _selected_signal_ids(win.classification):
            by_signal.setdefault(sig_id, []).append(win)

    out: list[WellbeingSignalTemporalEvidence] = []
    for signal in sorted(by_signal.keys()):
        wins = by_signal[signal]
        # Preserve chronological order; de-dupe indices if needed.
        indices = [w.window_index for w in wins]
        out.append(
            WellbeingSignalTemporalEvidence(
                signal=signal,
                window_count=len(indices),
                window_indices=indices,
                first_start=wins[0].start,
                last_end=wins[-1].end,
            ),
        )
    return out


def _window_has_distress(selected: Sequence[str]) -> bool:
    return any(s in DISTRESS_LIKE_SIGNAL_IDS for s in selected)


def _window_has_recovery(selected: Sequence[str]) -> bool:
    return any(s in RECOVERY_SIGNAL_IDS for s in selected)


def _build_temporal_evidence(
    window_results: Sequence[WellbeingShadowWindowResult],
) -> WellbeingTemporalEvidence:
    # Chronological order by window_index, then start.
    windows = sorted(
        window_results,
        key=lambda w: (w.window_index, w.start),
    )
    evaluated = len(windows)
    eligible_w = 0
    uncertain_w = 0
    not_eligible_w = 0
    eligible_windows: list[WellbeingShadowWindowResult] = []
    distress_count = 0
    recovery_count = 0
    mixed_count = 0

    for win in windows:
        clf = win.classification
        if clf is None:
            not_eligible_w += 1
            continue
        status = clf.eligibility_status
        if status == "eligible":
            eligible_w += 1
            eligible_windows.append(win)
            selected = _selected_signal_ids(clf)
            has_distress = _window_has_distress(selected)
            has_recovery = _window_has_recovery(selected)
            if has_distress:
                distress_count += 1
            if has_recovery:
                recovery_count += 1
            if has_distress and has_recovery:
                mixed_count += 1
        elif status == "uncertain":
            uncertain_w += 1
        else:
            not_eligible_w += 1

    runs = _eligible_runs(eligible_windows)
    longest_windows = 0
    longest_seconds = 0.0
    if runs:
        # Longest by window count; on ties prefer longer duration.
        best = max(runs, key=lambda r: (r.window_count, r.duration_seconds))
        longest_windows = best.window_count
        longest_seconds = best.duration_seconds

    fraction: Optional[float] = None
    if evaluated > 0:
        fraction = eligible_w / evaluated

    first_start = eligible_windows[0].start if eligible_windows else None
    last_end = eligible_windows[-1].end if eligible_windows else None

    return WellbeingTemporalEvidence(
        evaluated_window_count=evaluated,
        eligible_window_count=eligible_w,
        uncertain_window_count=uncertain_w,
        not_eligible_window_count=not_eligible_w,
        eligible_window_indices=[w.window_index for w in eligible_windows],
        eligible_window_fraction=fraction,
        first_eligible_start=first_start,
        last_eligible_end=last_end,
        eligible_runs=runs,
        longest_eligible_run_windows=longest_windows,
        longest_eligible_run_seconds=longest_seconds,
        distress_window_count=distress_count,
        recovery_window_count=recovery_count,
        mixed_signal_window_count=mixed_count,
        signal_evidence=_signal_temporal_evidence(eligible_windows),
    )


def _conflict_diagnostics(
    global_ev: WellbeingGlobalEvidence,
    temporal_ev: WellbeingTemporalEvidence,
) -> list[WellbeingConflictDiagnostic]:
    """Expose disagreements only — do not resolve them."""
    codes: list[WellbeingConflictDiagnostic] = []
    local_eligible = temporal_ev.eligible_window_count > 0
    global_eligible = (
        global_ev.source_present
        and global_ev.eligibility_status == "eligible"
    )
    global_not_eligible = (
        global_ev.source_present
        and global_ev.eligibility_status == "not_eligible"
    )

    if global_eligible and not local_eligible:
        codes.append("global_eligible_no_local_support")
    if global_not_eligible and local_eligible:
        codes.append("local_eligible_without_global_eligibility")

    global_selected = global_ev.selected_signals
    global_recovery = _window_has_recovery(global_selected)
    global_distress = _window_has_distress(global_selected)

    if global_recovery and temporal_ev.distress_window_count > 0:
        codes.append("global_recovery_local_distress")
    if global_distress and temporal_ev.recovery_window_count > 0:
        codes.append("global_distress_local_recovery")

    return codes


def _evidence_ids(
    global_ev: WellbeingGlobalEvidence,
    window_results: Sequence[WellbeingShadowWindowResult],
) -> list[str]:
    ids: list[str] = []
    if global_ev.source_present and global_ev.source_role is not None:
        ids.append(f"wellbeing-global-{global_ev.source_role}")
    for win in sorted(window_results, key=lambda w: w.window_index):
        ids.append(f"wellbeing-window-{win.window_index}")
    return ids


def _evidence_status(
    global_ev: WellbeingGlobalEvidence,
    temporal_ev: WellbeingTemporalEvidence,
    *,
    shadow_status: Optional[str],
) -> WellbeingEvidenceStatus:
    if shadow_status == "error":
        return "error"
    if (
        not global_ev.source_present
        and temporal_ev.evaluated_window_count == 0
    ):
        if shadow_status in {"insufficient_text", "classifier_unavailable"}:
            return "insufficient"
        return "empty"
    return "ok"


def build_wellbeing_evidence_context(
    *,
    source_results: Sequence[WellbeingShadowSourceResult] = (),
    window_results: Sequence[WellbeingShadowWindowResult] = (),
    shadow_status: Optional[str] = None,
) -> WellbeingEvidenceContext:
    """Build deterministic wellbeing evidence from shadow classification results.

    Consumes WellbeingShadow* classification outputs only. Never inspects
    visual sentiment, SigLIP, faces, OCR, negative_persistence, or trajectory.
    """
    global_ev = _build_global_evidence(source_results)
    temporal_ev = _build_temporal_evidence(window_results)
    status = _evidence_status(
        global_ev,
        temporal_ev,
        shadow_status=shadow_status,
    )
    return WellbeingEvidenceContext(
        status=status,
        global_evidence=global_ev,
        temporal_evidence=temporal_ev,
        evidence_ids=_evidence_ids(global_ev, window_results),
        conflict_diagnostics=_conflict_diagnostics(global_ev, temporal_ev),
        affects_final_assessment=False,
    )
