"""Build client-facing FinalTemporalAssessment from deterministic + advisory reasoning."""

from __future__ import annotations

from typing import Optional, Sequence

from src.schemas import (
    FinalAssessmentStatus,
    FinalTemporalAssessment,
    TemporalContext,
    TemporalHighlight,
    TemporalReasoningResult,
)
from src.temporal.wellbeing import compute_wellbeing_indicator


def format_timestamp_range(start: float, end: float) -> str:
    """Format seconds as MM:SS–MM:SS (or H:MM:SS when needed)."""

    def _fmt(seconds: float) -> str:
        total = max(0, int(round(float(seconds))))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    return f"{_fmt(start)}–{_fmt(end)}"


def _deterministic_highlights(
    temporal: TemporalContext,
    *,
    valid_evidence_ids: Optional[set[str]] = None,
    limit: int = 5,
) -> list[TemporalHighlight]:
    """Build 1–5 highlights from authoritative deterministic features only."""
    feats = temporal.features
    out: list[TemporalHighlight] = []
    allow = valid_evidence_ids

    def _ok(ids: Sequence[str]) -> list[str]:
        cleaned = [str(i) for i in ids if i]
        if allow is None:
            return cleaned
        return [i for i in cleaned if i in allow]

    sudden = feats.sudden_negative_change
    if sudden.detected and sudden.to_window is not None:
        to_idx = int(sudden.to_window)
        start = float(sudden.to_start if sudden.to_start is not None else to_idx * temporal.window_seconds)
        end = start + float(temporal.window_seconds)
        # Prefer matching window bounds when available.
        for w in temporal.windows:
            if w.index == to_idx:
                start, end = float(w.start), float(w.end)
                break
        evid = _ok([f"window-{to_idx}"])
        out.append(
            TemporalHighlight(
                start=start,
                end=end,
                timestamp_label=format_timestamp_range(start, end),
                description="Negative emotional shift detected",
                evidence_ids=evid,
            ),
        )

    strongest = feats.strongest_negative_window
    if strongest is not None and strongest.score is not None and float(strongest.score) >= 0.45:
        evid = _ok([f"window-{strongest.index}"]) if strongest.index is not None else []
        out.append(
            TemporalHighlight(
                start=float(strongest.start),
                end=float(strongest.end),
                timestamp_label=format_timestamp_range(strongest.start, strongest.end),
                description="Strongest negative period",
                evidence_ids=evid,
            ),
        )

    # Elevated negative run: consecutive usable negative windows (persistence cue).
    elevated_added = False
    run_windows = []
    for w in temporal.windows:
        if w.usable and (
            w.dominant_label == "negative"
            or (
                w.negative_probability is not None
                and float(w.negative_probability) >= 0.45
            )
        ):
            run_windows.append(w)
        else:
            if len(run_windows) >= 2 and not elevated_added:
                first, last = run_windows[0], run_windows[-1]
                out.append(
                    TemporalHighlight(
                        start=float(first.start),
                        end=float(last.end),
                        timestamp_label=format_timestamp_range(first.start, last.end),
                        description="Negative expression remained elevated",
                        evidence_ids=_ok(
                            [f"window-{rw.index}" for rw in run_windows[:3]],
                        ),
                    ),
                )
                elevated_added = True
            run_windows = []
    if len(run_windows) >= 2 and not elevated_added:
        first, last = run_windows[0], run_windows[-1]
        out.append(
            TemporalHighlight(
                start=float(first.start),
                end=float(last.end),
                timestamp_label=format_timestamp_range(first.start, last.end),
                description="Negative expression remained elevated",
                evidence_ids=_ok([f"window-{rw.index}" for rw in run_windows[:3]]),
            ),
        )

    for conflict in feats.cross_modal_conflicts[:2]:
        evid = _ok([f"conflict-window-{conflict.window_index}", f"window-{conflict.window_index}"])
        mods = " and ".join(conflict.modalities) if conflict.modalities else "modalities"
        out.append(
            TemporalHighlight(
                start=float(conflict.window_start),
                end=float(conflict.window_end),
                timestamp_label=format_timestamp_range(
                    conflict.window_start,
                    conflict.window_end,
                ),
                description=f"Cross-modal disagreement between {mods}",
                evidence_ids=evid,
            ),
        )

    # Deduplicate by timestamp_label + description, keep chronological.
    seen: set[tuple[str, str]] = set()
    unique: list[TemporalHighlight] = []
    for item in sorted(out, key=lambda h: (h.start, h.end)):
        key = (item.timestamp_label, item.description)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= limit:
            break

    if unique:
        return unique

    # Fallback: first usable window summary so the UI always has a timeline cue.
    for w in temporal.windows:
        if not w.usable:
            continue
        label = w.dominant_label or "unclear"
        desc = (
            "Lower-negative / non-negative expressed evidence"
            if label != "negative"
            else "Negative expressed evidence in this window"
        )
        return [
            TemporalHighlight(
                start=float(w.start),
                end=float(w.end),
                timestamp_label=format_timestamp_range(w.start, w.end),
                description=desc,
                evidence_ids=_ok([f"window-{w.index}"]),
            ),
        ]
    return []


def _highlights_from_reasoning(
    reasoning: TemporalReasoningResult,
    *,
    valid_evidence_ids: Optional[set[str]] = None,
    limit: int = 5,
) -> list[TemporalHighlight]:
    """Advisory only — client highlights prefer deterministic timestamps."""
    out: list[TemporalHighlight] = []
    allow = valid_evidence_ids
    for transition in reasoning.important_transitions:
        evid = list(transition.evidence_ids or [])
        if allow is not None:
            evid = [e for e in evid if e in allow]
        out.append(
            TemporalHighlight(
                start=float(transition.start),
                end=float(transition.end),
                timestamp_label=format_timestamp_range(transition.start, transition.end),
                description=(transition.description or "").strip() or "Notable temporal transition.",
                evidence_ids=evid,
            ),
        )
        if len(out) >= limit:
            break
    return out


def _evidence_summary(
    temporal: TemporalContext,
    reasoning: Optional[TemporalReasoningResult],
) -> str:
    feats = temporal.features
    cov = feats.evidence_coverage
    parts = [
        f"Trajectory: {feats.trajectory}.",
        f"Negative persistence: {feats.negative_persistence:.2f}.",
        f"Usable coverage: {cov.overall_usable_coverage:.2f}.",
        f"Cross-modal agreement: {feats.cross_modal_agreement}.",
    ]
    if feats.cross_modal_conflicts:
        parts.append(f"Conflicts preserved: {len(feats.cross_modal_conflicts)}.")
    if reasoning is not None and reasoning.status == "ok" and reasoning.evidence:
        parts.append(f"Advisory evidence refs: {len(reasoning.evidence)}.")
    return " ".join(parts)


def _uncertainty_note(
    temporal: TemporalContext,
    reasoning: Optional[TemporalReasoningResult],
    *,
    explanation_ok: bool,
) -> str:
    notes: list[str] = []
    if not explanation_ok:
        notes.append("Context explanation temporarily unavailable.")
    cov = temporal.features.evidence_coverage
    if float(cov.overall_usable_coverage or 0.0) < 0.5:
        notes.append("Limited usable multimodal coverage across the timeline.")
    if reasoning is not None and reasoning.uncertainties:
        notes.extend(str(u).strip() for u in reasoning.uncertainties if str(u).strip())
    if not notes:
        notes.append("Interpretations are content-level only and may be incomplete.")
    # Cap length for UI.
    text = " ".join(notes)
    if len(text) > 600:
        text = text[:597] + "..."
    return text


def build_final_temporal_assessment(
    temporal: TemporalContext,
    reasoning: Optional[TemporalReasoningResult] = None,
    *,
    valid_evidence_ids: Optional[set[str]] = None,
    reasoner_configured: Optional[bool] = None,
    model_id: Optional[str] = None,
) -> FinalTemporalAssessment:
    """Compose the clean client-facing assessment.

    Deterministic temporal facts always survive. LLM failure → explanation
    unavailable; indicator still gated conservatively via context_type.
    """
    explanation_ok = bool(
        reasoning is not None
        and reasoning.status == "ok"
        and (reasoning.summary or "").strip()
    )
    context_type = (
        reasoning.context_type if reasoning is not None and reasoning.status == "ok" else "uncertain"
    )
    indicator = compute_wellbeing_indicator(temporal, context_type=context_type)

    # Client highlights are deterministic-only so the LLM cannot invent timestamps.
    highlights = _deterministic_highlights(
        temporal,
        valid_evidence_ids=valid_evidence_ids,
        limit=5,
    )

    if explanation_ok and reasoning is not None:
        summary = (reasoning.summary or "").strip()
        status: FinalAssessmentStatus = "ok"
        model = reasoning.model or model_id
    else:
        summary = ""
        status = "explanation_unavailable"
        model = model_id or (reasoning.model if reasoning is not None else None)
        if reasoning is not None and reasoning.status == "disabled":
            status = "disabled"

    if indicator == "insufficient_evidence" and status == "ok":
        # Keep explanation; indicator remains insufficient when context is non-personal.
        pass

    return FinalTemporalAssessment(
        overall_wellbeing_indicator=indicator,
        key_temporal_highlights=highlights[:5],
        summary_explanation=summary,
        evidence_summary=_evidence_summary(temporal, reasoning),
        uncertainty_note=_uncertainty_note(
            temporal,
            reasoning,
            explanation_ok=explanation_ok,
        ),
        context_type=context_type,  # type: ignore[arg-type]
        model=model,
        status=status,
        reasoner_configured=reasoner_configured,
    )
