"""Phase 4B wellbeing shadow orchestrator.

Runs the independent DeBERTa classifier in parallel with existing sentiment /
temporal analysis. Never alters FinalTemporalAssessment, fusion, sentiment,
OpenRouter reasoning, or the existing temporal wellbeing gate.

Lazy: when shadow mode is disabled, this module must not load DeBERTa.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from src.config import resolve_wellbeing_shadow_enabled
from src.wellbeing.schemas import (
    WellbeingClassificationResult,
    WellbeingShadowAnalysis,
    WellbeingShadowSourceResult,
    WellbeingShadowSummary,
    WellbeingShadowWindowResult,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ShadowItem:
    """Internal batch item (text kept only until classify_many returns)."""

    kind: str  # source | window
    source_role: str
    text: str
    window_index: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None
    usable_window: bool = False
    provenance: Optional[dict[str, Any]] = None


def _usable_text(value: Optional[str]) -> Optional[str]:
    if value is None or not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _window_speech_text(window: Any) -> Optional[str]:
    segments = getattr(window, "speech_segments", None) or []
    parts: list[str] = []
    for seg in segments:
        text = getattr(seg, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    if not parts:
        return None
    return " ".join(parts)


def collect_shadow_items(
    *,
    primary_text: Optional[str] = None,
    caption: Optional[str] = None,
    transcript: Optional[str] = None,
    temporal_context: Any = None,
) -> list[_ShadowItem]:
    """Collect textual sources for shadow classification (OCR excluded).

    Image/video OCR is intentionally not included — uncertain authorship.
    Faces / SigLIP / visual evidence are never inputs.
    """
    items: list[_ShadowItem] = []

    primary = _usable_text(primary_text)
    if primary is not None:
        items.append(
            _ShadowItem(
                kind="source",
                source_role="primary_text",
                text=primary,
                provenance={"role": "primary_text"},
            ),
        )

    cap = _usable_text(caption)
    if cap is not None:
        items.append(
            _ShadowItem(
                kind="source",
                source_role="caption",
                text=cap,
                provenance={"role": "caption"},
            ),
        )

    tr = _usable_text(transcript)
    if tr is not None:
        items.append(
            _ShadowItem(
                kind="source",
                source_role="transcript",
                text=tr,
                provenance={"role": "transcript"},
            ),
        )

    if temporal_context is not None:
        windows = getattr(temporal_context, "windows", None) or []
        for window in windows:
            speech_text = _window_speech_text(window)
            if speech_text is None:
                continue
            usable = bool(getattr(window, "usable", False))
            # Only usable windows that contain speech (Phase 4B scope).
            if not usable:
                continue
            items.append(
                _ShadowItem(
                    kind="window",
                    source_role="speech_window",
                    text=speech_text,
                    window_index=int(getattr(window, "index", 0)),
                    start=float(getattr(window, "start", 0.0)),
                    end=float(getattr(window, "end", 0.0)),
                    usable_window=usable,
                    provenance={
                        "role": "speech_window",
                        "window_index": int(getattr(window, "index", 0)),
                    },
                ),
            )

    return items


def _empty_summary() -> WellbeingShadowSummary:
    return WellbeingShadowSummary()


def _summarize(
    source_results: Sequence[WellbeingShadowSourceResult],
    window_results: Sequence[WellbeingShadowWindowResult],
    *,
    items_submitted: int,
    windows_submitted: int,
) -> WellbeingShadowSummary:
    selected_counts: dict[str, int] = {}
    eligible_s = uncertain_s = not_elig_s = 0
    for src in source_results:
        clf = src.classification
        if clf is None:
            continue
        status = clf.eligibility_status
        if status == "eligible":
            eligible_s += 1
        elif status == "uncertain":
            uncertain_s += 1
        else:
            not_elig_s += 1
        for sig in clf.signals:
            if sig.selected:
                selected_counts[sig.signal] = selected_counts.get(sig.signal, 0) + 1

    eligible_w = uncertain_w = not_elig_w = 0
    for win in window_results:
        clf = win.classification
        if clf is None:
            continue
        status = clf.eligibility_status
        if status == "eligible":
            eligible_w += 1
        elif status == "uncertain":
            uncertain_w += 1
        else:
            not_elig_w += 1
        for sig in clf.signals:
            if sig.selected:
                selected_counts[sig.signal] = selected_counts.get(sig.signal, 0) + 1

    return WellbeingShadowSummary(
        evaluated_source_count=len(source_results),
        eligible_source_count=eligible_s,
        uncertain_source_count=uncertain_s,
        not_eligible_source_count=not_elig_s,
        selected_signal_counts=selected_counts,
        evaluated_window_count=len(window_results),
        eligible_window_count=eligible_w,
        uncertain_window_count=uncertain_w,
        not_eligible_window_count=not_elig_w,
        items_submitted=items_submitted,
        windows_submitted=windows_submitted,
    )


def _status_for_results(
    source_results: Sequence[WellbeingShadowSourceResult],
    window_results: Sequence[WellbeingShadowWindowResult],
) -> str:
    statuses: list[str] = []
    for src in source_results:
        if src.classification is not None:
            statuses.append(src.classification.status)
    for win in window_results:
        if win.classification is not None:
            statuses.append(win.classification.status)
    if not statuses:
        return "insufficient_text"
    if all(s == "ok" for s in statuses):
        return "ok"
    if any(s == "ok" for s in statuses):
        return "partial"
    if any(s == "classifier_unavailable" for s in statuses):
        return "classifier_unavailable"
    if any(s == "insufficient_text" for s in statuses) and all(
        s in {"insufficient_text", "error"} for s in statuses
    ):
        return "insufficient_text"
    return "error"


def build_wellbeing_shadow(
    *,
    primary_text: Optional[str] = None,
    caption: Optional[str] = None,
    transcript: Optional[str] = None,
    temporal_context: Any = None,
    classifier: Any = None,
    enabled: Optional[bool] = None,
) -> Optional[WellbeingShadowAnalysis]:
    """Build shadow wellbeing analysis or return None when shadow is disabled.

    Fail-soft: unexpected errors become status=error and never raise to the
    caller (pipeline must still succeed).
    """
    if not resolve_wellbeing_shadow_enabled(override=enabled):
        return None

    t0 = time.perf_counter()
    try:
        items = collect_shadow_items(
            primary_text=primary_text,
            caption=caption,
            transcript=transcript,
            temporal_context=temporal_context,
        )
        if not items:
            return WellbeingShadowAnalysis(
                status="insufficient_text",
                summary=_empty_summary(),
                processing_seconds=round(time.perf_counter() - t0, 4),
                note=(
                    "Shadow mode enabled but no usable authored textual evidence "
                    "(OCR/visual/face evidence excluded). Missing != neutral."
                ),
            )

        if classifier is None:
            # Lazy import / singleton — only when shadow is enabled and texts exist.
            from src.wellbeing.classifier import get_wellbeing_classifier

            classifier = get_wellbeing_classifier()

        texts = [item.text for item in items]
        windows_submitted = sum(1 for item in items if item.kind == "window")
        results: list[WellbeingClassificationResult] = classifier.classify_many(
            texts,
            source_type="text",
        )

        source_results: list[WellbeingShadowSourceResult] = []
        window_results: list[WellbeingShadowWindowResult] = []
        model_id: Optional[str] = getattr(classifier, "model_id", None)

        for item, result in zip(items, results):
            if model_id is None:
                model_id = result.model_id
            if item.kind == "window":
                window_results.append(
                    WellbeingShadowWindowResult(
                        window_index=int(item.window_index or 0),
                        start=float(item.start or 0.0),
                        end=float(item.end or 0.0),
                        classification=result,
                        input_character_count=len(item.text),
                        usable_window=item.usable_window,
                    ),
                )
            else:
                source_results.append(
                    WellbeingShadowSourceResult(
                        source_role=item.source_role,  # type: ignore[arg-type]
                        classification=result,
                        input_character_count=len(item.text),
                        provenance=dict(item.provenance or {}),
                    ),
                )

        summary = _summarize(
            source_results,
            window_results,
            items_submitted=len(items),
            windows_submitted=windows_submitted,
        )
        return WellbeingShadowAnalysis(
            status=_status_for_results(source_results, window_results),  # type: ignore[arg-type]
            model_id=model_id,
            source_results=source_results,
            window_results=window_results,
            summary=summary,
            processing_seconds=round(time.perf_counter() - t0, 4),
        )
    except Exception as exc:  # noqa: BLE001 — fail soft for pipeline isolation
        logger.warning(
            "Wellbeing shadow analysis failed (%s)",
            type(exc).__name__,
        )
        return WellbeingShadowAnalysis(
            status="error",
            summary=_empty_summary(),
            processing_seconds=round(time.perf_counter() - t0, 4),
            error_code=type(exc).__name__,
            note=(
                "Shadow classifier failed; existing sentiment/temporal analysis "
                "is unaffected."
            ),
        )
