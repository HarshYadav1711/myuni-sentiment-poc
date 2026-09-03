"""Freeze deterministic reasoner inputs for fair model comparison.

Model comparison begins AFTER TemporalContext exists. Never re-run
video/SigLIP/Whisper/RoBERTa when comparing reasoners.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from src.schemas import SentimentEvidence, TemporalContext
from src.temporal.prompt import build_evidence_payload, collect_valid_evidence_ids

PathLike = Union[str, Path]


class FrozenReasonerPayload(BaseModel):
    """Canonical frozen reasoner input (no media bytes / model objects)."""

    fixture_id: str
    source: str = Field(
        description="Origin of the payload, e.g. synthetic fixture or controlled video export.",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    temporal_context: TemporalContext
    baseline_overall: Optional[SentimentEvidence] = None
    # Precomputed for convenience / offline inspection (also rebuildable).
    valid_evidence_ids: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    def evidence_payload(self, *, max_windows: int = 12, max_evidence_items: int = 16) -> dict[str, Any]:
        from src.config import TemporalReasonerConfig

        cfg = TemporalReasonerConfig(
            max_windows=max_windows,
            max_evidence_items=max_evidence_items,
        )
        return build_evidence_payload(
            self.temporal_context,
            baseline_overall=self.baseline_overall,
            config=cfg,
        )


def lean_temporal_context_for_reasoner(temporal: TemporalContext) -> TemporalContext:
    """Keep fields the reasoner/prompt need; drop bulky frame event lists.

    Preserves windows, features, duration, alignment metadata, and original
    Whisper top-level segments. Clears per-frame events and raw word lists
    (window-local speech text already lives on windows).
    """
    return temporal.model_copy(
        update={
            "events": [],
            "events_total": 0,
            "events_truncated": False,
            "speech_words": [],
            # Keep speech_word_count as a coverage diagnostic.
        },
    )


def export_frozen_payload(
    *,
    fixture_id: str,
    source: str,
    temporal: TemporalContext,
    baseline_overall: Optional[SentimentEvidence] = None,
    known_limitations: Optional[list[str]] = None,
    notes: Optional[list[str]] = None,
    meta: Optional[dict[str, Any]] = None,
    lean: bool = True,
) -> FrozenReasonerPayload:
    """Build a typed frozen payload and populate valid evidence IDs."""
    ctx = lean_temporal_context_for_reasoner(temporal) if lean else temporal
    payload = FrozenReasonerPayload(
        fixture_id=fixture_id,
        source=source,
        temporal_context=ctx,
        baseline_overall=baseline_overall,
        known_limitations=list(known_limitations or []),
        notes=list(notes or []),
        meta=dict(meta or {}),
    )
    evidence = payload.evidence_payload()
    payload.valid_evidence_ids = collect_valid_evidence_ids(evidence)
    # Round-trip validate through Pydantic.
    return FrozenReasonerPayload.model_validate(payload.model_dump(mode="json"))


def save_frozen_payload(payload: FrozenReasonerPayload, path: PathLike) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def load_frozen_payload(path: PathLike) -> FrozenReasonerPayload:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return FrozenReasonerPayload.model_validate(data)


def export_from_activity_analysis_dict(
    analysis: dict[str, Any],
    *,
    fixture_id: str,
    source: str,
    known_limitations: Optional[list[str]] = None,
    notes: Optional[list[str]] = None,
) -> FrozenReasonerPayload:
    """Export from a serialized ActivityAnalysisResult.analysis block."""
    temporal_raw = analysis.get("temporal_context")
    if temporal_raw is None and analysis.get("deterministic_context"):
        temporal_raw = analysis["deterministic_context"].get("context")
    if temporal_raw is None:
        raise ValueError("analysis dict missing temporal_context")
    temporal = TemporalContext.model_validate(temporal_raw)
    baseline = None
    if analysis.get("overall") is not None:
        baseline = SentimentEvidence.model_validate(analysis["overall"])
    return export_frozen_payload(
        fixture_id=fixture_id,
        source=source,
        temporal=temporal,
        baseline_overall=baseline,
        known_limitations=known_limitations,
        notes=notes,
        meta={
            "speech_alignment_source": temporal.speech_alignment_source,
            "speech_word_count": temporal.speech_word_count,
            "trajectory": temporal.features.trajectory,
        },
    )
