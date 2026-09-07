"""Parse and validate TemporalContextReasoner JSON output via Pydantic."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import ValidationError

from src.schemas import TemporalReasoningResult
from src.temporal.providers.openrouter_schema import OPENROUTER_TEMPORAL_REASONING_SCHEMA

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

MODEL_OUTPUT_REQUIRED_FIELDS: tuple[str, ...] = tuple(
    OPENROUTER_TEMPORAL_REASONING_SCHEMA["required"],
)


def extract_json_object(text: str) -> str:
    """Extract the outermost JSON object from model text.

    Compatibility fallback only. Primary path prefers direct ``json.loads``.
    A single markdown fence unwrap is allowed; arbitrary prose scraping is not.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("empty model output")

    fence = _FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    if cleaned.startswith("{"):
        return cleaned

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start : end + 1]
    raise ValueError("no JSON object found in model output")


def _loads_model_json(text: str) -> Any:
    """Prefer direct json.loads; fall back to bounded object extraction."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("empty model output")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        raw = extract_json_object(text)
        return json.loads(raw)


def assert_required_model_fields(data: dict[str, Any]) -> None:
    """Fail when schema-required top-level keys are absent (before defaults fill)."""
    missing = [key for key in MODEL_OUTPUT_REQUIRED_FIELDS if key not in data]
    if missing:
        raise ValueError(f"missing required field(s): {missing}")
    cmc = data.get("cross_modal_context")
    if isinstance(cmc, dict):
        cmc_required = OPENROUTER_TEMPORAL_REASONING_SCHEMA["properties"][
            "cross_modal_context"
        ]["required"]
        missing_cmc = [key for key in cmc_required if key not in cmc]
        if missing_cmc:
            raise ValueError(
                f"missing required cross_modal_context field(s): {missing_cmc}",
            )


def parse_reasoning_result(
    text: str,
    *,
    model_id: Optional[str] = None,
    valid_evidence_ids: Optional[set[str]] = None,
    valid_window_ranges: Optional[list[tuple[float, float]]] = None,
) -> TemporalReasoningResult:
    """Parse model text into TemporalReasoningResult or raise ValidationError/ValueError."""
    data: Any = _loads_model_json(text)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    assert_required_model_fields(data)
    if model_id and "model" not in data:
        data = {**data, "model": model_id}
    if "status" not in data:
        data = {**data, "status": "ok"}
    result = TemporalReasoningResult.model_validate(data)
    if valid_evidence_ids is not None:
        _validate_evidence_ids(result, valid_evidence_ids)
    if valid_window_ranges is not None:
        _validate_transition_timestamps(result, valid_window_ranges)
    return result


def _validate_evidence_ids(
    result: TemporalReasoningResult,
    valid_evidence_ids: set[str],
) -> None:
    bad: list[str] = []
    for item in result.evidence:
        if item.evidence_id not in valid_evidence_ids:
            bad.append(item.evidence_id)
    for transition in result.important_transitions:
        for evidence_id in transition.evidence_ids:
            if evidence_id not in valid_evidence_ids:
                bad.append(evidence_id)
    if bad:
        raise ValueError(f"unknown evidence_id(s): {sorted(dict.fromkeys(bad))}")


def _validate_transition_timestamps(
    result: TemporalReasoningResult,
    valid_window_ranges: list[tuple[float, float]],
) -> None:
    for transition in result.important_transitions:
        if transition.end < transition.start:
            raise ValueError(
                f"invalid transition range: end < start ({transition.start}, {transition.end})",
            )
        inside_any = any(
            float(start) <= float(transition.start) <= float(end)
            and float(start) <= float(transition.end) <= float(end)
            for start, end in valid_window_ranges
        )
        if not inside_any:
            raise ValueError(
                "transition timestamps must fall within a supplied window range",
            )


def format_validation_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return exc.errors().__repr__()[:800]
    return str(exc)[:800]
