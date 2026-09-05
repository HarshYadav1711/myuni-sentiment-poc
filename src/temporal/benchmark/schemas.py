"""Typed schemas for the temporal reasoner benchmark harness."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


HumanReviewGrade = Literal["PASS", "PARTIAL", "FAIL", ""]


class HumanReviewFields(BaseModel):
    """Blank fields for later manual rating — never pre-filled by a model."""

    context_quality: HumanReviewGrade = ""
    summary_grounding: HumanReviewGrade = ""
    useful_uncertainty: HumanReviewGrade = ""
    reviewer_notes: str = ""


class InvariantCheckResult(BaseModel):
    """Automatic structural / contract checks for one reasoner run.

    Optional bools use ``None`` for not-applicable (e.g. no model output,
    or the fixture does not exercise that check).
    """

    schema_valid: bool = False
    valid_evidence_ids: Optional[bool] = None
    deterministic_fact_preservation: Optional[bool] = None
    conflict_preservation: Optional[bool] = None
    transition_timestamps_valid: Optional[bool] = None
    uncertainty_requirement_met: Optional[bool] = None
    prompt_injection_resisted: Optional[bool] = None
    context_type_match: Optional[bool] = None
    unsupported_claim_flags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ReasonerBenchmarkResult(BaseModel):
    """One model × fixture run result for Phase 3B-A comparison."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    fixture_id: str
    run_id: str
    seed: int

    status: str = "unknown"
    schema_valid: bool = False
    repair_attempted: bool = False

    # None = not applicable (e.g. reasoner_unavailable / check not exercised).
    deterministic_fact_preservation: Optional[bool] = None
    valid_evidence_ids: Optional[bool] = None
    transition_timestamps_valid: Optional[bool] = None
    conflict_preservation: Optional[bool] = None
    uncertainty_requirement_met: Optional[bool] = None
    prompt_injection_resisted: Optional[bool] = None

    context_type: Optional[str] = None
    context_type_expected: Optional[str] = None
    context_type_match: Optional[bool] = None

    generation_seconds: Optional[float] = None
    parse_seconds: Optional[float] = None
    prompt_construction_seconds: Optional[float] = None
    # Per-reason() load() call timing (often a no-op cache check after session load).
    model_load_seconds: Optional[float] = None
    # Once-per-candidate tokenizer+model+device prepare time (session load).
    candidate_model_prepare_seconds: Optional[float] = Field(
        default=None,
        description=(
            "Wall seconds for the candidate session load "
            "(tokenizer + weights + device placement). Measured once per model."
        ),
    )
    repair_generation_seconds: Optional[float] = None
    total_seconds: Optional[float] = None

    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None
    peak_gpu_memory_mb: Optional[float] = None

    raw_output_preview: Optional[str] = None
    invariant_notes: list[str] = Field(default_factory=list)
    unsupported_claim_flags: list[str] = Field(default_factory=list)
    human_review: HumanReviewFields = Field(default_factory=HumanReviewFields)
    details: Optional[dict[str, Any]] = None
