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
    """Automatic structural / contract checks for one reasoner run."""

    schema_valid: bool = False
    valid_evidence_ids: bool = False
    deterministic_fact_preservation: bool = False
    conflict_preservation: bool = False
    transition_timestamps_valid: bool = False
    uncertainty_requirement_met: bool = False
    prompt_injection_resisted: bool = False
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

    deterministic_fact_preservation: bool = False
    valid_evidence_ids: bool = False
    transition_timestamps_valid: bool = False
    conflict_preservation: bool = False
    uncertainty_requirement_met: bool = False
    prompt_injection_resisted: bool = False

    context_type: Optional[str] = None
    context_type_expected: Optional[str] = None
    context_type_match: Optional[bool] = None

    generation_seconds: Optional[float] = None
    parse_seconds: Optional[float] = None
    prompt_construction_seconds: Optional[float] = None
    model_load_seconds: Optional[float] = None
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
