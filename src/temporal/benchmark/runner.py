"""Benchmark runner: frozen payloads → reasoner candidates → evaluated results.

Loads each candidate once per session. Unit tests must mock generation and
must not download Qwen3-4B.
"""

from __future__ import annotations

import uuid
from typing import Callable, Optional, Sequence

from src.config import (
    TEMPORAL_REASONER_CANDIDATE_1_7B,
    TEMPORAL_REASONER_CANDIDATE_4B,
    TemporalReasonerConfig,
    evaluation_reasoner_config,
)
from src.temporal.benchmark.evaluate import evaluate_invariants
from src.temporal.benchmark.export import FrozenReasonerPayload
from src.temporal.benchmark.fixtures import (
    BENCHMARK_FIXTURES,
    BenchmarkFixtureSpec,
    get_fixture_spec,
    load_benchmark_payload,
)
from src.temporal.benchmark.schemas import HumanReviewFields, ReasonerBenchmarkResult
from src.temporal.reasoner import TemporalContextReasoner

GenerateFn = Callable[[str, str], str]


DEFAULT_CANDIDATES = (
    TEMPORAL_REASONER_CANDIDATE_1_7B,
    TEMPORAL_REASONER_CANDIDATE_4B,
)


class ReasonerBenchmarkRunner:
    """Run frozen fixtures against one or more reasoner candidates."""

    def __init__(
        self,
        *,
        model_ids: Sequence[str] = DEFAULT_CANDIDATES,
        device: str = "cpu",
        seed: Optional[int] = None,
        generate_override: Optional[GenerateFn] = None,
        skip_missing_real: bool = False,
        forbid_model_ids: Optional[Sequence[str]] = None,
    ) -> None:
        self.model_ids = list(model_ids)
        self.device = device
        self.seed = seed
        self.generate_override = generate_override
        self.skip_missing_real = skip_missing_real
        # Safety: local Phase 3B-A/B must not download 4B unless explicitly allowed.
        self.forbid_model_ids = set(forbid_model_ids or ())

    def build_config(self, model_id: str) -> TemporalReasonerConfig:
        kwargs = {"device": self.device}
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        return evaluation_reasoner_config(model_id, **kwargs)

    def run_fixture(
        self,
        reasoner: TemporalContextReasoner,
        payload: FrozenReasonerPayload,
        *,
        spec: Optional[BenchmarkFixtureSpec] = None,
        run_id: Optional[str] = None,
    ) -> ReasonerBenchmarkResult:
        """Run one already-loaded reasoner on one frozen payload."""
        spec = spec or get_fixture_spec(payload.fixture_id)
        run_id = run_id or str(uuid.uuid4())
        seed = int(reasoner.config.seed)

        if self.generate_override is not None:
            reasoner._generate = self.generate_override  # type: ignore[method-assign]

        result, diagnostics = reasoner.reason(
            payload.temporal_context,
            baseline_overall=payload.baseline_overall,
        )
        schema_valid = result.status == "ok"
        parse_error = None
        if result.status == "invalid_model_output" and result.details:
            parse_error = str(result.details.get("error") or "")

        checks = evaluate_invariants(
            payload=payload,
            result=result,
            spec=spec,
            schema_valid=schema_valid,
            parse_error=parse_error,
        )

        return ReasonerBenchmarkResult(
            model_id=reasoner.model_id,
            fixture_id=payload.fixture_id,
            run_id=run_id,
            seed=seed,
            status=result.status,
            schema_valid=checks.schema_valid,
            repair_attempted=bool(diagnostics.repair_attempted),
            deterministic_fact_preservation=checks.deterministic_fact_preservation,
            valid_evidence_ids=checks.valid_evidence_ids,
            transition_timestamps_valid=checks.transition_timestamps_valid,
            conflict_preservation=checks.conflict_preservation,
            uncertainty_requirement_met=checks.uncertainty_requirement_met,
            prompt_injection_resisted=checks.prompt_injection_resisted,
            context_type=result.context_type,
            context_type_expected=spec.expected_context_type,
            context_type_match=checks.context_type_match,
            generation_seconds=diagnostics.generation_seconds,
            parse_seconds=diagnostics.parse_validation_seconds,
            prompt_construction_seconds=diagnostics.prompt_construction_seconds,
            model_load_seconds=diagnostics.model_load_seconds,
            total_seconds=diagnostics.total_reasoner_seconds,
            prompt_tokens=diagnostics.prompt_tokens,
            generated_tokens=diagnostics.generated_tokens,
            raw_output_preview=diagnostics.raw_output_preview,
            invariant_notes=list(checks.notes),
            unsupported_claim_flags=list(checks.unsupported_claim_flags),
            human_review=HumanReviewFields(),
            details={
                "prompt_chars": diagnostics.prompt_chars,
                "evidence_ids_supplied": diagnostics.evidence_ids_supplied,
                "generation_kwargs": diagnostics.generation_kwargs,
                "sampling_warning_detected": diagnostics.sampling_warning_detected,
            },
        )

    def run_model_on_payloads(
        self,
        model_id: str,
        payloads: Sequence[FrozenReasonerPayload],
        *,
        specs: Optional[dict[str, BenchmarkFixtureSpec]] = None,
    ) -> list[ReasonerBenchmarkResult]:
        if model_id in self.forbid_model_ids:
            raise RuntimeError(
                f"Model {model_id} is forbidden in this session "
                "(do not download/run Qwen3-4B locally without allow flag).",
            )
        cfg = self.build_config(model_id)
        reasoner = TemporalContextReasoner(cfg)
        # When generate_override is set, never call load().
        if self.generate_override is None:
            reasoner.load()
        results: list[ReasonerBenchmarkResult] = []
        try:
            for payload in payloads:
                spec = (specs or {}).get(payload.fixture_id)
                if spec is None:
                    try:
                        spec = get_fixture_spec(payload.fixture_id)
                    except KeyError:
                        spec = BenchmarkFixtureSpec(
                            fixture_id=payload.fixture_id,
                            description=payload.source,
                        )
                results.append(self.run_fixture(reasoner, payload, spec=spec))
        finally:
            if self.generate_override is None:
                reasoner.unload()
        return results

    def run_all(
        self,
        *,
        fixture_ids: Optional[Sequence[str]] = None,
    ) -> tuple[list[ReasonerBenchmarkResult], dict[str, FrozenReasonerPayload]]:
        """Load fixtures once, then run each model over the same payloads."""
        wanted = set(fixture_ids) if fixture_ids is not None else None
        payloads: list[FrozenReasonerPayload] = []
        payload_map: dict[str, FrozenReasonerPayload] = {}
        specs: dict[str, BenchmarkFixtureSpec] = {}
        for spec in BENCHMARK_FIXTURES:
            if wanted is not None and spec.fixture_id not in wanted:
                continue
            try:
                payload = load_benchmark_payload(spec)
            except FileNotFoundError:
                if self.skip_missing_real and spec.frozen_path is not None:
                    continue
                raise
            payloads.append(payload)
            payload_map[payload.fixture_id] = payload
            specs[spec.fixture_id] = spec

        all_results: list[ReasonerBenchmarkResult] = []
        for model_id in self.model_ids:
            try:
                all_results.extend(
                    self.run_model_on_payloads(model_id, payloads, specs=specs),
                )
            except Exception as exc:  # noqa: BLE001
                # Preserve prior model results; mark this candidate unavailable.
                all_results.append(
                    ReasonerBenchmarkResult(
                        model_id=model_id,
                        fixture_id="__session__",
                        run_id=str(uuid.uuid4()),
                        seed=int(self.seed if self.seed is not None else 42),
                        status="model_unavailable",
                        details={"error": str(exc)},
                    ),
                )
        return all_results, payload_map
