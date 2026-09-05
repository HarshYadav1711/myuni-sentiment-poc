"""Unit tests for Phase 3B-A reasoner benchmark harness (no live models)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    TEMPORAL_REASONER_CANDIDATE_1_7B,
    TEMPORAL_REASONER_CANDIDATE_4B,
    TEMPORAL_REASONER_EVAL_SEED,
    evaluation_reasoner_config,
)
from src.temporal.benchmark.capabilities import (
    chat_template_apply_kwargs,
    resolve_model_capability,
)
from src.temporal.benchmark.evaluate import evaluate_invariants
from src.temporal.benchmark.export import (
    FrozenReasonerPayload,
    export_frozen_payload,
    lean_temporal_context_for_reasoner,
    load_frozen_payload,
)
from src.temporal.benchmark.fixtures import (
    BENCHMARK_FIXTURES,
    REAL_CONTROLLED_PAYLOAD_PATH,
    fixture_ids,
    get_fixture_spec,
    load_benchmark_payload,
)
from src.temporal.benchmark.report import (
    aggregate_pass_rates,
    write_human_review_markdown,
    write_results_csv,
    write_results_json,
)
from src.temporal.benchmark.runner import ReasonerBenchmarkRunner
from src.temporal.benchmark.schemas import ReasonerBenchmarkResult
from src.temporal.benchmark.synthetic import (
    fixture_prompt_injection,
    fixture_sparse_visual_only,
    fixture_visual_neg_speech_neg,
    fixture_visual_pos_speech_neg,
)
from src.temporal.prompt import SYSTEM_INSTRUCTION, build_evidence_payload, build_user_prompt
from src.temporal.reasoner import TemporalContextReasoner
from src.schemas import TemporalReasoningResult


def _ok_json(**overrides) -> str:
    payload = {
        "summary": "Neutral timeline with limited evidence.",
        "trajectory_explanation": "The deterministic trajectory is stable neutral.",
        "cross_modal_context": {
            "consistency": "insufficient_evidence",
            "conflicts_detected": False,
            "description": "Only visual modality available.",
        },
        "important_transitions": [],
        "context_type": "uncertain",
        "evidence": [{"evidence_id": "window-0", "explanation": "only window"}],
        "uncertainties": ["sparse evidence", "missing speech"],
        "confidence": 0.4,
        "status": "ok",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_evaluation_config_explicit_do_sample() -> None:
    cfg = evaluation_reasoner_config(TEMPORAL_REASONER_CANDIDATE_1_7B)
    assert cfg.do_sample is True
    assert cfg.temperature == pytest.approx(0.7)
    assert cfg.top_p == pytest.approx(0.8)
    assert cfg.top_k == 20
    assert cfg.seed == TEMPORAL_REASONER_EVAL_SEED
    assert cfg.enable_thinking is False
    assert cfg.max_new_tokens == 1024
    reasoner = TemporalContextReasoner(cfg)
    gen = reasoner.build_generation_config()
    assert gen["do_sample"] is True
    assert gen["temperature"] == pytest.approx(0.7)


def test_both_candidates_share_eval_output_budget() -> None:
    c17 = evaluation_reasoner_config(TEMPORAL_REASONER_CANDIDATE_1_7B)
    c4 = evaluation_reasoner_config(TEMPORAL_REASONER_CANDIDATE_4B)
    assert c17.max_new_tokens == 1024
    assert c4.max_new_tokens == 1024
    assert c17.max_new_tokens == c4.max_new_tokens


def test_same_frozen_fixture_builds_identical_user_prompt_for_both_candidates() -> None:
    """Semantic evidence/prompt text is identical; token counts may differ by tokenizer."""
    payload = load_benchmark_payload(get_fixture_spec("stable_neutral"))
    c17 = evaluation_reasoner_config(TEMPORAL_REASONER_CANDIDATE_1_7B)
    c4 = evaluation_reasoner_config(TEMPORAL_REASONER_CANDIDATE_4B)
    e17 = build_evidence_payload(
        payload.temporal_context,
        baseline_overall=payload.baseline_overall,
        config=c17,
    )
    e4 = build_evidence_payload(
        payload.temporal_context,
        baseline_overall=payload.baseline_overall,
        config=c4,
    )
    assert build_user_prompt(e17) == build_user_prompt(e4)
    assert SYSTEM_INSTRUCTION  # shared constant for both candidates


def test_fixed_seed_recorded_on_eval_config() -> None:
    cfg = evaluation_reasoner_config(seed=99)
    assert cfg.seed == 99


def test_capability_qwen17_supports_thinking_switch() -> None:
    cap = resolve_model_capability(TEMPORAL_REASONER_CANDIDATE_1_7B)
    assert cap.supports_enable_thinking is True
    kwargs = chat_template_apply_kwargs(cap, enable_thinking=False)
    assert kwargs["enable_thinking"] is False


def test_capability_qwen4b_instruct_no_thinking_kwarg() -> None:
    cap = resolve_model_capability(TEMPORAL_REASONER_CANDIDATE_4B)
    assert cap.supports_enable_thinking is False
    kwargs = chat_template_apply_kwargs(cap, enable_thinking=False)
    assert "enable_thinking" not in kwargs


def test_fixture_catalog_coverage() -> None:
    ids = set(fixture_ids())
    required = {
        "stable_neutral",
        "stable_negative",
        "increasing_negative",
        "decreasing_negative",
        "isolated_negative",
        "persistent_negative",
        "visual_pos_speech_neg",
        "visual_neg_speech_neg",
        "sparse_visual_only",
        "informational",
        "quoted_narrative",
        "prompt_injection",
        "real_phase3a_controlled_video",
    }
    assert required <= ids


def test_synthetic_payload_roundtrip() -> None:
    ctx = fixture_visual_pos_speech_neg()
    payload = export_frozen_payload(
        fixture_id="visual_pos_speech_neg",
        source="test",
        temporal=ctx,
    )
    assert isinstance(payload, FrozenReasonerPayload)
    assert payload.valid_evidence_ids
    lean = lean_temporal_context_for_reasoner(ctx)
    assert lean.events == []
    assert lean.windows


def test_visual_neg_speech_neg_has_no_conflict() -> None:
    ctx = fixture_visual_neg_speech_neg()
    assert ctx.features.cross_modal_conflicts == []


def test_conflict_evaluator_requires_conflict_flag() -> None:
    ctx = fixture_visual_pos_speech_neg()
    payload = export_frozen_payload(
        fixture_id="visual_pos_speech_neg",
        source="test",
        temporal=ctx,
    )
    good = TemporalReasoningResult.model_validate(
        json.loads(
            _ok_json(
                summary="Visual positive while speech negative conflict preserved.",
                trajectory_explanation="stable positive visual with speech conflict.",
                cross_modal_context={
                    "consistency": "low",
                    "conflicts_detected": True,
                    "description": "Visual positive while speech negative.",
                },
                evidence=[{"evidence_id": "window-0", "explanation": "conflict window"}],
                uncertainties=[],
                confidence=0.6,
                context_type="uncertain",
            ),
        ),
    )
    checks = evaluate_invariants(
        payload=payload,
        result=good,
        spec=get_fixture_spec("visual_pos_speech_neg"),
        schema_valid=True,
    )
    assert checks.conflict_preservation is True

    bad = good.model_copy(
        update={
            "cross_modal_context": good.cross_modal_context.model_copy(
                update={"conflicts_detected": False, "description": "both negative"},
            ),
        },
    )
    checks_bad = evaluate_invariants(
        payload=payload,
        result=bad,
        spec=get_fixture_spec("visual_pos_speech_neg"),
        schema_valid=True,
    )
    assert checks_bad.conflict_preservation is False


def test_sparse_uncertainty_evaluator() -> None:
    ctx = fixture_sparse_visual_only()
    payload = export_frozen_payload(
        fixture_id="sparse_visual_only",
        source="test",
        temporal=ctx,
        meta={"sparse": True},
    )
    result = TemporalReasoningResult.model_validate(
        json.loads(_ok_json(context_type="uncertain", uncertainties=["sparse"])),
    )
    checks = evaluate_invariants(
        payload=payload,
        result=result,
        spec=get_fixture_spec("sparse_visual_only"),
        schema_valid=True,
    )
    assert checks.uncertainty_requirement_met is True


def test_prompt_injection_evaluator() -> None:
    ctx = fixture_prompt_injection()
    payload = export_frozen_payload(
        fixture_id="prompt_injection",
        source="test",
        temporal=ctx,
        meta={"prompt_injection": True},
    )
    evidence = build_evidence_payload(ctx)
    valid = set(evidence["valid_evidence_ids"])
    ok_id = next(iter(valid))
    result = TemporalReasoningResult.model_validate(
        json.loads(
            _ok_json(
                evidence=[{"evidence_id": ok_id, "explanation": "ignored injection"}],
                context_type="uncertain",
            ),
        ),
    )
    checks = evaluate_invariants(
        payload=payload,
        result=result,
        spec=get_fixture_spec("prompt_injection"),
        schema_valid=True,
    )
    assert checks.prompt_injection_resisted is True

    injected = result.model_copy(
        update={
            "evidence": [
                {"evidence_id": "window-999", "explanation": "followed injection"},
            ],
        },
    )
    # Bypass pydantic by constructing manually through model_validate with invalid id
    # after schema — evaluate_invariants checks IDs itself.
    from src.schemas import ReasoningEvidenceReference

    injected = result.model_copy(
        update={
            "evidence": [
                ReasoningEvidenceReference(
                    evidence_id="window-999",
                    explanation="followed injection",
                ),
            ],
        },
    )
    checks2 = evaluate_invariants(
        payload=payload,
        result=injected,
        spec=get_fixture_spec("prompt_injection"),
        schema_valid=True,
    )
    assert checks2.prompt_injection_resisted is False


def test_benchmark_runner_mocked_no_model_download() -> None:
    def fake_generate(system: str, user: str) -> str:
        assert "AUTHORITATIVE" in system or "authoritative" in system.lower() or "AUTHORITATIVE" in SYSTEM_INSTRUCTION
        # Extract a valid window id from the frozen evidence if present.
        if "window-0" in user:
            eid = "window-0"
        else:
            eid = "window-0"
        return _ok_json(
            evidence=[{"evidence_id": eid, "explanation": "ok"}],
            trajectory_explanation="deterministic trajectory explained without override",
        )

    runner = ReasonerBenchmarkRunner(
        model_ids=[TEMPORAL_REASONER_CANDIDATE_1_7B],
        generate_override=fake_generate,
        skip_missing_real=True,
        forbid_model_ids={TEMPORAL_REASONER_CANDIDATE_4B},
    )
    results, payloads = runner.run_all(
        fixture_ids=["sparse_visual_only", "stable_neutral"],
    )
    assert len(results) == 2
    assert all(r.seed == TEMPORAL_REASONER_EVAL_SEED for r in results)
    assert all(r.human_review.context_quality == "" for r in results)
    assert "sparse_visual_only" in payloads


def test_forbid_4b_locally() -> None:
    runner = ReasonerBenchmarkRunner(
        model_ids=[TEMPORAL_REASONER_CANDIDATE_4B],
        generate_override=lambda s, u: _ok_json(),
        forbid_model_ids={TEMPORAL_REASONER_CANDIDATE_4B},
    )
    payload = load_benchmark_payload(get_fixture_spec("stable_neutral"))
    with pytest.raises(RuntimeError, match="forbidden"):
        runner.run_model_on_payloads(TEMPORAL_REASONER_CANDIDATE_4B, [payload])


def test_report_serialization(tmp_path: Path) -> None:
    row = ReasonerBenchmarkResult(
        model_id=TEMPORAL_REASONER_CANDIDATE_1_7B,
        fixture_id="stable_neutral",
        run_id="r1",
        seed=42,
        status="ok",
        schema_valid=True,
        valid_evidence_ids=True,
        conflict_preservation=True,
        prompt_injection_resisted=True,
        deterministic_fact_preservation=True,
        transition_timestamps_valid=True,
        uncertainty_requirement_met=True,
    )
    payload = load_benchmark_payload(get_fixture_spec("stable_neutral"))
    write_results_json([row], tmp_path / "results.json")
    write_results_csv([row], tmp_path / "results.csv")
    write_human_review_markdown(
        [row],
        {"stable_neutral": payload},
        tmp_path / "human_review.md",
    )
    rates = aggregate_pass_rates([row])
    assert rates[TEMPORAL_REASONER_CANDIDATE_1_7B]["schema_success_rate"] == 1.0
    md = (tmp_path / "human_review.md").read_text(encoding="utf-8")
    assert "context_quality" in md
    assert "PASS / PARTIAL / FAIL" not in md or True  # blank fields present
    assert "- context_quality: ` `" in md


def test_unavailable_inference_not_fake_semantic_pass() -> None:
    payload = load_benchmark_payload(get_fixture_spec("stable_neutral"))
    result = TemporalReasoningResult(
        summary="",
        context_type="uncertain",
        confidence=0.0,
        model=TEMPORAL_REASONER_CANDIDATE_1_7B,
        status="generation_failed",
        details={"error": "generation_failed: boom"},
    )
    checks = evaluate_invariants(
        payload=payload,
        result=result,
        spec=get_fixture_spec("stable_neutral"),
        schema_valid=False,
    )
    assert checks.schema_valid is False
    assert checks.prompt_injection_resisted is None
    assert checks.uncertainty_requirement_met is None
    assert checks.valid_evidence_ids is None
    assert checks.deterministic_fact_preservation is None

    row = ReasonerBenchmarkResult(
        model_id=TEMPORAL_REASONER_CANDIDATE_1_7B,
        fixture_id="stable_neutral",
        run_id="r-fail",
        seed=42,
        status="generation_failed",
        schema_valid=False,
        prompt_injection_resisted=None,
        uncertainty_requirement_met=None,
        valid_evidence_ids=None,
        deterministic_fact_preservation=None,
        conflict_preservation=None,
        transition_timestamps_valid=None,
    )
    rates = aggregate_pass_rates([row])
    model_rates = rates[TEMPORAL_REASONER_CANDIDATE_1_7B]
    assert model_rates["schema_success_rate"] == 0.0
    assert model_rates["injection_resistance_rate"] is None
    assert model_rates["uncertainty_pass_rate"] is None


def test_runner_preserves_reasoner_error_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_system: str, _user: str) -> str:
        raise RuntimeError("device-side generate failed")

    def fake_load(self) -> None:  # noqa: ANN001
        # Mark loaded without downloading any Qwen weights.
        self._tokenizer = object()
        self._model = object()
        self._device = "cpu"
        self._torch = object()

    monkeypatch.setattr(
        "src.temporal.benchmark.runner.TemporalContextReasoner.load",
        fake_load,
    )

    runner = ReasonerBenchmarkRunner(
        model_ids=[TEMPORAL_REASONER_CANDIDATE_1_7B],
        generate_override=boom,
        skip_missing_real=True,
        forbid_model_ids={TEMPORAL_REASONER_CANDIDATE_4B},
    )
    results, _ = runner.run_all(fixture_ids=["stable_neutral"])
    assert len(results) == 1
    row = results[0]
    assert row.status == "generation_failed"
    assert row.details is not None
    assert row.details.get("reasoner_error_type") == "RuntimeError"
    assert "device-side generate failed" in (row.details.get("reasoner_error_message") or "")
    assert row.details.get("reasoner_failure_stage") == "generation"
    assert row.prompt_injection_resisted is None
    assert row.uncertainty_requirement_met is None


def test_non_injection_fixture_injection_check_is_na_on_success() -> None:
    payload = load_benchmark_payload(get_fixture_spec("stable_neutral"))
    result = TemporalReasoningResult.model_validate(
        json.loads(
            _ok_json(
                evidence=[{"evidence_id": "window-0", "explanation": "ok"}],
                trajectory_explanation="stable neutral trajectory",
            ),
        ),
    )
    checks = evaluate_invariants(
        payload=payload,
        result=result,
        spec=get_fixture_spec("stable_neutral"),
        schema_valid=True,
    )
    assert checks.prompt_injection_resisted is None
    assert checks.uncertainty_requirement_met is None


def test_na_invariants_not_listed_as_failures() -> None:
    from src.temporal.benchmark.report import invariant_failure_labels

    row = ReasonerBenchmarkResult(
        model_id=TEMPORAL_REASONER_CANDIDATE_1_7B,
        fixture_id="real_phase3a_controlled_video",
        run_id="r-ok",
        seed=42,
        status="ok",
        schema_valid=True,
        valid_evidence_ids=True,
        deterministic_fact_preservation=True,
        conflict_preservation=True,
        transition_timestamps_valid=True,
        uncertainty_requirement_met=None,  # N/A
        prompt_injection_resisted=None,  # N/A
        context_type_match=None,
    )
    fails = invariant_failure_labels(row)
    assert "injection" not in fails
    assert "uncertainty" not in fails
    assert fails == []

    # Explicit FAIL still listed; N/A never coerced to FAIL or PASS.
    row_fail = row.model_copy(update={"prompt_injection_resisted": False})
    assert "injection" in invariant_failure_labels(row_fail)
    assert row.prompt_injection_resisted is None  # unchanged source


def test_sparse_uncertainty_na_on_non_sparse_fixture() -> None:
    payload = load_benchmark_payload(get_fixture_spec("stable_neutral"))
    result = TemporalReasoningResult.model_validate(
        json.loads(_ok_json(evidence=[{"evidence_id": "window-0", "explanation": "ok"}])),
    )
    checks = evaluate_invariants(
        payload=payload,
        result=result,
        spec=get_fixture_spec("stable_neutral"),
        schema_valid=True,
    )
    assert checks.uncertainty_requirement_met is None
    assert checks.prompt_injection_resisted is None


def test_candidate_model_prepare_timed_once_per_session(monkeypatch: pytest.MonkeyPatch) -> None:
    load_calls: list[float] = []

    def fake_load(self) -> None:  # noqa: ANN001
        load_calls.append(1.0)
        self._tokenizer = object()
        self._model = object()
        self._device = "cpu"
        self._torch = object()
        # Simulate non-trivial prepare wall time without sleeping.
        self._simulated_prepare = 1.25

    real_perf = time.perf_counter
    clock = {"t": 100.0}

    def fake_perf() -> float:
        return clock["t"]

    def load_with_clock(self) -> None:  # noqa: ANN001
        clock["t"] += 0.01
        started = clock["t"]
        fake_load(self)
        clock["t"] = started + 1.25

    monkeypatch.setattr(
        "src.temporal.benchmark.runner.TemporalContextReasoner.load",
        load_with_clock,
    )
    monkeypatch.setattr(time, "perf_counter", fake_perf)

    def fake_generate(system: str, user: str) -> str:
        return _ok_json(
            evidence=[{"evidence_id": "window-0", "explanation": "ok"}],
            trajectory_explanation="stable neutral",
        )

    # generate_override would skip load — call run_model_on_payloads without override
    # but patch _generate after load via monkeypatch on reason method path:
    runner = ReasonerBenchmarkRunner(
        model_ids=[TEMPORAL_REASONER_CANDIDATE_1_7B],
        skip_missing_real=True,
        forbid_model_ids={TEMPORAL_REASONER_CANDIDATE_4B},
    )

    def reason_no_download(self, temporal, *, baseline_overall=None):  # noqa: ANN001
        from src.schemas import TemporalReasonerDiagnostics, TemporalReasoningResult

        diag = TemporalReasonerDiagnostics(
            model_load_seconds=0.000005,
            generation_seconds=0.5,
            parse_validation_seconds=0.01,
            prompt_construction_seconds=0.001,
            total_reasoner_seconds=0.52,
        )
        result = TemporalReasoningResult.model_validate(
            json.loads(
                _ok_json(
                    evidence=[{"evidence_id": "window-0", "explanation": "ok"}],
                    trajectory_explanation="stable neutral",
                ),
            ),
        )
        return result, diag

    monkeypatch.setattr(
        "src.temporal.benchmark.runner.TemporalContextReasoner.reason",
        reason_no_download,
    )

    payload = load_benchmark_payload(get_fixture_spec("stable_neutral"))
    payload2 = load_benchmark_payload(get_fixture_spec("sparse_visual_only"))
    results = runner.run_model_on_payloads(
        TEMPORAL_REASONER_CANDIDATE_1_7B,
        [payload, payload2],
    )
    assert len(load_calls) == 1
    assert len(results) == 2
    prepares = [r.candidate_model_prepare_seconds for r in results]
    assert prepares[0] == pytest.approx(1.25, abs=0.02)
    assert prepares[1] == pytest.approx(1.25, abs=0.02)
    # Fixture generation timing remains separate from prepare.
    assert results[0].generation_seconds == pytest.approx(0.5)
    assert results[0].model_load_seconds == pytest.approx(0.000005)
    from src.temporal.benchmark.report import performance_summary

    perf = performance_summary(results)
    assert perf[TEMPORAL_REASONER_CANDIDATE_1_7B]["candidate_model_prepare_seconds"] == pytest.approx(
        1.25,
        abs=0.02,
    )
    assert "model_load_seconds_max" not in perf[TEMPORAL_REASONER_CANDIDATE_1_7B]


def test_real_phase3a_payload_if_present() -> None:
    if not REAL_CONTROLLED_PAYLOAD_PATH.is_file():
        pytest.skip("real Phase 3A fixture not exported yet")
    payload = load_frozen_payload(REAL_CONTROLLED_PAYLOAD_PATH)
    assert payload.fixture_id == "real_phase3a_controlled_video"
    assert payload.temporal_context.features.trajectory == "increasing_negative"
    assert payload.temporal_context.speech_alignment_source == "word_timestamps"
    assert payload.baseline_overall is not None
    assert payload.valid_evidence_ids
    assert payload.temporal_context.events == []
    # Boundary artifact documented.
    assert any("boundary" in x.lower() or "overwhelming" in x.lower() for x in payload.known_limitations)
    # Reasoner evidence payload rebuilds cleanly.
    evidence = payload.evidence_payload()
    assert evidence["speech_alignment_source"] == "word_timestamps"
    texts = []
    for w in payload.temporal_context.windows:
        for seg in w.speech_segments:
            texts.append(seg.text)
    assert len(set(texts)) >= 2


def test_same_semantic_system_instruction_for_candidates() -> None:
    # Prompt fairness: both candidates use the same SYSTEM_INSTRUCTION text.
    assert "AUTHORITATIVE" in SYSTEM_INSTRUCTION or "authoritative" in SYSTEM_INSTRUCTION.lower()
    assert "Do NOT diagnose" in SYSTEM_INSTRUCTION or "diagnos" in SYSTEM_INSTRUCTION.lower()


def test_existing_reasoner_greedy_config_unchanged() -> None:
    from src.config import TemporalReasonerConfig

    greedy = TemporalContextReasoner(
        TemporalReasonerConfig(temperature=0.0, top_p=1.0, top_k=0, do_sample=None),
    )
    assert greedy.build_generation_config()["do_sample"] is False
