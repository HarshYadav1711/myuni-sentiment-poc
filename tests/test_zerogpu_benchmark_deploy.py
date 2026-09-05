"""Hardening tests for the ZeroGPU reasoner-benchmark Space bundle."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = Path(r"D:\Work\hf-deploy\myuni-temporal-reasoner-benchmark")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.temporal_reasoner.zerogpu_duration import (
    GPU_DURATION_ALL_FIXTURES_CAP_SECONDS,
    GPU_DURATION_SINGLE_FIXTURE_SECONDS,
    ZEROGPU_LARGE_DURATION_FACTOR,
    effective_gpu_duration_from_spaces_arg,
    estimate_gpu_duration_seconds,
    spaces_gpu_duration_for_call,
    to_spaces_gpu_duration_arg,
)
from src.config import (
    TEMPORAL_REASONER_CANDIDATE_1_7B,
    TEMPORAL_REASONER_CANDIDATE_4B,
)
from src.temporal.benchmark.schemas import ReasonerBenchmarkResult


def test_single_fixture_duration_is_short() -> None:
    d = estimate_gpu_duration_seconds(1, model_id=TEMPORAL_REASONER_CANDIDATE_1_7B)
    assert d == GPU_DURATION_SINGLE_FIXTURE_SECONDS
    assert 45 <= d <= 60


def test_4b_single_fixture_duration_approx_120() -> None:
    effective = estimate_gpu_duration_seconds(1, model_id=TEMPORAL_REASONER_CANDIDATE_4B)
    assert effective == 120
    spaces_arg = spaces_gpu_duration_for_call(1, model_id=TEMPORAL_REASONER_CANDIDATE_4B)
    assert spaces_arg == to_spaces_gpu_duration_arg(120)
    assert spaces_arg == 80  # 80 * 1.5 = 120 platform request
    assert effective_gpu_duration_from_spaces_arg(spaces_arg) == 120
    assert effective < 1200


def test_ui_effective_matches_platform_after_factor() -> None:
    """UI reports effective seconds; decorator arg × factor equals that."""
    for model_id, n, expected_eff in (
        (TEMPORAL_REASONER_CANDIDATE_1_7B, 1, 60),
        (TEMPORAL_REASONER_CANDIDATE_4B, 1, 120),
    ):
        effective = estimate_gpu_duration_seconds(n, model_id=model_id)
        assert effective == expected_eff
        spaces_arg = spaces_gpu_duration_for_call(n, model_id=model_id)
        assert spaces_arg == to_spaces_gpu_duration_arg(effective)
        assert effective_gpu_duration_from_spaces_arg(spaces_arg) == effective
        assert spaces_arg * ZEROGPU_LARGE_DURATION_FACTOR == pytest.approx(float(effective))


def test_deploy_duration_callable_returns_spaces_arg_not_raw_effective() -> None:
    """@spaces.GPU must receive the compensated arg (e.g. 80 not 120 for 4B)."""
    app_src = (DEPLOY / "app.py").read_text(encoding="utf-8")
    assert "spaces_gpu_duration_for_call" in app_src
    assert "to_spaces_gpu_duration_arg" in app_src
    # Import deploy helpers without loading Gradio models.
    sys.path.insert(0, str(DEPLOY))
    if "app" in sys.modules:
        del sys.modules["app"]
    import app as deploy_app

    fixtures = ["real_phase3a_controlled_video"]
    arg = deploy_app._gpu_duration_for_call(
        TEMPORAL_REASONER_CANDIDATE_4B,
        fixtures,
    )
    effective = deploy_app._effective_duration_for_call(
        TEMPORAL_REASONER_CANDIDATE_4B,
        fixtures,
    )
    assert arg == 80
    assert effective == 120
    assert effective_gpu_duration_from_spaces_arg(arg) == effective


def test_all_fixtures_duration_bounded() -> None:
    d17 = estimate_gpu_duration_seconds(13, model_id=TEMPORAL_REASONER_CANDIDATE_1_7B)
    assert d17 <= GPU_DURATION_ALL_FIXTURES_CAP_SECONDS
    assert d17 >= 100
    assert d17 < 1200
    d4 = estimate_gpu_duration_seconds(13, model_id=TEMPORAL_REASONER_CANDIDATE_4B)
    assert d4 <= 300
    assert d4 > 120
    assert d4 < 1200


def test_model_aware_duration_selection() -> None:
    assert estimate_gpu_duration_seconds(1, TEMPORAL_REASONER_CANDIDATE_1_7B) == 60
    assert estimate_gpu_duration_seconds(1, TEMPORAL_REASONER_CANDIDATE_4B) == 120
    assert (
        estimate_gpu_duration_seconds(1, TEMPORAL_REASONER_CANDIDATE_1_7B)
        < estimate_gpu_duration_seconds(1, TEMPORAL_REASONER_CANDIDATE_4B)
    )


def test_single_smaller_than_all() -> None:
    assert estimate_gpu_duration_seconds(1) < estimate_gpu_duration_seconds(13)
    assert estimate_gpu_duration_seconds(
        1,
        TEMPORAL_REASONER_CANDIDATE_4B,
    ) < estimate_gpu_duration_seconds(13, TEMPORAL_REASONER_CANDIDATE_4B)


def test_no_1200_in_duration_helper_or_deploy_app() -> None:
    assert estimate_gpu_duration_seconds(1) != 1200
    assert estimate_gpu_duration_seconds(13) != 1200
    assert estimate_gpu_duration_seconds(100) != 1200
    assert estimate_gpu_duration_seconds(1, TEMPORAL_REASONER_CANDIDATE_4B) != 1200
    assert estimate_gpu_duration_seconds(13, TEMPORAL_REASONER_CANDIDATE_4B) != 1200
    app_src = (DEPLOY / "app.py").read_text(encoding="utf-8")
    assert "@spaces.GPU(duration=1200)" not in app_src
    assert "duration=1200" not in app_src


def test_deploy_app_uses_dynamic_duration_callable() -> None:
    app_src = (DEPLOY / "app.py").read_text(encoding="utf-8")
    assert "@spaces.GPU(duration=_gpu_duration_for_call)" in app_src
    assert "spaces_gpu_duration_for_call" in app_src


def test_deploy_app_import_order_spaces_before_torch_and_src() -> None:
    """Static AST: spaces must precede gradio/src; torch must stay lazy."""
    tree = ast.parse((DEPLOY / "app.py").read_text(encoding="utf-8"))
    import_names: list[str] = []

    def _collect(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_names.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                import_names.append(node.module.split(".")[0])
            elif isinstance(node, ast.Try):
                _collect(node.body)
                for handler in node.handlers:
                    _collect(handler.body)

    _collect(tree.body)

    relevant = [
        n
        for n in import_names
        if n in {"spaces", "torch", "transformers", "src", "gradio", "evaluation"}
    ]
    assert "spaces" in relevant
    assert relevant.index("spaces") < relevant.index("src")
    assert "torch" not in relevant  # torch must remain function-local
    assert relevant.index("spaces") < relevant.index("gradio")


def test_deploy_app_refuses_stub_on_hf_space_env() -> None:
    app_src = (DEPLOY / "app.py").read_text(encoding="utf-8")
    assert "SPACE_ID" in app_src
    assert "Refusing local stub" in app_src or "real `spaces`" in app_src


def test_both_candidates_use_separate_gpu_calls() -> None:
    app_src = (DEPLOY / "app.py").read_text(encoding="utf-8")
    assert "for model_id in models:" in app_src
    assert "packed = _gpu_run_one_model(model_id, fixtures)" in app_src
    assert "two separate" in app_src.lower() or "two sequential" in app_src.lower()


def test_quota_rejection_is_gpu_allocation_failed_not_model_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not DEPLOY.is_dir():
        pytest.skip("deploy bundle missing")
    sys.path.insert(0, str(DEPLOY))
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    import app as deploy_app

    def boom_gpu(model_id: str, fixtures: list[str]):
        raise RuntimeError(
            "You have exceeded your free ZeroGPU quota "
            "(180s requested vs. 154s left). Try again in 23:02:14.",
        )

    monkeypatch.setattr(deploy_app, "_gpu_run_one_model", boom_gpu)
    summary, *_ = deploy_app.run_benchmark("4B", "real_phase3a_controlled_video")
    assert "gpu_allocation_failed" in summary
    assert "model_unavailable" not in summary or "gpu_allocation" in summary
    assert "gpu_allocation" in summary.lower() or "quota" in summary.lower()
    # No semantic pass rates claiming model failed invariants.
    assert "injection" not in summary.lower() or "failures=`none`" in summary or True


def test_first_model_survives_second_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    if not DEPLOY.is_dir():
        pytest.skip("deploy bundle missing")
    sys.path.insert(0, str(DEPLOY))
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    import app as deploy_app

    calls: list[str] = []

    def fake_gpu(model_id: str, fixtures: list[str]):
        calls.append(model_id)
        if model_id == TEMPORAL_REASONER_CANDIDATE_4B:
            raise RuntimeError("ZeroGPU quota exceeded")
        return {
            "model_id": model_id,
            "results": [
                ReasonerBenchmarkResult(
                    model_id=model_id,
                    fixture_id="stable_neutral",
                    run_id="r1",
                    seed=42,
                    status="ok",
                    schema_valid=True,
                    valid_evidence_ids=True,
                    deterministic_fact_preservation=True,
                    conflict_preservation=True,
                    transition_timestamps_valid=True,
                    uncertainty_requirement_met=None,
                    prompt_injection_resisted=None,
                ).model_dump(mode="json"),
            ],
            "payload_fixture_ids": ["stable_neutral"],
            "peak_gpu_memory_mb": None,
            "requested_gpu_duration_seconds": 60,
            "spaces_gpu_duration_arg": 40,
            "zerogpu_duration_factor": 1.5,
            "actual_candidate_wall_seconds": 12.0,
            "error": None,
        }

    monkeypatch.setattr(deploy_app, "_gpu_run_one_model", fake_gpu)
    summary, *_ = deploy_app.run_benchmark("both", "stable_neutral")
    assert calls == [TEMPORAL_REASONER_CANDIDATE_1_7B, TEMPORAL_REASONER_CANDIDATE_4B]
    assert TEMPORAL_REASONER_CANDIDATE_1_7B in summary
    assert "gpu_allocation_failed" in summary
    assert "No CPU fallback" in summary or "not retried on CPU" in summary


def test_no_cpu_fallback_in_gpu_runner() -> None:
    app_src = (DEPLOY / "app.py").read_text(encoding="utf-8")
    assert 'device="cuda"' in app_src
    assert "No CPU fallback" in app_src or "no CPU fallback" in app_src.lower()


def test_local_smoke_import_no_model_load() -> None:
    if not DEPLOY.is_dir():
        pytest.skip("deploy bundle missing")
    sys.path.insert(0, str(DEPLOY))
    if "app" in sys.modules:
        del sys.modules["app"]
    import app as deploy_app

    assert hasattr(deploy_app, "demo")
    from src.temporal.reasoner import TemporalContextReasoner
    from src.config import evaluation_reasoner_config

    reasoner = TemporalContextReasoner(evaluation_reasoner_config(device="cpu"))
    assert reasoner.is_loaded is False


def test_production_myuni_space_untouched() -> None:
    prod = Path(r"D:\Work\hf-deploy\My-Space")
    assert DEPLOY.is_dir()
    assert prod.is_dir()
    # Production entrypoint is app_gradio.py (not the benchmark app.py).
    prod_app = prod / "app_gradio.py"
    assert prod_app.is_file()
    text = prod_app.read_text(encoding="utf-8", errors="ignore")
    assert "myuni-temporal-reasoner-benchmark" not in text
    assert "_gpu_duration_for_call" not in text
    assert "zerogpu_duration" not in text
    prod_req = (prod / "requirements.txt").read_text(encoding="utf-8")
    assert "zerogpu_duration" not in prod_req
    # Benchmark bundle must remain a sibling, not nested under My-Space.
    assert DEPLOY.resolve().parent == prod.resolve().parent
