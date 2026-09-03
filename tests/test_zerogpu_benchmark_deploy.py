"""Hardening tests for the ZeroGPU reasoner-benchmark Space bundle."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = Path(r"D:\Work\hf-deploy\myuni-temporal-reasoner-benchmark")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.temporal_reasoner.zerogpu_duration import (
    GPU_DURATION_ALL_FIXTURES_CAP_SECONDS,
    GPU_DURATION_SINGLE_FIXTURE_SECONDS,
    estimate_gpu_duration_seconds,
)
from src.config import (
    TEMPORAL_REASONER_CANDIDATE_1_7B,
    TEMPORAL_REASONER_CANDIDATE_4B,
)
from src.temporal.benchmark.schemas import ReasonerBenchmarkResult


def test_single_fixture_duration_is_short() -> None:
    d = estimate_gpu_duration_seconds(1)
    assert d == GPU_DURATION_SINGLE_FIXTURE_SECONDS
    assert 45 <= d <= 60


def test_all_fixtures_duration_bounded() -> None:
    d = estimate_gpu_duration_seconds(13)
    assert d <= GPU_DURATION_ALL_FIXTURES_CAP_SECONDS
    assert d >= 100  # larger than single-fixture budget, still bounded
    assert d < 1200


def test_single_smaller_than_all() -> None:
    assert estimate_gpu_duration_seconds(1) < estimate_gpu_duration_seconds(13)


def test_no_1200_in_duration_helper_or_deploy_app() -> None:
    assert estimate_gpu_duration_seconds(1) != 1200
    assert estimate_gpu_duration_seconds(13) != 1200
    assert estimate_gpu_duration_seconds(100) != 1200
    app_src = (DEPLOY / "app.py").read_text(encoding="utf-8")
    assert "@spaces.GPU(duration=1200)" not in app_src
    assert "duration=1200" not in app_src


def test_deploy_app_uses_dynamic_duration_callable() -> None:
    app_src = (DEPLOY / "app.py").read_text(encoding="utf-8")
    assert "@spaces.GPU(duration=_gpu_duration_for_call)" in app_src
    assert "estimate_gpu_duration_seconds" in app_src


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
                    uncertainty_requirement_met=True,
                    prompt_injection_resisted=True,
                ).model_dump(mode="json"),
            ],
            "payload_fixture_ids": ["stable_neutral"],
            "peak_gpu_memory_mb": None,
            "requested_gpu_duration_seconds": 60,
            "actual_candidate_wall_seconds": 12.0,
            "error": None,
        }

    monkeypatch.setattr(deploy_app, "_gpu_run_one_model", fake_gpu)
    summary, *_ = deploy_app.run_benchmark("both", "stable_neutral")
    assert calls == [TEMPORAL_REASONER_CANDIDATE_1_7B, TEMPORAL_REASONER_CANDIDATE_4B]
    assert TEMPORAL_REASONER_CANDIDATE_1_7B in summary
    assert "quota" in summary.lower() or "model_unavailable" in summary
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
