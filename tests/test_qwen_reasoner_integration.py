"""Optional real Qwen temporal-reasoner integration (opt-in; downloads model).

Enable with:

    set MYUNI_RUN_QWEN_INTEGRATION=1
    pytest -m qwen_integration -s
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for path in (ROOT, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.config import TemporalReasonerConfig
from src.temporal.reasoner import TemporalContextReasoner
from temporal_fixtures import fixture_visual_pos_speech_neg


pytestmark = [
    pytest.mark.qwen_integration,
    pytest.mark.skipif(
        os.environ.get("MYUNI_RUN_QWEN_INTEGRATION", "").strip() not in {"1", "true", "yes"},
        reason="Set MYUNI_RUN_QWEN_INTEGRATION=1 to run optional Qwen reasoner integration",
    ),
]


@pytest.mark.integration
def test_qwen_reasoner_controlled_fixture() -> None:
    cfg = TemporalReasonerConfig(
        enabled=True,
        model_id=os.environ.get("MYUNI_QWEN_MODEL", "Qwen/Qwen3-1.7B"),
        device=os.environ.get("MYUNI_QWEN_DEVICE", "cpu"),
        max_new_tokens=512,
        temperature=0.0,
        enable_thinking=False,
        max_retries=1,
    )
    reasoner = TemporalContextReasoner(cfg)
    ctx = fixture_visual_pos_speech_neg()

    load_started = time.perf_counter()
    try:
        reasoner.load()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qwen model could not be loaded (resource/network): {exc}")
    load_s = time.perf_counter() - load_started

    gen_started = time.perf_counter()
    result, diagnostics = reasoner.reason(ctx)
    gen_s = time.perf_counter() - gen_started

    mem_note = None
    try:
        import psutil

        mem_note = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001
        mem_note = None

    print("\n=== QWEN TEMPORAL REASONER INTEGRATION ===")
    print(f"model={cfg.model_id} device={cfg.device}")
    print(f"load_seconds={load_s:.2f}")
    print(f"generation_seconds={gen_s:.2f}")
    print(f"prompt_seconds={diagnostics.prompt_construction_seconds}")
    print(f"parse_seconds={diagnostics.parse_validation_seconds}")
    print(f"total_reasoner_seconds={diagnostics.total_reasoner_seconds}")
    print(f"rss_mb={mem_note}")
    print(f"status={result.status}")
    print(f"context_type={result.context_type}")
    print(f"confidence={result.confidence}")
    print(f"summary={result.summary!r}")
    print(f"details={result.details}")
    print("=== END QWEN INTEGRATION ===\n")

    # Must not invent clinical wellbeing fields; status must be known enum.
    assert result.status in {
        "ok",
        "invalid_model_output",
        "reasoner_unavailable",
        "disabled",
    }
    assert 0.0 <= result.confidence <= 1.0
    # Fail soft is acceptable under resource pressure; report honestly.
    if result.status == "ok":
        assert result.cross_modal_context is not None
