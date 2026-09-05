"""ZeroGPU duration estimation for the temporal reasoner benchmark Space.

Pure helpers — no torch / spaces imports. Used by the deployment wrapper and
unit tests so requested GPU quotas stay small and explicit.

Durations are model-aware: Qwen3-4B needs a larger single-fixture budget than
1.7B based on measured ZeroGPU wall time (~82s for one truncated 4B run).
"""

from __future__ import annotations

from typing import Optional, Sequence

from src.config import (
    TEMPORAL_REASONER_CANDIDATE_1_7B,
    TEMPORAL_REASONER_CANDIDATE_4B,
)

# Bounded ZeroGPU request sizes (seconds). Never request 1200s.
GPU_DURATION_SINGLE_1_7B_SECONDS = 60
GPU_DURATION_SINGLE_4B_SECONDS = 120
GPU_DURATION_ALL_FIXTURES_CAP_1_7B_SECONDS = 120
GPU_DURATION_ALL_FIXTURES_CAP_4B_SECONDS = 300

# Backward-compatible aliases (1.7B / default profile).
GPU_DURATION_SINGLE_FIXTURE_SECONDS = GPU_DURATION_SINGLE_1_7B_SECONDS
GPU_DURATION_ALL_FIXTURES_CAP_SECONDS = GPU_DURATION_ALL_FIXTURES_CAP_1_7B_SECONDS

# Approximate load + first-token overhead for one candidate on ZeroGPU.
GPU_DURATION_LOAD_OVERHEAD_1_7B_SECONDS = 28
GPU_DURATION_LOAD_OVERHEAD_4B_SECONDS = 20
# Approximate per-fixture generation budget after the model is warm.
GPU_DURATION_PER_FIXTURE_1_7B_SECONDS = 7
# Observed ~67s gen at 768 tokens; budget for 1024-token capped outputs.
GPU_DURATION_PER_FIXTURE_4B_SECONDS = 55


def _is_4b_candidate(model_id: Optional[str]) -> bool:
    if not model_id:
        return False
    mid = str(model_id).strip()
    return mid == TEMPORAL_REASONER_CANDIDATE_4B or "Qwen3-4B" in mid


def estimate_gpu_duration_seconds(
    n_fixtures: int,
    model_id: Optional[str] = None,
) -> int:
    """Return the smallest realistic GPU duration for one candidate run.

    Rules:
    - 1.7B: single ≈ 60s; multi capped at 120s
    - 4B: single ≈ 120s; multi uses prepare + per-fixture budget, capped at 300s
    - never returns 1200 or other oversized static values
    """
    n = max(1, int(n_fixtures))
    if _is_4b_candidate(model_id):
        if n == 1:
            return int(GPU_DURATION_SINGLE_4B_SECONDS)
        estimated = int(
            GPU_DURATION_LOAD_OVERHEAD_4B_SECONDS
            + n * GPU_DURATION_PER_FIXTURE_4B_SECONDS,
        )
        return int(min(GPU_DURATION_ALL_FIXTURES_CAP_4B_SECONDS, estimated))

    if n == 1:
        return int(GPU_DURATION_SINGLE_1_7B_SECONDS)
    estimated = int(
        GPU_DURATION_LOAD_OVERHEAD_1_7B_SECONDS
        + n * GPU_DURATION_PER_FIXTURE_1_7B_SECONDS,
    )
    return int(min(GPU_DURATION_ALL_FIXTURES_CAP_1_7B_SECONDS, estimated))


def estimate_gpu_duration_for_fixture_ids(
    fixture_ids: Sequence[str] | None,
    *args,
    model_id: Optional[str] = None,
    **kwargs,
) -> int:
    """Duration callable helper; ignores Gradio extras via *args/**kwargs."""
    n = len(list(fixture_ids or [])) or 1
    # Allow model_id via kwargs when Gradio/spaces passes it.
    mid = model_id if model_id is not None else kwargs.get("model_id")
    return estimate_gpu_duration_seconds(n, model_id=mid)
