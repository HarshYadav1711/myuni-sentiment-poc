"""ZeroGPU duration estimation for the temporal reasoner benchmark Space.

Pure helpers — no torch / spaces imports. Used by the deployment wrapper and
unit tests so requested GPU quotas stay small and explicit.
"""

from __future__ import annotations

from typing import Sequence

# Bounded ZeroGPU request sizes (seconds). Never request 1200s.
GPU_DURATION_SINGLE_FIXTURE_SECONDS = 60
GPU_DURATION_ALL_FIXTURES_CAP_SECONDS = 120
# Approximate load + first-token overhead for one candidate on ZeroGPU.
GPU_DURATION_LOAD_OVERHEAD_SECONDS = 28
# Approximate per-fixture generation budget after the model is warm.
GPU_DURATION_PER_FIXTURE_SECONDS = 7


def estimate_gpu_duration_seconds(n_fixtures: int) -> int:
    """Return the smallest realistic GPU duration for one candidate run.

    Rules:
    - exactly 1 fixture → 60s
    - multiple fixtures → load overhead + per-fixture budget, capped at 120s
    - never exceeds ``GPU_DURATION_ALL_FIXTURES_CAP_SECONDS``
    - never returns 1200 or other oversized static values
    """
    n = max(1, int(n_fixtures))
    if n == 1:
        return int(GPU_DURATION_SINGLE_FIXTURE_SECONDS)
    estimated = int(
        GPU_DURATION_LOAD_OVERHEAD_SECONDS + n * GPU_DURATION_PER_FIXTURE_SECONDS,
    )
    return int(min(GPU_DURATION_ALL_FIXTURES_CAP_SECONDS, estimated))


def estimate_gpu_duration_for_fixture_ids(
    fixture_ids: Sequence[str] | None,
    *args,
    **kwargs,
) -> int:
    """Duration callable helper; ignores Gradio extras via *args/**kwargs."""
    n = len(list(fixture_ids or [])) or 1
    return estimate_gpu_duration_seconds(n)
