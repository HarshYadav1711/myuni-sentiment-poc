"""ZeroGPU duration estimation for the temporal reasoner benchmark Space.

Pure helpers — no torch / spaces imports. Used by the deployment wrapper and
unit tests so requested GPU quotas stay small and explicit.

Platform note (current ZeroGPU ``large`` GPUs, e.g. RTX PRO 6000):
the ``spaces`` client multiplies the decorator ``duration`` by
``duration_factor`` (1.5) before scheduling / quota checks:

    platform_requested = spaces_duration_arg * 1.5

Therefore estimators here speak in **effective** (platform) seconds — what
Hugging Face reports as "Xs requested" — and ``to_spaces_gpu_duration_arg``
converts that into the integer passed to ``@spaces.GPU(duration=...)``.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from src.config import (
    TEMPORAL_REASONER_CANDIDATE_1_7B,
    TEMPORAL_REASONER_CANDIDATE_4B,
)

# Observed from spaces.zero.configs.json for RTX PRO 6000 (default large).
# H200 configs use 1.0; we target the factor that caused 120→180 mismatch.
ZEROGPU_LARGE_DURATION_FACTOR = 1.5

# Effective (platform / quota) seconds. Never request 1200s.
GPU_DURATION_SINGLE_1_7B_SECONDS = 60
GPU_DURATION_SINGLE_4B_SECONDS = 120
GPU_DURATION_ALL_FIXTURES_CAP_1_7B_SECONDS = 120
GPU_DURATION_ALL_FIXTURES_CAP_4B_SECONDS = 300

# Backward-compatible aliases (1.7B / default profile) — effective seconds.
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


def to_spaces_gpu_duration_arg(
    effective_seconds: int,
    *,
    factor: float = ZEROGPU_LARGE_DURATION_FACTOR,
) -> int:
    """Convert desired platform/quota seconds into ``@spaces.GPU(duration=)`` arg.

    For factor 1.5: effective 120 → spaces arg 80 (because 80 * 1.5 = 120).
    """
    eff = max(1, int(effective_seconds))
    f = float(factor) if factor else 1.0
    if f <= 0:
        f = 1.0
    return max(1, int(math.ceil(eff / f)))


def effective_gpu_duration_from_spaces_arg(
    spaces_duration_arg: int,
    *,
    factor: float = ZEROGPU_LARGE_DURATION_FACTOR,
) -> int:
    """Inverse of ``to_spaces_gpu_duration_arg`` (what HF quota check sees)."""
    arg = max(1, int(spaces_duration_arg))
    f = float(factor) if factor else 1.0
    if f <= 0:
        f = 1.0
    return int(round(arg * f))


def estimate_gpu_duration_seconds(
    n_fixtures: int,
    model_id: Optional[str] = None,
) -> int:
    """Return desired **effective** (platform) GPU duration for one candidate.

    Rules (effective / quota seconds, after spaces duration_factor):
    - 1.7B: single ≈ 60s; multi capped at 120s
    - 4B: single ≈ 120s; multi uses prepare + per-fixture budget, capped at 300s
    - never returns 1200 or other oversized static values

    Pass the result through ``to_spaces_gpu_duration_arg`` for ``@spaces.GPU``.
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


def spaces_gpu_duration_for_call(
    n_fixtures: int,
    model_id: Optional[str] = None,
) -> int:
    """Integer for ``@spaces.GPU(duration=...)`` given fixtures × model."""
    effective = estimate_gpu_duration_seconds(n_fixtures, model_id=model_id)
    return to_spaces_gpu_duration_arg(effective)


def estimate_gpu_duration_for_fixture_ids(
    fixture_ids: Sequence[str] | None,
    *args,
    model_id: Optional[str] = None,
    **kwargs,
) -> int:
    """Duration callable helper returning spaces decorator arg."""
    n = len(list(fixture_ids or [])) or 1
    mid = model_id if model_id is not None else kwargs.get("model_id")
    return spaces_gpu_duration_for_call(n, model_id=mid)
