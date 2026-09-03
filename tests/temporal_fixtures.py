"""Deterministic synthetic TemporalContext fixtures for Phase 2 reasoner tests.

Builders live in ``src.temporal.benchmark.synthetic`` (shared with Phase 3B-A).
"""

from __future__ import annotations

from src.temporal.benchmark.synthetic import (  # noqa: F401
    ALL_SYNTHETIC_FIXTURES,
    CFG,
    _ev,
    _series_context,
    fixture_decreasing_negative,
    fixture_increasing_negative,
    fixture_informational,
    fixture_isolated_negative,
    fixture_persistent_negative,
    fixture_prompt_injection,
    fixture_quoted_narrative,
    fixture_sparse_visual_only,
    fixture_stable_negative,
    fixture_stable_neutral,
    fixture_visual_neg_speech_neg,
    fixture_visual_pos_speech_neg,
)

# Backward-compatible alias used by existing tests.
ALL_FIXTURES = {
    "stable_neutral": fixture_stable_neutral,
    "stable_negative": fixture_stable_negative,
    "increasing_negative": fixture_increasing_negative,
    "decreasing_negative": fixture_decreasing_negative,
    "isolated_negative": fixture_isolated_negative,
    "persistent_negative": fixture_persistent_negative,
    "visual_pos_speech_neg": fixture_visual_pos_speech_neg,
    "sparse_visual_only": fixture_sparse_visual_only,
    "informational": fixture_informational,
    "quoted_narrative": fixture_quoted_narrative,
}
