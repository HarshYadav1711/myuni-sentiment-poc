"""Benchmark fixture catalog metadata and loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.schemas import TemporalContext
from src.temporal.benchmark.export import (
    FrozenReasonerPayload,
    export_frozen_payload,
    load_frozen_payload,
)
from src.temporal.benchmark.synthetic import (
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

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "tests" / "benchmark" / "fixtures"
REAL_CONTROLLED_PAYLOAD_PATH = FIXTURES_DIR / "real_phase3a_controlled_video.json"


@dataclass(frozen=True)
class BenchmarkFixtureSpec:
    fixture_id: str
    description: str
    builder: Optional[Callable[[], TemporalContext]] = None
    frozen_path: Optional[Path] = None
    expected_context_type: Optional[str] = None
    sparse: bool = False
    prompt_injection: bool = False
    known_limitations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


BENCHMARK_FIXTURES: list[BenchmarkFixtureSpec] = [
    BenchmarkFixtureSpec(
        fixture_id="stable_neutral",
        description="Stable neutral visual timeline",
        builder=fixture_stable_neutral,
    ),
    BenchmarkFixtureSpec(
        fixture_id="stable_negative",
        description="Stable negative visual timeline",
        builder=fixture_stable_negative,
    ),
    BenchmarkFixtureSpec(
        fixture_id="increasing_negative",
        description="Worsening P(negative) across windows",
        builder=fixture_increasing_negative,
    ),
    BenchmarkFixtureSpec(
        fixture_id="decreasing_negative",
        description="Improving (decreasing negative) timeline",
        builder=fixture_decreasing_negative,
    ),
    BenchmarkFixtureSpec(
        fixture_id="isolated_negative",
        description="One negative window among neutrals",
        builder=fixture_isolated_negative,
    ),
    BenchmarkFixtureSpec(
        fixture_id="persistent_negative",
        description="Long consecutive negative run",
        builder=fixture_persistent_negative,
    ),
    BenchmarkFixtureSpec(
        fixture_id="visual_pos_speech_neg",
        description="Deterministic visual+/speech− conflict",
        builder=fixture_visual_pos_speech_neg,
    ),
    BenchmarkFixtureSpec(
        fixture_id="visual_neg_speech_neg",
        description="Visual−/speech− agreement (no conflict)",
        builder=fixture_visual_neg_speech_neg,
    ),
    BenchmarkFixtureSpec(
        fixture_id="sparse_visual_only",
        description="Sparse visual-only single-frame evidence",
        builder=fixture_sparse_visual_only,
        sparse=True,
        expected_context_type="uncertain",
    ),
    BenchmarkFixtureSpec(
        fixture_id="informational",
        description="Neutral visual + informational OCR",
        builder=fixture_informational,
        expected_context_type="informational",
    ),
    BenchmarkFixtureSpec(
        fixture_id="quoted_narrative",
        description="Quoted / narrative speech content",
        builder=fixture_quoted_narrative,
        expected_context_type="narrative_or_entertainment",
    ),
    BenchmarkFixtureSpec(
        fixture_id="prompt_injection",
        description="Transcript attempts prompt injection",
        builder=fixture_prompt_injection,
        prompt_injection=True,
    ),
    BenchmarkFixtureSpec(
        fixture_id="real_phase3a_controlled_video",
        description="Frozen Phase 3A controlled real-video payload",
        frozen_path=REAL_CONTROLLED_PAYLOAD_PATH,
        known_limitations=[
            "Speech-window boundary artifact: window 3 ends before "
            "'overwhelming.' which is assigned to window 4 by midpoint rule. "
            "Do not change word assignment during model comparison.",
        ],
        notes=[
            "Generated from demo_assets/temporal_progression_demo.mp4 via "
            "MyUniSentimentPipeline; analyzers not re-run for each candidate.",
        ],
    ),
]


def fixture_ids() -> list[str]:
    return [spec.fixture_id for spec in BENCHMARK_FIXTURES]


def get_fixture_spec(fixture_id: str) -> BenchmarkFixtureSpec:
    for spec in BENCHMARK_FIXTURES:
        if spec.fixture_id == fixture_id:
            return spec
    raise KeyError(f"unknown benchmark fixture: {fixture_id}")


def load_benchmark_payload(spec: BenchmarkFixtureSpec) -> FrozenReasonerPayload:
    if spec.frozen_path is not None:
        if not spec.frozen_path.is_file():
            raise FileNotFoundError(
                f"Frozen fixture missing: {spec.frozen_path}. "
                "Run scripts/export_reasoner_benchmark_payload.py first.",
            )
        payload = load_frozen_payload(spec.frozen_path)
        # Ensure catalog metadata is present even on older exports.
        if spec.known_limitations and not payload.known_limitations:
            payload = payload.model_copy(
                update={"known_limitations": list(spec.known_limitations)},
            )
        return payload
    if spec.builder is None:
        raise ValueError(f"fixture {spec.fixture_id} has no builder or frozen path")
    temporal = spec.builder()
    return export_frozen_payload(
        fixture_id=spec.fixture_id,
        source=f"synthetic:{spec.fixture_id}",
        temporal=temporal,
        baseline_overall=None,
        known_limitations=list(spec.known_limitations),
        notes=list(spec.notes) + [spec.description],
        meta={
            "sparse": spec.sparse,
            "prompt_injection": spec.prompt_injection,
            "expected_context_type": spec.expected_context_type,
        },
    )


def load_all_benchmark_payloads(
    *,
    skip_missing_real: bool = False,
) -> list[FrozenReasonerPayload]:
    out: list[FrozenReasonerPayload] = []
    for spec in BENCHMARK_FIXTURES:
        try:
            out.append(load_benchmark_payload(spec))
        except FileNotFoundError:
            if skip_missing_real and spec.frozen_path is not None:
                continue
            raise
    return out
