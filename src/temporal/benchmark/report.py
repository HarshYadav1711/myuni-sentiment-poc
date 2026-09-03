"""Comparison report / human-review sheet generation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from src.temporal.benchmark.export import FrozenReasonerPayload
from src.temporal.benchmark.schemas import ReasonerBenchmarkResult

PathLike = Union[str, Path]


def aggregate_pass_rates(
    results: Sequence[ReasonerBenchmarkResult],
) -> dict[str, dict[str, float]]:
    """Per-model factual rates — not a single unexplained quality score."""
    by_model: dict[str, list[ReasonerBenchmarkResult]] = defaultdict(list)
    for row in results:
        if row.fixture_id == "__session__":
            continue
        by_model[row.model_id].append(row)

    out: dict[str, dict[str, float]] = {}
    for model_id, rows in by_model.items():
        n = max(1, len(rows))
        out[model_id] = {
            "n": float(len(rows)),
            "schema_success_rate": sum(1 for r in rows if r.schema_valid) / n,
            "grounding_pass_rate": sum(1 for r in rows if r.valid_evidence_ids) / n,
            "conflict_preservation_rate": sum(
                1 for r in rows if r.conflict_preservation
            )
            / n,
            "injection_resistance_rate": sum(
                1 for r in rows if r.prompt_injection_resisted
            )
            / n,
            "repair_rate": sum(1 for r in rows if r.repair_attempted) / n,
            "fact_preservation_rate": sum(
                1 for r in rows if r.deterministic_fact_preservation
            )
            / n,
            "transition_grounding_rate": sum(
                1 for r in rows if r.transition_timestamps_valid
            )
            / n,
            "uncertainty_pass_rate": sum(
                1 for r in rows if r.uncertainty_requirement_met
            )
            / n,
        }
    return out


def performance_summary(
    results: Sequence[ReasonerBenchmarkResult],
) -> dict[str, dict[str, float | int | None]]:
    """Per-model latency / token summary (not a quality score)."""
    by_model: dict[str, list[ReasonerBenchmarkResult]] = defaultdict(list)
    for row in results:
        if row.fixture_id == "__session__":
            continue
        by_model[row.model_id].append(row)

    out: dict[str, dict[str, float | int | None]] = {}
    for model_id, rows in by_model.items():
        gens = [float(r.generation_seconds) for r in rows if r.generation_seconds is not None]
        gens_sorted = sorted(gens)
        n = len(gens_sorted)
        mean = sum(gens_sorted) / n if n else None
        median = gens_sorted[n // 2] if n else None
        p95 = (
            gens_sorted[min(n - 1, int(0.95 * (n - 1)))]
            if n >= 2
            else (gens_sorted[0] if n == 1 else None)
        )
        loads = [float(r.model_load_seconds) for r in rows if r.model_load_seconds]
        out[model_id] = {
            "n_fixtures": len(rows),
            "mean_generation_seconds": mean,
            "median_generation_seconds": median,
            "p95_generation_seconds": p95,
            "total_generation_seconds": sum(gens_sorted) if gens_sorted else None,
            "model_load_seconds_max": max(loads) if loads else None,
            "mean_prompt_tokens": (
                sum(r.prompt_tokens or 0 for r in rows) / len(rows) if rows else None
            ),
            "mean_generated_tokens": (
                sum(r.generated_tokens or 0 for r in rows) / len(rows) if rows else None
            ),
        }
    return out


def write_results_json(
    results: Sequence[ReasonerBenchmarkResult],
    path: PathLike,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": [r.model_dump(mode="json") for r in results],
        "pass_rates": aggregate_pass_rates(results),
        "performance": performance_summary(results),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def write_results_csv(
    results: Sequence[ReasonerBenchmarkResult],
    path: PathLike,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_id",
        "fixture_id",
        "run_id",
        "seed",
        "status",
        "schema_valid",
        "repair_attempted",
        "deterministic_fact_preservation",
        "valid_evidence_ids",
        "transition_timestamps_valid",
        "conflict_preservation",
        "uncertainty_requirement_met",
        "prompt_injection_resisted",
        "context_type",
        "context_type_expected",
        "context_type_match",
        "generation_seconds",
        "total_seconds",
        "prompt_tokens",
        "generated_tokens",
        "human_context_quality",
        "human_summary_grounding",
        "human_useful_uncertainty",
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "model_id": r.model_id,
                    "fixture_id": r.fixture_id,
                    "run_id": r.run_id,
                    "seed": r.seed,
                    "status": r.status,
                    "schema_valid": r.schema_valid,
                    "repair_attempted": r.repair_attempted,
                    "deterministic_fact_preservation": r.deterministic_fact_preservation,
                    "valid_evidence_ids": r.valid_evidence_ids,
                    "transition_timestamps_valid": r.transition_timestamps_valid,
                    "conflict_preservation": r.conflict_preservation,
                    "uncertainty_requirement_met": r.uncertainty_requirement_met,
                    "prompt_injection_resisted": r.prompt_injection_resisted,
                    "context_type": r.context_type,
                    "context_type_expected": r.context_type_expected,
                    "context_type_match": r.context_type_match,
                    "generation_seconds": r.generation_seconds,
                    "total_seconds": r.total_seconds,
                    "prompt_tokens": r.prompt_tokens,
                    "generated_tokens": r.generated_tokens,
                    "human_context_quality": r.human_review.context_quality,
                    "human_summary_grounding": r.human_review.summary_grounding,
                    "human_useful_uncertainty": r.human_review.useful_uncertainty,
                },
            )
    return out


def write_human_review_markdown(
    results: Sequence[ReasonerBenchmarkResult],
    payloads: dict[str, FrozenReasonerPayload],
    path: PathLike,
    *,
    model_order: Optional[Sequence[str]] = None,
) -> Path:
    """Side-by-side comparison sheet with blank human-review fields."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    by_fixture: dict[str, list[ReasonerBenchmarkResult]] = defaultdict(list)
    for row in results:
        by_fixture[row.fixture_id].append(row)

    models = list(model_order) if model_order else sorted({r.model_id for r in results})
    lines: list[str] = [
        "# Temporal Reasoner Benchmark — Human Review Sheet",
        "",
        "Human fields (`context_quality`, `summary_grounding`, `useful_uncertainty`) "
        "are intentionally blank. Do not pre-fill them with a model.",
        "",
        "## Automatic pass rates (not a single quality score)",
        "",
        "```json",
        json.dumps(aggregate_pass_rates(results), indent=2),
        "```",
        "",
    ]

    for fixture_id in sorted(by_fixture):
        payload = payloads.get(fixture_id)
        lines.append(f"## Fixture `{fixture_id}`")
        lines.append("")
        if payload is not None:
            feats = payload.temporal_context.features
            lines.append("### Deterministic facts")
            lines.append("")
            lines.append(f"- trajectory: `{feats.trajectory}`")
            lines.append(f"- negative_persistence: `{feats.negative_persistence}`")
            snw = feats.strongest_negative_window
            lines.append(
                f"- strongest_negative_window: "
                f"`{None if snw is None else snw.model_dump(mode='json')}`",
            )
            lines.append(
                f"- sudden_negative_change: "
                f"`{feats.sudden_negative_change.model_dump(mode='json')}`",
            )
            lines.append(f"- cross_modal_agreement: `{feats.cross_modal_agreement}`")
            lines.append(
                f"- conflicts: `{[c.model_dump(mode='json') for c in feats.cross_modal_conflicts]}`",
            )
            if payload.known_limitations:
                lines.append(f"- known_limitations: {payload.known_limitations}")
            lines.append("")

        rows = {r.model_id: r for r in by_fixture[fixture_id]}
        for model_id in models:
            row = rows.get(model_id)
            lines.append(f"### Model `{model_id}`")
            lines.append("")
            if row is None:
                lines.append("_No result_")
                lines.append("")
                continue
            lines.append(f"- status: `{row.status}`")
            lines.append(f"- schema_valid: `{row.schema_valid}`")
            lines.append(f"- repair_attempted: `{row.repair_attempted}`")
            lines.append(
                f"- invariants: fact=`{row.deterministic_fact_preservation}` "
                f"ids=`{row.valid_evidence_ids}` "
                f"conflict=`{row.conflict_preservation}` "
                f"transitions=`{row.transition_timestamps_valid}` "
                f"uncertainty=`{row.uncertainty_requirement_met}` "
                f"injection=`{row.prompt_injection_resisted}`",
            )
            lines.append(f"- context_type: `{row.context_type}`")
            lines.append(f"- generation_seconds: `{row.generation_seconds}`")
            lines.append(f"- raw_preview: `{row.raw_output_preview}`")
            lines.append("")
            lines.append("Human review (fill manually):")
            lines.append("")
            lines.append("- context_quality: ` `")
            lines.append("- summary_grounding: ` `")
            lines.append("- useful_uncertainty: ` `")
            lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
