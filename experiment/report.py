"""Export experiment results and generate Markdown reports."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Union

from experiment.runner import ExperimentRunResult, SampleExperimentResult

PathLike = Union[str, Path]


def export_results_json(result: ExperimentRunResult, path: PathLike) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def _flatten_sample(row: SampleExperimentResult) -> dict[str, Any]:
    base: dict[str, Any] = {
        "sample_id": row.sample_id,
        "modality": row.modality,
        "status": row.status,
        "user_id": row.user_id,
        "notes": row.notes,
        "gold_label": row.gold_label,
        "pred_label": row.pred_label,
        "pred_score": row.pred_score,
        "confidence": row.confidence,
        "label_match": row.label_match,
        "latency_seconds": row.latency_seconds,
        "modality_conflict": row.modality_conflict,
        "warnings_count": len(row.warnings),
        "warnings": " | ".join(row.warnings) if row.warnings else None,
        "error": row.error,
        "text_evidence_label": (row.modality_evidence.text or {}).get("label"),
        "visual_evidence_label": (row.modality_evidence.visual or {}).get("label"),
        "ocr_evidence_label": (row.modality_evidence.ocr or {}).get("label"),
        "speech_evidence_label": (row.modality_evidence.speech or {}).get("label"),
        "ocr_text_preview": (row.ocr_text or "")[:120] or None,
        "transcript_preview": (row.transcript or "")[:120] or None,
    }
    if row.video_strategies:
        fixed = row.video_strategies.get("fixed_fps")
        scene = row.video_strategies.get("scene_keyframe")
        if fixed:
            base.update(
                {
                    "fixed_fps_pred_label": fixed.pred_label,
                    "fixed_fps_frames": fixed.frames_extracted,
                    "fixed_fps_seconds": fixed.total_seconds,
                },
            )
        if scene:
            base.update(
                {
                    "scene_pred_label": scene.pred_label,
                    "scene_frames": scene.frames_extracted,
                    "scene_seconds": scene.total_seconds,
                    "scene_resolved_strategy": scene.resolved_strategy,
                    "sampling_sentiment_differs": (
                        fixed.pred_label != scene.pred_label
                        if fixed and scene
                        else None
                    ),
                },
            )
    return base


def export_results_csv(result: ExperimentRunResult, path: PathLike) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [_flatten_sample(s) for s in result.samples]
    if not rows:
        out.write_text("", encoding="utf-8")
        return out
    fieldnames = list(rows[0].keys())
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def generate_markdown_report(result: ExperimentRunResult) -> str:
    """Build a concise, non-promotional experiment report."""
    agg = result.aggregates
    lines: list[str] = [
        f"# POC Experiment Report: {result.experiment_id}",
        "",
        f"**Generated:** {result.finished_at}  ",
        f"**Manifest:** `{result.manifest_path}`  ",
    ]
    if result.name:
        lines.append(f"**Name:** {result.name}  ")
    if result.description:
        lines.append(f"**Description:** {result.description}  ")
    lines.extend(
        [
            "",
            "> " + result.note,
            "",
            "## Experiment configuration",
            "",
            f"- Samples requested: **{agg.get('n_samples_total', 0)}**",
            f"- Completed OK: **{agg.get('n_ok', 0)}**",
            f"- Errors: **{agg.get('n_errors', 0)}**",
            f"- Modalities: {', '.join(result.configuration.get('modalities', []))}",
            f"- Video strategy comparison enabled: **{result.configuration.get('video_compare_strategies')}**",
            "",
            "## Model identifiers",
            "",
        ],
    )
    models = result.runtime.get("models") or {}
    for key, value in models.items():
        lines.append(f"- **{key}:** `{value}`")
    vs = result.runtime.get("video_sampling") or {}
    if vs:
        lines.append(f"- **video_sampling defaults:** `{json.dumps(vs, ensure_ascii=False)}`")
    if result.runtime.get("fusion_source"):
        lines.append(f"- **fusion config:** `{result.runtime['fusion_source']}`")

    lines.extend(["", "## Per-modality results", ""])
    by_mod = agg.get("by_modality") or {}
    if not by_mod:
        lines.append("_No successful samples._")
    for modality, block in by_mod.items():
        lines.append(f"### {modality}")
        lines.append("")
        lines.append(f"- Samples: {block.get('n_samples', 0)}")
        lat = block.get("latency") or {}
        if lat.get("count"):
            lines.append(
                f"- Latency (s): mean={lat.get('mean_seconds')} "
                f"p50={lat.get('p50_seconds')} p95={lat.get('p95_seconds')} "
                f"max={lat.get('max_seconds')}",
            )
        cls = block.get("classification")
        if cls:
            lines.append(
                f"- Labeled subset (n={cls.get('n_labeled')}): "
                f"accuracy={_fmt_pct(cls.get('accuracy'))}, "
                f"macro-F1={cls.get('f1_macro'):.3f}, "
                f"weighted-F1={cls.get('f1_weighted'):.3f}",
            )
        lines.append(f"- Modality conflicts flagged: {block.get('n_modality_conflicts', 0)}")
        lines.append(f"- Warning strings (total): {block.get('n_warnings', 0)}")
        disagreements = block.get("disagreements") or []
        if disagreements:
            lines.append("")
            lines.append("**Gold vs prediction disagreements:**")
            for d in disagreements[:10]:
                lines.append(
                    f"- `{d['sample_id']}`: gold={d['gold_label']} → pred={d['pred_label']} "
                    f"(score={d.get('pred_score')})",
                )
            if len(disagreements) > 10:
                lines.append(f"- … and {len(disagreements) - 10} more")
        lines.append("")

    overall_lat = agg.get("latency_overall") or {}
    if overall_lat.get("count"):
        lines.extend(
            [
                "## Latency statistics (all OK samples)",
                "",
                f"- mean={overall_lat.get('mean_seconds')}s, "
                f"p50={overall_lat.get('p50_seconds')}s, "
                f"p95={overall_lat.get('p95_seconds')}s, "
                f"max={overall_lat.get('max_seconds')}s",
                "",
            ],
        )

    vcmp = agg.get("video_sampling_comparison") or {}
    if vcmp.get("n_videos_compared"):
        lines.extend(
            [
                "## Video sampling comparison (experimental)",
                "",
                "This section compares **fixed_fps** vs **scene_keyframe** on the same clips. "
                "It does **not** declare either strategy superior.",
                "",
                f"- Videos compared: {vcmp.get('n_videos_compared')}",
                f"- Final prediction differed: {vcmp.get('sentiment_differs_count', 0)}",
                "",
            ],
        )
        for comp in vcmp.get("comparisons") or []:
            lines.append(f"### `{comp['sample_id']}`")
            ff = comp["fixed_fps"]
            sc = comp["scene_keyframe"]
            lines.append(
                f"- fixed_fps → {ff.get('pred_label')} "
                f"({ff.get('frames_extracted')} frames, {ff.get('total_seconds')}s)",
            )
            lines.append(
                f"- scene_keyframe → {sc.get('pred_label')} "
                f"({sc.get('frames_extracted')} frames, {sc.get('total_seconds')}s, "
                f"resolved={sc.get('resolved_strategy')})",
            )
            lines.append(f"- sentiment differs: **{comp.get('sentiment_differs')}**")
            lines.append("")

    conflicts = agg.get("modality_conflict_examples") or []
    lines.extend(["## Modality conflict examples", ""])
    if not conflicts:
        lines.append("_None flagged in this run._")
    else:
        for ex in conflicts[:8]:
            lines.append(f"- `{ex['sample_id']}` ({ex['modality']}): overall={ex.get('pred_label')}")
            if ex.get("fusion_explanation"):
                lines.append(f"  - {ex['fusion_explanation']}")
        if len(conflicts) > 8:
            lines.append(f"- … and {len(conflicts) - 8} more")

    errors = agg.get("errors") or []
    lines.extend(["", "## Failures", ""])
    if not errors:
        lines.append("_No sample-level failures._")
    else:
        for err in errors:
            lines.append(f"- `{err['sample_id']}` ({err['modality']}): {err.get('error')}")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Metrics apply only to supplied gold labels on this local manifest; not a benchmark claim.",
            "- Fusion weights and thresholds are POC defaults (`config/fusion.yaml`), not client scoring rules.",
            "- Missing FFmpeg/Tesseract/scenedetect reduces modality coverage; warnings are recorded per sample.",
            "- First model load adds one-time download/latency not isolated in these statistics.",
            "- English-only models; results are not validated for production deployment.",
            "",
        ],
    )
    return "\n".join(lines)


def write_experiment_outputs(result: ExperimentRunResult, output_dir: PathLike) -> dict[str, Path]:
    """Write JSON, CSV, and Markdown report under ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "results_json": export_results_json(result, out / "results.json"),
        "results_csv": export_results_csv(result, out / "results.csv"),
    }
    report_path = out / "report.md"
    report_path.write_text(generate_markdown_report(result) + "\n", encoding="utf-8")
    paths["report_md"] = report_path
    return paths
