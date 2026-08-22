"""Run controlled POC experiments via the central pipeline."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from evaluation.metrics import classification_metrics
from src.pipeline import MyUniSentimentPipeline
from src.runtime_info import build_poc_runtime_info
from src.schemas import ActivityAnalysisResult, ActivityInput, SentimentEvidence

from experiment.manifest import ExperimentManifest, ExperimentSample

logger = logging.getLogger(__name__)

POC_EXPERIMENT_NOTE = (
    "POC experiment output — fusion weights and metrics are evaluation defaults only; "
    "not the final client business scoring methodology."
)


@dataclass
class ModalityEvidenceSummary:
    text: Optional[dict[str, Any]] = None
    visual: Optional[dict[str, Any]] = None
    ocr: Optional[dict[str, Any]] = None
    speech: Optional[dict[str, Any]] = None


@dataclass
class VideoStrategyResult:
    strategy: str
    resolved_strategy: Optional[str] = None
    pred_label: Optional[str] = None
    pred_score: Optional[float] = None
    confidence: Optional[float] = None
    frames_extracted: Optional[int] = None
    frames_analyzed: Optional[int] = None
    extraction_seconds: Optional[float] = None
    processing_seconds: Optional[float] = None
    total_seconds: Optional[float] = None
    visual_label: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class SampleExperimentResult:
    sample_id: str
    modality: str
    status: str  # ok | error
    user_id: Optional[str] = None
    notes: Optional[str] = None
    gold_label: Optional[str] = None
    pred_label: Optional[str] = None
    pred_score: Optional[float] = None
    confidence: Optional[float] = None
    label_match: Optional[bool] = None
    latency_seconds: Optional[float] = None
    warnings: list[str] = field(default_factory=list)
    modality_evidence: ModalityEvidenceSummary = field(default_factory=ModalityEvidenceSummary)
    modality_conflict: bool = False
    fusion_explanation: Optional[str] = None
    ocr_text: Optional[str] = None
    transcript: Optional[str] = None
    video_diagnostics: Optional[dict[str, Any]] = None
    video_strategies: Optional[dict[str, VideoStrategyResult]] = None
    error: Optional[str] = None
    raw_activity_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentRunResult:
    experiment_id: str
    name: Optional[str]
    description: Optional[str]
    manifest_path: str
    started_at: str
    finished_at: str
    runtime: dict[str, Any]
    configuration: dict[str, Any]
    samples: list[SampleExperimentResult] = field(default_factory=list)
    aggregates: dict[str, Any] = field(default_factory=dict)
    note: str = POC_EXPERIMENT_NOTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "manifest_path": self.manifest_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "runtime": self.runtime,
            "configuration": self.configuration,
            "samples": [s.to_dict() for s in self.samples],
            "aggregates": self.aggregates,
            "note": self.note,
        }


def _summarize_evidence(ev: Optional[SentimentEvidence]) -> Optional[dict[str, Any]]:
    if ev is None:
        return None
    out: dict[str, Any] = {
        "label": ev.label,
        "score": round(float(ev.score), 4),
        "confidence": round(float(ev.confidence), 4),
        "model": ev.model,
    }
    if ev.probabilities:
        out["probabilities"] = {k: round(float(v), 4) for k, v in ev.probabilities.items()}
    return out


def _extract_modality_evidence(result: ActivityAnalysisResult) -> ModalityEvidenceSummary:
    mods = result.analysis.modalities
    return ModalityEvidenceSummary(
        text=_summarize_evidence(mods.text),
        visual=_summarize_evidence(mods.visual),
        ocr=_summarize_evidence(mods.ocr),
        speech=_summarize_evidence(mods.speech),
    )


def _result_from_activity(
    sample: ExperimentSample,
    result: ActivityAnalysisResult,
    *,
    latency_seconds: float,
) -> SampleExperimentResult:
    overall = result.analysis.overall
    fusion = result.analysis.fusion
    label_match = None
    if sample.gold_label is not None:
        label_match = sample.gold_label == overall.label

    video_diag = None
    if result.analysis.video is not None:
        video_diag = result.analysis.video.model_dump(mode="json")

    return SampleExperimentResult(
        sample_id=sample.sample_id,
        modality=sample.modality,
        status="ok",
        user_id=sample.user_id,
        notes=sample.notes,
        gold_label=sample.gold_label,
        pred_label=overall.label,
        pred_score=round(float(overall.score), 4),
        confidence=round(float(overall.confidence), 4),
        label_match=label_match,
        latency_seconds=round(latency_seconds, 4),
        warnings=list(result.analysis.warnings),
        modality_evidence=_extract_modality_evidence(result),
        modality_conflict=bool(fusion.modality_conflict) if fusion else False,
        fusion_explanation=fusion.explanation if fusion else None,
        ocr_text=result.analysis.ocr_text,
        transcript=result.analysis.transcript,
        video_diagnostics=video_diag,
        raw_activity_id=result.activity_id,
    )


def _analyze_video_strategy(
    pipeline: MyUniSentimentPipeline,
    sample: ExperimentSample,
    *,
    strategy: str,
    activity_id: str,
) -> VideoStrategyResult:
    pipeline.video_analyzer.set_sampling_strategy(strategy)
    user_id = sample.user_id or "EXP-USER"
    activity = ActivityInput(
        activity_id=activity_id,
        user_id=user_id,
        activity_type="video",
        text=sample.text,
        media_path=str(Path(sample.path).resolve()),
        created_at=datetime.now(timezone.utc),
    )
    started = time.perf_counter()
    result = pipeline.analyze_activity(activity)
    total = time.perf_counter() - started
    analysis = result.analysis
    video = analysis.video
    visual = analysis.modalities.visual
    return VideoStrategyResult(
        strategy=strategy,
        resolved_strategy=video.sampling_strategy if video else strategy,
        pred_label=analysis.overall.label,
        pred_score=round(float(analysis.overall.score), 4),
        confidence=round(float(analysis.overall.confidence), 4),
        frames_extracted=video.frames_extracted if video else None,
        frames_analyzed=video.frames_analyzed if video else None,
        extraction_seconds=video.extraction_seconds if video else None,
        processing_seconds=video.processing_seconds if video else None,
        total_seconds=round(total, 4),
        visual_label=visual.label if visual else None,
        warnings=list(analysis.warnings),
    )


def _run_sample(
    pipeline: MyUniSentimentPipeline,
    sample: ExperimentSample,
    *,
    video_compare: bool,
) -> SampleExperimentResult:
    user_id = sample.user_id or "EXP-USER"
    activity_id = f"EXP-{sample.sample_id}-{uuid.uuid4().hex[:6].upper()}"

    try:
        if sample.modality == "video" and (
            video_compare or sample.compare_sampling is True
        ):
            fixed = _analyze_video_strategy(
                pipeline,
                sample,
                strategy="fixed_fps",
                activity_id=f"{activity_id}-FPS",
            )
            scene = _analyze_video_strategy(
                pipeline,
                sample,
                strategy="scene_keyframe",
                activity_id=f"{activity_id}-SCN",
            )
            sentiment_differs = fixed.pred_label != scene.pred_label
            # Primary row uses fixed_fps baseline for gold comparison consistency.
            row = SampleExperimentResult(
                sample_id=sample.sample_id,
                modality=sample.modality,
                status="ok",
                user_id=sample.user_id,
                notes=sample.notes,
                gold_label=sample.gold_label,
                pred_label=fixed.pred_label,
                pred_score=fixed.pred_score,
                confidence=fixed.confidence,
                label_match=(
                    sample.gold_label == fixed.pred_label if sample.gold_label else None
                ),
                latency_seconds=fixed.total_seconds,
                warnings=fixed.warnings,
                video_diagnostics={
                    "compare_sampling": True,
                    "sentiment_differs": sentiment_differs,
                    "fixed_fps_label": fixed.pred_label,
                    "scene_keyframe_label": scene.pred_label,
                },
                video_strategies={
                    "fixed_fps": fixed,
                    "scene_keyframe": scene,
                },
                raw_activity_id=activity_id,
            )
            if sample.gold_label is not None and scene.pred_label != sample.gold_label:
                row.notes = (row.notes or "") + " [scene_keyframe mismatch vs gold noted]"
            return row

        started = time.perf_counter()
        if sample.modality == "text":
            result = pipeline.analyze_text(
                sample.text,
                user_id=user_id,
                activity_id=activity_id,
            )
        else:
            activity = ActivityInput(
                activity_id=activity_id,
                user_id=user_id,
                activity_type=sample.modality,
                text=sample.text,
                media_path=str(Path(sample.path).resolve()),
                created_at=datetime.now(timezone.utc),
            )
            if sample.modality == "video":
                pipeline.video_analyzer.set_sampling_strategy("fixed_fps")
            result = pipeline.analyze_activity(activity)
        latency = time.perf_counter() - started
        return _result_from_activity(sample, result, latency_seconds=latency)
    except Exception as exc:  # noqa: BLE001 — isolate samples
        logger.exception("Experiment sample failed sample_id=%s", sample.sample_id)
        return SampleExperimentResult(
            sample_id=sample.sample_id,
            modality=sample.modality,
            status="error",
            user_id=sample.user_id,
            notes=sample.notes,
            gold_label=sample.gold_label,
            error=str(exc),
        )


def _latency_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    n = len(ordered)

    def _pct(p: float) -> float:
        if n == 1:
            return ordered[0]
        idx = min(n - 1, int(round(p * (n - 1))))
        return ordered[idx]

    return {
        "count": n,
        "mean_seconds": round(sum(ordered) / n, 4),
        "min_seconds": round(ordered[0], 4),
        "max_seconds": round(ordered[-1], 4),
        "p50_seconds": round(_pct(0.5), 4),
        "p95_seconds": round(_pct(0.95), 4),
    }


def compute_aggregates(results: list[SampleExperimentResult]) -> dict[str, Any]:
    """Compute experiment-level metrics from per-sample rows."""
    ok_rows = [r for r in results if r.status == "ok"]
    errors = [r for r in results if r.status == "error"]
    by_modality: dict[str, list[SampleExperimentResult]] = {}
    for row in ok_rows:
        by_modality.setdefault(row.modality, []).append(row)

    modality_metrics: dict[str, Any] = {}
    for modality, rows in by_modality.items():
        latencies = [r.latency_seconds for r in rows if r.latency_seconds is not None]
        labeled = [
            (r.gold_label, r.pred_label)
            for r in rows
            if r.gold_label is not None and r.pred_label is not None
        ]
        block: dict[str, Any] = {
            "n_samples": len(rows),
            "latency": _latency_stats([float(x) for x in latencies]),
            "n_warnings": sum(len(r.warnings) for r in rows),
            "n_modality_conflicts": sum(1 for r in rows if r.modality_conflict),
        }
        if labeled:
            y_true = [t for t, _ in labeled]
            y_pred = [p for _, p in labeled]
            cls = classification_metrics(y_true, y_pred)
            block["classification"] = {
                "n_labeled": cls["n_examples"],
                "accuracy": cls["accuracy"],
                "f1_macro": cls["f1_macro"],
                "f1_weighted": cls["f1_weighted"],
                "precision_macro": cls["precision_macro"],
                "recall_macro": cls["recall_macro"],
                "confusion_matrix": cls["confusion_matrix"],
            }
            block["disagreements"] = [
                {
                    "sample_id": r.sample_id,
                    "gold_label": r.gold_label,
                    "pred_label": r.pred_label,
                    "pred_score": r.pred_score,
                }
                for r in rows
                if r.gold_label is not None and r.pred_label is not None and r.gold_label != r.pred_label
            ]
        modality_metrics[modality] = block

    video_compare_rows = [
        r for r in ok_rows if r.video_strategies is not None
    ]
    video_compare: dict[str, Any] = {"n_videos_compared": len(video_compare_rows), "comparisons": []}
    for row in video_compare_rows:
        fixed = row.video_strategies["fixed_fps"]  # type: ignore[index]
        scene = row.video_strategies["scene_keyframe"]  # type: ignore[index]
        video_compare["comparisons"].append(
            {
                "sample_id": row.sample_id,
                "sentiment_differs": fixed.pred_label != scene.pred_label,
                "fixed_fps": {
                    "pred_label": fixed.pred_label,
                    "pred_score": fixed.pred_score,
                    "frames_extracted": fixed.frames_extracted,
                    "total_seconds": fixed.total_seconds,
                },
                "scene_keyframe": {
                    "pred_label": scene.pred_label,
                    "pred_score": scene.pred_score,
                    "frames_extracted": scene.frames_extracted,
                    "total_seconds": scene.total_seconds,
                    "resolved_strategy": scene.resolved_strategy,
                },
            },
        )
    if video_compare_rows:
        video_compare["sentiment_differs_count"] = sum(
            1 for c in video_compare["comparisons"] if c["sentiment_differs"]
        )

    conflict_examples = [
        {
            "sample_id": r.sample_id,
            "modality": r.modality,
            "pred_label": r.pred_label,
            "fusion_explanation": r.fusion_explanation,
            "modality_evidence": asdict(r.modality_evidence),
        }
        for r in ok_rows
        if r.modality_conflict
    ]

    all_latencies = [float(r.latency_seconds) for r in ok_rows if r.latency_seconds is not None]
    return {
        "n_samples_total": len(results),
        "n_ok": len(ok_rows),
        "n_errors": len(errors),
        "latency_overall": _latency_stats(all_latencies),
        "by_modality": modality_metrics,
        "video_sampling_comparison": video_compare,
        "modality_conflict_examples": conflict_examples,
        "errors": [
            {"sample_id": r.sample_id, "modality": r.modality, "error": r.error}
            for r in errors
        ],
    }


def run_experiment(
    manifest: ExperimentManifest,
    *,
    manifest_path: str,
    pipeline: Optional[MyUniSentimentPipeline] = None,
) -> ExperimentRunResult:
    """Analyze all manifest samples and return structured experiment output."""
    started_at = datetime.now(timezone.utc)
    pipe = pipeline or MyUniSentimentPipeline()
    runtime_info = build_poc_runtime_info(pipe)

    sample_results: list[SampleExperimentResult] = []
    for sample in manifest.samples:
        compare = manifest.video_compare_strategies
        if sample.compare_sampling is False:
            compare = False
        sample_results.append(
            _run_sample(pipe, sample, video_compare=compare),
        )

    finished_at = datetime.now(timezone.utc)
    aggregates = compute_aggregates(sample_results)

    return ExperimentRunResult(
        experiment_id=manifest.experiment_id,
        name=manifest.name,
        description=manifest.description,
        manifest_path=manifest_path,
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        runtime=runtime_info.model_dump(mode="json"),
        configuration={
            "video_compare_strategies": manifest.video_compare_strategies,
            "n_samples": len(manifest.samples),
            "modalities": sorted({s.modality for s in manifest.samples}),
        },
        samples=sample_results,
        aggregates=aggregates,
    )
