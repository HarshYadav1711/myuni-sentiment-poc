"""Unit tests for POC experiment runner (mocked pipeline, no model downloads)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.manifest import ExperimentManifest, ExperimentSample, load_manifest
from experiment.report import export_results_csv, generate_markdown_report, write_experiment_outputs
from experiment.runner import (
    ModalityEvidenceSummary,
    SampleExperimentResult,
    VideoStrategyResult,
    compute_aggregates,
    run_experiment,
)
from src.schemas import (
    ActivityAnalysisResult,
    AnalysisBlock,
    FusionDiagnostics,
    InputMetadata,
    ModalityBundle,
    PocRuntimeInfo,
    SentimentEvidence,
)


def _ev(label: str = "neutral", score: float = 0.0) -> SentimentEvidence:
    return SentimentEvidence(
        label=label,  # type: ignore[arg-type]
        score=score,
        confidence=0.7,
        model="stub",
    )


def _activity_result(label: str, score: float, *, conflict: bool = False) -> ActivityAnalysisResult:
    overall = _ev(label, score)
    fusion = FusionDiagnostics(
        modality_conflict=conflict,
        explanation="stub fusion" if conflict else "",
    )
    return ActivityAnalysisResult(
        activity_id="ACT-1",
        user_id="U1",
        activity_type="text",
        input=InputMetadata(text_preview="x"),
        analysis=AnalysisBlock(
            overall=overall,
            modalities=ModalityBundle(text=overall),
            fusion=fusion,
            runtime=PocRuntimeInfo(models={"text": "stub-text", "fusion": "poc-fusion"}),
        ),
    )


def test_manifest_validation_text_requires_text() -> None:
    with pytest.raises(ValueError, match="requires non-blank text"):
        ExperimentSample(sample_id="X", modality="text")


def test_load_manifest_fixture() -> None:
    manifest = load_manifest(ROOT / "experiment" / "fixtures" / "poc_manifest.json")
    assert manifest.experiment_id == "poc-synthetic-local-v1"
    assert len(manifest.samples) == 5
    assert manifest.video_compare_strategies is True


def test_compute_aggregates_classification_and_disagreements() -> None:
    rows = [
        SampleExperimentResult(
            sample_id="A",
            modality="text",
            status="ok",
            gold_label="positive",
            pred_label="positive",
            pred_score=0.5,
            latency_seconds=0.1,
        ),
        SampleExperimentResult(
            sample_id="B",
            modality="text",
            status="ok",
            gold_label="negative",
            pred_label="neutral",
            pred_score=0.0,
            latency_seconds=0.2,
            modality_conflict=True,
            fusion_explanation="text vs visual disagree",
            modality_evidence=ModalityEvidenceSummary(text=_ev("negative", -0.4)),
        ),
        SampleExperimentResult(
            sample_id="C",
            modality="image",
            status="error",
            error="missing file",
        ),
    ]
    agg = compute_aggregates(rows)
    assert agg["n_ok"] == 2
    assert agg["n_errors"] == 1
    text_block = agg["by_modality"]["text"]
    assert text_block["classification"]["n_labeled"] == 2
    assert text_block["classification"]["accuracy"] == 0.5
    assert len(text_block["disagreements"]) == 1
    assert len(agg["modality_conflict_examples"]) == 1


def test_compute_aggregates_video_sampling_comparison() -> None:
    rows = [
        SampleExperimentResult(
            sample_id="V1",
            modality="video",
            status="ok",
            pred_label="neutral",
            latency_seconds=1.0,
            video_strategies={
                "fixed_fps": VideoStrategyResult(
                    strategy="fixed_fps",
                    pred_label="neutral",
                    frames_extracted=2,
                    total_seconds=1.0,
                ),
                "scene_keyframe": VideoStrategyResult(
                    strategy="scene_keyframe",
                    pred_label="positive",
                    frames_extracted=1,
                    total_seconds=0.8,
                    resolved_strategy="scene_keyframe",
                ),
            },
        ),
    ]
    agg = compute_aggregates(rows)
    vcmp = agg["video_sampling_comparison"]
    assert vcmp["n_videos_compared"] == 1
    assert vcmp["sentiment_differs_count"] == 1


def test_run_experiment_with_stub_pipeline() -> None:
    manifest = ExperimentManifest(
        experiment_id="stub-exp",
        samples=[
            ExperimentSample(
                sample_id="T1",
                modality="text",
                text="hello world",
                gold_label="positive",
            ),
        ],
    )
    pipeline = MagicMock()
    pipeline.analyze_text.return_value = _activity_result("positive", 0.6)
    pipeline.text_analyzer.model_name = "stub-text"
    pipeline.audio_analyzer.whisper_model_name = "stub-asr"
    pipeline.video_analyzer.frame_sampler.name = "fixed_fps"
    pipeline.video_analyzer.sampling.fps = 1.0
    pipeline.video_analyzer.sampling.max_frames = 60
    pipeline.video_analyzer.sampling.max_ocr_frames = 8

    result = run_experiment(manifest, manifest_path="stub.json", pipeline=pipeline)
    assert result.aggregates["n_ok"] == 1
    assert result.samples[0].pred_label == "positive"
    assert result.samples[0].label_match is True
    pipeline.analyze_text.assert_called_once()


def test_run_experiment_video_compare_calls_both_strategies() -> None:
    manifest = ExperimentManifest(
        experiment_id="vid-exp",
        video_compare_strategies=True,
        samples=[
            ExperimentSample(
                sample_id="V1",
                modality="video",
                path="data/samples/synthetic_sample.mp4",
                text="clip",
            ),
        ],
    )
    pipeline = MagicMock()
    pipeline.text_analyzer.model_name = "stub-text"
    pipeline.audio_analyzer.whisper_model_name = "stub-asr"
    pipeline.video_analyzer.sampling.fps = 1.0
    pipeline.video_analyzer.sampling.max_frames = 60
    pipeline.video_analyzer.sampling.max_ocr_frames = 8

    def _analyze(activity):  # type: ignore[no-untyped-def]
        label = "neutral" if "FPS" in activity.activity_id else "positive"
        return _activity_result(label, 0.1 if label == "neutral" else 0.5)

    pipeline.analyze_activity.side_effect = _analyze
    pipeline.video_analyzer.frame_sampler.name = "fixed_fps"

    result = run_experiment(manifest, manifest_path="stub.json", pipeline=pipeline)
    assert pipeline.video_analyzer.set_sampling_strategy.call_count == 2
    row = result.samples[0]
    assert row.video_strategies is not None
    assert row.video_strategies["fixed_fps"].pred_label == "neutral"
    assert row.video_strategies["scene_keyframe"].pred_label == "positive"


def test_markdown_report_and_exports(tmp_path: Path) -> None:
    from experiment.runner import ExperimentRunResult

    result = ExperimentRunResult(
        experiment_id="r1",
        name="demo",
        description="test",
        manifest_path="m.json",
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=datetime.now(timezone.utc).isoformat(),
        runtime={"models": {"text": "stub"}, "video_sampling": {"fps": 1.0}},
        configuration={"modalities": ["text"], "video_compare_strategies": False, "n_samples": 1},
        samples=[
            SampleExperimentResult(
                sample_id="T1",
                modality="text",
                status="ok",
                gold_label="positive",
                pred_label="positive",
                pred_score=0.4,
                confidence=0.8,
                latency_seconds=0.05,
            ),
        ],
        aggregates=compute_aggregates(
            [
                SampleExperimentResult(
                    sample_id="T1",
                    modality="text",
                    status="ok",
                    gold_label="positive",
                    pred_label="positive",
                    pred_score=0.4,
                    latency_seconds=0.05,
                ),
            ],
        ),
    )
    md = generate_markdown_report(result)
    assert "POC Experiment Report" in md
    assert "Limitations" in md
    assert "not a benchmark claim" in md.lower() or "not a benchmark" in md.lower()

    paths = write_experiment_outputs(result, tmp_path / "out")
    assert paths["results_json"].is_file()
    assert paths["results_csv"].is_file()
    assert paths["report_md"].is_file()
    payload = json.loads(paths["results_json"].read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "r1"
    csv_text = export_results_csv(result, tmp_path / "out2" / "r.csv").read_text(encoding="utf-8")
    assert "sample_id" in csv_text
    assert "TXT" not in csv_text or "T1" in csv_text
