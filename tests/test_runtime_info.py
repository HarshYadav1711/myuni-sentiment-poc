"""Tests for POC runtime metadata in analysis output."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime_info import build_poc_runtime_info


def test_runtime_info_defaults_without_pipeline() -> None:
    info = build_poc_runtime_info()
    assert info.models["text"]
    assert info.models["visual"]
    assert info.video_sampling is not None
    assert info.video_sampling["fps"] == 1.0


def test_runtime_info_from_pipeline_stub() -> None:
    pipeline = MagicMock()
    pipeline.text_analyzer.model_name = "text-model"
    pipeline.audio_analyzer.whisper_model_name = "whisper-model"
    pipeline.video_analyzer.frame_sampler.name = "fixed_fps"
    pipeline.video_analyzer.sampling.fps = 1.0
    pipeline.video_analyzer.sampling.max_frames = 60
    pipeline.video_analyzer.sampling.max_ocr_frames = 8
    pipeline.video_analyzer.temporal_config.window_seconds = 5.0
    pipeline.video_analyzer.temporal_reasoner_config.model_id = "Qwen/Qwen3-1.7B"

    info = build_poc_runtime_info(pipeline)
    assert info.models["text"] == "text-model"
    assert info.models["asr"] == "whisper-model"
    assert info.models["temporal_reasoner"] == "Qwen/Qwen3-1.7B"
    assert info.video_sampling["strategy"] == "fixed_fps"
    assert info.video_sampling["temporal_window_seconds"] == 5.0


def test_runtime_info_defaults_include_temporal_window() -> None:
    info = build_poc_runtime_info()
    assert info.video_sampling is not None
    assert info.video_sampling["temporal_window_seconds"] == 5.0
