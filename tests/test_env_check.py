"""Tests for environment health reporting."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_check import build_health_report, configured_models, video_sampling_defaults


def test_configured_models_contains_expected_keys() -> None:
    models = configured_models()
    assert "text" in models
    assert "visual" in models
    assert "asr" in models
    assert "cardiffnlp" in models["text"]
    assert "siglip" in models["visual"].lower()


def test_video_sampling_defaults() -> None:
    cfg = video_sampling_defaults()
    assert cfg["fps"] == 1.0
    assert cfg["max_frames"] == 12
    assert cfg["default_strategy"] == "fixed_fps"


def test_build_health_report_structure() -> None:
    report = build_health_report(db_path="data/myuni_poc.db")
    assert "python_version" in report
    assert "dependencies" in report
    assert "ffmpeg" in report["dependencies"]
    assert "tesseract" in report["dependencies"]
    assert "cuda" in report["dependencies"]
    assert "transformers" in report["dependencies"]
    assert "models" in report
    assert "video_sampling" in report
    assert "sqlite" in report
    assert "NOT the final client" in report["note"] or "not client" in report["note"].lower()
