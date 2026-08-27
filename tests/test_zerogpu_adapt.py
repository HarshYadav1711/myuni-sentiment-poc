"""Hugging Face / ZeroGPU adaptation tests (no model download)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyzers.text import TextSentimentAnalyzer
from src.analyzers.visual import VisualSentimentAnalyzer, _siglip_gpu_forward
from src.pipeline import MyUniSentimentPipeline


def test_text_analyzer_defaults_to_cpu() -> None:
    analyzer = TextSentimentAnalyzer()
    assert analyzer._resolve_device().type == "cpu"


def test_audio_analyzer_stays_cpu_int8() -> None:
    pipeline = MyUniSentimentPipeline()
    assert pipeline.audio_analyzer.device == "cpu"
    assert pipeline.audio_analyzer.compute_type == "int8"


def test_siglip_gpu_forward_is_spaces_wrapped() -> None:
    source = (ROOT / "src" / "analyzers" / "visual.py").read_text(encoding="utf-8")
    assert "import spaces" in source
    assert "@spaces.GPU" in source
    assert "def _siglip_gpu_forward" in source
    assert callable(_siglip_gpu_forward)


def test_visual_analyze_image_delegates_to_batch(monkeypatch) -> None:
    from PIL import Image

    analyzer = VisualSentimentAnalyzer(device="cpu")
    called: list[int] = []

    def fake_batch(images):
        called.append(len(images))
        from src.schemas import SentimentEvidence

        return [
            SentimentEvidence(
                label="neutral",
                score=0.0,
                confidence=0.5,
                probabilities={"negative": 0.2, "neutral": 0.6, "positive": 0.2},
                model=analyzer.model_name,
                details={"method": "zero-shot concept scoring"},
            )
        ]

    monkeypatch.setattr(analyzer, "analyze_images", fake_batch)
    evidence = analyzer.analyze_image(Image.new("RGB", (8, 8), color=(10, 20, 30)))
    assert called == [1]
    assert evidence.label == "neutral"


def test_pipeline_still_uses_single_visual_analyzer() -> None:
    pipeline = MyUniSentimentPipeline()
    visual = pipeline.image_analyzer._visual
    assert isinstance(visual, VisualSentimentAnalyzer)
    assert pipeline.video_analyzer._image._visual is visual
