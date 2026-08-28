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
    assert TextSentimentAnalyzer(device="cuda")._resolve_device().type == "cpu"


def test_text_analyzer_source_never_calls_cuda() -> None:
    source = (ROOT / "src" / "analyzers" / "text.py").read_text(encoding="utf-8")
    assert "torch.cuda." not in source
    assert 'device("cuda")' not in source
    assert '.to("cuda")' not in source
    assert "device_map" not in source


def test_text_load_does_not_invoke_torch_cuda(monkeypatch) -> None:
    import torch
    from unittest.mock import MagicMock

    from src.analyzers import text as text_mod

    def boom(*_args, **_kwargs):
        raise AssertionError("torch.cuda API must not run during text analysis")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    monkeypatch.setattr(torch.cuda, "current_device", boom)
    monkeypatch.setattr(torch.cuda, "device_count", boom)
    monkeypatch.setattr(torch.cuda, "_lazy_init", boom)

    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    encoded = {"input_ids": torch.tensor([[1, 2, 3]])}
    tokenizer.return_value = encoded

    logits = MagicMock()
    logits.detach.return_value.cpu.return_value.numpy.return_value = [[0.1, 0.2, 0.7]]
    model = MagicMock()
    model.to.return_value = model
    model.return_value.logits = logits

    config = MagicMock()
    config.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    monkeypatch.setattr(
        text_mod,
        "AutoTokenizer",
        MagicMock(from_pretrained=MagicMock(return_value=tokenizer)),
    )
    monkeypatch.setattr(
        text_mod,
        "AutoConfig",
        MagicMock(from_pretrained=MagicMock(return_value=config)),
    )
    monkeypatch.setattr(
        text_mod,
        "AutoModelForSequenceClassification",
        MagicMock(from_pretrained=MagicMock(return_value=model)),
    )

    analyzer = text_mod.TextSentimentAnalyzer()
    evidence = analyzer.analyze("Campus wifi is working well today.")
    assert evidence.label in {"positive", "neutral", "negative"}
    assert analyzer._device is not None
    assert analyzer._device.type == "cpu"
    model.to.assert_called_with("cpu")


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
    assert "return outputs.logits_per_image.detach().float().cpu().numpy()" in source


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
