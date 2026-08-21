"""Milestone 3 image analysis tests (no brittle ML probability assertions)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyzers.image import ImageAnalyzer
from src.analyzers.ocr import OcrExtraction, OcrExtractor, is_meaningful_ocr_text, normalize_ocr_text
from src.analyzers.visual import VisualSentimentAnalyzer
from src.pipeline import MyUniSentimentPipeline
from src.schemas import ActivityInput, SentimentEvidence


SAMPLES = ROOT / "data" / "samples"
SYNTHETIC = SAMPLES / "synthetic_sample.png"


@pytest.fixture(scope="module")
def sample_image_path() -> Path:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    if not SYNTHETIC.is_file():
        img = Image.new("RGB", (320, 180), color=(70, 130, 180))
        draw = ImageDraw.Draw(img)
        draw.rectangle((20, 20, 300, 160), outline=(255, 255, 255), width=3)
        draw.ellipse((120, 50, 200, 130), fill=(255, 215, 0))
        draw.text((40, 150), "CAMPUS EVENT", fill=(255, 255, 255))
        img.save(SYNTHETIC)
    assert SYNTHETIC.is_file()
    return SYNTHETIC


def _stub_visual_evidence() -> SentimentEvidence:
    return SentimentEvidence(
        label="positive",
        score=0.4,
        confidence=0.55,
        probabilities={"negative": 0.2, "neutral": 0.25, "positive": 0.55},
        model="stub-visual",
        details={"method": "zero-shot concept scoring"},
    )


def _stub_text_evidence(*, source: str = "caption") -> SentimentEvidence:
    return SentimentEvidence(
        label="positive",
        score=0.8,
        confidence=0.9,
        probabilities={"negative": 0.05, "neutral": 0.05, "positive": 0.9},
        model="stub-text",
        details={"source": source},
    )


def test_valid_local_image_opens(sample_image_path: Path) -> None:
    image = VisualSentimentAnalyzer.load_image(sample_image_path)
    assert image.mode == "RGB"
    assert image.size[0] > 0 and image.size[1] > 0


def test_missing_image_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        VisualSentimentAnalyzer.load_image(SAMPLES / "does_not_exist.png")


def test_corrupt_or_unsupported_file(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"not-an-image-at-all")
    with pytest.raises(ValueError, match="Unsupported or corrupt"):
        VisualSentimentAnalyzer.load_image(bad)


def test_ocr_gracefully_unavailable(sample_image_path: Path) -> None:
    visual = MagicMock()
    visual.analyze_image.return_value = _stub_visual_evidence()

    ocr = MagicMock(spec=OcrExtractor)
    ocr.extract.return_value = OcrExtraction(
        available=False,
        text=None,
        warning="OCR unavailable: Tesseract executable not found",
    )

    text = MagicMock()
    analyzer = ImageAnalyzer(visual_analyzer=visual, ocr_extractor=ocr, text_analyzer=text)
    image = VisualSentimentAnalyzer.load_image(sample_image_path)
    result = analyzer.analyze_image(image)

    assert result.visual.label in {"positive", "neutral", "negative"}
    assert result.ocr_sentiment is None
    assert any("OCR unavailable" in w for w in result.warnings)
    text.analyze.assert_not_called()


def test_ocr_empty_result(sample_image_path: Path) -> None:
    visual = MagicMock()
    visual.analyze_image.return_value = _stub_visual_evidence()
    ocr = MagicMock(spec=OcrExtractor)
    ocr.extract.return_value = OcrExtraction(
        available=True,
        text=None,
        warning="OCR returned no text",
    )
    text = MagicMock()
    analyzer = ImageAnalyzer(visual_analyzer=visual, ocr_extractor=ocr, text_analyzer=text)
    result = analyzer.analyze_image(VisualSentimentAnalyzer.load_image(sample_image_path))
    assert result.ocr_text is None
    assert result.ocr_sentiment is None
    assert any("no text" in w.lower() for w in result.warnings)
    text.analyze.assert_not_called()


def test_caption_remains_independently_represented(sample_image_path: Path) -> None:
    visual = MagicMock()
    visual.analyze_path.return_value = type(
        "IE",
        (),
        {
            "visual": _stub_visual_evidence(),
            "ocr_text": "SALE 50% OFF",
            "ocr_sentiment": _stub_text_evidence(source="ocr"),
            "warnings": [],
        },
    )()

    # Use a real pipeline but stub image analyzer + text analyzer methods.
    pipeline = MyUniSentimentPipeline()
    pipeline._image_analyzer = visual  # type: ignore[attr-defined]

    def fake_text_analyze(text: object) -> SentimentEvidence:
        ev = _stub_text_evidence(source="caption")
        return ev

    pipeline._text_analyzer.analyze = fake_text_analyze  # type: ignore[method-assign]
    pipeline._text_analyzer.validate_text = lambda t: str(t).strip()  # type: ignore[method-assign]

    activity = ActivityInput(
        activity_id="ACT-IMG-1",
        user_id="U100",
        activity_type="image",
        text="Loved the campus fest!",
        media_path=str(sample_image_path),
        created_at=datetime.now(timezone.utc),
    )
    result = pipeline.analyze_activity(activity)

    assert result.activity_type == "image"
    assert result.analysis.modalities.text is not None
    assert result.analysis.modalities.visual is not None
    assert result.analysis.modalities.ocr is not None
    assert result.analysis.modalities.text.details is not None
    assert result.analysis.modalities.text.details.get("source") == "caption"
    assert result.analysis.modalities.ocr.details is not None
    assert result.analysis.modalities.ocr.details.get("source") == "ocr"
    # Caption and OCR are separate objects
    assert result.analysis.modalities.text is not result.analysis.modalities.ocr
    assert result.analysis.ocr_text == "SALE 50% OFF"
    # Overall fusion present and schema-valid
    assert result.analysis.overall.model == "poc-fusion"
    assert -1.0 <= result.analysis.overall.score <= 1.0


def test_standardized_output_schema_image_without_caption(sample_image_path: Path) -> None:
    visual_stub = MagicMock()
    visual_stub.analyze_path.return_value = type(
        "IE",
        (),
        {
            "visual": _stub_visual_evidence(),
            "ocr_text": None,
            "ocr_sentiment": None,
            "warnings": ["OCR unavailable: Tesseract executable not found"],
        },
    )()

    pipeline = MyUniSentimentPipeline()
    pipeline._image_analyzer = visual_stub  # type: ignore[attr-defined]

    activity = ActivityInput(
        activity_id="ACT-IMG-2",
        user_id="U101",
        activity_type="image",
        media_path=str(sample_image_path),
        created_at=datetime.now(timezone.utc),
    )
    result = pipeline.analyze_activity(activity)
    payload = result.model_dump_json_compatible()
    assert payload["activity_type"] == "image"
    assert payload["analysis"]["modalities"]["text"] is None
    assert payload["analysis"]["modalities"]["visual"]["model"] == "stub-visual"
    assert payload["analysis"]["modalities"]["ocr"] is None
    assert any("OCR unavailable" in w for w in payload["analysis"]["warnings"])


def test_normalize_and_meaningful_ocr_helpers() -> None:
    assert normalize_ocr_text("  hello \n\n\n world  ") == "hello\nworld"
    assert is_meaningful_ocr_text("AB") is False
    assert is_meaningful_ocr_text("ABC") is True


@pytest.mark.integration
def test_real_visual_zero_shot_runs(sample_image_path: Path) -> None:
    """Optional real SigLIP 2 smoke: structure only, no brittle probabilities."""
    analyzer = VisualSentimentAnalyzer(device="cpu")
    evidence = analyzer.analyze_path(sample_image_path)
    assert evidence.model == "google/siglip2-base-patch16-224"
    assert evidence.label in {"positive", "neutral", "negative"}
    assert -1.0 <= evidence.score <= 1.0
    assert evidence.probabilities is not None
    assert set(evidence.probabilities) == {"negative", "neutral", "positive"}
    assert evidence.details is not None
    assert evidence.details.get("method") == "zero-shot concept scoring"
