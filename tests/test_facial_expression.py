"""Unit tests for facial helpers and text chunking (no model download)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyzers.facial_expression import FacialExpressionAnalyzer
from src.analyzers.faces import FaceBox, crop_faces
from src.analyzers.text import TextSentimentAnalyzer
from src.config import DEFAULT_VISUAL_MODEL, MAX_VIDEO_FRAMES, VIDEO_SAMPLE_FPS
from src.routing.input_router import CLIENT_ENABLED_MODALITIES, InputType
from src.schemas import SentimentEvidence


def test_client_capabilities_include_audio() -> None:
    assert InputType.AUDIO in CLIENT_ENABLED_MODALITIES


def test_video_sampling_client_defaults() -> None:
    assert VIDEO_SAMPLE_FPS == 1.0
    assert MAX_VIDEO_FRAMES == 12


def test_crop_faces_respects_bounds() -> None:
    image = Image.new("RGB", (200, 200), color=(30, 30, 30))
    boxes = [FaceBox(x=10, y=10, w=40, h=40)]
    crops = crop_faces(image, boxes, pad_ratio=0.25)
    assert len(crops) == 1
    assert crops[0].size[0] > 40


def test_facial_analyzer_scores_haar_crops() -> None:
    visual = MagicMock()
    visual.prompts = {}
    visual.analyze_image.return_value = SentimentEvidence(
        label="positive",
        score=0.5,
        confidence=0.6,
        probabilities={"positive": 0.6, "neutral": 0.3, "negative": 0.1},
        model=DEFAULT_VISUAL_MODEL,
        details={},
    )
    analyzer = FacialExpressionAnalyzer(visual_analyzer=visual)
    image = Image.new("RGB", (160, 160), color=(80, 80, 80))
    with patch(
        "src.analyzers.facial_expression.detect_faces",
        return_value=[FaceBox(20, 20, 60, 60)],
    ):
        outcome = analyzer.analyze_image(image)
    assert outcome.ok
    assert outcome.evidence is not None


def test_text_chunk_aggregation_length_weighted() -> None:
    analyzer = TextSentimentAnalyzer.__new__(TextSentimentAnalyzer)
    analyzer.model_name = "stub"
    analyzer.max_length = 512
    analyzer.chunk_size = 8
    analyzer.chunk_stride = 2
    analyzer._device = "cpu"
    analyzer._tokenizer = MagicMock()
    analyzer._model = MagicMock()
    analyzer._id2label = {0: "negative", 1: "neutral", 2: "positive"}

    chunks = [
        (
            SentimentEvidence(
                label="positive",
                score=0.8,
                confidence=0.9,
                probabilities={"positive": 0.9, "neutral": 0.05, "negative": 0.05},
                model="stub",
            ),
            10,
        ),
        (
            SentimentEvidence(
                label="negative",
                score=-0.6,
                confidence=0.7,
                probabilities={"positive": 0.1, "neutral": 0.2, "negative": 0.7},
                model="stub",
            ),
            30,
        ),
    ]
    agg = TextSentimentAnalyzer._aggregate_chunk_results(
        analyzer,
        chunks,
        total_tokens=40,
    )
    assert agg.details is not None
    assert agg.details["chunking"]["chunks"] == 2
    assert agg.details["chunking"]["method"] == "length_weighted_probability_average"
    # Heavier weight on negative chunk.
    assert agg.probabilities is not None
    assert agg.probabilities["negative"] > agg.probabilities["positive"]
