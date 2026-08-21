"""Tests for ActivityInput validation and JSONL batch ingestion (Milestone 2).

Most tests are pure validation / routing and do not download models.
One mixed-batch integration test exercises real text inference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.batch import BatchIngestor, parse_activity_record
from src.pipeline import MyUniSentimentPipeline
from src.schemas import ActivityInput


def _text_record(**overrides: object) -> dict:
    base: dict = {
        "activity_id": "ACT001",
        "user_id": "U001",
        "activity_type": "text",
        "text": "I loved today's event.",
        "created_at": "2026-08-21T10:00:00+05:30",
    }
    base.update(overrides)
    return base


def test_valid_text_record() -> None:
    activity = parse_activity_record(_text_record())
    assert activity.activity_id == "ACT001"
    assert activity.user_id == "U001"
    assert activity.activity_type == "text"
    assert activity.text == "I loved today's event."
    assert activity.media_path is None


def test_missing_activity_id_rejected() -> None:
    payload = _text_record()
    del payload["activity_id"]
    with pytest.raises(ValidationError) as exc_info:
        ActivityInput.model_validate(payload)
    assert "activity_id" in str(exc_info.value)


def test_missing_user_id_rejected() -> None:
    payload = _text_record()
    del payload["user_id"]
    with pytest.raises(ValidationError) as exc_info:
        ActivityInput.model_validate(payload)
    assert "user_id" in str(exc_info.value)


def test_blank_ids_rejected() -> None:
    with pytest.raises(ValidationError):
        ActivityInput.model_validate(_text_record(activity_id="  "))
    with pytest.raises(ValidationError):
        ActivityInput.model_validate(_text_record(user_id=""))


def test_malformed_activity_type_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ActivityInput.model_validate(_text_record(activity_type="story"))
    assert "activity_type" in str(exc_info.value)


def test_missing_media_path_for_image_rejected() -> None:
    payload = {
        "activity_id": "ACT010",
        "user_id": "U010",
        "activity_type": "image",
        "text": "optional caption",
        "created_at": "2026-08-21T10:00:00+05:30",
    }
    with pytest.raises(ValidationError) as exc_info:
        ActivityInput.model_validate(payload)
    assert "media_path" in str(exc_info.value)


def test_missing_media_path_for_video_rejected() -> None:
    payload = {
        "activity_id": "ACT011",
        "user_id": "U011",
        "activity_type": "video",
        "created_at": "2026-08-21T10:00:00+05:30",
    }
    with pytest.raises(ValidationError) as exc_info:
        ActivityInput.model_validate(payload)
    assert "media_path" in str(exc_info.value)


def test_image_with_media_path_validates_without_caption() -> None:
    activity = ActivityInput.model_validate(
        {
            "activity_id": "ACT012",
            "user_id": "U012",
            "activity_type": "image",
            "media_path": "data/samples/placeholder_image.jpg",
            "created_at": "2026-08-21T10:00:00+05:30",
        },
    )
    assert activity.media_path is not None
    assert activity.text is None


def test_text_activity_rejects_blank_text() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ActivityInput.model_validate(_text_record(text="   "))
    assert "text" in str(exc_info.value).lower()


def test_mixed_batch_continues_after_invalid_record() -> None:
    """Invalid record must not stop later valid/unsupported records."""
    records = [
        _text_record(activity_id="ACT-OK-1", text="Great workshop today!"),
        {"activity_id": "ACT-BAD", "user_id": "U9", "activity_type": "nope"},
        {
            "activity_id": "ACT-IMG",
            "user_id": "U8",
            "activity_type": "image",
            "media_path": "data/samples/synthetic_sample.png",
            "created_at": "2026-08-21T11:00:00+05:30",
        },
        {
            "activity_id": "ACT-VID",
            "user_id": "U7",
            "activity_type": "video",
            "media_path": "data/samples/placeholder_video.mp4",
            "created_at": "2026-08-21T12:00:00+05:30",
        },
        _text_record(activity_id="ACT-OK-2", text="This was awful."),
    ]

    class StubPipeline(MyUniSentimentPipeline):
        def analyze_activity(self, activity):  # type: ignore[no-untyped-def]
            from src.schemas import (
                ActivityAnalysisResult,
                AnalysisBlock,
                InputMetadata,
                ModalityBundle,
                SentimentEvidence,
            )

            evidence = SentimentEvidence(
                label="positive",
                score=0.5,
                confidence=0.9,
                model="stub",
            )
            return ActivityAnalysisResult(
                activity_id=activity.activity_id,
                user_id=activity.user_id,
                activity_type=activity.activity_type,
                input=InputMetadata(text_preview=activity.text, media_path=activity.media_path),
                analysis=AnalysisBlock(
                    overall=evidence,
                    modalities=ModalityBundle(text=evidence if activity.activity_type == "text" else None),
                ),
            )

    result = BatchIngestor(pipeline=StubPipeline()).process_records(records)
    assert result.summary.total == 5
    assert result.summary.invalid == 1
    assert result.summary.valid == 4
    assert result.summary.processed == 4  # 2 text + 1 image + 1 video
    assert result.summary.unsupported == 0
    assert result.summary.failed == 0
    assert [r.status for r in result.records] == [
        "processed",
        "invalid",
        "processed",
        "processed",
        "processed",
    ]


def test_sample_jsonl_file_metrics_with_stub() -> None:
    sample = ROOT / "data" / "samples" / "activities.jsonl"
    assert sample.is_file()

    class StubPipeline(MyUniSentimentPipeline):
        def analyze_activity(self, activity):  # type: ignore[no-untyped-def]
            from src.schemas import (
                ActivityAnalysisResult,
                AnalysisBlock,
                InputMetadata,
                ModalityBundle,
                SentimentEvidence,
            )

            evidence = SentimentEvidence(
                label="neutral",
                score=0.0,
                confidence=0.5,
                model="stub",
            )
            return ActivityAnalysisResult(
                activity_id=activity.activity_id,
                user_id=activity.user_id,
                activity_type=activity.activity_type,
                input=InputMetadata(text_preview=activity.text, media_path=activity.media_path),
                analysis=AnalysisBlock(
                    overall=evidence,
                    modalities=ModalityBundle(text=evidence if activity.activity_type == "text" else None),
                ),
            )

    result = BatchIngestor(pipeline=StubPipeline()).process_file(sample)
    # 3 text + 1 image + 1 video processed; 1 invalid blank text
    assert result.summary.total == 6
    assert result.summary.processed == 5
    assert result.summary.unsupported == 0
    assert result.summary.invalid == 1
    assert result.summary.valid == 5
    assert result.summary.failed == 0
    payload = result.model_dump_json_compatible()
    json.dumps(payload)


@pytest.mark.integration
def test_batch_processes_real_text_activity() -> None:
    records = [
        _text_record(
            activity_id="ACT-REAL",
            text="I really enjoyed today's workshop.",
        ),
    ]
    result = BatchIngestor().process_records(records, source="<test>")
    assert result.summary.processed == 1
    assert result.records[0].result is not None
    assert result.records[0].result.analysis.overall.label == "positive"
    assert result.records[0].result.activity_id == "ACT-REAL"
    assert result.records[0].result.user_id == "U001"
