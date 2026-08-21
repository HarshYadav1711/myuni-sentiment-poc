"""SQLite persistence and daily aggregation tests (Milestone 7)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import MyUniSentimentPipeline
from src.schemas import (
    ActivityAnalysisResult,
    AnalysisBlock,
    InputMetadata,
    ModalityBundle,
    SentimentEvidence,
)
from src.storage.aggregation import compute_daily_user_score, refresh_daily_user_score
from src.storage.repository import SentimentRepository
from src.storage.service import PersistentBatchRunner


def _evidence(label: str, score: float) -> SentimentEvidence:
    return SentimentEvidence(
        label=label,  # type: ignore[arg-type]
        score=score,
        confidence=0.9,
        model="stub",
    )


def _result(activity_id: str, user_id: str, label: str, score: float) -> ActivityAnalysisResult:
    ev = _evidence(label, score)
    return ActivityAnalysisResult(
        activity_id=activity_id,
        user_id=user_id,
        activity_type="text",
        input=InputMetadata(text_preview="x"),
        analysis=AnalysisBlock(
            overall=ev,
            modalities=ModalityBundle(text=ev),
        ),
        processed_at=datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc),
    )


class StubPipeline(MyUniSentimentPipeline):
    def __init__(self, mapping: dict[str, ActivityAnalysisResult] | None = None) -> None:
        # Avoid loading real models.
        self._mapping = mapping or {}
        self._fail_ids: set[str] = set()

    def analyze_activity(self, activity):  # type: ignore[no-untyped-def]
        if activity.activity_id in self._fail_ids:
            raise RuntimeError(f"forced failure for {activity.activity_id}")
        if activity.activity_id in self._mapping:
            return self._mapping[activity.activity_id]
        return _result(activity.activity_id, activity.user_id, "positive", 0.8)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    import json

    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_inserts_and_per_user_separation(tmp_path: Path) -> None:
    db = tmp_path / "poc.db"
    jsonl = tmp_path / "batch.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "activity_id": "A1",
                "user_id": "U1",
                "activity_type": "text",
                "text": "great day",
                "created_at": "2026-08-21T10:00:00+00:00",
            },
            {
                "activity_id": "A2",
                "user_id": "U2",
                "activity_type": "text",
                "text": "awful day",
                "created_at": "2026-08-21T11:00:00+00:00",
            },
        ],
    )
    pipeline = StubPipeline(
        {
            "A1": _result("A1", "U1", "positive", 0.9),
            "A2": _result("A2", "U2", "negative", -0.8),
        },
    )
    result = PersistentBatchRunner(db, pipeline=pipeline).process_file(jsonl)
    assert result.summary.processed == 2
    assert result.batch_id is not None

    repo = SentimentRepository(db)
    assert repo.activity_exists("A1")
    assert repo.get_analysis("A1")["user_id"] == "U1"
    assert repo.get_analysis("A2")["overall_label"] == "negative"

    scores = repo.get_daily_scores(score_date="2026-08-21")
    users = {s.user_id: s for s in scores}
    assert set(users) == {"U1", "U2"}
    assert users["U1"].positive_count == 1
    assert users["U2"].negative_count == 1
    assert users["U1"].daily_sentiment_label == "positive"
    assert "NOT the future client business score" in users["U1"].note


def test_duplicate_activity_skipped_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "poc.db"
    jsonl = tmp_path / "batch.jsonl"
    rows = [
        {
            "activity_id": "DUP1",
            "user_id": "U9",
            "activity_type": "text",
            "text": "hello",
            "created_at": "2026-08-21T10:00:00+00:00",
        },
    ]
    _write_jsonl(jsonl, rows)
    runner = PersistentBatchRunner(db, pipeline=StubPipeline())
    first = runner.process_file(jsonl)
    second = runner.process_file(jsonl)

    assert first.summary.processed == 1
    assert second.summary.processed == 0
    assert second.summary.skipped == 1
    assert second.records[0].status == "skipped"
    assert "already stored" in (second.records[0].note or "").lower()

    repo = SentimentRepository(db)
    with repo.connection() as conn:
        n = conn.execute("SELECT COUNT(*) AS c FROM activities").fetchone()["c"]
    assert n == 1


def test_failed_record_does_not_corrupt_successes(tmp_path: Path) -> None:
    db = tmp_path / "poc.db"
    jsonl = tmp_path / "batch.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "activity_id": "OK1",
                "user_id": "U1",
                "activity_type": "text",
                "text": "nice",
                "created_at": "2026-08-21T10:00:00+00:00",
            },
            {
                "activity_id": "BAD1",
                "user_id": "U1",
                "activity_type": "text",
                "text": "boom",
                "created_at": "2026-08-21T10:05:00+00:00",
            },
            {
                "activity_id": "OK2",
                "user_id": "U1",
                "activity_type": "text",
                "text": "fine",
                "created_at": "2026-08-21T10:10:00+00:00",
            },
        ],
    )
    pipeline = StubPipeline(
        {
            "OK1": _result("OK1", "U1", "positive", 0.7),
            "OK2": _result("OK2", "U1", "neutral", 0.0),
        },
    )
    pipeline._fail_ids.add("BAD1")
    result = PersistentBatchRunner(db, pipeline=pipeline).process_file(jsonl)

    assert result.summary.processed == 2
    assert result.summary.failed == 1
    assert [r.status for r in result.records] == ["processed", "failed", "processed"]

    repo = SentimentRepository(db)
    assert repo.get_analysis("OK1")["status"] == "processed"
    assert repo.get_analysis("BAD1")["status"] == "failed"
    assert repo.get_analysis("OK2")["status"] == "processed"

    daily = repo.get_daily_scores(user_id="U1", score_date="2026-08-21")
    assert len(daily) == 1
    assert daily[0].valid_analysis_count == 2
    assert daily[0].activity_count == 3  # includes failed analysis row


def test_daily_aggregation_helpers() -> None:
    rows = [
        {"status": "processed", "overall_label": "positive", "overall_score": 0.5},
        {"status": "processed", "overall_label": "negative", "overall_score": -0.5},
        {"status": "failed", "overall_label": None, "overall_score": None},
    ]
    score = compute_daily_user_score(
        user_id="U1",
        score_date="2026-08-21",
        rows=[r for r in rows if r["status"] == "processed"],
        activity_count=3,
    )
    assert score.valid_analysis_count == 2
    assert score.activity_count == 3
    assert score.positive_count == 1
    assert score.negative_count == 1
    assert abs((score.mean_sentiment_score or 0) - 0.0) < 1e-9
    assert score.daily_sentiment_label == "neutral"


def test_batch_run_tracking(tmp_path: Path) -> None:
    db = tmp_path / "poc.db"
    jsonl = tmp_path / "batch.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "activity_id": "T1",
                "user_id": "U1",
                "activity_type": "text",
                "text": "ok",
                "created_at": "2026-08-21T10:00:00+00:00",
            },
        ],
    )
    result = PersistentBatchRunner(db, pipeline=StubPipeline()).process_file(jsonl)
    repo = SentimentRepository(db)
    batch = repo.get_batch_run(result.batch_id or "")
    assert batch is not None
    assert batch["status"] == "completed"
    assert batch["succeeded"] == 1
    assert batch["started_at"]
    assert batch["completed_at"]
