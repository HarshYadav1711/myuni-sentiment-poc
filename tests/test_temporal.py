"""Unit tests for Phase 1 temporal context (no live models)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyzers.video import VideoAnalyzer
from src.config import TemporalConfig, TemporalReasonerConfig, VideoSamplingConfig
from src.media.ffmpeg_utils import VideoProbeInfo
from src.pipeline import MyUniSentimentPipeline
from src.schemas import (
    ActivityInput,
    SentimentEvidence,
    SpeechAnalysisResult,
    SpeechSegment,
    TemporalEvent,
    TemporalOcrEvidence,
)
from src.temporal.aggregation import (
    length_weighted_probability_distribution,
    mean_probability_distribution,
)
from src.temporal.builder import build_temporal_context
from src.temporal.events import build_frame_events, segments_overlapping_time, sort_events
from src.temporal.features import TemporalFeatureExtractor
from src.temporal.windows import build_temporal_windows, create_windows


def _probs(neg: float, neu: float, pos: float) -> dict[str, float]:
    return {"negative": neg, "neutral": neu, "positive": pos}


def _ev(
    label: str,
    *,
    neg: float = 0.1,
    neu: float = 0.1,
    pos: float = 0.8,
    conf: float | None = None,
) -> SentimentEvidence:
    probs = _probs(neg, neu, pos)
    if label == "negative":
        probs = _probs(max(neg, 0.6), neu, pos if pos < 0.3 else 0.1)
    elif label == "positive":
        probs = _probs(neg if neg < 0.3 else 0.1, neu, max(pos, 0.6))
    elif label == "neutral":
        probs = _probs(neg if neg < 0.35 else 0.2, max(neu, 0.5), pos if pos < 0.35 else 0.2)
    score = probs["positive"] - probs["negative"]
    return SentimentEvidence(
        label=label,  # type: ignore[arg-type]
        score=score,
        confidence=conf if conf is not None else max(probs.values()),
        probabilities=probs,
        model="stub",
    )


def _neg(p: float = 0.8) -> SentimentEvidence:
    rem = (1.0 - p) / 2.0
    return SentimentEvidence(
        label="negative",
        score=rem - p,
        confidence=p,
        probabilities=_probs(p, rem, rem),
        model="stub",
    )


def _pos(p: float = 0.8) -> SentimentEvidence:
    rem = (1.0 - p) / 2.0
    return SentimentEvidence(
        label="positive",
        score=p - rem,
        confidence=p,
        probabilities=_probs(rem, rem, p),
        model="stub",
    )


def _neu(p: float = 0.7) -> SentimentEvidence:
    rem = (1.0 - p) / 2.0
    return SentimentEvidence(
        label="neutral",
        score=0.0,
        confidence=p,
        probabilities=_probs(rem, p, rem),
        model="stub",
    )


CFG = TemporalConfig(
    window_seconds=5.0,
    negative_prob_threshold=0.45,
    positive_prob_threshold=0.45,
    min_usable_evidence=0.30,
    sudden_negative_delta=0.25,
    trajectory_slope_threshold=0.05,
)


# ---------------------------------------------------------------------------
# 1–4 Events / timestamps / speech / OCR
# ---------------------------------------------------------------------------


def test_event_sorting() -> None:
    events = [
        TemporalEvent(timestamp=3.0, event_id="frame-0002"),
        TemporalEvent(timestamp=1.0, event_id="frame-0000"),
        TemporalEvent(timestamp=2.0, event_id="frame-0001"),
    ]
    sorted_events = sort_events(events)
    assert [e.timestamp for e in sorted_events] == [1.0, 2.0, 3.0]


def test_frame_timestamps_preserved_in_events() -> None:
    events = build_frame_events(
        timestamps=[0.0, 1.0, 2.5],
        visuals=[_pos(), _neu(), _neg()],
        ocr_texts=[None, None, None],
        ocr_sentiments=[None, None, None],
    )
    assert [e.timestamp for e in events] == [0.0, 1.0, 2.5]
    assert events[0].event_id == "frame-0000"
    assert events[2].visual is not None
    assert events[2].visual.label == "negative"


def test_speech_segment_alignment_by_timestamp() -> None:
    segments = [
        SpeechSegment(start=0.0, end=2.0, text="hello"),
        SpeechSegment(start=4.0, end=6.0, text="world"),
    ]
    assert [s.text for s in segments_overlapping_time(segments, 1.0)] == ["hello"]
    assert [s.text for s in segments_overlapping_time(segments, 5.0)] == ["world"]
    assert segments_overlapping_time(segments, 3.0) == []

    events = build_frame_events(
        timestamps=[1.0, 5.0],
        visuals=[_neu(), _neu()],
        ocr_texts=[None, None],
        ocr_sentiments=[None, None],
        speech_segments=segments,
        speech_sentiments_by_segment=[_pos(), _neg()],
    )
    assert events[0].speech is not None
    assert events[0].speech.text == "hello"
    assert events[0].speech.segment_start == 0.0
    assert events[0].speech.segment_end == 2.0
    assert events[1].speech is not None
    assert events[1].speech.text == "world"
    assert events[1].speech.sentiment is not None
    assert events[1].speech.sentiment.label == "negative"


def test_ocr_timestamp_inheritance() -> None:
    events = build_frame_events(
        timestamps=[0.0, 7.5],
        visuals=[_pos(), _neu()],
        ocr_texts=["Sale!", "Closed"],
        ocr_sentiments=[_pos(), _neg()],
    )
    assert events[0].ocr is not None
    assert events[0].ocr.text == "Sale!"
    assert events[0].timestamp == 0.0
    assert events[1].ocr is not None
    assert events[1].timestamp == 7.5
    # OCR has no independent clock — tied to frame event timestamp.
    assert isinstance(events[1].ocr, TemporalOcrEvidence)


# ---------------------------------------------------------------------------
# 5–9 Windows / aggregation / missing modalities
# ---------------------------------------------------------------------------


def test_window_creation() -> None:
    bounds = create_windows(20.0, window_seconds=5.0)
    assert bounds == [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0)]


def test_short_final_window() -> None:
    bounds = create_windows(12.0, window_seconds=5.0)
    assert bounds == [(0.0, 5.0), (5.0, 10.0), (10.0, 12.0)]
    assert bounds[-1][1] - bounds[-1][0] == 2.0


def test_visual_aggregation_mean_probabilities() -> None:
    a = SentimentEvidence(
        label="negative",
        score=-0.4,
        confidence=0.6,
        probabilities=_probs(0.6, 0.2, 0.2),
        model="stub",
    )
    b = SentimentEvidence(
        label="negative",
        score=-0.2,
        confidence=0.4,
        probabilities=_probs(0.4, 0.4, 0.2),
        model="stub",
    )
    mean = mean_probability_distribution([a, b])
    assert mean is not None
    assert abs(mean["negative"] - 0.5) < 1e-9
    assert abs(mean["neutral"] - 0.3) < 1e-9
    assert abs(mean["positive"] - 0.2) < 1e-9


def test_speech_aggregation_length_weighted() -> None:
    short = _pos(0.9)
    long = _neg(0.9)
    # weight 1 vs 10 → strongly negative
    agg = length_weighted_probability_distribution([short, long], [1.0, 10.0])
    assert agg is not None
    assert agg["negative"] > agg["positive"]


def test_missing_modalities_excluded_not_neutral() -> None:
    ctx = build_temporal_context(
        duration_seconds=5.0,
        timestamps=[1.0],
        visuals=[_pos()],
        ocr_texts=[None],
        ocr_sentiments=[None],
        speech_segments=[],
        config=CFG,
    )
    window = ctx.windows[0]
    assert "visual" in window.available_modalities
    assert "speech" not in window.available_modalities
    assert "ocr" not in window.available_modalities
    assert window.speech_probabilities is None
    assert window.ocr_probabilities is None
    assert window.visual_probabilities is not None


# ---------------------------------------------------------------------------
# 10–17 Trajectory / persistence / runs / sudden change
# ---------------------------------------------------------------------------


def _windows_from_visual_series(
    series: list[SentimentEvidence],
    *,
    duration: float = 20.0,
) -> list:
    timestamps = [i * 5.0 + 1.0 for i in range(len(series))]
    ctx = build_temporal_context(
        duration_seconds=duration,
        timestamps=timestamps,
        visuals=series,
        ocr_texts=[None] * len(series),
        ocr_sentiments=[None] * len(series),
        config=CFG,
    )
    return ctx.windows


def test_trajectory_stable_positive() -> None:
    windows = _windows_from_visual_series([_pos(), _pos(), _pos(), _pos()])
    feats = TemporalFeatureExtractor(CFG).extract(windows)
    assert feats.trajectory == "stable_positive"


def test_trajectory_worsening_increasing_negative() -> None:
    series = [
        SentimentEvidence(
            label="positive",
            score=0.5,
            confidence=0.6,
            probabilities=_probs(0.1, 0.3, 0.6),
            model="stub",
        ),
        SentimentEvidence(
            label="neutral",
            score=0.0,
            confidence=0.5,
            probabilities=_probs(0.35, 0.4, 0.25),
            model="stub",
        ),
        SentimentEvidence(
            label="negative",
            score=-0.5,
            confidence=0.7,
            probabilities=_probs(0.7, 0.2, 0.1),
            model="stub",
        ),
        SentimentEvidence(
            label="negative",
            score=-0.8,
            confidence=0.85,
            probabilities=_probs(0.85, 0.1, 0.05),
            model="stub",
        ),
    ]
    windows = _windows_from_visual_series(series)
    feats = TemporalFeatureExtractor(CFG).extract(windows)
    assert feats.trajectory == "increasing_negative"


def test_trajectory_improving_decreasing_negative() -> None:
    series = [
        SentimentEvidence(
            label="negative",
            score=-0.8,
            confidence=0.85,
            probabilities=_probs(0.85, 0.1, 0.05),
            model="stub",
        ),
        SentimentEvidence(
            label="negative",
            score=-0.5,
            confidence=0.7,
            probabilities=_probs(0.7, 0.2, 0.1),
            model="stub",
        ),
        SentimentEvidence(
            label="neutral",
            score=0.0,
            confidence=0.5,
            probabilities=_probs(0.3, 0.45, 0.25),
            model="stub",
        ),
        SentimentEvidence(
            label="positive",
            score=0.5,
            confidence=0.6,
            probabilities=_probs(0.1, 0.3, 0.6),
            model="stub",
        ),
    ]
    windows = _windows_from_visual_series(series)
    feats = TemporalFeatureExtractor(CFG).extract(windows)
    assert feats.trajectory == "decreasing_negative"


def test_trajectory_mixed() -> None:
    # Mirror-symmetric P(neg) → ~0 normalized slope, but mixed labels.
    series = [
        SentimentEvidence(
            label="positive",
            score=0.5,
            confidence=0.7,
            probabilities=_probs(0.1, 0.2, 0.7),
            model="stub",
        ),
        SentimentEvidence(
            label="negative",
            score=-0.6,
            confidence=0.7,
            probabilities=_probs(0.7, 0.2, 0.1),
            model="stub",
        ),
        SentimentEvidence(
            label="negative",
            score=-0.6,
            confidence=0.7,
            probabilities=_probs(0.7, 0.2, 0.1),
            model="stub",
        ),
        SentimentEvidence(
            label="positive",
            score=0.5,
            confidence=0.7,
            probabilities=_probs(0.1, 0.2, 0.7),
            model="stub",
        ),
    ]
    windows = _windows_from_visual_series(series)
    feats = TemporalFeatureExtractor(CFG).extract(windows)
    assert feats.trajectory == "mixed"


def test_negative_persistence() -> None:
    windows = _windows_from_visual_series([_neg(), _neg(), _pos(), _neu()])
    feats = TemporalFeatureExtractor(CFG).extract(windows)
    assert feats.negative_persistence is not None
    # 2 of 4 usable windows meaningfully negative
    assert abs(feats.negative_persistence - 0.5) < 1e-9


def test_longest_negative_run() -> None:
    windows = _windows_from_visual_series([_pos(), _neg(), _neg(), _neg()])
    feats = TemporalFeatureExtractor(CFG).extract(windows)
    assert feats.longest_negative_run == 3
    assert feats.longest_negative_run_seconds is not None
    assert feats.longest_negative_run_seconds == pytest.approx(15.0)


def test_strongest_negative_window() -> None:
    series = [
        _neg(0.5),
        SentimentEvidence(
            label="negative",
            score=-0.9,
            confidence=0.9,
            probabilities=_probs(0.9, 0.05, 0.05),
            model="stub",
        ),
        _neu(),
    ]
    windows = _windows_from_visual_series(series, duration=15.0)
    feats = TemporalFeatureExtractor(CFG).extract(windows)
    assert feats.strongest_negative_window is not None
    assert feats.strongest_negative_window.index == 1
    assert feats.strongest_negative_window.score == pytest.approx(0.9)


def test_sudden_negative_change() -> None:
    series = [
        SentimentEvidence(
            label="positive",
            score=0.5,
            confidence=0.6,
            probabilities=_probs(0.1, 0.3, 0.6),
            model="stub",
        ),
        SentimentEvidence(
            label="negative",
            score=-0.7,
            confidence=0.75,
            probabilities=_probs(0.75, 0.15, 0.1),
            model="stub",
        ),
    ]
    windows = _windows_from_visual_series(series, duration=10.0)
    feats = TemporalFeatureExtractor(CFG).extract(windows)
    assert feats.sudden_negative_change.detected is True
    assert feats.sudden_negative_change.delta is not None
    assert feats.sudden_negative_change.delta >= CFG.sudden_negative_delta
    assert feats.sudden_negative_change.from_window == 0
    assert feats.sudden_negative_change.to_window == 1


# ---------------------------------------------------------------------------
# 18–19 Cross-modal agreement / conflict
# ---------------------------------------------------------------------------


def test_cross_modal_agreement_high() -> None:
    ctx = build_temporal_context(
        duration_seconds=5.0,
        timestamps=[1.0],
        visuals=[_pos(0.8)],
        ocr_texts=["great day"],
        ocr_sentiments=[_pos(0.75)],
        speech_segments=[SpeechSegment(start=0.0, end=3.0, text="lovely")],
        speech_sentiments_by_segment=[_pos(0.7)],
        config=CFG,
    )
    assert ctx.features.cross_modal_agreement == "high"
    assert ctx.features.cross_modal_conflicts == []


def test_cross_modal_contradiction() -> None:
    ctx = build_temporal_context(
        duration_seconds=5.0,
        timestamps=[1.0],
        visuals=[_pos(0.85)],
        ocr_texts=[None],
        ocr_sentiments=[None],
        speech_segments=[SpeechSegment(start=0.0, end=4.0, text="this is awful")],
        speech_sentiments_by_segment=[_neg(0.85)],
        config=CFG,
    )
    assert len(ctx.features.cross_modal_conflicts) >= 1
    conflict = ctx.features.cross_modal_conflicts[0]
    assert "visual" in conflict.modalities
    assert "speech" in conflict.modalities
    assert conflict.labels["visual"] == "positive"
    assert conflict.labels["speech"] == "negative"
    # Agreement should not be high when modalities contradict.
    assert ctx.features.cross_modal_agreement in {"low", "moderate", "insufficient_evidence"}


# ---------------------------------------------------------------------------
# 20–23 Sparse / no speech / no OCR / one-frame
# ---------------------------------------------------------------------------


def test_sparse_evidence_insufficient_trajectory() -> None:
    # Probabilities too flat / weak to clear min_usable_evidence.
    weak = SentimentEvidence(
        label="neutral",
        score=0.0,
        confidence=0.2,
        probabilities=_probs(0.2, 0.25, 0.2),
        model="stub",
    )
    # Normalize would make them usable possibly — use very low max class.
    # After normalize 0.2/0.65 etc. max ~0.38 which may still be usable with thr=0.30.
    # Force empty visuals so no usable windows.
    ctx = build_temporal_context(
        duration_seconds=10.0,
        timestamps=[1.0, 6.0],
        visuals=[None, None],
        ocr_texts=[None, None],
        ocr_sentiments=[None, None],
        config=CFG,
    )
    assert ctx.features.trajectory == "insufficient_evidence"
    assert ctx.features.negative_persistence is None
    assert ctx.features.evidence_coverage.usable_windows == 0
    _ = weak  # documented alternative for weak-but-present evidence


def test_video_without_speech() -> None:
    ctx = build_temporal_context(
        duration_seconds=5.0,
        timestamps=[0.0, 1.0],
        visuals=[_pos(), _neu()],
        ocr_texts=[None, None],
        ocr_sentiments=[None, None],
        speech_segments=[],
        config=CFG,
    )
    assert ctx.features.evidence_coverage.speech_coverage == 0.0
    assert all("speech" not in w.available_modalities for w in ctx.windows)


def test_video_without_ocr() -> None:
    ctx = build_temporal_context(
        duration_seconds=5.0,
        timestamps=[0.5],
        visuals=[_neu()],
        ocr_texts=[None],
        ocr_sentiments=[None],
        speech_segments=[SpeechSegment(start=0.0, end=2.0, text="hi")],
        speech_sentiments_by_segment=[_neu()],
        config=CFG,
    )
    assert ctx.features.evidence_coverage.ocr_coverage == 0.0
    assert all("ocr" not in w.available_modalities for w in ctx.windows)


def test_one_frame_video() -> None:
    ctx = build_temporal_context(
        duration_seconds=2.0,
        timestamps=[0.0],
        visuals=[_pos()],
        ocr_texts=[None],
        ocr_sentiments=[None],
        config=CFG,
    )
    assert len(ctx.events) == 1
    assert len(ctx.windows) == 1
    assert ctx.windows[0].end == 2.0
    assert ctx.features.trajectory == "stable_positive"
    assert ctx.features.sudden_negative_change.detected is False


# ---------------------------------------------------------------------------
# 24 Backward compatibility of existing video result fields
# ---------------------------------------------------------------------------


def test_existing_video_result_backward_compatibility(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    pipeline = MyUniSentimentPipeline()
    from src.schemas import TemporalContext, TemporalFeatures, VideoDiagnostics

    temporal = TemporalContext(
        window_seconds=5.0,
        duration_seconds=2.0,
        features=TemporalFeatures(trajectory="stable_neutral"),
    )
    fake_bundle = SimpleNamespace(
        visual=_ev("positive"),
        ocr=None,
        ocr_text=None,
        speech=_ev("negative", neg=0.6, neu=0.2, pos=0.2),
        transcript="this was bad",
        speech_result=SpeechAnalysisResult(
            transcript="this was bad",
            language="en",
            asr_model="base.en",
            sentiment=_ev("negative", neg=0.6, neu=0.2, pos=0.2),
            segments=[SpeechSegment(start=0.0, end=1.5, text="this was bad")],
        ),
        diagnostics=VideoDiagnostics(
            duration_seconds=2.0,
            sampling_strategy="fixed_fps",
            sampling_fps=1.0,
            frames_extracted=2,
            frames_analyzed=2,
            frame_timestamps=[0.0, 1.0],
            extraction_seconds=0.1,
            processing_seconds=0.5,
            has_audio=True,
        ),
        warnings=[],
        overall=_ev("neutral"),
        temporal_context=temporal,
        temporal_reasoning=None,
    )
    pipeline._video_analyzer.analyze = MagicMock(return_value=fake_bundle)  # type: ignore[method-assign]
    pipeline._text_analyzer.analyze = MagicMock(return_value=_ev("positive"))  # type: ignore[method-assign]
    pipeline._text_analyzer.validate_text = lambda t: str(t).strip()  # type: ignore[method-assign]

    activity = ActivityInput(
        activity_id="ACT-T1",
        user_id="U1",
        activity_type="video",
        text="Campus clip",
        media_path=str(video),
        created_at=datetime.now(timezone.utc),
    )
    result = pipeline.analyze_activity(activity)
    dumped = result.model_dump_json_compatible()

    # Existing fields still present.
    assert "analysis" in dumped
    assert "overall" in dumped["analysis"]
    assert "modalities" in dumped["analysis"]
    assert "fusion" in dumped["analysis"]
    assert "video" in dumped["analysis"]
    assert "transcript" in dumped["analysis"]
    # New field added without removing old ones.
    assert "temporal_context" in dumped["analysis"]
    assert dumped["analysis"]["temporal_context"]["features"]["trajectory"] == "stable_neutral"
    assert dumped["analysis"]["modalities"]["visual"]["label"] == "positive"
    assert dumped["analysis"]["modalities"]["speech"]["label"] == "negative"


def test_window_speech_spans_multiple_windows() -> None:
    segments = [SpeechSegment(start=3.0, end=12.0, text="long speech spanning windows")]
    sentiment = _neg(0.7)
    ctx = build_temporal_context(
        duration_seconds=15.0,
        timestamps=[1.0, 6.0, 11.0],
        visuals=[_neu(), _neu(), _neu()],
        ocr_texts=[None, None, None],
        ocr_sentiments=[None, None, None],
        speech_segments=segments,
        speech_sentiments_by_segment=[sentiment],
        config=CFG,
    )
    # Segment overlaps windows 0 (0-5), 1 (5-10), 2 (10-15).
    overlapping_windows = [w for w in ctx.windows if w.speech_segments]
    assert len(overlapping_windows) >= 2


def test_video_analyzer_attaches_temporal_context(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    image = MagicMock()
    image._visual = MagicMock()
    image._visual.analyze_images.return_value = [_pos(), _neu()]
    image.extract_ocr_evidence.return_value = ("Hello World Text", _pos(), [])

    audio = MagicMock()
    audio.analyze.return_value = SpeechAnalysisResult(
        transcript="bad news today",
        language="en",
        asr_model="base.en",
        sentiment=_neg(),
        segments=[SpeechSegment(start=0.5, end=1.5, text="bad news today")],
    )

    text = MagicMock()
    text.analyze.return_value = _neg()

    analyzer = VideoAnalyzer(
        image_analyzer=image,
        audio_analyzer=audio,
        text_analyzer=text,
        sampling=VideoSamplingConfig(fps=1.0, max_frames=5, max_ocr_frames=5),
        temporal_config=CFG,
        temporal_reasoner_config=TemporalReasonerConfig(enabled=False),
    )

    frame1 = tmp_path / "frame_00001.jpg"
    frame2 = tmp_path / "frame_00002.jpg"
    frame1.write_bytes(b"x")
    frame2.write_bytes(b"y")
    probe = VideoProbeInfo(duration_seconds=2.0, has_video=True, has_audio=True)
    loaded = MagicMock()

    with patch("src.analyzers.video.probe_video", return_value=probe), patch(
        "src.media.samplers.extract_frames_at_fps",
        return_value=[frame1, frame2],
    ), patch(
        "src.analyzers.visual.VisualSentimentAnalyzer.load_image",
        return_value=loaded,
    ):
        bundle = analyzer.analyze(video)

    assert bundle.visual is not None
    assert bundle.temporal_context is not None
    assert bundle.temporal_context.window_seconds == 5.0
    assert len(bundle.temporal_context.events) == 2
    assert bundle.frame_evidence[0].timestamp_seconds == 0.0
    # OCR inherited frame timestamp via events.
    ocr_events = [e for e in bundle.temporal_context.events if e.ocr and e.ocr.text]
    assert ocr_events
    assert ocr_events[0].timestamp in {0.0, 1.0}
