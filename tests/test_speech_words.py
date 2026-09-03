"""Unit tests for Phase 3A word-timestamp temporal speech (no live models)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyzers.audio import AudioAnalyzer
from src.config import TemporalConfig
from src.schemas import (
    SentimentEvidence,
    SpeechAnalysisResult,
    SpeechSegment,
    SpeechWord,
)
from src.temporal.builder import build_temporal_context
from src.temporal.prompt import build_evidence_payload, collect_valid_evidence_ids
from src.temporal.speech_words import (
    assign_words_to_windows,
    join_words,
    usable_word_timestamps,
    word_midpoint,
    window_speech_units_from_words,
)


def _sent(
    label: str,
    *,
    neg: float,
    neu: float = 0.1,
    pos: float = 0.1,
) -> SentimentEvidence:
    probs = {"negative": neg, "neutral": neu, "positive": pos}
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}
    score = probs["positive"] - probs["negative"]
    return SentimentEvidence(
        label=label,  # type: ignore[arg-type]
        score=score,
        confidence=max(probs.values()),
        probabilities=probs,
        model="mock-roberta",
    )


def _words_phrase(start: float, texts: list[str], *, dur: float = 0.35) -> list[SpeechWord]:
    out: list[SpeechWord] = []
    t = start
    for i, text in enumerate(texts):
        out.append(SpeechWord(start=t, end=t + dur, text=text, word_id=f"w-{start:.1f}-{i}"))
        t += dur + 0.05
    return out


def test_word_midpoint_calculation() -> None:
    w = SpeechWord(start=1.0, end=3.0, text="hello")
    assert word_midpoint(w) == 2.0
    swapped = SpeechWord(start=3.0, end=1.0, text="x")
    assert word_midpoint(swapped) == 2.0


def test_first_window_assignment() -> None:
    words = [SpeechWord(start=0.2, end=0.6, text="Today")]
    buckets = assign_words_to_windows(words, duration_seconds=20.0, window_seconds=5.0)
    assert [w.text for w in buckets[0]] == ["Today"]
    assert buckets[1] == []


def test_exact_boundary_behavior() -> None:
    """Midpoint exactly at end of non-last window belongs to the NEXT window.

    Window 0 is [0, 5), window 1 is [5, 10). Midpoint 5.0 → window 1.
    """
    # start=4.5 end=5.5 → midpoint 5.0
    words = [SpeechWord(start=4.5, end=5.5, text="boundary")]
    buckets = assign_words_to_windows(words, duration_seconds=20.0, window_seconds=5.0)
    assert buckets[0] == []
    assert [w.text for w in buckets[1]] == ["boundary"]


def test_word_crossing_boundary_assigned_once() -> None:
    words = [SpeechWord(start=4.8, end=5.4, text="crossing")]  # mid=5.1 → win 1
    buckets = assign_words_to_windows(words, duration_seconds=20.0, window_seconds=5.0)
    assigned = sum(1 for b in buckets for _ in b)
    assert assigned == 1
    assert [w.text for w in buckets[1]] == ["crossing"]
    assert buckets[0] == []


def test_final_partial_window() -> None:
    words = [SpeechWord(start=20.1, end=20.5, text="end")]
    buckets = assign_words_to_windows(words, duration_seconds=22.7, window_seconds=5.0)
    assert len(buckets) == 5
    assert [w.text for w in buckets[4]] == ["end"]


def test_chronological_word_ordering() -> None:
    words = [
        SpeechWord(start=1.5, end=1.8, text="B"),
        SpeechWord(start=0.5, end=0.8, text="A"),
        SpeechWord(start=2.0, end=2.3, text="C"),
    ]
    buckets = assign_words_to_windows(words, duration_seconds=10.0, window_seconds=5.0)
    assert [w.text for w in buckets[0]] == ["A", "B", "C"]


def test_text_construction_per_window() -> None:
    words = _words_phrase(0.5, ["Today", "has", "been"]) + _words_phrase(
        5.5,
        ["three", "assignments"],
    )
    buckets = assign_words_to_windows(words, duration_seconds=20.0, window_seconds=5.0)
    assert join_words(buckets[0]) == "Today has been"
    assert join_words(buckets[1]) == "three assignments"


def test_no_duplicate_and_no_missing_assignment() -> None:
    words = [
        SpeechWord(start=0.1, end=0.3, text="a"),
        SpeechWord(start=5.1, end=5.3, text="b"),
        SpeechWord(start=10.1, end=10.3, text="c"),
        SpeechWord(start=15.1, end=15.3, text="d"),
    ]
    buckets = assign_words_to_windows(words, duration_seconds=20.0, window_seconds=5.0)
    flat = [w.text for b in buckets for w in b]
    assert flat == ["a", "b", "c", "d"]
    assert len(flat) == len(set(flat))


def test_empty_speech_window() -> None:
    words = [SpeechWord(start=0.2, end=0.5, text="only")]
    buckets = assign_words_to_windows(words, duration_seconds=20.0, window_seconds=5.0)
    assert join_words(buckets[2]) == ""
    assert buckets[1] == []


def test_malformed_and_empty_words_skipped() -> None:
    words = [
        SpeechWord(start=float("nan"), end=1.0, text="bad"),
        SpeechWord(start=0.2, end=0.4, text="  "),
        SpeechWord(start=0.5, end=0.7, text="ok"),
    ]
    assert usable_word_timestamps(words) is True
    buckets = assign_words_to_windows(words, duration_seconds=10.0, window_seconds=5.0)
    assert [w.text for w in buckets[0]] == ["ok"]


def test_long_single_segment_produces_multiple_window_texts() -> None:
    """Controlled fixture: ONE Whisper segment 0.4→21.7 with word timings."""
    phrases = [
        (0.5, ["Today", "has", "been", "pretty", "normal"]),
        (5.5, ["I", "have", "got", "three", "assignments", "due", "tomorrow"]),
        (10.5, ["I", "am", "getting", "really", "exhausted", "trying", "to", "finish", "everything"]),
        (15.5, ["it's", "honestly", "becoming", "pretty", "overwhelming"]),
    ]
    words: list[SpeechWord] = []
    for start, toks in phrases:
        words.extend(_words_phrase(start, toks))

    long_seg = SpeechSegment(
        start=0.4,
        end=21.7,
        text=" ".join(w.text for w in words),
    )

    scored_calls: list[str] = []

    def scorer(text: str) -> SentimentEvidence:
        scored_calls.append(text)
        # Distinct mocked sentiments per call order — not real RoBERTa.
        neg = 0.2 + 0.15 * len(scored_calls)
        return _sent("negative" if neg >= 0.45 else "neutral", neg=min(neg, 0.9), neu=0.3, pos=0.1)

    ctx = build_temporal_context(
        duration_seconds=22.7,
        timestamps=[0.0, 2.0, 6.0, 11.0, 16.0, 21.0],
        visuals=[_sent("negative", neg=0.8)] * 6,
        ocr_texts=[None] * 6,
        ocr_sentiments=[None] * 6,
        speech_segments=[long_seg],
        speech_words=words,
        speech_scorer=scorer,
        config=TemporalConfig(window_seconds=5.0),
    )

    assert ctx.speech_alignment_source == "word_timestamps"
    assert ctx.speech_word_count == len(words)

    texts = [
        (w.speech_segments[0].text if w.speech_segments else "")
        for w in ctx.windows
    ]
    assert texts[0] != texts[1]
    assert texts[1] != texts[2]
    assert "Today" in texts[0]
    assert "assignments" in texts[1]
    assert "exhausted" in texts[2]
    assert "overwhelming" in texts[3]

    # Each non-empty window got its own sentiment score (not one shared segment score).
    speech_negs = [
        (w.speech_probabilities or {}).get("negative")
        for w in ctx.windows
        if w.speech_probabilities
    ]
    assert len(set(round(x, 5) for x in speech_negs if x is not None)) >= 2
    assert len(scored_calls) >= 4


def test_multiple_whisper_segments_also_work_with_words() -> None:
    words = _words_phrase(0.5, ["hello", "there"]) + _words_phrase(6.0, ["goodbye", "now"])
    segs = [
        SpeechSegment(start=0.4, end=2.0, text="hello there"),
        SpeechSegment(start=5.8, end=7.0, text="goodbye now"),
    ]
    ctx = build_temporal_context(
        duration_seconds=10.0,
        timestamps=[0.5, 6.5],
        visuals=[_sent("neutral", neg=0.2, neu=0.6, pos=0.2)] * 2,
        ocr_texts=[None, None],
        ocr_sentiments=[None, None],
        speech_segments=segs,
        speech_words=words,
        speech_scorer=lambda t: _sent("neutral", neg=0.2, neu=0.7, pos=0.1),
    )
    assert ctx.speech_alignment_source == "word_timestamps"
    assert ctx.windows[0].speech_segments[0].text == "hello there"
    assert ctx.windows[1].speech_segments[0].text == "goodbye now"


def test_missing_word_timestamps_falls_back_to_segments() -> None:
    seg = SpeechSegment(start=0.0, end=4.0, text="this is awful and upsetting")
    ctx = build_temporal_context(
        duration_seconds=5.0,
        timestamps=[1.0],
        visuals=[_sent("positive", neg=0.1, neu=0.1, pos=0.8)],
        ocr_texts=[None],
        ocr_sentiments=[None],
        speech_segments=[seg],
        speech_words=[],
        speech_scorer=lambda t: _sent("negative", neg=0.8),
    )
    assert ctx.speech_alignment_source == "segment_fallback"
    assert ctx.windows[0].speech_segments
    assert ctx.windows[0].speech_probabilities is not None


def test_window_local_sentiment_plumbing() -> None:
    words = _words_phrase(0.5, ["nice", "day"]) + _words_phrase(6.0, ["terrible", "news"])

    def scorer(text: str) -> SentimentEvidence:
        if "terrible" in text:
            return _sent("negative", neg=0.85)
        return _sent("positive", neg=0.05, neu=0.1, pos=0.85)

    _b, segs, sents = window_speech_units_from_words(
        words,
        duration_seconds=10.0,
        window_seconds=5.0,
        speech_scorer=scorer,
    )
    assert segs[0] is not None and sents[0] is not None
    assert sents[0].label == "positive"
    assert segs[1] is not None and sents[1] is not None
    assert sents[1].label == "negative"


def test_full_transcript_sentiment_and_global_fusion_unchanged_contract() -> None:
    """AudioAnalyzer still scores the full transcript once; words are additive."""
    media = Path("dummy.wav")
    text = MagicMock()
    full = _sent("negative", neg=0.55, neu=0.3, pos=0.15)
    text.analyze.return_value = full

    analyzer = AudioAnalyzer(text_analyzer=text)
    analyzer._whisper_model = MagicMock()

    word_objs = [
        SimpleNamespace(start=0.5, end=0.8, word="Today"),
        SimpleNamespace(start=5.5, end=5.8, word="exhausted"),
    ]
    seg = SimpleNamespace(
        start=0.4,
        end=6.0,
        text=" Today exhausted",
        words=word_objs,
    )
    info = SimpleNamespace(language="en", duration=6.0)
    analyzer._whisper_model.transcribe.return_value = (iter([seg]), info)

    with patch("src.analyzers.audio.extract_audio_wav", return_value=Path("x.wav")), patch(
        "src.analyzers.audio.AudioAnalyzer.validate_media_path",
        return_value=media,
    ):
        result = analyzer.analyze(media, word_timestamps=True)

    assert result.transcript == "Today exhausted"
    assert result.sentiment is not None
    assert result.sentiment.label == "negative"
    assert abs(result.sentiment.probabilities["negative"] - 0.55) < 1e-6
    assert len(result.words) == 2
    text.analyze.assert_called_once_with("Today exhausted")
    # Ensure Whisper was called once with word_timestamps=True
    kwargs = analyzer._whisper_model.transcribe.call_args.kwargs
    assert kwargs.get("word_timestamps") is True


def test_audio_default_analyze_does_not_request_word_timestamps() -> None:
    media = Path("dummy.wav")
    analyzer = AudioAnalyzer()
    analyzer._whisper_model = MagicMock()
    seg = SimpleNamespace(start=0.0, end=1.0, text=" hi", words=None)
    info = SimpleNamespace(language="en", duration=1.0)
    analyzer._whisper_model.transcribe.return_value = (iter([seg]), info)

    with patch("src.analyzers.audio.extract_audio_wav", return_value=Path("x.wav")), patch(
        "src.analyzers.audio.AudioAnalyzer.validate_media_path",
        return_value=media,
    ):
        result = analyzer.analyze(media)

    assert result.words == []
    assert analyzer._whisper_model.transcribe.call_args.kwargs.get("word_timestamps") is False


def test_evidence_ids_valid_for_word_alignment() -> None:
    words = _words_phrase(0.5, ["hello"]) + _words_phrase(6.0, ["world"])
    ctx = build_temporal_context(
        duration_seconds=10.0,
        timestamps=[0.5, 6.5],
        visuals=[_sent("neutral", neg=0.2, neu=0.6, pos=0.2)] * 2,
        ocr_texts=[None, None],
        ocr_sentiments=[None, None],
        speech_segments=[SpeechSegment(start=0.4, end=7.0, text="hello world")],
        speech_words=words,
        speech_scorer=lambda t: _sent("neutral", neg=0.2, neu=0.7, pos=0.1),
    )
    payload = build_evidence_payload(ctx)
    ids = set(collect_valid_evidence_ids(payload))
    assert "speech-window-0" in ids
    assert "speech-window-1" in ids
    assert payload["speech_alignment_source"] == "word_timestamps"


def test_reasoner_payload_accepts_speech_window_ids() -> None:
    from src.temporal.parse import parse_reasoning_result

    words = _words_phrase(0.5, ["hello"])
    ctx = build_temporal_context(
        duration_seconds=5.0,
        timestamps=[0.5],
        visuals=[_sent("neutral", neg=0.2, neu=0.6, pos=0.2)],
        ocr_texts=[None],
        ocr_sentiments=[None],
        speech_segments=[SpeechSegment(start=0.4, end=1.0, text="hello")],
        speech_words=words,
        speech_scorer=lambda t: _sent("neutral", neg=0.2, neu=0.7, pos=0.1),
    )
    payload = build_evidence_payload(ctx)
    valid = set(payload["valid_evidence_ids"])
    raw = {
        "summary": "neutral greeting",
        "trajectory_explanation": "stable neutral as given",
        "cross_modal_context": {
            "consistency": "high",
            "conflicts_detected": False,
            "description": "aligned",
        },
        "important_transitions": [],
        "context_type": "uncertain",
        "evidence": [
            {"evidence_id": "speech-window-0", "explanation": "local greeting text"},
            {"evidence_id": "window-0", "explanation": "only window"},
        ],
        "uncertainties": [],
        "confidence": 0.5,
        "status": "ok",
    }
    import json

    result = parse_reasoning_result(
        json.dumps(raw),
        valid_evidence_ids=valid,
        valid_window_ranges=[(0.0, 5.0)],
    )
    assert result.status == "ok"
    assert {e.evidence_id for e in result.evidence} <= valid
