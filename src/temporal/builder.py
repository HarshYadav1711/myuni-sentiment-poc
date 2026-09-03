"""Orchestrate TemporalContext from frame/speech evidence (CPU-only)."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from src.config import DEFAULT_TEMPORAL, TemporalConfig
from src.schemas import (
    SentimentEvidence,
    SpeechAlignmentSource,
    SpeechSegment,
    SpeechWord,
    TemporalContext,
    TemporalEvent,
)
from src.temporal.events import build_frame_events, sort_events
from src.temporal.features import TemporalFeatureExtractor
from src.temporal.speech_words import (
    normalize_word,
    usable_word_timestamps,
    window_speech_units_from_words,
)
from src.temporal.windows import build_temporal_windows


SentimentScorer = Callable[[str], Optional[SentimentEvidence]]


class TemporalContextBuilder:
    """Build structured temporal context parallel to existing sentiment fusion."""

    def __init__(
        self,
        config: TemporalConfig = DEFAULT_TEMPORAL,
        *,
        speech_scorer: Optional[SentimentScorer] = None,
        include_all_events: bool = False,
    ) -> None:
        self.config = config
        self.speech_scorer = speech_scorer
        self.include_all_events = include_all_events
        self._features = TemporalFeatureExtractor(config)

    def build(
        self,
        *,
        duration_seconds: float,
        timestamps: Sequence[float],
        visuals: Sequence[Optional[SentimentEvidence]],
        ocr_texts: Sequence[Optional[str]],
        ocr_sentiments: Sequence[Optional[SentimentEvidence]],
        speech_segments: Sequence[SpeechSegment] = (),
        speech_sentiments_by_segment: Optional[Sequence[Optional[SentimentEvidence]]] = None,
        speech_words: Sequence[SpeechWord] = (),
    ) -> TemporalContext:
        """Assemble events → windows → features.

        ``duration_seconds`` should come from existing ffprobe (do not re-probe).
        Frame timestamps should come from the sampler (index/fps or seek times).

        When usable ``speech_words`` are present, window speech text/sentiment
        is derived from word midpoints (``speech_alignment_source=word_timestamps``).
        Otherwise the legacy segment-overlap path is used (``segment_fallback``).
        """
        scored_segments = self._resolve_speech_sentiments(
            speech_segments,
            speech_sentiments_by_segment,
        )

        events = build_frame_events(
            timestamps=timestamps,
            visuals=visuals,
            ocr_texts=ocr_texts,
            ocr_sentiments=ocr_sentiments,
            speech_segments=speech_segments,
            speech_sentiments_by_segment=scored_segments,
        )
        events = sort_events(events)

        alignment: Optional[SpeechAlignmentSource] = None
        speech_word_count: Optional[int] = None
        retained_words: list[SpeechWord] = []
        window_kwargs: dict = {
            "duration_seconds": float(duration_seconds),
            "events": events,
            "speech_segments": speech_segments,
            "speech_sentiments_by_segment": scored_segments,
            "config": self.config,
        }

        if usable_word_timestamps(speech_words):
            retained_words = [
                cleaned
                for idx, word in enumerate(speech_words)
                if (cleaned := normalize_word(word, index=idx)) is not None
            ]
            _buckets, win_segs, win_sents = window_speech_units_from_words(
                retained_words,
                duration_seconds=float(duration_seconds),
                window_seconds=float(self.config.window_seconds),
                speech_scorer=self.speech_scorer,
            )
            window_kwargs["window_speech_segments"] = win_segs
            window_kwargs["window_speech_sentiments"] = win_sents
            alignment = "word_timestamps"
            speech_word_count = sum(len(b) for b in _buckets)
            # Prefer window-local speech on frame events so events do not
            # misleadingly repeat a single long Whisper segment.
            events = _attach_window_speech_to_events(
                events,
                windows_bounds=[(s.start, s.end) if s else None for s in win_segs],
                window_segments=win_segs,
                window_sentiments=win_sents,
                window_seconds=float(self.config.window_seconds),
                duration_seconds=float(duration_seconds),
            )
            window_kwargs["events"] = events
        elif speech_segments:
            alignment = "segment_fallback"
        elif speech_words:
            # Words present but unusable timings → fall back (no segment text).
            alignment = "segment_fallback"

        windows = build_temporal_windows(**window_kwargs)
        features = self._features.extract(windows)

        events_total = len(events)
        events_out, truncated = self._maybe_truncate_events(events)

        return TemporalContext(
            window_seconds=float(self.config.window_seconds),
            duration_seconds=float(duration_seconds) if duration_seconds > 0 else None,
            events=events_out,
            events_truncated=truncated,
            events_total=events_total,
            windows=windows,
            features=features,
            speech_alignment_source=alignment,
            speech_word_count=speech_word_count,
            speech_words=retained_words,
            source_speech_segments=list(speech_segments),
        )

    def _resolve_speech_sentiments(
        self,
        segments: Sequence[SpeechSegment],
        provided: Optional[Sequence[Optional[SentimentEvidence]]],
    ) -> list[Optional[SentimentEvidence]]:
        if provided is not None:
            out = list(provided)
            if len(out) < len(segments):
                out.extend([None] * (len(segments) - len(out)))
            return out[: len(segments)]

        if self.speech_scorer is None:
            return [None] * len(segments)

        scored: list[Optional[SentimentEvidence]] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                scored.append(None)
                continue
            try:
                scored.append(self.speech_scorer(text))
            except Exception:  # noqa: BLE001 — never invent sentiment on failure
                scored.append(None)
        return scored

    def _maybe_truncate_events(
        self,
        events: Sequence[TemporalEvent],
    ) -> tuple[list[TemporalEvent], bool]:
        if self.include_all_events or len(events) <= self.config.max_events_in_output:
            return list(events), False
        return list(events[: self.config.max_events_in_output]), True


def _attach_window_speech_to_events(
    events: Sequence[TemporalEvent],
    *,
    windows_bounds: Sequence[Optional[tuple[float, float]]],
    window_segments: Sequence[Optional[SpeechSegment]],
    window_sentiments: Sequence[Optional[SentimentEvidence]],
    window_seconds: float,
    duration_seconds: float,
) -> list[TemporalEvent]:
    """Replace event speech with the window-local unit containing the frame time.

    Uses the same half-open / closed-last window convention as word assignment.
    """
    from src.schemas import TemporalSpeechEvidence
    from src.temporal.windows import create_windows

    bounds = create_windows(duration_seconds, window_seconds=window_seconds)
    out: list[TemporalEvent] = []
    for event in events:
        ts = float(event.timestamp)
        win_idx: Optional[int] = None
        for i, (start, end) in enumerate(bounds):
            is_last = i == len(bounds) - 1
            if is_last:
                if start <= ts <= end:
                    win_idx = i
                    break
            elif start <= ts < end:
                win_idx = i
                break
        speech = None
        if win_idx is not None and win_idx < len(window_segments):
            seg = window_segments[win_idx]
            sent = window_sentiments[win_idx] if win_idx < len(window_sentiments) else None
            if seg is not None:
                speech = TemporalSpeechEvidence(
                    text=seg.text,
                    sentiment=sent,
                    segment_start=float(seg.start),
                    segment_end=float(seg.end),
                )
        out.append(
            event.model_copy(update={"speech": speech}),
        )
        _ = windows_bounds
    return out


def build_temporal_context(
    *,
    duration_seconds: float,
    timestamps: Sequence[float],
    visuals: Sequence[Optional[SentimentEvidence]],
    ocr_texts: Sequence[Optional[str]],
    ocr_sentiments: Sequence[Optional[SentimentEvidence]],
    speech_segments: Sequence[SpeechSegment] = (),
    speech_sentiments_by_segment: Optional[Sequence[Optional[SentimentEvidence]]] = None,
    speech_words: Sequence[SpeechWord] = (),
    config: TemporalConfig = DEFAULT_TEMPORAL,
    speech_scorer: Optional[SentimentScorer] = None,
    include_all_events: bool = False,
) -> TemporalContext:
    """Convenience wrapper around :class:`TemporalContextBuilder`."""
    builder = TemporalContextBuilder(
        config,
        speech_scorer=speech_scorer,
        include_all_events=include_all_events,
    )
    return builder.build(
        duration_seconds=duration_seconds,
        timestamps=timestamps,
        visuals=visuals,
        ocr_texts=ocr_texts,
        ocr_sentiments=ocr_sentiments,
        speech_segments=speech_segments,
        speech_sentiments_by_segment=speech_sentiments_by_segment,
        speech_words=speech_words,
    )
