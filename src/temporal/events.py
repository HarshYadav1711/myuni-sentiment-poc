"""Build sorted TemporalEvent timeline from frame + speech evidence."""

from __future__ import annotations

from typing import Optional, Sequence

from src.schemas import (
    SentimentEvidence,
    SpeechSegment,
    TemporalEvent,
    TemporalOcrEvidence,
    TemporalSpeechEvidence,
)


def segment_overlaps_time(segment: SpeechSegment, timestamp: float) -> bool:
    """True when ``timestamp`` falls inside [start, end] (inclusive).

    Zero-length segments match only exact start equality.
    """
    start = float(segment.start)
    end = float(segment.end)
    if end < start:
        start, end = end, start
    if end == start:
        return abs(timestamp - start) < 1e-9
    return start <= timestamp <= end


def segments_overlapping_time(
    segments: Sequence[SpeechSegment],
    timestamp: float,
) -> list[SpeechSegment]:
    """Return speech segments whose temporal span overlaps ``timestamp``."""
    return [seg for seg in segments if segment_overlaps_time(seg, timestamp)]


def segments_overlapping_interval(
    segments: Sequence[SpeechSegment],
    start: float,
    end: float,
) -> list[SpeechSegment]:
    """Return segments overlapping [start, end) (last window may use inclusive end)."""
    if end < start:
        start, end = end, start
    out: list[SpeechSegment] = []
    for seg in segments:
        seg_start = float(seg.start)
        seg_end = float(seg.end)
        if seg_end < seg_start:
            seg_start, seg_end = seg_end, seg_start
        # Overlap test for half-open [start, end) with closed segment [s, e].
        if seg_end < start:
            continue
        if seg_start > end:
            continue
        # Exact boundary: segment starting exactly at end belongs to next window,
        # unless this is a degenerate zero-width query.
        if end > start and seg_start >= end:
            continue
        out.append(seg)
    return out


def build_frame_events(
    *,
    timestamps: Sequence[float],
    visuals: Sequence[Optional[SentimentEvidence]],
    ocr_texts: Sequence[Optional[str]],
    ocr_sentiments: Sequence[Optional[SentimentEvidence]],
    speech_segments: Sequence[SpeechSegment] = (),
    speech_sentiments_by_segment: Optional[Sequence[Optional[SentimentEvidence]]] = None,
) -> list[TemporalEvent]:
    """Create one TemporalEvent per frame timestamp (sorted ascending).

    OCR inherits the parent frame timestamp (no artificial OCR clock).
    Speech is attached when a Faster-Whisper segment span overlaps the frame time.
    When multiple segments overlap, the longest-overlap segment is preferred for the
    single optional ``speech`` slot; others remain available via window alignment.
    """
    n = len(timestamps)
    if not (len(visuals) == len(ocr_texts) == len(ocr_sentiments) == n):
        raise ValueError(
            "timestamps, visuals, ocr_texts, and ocr_sentiments must have equal length",
        )

    seg_sentiments: Sequence[Optional[SentimentEvidence]]
    if speech_sentiments_by_segment is None:
        seg_sentiments = [None] * len(speech_segments)
    else:
        if len(speech_sentiments_by_segment) != len(speech_segments):
            raise ValueError(
                "speech_sentiments_by_segment must align with speech_segments",
            )
        seg_sentiments = speech_sentiments_by_segment

    indexed = sorted(enumerate(timestamps), key=lambda pair: (float(pair[1]), pair[0]))
    events: list[TemporalEvent] = []

    for order, (frame_idx, raw_ts) in enumerate(indexed):
        ts = float(raw_ts)
        visual = visuals[frame_idx]
        ocr_text = ocr_texts[frame_idx]
        ocr_sentiment = ocr_sentiments[frame_idx]

        ocr_block: Optional[TemporalOcrEvidence] = None
        if ocr_text or ocr_sentiment is not None:
            ocr_block = TemporalOcrEvidence(text=ocr_text, sentiment=ocr_sentiment)

        overlapping = segments_overlapping_time(speech_segments, ts)
        speech_block: Optional[TemporalSpeechEvidence] = None
        if overlapping:
            # Prefer the segment covering the most time around this timestamp.
            best = max(
                overlapping,
                key=lambda s: (float(s.end) - float(s.start), -(float(s.start))),
            )
            best_idx = next(
                i for i, s in enumerate(speech_segments) if s is best or (
                    float(s.start) == float(best.start)
                    and float(s.end) == float(best.end)
                    and s.text == best.text
                )
            )
            speech_block = TemporalSpeechEvidence(
                text=best.text,
                sentiment=seg_sentiments[best_idx],
                segment_start=float(best.start),
                segment_end=float(best.end),
            )

        events.append(
            TemporalEvent(
                timestamp=ts,
                event_id=f"frame-{frame_idx:04d}",
                visual=visual,
                ocr=ocr_block,
                speech=speech_block,
            ),
        )
        # ``order`` retained for stable sort documentation; events already sorted.
        _ = order

    return events


def sort_events(events: Sequence[TemporalEvent]) -> list[TemporalEvent]:
    """Return events sorted by timestamp, then event_id."""
    return sorted(events, key=lambda e: (float(e.timestamp), e.event_id))
