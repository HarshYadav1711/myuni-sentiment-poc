"""Word-timestamp assignment into fixed TemporalWindows.

Boundary convention (documented):
- Windows covering [0, duration] are half-open [start, end) except the final
  window which is closed [start, end] so the last sample is not dropped.
- Each word is assigned to exactly one window using its midpoint:
      midpoint = (word.start + word.end) / 2
- A word belongs to window i when start_i <= midpoint < end_i (non-last),
  or start_i <= midpoint <= end_i (last window).
- Words with malformed timing (non-finite, end < start after swap, empty text)
  are skipped — never invented into a window.
- Midpoints before the first window or after the last window are skipped.

This avoids duplicating a boundary word across two windows.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from src.schemas import SentimentEvidence, SpeechSegment, SpeechWord
from src.temporal.windows import create_windows


def word_midpoint(word: SpeechWord) -> float:
    """Return the midpoint used for exclusive window assignment."""
    start = float(word.start)
    end = float(word.end)
    if end < start:
        start, end = end, start
    return (start + end) / 2.0


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def normalize_word(word: SpeechWord, *, index: int) -> Optional[SpeechWord]:
    """Return a cleaned word or None when timing/text is unusable."""
    text = (word.text or "").strip()
    if not text:
        return None
    try:
        start = float(word.start)
        end = float(word.end)
    except (TypeError, ValueError):
        return None
    if not (_is_finite(start) and _is_finite(end)):
        return None
    if end < start:
        start, end = end, start
    word_id = word.word_id or f"word-{index:04d}"
    return SpeechWord(start=start, end=end, text=text, word_id=word_id)


def usable_word_timestamps(words: Sequence[SpeechWord]) -> bool:
    """True when at least one normalized word timing is available."""
    for idx, word in enumerate(words):
        if normalize_word(word, index=idx) is not None:
            return True
    return False


def join_words(words: Sequence[SpeechWord]) -> str:
    """Join assigned words chronologically into window-local speech text.

    Faster-Whisper often prefixes tokens with a leading space; we strip each
    token and re-join with single spaces so punctuation attached to words is
    preserved without inventing separators.
    """
    parts = [(w.text or "").strip() for w in words if (w.text or "").strip()]
    return " ".join(parts).strip()


def assign_words_to_windows(
    words: Sequence[SpeechWord],
    *,
    duration_seconds: float,
    window_seconds: float,
) -> list[list[SpeechWord]]:
    """Assign each usable word to exactly one temporal window by midpoint.

    Returns a list aligned with ``create_windows(...)``: one word list per
    window, each list sorted chronologically by (start, end, word_id).
    """
    bounds = create_windows(duration_seconds, window_seconds=window_seconds)
    buckets: list[list[SpeechWord]] = [[] for _ in bounds]

    normalized: list[SpeechWord] = []
    for idx, word in enumerate(words):
        cleaned = normalize_word(word, index=idx)
        if cleaned is not None:
            normalized.append(cleaned)

    for word in normalized:
        mid = word_midpoint(word)
        assigned = False
        for i, (start, end) in enumerate(bounds):
            is_last = i == len(bounds) - 1
            if is_last:
                if start <= mid <= end:
                    buckets[i].append(word)
                    assigned = True
                    break
            else:
                if start <= mid < end:
                    buckets[i].append(word)
                    assigned = True
                    break
        _ = assigned  # words outside [0, duration] are intentionally dropped

    for bucket in buckets:
        bucket.sort(
            key=lambda w: (float(w.start), float(w.end), w.word_id or "", w.text),
        )
    return buckets


def window_speech_units_from_words(
    words: Sequence[SpeechWord],
    *,
    duration_seconds: float,
    window_seconds: float,
    speech_scorer=None,
) -> tuple[list[list[SpeechWord]], list[Optional[SpeechSegment]], list[Optional[SentimentEvidence]]]:
    """Build per-window word lists, synthetic segments, and optional sentiments.

    Synthetic ``SpeechSegment`` spans the first/last assigned word timing and
    carries the window-local joined text. Empty windows yield None segment and
    None sentiment (speech remains unavailable — never neutral).
    """
    buckets = assign_words_to_windows(
        words,
        duration_seconds=duration_seconds,
        window_seconds=window_seconds,
    )
    segments: list[Optional[SpeechSegment]] = []
    sentiments: list[Optional[SentimentEvidence]] = []

    for bucket in buckets:
        text = join_words(bucket)
        if not text:
            segments.append(None)
            sentiments.append(None)
            continue
        seg = SpeechSegment(
            start=float(bucket[0].start),
            end=float(bucket[-1].end),
            text=text,
        )
        segments.append(seg)
        sentiment: Optional[SentimentEvidence] = None
        if speech_scorer is not None:
            try:
                sentiment = speech_scorer(text)
            except Exception:  # noqa: BLE001 — never invent sentiment on failure
                sentiment = None
        sentiments.append(sentiment)

    return buckets, segments, sentiments
