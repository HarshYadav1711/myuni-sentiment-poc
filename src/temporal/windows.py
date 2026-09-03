"""Fixed temporal windows and transparent per-window aggregation."""

from __future__ import annotations

import math
from typing import Optional, Sequence

from src.config import DEFAULT_TEMPORAL, TemporalConfig
from src.schemas import (
    SentimentEvidence,
    SpeechSegment,
    TemporalEvent,
    TemporalWindow,
)
from src.temporal.aggregation import (
    combine_modality_probabilities,
    label_from_probabilities,
    length_weighted_probability_distribution,
    max_class_probability,
    mean_probability_distribution,
)
from src.temporal.events import segments_overlapping_interval


def create_windows(
    duration_seconds: float,
    *,
    window_seconds: float,
) -> list[tuple[float, float]]:
    """Create fixed [start, end) windows covering ``[0, duration]``.

    The final window may be shorter than ``window_seconds``. Returns a single
    degenerate window ``(0, 0)`` when duration is non-positive so callers can
    still attach zero-coverage features without inventing evidence.
    """
    if duration_seconds <= 0 or window_seconds <= 0:
        return [(0.0, 0.0)]

    n = max(1, int(math.ceil(duration_seconds / window_seconds)))
    windows: list[tuple[float, float]] = []
    for i in range(n):
        start = i * window_seconds
        end = min(duration_seconds, (i + 1) * window_seconds)
        if end <= start and i > 0:
            break
        # Guarantee progress on tiny durations.
        if end <= start:
            end = duration_seconds if duration_seconds > start else start
        windows.append((float(start), float(end)))
    return windows or [(0.0, float(duration_seconds))]


def _event_in_window(timestamp: float, start: float, end: float, *, is_last: bool) -> bool:
    if is_last:
        return start <= timestamp <= end
    return start <= timestamp < end


def build_temporal_windows(
    *,
    duration_seconds: float,
    events: Sequence[TemporalEvent],
    speech_segments: Sequence[SpeechSegment] = (),
    speech_sentiments_by_segment: Optional[Sequence[Optional[SentimentEvidence]]] = None,
    window_speech_segments: Optional[Sequence[Optional[SpeechSegment]]] = None,
    window_speech_sentiments: Optional[Sequence[Optional[SentimentEvidence]]] = None,
    config: TemporalConfig = DEFAULT_TEMPORAL,
) -> list[TemporalWindow]:
    """Bucket events + overlapping speech into fixed temporal windows.

    Aggregation rules (documented):
    - visual: unweighted mean of probability distributions among frames in window
    - speech (default / segment_fallback): length-weighted mean of per-segment
      sentiment probabilities (weight = max(1, len(text))); segments without
      sentiment are omitted from probability aggregation but still listed in
      ``speech_segments``
    - speech (word_timestamps): when ``window_speech_segments`` is provided,
      each window uses its own pre-built local speech unit (exactly one
      segment or none) instead of segment-overlap duplication
    - OCR: unweighted mean of OCR sentiment probabilities; texts preserved raw

    Missing modalities are excluded — never filled with neutral/zero.
    """
    bounds = create_windows(duration_seconds, window_seconds=config.window_seconds)
    seg_sents = list(speech_sentiments_by_segment or [None] * len(speech_segments))
    if len(seg_sents) < len(speech_segments):
        seg_sents.extend([None] * (len(speech_segments) - len(seg_sents)))

    use_window_speech = window_speech_segments is not None
    win_segs = list(window_speech_segments or [])
    win_sents = list(window_speech_sentiments or [])
    if use_window_speech:
        while len(win_segs) < len(bounds):
            win_segs.append(None)
        while len(win_sents) < len(bounds):
            win_sents.append(None)

    windows: list[TemporalWindow] = []
    for idx, (start, end) in enumerate(bounds):
        is_last = idx == len(bounds) - 1
        visual_ev: list[SentimentEvidence] = []
        ocr_texts: list[str] = []
        ocr_sents: list[SentimentEvidence] = []

        for event in events:
            if not _event_in_window(float(event.timestamp), start, end, is_last=is_last):
                continue
            if event.visual is not None:
                visual_ev.append(event.visual)
            if event.ocr is not None:
                if event.ocr.text:
                    ocr_texts.append(event.ocr.text)
                if event.ocr.sentiment is not None:
                    ocr_sents.append(event.ocr.sentiment)

        if use_window_speech:
            local_seg = win_segs[idx]
            overlapping = [local_seg] if local_seg is not None else []
            speech_ev: list[SentimentEvidence] = []
            speech_lengths: list[float] = []
            local_sent = win_sents[idx] if idx < len(win_sents) else None
            if local_sent is not None and local_seg is not None:
                speech_ev.append(local_sent)
                speech_lengths.append(float(max(1, len(local_seg.text or ""))))
        else:
            # Inclusive end for last window so trailing speech is not dropped.
            if is_last:
                overlapping = [
                    seg
                    for seg in speech_segments
                    if float(seg.end) >= start and float(seg.start) <= end
                ]
            else:
                overlapping = segments_overlapping_interval(speech_segments, start, end)

            speech_ev = []
            speech_lengths = []
            for seg in overlapping:
                try:
                    seg_idx = next(
                        i
                        for i, s in enumerate(speech_segments)
                        if float(s.start) == float(seg.start)
                        and float(s.end) == float(seg.end)
                        and s.text == seg.text
                    )
                except StopIteration:
                    continue
                sent = seg_sents[seg_idx]
                if sent is not None:
                    speech_ev.append(sent)
                    speech_lengths.append(float(max(1, len(seg.text or ""))))

        visual_probs = mean_probability_distribution(visual_ev)
        ocr_probs = mean_probability_distribution(ocr_sents)
        speech_probs = length_weighted_probability_distribution(speech_ev, speech_lengths)

        available: list[str] = []
        modality_probs: dict[str, dict[str, float]] = {}
        if visual_probs is not None:
            available.append("visual")
            modality_probs["visual"] = visual_probs
        if speech_probs is not None:
            available.append("speech")
            modality_probs["speech"] = speech_probs
        elif overlapping:
            # Speech text present but no per-segment sentiment — still mark coverage.
            available.append("speech")
        if ocr_probs is not None:
            available.append("ocr")
            modality_probs["ocr"] = ocr_probs
        elif ocr_texts:
            available.append("ocr")

        combined = combine_modality_probabilities(modality_probs)
        usable = False
        dominant = None
        neg_prob = None
        if combined is not None:
            neg_prob = float(combined.get("negative", 0.0))
            if max_class_probability(combined) >= config.min_usable_evidence:
                usable = True
                dominant = label_from_probabilities(
                    combined,
                    negative_threshold=config.negative_prob_threshold,
                    positive_threshold=config.positive_prob_threshold,
                )

        windows.append(
            TemporalWindow(
                start=float(start),
                end=float(end),
                index=idx,
                visual_evidence=visual_ev,
                speech_segments=list(overlapping),
                speech_sentiments=speech_ev,
                ocr_texts=list(dict.fromkeys(ocr_texts)),
                ocr_sentiments=ocr_sents,
                visual_probabilities=visual_probs,
                speech_probabilities=speech_probs,
                ocr_probabilities=ocr_probs,
                available_modalities=available,
                usable=usable,
                dominant_label=dominant,
                negative_probability=neg_prob,
            ),
        )

    return windows
