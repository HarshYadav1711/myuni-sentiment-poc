"""Deterministic TemporalFeatureExtractor (no LLM).

Each feature formula is documented in the corresponding method docstring.
"""

from __future__ import annotations

from typing import Optional, Sequence

from src.config import DEFAULT_TEMPORAL, TemporalConfig
from src.schemas import (
    AgreementLevel,
    CrossModalConflict,
    EvidenceCoverage,
    SentimentEvidence,
    SentimentLabel,
    SuddenNegativeChange,
    StrongestNegativeWindow,
    TemporalFeatures,
    TemporalWindow,
    TrajectoryLabel,
)
from src.temporal.aggregation import (
    score_from_probabilities,
)


def _linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Ordinary least-squares slope for equal-length x/y sequences."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 0:
        return 0.0
    numer = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return numer / denom


def _normalized_temporal_positions(centers: Sequence[float]) -> list[float]:
    """Map usable-window centers to [0, 1] preserving relative spacing.

    first center → 0.0, last center → 1.0. Single center → [0.0] (no span).
    Identical centers → all 0.0 (degenerate; slope will be 0).
    """
    if not centers:
        return []
    if len(centers) == 1:
        return [0.0]
    first = float(centers[0])
    last = float(centers[-1])
    span = last - first
    if span <= 0:
        return [0.0] * len(centers)
    return [(float(c) - first) / span for c in centers]


class TemporalFeatureExtractor:
    """Extract explainable temporal features from windowed evidence."""

    def __init__(self, config: TemporalConfig = DEFAULT_TEMPORAL) -> None:
        self.config = config

    def extract(self, windows: Sequence[TemporalWindow]) -> TemporalFeatures:
        usable = [w for w in windows if w.usable]
        run_count, run_seconds = self.longest_negative_run(usable)
        return TemporalFeatures(
            trajectory=self.trajectory(usable),
            negative_persistence=self.negative_persistence(usable),
            longest_negative_run=run_count,
            longest_negative_run_seconds=run_seconds,
            strongest_negative_window=self.strongest_negative_window(usable),
            sudden_negative_change=self.sudden_negative_change(usable),
            cross_modal_agreement=self.cross_modal_agreement(windows),
            cross_modal_conflicts=self.cross_modal_conflicts(windows),
            evidence_coverage=self.evidence_coverage(windows),
        )

    def trajectory(self, usable: Sequence[TemporalWindow]) -> TrajectoryLabel:
        """Classify sentiment trajectory across usable windows.

        Formula:
        1. Require >= 1 usable window with dominant labels; else
           ``insufficient_evidence``.
        2. With exactly one usable window: no OLS is computed. Return
           ``stable_{label}`` from that window's dominant label (or
           ``insufficient_evidence`` if label missing).
        3. With >= 2 usable windows:
           a. centers_i = (start_i + end_i) / 2
           b. x_i = normalized position of centers_i onto [0, 1]
              (first usable center → 0.0, last → 1.0; intermediates keep
              relative spacing). Slope is therefore ≈ change in P(negative)
              over the observed usable timeline, not per raw second.
           c. y_i = negative_probability_i (0 if missing)
           d. slope = OLS(y ~ x)
           e. Compare |slope| to ``trajectory_slope_threshold``.
        4. Mapping (slope checked before unanimous-label stables):
           - slope >= thr → increasing_negative
           - slope <= -thr → decreasing_negative
           - all labels positive and |slope| < thr → stable_positive
           - all labels negative and |slope| < thr → stable_negative
           - all labels neutral and |slope| < thr → stable_neutral
           - otherwise → mixed
        """
        if not usable:
            return "insufficient_evidence"

        labels = [w.dominant_label for w in usable if w.dominant_label is not None]
        if not labels:
            return "insufficient_evidence"

        # Single usable window: no fabricated regression.
        if len(usable) == 1:
            only = labels[0]
            if only == "positive":
                return "stable_positive"
            if only == "negative":
                return "stable_negative"
            if only == "neutral":
                return "stable_neutral"
            return "insufficient_evidence"

        centers = [(w.start + w.end) / 2.0 for w in usable]
        xs = _normalized_temporal_positions(centers)
        ys = [float(w.negative_probability or 0.0) for w in usable]
        slope = _linear_slope(xs, ys)
        thr = self.config.trajectory_slope_threshold

        unique = set(labels)
        if slope >= thr:
            return "increasing_negative"
        if slope <= -thr:
            return "decreasing_negative"
        if unique == {"positive"}:
            return "stable_positive"
        if unique == {"negative"}:
            return "stable_negative"
        if unique == {"neutral"}:
            return "stable_neutral"
        return "mixed"

    def negative_persistence(self, usable: Sequence[TemporalWindow]) -> Optional[float]:
        """Fraction of usable windows that are meaningfully negative.

        denominator = count(usable windows)
        numerator = count(usable windows where dominant_label == negative
                          OR negative_probability >= negative_prob_threshold)
        Returns None when denominator is 0 (not 0.0 pretending full coverage).
        """
        if not usable:
            return None
        thr = self.config.negative_prob_threshold
        neg_count = 0
        for w in usable:
            neg_p = float(w.negative_probability or 0.0)
            if w.dominant_label == "negative" or neg_p >= thr:
                neg_count += 1
        return neg_count / float(len(usable))

    def longest_negative_run(
        self,
        usable: Sequence[TemporalWindow],
    ) -> tuple[Optional[int], Optional[float]]:
        """Longest consecutive usable-window run classified as meaningfully negative.

        Returns (window_count, approximate_duration_seconds).
        Duration sums (end-start) of windows in the longest run.
        """
        if not usable:
            return None, None
        thr = self.config.negative_prob_threshold

        def is_neg(w: TemporalWindow) -> bool:
            neg_p = float(w.negative_probability or 0.0)
            return w.dominant_label == "negative" or neg_p >= thr

        best_len = 0
        best_dur = 0.0
        cur_len = 0
        cur_dur = 0.0
        for w in usable:
            if is_neg(w):
                cur_len += 1
                cur_dur += max(0.0, float(w.end) - float(w.start))
                if cur_len > best_len:
                    best_len = cur_len
                    best_dur = cur_dur
            else:
                cur_len = 0
                cur_dur = 0.0
        if best_len <= 0:
            return 0, 0.0
        return best_len, best_dur

    def strongest_negative_window(
        self,
        usable: Sequence[TemporalWindow],
    ) -> Optional[StrongestNegativeWindow]:
        """Usable window with maximum negative_probability among meaningfully negative ones.

        A window qualifies when ``dominant_label == negative`` OR
        ``negative_probability >= negative_prob_threshold`` (same rule as
        negative persistence / longest negative run).
        """
        thr = self.config.negative_prob_threshold
        candidates = [
            w
            for w in usable
            if w.negative_probability is not None
            and (
                w.dominant_label == "negative"
                or float(w.negative_probability) >= thr
            )
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda w: float(w.negative_probability or 0.0))
        return StrongestNegativeWindow(
            start=float(best.start),
            end=float(best.end),
            score=float(best.negative_probability or 0.0),
            index=int(best.index),
        )

    def sudden_negative_change(
        self,
        usable: Sequence[TemporalWindow],
    ) -> SuddenNegativeChange:
        """Detect max ΔP(negative) between consecutive usable windows.

        detected when max(neg[i+1] - neg[i]) >= sudden_negative_delta.
        Reports the first pair achieving the maximum delta.
        """
        if len(usable) < 2:
            return SuddenNegativeChange(detected=False)

        best_delta = 0.0
        best_pair: Optional[tuple[TemporalWindow, TemporalWindow]] = None
        for a, b in zip(usable, usable[1:]):
            na = float(a.negative_probability or 0.0)
            nb = float(b.negative_probability or 0.0)
            delta = nb - na
            if delta > best_delta:
                best_delta = delta
                best_pair = (a, b)

        if best_pair is None or best_delta < self.config.sudden_negative_delta:
            return SuddenNegativeChange(detected=False, delta=best_delta if best_pair else None)

        a, b = best_pair
        return SuddenNegativeChange(
            detected=True,
            from_window=int(a.index),
            to_window=int(b.index),
            from_start=float(a.start),
            to_start=float(b.start),
            delta=float(best_delta),
        )

    def cross_modal_agreement(self, windows: Sequence[TemporalWindow]) -> AgreementLevel:
        """Categorical agreement among available modalities across windows.

        Per window with >= 2 modalities that have probability distributions:
          - Derive each modality label via argmax of its probability map.
          - agreement_score = fraction of modality pairs that share a label.
        Video-level score = mean of per-window agreement_scores.
        Mapping:
          - insufficient_evidence if no window had >= 2 scored modalities
          - high    if mean >= 0.75
          - moderate if mean >= 0.40
          - low     otherwise
        """
        scores: list[float] = []
        for w in windows:
            labels = self._modality_labels(w)
            names = list(labels.keys())
            if len(names) < 2:
                continue
            pairs = 0
            agree = 0
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    pairs += 1
                    if labels[names[i]] == labels[names[j]]:
                        agree += 1
            if pairs:
                scores.append(agree / float(pairs))

        if not scores:
            return "insufficient_evidence"
        mean_score = sum(scores) / float(len(scores))
        if mean_score >= 0.75:
            return "high"
        if mean_score >= 0.40:
            return "moderate"
        return "low"

    def cross_modal_conflicts(
        self,
        windows: Sequence[TemporalWindow],
    ) -> list[CrossModalConflict]:
        """Detect strong opposing polarity between modalities in a window.

        Conflict when at least one modality has score >= +min_polarity and
        another has score <= -min_polarity, each with confidence >= min_confidence
        (confidence approximated as max class probability when SentimentEvidence
        is reconstructed from window probability maps).
        """
        conflicts: list[CrossModalConflict] = []
        min_pol = self.config.cross_modal_min_polarity
        min_conf = self.config.cross_modal_min_confidence

        for w in windows:
            scored = self._modality_evidence(w)
            strong_pos = {
                name: ev
                for name, ev in scored.items()
                if ev.confidence >= min_conf and ev.score >= min_pol
            }
            strong_neg = {
                name: ev
                for name, ev in scored.items()
                if ev.confidence >= min_conf and ev.score <= -min_pol
            }
            if not strong_pos or not strong_neg:
                continue
            involved = {**strong_pos, **strong_neg}
            conflicts.append(
                CrossModalConflict(
                    window_index=int(w.index),
                    window_start=float(w.start),
                    window_end=float(w.end),
                    modalities=sorted(involved.keys()),
                    labels={k: v.label for k, v in involved.items()},
                    scores={k: float(v.score) for k, v in involved.items()},
                    probabilities={
                        k: dict(v.probabilities or {}) for k, v in involved.items()
                    },
                ),
            )
        return conflicts

    def evidence_coverage(self, windows: Sequence[TemporalWindow]) -> EvidenceCoverage:
        """Coverage fractions over all windows (not only usable).

        visual_coverage  = windows with 'visual' in available_modalities / total
        speech_coverage  = windows with 'speech' / total
        ocr_coverage     = windows with 'ocr' / total
        overall_usable   = usable_windows / total
        Empty timeline → all zeros with total_windows=0.
        """
        total = len(windows)
        if total == 0:
            return EvidenceCoverage()
        usable_n = sum(1 for w in windows if w.usable)
        visual_n = sum(1 for w in windows if "visual" in w.available_modalities)
        speech_n = sum(1 for w in windows if "speech" in w.available_modalities)
        ocr_n = sum(1 for w in windows if "ocr" in w.available_modalities)
        t = float(total)
        return EvidenceCoverage(
            total_windows=total,
            usable_windows=usable_n,
            visual_coverage=visual_n / t,
            speech_coverage=speech_n / t,
            ocr_coverage=ocr_n / t,
            overall_usable_coverage=usable_n / t,
        )

    def _modality_labels(self, window: TemporalWindow) -> dict[str, SentimentLabel]:
        out: dict[str, SentimentLabel] = {}
        for name, probs in (
            ("visual", window.visual_probabilities),
            ("speech", window.speech_probabilities),
            ("ocr", window.ocr_probabilities),
        ):
            if not probs:
                continue
            out[name] = max(
                ("negative", "neutral", "positive"),
                key=lambda k: float(probs.get(k, 0.0)),
            )  # type: ignore[assignment]
        return out

    def _modality_evidence(self, window: TemporalWindow) -> dict[str, SentimentEvidence]:
        out: dict[str, SentimentEvidence] = {}
        for name, probs in (
            ("visual", window.visual_probabilities),
            ("speech", window.speech_probabilities),
            ("ocr", window.ocr_probabilities),
        ):
            if not probs:
                continue
            label = max(
                ("negative", "neutral", "positive"),
                key=lambda k: float(probs.get(k, 0.0)),
            )
            conf = float(probs.get(label, 0.0))
            out[name] = SentimentEvidence(
                label=label,  # type: ignore[arg-type]
                score=score_from_probabilities(probs),
                confidence=conf,
                probabilities=dict(probs),
                model="temporal-window-aggregate",
            )
        return out
