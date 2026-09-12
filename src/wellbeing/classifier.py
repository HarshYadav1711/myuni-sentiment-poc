"""Independent wellbeing / context classifier (Phase 4 foundation).

Runs locally via Hugging Face Transformers zero-shot classification.
Does NOT call OpenRouter or any external inference API.
Does NOT use visual / facial / SigLIP evidence.
Does NOT use sentiment score as a wellbeing substitute.

Lazy-load only — importing this module must not download or load the model.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Mapping, Optional

from src.config import resolve_wellbeing_classifier_config
from src.wellbeing.labels import (
    RELEVANCE_CANDIDATES,
    RELEVANCE_HYPOTHESIS_TEMPLATE,
    RELEVANCE_LABELS,
    SIGNAL_CANDIDATES,
    SIGNAL_HYPOTHESIS_TEMPLATE,
    SIGNAL_LABELS,
    TARGET_CANDIDATES,
    TARGET_HYPOTHESIS_TEMPLATE,
    TARGET_LABELS,
    candidate_list,
    verbalization_to_label,
)
from src.wellbeing.schemas import (
    ClassifierInputMetadata,
    ClassifierUncertainty,
    ExclusiveClassification,
    SignalScore,
    SourceType,
    WellbeingClassificationResult,
)

logger = logging.getLogger(__name__)

# Module-level singleton for lazy reuse across calls.
_default_classifier: Optional["WellbeingClassifier"] = None
_default_lock = threading.Lock()


def _top1_top2_margin(scores: Mapping[str, float]) -> Optional[float]:
    if len(scores) < 2:
        if len(scores) == 1:
            return float(next(iter(scores.values())))
        return None
    ordered = sorted(scores.values(), reverse=True)
    return float(ordered[0] - ordered[1])


def _empty_exclusive(labels: tuple[str, ...]) -> ExclusiveClassification:
    return ExclusiveClassification(
        label=None,
        scores={label: 0.0 for label in labels},
        confidence=None,
    )


def _empty_signals() -> list[SignalScore]:
    return [
        SignalScore(signal=label, score=0.0, selected=False)
        for label in SIGNAL_LABELS
    ]


class WellbeingClassifier:
    """Lazy reusable zero-shot wellbeing relevance / target / signals classifier.

    Default device is CPU. This pass does not auto-select CUDA via
    ``torch.cuda.is_available()``.
    """

    def __init__(
        self,
        *,
        model_id: Optional[str] = None,
        enabled: Optional[bool] = None,
        signal_threshold: Optional[float] = None,
        min_chars: Optional[int] = None,
        max_chars: Optional[int] = None,
        pipeline_factory: Optional[Any] = None,
    ) -> None:
        cfg = resolve_wellbeing_classifier_config()
        self.model_id = model_id or cfg.model_id
        self.enabled = cfg.enabled if enabled is None else bool(enabled)
        self.signal_threshold = (
            cfg.signal_poc_threshold
            if signal_threshold is None
            else float(signal_threshold)
        )
        self.min_chars = cfg.min_chars if min_chars is None else int(min_chars)
        self.max_chars = cfg.max_chars if max_chars is None else int(max_chars)
        self._pipeline_factory = pipeline_factory
        self._pipeline: Any = None
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> None:
        """Load tokenizer/model once onto CPU in eval mode.

        Safe to call repeatedly. Failures are recorded without raising so
        ``classify`` can return a typed fail-soft result.
        """
        if self.is_loaded:
            return
        with self._lock:
            if self.is_loaded:
                return
            if not self.enabled:
                self._load_error = "classifier_disabled"
                return
            try:
                self._pipeline = self._create_pipeline()
                self._load_error = None
                logger.info(
                    "Wellbeing classifier ready (model_id=%s, device=cpu)",
                    self.model_id,
                )
            except Exception as exc:  # noqa: BLE001 — fail-soft foundation
                # Never log user text here (none available); keep exception type only.
                self._pipeline = None
                self._load_error = type(exc).__name__
                logger.warning(
                    "Wellbeing classifier load failed (%s)",
                    type(exc).__name__,
                )

    def _create_pipeline(self) -> Any:
        if self._pipeline_factory is not None:
            return self._pipeline_factory(self.model_id)

        # Import Transformers only at load time — not at module import.
        from transformers import pipeline

        # Explicit CPU; do not call torch.cuda.is_available() in this pass.
        return pipeline(
            "zero-shot-classification",
            model=self.model_id,
            device=-1,
        )

    def classify(
        self,
        text: Optional[str],
        *,
        source_type: SourceType = "text",
    ) -> WellbeingClassificationResult:
        """Classify content-level wellbeing relevance, target, and signals."""
        meta = ClassifierInputMetadata(
            source_type=source_type,
            input_character_count=0,
            truncated=False,
        )

        if not self.enabled:
            return WellbeingClassificationResult(
                model_id=self.model_id,
                status="classifier_unavailable",
                relevance=_empty_exclusive(RELEVANCE_LABELS),
                target=_empty_exclusive(TARGET_LABELS),
                signals=_empty_signals(),
                input_metadata=meta,
                error_code="classifier_disabled",
                error_message="Wellbeing classifier is disabled by configuration.",
            )

        prepared = self._prepare_text(text)
        if prepared is None:
            char_count = 0 if text is None else len(str(text))
            meta = ClassifierInputMetadata(
                source_type=source_type,
                input_character_count=char_count,
                truncated=False,
            )
            return WellbeingClassificationResult(
                model_id=self.model_id,
                status="insufficient_text",
                relevance=_empty_exclusive(RELEVANCE_LABELS),
                target=_empty_exclusive(TARGET_LABELS),
                signals=_empty_signals(),
                input_metadata=meta,
                error_code="insufficient_text",
                error_message=(
                    "Input text is missing, blank, or too short for "
                    "wellbeing classification. Missing is not neutral."
                ),
            )

        cleaned, truncated = prepared
        meta = ClassifierInputMetadata(
            source_type=source_type,
            input_character_count=len(cleaned) if not truncated else len(str(text).strip()),
            truncated=truncated,
        )
        # Prefer original stripped length for metadata when truncated.
        if text is not None:
            meta = ClassifierInputMetadata(
                source_type=source_type,
                input_character_count=len(str(text).strip()),
                truncated=truncated,
            )

        if not self.is_loaded:
            self.load()

        if not self.is_loaded:
            return WellbeingClassificationResult(
                model_id=self.model_id,
                status="classifier_unavailable",
                relevance=_empty_exclusive(RELEVANCE_LABELS),
                target=_empty_exclusive(TARGET_LABELS),
                signals=_empty_signals(),
                input_metadata=meta,
                error_code=self._load_error or "classifier_unavailable",
                error_message="Wellbeing classifier model is unavailable.",
            )

        try:
            relevance = self._run_exclusive(
                cleaned,
                labels=RELEVANCE_LABELS,
                candidates=RELEVANCE_CANDIDATES,
                hypothesis_template=RELEVANCE_HYPOTHESIS_TEMPLATE,
                multi_label=False,
            )
            target = self._run_exclusive(
                cleaned,
                labels=TARGET_LABELS,
                candidates=TARGET_CANDIDATES,
                hypothesis_template=TARGET_HYPOTHESIS_TEMPLATE,
                multi_label=False,
            )
            signals = self._run_signals(cleaned)
        except Exception as exc:  # noqa: BLE001 — fail-soft; do not echo user text
            logger.warning(
                "Wellbeing classifier inference failed (%s)",
                type(exc).__name__,
            )
            return WellbeingClassificationResult(
                model_id=self.model_id,
                status="error",
                relevance=_empty_exclusive(RELEVANCE_LABELS),
                target=_empty_exclusive(TARGET_LABELS),
                signals=_empty_signals(),
                input_metadata=meta,
                error_code=type(exc).__name__,
                error_message="Wellbeing classifier inference failed.",
            )

        return WellbeingClassificationResult(
            model_id=self.model_id,
            status="ok",
            relevance=relevance,
            target=target,
            signals=signals,
            input_metadata=meta,
            uncertainty=ClassifierUncertainty(
                relevance_top1_top2_margin=_top1_top2_margin(relevance.scores),
                target_top1_top2_margin=_top1_top2_margin(target.scores),
            ),
        )

    def _prepare_text(self, text: Optional[str]) -> Optional[tuple[str, bool]]:
        if text is None:
            return None
        if not isinstance(text, str):
            return None
        cleaned = text.strip()
        if not cleaned:
            return None
        # Extremely short unusable text — no model inference.
        if len(cleaned) < self.min_chars:
            return None
        truncated = False
        if len(cleaned) > self.max_chars:
            cleaned = cleaned[: self.max_chars]
            truncated = True
        return cleaned, truncated

    def _invoke_pipeline(
        self,
        text: str,
        *,
        candidate_labels: list[str],
        hypothesis_template: str,
        multi_label: bool,
    ) -> Mapping[str, Any]:
        pipe = self._pipeline
        if pipe is None:
            raise RuntimeError("pipeline_not_loaded")
        raw = pipe(
            text,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=multi_label,
        )
        if not isinstance(raw, Mapping):
            raise ValueError("unexpected_pipeline_output_type")
        return raw

    def _run_exclusive(
        self,
        text: str,
        *,
        labels: tuple[str, ...],
        candidates: Mapping[str, str],
        hypothesis_template: str,
        multi_label: bool,
    ) -> ExclusiveClassification:
        raw = self._invoke_pipeline(
            text,
            candidate_labels=candidate_list(candidates, labels),
            hypothesis_template=hypothesis_template,
            multi_label=multi_label,
        )
        scores = self._scores_from_pipeline(raw, candidates=candidates, labels=labels)
        if not scores:
            raise ValueError("empty_or_unknown_exclusive_scores")
        top_label = max(scores, key=scores.get)
        return ExclusiveClassification(
            label=top_label,
            scores=scores,
            confidence=float(scores[top_label]),
        )

    def _run_signals(self, text: str) -> list[SignalScore]:
        raw = self._invoke_pipeline(
            text,
            candidate_labels=candidate_list(SIGNAL_CANDIDATES, SIGNAL_LABELS),
            hypothesis_template=SIGNAL_HYPOTHESIS_TEMPLATE,
            multi_label=True,
        )
        scores = self._scores_from_pipeline(
            raw,
            candidates=SIGNAL_CANDIDATES,
            labels=SIGNAL_LABELS,
        )
        if len(scores) != len(SIGNAL_LABELS):
            raise ValueError("incomplete_or_unknown_signal_scores")

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            SignalScore(
                signal=label,
                score=float(score),
                selected=float(score) >= self.signal_threshold,
            )
            for label, score in ordered
        ]

    def _scores_from_pipeline(
        self,
        raw: Mapping[str, Any],
        *,
        candidates: Mapping[str, str],
        labels: tuple[str, ...],
    ) -> dict[str, float]:
        raw_labels = raw.get("labels")
        raw_scores = raw.get("scores")
        if not isinstance(raw_labels, (list, tuple)) or not isinstance(
            raw_scores,
            (list, tuple),
        ):
            raise ValueError("missing_labels_or_scores")
        if len(raw_labels) != len(raw_scores):
            raise ValueError("labels_scores_length_mismatch")

        scores: dict[str, float] = {label: 0.0 for label in labels}
        seen: set[str] = set()
        for verbalization, score in zip(raw_labels, raw_scores):
            try:
                label = verbalization_to_label(str(verbalization), candidates)
            except KeyError as exc:
                raise ValueError("unknown_hf_candidate_label") from exc
            if label not in scores:
                raise ValueError("unexpected_taxonomy_label")
            scores[label] = float(score)
            seen.add(label)

        # Exclusive heads may omit nothing; require full coverage for safety.
        if seen != set(labels):
            raise ValueError("incomplete_candidate_coverage")
        return scores


def get_wellbeing_classifier() -> WellbeingClassifier:
    """Return the process-wide lazy classifier singleton."""
    global _default_classifier
    if _default_classifier is None:
        with _default_lock:
            if _default_classifier is None:
                _default_classifier = WellbeingClassifier()
    return _default_classifier


def classify_wellbeing(
    text: Optional[str],
    *,
    source_type: SourceType = "text",
) -> WellbeingClassificationResult:
    """Convenience entry point using the process-wide singleton."""
    return get_wellbeing_classifier().classify(text, source_type=source_type)


def reset_wellbeing_classifier_for_tests() -> None:
    """Clear the singleton (unit tests only)."""
    global _default_classifier
    with _default_lock:
        _default_classifier = None
