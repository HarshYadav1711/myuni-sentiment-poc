"""Independent wellbeing / context classifier (Phase 4 foundation).

Runs locally via Hugging Face Transformers zero-shot classification.
Does NOT call OpenRouter or any external inference API.
Does NOT use visual / facial / SigLIP evidence.
Does NOT use sentiment score as a wellbeing substitute.

Lazy-load only — importing this module must not download or load the model.

Phase 4A.5:
- dual independent attribution evidence (direct_self + reported_other)
- multi_label=True for attribution heads
- legacy binary attribution available for evaluation comparison only
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Mapping, Optional, Sequence

from src.config import resolve_wellbeing_classifier_config
from src.wellbeing.labels import (
    ATTRIBUTION_CANDIDATES,
    ATTRIBUTION_HYPOTHESIS_TEMPLATE,
    DUAL_ATTRIBUTION_CANDIDATES,
    DUAL_ATTRIBUTION_HYPOTHESIS_TEMPLATE,
    DUAL_ATTRIBUTION_LABELS,
    DUAL_ATTRIBUTION_MULTI_LABEL,
    RAW_ATTRIBUTION_LABELS,
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
from src.wellbeing.policy import decide_eligibility
from src.wellbeing.schemas import (
    AttributionEvidence,
    ClassifierInputMetadata,
    ClassifierUncertainty,
    ExclusiveClassification,
    LegacyBinaryAttribution,
    SelfAttributionResult,
    SignalScore,
    SourceType,
    WellbeingClassificationResult,
)

logger = logging.getLogger(__name__)

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
        SignalScore(
            signal=label,
            score=0.0,
            threshold_passed=False,
            selected=False,
        )
        for label in SIGNAL_LABELS
    ]


def _not_evaluated_attribution() -> SelfAttributionResult:
    return SelfAttributionResult(status="not_evaluated")


def _require_all(
    results: Sequence[Optional[WellbeingClassificationResult]],
) -> list[WellbeingClassificationResult]:
    out: list[WellbeingClassificationResult] = []
    for item in results:
        if item is None:
            raise RuntimeError("incomplete_classify_many_results")
        out.append(item)
    return out


def _signals_from_scores(
    scores: Mapping[str, float],
    *,
    threshold: float,
    personal_eligible: bool,
) -> list[SignalScore]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        SignalScore(
            signal=label,
            score=float(score),
            threshold_passed=float(score) >= threshold,
            selected=bool(personal_eligible and float(score) >= threshold),
        )
        for label, score in ordered
    ]


def _dual_attribution_from_scores(
    scores: Mapping[str, float],
) -> SelfAttributionResult:
    """Build RAW dual-head attribution evidence (final_label filled by policy)."""
    direct = float(scores.get("direct_self_experience", 0.0))
    other = float(scores.get("reported_other_experience", 0.0))
    dual = {
        "direct_self_experience": direct,
        "reported_other_experience": other,
    }
    return SelfAttributionResult(
        status="ok",
        raw_label=None,
        final_label=None,
        label=None,
        scores=dual,
        top_score=max(direct, other),
        top1_top2_margin=abs(direct - other),
        attribution_evidence=AttributionEvidence(
            direct_self_score=direct,
            reported_other_score=other,
            evidence_status="ok",
        ),
    )


def _legacy_binary_from_scores(
    scores: Mapping[str, float],
) -> LegacyBinaryAttribution:
    binary = {
        "self_experience": float(scores.get("self_experience", 0.0)),
        "not_self_experience": float(scores.get("not_self_experience", 0.0)),
    }
    top_label = max(binary, key=binary.get)
    return LegacyBinaryAttribution(
        status="ok",
        raw_label=top_label,
        scores=binary,
        top_score=float(binary[top_label]),
        top1_top2_margin=_top1_top2_margin(binary),
    )


class WellbeingClassifier:
    """Lazy reusable zero-shot wellbeing classifier (CPU default)."""

    def __init__(
        self,
        *,
        model_id: Optional[str] = None,
        enabled: Optional[bool] = None,
        signal_threshold: Optional[float] = None,
        min_chars: Optional[int] = None,
        max_chars: Optional[int] = None,
        batch_size: Optional[int] = None,
        relevance_min_margin: Optional[float] = None,
        target_min_margin: Optional[float] = None,
        direct_self_min_score: Optional[float] = None,
        reported_other_block_score: Optional[float] = None,
        attribution_min_margin: Optional[float] = None,
        attribution_min_top_score: Optional[float] = None,
        include_legacy_binary_attribution: bool = False,
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
        self.batch_size = cfg.batch_size if batch_size is None else int(batch_size)
        self.relevance_min_margin = (
            cfg.relevance_min_margin
            if relevance_min_margin is None
            else float(relevance_min_margin)
        )
        self.target_min_margin = (
            cfg.target_min_margin
            if target_min_margin is None
            else float(target_min_margin)
        )
        self.direct_self_min_score = (
            cfg.direct_self_min_score
            if direct_self_min_score is None
            else float(direct_self_min_score)
        )
        self.reported_other_block_score = (
            cfg.reported_other_block_score
            if reported_other_block_score is None
            else float(reported_other_block_score)
        )
        # Legacy binary gates — comparison tooling only.
        self.attribution_min_margin = (
            cfg.attribution_min_margin
            if attribution_min_margin is None
            else float(attribution_min_margin)
        )
        self.attribution_min_top_score = (
            cfg.attribution_min_top_score
            if attribution_min_top_score is None
            else float(attribution_min_top_score)
        )
        self.include_legacy_binary_attribution = bool(
            include_legacy_binary_attribution,
        )
        self._pipeline_factory = pipeline_factory
        self._pipeline: Any = None
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()
        self.last_attribution_call_count: int = 0
        self.last_legacy_binary_call_count: int = 0

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> None:
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
            except Exception as exc:  # noqa: BLE001
                self._pipeline = None
                self._load_error = type(exc).__name__
                logger.warning(
                    "Wellbeing classifier load failed (%s)",
                    type(exc).__name__,
                )

    def _create_pipeline(self) -> Any:
        if self._pipeline_factory is not None:
            return self._pipeline_factory(self.model_id)
        from transformers import pipeline

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
        return self.classify_many([text], source_type=source_type)[0]

    def classify_many(
        self,
        texts: Sequence[Optional[str]],
        *,
        source_type: SourceType = "text",
    ) -> list[WellbeingClassificationResult]:
        """Classify many texts; attribution runs only for personal+self candidates."""
        self.last_attribution_call_count = 0
        self.last_legacy_binary_call_count = 0
        n = len(texts)
        if n == 0:
            return []

        if not self.enabled:
            return [
                self._terminal_result(
                    text,
                    source_type=source_type,
                    status="classifier_unavailable",
                    error_code="classifier_disabled",
                    error_message=(
                        "Wellbeing classifier is disabled by configuration."
                    ),
                )
                for text in texts
            ]

        prepared: list[Optional[tuple[str, bool, int]]] = []
        usable_indices: list[int] = []
        usable_texts: list[str] = []

        for idx, text in enumerate(texts):
            item = self._prepare_text(text)
            if item is None:
                prepared.append(None)
                continue
            cleaned, truncated = item
            char_count = 0 if text is None else len(str(text).strip())
            prepared.append((cleaned, truncated, char_count))
            usable_indices.append(idx)
            usable_texts.append(cleaned)

        results: list[Optional[WellbeingClassificationResult]] = [None] * n
        for idx, prep in enumerate(prepared):
            if prep is None:
                results[idx] = self._terminal_result(
                    texts[idx],
                    source_type=source_type,
                    status="insufficient_text",
                    error_code="insufficient_text",
                    error_message=(
                        "Input text is missing, blank, or too short for "
                        "wellbeing classification. Missing is not neutral."
                    ),
                )

        if not usable_texts:
            return _require_all(results)

        if not self.is_loaded:
            self.load()

        if not self.is_loaded:
            for idx in usable_indices:
                results[idx] = self._terminal_result(
                    texts[idx],
                    source_type=source_type,
                    status="classifier_unavailable",
                    error_code=self._load_error or "classifier_unavailable",
                    error_message="Wellbeing classifier model is unavailable.",
                    prepared=prepared[idx],
                )
            return _require_all(results)

        try:
            relevance_list = self._run_exclusive_batch(
                usable_texts,
                labels=RELEVANCE_LABELS,
                candidates=RELEVANCE_CANDIDATES,
                hypothesis_template=RELEVANCE_HYPOTHESIS_TEMPLATE,
            )
            target_list = self._run_exclusive_batch(
                usable_texts,
                labels=TARGET_LABELS,
                candidates=TARGET_CANDIDATES,
                hypothesis_template=TARGET_HYPOTHESIS_TEMPLATE,
            )
            signal_score_maps = self._run_signals_batch(usable_texts)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Wellbeing classifier inference failed (%s)",
                type(exc).__name__,
            )
            for idx in usable_indices:
                results[idx] = self._terminal_result(
                    texts[idx],
                    source_type=source_type,
                    status="error",
                    error_code=type(exc).__name__,
                    error_message="Wellbeing classifier inference failed.",
                    prepared=prepared[idx],
                )
            return _require_all(results)

        # Conditional attribution: only personal_wellbeing + self candidates.
        attr_batch_indices: list[int] = []
        attr_texts: list[str] = []
        for batch_i, idx in enumerate(usable_indices):
            rel = relevance_list[batch_i]
            tgt = target_list[batch_i]
            if rel.label == "personal_wellbeing" and tgt.label == "self":
                attr_batch_indices.append(batch_i)
                attr_texts.append(usable_texts[batch_i])

        attributions: dict[int, SelfAttributionResult] = {}
        if attr_texts:
            self.last_attribution_call_count = len(attr_texts)
            try:
                dual_maps = self._run_dual_attribution_batch(attr_texts)
                legacy_maps: list[Optional[dict[str, float]]] = [
                    None
                ] * len(attr_texts)
                if self.include_legacy_binary_attribution:
                    self.last_legacy_binary_call_count = len(attr_texts)
                    legacy_exclusive = self._run_exclusive_batch(
                        attr_texts,
                        labels=RAW_ATTRIBUTION_LABELS,
                        candidates=ATTRIBUTION_CANDIDATES,
                        hypothesis_template=ATTRIBUTION_HYPOTHESIS_TEMPLATE,
                    )
                    legacy_maps = [item.scores for item in legacy_exclusive]
                for local_i, batch_i in enumerate(attr_batch_indices):
                    attr = _dual_attribution_from_scores(dual_maps[local_i])
                    if legacy_maps[local_i] is not None:
                        attr = attr.model_copy(
                            update={
                                "legacy_binary": _legacy_binary_from_scores(
                                    legacy_maps[local_i] or {},
                                ),
                            },
                        )
                    attributions[batch_i] = attr
            except Exception as exc:  # noqa: BLE001 — fail closed per item
                logger.warning(
                    "Wellbeing attribution inference failed (%s)",
                    type(exc).__name__,
                )
                for batch_i in attr_batch_indices:
                    attributions[batch_i] = SelfAttributionResult(
                        status="error",
                        error_code=type(exc).__name__,
                        attribution_evidence=AttributionEvidence(
                            evidence_status="error",
                            error_code=type(exc).__name__,
                        ),
                    )

        for batch_i, idx in enumerate(usable_indices):
            prep = prepared[idx]
            assert prep is not None
            _cleaned, truncated, char_count = prep
            relevance = relevance_list[batch_i]
            target = target_list[batch_i]
            rel_margin = _top1_top2_margin(relevance.scores)
            tgt_margin = _top1_top2_margin(target.scores)

            attribution = attributions.get(batch_i, _not_evaluated_attribution())
            decision = decide_eligibility(
                classifier_status="ok",
                relevance_label=relevance.label,
                target_label=target.label,
                relevance_margin=rel_margin,
                target_margin=tgt_margin,
                attribution=attribution,
                relevance_min_margin=self.relevance_min_margin,
                target_min_margin=self.target_min_margin,
                direct_self_min_score=self.direct_self_min_score,
                reported_other_block_score=self.reported_other_block_score,
            )
            signals = _signals_from_scores(
                signal_score_maps[batch_i],
                threshold=self.signal_threshold,
                personal_eligible=decision.personal_wellbeing_eligible,
            )
            results[idx] = WellbeingClassificationResult(
                model_id=self.model_id,
                status="ok",
                relevance=relevance,
                target=target,
                signals=signals,
                self_attribution=decision.finalized_attribution,
                input_metadata=ClassifierInputMetadata(
                    source_type=source_type,
                    input_character_count=char_count,
                    truncated=truncated,
                ),
                uncertainty=ClassifierUncertainty(
                    relevance_top1_top2_margin=rel_margin,
                    target_top1_top2_margin=tgt_margin,
                    attribution_top1_top2_margin=(
                        decision.finalized_attribution.top1_top2_margin
                    ),
                ),
                eligibility_status=decision.status,
                personal_wellbeing_eligible=decision.personal_wellbeing_eligible,
                eligibility_reasons=decision.reasons,
            )

        return _require_all(results)

    def _terminal_result(
        self,
        text: Optional[str],
        *,
        source_type: SourceType,
        status: str,
        error_code: str,
        error_message: str,
        prepared: Optional[tuple[str, bool, int]] = None,
    ) -> WellbeingClassificationResult:
        if prepared is not None:
            _cleaned, truncated, char_count = prepared
            meta = ClassifierInputMetadata(
                source_type=source_type,
                input_character_count=char_count,
                truncated=truncated,
            )
        else:
            meta = ClassifierInputMetadata(
                source_type=source_type,
                input_character_count=0 if text is None else len(str(text)),
                truncated=False,
            )
        decision = decide_eligibility(
            classifier_status=status,
            relevance_label=None,
            target_label=None,
            relevance_margin=None,
            target_margin=None,
            attribution=_not_evaluated_attribution(),
            relevance_min_margin=self.relevance_min_margin,
            target_min_margin=self.target_min_margin,
            direct_self_min_score=self.direct_self_min_score,
            reported_other_block_score=self.reported_other_block_score,
        )
        return WellbeingClassificationResult(
            model_id=self.model_id,
            status=status,  # type: ignore[arg-type]
            relevance=_empty_exclusive(RELEVANCE_LABELS),
            target=_empty_exclusive(TARGET_LABELS),
            signals=_empty_signals(),
            self_attribution=decision.finalized_attribution,
            input_metadata=meta,
            eligibility_status=decision.status,
            personal_wellbeing_eligible=decision.personal_wellbeing_eligible,
            eligibility_reasons=decision.reasons,
            error_code=error_code,
            error_message=error_message,
        )

    def _prepare_text(self, text: Optional[str]) -> Optional[tuple[str, bool]]:
        if text is None or not isinstance(text, str):
            return None
        cleaned = text.strip()
        if not cleaned or len(cleaned) < self.min_chars:
            return None
        truncated = False
        if len(cleaned) > self.max_chars:
            cleaned = cleaned[: self.max_chars]
            truncated = True
        return cleaned, truncated

    def _chunk_texts(self, texts: Sequence[str]) -> list[list[str]]:
        size = max(1, int(self.batch_size))
        return [list(texts[i : i + size]) for i in range(0, len(texts), size)]

    def _invoke_pipeline(
        self,
        sequences: str | Sequence[str],
        *,
        candidate_labels: list[str],
        hypothesis_template: str,
        multi_label: bool,
    ) -> Mapping[str, Any] | list[Mapping[str, Any]]:
        pipe = self._pipeline
        if pipe is None:
            raise RuntimeError("pipeline_not_loaded")
        return pipe(  # type: ignore[no-any-return]
            sequences,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=multi_label,
            batch_size=max(1, int(self.batch_size)),
        )

    def _normalize_batch_output(
        self,
        raw: Mapping[str, Any] | list[Any],
        *,
        expected: int,
    ) -> list[Mapping[str, Any]]:
        if expected == 1 and isinstance(raw, Mapping):
            return [raw]
        if isinstance(raw, list):
            if len(raw) != expected:
                raise ValueError("unexpected_batch_output_length")
            out: list[Mapping[str, Any]] = []
            for item in raw:
                if not isinstance(item, Mapping):
                    raise ValueError("unexpected_pipeline_output_type")
                out.append(item)
            return out
        raise ValueError("unexpected_pipeline_output_type")

    def _run_exclusive_batch(
        self,
        texts: Sequence[str],
        *,
        labels: tuple[str, ...],
        candidates: Mapping[str, str],
        hypothesis_template: str,
    ) -> list[ExclusiveClassification]:
        results: list[ExclusiveClassification] = []
        cands = candidate_list(candidates, labels)
        for chunk in self._chunk_texts(texts):
            raw = self._invoke_pipeline(
                chunk,
                candidate_labels=cands,
                hypothesis_template=hypothesis_template,
                multi_label=False,
            )
            for item in self._normalize_batch_output(raw, expected=len(chunk)):
                scores = self._scores_from_pipeline(
                    item,
                    candidates=candidates,
                    labels=labels,
                )
                top_label = max(scores, key=scores.get)
                results.append(
                    ExclusiveClassification(
                        label=top_label,
                        scores=scores,
                        confidence=float(scores[top_label]),
                    ),
                )
        return results

    def _run_signals_batch(self, texts: Sequence[str]) -> list[dict[str, float]]:
        results: list[dict[str, float]] = []
        cands = candidate_list(SIGNAL_CANDIDATES, SIGNAL_LABELS)
        for chunk in self._chunk_texts(texts):
            raw = self._invoke_pipeline(
                chunk,
                candidate_labels=cands,
                hypothesis_template=SIGNAL_HYPOTHESIS_TEMPLATE,
                multi_label=True,
            )
            for item in self._normalize_batch_output(raw, expected=len(chunk)):
                scores = self._scores_from_pipeline(
                    item,
                    candidates=SIGNAL_CANDIDATES,
                    labels=SIGNAL_LABELS,
                )
                if len(scores) != len(SIGNAL_LABELS):
                    raise ValueError("incomplete_or_unknown_signal_scores")
                results.append(scores)
        return results

    def _run_dual_attribution_batch(
        self,
        texts: Sequence[str],
    ) -> list[dict[str, float]]:
        """Independent dual-head attribution (multi_label=True)."""
        results: list[dict[str, float]] = []
        cands = candidate_list(DUAL_ATTRIBUTION_CANDIDATES, DUAL_ATTRIBUTION_LABELS)
        assert DUAL_ATTRIBUTION_MULTI_LABEL is True
        for chunk in self._chunk_texts(texts):
            raw = self._invoke_pipeline(
                chunk,
                candidate_labels=cands,
                hypothesis_template=DUAL_ATTRIBUTION_HYPOTHESIS_TEMPLATE,
                multi_label=True,
            )
            for item in self._normalize_batch_output(raw, expected=len(chunk)):
                scores = self._scores_from_pipeline(
                    item,
                    candidates=DUAL_ATTRIBUTION_CANDIDATES,
                    labels=DUAL_ATTRIBUTION_LABELS,
                )
                if len(scores) != len(DUAL_ATTRIBUTION_LABELS):
                    raise ValueError("incomplete_dual_attribution_scores")
                results.append(scores)
        return results

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
        if seen != set(labels):
            raise ValueError("incomplete_candidate_coverage")
        return scores


def get_wellbeing_classifier() -> WellbeingClassifier:
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
    return get_wellbeing_classifier().classify(text, source_type=source_type)


def reset_wellbeing_classifier_for_tests() -> None:
    global _default_classifier
    with _default_lock:
        _default_classifier = None
