"""English social-media text sentiment analyzer (RoBERTa)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.schemas import SentimentEvidence, SentimentLabel

logger = logging.getLogger(__name__)

DEFAULT_TEXT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Canonical label order used for score = P(pos) - P(neg).
_LABEL_KEYS: tuple[SentimentLabel, ...] = ("negative", "neutral", "positive")


class TextSentimentAnalyzer:
    """Lazy-loaded Twitter-RoBERTa sentiment classifier for English text.

    Score convention (POC, not client business scoring):
        score = positive_probability - negative_probability  ∈ [-1, +1]
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self._device_preference = device
        self._tokenizer = None
        self._model = None
        self._id2label: dict[int, str] = {}
        self._device: Optional[torch.device] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _resolve_device(self) -> torch.device:
        if self._device_preference:
            return torch.device(self._device_preference)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _normalize_label(self, raw: str) -> SentimentLabel:
        key = raw.strip().lower()
        # Some checkpoints expose LABEL_0 / negative / Negative.
        aliases = {
            "label_0": "negative",
            "label_1": "neutral",
            "label_2": "positive",
            "neg": "negative",
            "neu": "neutral",
            "pos": "positive",
            "negative": "negative",
            "neutral": "neutral",
            "positive": "positive",
        }
        if key not in aliases:
            raise ValueError(f"Unexpected sentiment label from model: {raw!r}")
        return aliases[key]  # type: ignore[return-value]

    def load(self) -> None:
        """Download (if needed) and load tokenizer + model onto device."""
        if self.is_loaded:
            return

        logger.info("Loading text sentiment model: %s", self.model_name)
        self._device = self._resolve_device()

        # CardiffNLP checkpoints commonly emit an unused-pooler warning; keep CLI quiet.
        from transformers.utils import logging as hf_logging

        prev_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            config = AutoConfig.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
            )
        finally:
            hf_logging.set_verbosity(prev_verbosity)

        self._model.to(self._device)
        self._model.eval()

        self._id2label = {
            int(i): self._normalize_label(str(label))
            for i, label in config.id2label.items()
        }
        logger.info(
            "Text sentiment model ready on %s (labels=%s)",
            self._device,
            self._id2label,
        )

    @staticmethod
    def validate_text(text: object) -> str:
        """Require a non-blank string. Raises ValueError otherwise."""
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("text must be a non-blank string")
        return cleaned

    def analyze(self, text: object) -> SentimentEvidence:
        """Run inference and return structured modality evidence."""
        cleaned = self.validate_text(text)
        self.load()

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None

        encoded = self._tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self._model(**encoded)
            logits = outputs.logits.detach().cpu().numpy()[0]

        probs = softmax(logits)
        probability_map: dict[str, float] = {
            self._id2label[i]: float(probs[i]) for i in range(len(probs))
        }

        # Ensure all three keys exist for a stable score formula.
        for key in _LABEL_KEYS:
            probability_map.setdefault(key, 0.0)

        positive_p = probability_map["positive"]
        negative_p = probability_map["negative"]
        score = float(positive_p - negative_p)

        label = max(_LABEL_KEYS, key=lambda k: probability_map[k])
        confidence = float(probability_map[label])

        return SentimentEvidence(
            label=label,
            score=float(np.clip(score, -1.0, 1.0)),
            confidence=confidence,
            probabilities={k: float(probability_map[k]) for k in _LABEL_KEYS},
            model=self.model_name,
            details={"device": str(self._device)},
        )
