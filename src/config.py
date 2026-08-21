"""POC configuration for models, OCR, and explainable fusion weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


# Exact Hugging Face checkpoint used for zero-shot visual concept scoring.
DEFAULT_VISUAL_MODEL = "google/siglip2-base-patch16-224"

# Zero-shot candidate prompts → sentiment labels (transparent, not a trained classifier).
DEFAULT_VISUAL_PROMPTS: dict[str, str] = {
    "positive": "a positive and pleasant situation",
    "neutral": "a neutral everyday situation",
    "negative": "a negative or unpleasant situation",
}

# Minimum alphanumeric characters before OCR text is treated as meaningful.
OCR_MIN_ALNUM_CHARS = 3

# faster-whisper model size. base.en is CPU-friendly on ~16 GB RAM.
DEFAULT_WHISPER_MODEL = "base.en"
# int8 is the conservative CPU compute type for faster-whisper.
DEFAULT_WHISPER_COMPUTE_TYPE = "int8"
# Assumed language for English-only MVP (also passed to Whisper).
DEFAULT_ASR_LANGUAGE = "en"


@dataclass(frozen=True)
class FusionConfig:
    """POC-only late fusion weights (not client business scoring).

    Image overall score is a confidence-weighted average of available modality
    scores (caption text, visual, OCR text). Missing modalities are skipped.
    Label is derived from the fused score with a small neutral band.
    """

    modality_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "text": 1.0,  # caption
            "visual": 1.0,
            "ocr": 0.8,
        },
    )
    neutral_band: float = 0.15


DEFAULT_FUSION = FusionConfig()
