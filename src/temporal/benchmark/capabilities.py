"""Model-specific chat-template / generation capabilities.

Do not assume every model accepts ``enable_thinking``. Capabilities are
declared here without downloading model weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.config import (
    TEMPORAL_REASONER_CANDIDATE_1_7B,
    TEMPORAL_REASONER_CANDIDATE_4B,
)


@dataclass(frozen=True)
class ModelCapability:
    """Capability profile for one reasoner candidate."""

    model_id: str
    display_name: str
    # When True, apply_chat_template may receive enable_thinking=.
    supports_enable_thinking: bool
    default_enable_thinking: bool = False
    # Extra kwargs always passed to apply_chat_template (model-specific).
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


# Declared without downloading Qwen3-4B. Instruct-2507 is treated as a
# non-thinking Instruct chat model: do not pass enable_thinking blindly.
MODEL_CANDIDATES: dict[str, ModelCapability] = {
    TEMPORAL_REASONER_CANDIDATE_1_7B: ModelCapability(
        model_id=TEMPORAL_REASONER_CANDIDATE_1_7B,
        display_name="Qwen3-1.7B",
        supports_enable_thinking=True,
        default_enable_thinking=False,
        notes=(
            "Qwen3 base/chat supports enable_thinking via chat template. "
            "Benchmark forces enable_thinking=False."
        ),
    ),
    TEMPORAL_REASONER_CANDIDATE_4B: ModelCapability(
        model_id=TEMPORAL_REASONER_CANDIDATE_4B,
        display_name="Qwen3-4B-Instruct-2507",
        supports_enable_thinking=False,
        default_enable_thinking=False,
        notes=(
            "Instruct-2507: do not pass enable_thinking unless a future "
            "tokenizer documents support. Capability declared offline; "
            "weights must not be downloaded on local CPU during Phase 3B-A."
        ),
    ),
}


def resolve_model_capability(model_id: str) -> ModelCapability:
    """Return a known capability profile or a conservative default."""
    if model_id in MODEL_CANDIDATES:
        return MODEL_CANDIDATES[model_id]
    # Unknown models: never invent thinking support.
    return ModelCapability(
        model_id=model_id,
        display_name=model_id,
        supports_enable_thinking=False,
        default_enable_thinking=False,
        notes="Unknown model — enable_thinking not passed.",
    )


def chat_template_apply_kwargs(
    capability: ModelCapability,
    *,
    enable_thinking: Optional[bool] = None,
) -> dict[str, Any]:
    """Build apply_chat_template kwargs for this model.

    Thinking switches are only included when the capability supports them.
    """
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        **dict(capability.chat_template_kwargs),
    }
    if capability.supports_enable_thinking:
        thinking = (
            capability.default_enable_thinking
            if enable_thinking is None
            else bool(enable_thinking)
        )
        kwargs["enable_thinking"] = thinking
    return kwargs
