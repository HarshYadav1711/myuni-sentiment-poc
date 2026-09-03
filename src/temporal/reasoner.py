"""TemporalContextReasoner — text LLM over structured temporal evidence.

Additive only: failures never break deterministic temporal_context or fusion.
Does not receive raw video/images/audio.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from src.config import DEFAULT_TEMPORAL_REASONER, TemporalReasonerConfig
from src.schemas import (
    SentimentEvidence,
    TemporalContext,
    TemporalReasonerDiagnostics,
    TemporalReasoningResult,
)
from src.temporal.parse import format_validation_error, parse_reasoning_result
from src.temporal.prompt import (
    SYSTEM_INSTRUCTION,
    build_evidence_payload,
    build_repair_prompt,
    build_user_prompt,
    collect_valid_evidence_ids,
)

logger = logging.getLogger(__name__)


class TemporalContextReasoner:
    """Lazy-loaded text reasoner for video temporal context (Phase 2)."""

    def __init__(self, config: TemporalReasonerConfig = DEFAULT_TEMPORAL_REASONER) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None
        self._load_error: Optional[str] = None
        self._last_generation_meta: dict[str, Any] = {}
        self._device: str = (config.device or "cpu").strip() or "cpu"
        self._torch = None

    def unload(self) -> None:
        """Drop model/tokenizer references. GPU cache clearing is deployment-layer."""
        self._model = None
        self._tokenizer = None
        self._last_generation_meta = {}
        self._load_error = None
        try:
            import gc

            gc.collect()
        except Exception:  # noqa: BLE001
            pass

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def load(self) -> None:
        """Download/load the configured causal LM onto the configured device."""
        if self.is_loaded:
            return
        if self._load_error is not None:
            raise RuntimeError(self._load_error)

        device = (self.config.device or "cpu").strip() or "cpu"
        # Explicit device only — never torch.cuda.is_available() (ZeroGPU).
        logger.info(
            "Loading temporal reasoner model=%s device=%s",
            self.config.model_id,
            device,
        )
        started = time.perf_counter()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(self.config.model_id)
            self._model.to(device)
            self._model.eval()
            self._device = device
            self._torch = torch
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"Failed to load temporal reasoner: {exc}"
            self._tokenizer = None
            self._model = None
            logger.exception("Temporal reasoner load failed")
            raise RuntimeError(self._load_error) from exc

        logger.info(
            "Temporal reasoner ready in %.2fs model=%s device=%s",
            time.perf_counter() - started,
            self.config.model_id,
            device,
        )

    def reason(
        self,
        temporal: TemporalContext,
        *,
        baseline_overall: Optional[SentimentEvidence] = None,
    ) -> tuple[TemporalReasoningResult, TemporalReasonerDiagnostics]:
        """Produce validated contextual interpretation or a fail-soft status."""
        total_started = time.perf_counter()
        diagnostics = TemporalReasonerDiagnostics()
        if not self.config.enabled:
            diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
            return (
                TemporalReasoningResult(
                    summary="",
                    context_type="uncertain",
                    confidence=0.0,
                    model=self.config.model_id,
                    status="disabled",
                    details={"reason": "TEMPORAL_REASONER_ENABLED=false"},
                ),
                diagnostics,
            )

        try:
            load_started = time.perf_counter()
            self.load()
            diagnostics.model_load_seconds = time.perf_counter() - load_started
        except Exception as exc:  # noqa: BLE001
            diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
            return (
                TemporalReasoningResult(
                    summary="",
                    context_type="uncertain",
                    confidence=0.0,
                    model=self.config.model_id,
                    status="reasoner_unavailable",
                    details={"error": str(exc)},
                ),
                diagnostics,
            )

        prompt_started = time.perf_counter()
        evidence = build_evidence_payload(
            temporal,
            baseline_overall=baseline_overall,
            config=self.config,
        )
        user_prompt = build_user_prompt(evidence)
        diagnostics.prompt_construction_seconds = time.perf_counter() - prompt_started
        diagnostics.prompt_chars = len(user_prompt)
        diagnostics.prompt_windows_included = int(evidence.get("windows_included", 0))
        diagnostics.evidence_ids_supplied = collect_valid_evidence_ids(evidence)
        valid_evidence_ids = set(diagnostics.evidence_ids_supplied)
        valid_window_ranges = [
            (float(window.start), float(window.end))
            for window in temporal.windows
        ]

        try:
            gen_started = time.perf_counter()
            raw = self._generate(SYSTEM_INSTRUCTION, user_prompt)
            diagnostics.generation_seconds = time.perf_counter() - gen_started
            diagnostics.raw_output_preview = raw[:500]
            self._apply_generation_meta(diagnostics)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Temporal reasoner generation failed: %s", exc)
            diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
            self._apply_generation_meta(diagnostics)
            return (
                TemporalReasoningResult(
                    summary="",
                    context_type="uncertain",
                    confidence=0.0,
                    model=self.config.model_id,
                    status="reasoner_unavailable",
                    details={"error": f"generation_failed: {exc}"},
                ),
                diagnostics,
            )

        try:
            parse_started = time.perf_counter()
            result = parse_reasoning_result(
                raw,
                model_id=self.config.model_id,
                valid_evidence_ids=valid_evidence_ids,
                valid_window_ranges=valid_window_ranges,
            )
            diagnostics.parse_validation_seconds = time.perf_counter() - parse_started
            diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
            return result, diagnostics
        except Exception as first_exc:  # noqa: BLE001
            if self.config.max_retries < 1:
                diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
                return self._invalid(raw, first_exc), diagnostics

            diagnostics.repair_attempted = True
            repair = build_repair_prompt(
                validation_error=format_validation_error(first_exc),
                previous_output=raw,
            )
            try:
                gen_started = time.perf_counter()
                raw_retry = self._generate(SYSTEM_INSTRUCTION, repair)
                diagnostics.generation_seconds = (
                    diagnostics.generation_seconds or 0.0
                ) + (time.perf_counter() - gen_started)
                diagnostics.raw_output_preview = raw_retry[:500]
                self._apply_generation_meta(diagnostics)
                parse_started = time.perf_counter()
                result = parse_reasoning_result(
                    raw_retry,
                    model_id=self.config.model_id,
                    valid_evidence_ids=valid_evidence_ids,
                    valid_window_ranges=valid_window_ranges,
                )
                diagnostics.parse_validation_seconds = (
                    diagnostics.parse_validation_seconds or 0.0
                ) + (time.perf_counter() - parse_started)
                diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
                return result, diagnostics
            except Exception as second_exc:  # noqa: BLE001
                diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
                self._apply_generation_meta(diagnostics)
                return self._invalid(raw, second_exc, first_error=first_exc), diagnostics

    def _apply_generation_meta(self, diagnostics: TemporalReasonerDiagnostics) -> None:
        meta = self._last_generation_meta or {}
        if meta.get("generation_kwargs") is not None:
            diagnostics.generation_kwargs = dict(meta["generation_kwargs"])
        if meta.get("prompt_tokens") is not None:
            diagnostics.prompt_tokens = int(meta["prompt_tokens"])
        if meta.get("generated_tokens") is not None:
            diagnostics.generated_tokens = int(meta["generated_tokens"])
        if meta.get("sampling_warning_detected"):
            diagnostics.sampling_warning_detected = True

    def _invalid(
        self,
        raw: str,
        exc: Exception,
        *,
        first_error: Optional[Exception] = None,
    ) -> TemporalReasoningResult:
        details: dict[str, Any] = {
            "error": format_validation_error(exc),
            "raw_preview": (raw or "")[:500],
        }
        if first_error is not None:
            details["first_error"] = format_validation_error(first_error)
        return TemporalReasoningResult(
            summary="",
            context_type="uncertain",
            confidence=0.0,
            model=self.config.model_id,
            status="invalid_model_output",
            details=details,
        )

    def _generate(self, system: str, user: str) -> str:
        assert self._tokenizer is not None
        assert self._model is not None
        torch = self._torch
        device = self._device

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        from src.temporal.benchmark.capabilities import (
            chat_template_apply_kwargs,
            resolve_model_capability,
        )

        capability = resolve_model_capability(self.config.model_id)
        apply_kwargs = chat_template_apply_kwargs(
            capability,
            enable_thinking=bool(self.config.enable_thinking),
        )
        try:
            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                **apply_kwargs,
            )
        except TypeError:
            # Older / unexpected chat templates may reject capability kwargs.
            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self._tokenizer([prompt_text], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = self.build_generation_config()
        gen_kwargs.update({
            "max_new_tokens": int(self.config.max_new_tokens),
        })
        if gen_kwargs.get("do_sample", False):
            try:
                gen_kwargs["generator"] = torch.Generator(device=device).manual_seed(
                    int(self.config.seed),
                )
            except Exception:
                gen_kwargs["generator"] = torch.Generator().manual_seed(
                    int(self.config.seed),
                )

        recorded_kwargs = {
            k: v
            for k, v in gen_kwargs.items()
            if k != "generator" and v is not None
        }
        recorded_kwargs["seed"] = int(self.config.seed)
        recorded_kwargs["device"] = device
        recorded_kwargs["model_id"] = self.config.model_id
        recorded_kwargs["chat_template_kwargs"] = {
            k: v for k, v in apply_kwargs.items() if k != "tokenize"
        }

        import warnings

        sampling_warning = False
        with torch.inference_mode():
            clean_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
            if not clean_kwargs.get("do_sample", False):
                for key in ("temperature", "top_p", "top_k", "generator"):
                    clean_kwargs.pop(key, None)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                output_ids = self._model.generate(**inputs, **clean_kwargs)
            for item in caught:
                msg = str(item.message).lower()
                if "temperature" in msg or "top_p" in msg or "top_k" in msg:
                    if "not valid" in msg or "ignored" in msg:
                        sampling_warning = True

        input_len = int(inputs["input_ids"].shape[-1])
        generated = output_ids[0][input_len:]
        self._last_generation_meta = {
            "generation_kwargs": recorded_kwargs,
            "prompt_tokens": input_len,
            "generated_tokens": int(generated.shape[-1]),
            "sampling_warning_detected": sampling_warning,
        }
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def build_generation_config(self) -> dict[str, Any]:
        """Build generation kwargs from explicit config for evaluation switching.

        Explicit ``config.do_sample`` wins. Otherwise infer from temperature /
        top_p / top_k. Evaluation profiles set do_sample=True so sampling
        parameters are not ignored under greedy decoding.
        """
        if self.config.do_sample is None:
            do_sample = (
                float(self.config.temperature) > 0
                or float(self.config.top_p) < 1.0
                or int(self.config.top_k) > 0
            )
        else:
            do_sample = bool(self.config.do_sample)
        return {
            "do_sample": do_sample,
            "temperature": float(self.config.temperature),
            "top_p": float(self.config.top_p),
            "top_k": int(self.config.top_k),
        }


def disabled_reasoning_result(
    *,
    model_id: str = DEFAULT_TEMPORAL_REASONER.model_id,
    reason: str = "disabled",
) -> TemporalReasoningResult:
    return TemporalReasoningResult(
        summary="",
        context_type="uncertain",
        confidence=0.0,
        model=model_id,
        status="disabled",
        details={"reason": reason},
    )
