"""TemporalContextReasoner — text LLM over structured temporal evidence.

Additive only: failures never break deterministic temporal_context or fusion.
Does not receive raw video/images/audio.
"""

from __future__ import annotations

import logging
import re
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

_SECRET_RE = re.compile(
    r"(hf_[A-Za-z0-9]+)|(Bearer\s+\S+)|(token[=:]\s*\S+)|(api[_-]?key[=:]\s*\S+)",
    re.IGNORECASE,
)


def _safe_error_message(exc: BaseException, *, limit: int = 400) -> str:
    """Truncate and redact obvious secret-like substrings for artifact storage."""
    text = str(exc).replace("\n", " ").strip()
    text = _SECRET_RE.sub("[REDACTED]", text)
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def _apply_failure_diagnostics(
    diagnostics: TemporalReasonerDiagnostics,
    exc: BaseException,
    *,
    stage: str,
) -> None:
    diagnostics.reasoner_error_type = type(exc).__name__
    diagnostics.reasoner_error_message = _safe_error_message(exc)
    diagnostics.reasoner_failure_stage = stage


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
        self._failure_stage_hint: str = "unknown"

    def unload(self) -> None:
        """Drop model/tokenizer references. GPU cache clearing is deployment-layer."""
        self._model = None
        self._tokenizer = None
        self._last_generation_meta = {}
        self._load_error = None
        self._failure_stage_hint = "unknown"
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

            self._failure_stage_hint = "tokenizer_load"
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            self._failure_stage_hint = "model_load"
            self._model = AutoModelForCausalLM.from_pretrained(self.config.model_id)
            self._failure_stage_hint = "device_placement"
            self._model.to(device)
            self._model.eval()
            self._device = device
            self._torch = torch
            self._failure_stage_hint = "unknown"
        except Exception as exc:  # noqa: BLE001
            stage = self._failure_stage_hint or "model_load"
            self._load_error = f"Failed to load temporal reasoner: {exc}"
            self._tokenizer = None
            self._model = None
            logger.exception(
                "Temporal reasoner load failed stage=%s model=%s",
                stage,
                self.config.model_id,
            )
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
        diagnostics = TemporalReasonerDiagnostics(provider="qwen_local_or_zerogpu")
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
            stage = self._failure_stage_hint or "model_load"
            if stage not in (
                "tokenizer_load",
                "model_load",
                "device_placement",
            ):
                stage = "model_load"
            _apply_failure_diagnostics(diagnostics, exc, stage=stage)
            diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
            # load() already logs with traceback; keep fail-soft return.
            return (
                TemporalReasoningResult(
                    summary="",
                    context_type="uncertain",
                    confidence=0.0,
                    model=self.config.model_id,
                    status="reasoner_unavailable",
                    details={
                        "error": _safe_error_message(exc),
                        "reasoner_error_type": diagnostics.reasoner_error_type,
                        "reasoner_error_message": diagnostics.reasoner_error_message,
                        "reasoner_failure_stage": diagnostics.reasoner_failure_stage,
                    },
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
            stage = self._failure_stage_hint or "generation"
            if stage not in (
                "prompt_construction",
                "device_placement",
                "generation",
            ):
                stage = "generation"
            logger.exception(
                "Temporal reasoner generation failed stage=%s model=%s",
                stage,
                self.config.model_id,
            )
            _apply_failure_diagnostics(diagnostics, exc, stage=stage)
            diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
            self._apply_generation_meta(diagnostics)
            return (
                TemporalReasoningResult(
                    summary="",
                    context_type="uncertain",
                    confidence=0.0,
                    model=self.config.model_id,
                    status="generation_failed",
                    details={
                        "error": f"generation_failed: {_safe_error_message(exc)}",
                        "reasoner_error_type": diagnostics.reasoner_error_type,
                        "reasoner_error_message": diagnostics.reasoner_error_message,
                        "reasoner_failure_stage": diagnostics.reasoner_failure_stage,
                    },
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
            self._apply_truncation_flags(diagnostics, parse_error=None)
            diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
            return result, diagnostics
        except Exception as first_exc:  # noqa: BLE001
            self._apply_truncation_flags(diagnostics, parse_error=first_exc)
            if self.config.max_retries < 1:
                diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
                _apply_failure_diagnostics(diagnostics, first_exc, stage="parse")
                return self._invalid(raw, first_exc, diagnostics=diagnostics), diagnostics

            repair = build_repair_prompt(
                validation_error=format_validation_error(first_exc),
                previous_output=raw,
            )
            # Do not set repair_attempted until repair generation is invoked.
            try:
                gen_started = time.perf_counter()
                raw_retry = self._generate(SYSTEM_INSTRUCTION, repair)
                repair_s = time.perf_counter() - gen_started
                diagnostics.repair_attempted = True
                diagnostics.repair_generation_seconds = repair_s
                diagnostics.generation_seconds = (
                    diagnostics.generation_seconds or 0.0
                ) + repair_s
                diagnostics.raw_output_preview = raw_retry[:500]
                self._apply_generation_meta(diagnostics)
            except Exception as repair_gen_exc:  # noqa: BLE001
                diagnostics.repair_attempted = True
                diagnostics.repair_generation_seconds = (
                    time.perf_counter() - gen_started
                )
                logger.exception(
                    "Temporal reasoner repair generation failed model=%s",
                    self.config.model_id,
                )
                _apply_failure_diagnostics(
                    diagnostics,
                    repair_gen_exc,
                    stage="generation",
                )
                diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
                self._apply_generation_meta(diagnostics)
                return (
                    TemporalReasoningResult(
                        summary="",
                        context_type="uncertain",
                        confidence=0.0,
                        model=self.config.model_id,
                        status="generation_failed",
                        details={
                            "error": (
                                f"repair_generation_failed: "
                                f"{_safe_error_message(repair_gen_exc)}"
                            ),
                            "first_error": format_validation_error(first_exc),
                            "reasoner_error_type": diagnostics.reasoner_error_type,
                            "reasoner_error_message": diagnostics.reasoner_error_message,
                            "reasoner_failure_stage": diagnostics.reasoner_failure_stage,
                            "output_hit_token_limit": diagnostics.output_hit_token_limit,
                            "likely_output_truncation": diagnostics.likely_output_truncation,
                        },
                    ),
                    diagnostics,
                )

            try:
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
                self._apply_truncation_flags(diagnostics, parse_error=None)
                diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
                return result, diagnostics
            except Exception as second_exc:  # noqa: BLE001
                self._apply_truncation_flags(diagnostics, parse_error=second_exc)
                diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
                self._apply_generation_meta(diagnostics)
                _apply_failure_diagnostics(diagnostics, second_exc, stage="parse")
                return (
                    self._invalid(
                        raw_retry,
                        second_exc,
                        first_error=first_exc,
                        diagnostics=diagnostics,
                    ),
                    diagnostics,
                )

    def _apply_truncation_flags(
        self,
        diagnostics: TemporalReasonerDiagnostics,
        *,
        parse_error: Optional[BaseException],
    ) -> None:
        max_tok = int(self.config.max_new_tokens)
        gen_tok = diagnostics.generated_tokens
        hit = gen_tok is not None and int(gen_tok) >= max_tok
        diagnostics.output_hit_token_limit = bool(hit)
        likely = False
        if hit and parse_error is not None:
            msg = str(parse_error).lower()
            if any(
                needle in msg
                for needle in (
                    "unterminated",
                    "jsondecodeerror",
                    "expecting value",
                    "expecting ','",
                    "expecting property",
                    "invalid control character",
                )
            ):
                likely = True
        diagnostics.likely_output_truncation = bool(likely)

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
        diagnostics: Optional[TemporalReasonerDiagnostics] = None,
    ) -> TemporalReasoningResult:
        details: dict[str, Any] = {
            "error": format_validation_error(exc),
            "raw_preview": (raw or "")[:500],
            "reasoner_failure_stage": "parse",
            "reasoner_error_type": type(exc).__name__,
            "reasoner_error_message": _safe_error_message(exc),
        }
        if first_error is not None:
            details["first_error"] = format_validation_error(first_error)
        if diagnostics is not None:
            details["output_hit_token_limit"] = bool(diagnostics.output_hit_token_limit)
            details["likely_output_truncation"] = bool(
                diagnostics.likely_output_truncation,
            )
            if diagnostics.repair_generation_seconds is not None:
                details["repair_generation_seconds"] = (
                    diagnostics.repair_generation_seconds
                )
        return TemporalReasoningResult(
            summary="",
            context_type="uncertain",
            confidence=0.0,
            model=self.config.model_id,
            status="invalid_model_output",
            details=details,
        )

    def _seed_rng_for_generation(self, torch: Any, *, seed: int, device: str) -> None:
        """Reset global RNG state before each generate() for seeded sampling.

        Does not pass ``generator`` / ``seed`` into ``model.generate`` (rejected
        by current Qwen Transformers paths). Documents seeded sampling only —
        not bit-for-bit deterministic GPU inference.
        """
        torch.manual_seed(int(seed))
        if str(device).startswith("cuda"):
            try:
                torch.cuda.manual_seed_all(int(seed))
            except Exception:  # noqa: BLE001
                pass
        try:
            import random

            random.seed(int(seed))
        except Exception:  # noqa: BLE001
            pass
        try:
            import numpy as np

            np.random.seed(int(seed) % (2**32 - 1))
        except Exception:  # noqa: BLE001
            pass

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
        self._failure_stage_hint = "prompt_construction"
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

        self._failure_stage_hint = "device_placement"
        inputs = self._tokenizer([prompt_text], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = self.build_generation_config()
        gen_kwargs.update({
            "max_new_tokens": int(self.config.max_new_tokens),
        })
        # Qwen / current Transformers reject unused model_kwargs including
        # ``generator`` and ``seed``. Seed via global PyTorch RNG instead —
        # reset before *each* generation for per-fixture reproducibility.
        seed = int(self.config.seed)
        self._seed_rng_for_generation(torch, seed=seed, device=device)

        recorded_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        recorded_kwargs["seed"] = seed
        recorded_kwargs["device"] = device
        recorded_kwargs["model_id"] = self.config.model_id
        recorded_kwargs["chat_template_kwargs"] = {
            k: v for k, v in apply_kwargs.items() if k != "tokenize"
        }
        # Never record unsupported generate kwargs.
        recorded_kwargs.pop("generator", None)

        import warnings

        sampling_warning = False
        self._failure_stage_hint = "generation"
        with torch.inference_mode():
            clean_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
            # Never pass generator/seed into model.generate (rejected by Qwen).
            clean_kwargs.pop("generator", None)
            clean_kwargs.pop("seed", None)
            if not clean_kwargs.get("do_sample", False):
                for key in ("temperature", "top_p", "top_k"):
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
        self._failure_stage_hint = "unknown"
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
