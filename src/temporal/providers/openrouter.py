"""OpenRouter temporal reasoner — HTTP Chat Completions with structured outputs.

Uses stdlib urllib only (no OpenAI SDK / LangChain). Never logs API keys.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
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
    build_evidence_payload,
    build_user_prompt,
    collect_valid_evidence_ids,
)
from src.temporal.providers.openrouter_schema import (
    OPENROUTER_SYSTEM_INSTRUCTION,
    OPENROUTER_TEMPORAL_REASONING_SCHEMA,
)
from src.temporal.reasoner import _apply_failure_diagnostics, _safe_error_message

logger = logging.getLogger(__name__)

# Module-level cap used when parsing Retry-After before config is available.
OPENROUTER_MAX_RETRY_CAP = 5.0

_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]+)|(Bearer\s+\S+)|(api[_-]?key[=:]\s*\S+)|(or-[A-Za-z0-9_-]+)",
    re.IGNORECASE,
)


def _redact(text: str) -> str:
    return _SECRET_RE.sub("[REDACTED]", text or "")


def openrouter_api_key_configured(api_key: Optional[str] = None) -> bool:
    """Safe configured/unconfigured check — never returns the secret."""
    key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
    return bool(key and str(key).strip())


def resolve_openrouter_api_key() -> Optional[str]:
    """Read OPENROUTER_API_KEY from the environment only."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key is None:
        return None
    cleaned = str(key).strip()
    return cleaned or None


class OpenRouterError(Exception):
    """Fail-soft transport / API error with safe metadata."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retryable: bool = False,
        retry_after: Optional[float] = None,
        error_kind: str = "upstream",
        retry_attempted: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after
        self.error_kind = error_kind
        self.retry_attempted = retry_attempted


def build_openrouter_request_body(
    *,
    model_id: str,
    system: str,
    user: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Exact Chat Completions payload shape for the temporal reasoner."""
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "temporal_reasoning_result",
                "strict": True,
                "schema": OPENROUTER_TEMPORAL_REASONING_SCHEMA,
            },
        },
        "provider": {
            "require_parameters": True,
        },
    }


def _parse_retry_after(headers: Any, *, max_wait: float) -> Optional[float]:
    if headers is None:
        return None
    raw = None
    try:
        raw = headers.get("Retry-After") or headers.get("retry-after")
    except Exception:  # noqa: BLE001
        return None
    if raw is None:
        return None
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return min(seconds, max_wait)


def post_openrouter_chat_completion(
    body: dict[str, Any],
    *,
    api_key: str,
    api_url: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """POST once to OpenRouter. Raises OpenRouterError on failure."""
    # Never include the key in exception messages or logs.
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "HTTP-Referer": "https://myuni.local/temporal-demo",
            "X-Title": "MyUni Temporal Demo",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            raw = ""
        retry_after = _parse_retry_after(
            getattr(exc, "headers", None),
            max_wait=OPENROUTER_MAX_RETRY_CAP,
        )
        body_lower = (raw or "").lower()
        if status in (401, 403):
            raise OpenRouterError(
                "OpenRouter authentication failed",
                status_code=status,
                retryable=False,
                error_kind="auth",
            ) from exc
        if status == 429:
            raise OpenRouterError(
                "OpenRouter rate limited",
                status_code=status,
                retryable=True,
                retry_after=retry_after,
                error_kind="rate_limit",
            ) from exc
        if status >= 500:
            # Free endpoint unavailable often surfaces as 5xx / 503.
            kind = "free_endpoint_unavailable" if status in (502, 503, 504) else "server_error"
            raise OpenRouterError(
                f"OpenRouter upstream failure ({status})",
                status_code=status,
                retryable=True,
                retry_after=retry_after,
                error_kind=kind,
            ) from exc
        if status == 400 and (
            "response_format" in body_lower
            or "json_schema" in body_lower
            or "require_parameters" in body_lower
            or "structured" in body_lower
        ):
            raise OpenRouterError(
                "Structured-output unsupported by routed provider",
                status_code=status,
                retryable=False,
                error_kind="structured_output_unsupported",
            ) from exc
        raise OpenRouterError(
            f"OpenRouter HTTP error ({status})",
            status_code=status,
            retryable=False,
            error_kind="http_error",
        ) from exc
    except TimeoutError as exc:
        raise OpenRouterError(
            "OpenRouter request timed out",
            retryable=False,
            error_kind="timeout",
        ) from exc
    except urllib.error.URLError as exc:
        reason = str(getattr(exc, "reason", exc))
        if "timed out" in reason.lower() or "timeout" in reason.lower():
            raise OpenRouterError(
                "OpenRouter request timed out",
                retryable=False,
                error_kind="timeout",
            ) from exc
        raise OpenRouterError(
            "OpenRouter connection failed",
            retryable=False,
            error_kind="connection",
        ) from exc

    if status < 200 or status >= 300:
        raise OpenRouterError(
            f"OpenRouter unexpected status ({status})",
            status_code=status,
            retryable=False,
            error_kind="http_error",
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OpenRouterError(
            "OpenRouter returned invalid JSON",
            status_code=status,
            retryable=False,
            error_kind="invalid_response",
        ) from exc
    if not isinstance(data, dict):
        raise OpenRouterError(
            "OpenRouter response root must be an object",
            status_code=status,
            retryable=False,
            error_kind="invalid_response",
        )
    return data


def extract_message_content(response: dict[str, Any]) -> str:
    """Pull assistant message content from a Chat Completions response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenRouterError(
            "OpenRouter response missing choices",
            error_kind="missing_content",
        )
    first = choices[0]
    if not isinstance(first, dict):
        raise OpenRouterError(
            "OpenRouter choice malformed",
            error_kind="missing_content",
        )
    message = first.get("message")
    if not isinstance(message, dict):
        raise OpenRouterError(
            "OpenRouter message missing",
            error_kind="missing_content",
        )
    content = message.get("content")
    if content is None:
        raise OpenRouterError(
            "OpenRouter message content missing",
            error_kind="missing_content",
        )
    if isinstance(content, list):
        # Some providers return content parts.
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                texts.append(part)
        content = "".join(texts)
    text = str(content).strip()
    if not text:
        raise OpenRouterError(
            "OpenRouter message content empty",
            error_kind="missing_content",
        )
    return text


class OpenRouterTemporalReasoner:
    """HTTP temporal reasoner using OpenRouter structured outputs."""

    def __init__(self, config: TemporalReasonerConfig = DEFAULT_TEMPORAL_REASONER) -> None:
        self.config = config
        self._last_generation_meta: dict[str, Any] = {}

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def is_configured(self) -> bool:
        return openrouter_api_key_configured()

    def reason(
        self,
        temporal: TemporalContext,
        *,
        baseline_overall: Optional[SentimentEvidence] = None,
    ) -> tuple[TemporalReasoningResult, TemporalReasonerDiagnostics]:
        total_started = time.perf_counter()
        diagnostics = TemporalReasonerDiagnostics(
            provider="openrouter",
            reasoner_configured=self.is_configured,
        )
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

        api_key = resolve_openrouter_api_key()
        diagnostics.reasoner_configured = bool(api_key)
        if not api_key:
            diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
            diagnostics.reasoner_failure_stage = "auth"
            diagnostics.reasoner_error_type = "MissingAPIKey"
            diagnostics.reasoner_error_message = "OPENROUTER_API_KEY not configured"
            logger.warning("OpenRouter reasoner unavailable: API key not configured")
            return (
                TemporalReasoningResult(
                    summary="",
                    context_type="uncertain",
                    confidence=0.0,
                    model=self.config.model_id,
                    status="reasoner_unavailable",
                    details={
                        "error": "OPENROUTER_API_KEY not configured",
                        "reasoner_configured": False,
                        "provider": "openrouter",
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

        body = build_openrouter_request_body(
            model_id=self.config.model_id,
            system=OPENROUTER_SYSTEM_INSTRUCTION,
            user=user_prompt,
            max_tokens=int(self.config.max_new_tokens),
        )
        # Safe meta only — never store Authorization or API key.
        diagnostics.generation_kwargs = {
            "model": self.config.model_id,
            "api_url": self.config.openrouter_api_url,
            "response_format_type": "json_schema",
            "json_schema_strict": True,
            "provider_require_parameters": True,
            "max_tokens": int(self.config.max_new_tokens),
        }

        try:
            raw_text, http_meta = self._generate_with_retry(body, api_key=api_key)
            diagnostics.http_status = http_meta.get("http_status")
            diagnostics.retry_attempted = bool(http_meta.get("retry_attempted"))
            diagnostics.generation_seconds = http_meta.get("generation_seconds")
            diagnostics.raw_output_preview = _redact(raw_text[:500])
            usage = http_meta.get("usage") or {}
            if usage.get("prompt_tokens") is not None:
                diagnostics.prompt_tokens = int(usage["prompt_tokens"])
            if usage.get("completion_tokens") is not None:
                diagnostics.generated_tokens = int(usage["completion_tokens"])
        except OpenRouterError as exc:
            logger.warning(
                "OpenRouter generation failed kind=%s status=%s",
                exc.error_kind,
                exc.status_code,
            )
            _apply_failure_diagnostics(diagnostics, exc, stage="generation")
            diagnostics.http_status = exc.status_code
            diagnostics.retry_attempted = bool(exc.retry_attempted)
            diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
            status = (
                "reasoner_unavailable"
                if exc.error_kind in {"auth", "connection", "free_endpoint_unavailable"}
                else "generation_failed"
            )
            return (
                TemporalReasoningResult(
                    summary="",
                    context_type="uncertain",
                    confidence=0.0,
                    model=self.config.model_id,
                    status=status,
                    details={
                        "error": _safe_error_message(exc),
                        "error_kind": exc.error_kind,
                        "http_status": exc.status_code,
                        "provider": "openrouter",
                        "reasoner_configured": True,
                    },
                ),
                diagnostics,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("OpenRouter unexpected generation failure")
            _apply_failure_diagnostics(diagnostics, exc, stage="generation")
            diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
            return (
                TemporalReasoningResult(
                    summary="",
                    context_type="uncertain",
                    confidence=0.0,
                    model=self.config.model_id,
                    status="generation_failed",
                    details={
                        "error": _safe_error_message(exc),
                        "provider": "openrouter",
                    },
                ),
                diagnostics,
            )

        try:
            parse_started = time.perf_counter()
            result = parse_reasoning_result(
                raw_text,
                model_id=self.config.model_id,
                valid_evidence_ids=valid_evidence_ids,
                valid_window_ranges=valid_window_ranges,
            )
            diagnostics.parse_validation_seconds = time.perf_counter() - parse_started
            diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
            return result, diagnostics
        except Exception as exc:  # noqa: BLE001
            _apply_failure_diagnostics(diagnostics, exc, stage="parse")
            diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
            return (
                TemporalReasoningResult(
                    summary="",
                    context_type="uncertain",
                    confidence=0.0,
                    model=self.config.model_id,
                    status="invalid_model_output",
                    details={
                        "error": format_validation_error(exc),
                        "raw_preview": _redact(raw_text[:500]),
                        "provider": "openrouter",
                    },
                ),
                diagnostics,
            )

    def _generate_with_retry(
        self,
        body: dict[str, Any],
        *,
        api_key: str,
    ) -> tuple[str, dict[str, Any]]:
        """One normal request + at most one bounded retry for transient 429/5xx."""
        max_retries = int(self.config.openrouter_max_transient_retries)
        attempt = 0
        retry_attempted = False
        started = time.perf_counter()
        while True:
            try:
                response = post_openrouter_chat_completion(
                    body,
                    api_key=api_key,
                    api_url=self.config.openrouter_api_url,
                    timeout_seconds=self.config.openrouter_timeout_seconds,
                )
                content = extract_message_content(response)
                usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
                return content, {
                    "http_status": 200,
                    "retry_attempted": retry_attempted,
                    "generation_seconds": time.perf_counter() - started,
                    "usage": usage,
                }
            except OpenRouterError as exc:
                can_retry = (
                    exc.retryable
                    and attempt < max_retries
                    and exc.error_kind
                    in {
                        "rate_limit",
                        "server_error",
                        "free_endpoint_unavailable",
                    }
                )
                if not can_retry:
                    raise OpenRouterError(
                        str(exc),
                        status_code=exc.status_code,
                        retryable=False,
                        retry_after=exc.retry_after,
                        error_kind=exc.error_kind,
                        retry_attempted=retry_attempted,
                    ) from exc
                attempt += 1
                retry_attempted = True
                wait = exc.retry_after
                if wait is None:
                    wait = 0.5
                wait = min(
                    float(wait),
                    float(self.config.openrouter_max_retry_after_seconds),
                )
                logger.warning(
                    "OpenRouter transient error; bounded retry attempt=%s wait=%.2fs",
                    attempt,
                    wait,
                )
                if wait > 0:
                    time.sleep(wait)
