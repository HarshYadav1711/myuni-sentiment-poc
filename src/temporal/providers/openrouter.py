"""OpenRouter temporal reasoner — HTTP Chat Completions with structured outputs.

Uses stdlib urllib only (no OpenAI SDK / LangChain). Never logs API keys.

Failure stages (openrouter_failure_stage):
- payload_build
- request_build
- connection
- http_response
- response_parse
- schema_validation
- unknown
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from enum import Enum
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
from src.temporal.reasoner import _safe_error_message

logger = logging.getLogger(__name__)

# Module-level cap used when parsing Retry-After before config is available.
OPENROUTER_MAX_RETRY_CAP = 5.0

OpenRouterFailureStage = str  # documented enum-like strings above

_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]+)|(Bearer\s+\S+)|(api[_-]?key[=:]\s*\S+)|(or-[A-Za-z0-9_-]+)",
    re.IGNORECASE,
)


def _redact(text: str) -> str:
    return _SECRET_RE.sub("[REDACTED]", text or "")


def sanitize_openrouter_error_message(exc: BaseException, *, limit: int = 400) -> str:
    """Public sanitizer for diagnostics / UI — never includes secrets."""
    return _redact(_safe_error_message(exc, limit=limit))


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


def to_jsonable(value: Any) -> Any:
    """Convert nested values to plain JSON-serializable Python types.

    Handles numpy scalars, Pydantic models, Enums, datetime, sets, and
    non-finite floats (replaced with None). Does not stringify arbitrary objects.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, set):
        return [to_jsonable(v) for v in sorted(value, key=lambda x: str(x))]
    # Pydantic v2
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return to_jsonable(model_dump(mode="json"))
    # numpy scalar / array-ish
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except Exception:  # noqa: BLE001
            pass
    # Reject unknown objects rather than str()-ing them.
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def assert_json_serializable(payload: Any) -> bytes:
    """Serialize payload to UTF-8 JSON bytes or raise TypeError/ValueError."""
    jsonable = to_jsonable(payload)
    return json.dumps(jsonable, ensure_ascii=False, allow_nan=False).encode("utf-8")


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
        failure_stage: str = "unknown",
        response_body_preview: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after
        self.error_kind = error_kind
        self.retry_attempted = retry_attempted
        self.failure_stage = failure_stage
        self.response_body_preview = response_body_preview


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


def _safe_http_body_preview(raw: str, *, limit: int = 500) -> str:
    text = _redact((raw or "").replace("\n", " ").strip())
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def post_openrouter_chat_completion(
    body: dict[str, Any],
    *,
    api_key: str,
    api_url: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """POST once to OpenRouter. Raises OpenRouterError on failure.

    Stages:
    - payload_build: JSON serialization
    - request_build: urllib Request construction
    - connection: DNS/TLS/network before HTTP response
    - http_response: non-2xx HTTP
    - response_parse: response body JSON decode
    """
    # --- payload_build ---
    try:
        payload = assert_json_serializable(body)
    except Exception as exc:  # noqa: BLE001
        raise OpenRouterError(
            f"Request payload JSON serialization failed: {type(exc).__name__}",
            retryable=False,
            error_kind="payload_serialization",
            failure_stage="payload_build",
        ) from exc

    # --- request_build ---
    try:
        request = urllib.request.Request(
            api_url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "HTTP-Referer": "https://huggingface.co/spaces/myuni-temporal-demo",
                "X-Title": "MyUni Temporal Demo",
            },
        )
        # Sanity: POST + body present.
        if (request.get_method() or "").upper() != "POST":
            raise RuntimeError("urllib Request method is not POST")
        if request.data is None:
            raise RuntimeError("urllib Request data is empty")
    except OpenRouterError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise OpenRouterError(
            f"urllib Request construction failed: {type(exc).__name__}",
            retryable=False,
            error_kind="request_build",
            failure_stage="request_build",
        ) from exc

    # --- network I/O ---
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
        preview = _safe_http_body_preview(raw)
        retry_after = _parse_retry_after(
            getattr(exc, "headers", None),
            max_wait=OPENROUTER_MAX_RETRY_CAP,
        )
        body_lower = (raw or "").lower()
        if status in (401, 403):
            raise OpenRouterError(
                f"OpenRouter authentication failed ({status}): {preview or 'no body'}",
                status_code=status,
                retryable=False,
                error_kind="auth",
                failure_stage="http_response",
                response_body_preview=preview,
            ) from exc
        if status == 429:
            raise OpenRouterError(
                f"OpenRouter rate limited (429): {preview or 'no body'}",
                status_code=status,
                retryable=True,
                retry_after=retry_after,
                error_kind="rate_limit",
                failure_stage="http_response",
                response_body_preview=preview,
            ) from exc
        if status >= 500:
            kind = "free_endpoint_unavailable" if status in (502, 503, 504) else "server_error"
            raise OpenRouterError(
                f"OpenRouter upstream failure ({status}): {preview or 'no body'}",
                status_code=status,
                retryable=True,
                retry_after=retry_after,
                error_kind=kind,
                failure_stage="http_response",
                response_body_preview=preview,
            ) from exc
        if status == 400 and (
            "response_format" in body_lower
            or "json_schema" in body_lower
            or "require_parameters" in body_lower
            or "structured" in body_lower
        ):
            raise OpenRouterError(
                f"Structured-output unsupported by routed provider: {preview or 'no body'}",
                status_code=status,
                retryable=False,
                error_kind="structured_output_unsupported",
                failure_stage="http_response",
                response_body_preview=preview,
            ) from exc
        raise OpenRouterError(
            f"OpenRouter HTTP error ({status}): {preview or 'no body'}",
            status_code=status,
            retryable=False,
            error_kind="http_error",
            failure_stage="http_response",
            response_body_preview=preview,
        ) from exc
    except TimeoutError as exc:
        raise OpenRouterError(
            "OpenRouter request timed out",
            retryable=False,
            error_kind="timeout",
            failure_stage="connection",
        ) from exc
    except urllib.error.URLError as exc:
        reason_obj = getattr(exc, "reason", exc)
        reason = str(reason_obj)
        reason_l = reason.lower()
        if "timed out" in reason_l or "timeout" in reason_l:
            raise OpenRouterError(
                f"OpenRouter request timed out: {_redact(reason)[:200]}",
                retryable=False,
                error_kind="timeout",
                failure_stage="connection",
            ) from exc
        # SSL / certificate / DNS / refused — still connection stage.
        if isinstance(reason_obj, ssl.SSLError) or "ssl" in reason_l or "certificate" in reason_l:
            raise OpenRouterError(
                f"OpenRouter TLS/SSL connection failed: {_redact(reason)[:200]}",
                retryable=False,
                error_kind="ssl",
                failure_stage="connection",
            ) from exc
        raise OpenRouterError(
            f"OpenRouter connection failed: {_redact(reason)[:200]}",
            retryable=False,
            error_kind="connection",
            failure_stage="connection",
        ) from exc
    except ssl.SSLError as exc:
        raise OpenRouterError(
            f"OpenRouter TLS/SSL error: {_redact(str(exc))[:200]}",
            retryable=False,
            error_kind="ssl",
            failure_stage="connection",
        ) from exc
    except OSError as exc:
        # Includes socket errors not wrapped as URLError on some platforms.
        raise OpenRouterError(
            f"OpenRouter network OSError: {type(exc).__name__}: {_redact(str(exc))[:200]}",
            retryable=False,
            error_kind="connection",
            failure_stage="connection",
        ) from exc

    if status < 200 or status >= 300:
        raise OpenRouterError(
            f"OpenRouter unexpected status ({status})",
            status_code=status,
            retryable=False,
            error_kind="http_error",
            failure_stage="http_response",
        )

    # --- response_parse ---
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OpenRouterError(
            "OpenRouter returned invalid JSON",
            status_code=status,
            retryable=False,
            error_kind="invalid_response",
            failure_stage="response_parse",
            response_body_preview=_safe_http_body_preview(raw),
        ) from exc
    if not isinstance(data, dict):
        raise OpenRouterError(
            "OpenRouter response root must be an object",
            status_code=status,
            retryable=False,
            error_kind="invalid_response",
            failure_stage="response_parse",
        )
    return data


def extract_message_content(response: dict[str, Any]) -> str:
    """Pull assistant message content from a Chat Completions response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenRouterError(
            "OpenRouter response missing choices",
            error_kind="missing_content",
            failure_stage="response_parse",
        )
    first = choices[0]
    if not isinstance(first, dict):
        raise OpenRouterError(
            "OpenRouter choice malformed",
            error_kind="missing_content",
            failure_stage="response_parse",
        )
    message = first.get("message")
    if not isinstance(message, dict):
        raise OpenRouterError(
            "OpenRouter message missing",
            error_kind="missing_content",
            failure_stage="response_parse",
        )
    content = message.get("content")
    if content is None:
        raise OpenRouterError(
            "OpenRouter message content missing",
            error_kind="missing_content",
            failure_stage="response_parse",
        )
    if isinstance(content, list):
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
            failure_stage="response_parse",
        )
    return text


def _apply_openrouter_failure(
    diagnostics: TemporalReasonerDiagnostics,
    exc: BaseException,
    *,
    stage: str,
) -> None:
    diagnostics.openrouter_failure_stage = stage
    diagnostics.reasoner_failure_stage = stage
    diagnostics.openrouter_error_type = type(exc).__name__
    diagnostics.reasoner_error_type = type(exc).__name__
    msg = sanitize_openrouter_error_message(exc)
    diagnostics.openrouter_error_message = msg
    diagnostics.reasoner_error_message = msg
    if isinstance(exc, OpenRouterError):
        diagnostics.openrouter_http_status = exc.status_code
        diagnostics.http_status = exc.status_code
        if exc.response_body_preview:
            diagnostics.openrouter_response_preview = _redact(exc.response_body_preview)


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
            diagnostics.openrouter_failure_stage = "unknown"
            diagnostics.reasoner_failure_stage = "auth"
            diagnostics.openrouter_error_type = "MissingAPIKey"
            diagnostics.reasoner_error_type = "MissingAPIKey"
            diagnostics.openrouter_error_message = "OPENROUTER_API_KEY not configured"
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
                        "openrouter_failure_stage": "unknown",
                    },
                ),
                diagnostics,
            )

        # --- payload / prompt construction ---
        try:
            prompt_started = time.perf_counter()
            evidence = build_evidence_payload(
                temporal,
                baseline_overall=baseline_overall,
                config=self.config,
            )
            # Ensure evidence itself is JSON-serializable before embedding.
            evidence = to_jsonable(evidence)
            user_prompt = build_user_prompt(evidence)
            body = build_openrouter_request_body(
                model_id=self.config.model_id,
                system=OPENROUTER_SYSTEM_INSTRUCTION,
                user=user_prompt,
                max_tokens=int(self.config.max_new_tokens),
            )
            # Preflight serialize (catches schema / numpy / set issues before I/O).
            assert_json_serializable(body)
            diagnostics.prompt_construction_seconds = time.perf_counter() - prompt_started
            diagnostics.prompt_chars = len(user_prompt)
            diagnostics.prompt_windows_included = int(evidence.get("windows_included", 0))
            diagnostics.evidence_ids_supplied = collect_valid_evidence_ids(evidence)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "OpenRouter payload_build failed before HTTP request model=%s",
                self.config.model_id,
            )
            _apply_openrouter_failure(diagnostics, exc, stage="payload_build")
            diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
            return (
                TemporalReasoningResult(
                    summary="",
                    context_type="uncertain",
                    confidence=0.0,
                    model=self.config.model_id,
                    status="generation_failed",
                    details={
                        "error": sanitize_openrouter_error_message(exc),
                        "error_kind": "payload_serialization",
                        "openrouter_failure_stage": "payload_build",
                        "provider": "openrouter",
                        "reasoner_configured": True,
                    },
                ),
                diagnostics,
            )

        valid_evidence_ids = set(diagnostics.evidence_ids_supplied)
        valid_window_ranges = [
            (float(window.start), float(window.end))
            for window in temporal.windows
        ]

        # Safe meta only — never store Authorization or API key.
        diagnostics.generation_kwargs = {
            "model": self.config.model_id,
            "api_url": self.config.openrouter_api_url,
            "response_format_type": "json_schema",
            "json_schema_strict": True,
            "provider_require_parameters": True,
            "max_tokens": int(self.config.max_new_tokens),
            "method": "POST",
            "content_type": "application/json",
        }

        try:
            raw_text, http_meta = self._generate_with_retry(body, api_key=api_key)
            diagnostics.http_status = http_meta.get("http_status")
            diagnostics.openrouter_http_status = http_meta.get("http_status")
            diagnostics.retry_attempted = bool(http_meta.get("retry_attempted"))
            diagnostics.generation_seconds = http_meta.get("generation_seconds")
            diagnostics.raw_output_preview = _redact(raw_text[:500])
            usage = http_meta.get("usage") or {}
            if usage.get("prompt_tokens") is not None:
                diagnostics.prompt_tokens = int(usage["prompt_tokens"])
            if usage.get("completion_tokens") is not None:
                diagnostics.generated_tokens = int(usage["completion_tokens"])
        except OpenRouterError as exc:
            stage = exc.failure_stage or "unknown"
            logger.exception(
                "OpenRouter generation failed stage=%s kind=%s status=%s model=%s",
                stage,
                exc.error_kind,
                exc.status_code,
                self.config.model_id,
            )
            _apply_openrouter_failure(diagnostics, exc, stage=stage)
            diagnostics.retry_attempted = bool(exc.retry_attempted)
            diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
            if exc.error_kind == "auth" or exc.error_kind in {
                "connection",
                "ssl",
                "free_endpoint_unavailable",
            }:
                status = "reasoner_unavailable"
            else:
                # timeout, http_error, payload_*, request_build, structured_output, etc.
                status = "generation_failed"
            return (
                TemporalReasoningResult(
                    summary="",
                    context_type="uncertain",
                    confidence=0.0,
                    model=self.config.model_id,
                    status=status,
                    details={
                        "error": sanitize_openrouter_error_message(exc),
                        "error_kind": exc.error_kind,
                        "http_status": exc.status_code,
                        "openrouter_failure_stage": stage,
                        "openrouter_error_type": type(exc).__name__,
                        "openrouter_error_message": sanitize_openrouter_error_message(exc),
                        "openrouter_http_status": exc.status_code,
                        "provider": "openrouter",
                        "reasoner_configured": True,
                    },
                ),
                diagnostics,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "OpenRouter unexpected generation failure model=%s",
                self.config.model_id,
            )
            _apply_openrouter_failure(diagnostics, exc, stage="unknown")
            diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
            return (
                TemporalReasoningResult(
                    summary="",
                    context_type="uncertain",
                    confidence=0.0,
                    model=self.config.model_id,
                    status="generation_failed",
                    details={
                        "error": sanitize_openrouter_error_message(exc),
                        "openrouter_failure_stage": "unknown",
                        "openrouter_error_type": type(exc).__name__,
                        "openrouter_error_message": sanitize_openrouter_error_message(exc),
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
            logger.exception(
                "OpenRouter schema_validation failed model=%s",
                self.config.model_id,
            )
            _apply_openrouter_failure(diagnostics, exc, stage="schema_validation")
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
                        "openrouter_failure_stage": "schema_validation",
                        "openrouter_error_type": type(exc).__name__,
                        "openrouter_error_message": sanitize_openrouter_error_message(exc),
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
                        failure_stage=exc.failure_stage,
                        response_body_preview=exc.response_body_preview,
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
