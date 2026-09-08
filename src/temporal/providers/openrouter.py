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
from typing import Any, Optional, Sequence

from src.config import (
    DEFAULT_TEMPORAL_REASONER,
    OPENROUTER_MAX_HTTP_REQUESTS_PER_CALL,
    OPENROUTER_REASONER_MAX_TOKENS,
    TemporalReasonerConfig,
)
from src.schemas import (
    SentimentEvidence,
    TemporalContext,
    TemporalReasonerDiagnostics,
    TemporalReasoningResult,
)
from src.temporal.parse import format_validation_error, parse_reasoning_result
from src.temporal.prompt import (
    build_evidence_payload,
    build_repair_prompt,
    build_user_prompt,
    collect_valid_evidence_ids,
)
from src.temporal.providers.openrouter_schema import OPENROUTER_SYSTEM_INSTRUCTION
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
        response_shape: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after
        self.error_kind = error_kind
        self.retry_attempted = retry_attempted
        self.failure_stage = failure_stage
        self.response_body_preview = response_body_preview
        # Safe structural summary only — never raw content / reasoning text.
        self.response_shape = response_shape


def normalize_openrouter_fallback_models(
    fallback_models: Optional[Sequence[str]],
    *,
    primary: str,
) -> list[str]:
    """Deduplicate fallbacks and exclude the primary model id."""
    primary_id = (primary or "").strip()
    seen: set[str] = {primary_id} if primary_id else set()
    out: list[str] = []
    for model in fallback_models or []:
        cleaned = str(model).strip()
        if not cleaned or cleaned in seen:
            continue
        out.append(cleaned)
        seen.add(cleaned)
    return out


def build_openrouter_model_chain(
    *,
    primary: str,
    fallback_models: Optional[Sequence[str]] = None,
) -> list[str]:
    """Ordered free-model chain for raw OpenRouter REST ``models`` field.

    Deduplicates while preserving priority: primary first, then fallbacks.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for model in [primary, *(fallback_models or [])]:
        cleaned = str(model or "").strip()
        if not cleaned or cleaned in seen:
            continue
        ordered.append(cleaned)
        seen.add(cleaned)
    return ordered


def build_openrouter_request_body(
    *,
    model_id: str,
    system: str,
    user: str,
    max_tokens: int,
    fallback_models: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Exact Chat Completions payload shape for the temporal reasoner.

    Free-model path uses ``json_object`` (not remote json_schema). Strict
    TemporalReasoningResult validation remains local after the response.

    Raw urllib client sends OpenRouter's documented ordered ``models`` array
    only — no separate top-level ``model`` field — so fallbacks are tried when
    an earlier free model is unavailable / rate-limited.

    Reasoning is minimized and excluded from the returned message: the reasoner
    receives deterministic evidence and must emit final JSON in ``message.content``.
    """
    ordered_models = build_openrouter_model_chain(
        primary=model_id,
        fallback_models=fallback_models,
    )
    if not ordered_models:
        raise ValueError("OpenRouter model chain is empty")
    return {
        "models": ordered_models,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "response_format": {
            "type": "json_object",
        },
        "provider": {
            "require_parameters": True,
        },
        "reasoning": {
            "effort": "minimal",
            "exclude": True,
        },
    }


def remaining_models_after_routed(
    model_chain: Sequence[str],
    routed_model: Optional[str],
) -> list[str]:
    """Models strictly after the routed (or primary) model — never cycle backward."""
    chain = [str(m).strip() for m in model_chain if str(m).strip()]
    if not chain:
        return []
    routed = (routed_model or "").strip()
    if routed:
        for idx, model in enumerate(chain):
            if model == routed:
                return list(chain[idx + 1 :])
        # Remapped provider id: drop the first requested model only.
        return list(chain[1:])
    return list(chain[1:])


def is_application_fallback_eligible(exc: BaseException) -> bool:
    """True when a 2xx response was unusable (null/empty content), not transport/auth."""
    if not isinstance(exc, OpenRouterError):
        return False
    if exc.error_kind != "missing_content":
        return False
    if exc.status_code is not None and int(exc.status_code) != 200:
        return False
    return True


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
        if status == 404 or (
            status == 400
            and (
                "unavailable for free" in body_lower
                or "model is unavailable" in body_lower
                or "no endpoints found" in body_lower
            )
        ):
            raise OpenRouterError(
                f"OpenRouter model unavailable ({status}): {preview or 'no body'}",
                status_code=status,
                retryable=False,
                error_kind="model_unavailable",
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


def extract_routed_model(response: dict[str, Any]) -> Optional[str]:
    """Return the actual model OpenRouter reports, if present.

    Does not invent a model ID. Prefer top-level ``model``; otherwise None.
    """
    model = response.get("model")
    if isinstance(model, str):
        cleaned = model.strip()
        if cleaned:
            return cleaned
    return None


def _content_structural_meta(content: Any) -> tuple[bool, Optional[str], Optional[int]]:
    """Return (content_present, content_type, content_length) without storing text."""
    if content is None:
        return False, "null", None
    if isinstance(content, str):
        return bool(content.strip()), "str", len(content)
    if isinstance(content, list):
        length = 0
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                length += len(str(part.get("text") or ""))
            elif isinstance(part, str):
                length += len(part)
        return length > 0, "list", length
    # Unexpected content types: report type only; never stringify payload into diagnostics.
    return True, type(content).__name__, None


def _reasoning_structural_meta(message: dict[str, Any]) -> tuple[bool, Optional[int], bool, Optional[int]]:
    """Presence/length only for reasoning fields — never store reasoning text."""
    reasoning_present = False
    reasoning_length: Optional[int] = None
    reasoning_details_present = False
    reasoning_details_count: Optional[int] = None

    for key in ("reasoning", "reasoning_content"):
        val = message.get(key)
        if isinstance(val, str) and val:
            reasoning_present = True
            reasoning_length = (reasoning_length or 0) + len(val)
        elif val is not None:
            # Non-string reasoning payload: mark present, do not coerce to text.
            reasoning_present = True

    details = message.get("reasoning_details")
    if isinstance(details, list):
        reasoning_details_present = True
        reasoning_details_count = len(details)
        if details:
            reasoning_present = True
    elif details is not None:
        reasoning_details_present = True
        reasoning_present = True

    return (
        reasoning_present,
        reasoning_length,
        reasoning_details_present,
        reasoning_details_count,
    )


def summarize_openrouter_response_shape(
    response: dict[str, Any],
    *,
    http_status: Optional[int] = None,
) -> dict[str, Any]:
    """Safe structural summary of a Chat Completions response.

    NEVER stores Authorization, API keys, prompts, message content text,
    reasoning text, or reasoning_details payloads.
    """
    top_keys = sorted(str(k) for k in response.keys())
    choices = response.get("choices")
    choices_count = len(choices) if isinstance(choices, list) else 0

    finish_reason: Optional[str] = None
    native_finish_reason: Optional[str] = None
    message_keys: Optional[list[str]] = None
    content_present = False
    content_type: Optional[str] = None
    content_length: Optional[int] = None
    reasoning_present = False
    reasoning_length: Optional[int] = None
    reasoning_details_present = False
    reasoning_details_count: Optional[int] = None

    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        first = choices[0]
        fr = first.get("finish_reason")
        if isinstance(fr, str) and fr.strip():
            finish_reason = fr.strip()
        nfr = first.get("native_finish_reason")
        if isinstance(nfr, str) and nfr.strip():
            native_finish_reason = nfr.strip()
        message = first.get("message")
        if isinstance(message, dict):
            message_keys = sorted(str(k) for k in message.keys())
            content_present, content_type, content_length = _content_structural_meta(
                message.get("content"),
            )
            (
                reasoning_present,
                reasoning_length,
                reasoning_details_present,
                reasoning_details_count,
            ) = _reasoning_structural_meta(message)

    usage_out: dict[str, int] = {}
    usage = response.get("usage")
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            raw = usage.get(key)
            if raw is None:
                continue
            try:
                usage_out[key] = int(raw)
            except (TypeError, ValueError):
                continue
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
            try:
                usage_out["reasoning_tokens"] = int(details["reasoning_tokens"])
            except (TypeError, ValueError):
                pass

    resp_id = response.get("id")
    response_id_prefix: Optional[str] = None
    if isinstance(resp_id, str) and resp_id.strip():
        response_id_prefix = resp_id.strip()[:16]

    shape: dict[str, Any] = {
        "http_status": http_status,
        "top_level_keys": top_keys,
        "routed_model": extract_routed_model(response),
        "choices_count": choices_count,
        "finish_reason": finish_reason,
        "native_finish_reason": native_finish_reason,
        "message_keys": message_keys,
        "content_present": content_present,
        "content_type": content_type,
        "content_length": content_length,
        "reasoning_present": reasoning_present,
        "reasoning_length": reasoning_length,
        "reasoning_details_present": reasoning_details_present,
        "reasoning_details_count": reasoning_details_count,
        "usage": usage_out or None,
        "response_id_prefix": response_id_prefix,
    }
    return shape


def format_missing_content_error(
    shape: dict[str, Any],
    *,
    empty: bool = False,
) -> str:
    """Compact structural missing/empty-content error — never includes reasoning text."""
    label = "empty_content" if empty else "missing_content"
    parts: list[str] = [f"{label}:"]
    model = shape.get("routed_model")
    if isinstance(model, str) and model.strip():
        parts.append(f"model={model.strip()}")
    finish = shape.get("finish_reason")
    if isinstance(finish, str) and finish.strip():
        parts.append(f"finish_reason={finish.strip()}")
    native = shape.get("native_finish_reason")
    if isinstance(native, str) and native.strip():
        parts.append(f"native_finish_reason={native.strip()}")
    parts.append(f"content_type={shape.get('content_type')}")
    parts.append(f"content_present={bool(shape.get('content_present'))}")
    if shape.get("content_length") is not None:
        parts.append(f"content_length={shape.get('content_length')}")
    parts.append(f"reasoning_present={bool(shape.get('reasoning_present'))}")
    if shape.get("reasoning_length") is not None:
        parts.append(f"reasoning_length={shape.get('reasoning_length')}")
    parts.append(
        f"reasoning_details_present={bool(shape.get('reasoning_details_present'))}"
    )
    if shape.get("reasoning_details_count") is not None:
        parts.append(f"reasoning_details_count={shape.get('reasoning_details_count')}")
    usage = shape.get("usage") if isinstance(shape.get("usage"), dict) else {}
    if usage.get("completion_tokens") is not None:
        parts.append(f"completion_tokens={usage.get('completion_tokens')}")
    if usage.get("reasoning_tokens") is not None:
        parts.append(f"reasoning_tokens={usage.get('reasoning_tokens')}")
    if usage.get("prompt_tokens") is not None:
        parts.append(f"prompt_tokens={usage.get('prompt_tokens')}")
    return " ".join(parts)


def extract_message_content(response: dict[str, Any]) -> str:
    """Pull assistant message content from a Chat Completions response.

    Final application payload MUST come from ``message.content`` only.
    Never treat ``reasoning`` / ``reasoning_content`` / ``reasoning_details``
    as the TemporalReasoningResult source.
    """
    shape = summarize_openrouter_response_shape(response)
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenRouterError(
            format_missing_content_error(shape),
            error_kind="missing_content",
            failure_stage="response_parse",
            response_shape=shape,
        )
    first = choices[0]
    if not isinstance(first, dict):
        raise OpenRouterError(
            format_missing_content_error(shape),
            error_kind="missing_content",
            failure_stage="response_parse",
            response_shape=shape,
        )
    message = first.get("message")
    if not isinstance(message, dict):
        raise OpenRouterError(
            format_missing_content_error(shape),
            error_kind="missing_content",
            failure_stage="response_parse",
            response_shape=shape,
        )
    content = message.get("content")
    if content is None:
        raise OpenRouterError(
            format_missing_content_error(shape, empty=False),
            error_kind="missing_content",
            failure_stage="response_parse",
            response_shape=shape,
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
            format_missing_content_error(shape, empty=True),
            error_kind="missing_content",
            failure_stage="response_parse",
            response_shape=shape,
        )
    return text


def _resolve_openrouter_max_tokens(config: TemporalReasonerConfig) -> int:
    """Production OpenRouter budget is 1280; honor explicit non-default overrides."""
    from src.config import TEMPORAL_REASONER_MAX_NEW_TOKENS

    cfg_tokens = int(config.max_new_tokens)
    if cfg_tokens in {
        int(TEMPORAL_REASONER_MAX_NEW_TOKENS),
        int(OPENROUTER_REASONER_MAX_TOKENS),
    }:
        return int(OPENROUTER_REASONER_MAX_TOKENS)
    return cfg_tokens


def _safe_attempt_summary(
    *,
    kind: str,
    requested_models: Sequence[str],
    routed_model: Optional[str],
    http_status: Optional[int],
    shape: Optional[dict[str, Any]],
    failure_kind: Optional[str],
) -> dict[str, Any]:
    """Safe per-attempt record — never prompts, secrets, or reasoning text."""
    usage = shape.get("usage") if isinstance(shape, dict) else None
    return {
        "kind": kind,
        "requested_models": [str(m) for m in requested_models],
        "routed_model": routed_model,
        "http_status": http_status,
        "finish_reason": (shape or {}).get("finish_reason") if shape else None,
        "content_present": (shape or {}).get("content_present") if shape else None,
        "content_length": (shape or {}).get("content_length") if shape else None,
        "reasoning_present": (shape or {}).get("reasoning_present") if shape else None,
        "reasoning_length": (shape or {}).get("reasoning_length") if shape else None,
        "completion_tokens": (usage or {}).get("completion_tokens") if usage else None,
        "reasoning_tokens": (usage or {}).get("reasoning_tokens") if usage else None,
        "prompt_tokens": (usage or {}).get("prompt_tokens") if usage else None,
        "failure_kind": failure_kind,
    }


class _OpenRouterRequestBudget:
    """Shared hard cap on HTTP POSTs inside one ``reason()`` call.

    Documented maximum (``OPENROUTER_MAX_HTTP_REQUESTS_PER_CALL`` = 4):
    1. primary models-chain request
    2. at most one transient 429/5xx retry of the current request
    3. at most one application-level fallback request (remaining models)
    4. at most one schema-repair request (no nested app fallback)
    """

    def __init__(self, *, max_requests: int) -> None:
        self.max_requests = int(max_requests)
        self.count = 0
        self.attempts: list[dict[str, Any]] = []
        self.application_fallback_attempted = False
        self.application_fallback_from_model: Optional[str] = None
        self.application_fallback_remaining_models: list[str] = []

    def consume(self) -> None:
        if self.count >= self.max_requests:
            raise OpenRouterError(
                f"OpenRouter request budget exhausted ({self.max_requests})",
                retryable=False,
                error_kind="request_budget_exhausted",
                failure_stage="unknown",
            )
        self.count += 1


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
        shape = exc.response_shape if isinstance(exc.response_shape, dict) else None
        if shape:
            diagnostics.openrouter_response_shape = shape
            routed = shape.get("routed_model")
            if isinstance(routed, str) and routed.strip():
                diagnostics.openrouter_routed_model = routed.strip()
            usage = shape.get("usage") if isinstance(shape.get("usage"), dict) else {}
            if usage.get("prompt_tokens") is not None:
                diagnostics.prompt_tokens = int(usage["prompt_tokens"])
            if usage.get("completion_tokens") is not None:
                diagnostics.generated_tokens = int(usage["completion_tokens"])
            # Hint truncation when finish_reason=length with no usable content.
            if shape.get("finish_reason") == "length" and not shape.get("content_present"):
                diagnostics.output_hit_token_limit = True
                diagnostics.likely_output_truncation = True


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
            fallback_models = list(self.config.openrouter_fallback_models or [])
            body = build_openrouter_request_body(
                model_id=self.config.model_id,
                system=OPENROUTER_SYSTEM_INSTRUCTION,
                user=user_prompt,
                max_tokens=int(self.config.max_new_tokens),
                fallback_models=fallback_models,
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
        model_chain = build_openrouter_model_chain(
            primary=self.config.model_id,
            fallback_models=self.config.openrouter_fallback_models,
        )
        fallback_models = model_chain[1:]
        openrouter_max_tokens = _resolve_openrouter_max_tokens(self.config)
        diagnostics.generation_kwargs = {
            "model": self.config.model_id,
            "requested_model": model_chain[0] if model_chain else self.config.model_id,
            "fallback_models": fallback_models,
            "model_chain": model_chain,
            "api_url": self.config.openrouter_api_url,
            "response_format_type": "json_object",
            "provider_require_parameters": True,
            "max_tokens": openrouter_max_tokens,
            "reasoning_effort": "minimal",
            "reasoning_exclude": True,
            "method": "POST",
            "content_type": "application/json",
            "max_http_requests": int(OPENROUTER_MAX_HTTP_REQUESTS_PER_CALL),
        }

        # Rebuild body with the resolved OpenRouter token budget + reasoning knobs
        # (payload_build above may have used a stale max_new_tokens default).
        body = build_openrouter_request_body(
            model_id=self.config.model_id,
            system=OPENROUTER_SYSTEM_INSTRUCTION,
            user=user_prompt,
            max_tokens=openrouter_max_tokens,
            fallback_models=list(self.config.openrouter_fallback_models or []),
        )

        request_budget = _OpenRouterRequestBudget(
            max_requests=int(OPENROUTER_MAX_HTTP_REQUESTS_PER_CALL),
        )
        try:
            raw_text, http_meta = self._generate_with_application_fallback(
                body,
                api_key=api_key,
                model_chain=model_chain,
                system=OPENROUTER_SYSTEM_INSTRUCTION,
                user=user_prompt,
                max_tokens=openrouter_max_tokens,
                budget=request_budget,
                allow_application_fallback=True,
            )
            self._apply_http_meta(diagnostics, http_meta)
            self._apply_attempt_diagnostics(diagnostics, request_budget)
            reported_model = diagnostics.openrouter_routed_model or self.config.model_id
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
            self._apply_attempt_diagnostics(diagnostics, request_budget)
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
                        "openrouter_routed_model": diagnostics.openrouter_routed_model,
                        "openrouter_response_shape": diagnostics.openrouter_response_shape,
                        "openrouter_application_fallback_attempted": (
                            diagnostics.openrouter_application_fallback_attempted
                        ),
                        "openrouter_attempt_count": diagnostics.openrouter_attempt_count,
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
            self._apply_attempt_diagnostics(diagnostics, request_budget)
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
                model_id=reported_model,
                valid_evidence_ids=valid_evidence_ids,
                valid_window_ranges=valid_window_ranges,
            )
            # Prefer OpenRouter-reported routed model over any model string in LLM JSON.
            result = result.model_copy(update={"model": reported_model})
            diagnostics.parse_validation_seconds = time.perf_counter() - parse_started
            diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
            return result, diagnostics
        except Exception as first_exc:  # noqa: BLE001
            if int(self.config.max_retries) < 1:
                logger.exception(
                    "OpenRouter schema_validation failed model=%s",
                    reported_model,
                )
                _apply_openrouter_failure(diagnostics, first_exc, stage="schema_validation")
                diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
                return (
                    TemporalReasoningResult(
                        summary="",
                        context_type="uncertain",
                        confidence=0.0,
                        model=reported_model,
                        status="invalid_model_output",
                        details={
                            "error": format_validation_error(first_exc),
                            "raw_preview": _redact(raw_text[:500]),
                            "openrouter_failure_stage": "schema_validation",
                            "openrouter_error_type": type(first_exc).__name__,
                            "openrouter_error_message": sanitize_openrouter_error_message(
                                first_exc,
                            ),
                            "provider": "openrouter",
                        },
                    ),
                    diagnostics,
                )

            repair_user = build_repair_prompt(
                validation_error=format_validation_error(first_exc),
                previous_output=raw_text,
            )
            repair_body = build_openrouter_request_body(
                model_id=self.config.model_id,
                system=OPENROUTER_SYSTEM_INSTRUCTION,
                user=repair_user,
                max_tokens=openrouter_max_tokens,
                fallback_models=list(self.config.openrouter_fallback_models or []),
            )
            try:
                repair_started = time.perf_counter()
                # Schema repair: reuse shared HTTP budget; NO second app-level fallback.
                raw_retry, repair_meta = self._generate_with_application_fallback(
                    repair_body,
                    api_key=api_key,
                    model_chain=model_chain,
                    system=OPENROUTER_SYSTEM_INSTRUCTION,
                    user=repair_user,
                    max_tokens=openrouter_max_tokens,
                    budget=request_budget,
                    allow_application_fallback=False,
                )
                repair_s = time.perf_counter() - repair_started
                diagnostics.repair_attempted = True
                diagnostics.repair_generation_seconds = repair_s
                diagnostics.raw_output_preview = _redact(raw_retry[:500])
                self._apply_http_meta(diagnostics, repair_meta, merge_generation_seconds=True)
                self._apply_attempt_diagnostics(diagnostics, request_budget)
                routed_repair = repair_meta.get("routed_model")
                if isinstance(routed_repair, str) and routed_repair.strip():
                    diagnostics.openrouter_routed_model = routed_repair.strip()
                    reported_model = diagnostics.openrouter_routed_model
            except Exception as repair_gen_exc:  # noqa: BLE001
                diagnostics.repair_attempted = True
                diagnostics.repair_generation_seconds = (
                    time.perf_counter() - repair_started
                )
                self._apply_attempt_diagnostics(diagnostics, request_budget)
                logger.exception(
                    "OpenRouter repair generation failed model=%s",
                    reported_model,
                )
                stage = (
                    repair_gen_exc.failure_stage
                    if isinstance(repair_gen_exc, OpenRouterError)
                    else "unknown"
                )
                _apply_openrouter_failure(diagnostics, repair_gen_exc, stage=stage)
                diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
                status = "generation_failed"
                if isinstance(repair_gen_exc, OpenRouterError) and repair_gen_exc.error_kind in {
                    "auth",
                    "connection",
                    "ssl",
                    "free_endpoint_unavailable",
                }:
                    status = "reasoner_unavailable"
                return (
                    TemporalReasoningResult(
                        summary="",
                        context_type="uncertain",
                        confidence=0.0,
                        model=reported_model,
                        status=status,
                        details={
                            "error": (
                                "repair_generation_failed: "
                                f"{sanitize_openrouter_error_message(repair_gen_exc)}"
                            ),
                            "first_error": format_validation_error(first_exc),
                            "openrouter_failure_stage": stage,
                            "openrouter_error_type": type(repair_gen_exc).__name__,
                            "openrouter_error_message": sanitize_openrouter_error_message(
                                repair_gen_exc,
                            ),
                            "provider": "openrouter",
                            "openrouter_attempt_count": diagnostics.openrouter_attempt_count,
                        },
                    ),
                    diagnostics,
                )

            try:
                parse_started = time.perf_counter()
                result = parse_reasoning_result(
                    raw_retry,
                    model_id=reported_model,
                    valid_evidence_ids=valid_evidence_ids,
                    valid_window_ranges=valid_window_ranges,
                )
                result = result.model_copy(update={"model": reported_model})
                diagnostics.parse_validation_seconds = (
                    diagnostics.parse_validation_seconds or 0.0
                ) + (time.perf_counter() - parse_started)
                diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
                return result, diagnostics
            except Exception as second_exc:  # noqa: BLE001
                logger.exception(
                    "OpenRouter schema_validation failed after repair model=%s",
                    reported_model,
                )
                _apply_openrouter_failure(
                    diagnostics,
                    second_exc,
                    stage="schema_validation",
                )
                diagnostics.total_reasoner_seconds = time.perf_counter() - total_started
                return (
                    TemporalReasoningResult(
                        summary="",
                        context_type="uncertain",
                        confidence=0.0,
                        model=reported_model,
                        status="invalid_model_output",
                        details={
                            "error": format_validation_error(second_exc),
                            "first_error": format_validation_error(first_exc),
                            "raw_preview": _redact(raw_retry[:500]),
                            "openrouter_failure_stage": "schema_validation",
                            "openrouter_error_type": type(second_exc).__name__,
                            "openrouter_error_message": sanitize_openrouter_error_message(
                                second_exc,
                            ),
                            "provider": "openrouter",
                            "repair_attempted": True,
                        },
                    ),
                    diagnostics,
                )

    def _apply_http_meta(
        self,
        diagnostics: TemporalReasonerDiagnostics,
        http_meta: dict[str, Any],
        *,
        merge_generation_seconds: bool = True,
    ) -> None:
        diagnostics.http_status = http_meta.get("http_status")
        diagnostics.openrouter_http_status = http_meta.get("http_status")
        diagnostics.retry_attempted = bool(
            diagnostics.retry_attempted or http_meta.get("retry_attempted")
        )
        gen_s = http_meta.get("generation_seconds")
        if gen_s is not None:
            if merge_generation_seconds and diagnostics.generation_seconds is not None:
                diagnostics.generation_seconds = float(diagnostics.generation_seconds) + float(
                    gen_s
                )
            else:
                diagnostics.generation_seconds = float(gen_s)
        raw = http_meta.get("raw_text")
        if isinstance(raw, str) and raw:
            diagnostics.raw_output_preview = _redact(raw[:500])
        usage = http_meta.get("usage") or {}
        if usage.get("prompt_tokens") is not None:
            diagnostics.prompt_tokens = int(usage["prompt_tokens"])
        if usage.get("completion_tokens") is not None:
            diagnostics.generated_tokens = int(usage["completion_tokens"])
        routed_model = http_meta.get("routed_model")
        if isinstance(routed_model, str) and routed_model.strip():
            diagnostics.openrouter_routed_model = routed_model.strip()
        shape = http_meta.get("response_shape")
        if isinstance(shape, dict):
            diagnostics.openrouter_response_shape = shape
        if http_meta.get("application_fallback_attempted"):
            diagnostics.openrouter_application_fallback_attempted = True
            from_model = http_meta.get("application_fallback_from_model")
            if isinstance(from_model, str) and from_model.strip():
                diagnostics.openrouter_application_fallback_from_model = from_model.strip()
            remaining = http_meta.get("application_fallback_remaining_models")
            if isinstance(remaining, list):
                diagnostics.openrouter_application_fallback_remaining_models = [
                    str(m) for m in remaining
                ]

    def _apply_attempt_diagnostics(
        self,
        diagnostics: TemporalReasonerDiagnostics,
        budget: _OpenRouterRequestBudget,
    ) -> None:
        diagnostics.openrouter_attempt_count = int(budget.count)
        diagnostics.openrouter_attempts = list(budget.attempts)
        if budget.application_fallback_attempted:
            diagnostics.openrouter_application_fallback_attempted = True
            diagnostics.openrouter_application_fallback_from_model = (
                budget.application_fallback_from_model
            )
            diagnostics.openrouter_application_fallback_remaining_models = list(
                budget.application_fallback_remaining_models
            )

    def _generate_with_retry(
        self,
        body: dict[str, Any],
        *,
        api_key: str,
    ) -> tuple[str, dict[str, Any]]:
        """One request + at most one bounded transient 429/5xx retry (shared budget).

        Signature kept as ``(body, *, api_key)`` for unit-test monkeypatches.
        Shared request budget / attempt kind are threaded via instance attributes
        set by ``_generate_with_application_fallback``.
        """
        budget = getattr(self, "_request_budget", None)
        if budget is None:
            budget = _OpenRouterRequestBudget(
                max_requests=int(OPENROUTER_MAX_HTTP_REQUESTS_PER_CALL),
            )
            self._request_budget = budget
        attempt_kind = str(getattr(self, "_attempt_kind", "primary") or "primary")
        max_retries = int(self.config.openrouter_max_transient_retries)
        attempt = 0
        retry_attempted = False
        started = time.perf_counter()
        requested_models = body.get("models") if isinstance(body.get("models"), list) else []
        while True:
            budget.consume()
            try:
                response = post_openrouter_chat_completion(
                    body,
                    api_key=api_key,
                    api_url=self.config.openrouter_api_url,
                    timeout_seconds=self.config.openrouter_timeout_seconds,
                )
                shape = summarize_openrouter_response_shape(response, http_status=200)
                usage = (
                    response.get("usage")
                    if isinstance(response.get("usage"), dict)
                    else {}
                )
                routed = extract_routed_model(response)
                try:
                    content = extract_message_content(response)
                except OpenRouterError as parse_exc:
                    shape_err = (
                        parse_exc.response_shape
                        if isinstance(parse_exc.response_shape, dict)
                        else shape
                    )
                    if isinstance(shape_err, dict) and shape_err.get("http_status") is None:
                        shape_err = {**shape_err, "http_status": 200}
                    budget.attempts.append(
                        _safe_attempt_summary(
                            kind=attempt_kind if not retry_attempted else f"{attempt_kind}_transient_retry",
                            requested_models=requested_models,
                            routed_model=routed or (shape_err or {}).get("routed_model"),
                            http_status=200,
                            shape=shape_err,
                            failure_kind=parse_exc.error_kind,
                        )
                    )
                    raise OpenRouterError(
                        str(parse_exc),
                        status_code=200,
                        retryable=False,
                        error_kind=parse_exc.error_kind,
                        retry_attempted=retry_attempted,
                        failure_stage=parse_exc.failure_stage or "response_parse",
                        response_body_preview=parse_exc.response_body_preview,
                        response_shape=shape_err,
                    ) from parse_exc
                budget.attempts.append(
                    _safe_attempt_summary(
                        kind=attempt_kind if not retry_attempted else f"{attempt_kind}_transient_retry",
                        requested_models=requested_models,
                        routed_model=routed,
                        http_status=200,
                        shape=shape,
                        failure_kind=None,
                    )
                )
                return content, {
                    "http_status": 200,
                    "retry_attempted": retry_attempted,
                    "generation_seconds": time.perf_counter() - started,
                    "usage": usage,
                    "routed_model": routed,
                    "response_shape": shape,
                    "raw_text": content,
                }
            except OpenRouterError as exc:
                if exc.error_kind != "missing_content":
                    budget.attempts.append(
                        _safe_attempt_summary(
                            kind=(
                                attempt_kind
                                if not retry_attempted
                                else f"{attempt_kind}_transient_retry"
                            ),
                            requested_models=requested_models,
                            routed_model=(
                                (exc.response_shape or {}).get("routed_model")
                                if isinstance(exc.response_shape, dict)
                                else None
                            ),
                            http_status=exc.status_code,
                            shape=exc.response_shape if isinstance(exc.response_shape, dict) else None,
                            failure_kind=exc.error_kind,
                        )
                    )
                can_retry = (
                    exc.retryable
                    and attempt < max_retries
                    and exc.error_kind
                    in {
                        "rate_limit",
                        "server_error",
                        "free_endpoint_unavailable",
                    }
                    and budget.count < budget.max_requests
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
                        response_shape=exc.response_shape,
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

    def _generate_with_application_fallback(
        self,
        body: dict[str, Any],
        *,
        api_key: str,
        model_chain: Sequence[str],
        system: str,
        user: str,
        max_tokens: int,
        budget: _OpenRouterRequestBudget,
        allow_application_fallback: bool,
    ) -> tuple[str, dict[str, Any]]:
        """Primary chain request, optional transient retry, optional ONE app fallback.

        Application fallback triggers only on HTTP 2xx with unusable final content
        (null/empty ``message.content``). It requests only models AFTER the routed
        model and never retries models that already returned an unusable 2xx.
        """
        self._request_budget = budget
        kind = "primary" if allow_application_fallback else "schema_repair"
        self._attempt_kind = kind
        try:
            content, meta = self._generate_with_retry(body, api_key=api_key)
            return content, meta
        except OpenRouterError as exc:
            if not allow_application_fallback or not is_application_fallback_eligible(exc):
                raise
            if budget.application_fallback_attempted:
                raise
            shape = exc.response_shape if isinstance(exc.response_shape, dict) else {}
            routed = None
            if isinstance(shape.get("routed_model"), str):
                routed = shape["routed_model"].strip() or None
            remaining = remaining_models_after_routed(model_chain, routed)
            if not remaining:
                raise
            from_model = routed or (model_chain[0] if model_chain else self.config.model_id)
            budget.application_fallback_attempted = True
            budget.application_fallback_from_model = str(from_model)
            budget.application_fallback_remaining_models = list(remaining)
            fallback_body = build_openrouter_request_body(
                model_id=remaining[0],
                system=system,
                user=user,
                max_tokens=int(max_tokens),
                fallback_models=remaining[1:],
            )
            self._attempt_kind = "application_fallback"
            content, meta = self._generate_with_retry(fallback_body, api_key=api_key)
            meta = {
                **meta,
                "application_fallback_attempted": True,
                "application_fallback_from_model": str(from_model),
                "application_fallback_remaining_models": list(remaining),
            }
            return content, meta
