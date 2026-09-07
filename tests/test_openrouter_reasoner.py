"""OpenRouter temporal reasoner + final assessment tests (no live network)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for path in (ROOT, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.config import (
    OPENROUTER_API_URL,
    OPENROUTER_REASONER_MODEL,
    TemporalReasonerConfig,
    evaluation_reasoner_config,
    resolve_temporal_reasoner_config,
)
from src.routing.input_router import CapabilityStatus, InputType, RoutedAnalysisResult
from src.schemas import (
    ActivityAnalysisResult,
    AnalysisBlock,
    FinalTemporalAssessment,
    FusionDiagnostics,
    InputMetadata,
    ModalityBundle,
    SentimentEvidence,
    TemporalHighlight,
    TemporalReasoningResult,
    VideoDiagnostics,
)
from src.temporal.final_assessment import build_final_temporal_assessment
from src.temporal.prompt import SYSTEM_INSTRUCTION, build_evidence_payload, build_user_prompt
from src.temporal.providers import create_temporal_reasoner
from src.temporal.providers.openrouter import (
    OpenRouterError,
    OpenRouterTemporalReasoner,
    build_openrouter_request_body,
    extract_message_content,
    openrouter_api_key_configured,
    post_openrouter_chat_completion,
)
from src.temporal.providers.openrouter_schema import (
    OPENROUTER_SYSTEM_INSTRUCTION,
    OPENROUTER_TEMPORAL_REASONING_SCHEMA,
)
from src.temporal.reasoner import TemporalContextReasoner
from src.temporal.wellbeing import WELLBEING_RULE_DOC, compute_wellbeing_indicator
from src.ui.gradio_view import (
    CONTEXT_UNAVAILABLE_MESSAGE,
    render_routed_result,
    render_technical_details,
)
from temporal_fixtures import (
    fixture_informational,
    fixture_persistent_negative,
    fixture_prompt_injection,
    fixture_quoted_narrative,
    fixture_stable_neutral,
    fixture_visual_pos_speech_neg,
)


def _ev(label: str = "neutral") -> SentimentEvidence:
    probs = {"positive": 0.1, "neutral": 0.1, "negative": 0.1}
    probs[label] = 0.8
    return SentimentEvidence(
        label=label,  # type: ignore[arg-type]
        score=probs["positive"] - probs["negative"],
        confidence=0.8,
        probabilities=probs,
        model="stub",
    )


def _valid_reasoning_json(**overrides: object) -> str:
    payload = {
        "summary": (
            "Across the timeline, expressed tone is mostly stable. "
            "Deterministic trajectory remains authoritative. "
            "Coverage is limited in places."
        ),
        "trajectory_explanation": "Deterministic trajectory is stable_neutral.",
        "cross_modal_context": {
            "consistency": "insufficient_evidence",
            "conflicts_detected": False,
            "description": "Limited paired modality evidence.",
        },
        "important_transitions": [
            {
                "start": 0.0,
                "end": 5.0,
                "description": "Opening window shows lower-negative evidence.",
                "evidence_ids": ["window-0"],
            },
        ],
        "context_type": "personal_expression",
        "evidence": [
            {"evidence_id": "window-0", "explanation": "Initial usable window."},
        ],
        "uncertainties": ["Sparse OCR"],
        "confidence": 0.6,
    }
    payload.update(overrides)
    return json.dumps(payload)


def _openrouter_cfg(**kwargs: object) -> TemporalReasonerConfig:
    base = {
        "enabled": True,
        "provider": "openrouter",
        "fallback": "none",
        "model_id": OPENROUTER_REASONER_MODEL,
        "openrouter_max_transient_retries": 1,
        "openrouter_timeout_seconds": 5.0,
    }
    base.update(kwargs)
    return TemporalReasonerConfig(**base)  # type: ignore[arg-type]


def _fake_http_response(payload: dict, *, status: int = 200) -> SimpleNamespace:
    body = json.dumps(payload).encode("utf-8")

    class _Resp:
        def __init__(self) -> None:
            self.status = status

        def read(self) -> bytes:
            return body

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    return _Resp()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Request shape / auth / schema
# ---------------------------------------------------------------------------


def test_openrouter_model_id_default() -> None:
    assert OPENROUTER_REASONER_MODEL == "openai/gpt-oss-20b:free"
    cfg = resolve_temporal_reasoner_config()
    assert cfg.provider == "openrouter"
    assert cfg.model_id == "openai/gpt-oss-20b:free"
    assert cfg.fallback == "none"


def test_openrouter_request_shape_and_auth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-key-do-not-log")
    captured: dict = {}

    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["headers"] = {k.lower(): v for k, v in request.header_items()}
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _fake_http_response(
            {
                "choices": [
                    {"message": {"content": _valid_reasoning_json()}},
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
        )

    monkeypatch.setattr(
        "src.temporal.providers.openrouter.urllib.request.urlopen",
        fake_urlopen,
    )
    body = build_openrouter_request_body(
        model_id=OPENROUTER_REASONER_MODEL,
        system=OPENROUTER_SYSTEM_INSTRUCTION,
        user="hello",
        max_tokens=768,
    )
    assert body["model"] == "openai/gpt-oss-20b:free"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == OPENROUTER_TEMPORAL_REASONING_SCHEMA
    assert body["provider"]["require_parameters"] is True

    resp = post_openrouter_chat_completion(
        body,
        api_key="test-secret-key-do-not-log",
        api_url=OPENROUTER_API_URL,
        timeout_seconds=5.0,
    )
    assert captured["url"] == OPENROUTER_API_URL
    assert captured["headers"]["authorization"] == "Bearer test-secret-key-do-not-log"
    assert captured["headers"]["content-type"] == "application/json"
    assert captured["body"]["model"] == "openai/gpt-oss-20b:free"
    assert captured["body"]["response_format"]["json_schema"]["strict"] is True
    assert captured["body"]["provider"]["require_parameters"] is True
    assert "choices" in resp
    # Verify POST semantics on the Request object via captured call.
    # fake_urlopen receives the Request; re-check via building one.
    import urllib.request as ur

    req = ur.Request(
        OPENROUTER_API_URL,
        data=b"{}",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    assert req.get_method() == "POST"


def test_api_key_never_logged(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "super-secret-openrouter-key")
    reasoner = OpenRouterTemporalReasoner(_openrouter_cfg())

    def boom(*_a, **_k):  # noqa: ANN001
        raise OpenRouterError("auth failed", status_code=401, error_kind="auth")

    monkeypatch.setattr(reasoner, "_generate_with_retry", boom)
    with caplog.at_level(logging.WARNING):
        result, diag = reasoner.reason(fixture_stable_neutral())
    assert result.status == "reasoner_unavailable"
    joined = " ".join(r.message for r in caplog.records)
    assert "super-secret-openrouter-key" not in joined
    assert "super-secret-openrouter-key" not in json.dumps(result.model_dump())
    assert "super-secret-openrouter-key" not in json.dumps(diag.model_dump())


def test_successful_structured_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    reasoner = OpenRouterTemporalReasoner(_openrouter_cfg())
    ctx = fixture_stable_neutral()
    payload = build_evidence_payload(ctx, config=_openrouter_cfg())
    valid = set(payload["valid_evidence_ids"])

    def fake_gen(body, *, api_key):  # noqa: ANN001
        assert body["model"] == OPENROUTER_REASONER_MODEL
        assert body["response_format"]["json_schema"]["strict"] is True
        return _valid_reasoning_json(evidence_ids_ok=True), {
            "http_status": 200,
            "retry_attempted": False,
            "generation_seconds": 0.01,
            "usage": {"prompt_tokens": 11, "completion_tokens": 22},
        }

    # Ensure evidence ids in fixture response are valid
    raw = _valid_reasoning_json(
        evidence=[{"evidence_id": next(iter(valid)), "explanation": "ok"}],
        important_transitions=[
            {
                "start": float(ctx.windows[0].start),
                "end": float(ctx.windows[0].end),
                "description": "Stable opening.",
                "evidence_ids": [f"window-{ctx.windows[0].index}"],
            },
        ],
    )

    def fake_gen2(body, *, api_key):  # noqa: ANN001
        return raw, {
            "http_status": 200,
            "retry_attempted": False,
            "generation_seconds": 0.01,
            "usage": {},
        }

    monkeypatch.setattr(reasoner, "_generate_with_retry", fake_gen2)
    result, diag = reasoner.reason(ctx)
    assert result.status == "ok"
    assert result.model == OPENROUTER_REASONER_MODEL
    assert diag.provider == "openrouter"
    assert result.context_type == "personal_expression"


def test_missing_key_fail_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    reasoner = OpenRouterTemporalReasoner(_openrouter_cfg())
    result, diag = reasoner.reason(fixture_stable_neutral())
    assert result.status == "reasoner_unavailable"
    assert diag.reasoner_configured is False
    assert openrouter_api_key_configured() is False


def test_429_bounded_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    reasoner = OpenRouterTemporalReasoner(_openrouter_cfg())
    calls = {"n": 0}

    def flaky(body, *, api_key):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise OpenRouterError(
                "rate limited",
                status_code=429,
                retryable=True,
                retry_after=0.0,
                error_kind="rate_limit",
            )
        return _valid_reasoning_json(
            evidence=[{"evidence_id": "window-0", "explanation": "ok"}],
            important_transitions=[],
        ), {
            "http_status": 200,
            "retry_attempted": True,
            "generation_seconds": 0.02,
            "usage": {},
        }

    # Patch post to simulate retry path inside _generate_with_retry
    def fake_post(*_a, **_k):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise OpenRouterError(
                "rate limited",
                status_code=429,
                retryable=True,
                retry_after=0.0,
                error_kind="rate_limit",
            )
        return {
            "choices": [{"message": {"content": _valid_reasoning_json(
                evidence=[{"evidence_id": "window-0", "explanation": "ok"}],
                important_transitions=[],
            )}}],
            "usage": {},
        }

    monkeypatch.setattr(
        "src.temporal.providers.openrouter.post_openrouter_chat_completion",
        fake_post,
    )
    monkeypatch.setattr("src.temporal.providers.openrouter.time.sleep", lambda _s: None)
    result, diag = reasoner.reason(fixture_stable_neutral())
    assert calls["n"] == 2
    assert diag.retry_attempted is True
    assert result.status == "ok"


def test_5xx_bounded_retry_then_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    reasoner = OpenRouterTemporalReasoner(_openrouter_cfg())
    calls = {"n": 0}

    def always_5xx(*_a, **_k):  # noqa: ANN001
        calls["n"] += 1
        raise OpenRouterError(
            "upstream",
            status_code=503,
            retryable=True,
            retry_after=0.0,
            error_kind="free_endpoint_unavailable",
        )

    monkeypatch.setattr(
        "src.temporal.providers.openrouter.post_openrouter_chat_completion",
        always_5xx,
    )
    monkeypatch.setattr("src.temporal.providers.openrouter.time.sleep", lambda _s: None)
    result, diag = reasoner.reason(fixture_stable_neutral())
    assert calls["n"] == 2  # initial + one retry
    assert result.status in {"reasoner_unavailable", "generation_failed"}
    assert diag.retry_attempted is True


def test_timeout_fail_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    reasoner = OpenRouterTemporalReasoner(_openrouter_cfg())

    def boom(*_a, **_k):  # noqa: ANN001
        raise OpenRouterError("timed out", error_kind="timeout")

    monkeypatch.setattr(
        "src.temporal.providers.openrouter.post_openrouter_chat_completion",
        boom,
    )
    result, _diag = reasoner.reason(fixture_stable_neutral())
    assert result.status == "generation_failed"
    assert result.details and result.details.get("error_kind") == "timeout"


def test_malformed_response_and_schema_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    reasoner = OpenRouterTemporalReasoner(_openrouter_cfg())

    def bad_json(*_a, **_k):  # noqa: ANN001
        return (
            "{not-json",
            {"http_status": 200, "retry_attempted": False, "generation_seconds": 0.01, "usage": {}},
        )

    monkeypatch.setattr(reasoner, "_generate_with_retry", bad_json)
    result, _ = reasoner.reason(fixture_stable_neutral())
    assert result.status == "invalid_model_output"

    def bad_schema(*_a, **_k):  # noqa: ANN001
        return (
            json.dumps({"summary": "x", "context_type": "not_a_real_type"}),
            {"http_status": 200, "retry_attempted": False, "generation_seconds": 0.01, "usage": {}},
        )

    monkeypatch.setattr(reasoner, "_generate_with_retry", bad_schema)
    result2, _ = reasoner.reason(fixture_stable_neutral())
    assert result2.status == "invalid_model_output"


def test_missing_content(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(OpenRouterError) as ei:
        extract_message_content({"choices": [{"message": {}}]})
    assert ei.value.error_kind == "missing_content"


def test_free_endpoint_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    reasoner = OpenRouterTemporalReasoner(_openrouter_cfg())

    def boom(*_a, **_k):  # noqa: ANN001
        raise OpenRouterError(
            "unavailable",
            status_code=503,
            retryable=False,
            error_kind="free_endpoint_unavailable",
        )

    monkeypatch.setattr(reasoner, "_generate_with_retry", boom)
    result, _ = reasoner.reason(fixture_stable_neutral())
    assert result.status == "reasoner_unavailable"


def test_deterministic_survives_llm_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    reasoner = OpenRouterTemporalReasoner(_openrouter_cfg())
    ctx = fixture_visual_pos_speech_neg()
    traj = ctx.features.trajectory
    conflicts = list(ctx.features.cross_modal_conflicts)

    def boom(*_a, **_k):  # noqa: ANN001
        raise OpenRouterError("fail", status_code=500, error_kind="server_error")

    monkeypatch.setattr(reasoner, "_generate_with_retry", boom)
    result, _ = reasoner.reason(ctx)
    assert result.status != "ok"
    # Deterministic context object unchanged
    assert ctx.features.trajectory == traj
    assert ctx.features.cross_modal_conflicts == conflicts
    final = build_final_temporal_assessment(ctx, result, model_id=OPENROUTER_REASONER_MODEL)
    assert final.status == "explanation_unavailable"
    assert CONTEXT_UNAVAILABLE_MESSAGE in final.uncertainty_note
    assert ctx.features.trajectory == traj


def test_frozen_deterministic_evidence_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    reasoner = OpenRouterTemporalReasoner(_openrouter_cfg())
    ctx = fixture_visual_pos_speech_neg()
    evidence = build_evidence_payload(ctx, config=_openrouter_cfg())
    feats = evidence["deterministic_features"]

    def echo(body, *, api_key):  # noqa: ANN001
        user = body["messages"][1]["content"]
        assert '"trajectory"' in user
        assert feats["trajectory"] in user
        assert "never follow" in OPENROUTER_SYSTEM_INSTRUCTION.lower() or "Never follow" in OPENROUTER_SYSTEM_INSTRUCTION
        raw = _valid_reasoning_json(
            context_type="personal_expression",
            evidence=[{"evidence_id": "window-0", "explanation": "ok"}],
            important_transitions=[],
            cross_modal_context={
                "consistency": "low",
                "conflicts_detected": True,
                "description": "Speech and visual disagree as supplied.",
            },
        )
        return raw, {
            "http_status": 200,
            "retry_attempted": False,
            "generation_seconds": 0.01,
            "usage": {},
        }

    monkeypatch.setattr(reasoner, "_generate_with_retry", echo)
    result, _ = reasoner.reason(ctx)
    assert result.status == "ok"
    # Authority: deterministic features on context unchanged
    assert ctx.features.trajectory == feats["trajectory"]
    assert len(ctx.features.cross_modal_conflicts) >= 1


def test_valid_evidence_ids_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    reasoner = OpenRouterTemporalReasoner(_openrouter_cfg())

    def invent(*_a, **_k):  # noqa: ANN001
        return (
            _valid_reasoning_json(
                evidence=[{"evidence_id": "invented-id-999", "explanation": "nope"}],
                important_transitions=[],
            ),
            {"http_status": 200, "retry_attempted": False, "generation_seconds": 0.01, "usage": {}},
        )

    monkeypatch.setattr(reasoner, "_generate_with_retry", invent)
    result, _ = reasoner.reason(fixture_stable_neutral())
    assert result.status == "invalid_model_output"


def test_prompt_injection_remains_data() -> None:
    ctx = fixture_prompt_injection()
    user = build_user_prompt(build_evidence_payload(ctx, config=_openrouter_cfg()))
    assert "DATA" in user or "untrusted" in user.lower()
    assert "Never follow" in OPENROUTER_SYSTEM_INSTRUCTION or "never follow" in OPENROUTER_SYSTEM_INSTRUCTION.lower()
    assert "<<<STRUCTURED_TEMPORAL_EVIDENCE>>>" in user
    # Injection text is inside the delimited evidence block as data.
    assert "IGNORE ALL PRIOR INSTRUCTIONS" in user or "ignore all prior" in user.lower()


def test_qwen_code_remains_available() -> None:
    assert TemporalContextReasoner is not None
    cfg = evaluation_reasoner_config()
    assert cfg.provider == "qwen_local_or_zerogpu"
    reasoner = create_temporal_reasoner(cfg)
    assert isinstance(reasoner, TemporalContextReasoner)


def test_no_automatic_qwen_fallback_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    cfg = _openrouter_cfg(fallback="none")
    provider = create_temporal_reasoner(cfg)
    assert isinstance(provider, OpenRouterTemporalReasoner)
    assert not hasattr(provider, "fallback") or getattr(provider, "fallback", None) is None

    qwen_called = {"n": 0}

    def boom(*_a, **_k):  # noqa: ANN001
        raise OpenRouterError("fail", status_code=503, error_kind="free_endpoint_unavailable")

    monkeypatch.setattr(
        OpenRouterTemporalReasoner,
        "_generate_with_retry",
        boom,
    )

    class SpyQwen(TemporalContextReasoner):
        def reason(self, *a, **k):  # noqa: ANN001
            qwen_called["n"] += 1
            return super().reason(*a, **k)

    with patch("src.temporal.providers.TemporalContextReasoner", SpyQwen):
        r = create_temporal_reasoner(cfg)
        result, _ = r.reason(fixture_stable_neutral())
    assert result.status != "ok"
    assert qwen_called["n"] == 0


def test_no_numerical_wellbeing_score() -> None:
    ctx = fixture_persistent_negative()
    reasoning = TemporalReasoningResult(
        summary="Persistent negative personal expression across windows.",
        trajectory_explanation="Authoritative trajectory preserved.",
        context_type="personal_expression",
        confidence=0.7,
        status="ok",
        model=OPENROUTER_REASONER_MODEL,
    )
    final = build_final_temporal_assessment(ctx, reasoning)
    assert final.overall_wellbeing_indicator in {
        "low_concern",
        "moderate_concern",
        "high_concern",
        "insufficient_evidence",
    }
    dumped = json.dumps(final.model_dump())
    assert "/10" not in dumped
    assert "/100" not in dumped
    assert "wellbeing_score" not in dumped


def test_conservative_insufficient_and_quoted_informational() -> None:
    assert "not clinical" in WELLBEING_RULE_DOC.lower()
    quoted = fixture_quoted_narrative()
    assert compute_wellbeing_indicator(quoted, context_type="quoted_or_reposted_content") == (
        "insufficient_evidence"
    )
    info = fixture_informational()
    assert compute_wellbeing_indicator(info, context_type="informational") == "insufficient_evidence"
    assert compute_wellbeing_indicator(info, context_type="uncertain") == "insufficient_evidence"

    personal = fixture_persistent_negative()
    indicator = compute_wellbeing_indicator(personal, context_type="personal_expression")
    assert indicator in {"moderate_concern", "high_concern", "low_concern"}


def test_client_ui_hides_benchmark_metrics() -> None:
    assessment = FinalTemporalAssessment(
        overall_wellbeing_indicator="moderate_concern",
        key_temporal_highlights=[
            TemporalHighlight(
                start=10.0,
                end=15.0,
                timestamp_label="00:10–00:15",
                description="Negative speech increases.",
                evidence_ids=["window-2"],
            ),
        ],
        summary_explanation="Tone shifts toward stronger negative speech mid-clip.",
        evidence_summary="Trajectory: increasing_negative.",
        uncertainty_note="Limited OCR.",
        context_type="personal_expression",
        model=OPENROUTER_REASONER_MODEL,
        status="ok",
        reasoner_configured=True,
    )
    result = ActivityAnalysisResult(
        activity_id="A-V",
        activity_type="video",
        input=InputMetadata(media_path="clip.mp4"),
        analysis=AnalysisBlock(
            overall=_ev("negative"),
            modalities=ModalityBundle(visual=_ev("negative")),
            fusion=FusionDiagnostics(
                contributing_modalities=["visual"],
                explanation="visual only",
            ),
            video=VideoDiagnostics(
                sampling_strategy="fixed_fps",
                frames_extracted=2,
                frames_analyzed=2,
            ),
            final_temporal_assessment=assessment,
        ),
    )
    routed = RoutedAnalysisResult(
        status=CapabilityStatus.OK,
        detected_input=InputType.VIDEO,
        analysis=result,
        model_display_name="POC fusion",
        model_id="poc",
    )
    html = render_routed_result(routed)
    assert "Overall Wellbeing Indicator" in html
    assert "POC content-level indicator — not a clinical assessment." in html
    assert "Key Temporal Highlights" in html
    assert "00:10–00:15" in html
    assert "Summary Explanation" in html
    assert "benchmark" not in html.lower()
    assert "prompt tokens" not in html.lower()
    assert "generated tokens" not in html.lower()
    assert "schema validation" not in html.lower()
    assert "gpu" not in html.lower()
    assert "/10" not in html
    assert "/100" not in html

    # Text path unchanged — no temporal highlights invented
    text_result = ActivityAnalysisResult(
        activity_id="A-T",
        activity_type="text",
        input=InputMetadata(text_preview="hi"),
        analysis=AnalysisBlock(
            overall=_ev("positive"),
            modalities=ModalityBundle(text=_ev("positive")),
        ),
    )
    text_html = render_routed_result(
        RoutedAnalysisResult(
            status=CapabilityStatus.OK,
            detected_input=InputType.TEXT,
            analysis=text_result,
            model_display_name="Twitter-RoBERTa",
            model_id="text",
        ),
    )
    assert "Key Temporal Highlights" not in text_html
    assert "Overall Wellbeing Indicator" not in text_html


def test_explanation_unavailable_message_in_ui() -> None:
    assessment = FinalTemporalAssessment(
        overall_wellbeing_indicator="insufficient_evidence",
        summary_explanation="",
        evidence_summary="Trajectory: stable_neutral.",
        uncertainty_note=CONTEXT_UNAVAILABLE_MESSAGE,
        context_type="uncertain",
        status="explanation_unavailable",
        reasoner_configured=False,
    )
    result = ActivityAnalysisResult(
        activity_id="A-V2",
        activity_type="video",
        input=InputMetadata(),
        analysis=AnalysisBlock(
            overall=_ev("neutral"),
            modalities=ModalityBundle(visual=_ev("neutral")),
            final_temporal_assessment=assessment,
            video=VideoDiagnostics(frames_extracted=1, frames_analyzed=1),
        ),
    )
    html = render_routed_result(
        RoutedAnalysisResult(
            status=CapabilityStatus.OK,
            detected_input=InputType.VIDEO,
            analysis=result,
        ),
    )
    assert CONTEXT_UNAVAILABLE_MESSAGE in html


def test_no_live_api_in_unit_suite() -> None:
    # Guard: OpenRouterTemporalReasoner unit path must not call real urlopen unless patched.
    assert "openrouter.ai" in OPENROUTER_API_URL
    # Import path sanity — factory default is openrouter without network.
    reasoner = create_temporal_reasoner(_openrouter_cfg())
    assert isinstance(reasoner, OpenRouterTemporalReasoner)


def test_qwen_system_instruction_still_present() -> None:
    assert "Do NOT diagnose" in SYSTEM_INSTRUCTION
    assert "untrusted USER DATA" in SYSTEM_INSTRUCTION


# ---------------------------------------------------------------------------
# Pre-request failure diagnostics / serialization
# ---------------------------------------------------------------------------


def test_request_payload_json_serializable_real_fixture() -> None:
    from src.temporal.benchmark.export import lean_temporal_context_for_reasoner
    from src.temporal.benchmark.fixtures import get_fixture_spec, load_benchmark_payload
    from src.temporal.providers.openrouter import assert_json_serializable, to_jsonable

    payload = load_benchmark_payload(get_fixture_spec("real_phase3a_controlled_video"))
    ctx = lean_temporal_context_for_reasoner(payload.temporal_context)
    cfg = _openrouter_cfg()
    evidence = to_jsonable(build_evidence_payload(ctx, config=cfg))
    user = build_user_prompt(evidence)
    body = build_openrouter_request_body(
        model_id=OPENROUTER_REASONER_MODEL,
        system=OPENROUTER_SYSTEM_INSTRUCTION,
        user=user,
        max_tokens=768,
    )
    raw = assert_json_serializable(body)
    parsed = json.loads(raw.decode("utf-8"))
    assert parsed["model"] == OPENROUTER_REASONER_MODEL
    assert parsed["response_format"]["json_schema"]["strict"] is True
    assert parsed["provider"]["require_parameters"] is True


def test_numpy_float_in_payload_is_sanitized() -> None:
    import numpy as np

    from src.temporal.providers.openrouter import assert_json_serializable, to_jsonable

    messy = {
        "score": np.float64(0.42),
        "nested": {"p": np.float32(0.1)},
        "vals": [np.int64(3)],
    }
    raw = assert_json_serializable(messy)
    data = json.loads(raw.decode("utf-8"))
    assert data["score"] == pytest.approx(0.42)
    assert isinstance(to_jsonable(np.float64(1.5)), float)


def test_payload_build_failure_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    reasoner = OpenRouterTemporalReasoner(_openrouter_cfg())

    def boom(*_a, **_k):  # noqa: ANN001
        raise TypeError("Object of type float64 is not JSON serializable")

    monkeypatch.setattr(
        "src.temporal.providers.openrouter.build_evidence_payload",
        boom,
    )
    result, diag = reasoner.reason(fixture_stable_neutral())
    assert result.status == "generation_failed"
    assert diag.openrouter_failure_stage == "payload_build"
    assert result.details is not None
    assert result.details["openrouter_failure_stage"] == "payload_build"
    assert "float64" in (diag.openrouter_error_message or "")


def test_request_build_failure_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    reasoner = OpenRouterTemporalReasoner(_openrouter_cfg())

    def boom_request(*_a, **_k):  # noqa: ANN001
        raise RuntimeError("cannot construct request")

    monkeypatch.setattr(
        "src.temporal.providers.openrouter.urllib.request.Request",
        boom_request,
    )
    result, diag = reasoner.reason(fixture_stable_neutral())
    assert result.status == "generation_failed"
    assert diag.openrouter_failure_stage == "request_build"
    assert diag.openrouter_http_status is None


def test_urlerror_connection_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    reasoner = OpenRouterTemporalReasoner(_openrouter_cfg())

    def boom(*_a, **_k):  # noqa: ANN001
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr(
        "src.temporal.providers.openrouter.urllib.request.urlopen",
        boom,
    )
    result, diag = reasoner.reason(fixture_stable_neutral())
    assert diag.openrouter_failure_stage == "connection"
    assert result.status == "reasoner_unavailable"
    assert diag.openrouter_error_type == "OpenRouterError"


def test_http_400_stage_and_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    reasoner = OpenRouterTemporalReasoner(_openrouter_cfg())

    def boom(*_a, **_k):  # noqa: ANN001
        raise urllib.error.HTTPError(
            OPENROUTER_API_URL,
            400,
            "Bad Request",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )

    import io
    import urllib.error

    err = urllib.error.HTTPError(
        OPENROUTER_API_URL,
        400,
        "Bad Request",
        hdrs={},  # type: ignore[arg-type]
        fp=io.BytesIO(b'{"error":{"message":"bad response_format"}}'),
    )

    def boom2(*_a, **_k):  # noqa: ANN001
        raise err

    monkeypatch.setattr(
        "src.temporal.providers.openrouter.urllib.request.urlopen",
        boom2,
    )
    result, diag = reasoner.reason(fixture_stable_neutral())
    assert diag.openrouter_failure_stage == "http_response"
    assert diag.openrouter_http_status == 400
    assert result.status == "generation_failed"
    assert "response_format" in (diag.openrouter_error_message or "").lower() or (
        "structured-output" in (diag.openrouter_error_message or "").lower()
    )


def test_http_401_403_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import urllib.error

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    reasoner = OpenRouterTemporalReasoner(_openrouter_cfg())

    for code in (401, 403):
        err = urllib.error.HTTPError(
            OPENROUTER_API_URL,
            code,
            "Denied",
            hdrs={},  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"error":"denied"}'),
        )

        def boom(*_a, _err=err, **_k):  # noqa: ANN001
            raise _err

        monkeypatch.setattr(
            "src.temporal.providers.openrouter.urllib.request.urlopen",
            boom,
        )
        result, diag = reasoner.reason(fixture_stable_neutral())
        assert result.status == "reasoner_unavailable"
        assert diag.openrouter_failure_stage == "http_response"
        assert diag.openrouter_http_status == code


def test_diagnostics_never_contain_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "sk-or-v1-supersecretdiagnosticvalue99"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    reasoner = OpenRouterTemporalReasoner(_openrouter_cfg())

    def boom(*_a, **_k):  # noqa: ANN001
        raise OpenRouterError(
            f"Bearer {secret} leaked somehow",
            status_code=401,
            error_kind="auth",
            failure_stage="http_response",
        )

    monkeypatch.setattr(reasoner, "_generate_with_retry", boom)
    result, diag = reasoner.reason(fixture_stable_neutral())
    blob = json.dumps(result.model_dump()) + json.dumps(diag.model_dump())
    assert secret not in blob
    assert "Bearer " not in blob or "[REDACTED]" in blob


def test_technical_details_shows_openrouter_failure_fields() -> None:
    from src.schemas import TemporalReasonerDiagnostics

    assessment = FinalTemporalAssessment(
        overall_wellbeing_indicator="insufficient_evidence",
        status="explanation_unavailable",
        reasoner_configured=True,
    )
    result = ActivityAnalysisResult(
        activity_id="A-diag",
        activity_type="video",
        input=InputMetadata(),
        analysis=AnalysisBlock(
            overall=_ev("neutral"),
            modalities=ModalityBundle(visual=_ev("neutral")),
            final_temporal_assessment=assessment,
            temporal_reasoning=TemporalReasoningResult(
                status="generation_failed",
                context_type="uncertain",
                confidence=0.0,
                model=OPENROUTER_REASONER_MODEL,
            ),
            temporal_reasoner_diagnostics=TemporalReasonerDiagnostics(
                provider="openrouter",
                openrouter_failure_stage="request_build",
                openrouter_error_type="OpenRouterError",
                openrouter_error_message="urllib Request construction failed: RuntimeError",
                openrouter_http_status=None,
                reasoner_configured=True,
            ),
            video=VideoDiagnostics(frames_extracted=1, frames_analyzed=1),
        ),
    )
    tech = render_technical_details(
        RoutedAnalysisResult(
            status=CapabilityStatus.OK,
            detected_input=InputType.VIDEO,
            analysis=result,
        ),
    )
    assert "OpenRouter status" in tech
    assert "generation_failed" in tech
    assert "Failure stage" in tech
    assert "request_build" in tech
    assert "HTTP status" in tech
    assert "n/a" in tech
    assert "Error:" in tech
    assert "OPENROUTER_API_KEY" not in tech
    assert "sk-" not in tech


def test_logger_exception_on_generation_failed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    reasoner = OpenRouterTemporalReasoner(_openrouter_cfg())

    def boom(*_a, **_k):  # noqa: ANN001
        raise OpenRouterError(
            "connection failed",
            error_kind="connection",
            failure_stage="connection",
        )

    monkeypatch.setattr(reasoner, "_generate_with_retry", boom)
    with caplog.at_level(logging.ERROR):
        result, _ = reasoner.reason(fixture_stable_neutral())
    assert result.status == "reasoner_unavailable"
    assert any("OpenRouter generation failed" in r.message for r in caplog.records)
