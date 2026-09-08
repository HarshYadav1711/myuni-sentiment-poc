"""Presentation helpers for the Gradio Hugging Face client (no inference)."""

from __future__ import annotations

from html import escape
from typing import Any, Optional

from src.routing.input_router import CapabilityStatus, InputType
from src.schemas import FinalTemporalAssessment, SentimentEvidence
from src.temporal.wellbeing import wellbeing_indicator_label
from src.ui.display import format_confidence_pct, format_probability_pct

BOTH_INPUTS_MESSAGE = "Please analyze one content item at a time."
NO_SPEECH_MESSAGE = "No meaningful speech was detected."
EMPTY_INPUT_MESSAGE = "Enter text or upload a supported image, audio, or video to analyze."
CONTEXT_UNAVAILABLE_MESSAGE = "Context explanation temporarily unavailable."

_LABEL_COLORS = {
    "positive": "#15803d",
    "neutral": "#475569",
    "negative": "#b91c1c",
}

_WELLBEING_COLORS = {
    "low_concern": "#15803d",
    "moderate_concern": "#b45309",
    "high_concern": "#b91c1c",
    "insufficient_evidence": "#475569",
}


def _color(label: Optional[str]) -> str:
    return _LABEL_COLORS.get((label or "").lower(), "#475569")


def _dist_html(probs: Optional[dict[str, Any]]) -> str:
    probs = probs or {}
    return f"""
    <div class="mu-dist">
      <div class="mu-dist-card">
        <div class="mu-dist-label">Positive</div>
        <div class="mu-dist-value">{escape(format_probability_pct(float(probs.get("positive", 0.0))))}</div>
      </div>
      <div class="mu-dist-card">
        <div class="mu-dist-label">Neutral</div>
        <div class="mu-dist-value">{escape(format_probability_pct(float(probs.get("neutral", 0.0))))}</div>
      </div>
      <div class="mu-dist-card">
        <div class="mu-dist-label">Negative</div>
        <div class="mu-dist-value">{escape(format_probability_pct(float(probs.get("negative", 0.0))))}</div>
      </div>
    </div>
    """


def _pill(label: str) -> str:
    color = _color(label)
    return (
        f'<span class="mu-pill" style="background:{color}18;color:{color};'
        f'border:1px solid {color}55;">● {escape(label.upper())}</span>'
    )


def _evidence_block(title: str, evidence: SentimentEvidence, model_name: str) -> str:
    return f"""
    <div class="mu-metric">
      <div class="mu-metric-label">{escape(title)}</div>
      {_pill(evidence.label)}
      <div class="mu-conf">Confidence <strong>{escape(format_confidence_pct(evidence.confidence))}</strong></div>
      <div class="mu-model">Model · {escape(model_name)}</div>
    </div>
    {_dist_html(evidence.probabilities)}
    """


def _wellbeing_block(assessment: FinalTemporalAssessment) -> str:
    indicator = assessment.overall_wellbeing_indicator
    color = _WELLBEING_COLORS.get(indicator, "#475569")
    label = wellbeing_indicator_label(indicator)
    return f"""
    <div class="mu-wellbeing">
      <div class="mu-wellbeing-label">Overall Well-Being Score</div>
      <div class="mu-wellbeing-value" style="color:{color};">{escape(label)}</div>
    </div>
    """


def _highlights_block(assessment: FinalTemporalAssessment) -> str:
    highlights = list(assessment.key_temporal_highlights or [])
    if not highlights:
        return (
            '<div class="mu-sub">Key Temporal Highlights</div>'
            '<p class="mu-copy">No temporal highlights available.</p>'
        )
    cards = []
    for hl in highlights[:5]:
        cards.append(
            f'<div class="mu-hl-card">'
            f'<div class="mu-hl-time">{escape(hl.timestamp_label)}</div>'
            f'<div class="mu-hl-desc">{escape(hl.description)}</div>'
            f"</div>"
        )
    return (
        '<div class="mu-sub">Key Temporal Highlights</div>'
        f'<div class="mu-highlights">{"".join(cards)}</div>'
    )


def _summary_block(assessment: FinalTemporalAssessment) -> str:
    if assessment.status != "ok" or not (assessment.summary_explanation or "").strip():
        return (
            '<div class="mu-sub">Summary Explanation</div>'
            f'<p class="mu-copy">{escape(CONTEXT_UNAVAILABLE_MESSAGE)}</p>'
        )
    return (
        '<div class="mu-sub">Summary Explanation</div>'
        f'<p class="mu-copy">{escape(assessment.summary_explanation.strip())}</p>'
    )


def _poc_note() -> str:
    return (
        '<p class="mu-wellbeing-note">'
        "POC content-level wellbeing indicator — not a clinical assessment."
        "</p>"
    )


def _shell(detected: str, body: str) -> str:
    return f"""
    <div class="mu-card">
      <div class="mu-kicker">Detected Input</div>
      <div class="mu-detected">{escape(detected)}</div>
      {body}
    </div>
    """


def _message_card(title: str, message: str, detected: Optional[str] = None) -> str:
    detected_html = ""
    if detected:
        detected_html = (
            f'<div class="mu-kicker">Detected Input</div>'
            f'<div class="mu-detected">{escape(detected)}</div>'
        )
    return f"""
    <div class="mu-card">
      {detected_html}
      <div class="mu-title">{escape(title)}</div>
      <p class="mu-copy">{escape(message)}</p>
    </div>
    """


def _ocr_unavailable(warnings: Optional[list[str]]) -> bool:
    joined = " ".join(warnings or []).lower()
    return "ocr unavailable" in joined or "tesseract" in joined


def render_idle() -> str:
    return """
    <div class="mu-card">
      <div class="mu-title">Analysis</div>
      <p class="mu-copy">Submit one text post or one media file. Content type is detected automatically.</p>
      <ul class="mu-list">
        <li>Text — Twitter-RoBERTa</li>
        <li>Image — SigLIP 2 visual sentiment</li>
        <li>Audio — Faster-Whisper transcript, then Twitter-RoBERTa</li>
        <li>Video — sampled frames (SigLIP 2) + optional speech, temporal context</li>
      </ul>
    </div>
    """


def render_validation(message: str) -> str:
    return _message_card("Unable to analyze", message)


def render_technical_details(routed: Any) -> str:
    """Client-safe technical notes. OCR gaps stay here, not in the main result."""
    detected = routed.detected_input
    kind = detected.value.upper() if detected else "UNKNOWN"
    lines = [
        f"**Detected input:** {kind}",
        f"**Display model:** {routed.model_display_name or '—'}",
        f"**Technical identifier:** `{routed.model_id or '—'}`",
    ]
    analysis = getattr(routed, "analysis", None)
    warnings: list[str] = []
    if analysis is not None:
        warnings = list(analysis.analysis.warnings or [])
        runtime = analysis.analysis.runtime
        if runtime is not None:
            models = runtime.models or {}
            if models.get("text"):
                lines.append(f"**Twitter-RoBERTa:** `{models['text']}`")
            if models.get("visual"):
                lines.append(f"**SigLIP 2:** `{models['visual']}`")
            if models.get("asr"):
                lines.append(
                    f"**Faster-Whisper:** `{models['asr']}` "
                    f"({models.get('asr_compute_type', 'int8')}, "
                    f"{models.get('asr_language', 'en')})"
                )
        if kind == "IMAGE":
            lines.append("**Visual checkpoint:** `google/siglip2-base-patch16-224`")
            if _ocr_unavailable(warnings):
                lines.append(
                    "**OCR:** unavailable in this environment. "
                    "Visual sentiment still ran; OCR text was not used."
                )
            elif analysis.analysis.ocr_text:
                lines.append("**OCR:** text extracted and scored when meaningful.")
            else:
                lines.append("**OCR:** no meaningful embedded text used.")
        if kind == "TEXT":
            lines.append(
                "**Text checkpoint:** `cardiffnlp/twitter-roberta-base-sentiment-latest`. "
                "Label is the highest class probability."
            )
        if kind == "AUDIO":
            lines.append(
                "**Audio path:** Faster-Whisper `base.en` (CPU int8) → Twitter-RoBERTa on the transcript."
            )
        if kind == "VIDEO":
            video = analysis.analysis.video
            if video is not None:
                lines.append(
                    f"**Sampling:** `{video.sampling_strategy}` · "
                    f"extracted {video.frames_extracted} · analyzed {video.frames_analyzed}"
                )
            lines.append(
                "**Video path:** FFmpeg frame sampling (CPU) · SigLIP 2 (ZeroGPU) · "
                "Faster-Whisper CPU int8 → Twitter-RoBERTa. Fusion is a POC baseline only."
            )
            overall = analysis.analysis.overall
            if overall is not None:
                lines.append(
                    f"**Overall sentiment (fusion POC):** `{overall.label}` · "
                    f"confidence={overall.confidence:.2f}"
                )
                if overall.probabilities:
                    lines.append(
                        "**RoBERTa/fusion class probs:** "
                        f"pos={float(overall.probabilities.get('positive', 0)):.3f} · "
                        f"neu={float(overall.probabilities.get('neutral', 0)):.3f} · "
                        f"neg={float(overall.probabilities.get('negative', 0)):.3f}"
                    )
            modalities = analysis.analysis.modalities
            if modalities.visual is not None:
                lines.append(
                    f"**Visual evidence:** `{modalities.visual.label}` "
                    f"(SigLIP 2)"
                )
            if modalities.speech is not None:
                lines.append(
                    f"**Speech evidence:** `{modalities.speech.label}` "
                    f"(Whisper → Twitter-RoBERTa)"
                )
            elif analysis.analysis.transcript:
                lines.append("**Speech:** transcript present; see pipeline notes if unused.")
            else:
                lines.append("**Speech:** no meaningful speech detected.")
            fusion = analysis.analysis.fusion
            if fusion is not None:
                used = ", ".join(fusion.contributing_modalities) or "none"
                lines.append(f"**Fusion modalities used:** {used}")
                if fusion.explanation:
                    lines.append(f"**Fusion note:** {fusion.explanation}")
            temporal = analysis.analysis.temporal_context
            if temporal is not None:
                feats = temporal.features
                cov = feats.evidence_coverage
                lines.append(
                    f"**Deterministic temporal context:** window={temporal.window_seconds}s · "
                    f"windows={len(temporal.windows)} · "
                    f"trajectory=`{feats.trajectory}` · "
                    f"usable_coverage={cov.overall_usable_coverage:.2f}"
                )
            reasoning = analysis.analysis.temporal_reasoning
            if reasoning is not None:
                lines.append(
                    f"**Contextual reasoning (advisory):** status=`{reasoning.status}` · "
                    f"context_type=`{reasoning.context_type}` · "
                    f"confidence={reasoning.confidence:.2f}"
                )
            final = analysis.analysis.final_temporal_assessment
            if final is not None:
                lines.append(
                    f"**Final temporal assessment:** status=`{final.status}` · "
                    f"indicator=`{final.overall_wellbeing_indicator}` · "
                    f"client_label=`{wellbeing_indicator_label(final.overall_wellbeing_indicator)}` · "
                    f"reasoner_configured=`{final.reasoner_configured}`"
                )
                if final.evidence_summary:
                    lines.append(f"**Evidence summary:** {final.evidence_summary}")
                if final.uncertainty_note:
                    lines.append(f"**Uncertainty:** {final.uncertainty_note}")
            reasoner_diag = analysis.analysis.temporal_reasoner_diagnostics
            if reasoner_diag is not None:
                lines.append(
                    f"**Reasoner timings:** prompt={reasoner_diag.prompt_construction_seconds or 0:.2f}s · "
                    f"generation={reasoner_diag.generation_seconds or 0:.2f}s · "
                    f"parse={reasoner_diag.parse_validation_seconds or 0:.2f}s · "
                    f"total={reasoner_diag.total_reasoner_seconds or 0:.2f}s"
                )
                if reasoner_diag.provider:
                    lines.append(f"**Reasoner provider:** `{reasoner_diag.provider}`")
                if reasoner_diag.provider == "openrouter":
                    req_model = None
                    fallback_models = None
                    model_chain = None
                    if reasoner_diag.generation_kwargs:
                        raw_chain = reasoner_diag.generation_kwargs.get("model_chain")
                        if isinstance(raw_chain, list) and raw_chain:
                            model_chain = [
                                str(item).strip()
                                for item in raw_chain
                                if str(item).strip()
                            ]
                        raw_req = reasoner_diag.generation_kwargs.get("requested_model")
                        if not isinstance(raw_req, str):
                            raw_req = reasoner_diag.generation_kwargs.get("model")
                        if isinstance(raw_req, str) and raw_req.strip():
                            req_model = raw_req.strip()
                        elif model_chain:
                            req_model = model_chain[0]
                        raw_fb = reasoner_diag.generation_kwargs.get("fallback_models")
                        if isinstance(raw_fb, list) and raw_fb:
                            fallback_models = [
                                str(item).strip()
                                for item in raw_fb
                                if str(item).strip()
                            ]
                        elif model_chain and len(model_chain) > 1:
                            fallback_models = model_chain[1:]
                    if req_model:
                        lines.append(f"**OpenRouter requested model:** `{req_model}`")
                    if fallback_models:
                        lines.append(
                            "**OpenRouter fallback models:** `"
                            + "`, `".join(fallback_models)
                            + "`"
                        )
                    if model_chain:
                        lines.append(
                            "**OpenRouter model chain:** `"
                            + "` → `".join(model_chain)
                            + "`"
                        )
                    if reasoner_diag.openrouter_routed_model:
                        lines.append(
                            f"**OpenRouter routed model:** `{reasoner_diag.openrouter_routed_model}`"
                        )
                    shape = reasoner_diag.openrouter_response_shape
                    if isinstance(shape, dict) and shape:
                        fr = shape.get("finish_reason")
                        if isinstance(fr, str) and fr.strip():
                            lines.append(f"**OpenRouter finish_reason:** `{fr.strip()}`")
                        lines.append(
                            "**OpenRouter content present:** "
                            f"`{bool(shape.get('content_present'))}` "
                            f"(type=`{shape.get('content_type')}`"
                            + (
                                f", length={shape.get('content_length')}"
                                if shape.get("content_length") is not None
                                else ""
                            )
                            + ")"
                        )
                        lines.append(
                            "**OpenRouter reasoning present:** "
                            f"`{bool(shape.get('reasoning_present'))}`"
                            + (
                                f" (length={shape.get('reasoning_length')})"
                                if shape.get("reasoning_length") is not None
                                else ""
                            )
                        )
                        if shape.get("reasoning_details_present"):
                            lines.append(
                                "**OpenRouter reasoning_details:** "
                                f"present=`True` count=`{shape.get('reasoning_details_count')}`"
                            )
                        usage = shape.get("usage") if isinstance(shape.get("usage"), dict) else {}
                        if usage:
                            usage_bits = []
                            for key in (
                                "prompt_tokens",
                                "completion_tokens",
                                "total_tokens",
                                "reasoning_tokens",
                            ):
                                if usage.get(key) is not None:
                                    usage_bits.append(f"{key}={usage[key]}")
                            if usage_bits:
                                lines.append(
                                    "**OpenRouter usage:** `" + ", ".join(usage_bits) + "`"
                                )
                    if reasoner_diag.openrouter_application_fallback_attempted:
                        lines.append("**OpenRouter application fallback:** `attempted`")
                        if reasoner_diag.openrouter_application_fallback_from_model:
                            lines.append(
                                "**OpenRouter fallback from model:** "
                                f"`{reasoner_diag.openrouter_application_fallback_from_model}`"
                            )
                        if reasoner_diag.openrouter_application_fallback_remaining_models:
                            lines.append(
                                "**OpenRouter fallback remaining models:** `"
                                + "`, `".join(
                                    reasoner_diag.openrouter_application_fallback_remaining_models
                                )
                                + "`"
                            )
                    if reasoner_diag.openrouter_attempt_count:
                        lines.append(
                            f"**OpenRouter HTTP attempts:** `{reasoner_diag.openrouter_attempt_count}`"
                        )
                if reasoning is not None and reasoner_diag.provider == "openrouter":
                    http_disp = (
                        reasoner_diag.openrouter_http_status
                        if reasoner_diag.openrouter_http_status is not None
                        else reasoner_diag.http_status
                    )
                    http_label = "n/a" if http_disp is None else str(http_disp)
                    stage = reasoner_diag.openrouter_failure_stage or reasoner_diag.reasoner_failure_stage or "n/a"
                    err = reasoner_diag.openrouter_error_message or reasoner_diag.reasoner_error_message or "n/a"
                    lines.append(f"**OpenRouter status:** `{reasoning.status}`")
                    lines.append(f"**Failure stage:** `{stage}`")
                    lines.append(f"**HTTP status:** {http_label}")
                    lines.append(f"**Error:** {err}")
                    if reasoner_diag.openrouter_error_type:
                        lines.append(
                            f"**Error type:** `{reasoner_diag.openrouter_error_type}`"
                        )
    if warnings:
        lines.append("**Pipeline notes:**")
        for warning in warnings[:8]:
            lines.append(f"- {warning}")
    return "\n\n".join(lines)


def render_routed_result(routed: Any) -> str:
    status = routed.status
    detected = routed.detected_input
    kind = detected.value.upper() if detected else "UNKNOWN"

    if status == CapabilityStatus.VALIDATION_ERROR:
        return render_validation(routed.message or "Unable to complete analysis.")

    if status == CapabilityStatus.INSUFFICIENT_EVIDENCE:
        if detected == InputType.AUDIO:
            return _shell(
                "AUDIO",
                f'<p class="mu-copy">{escape(NO_SPEECH_MESSAGE)}</p>'
                '<p class="mu-note">No transcript sentiment was invented.</p>',
            )
        return _message_card(
            "Insufficient evidence",
            routed.message or "Insufficient evidence for a sentiment result.",
            detected=kind,
        )

    if status != CapabilityStatus.OK or routed.analysis is None:
        return render_validation(routed.message or "Unable to complete analysis.")

    analysis = routed.analysis
    block = analysis.analysis
    modalities = block.modalities

    if kind == "TEXT" and modalities.text is not None:
        body = _evidence_block("Overall Sentiment", modalities.text, "Twitter-RoBERTa")
        return _shell("TEXT", body)

    if kind == "IMAGE" and modalities.visual is not None:
        body = _evidence_block("Visual Sentiment", modalities.visual, "SigLIP 2")
        return _shell("IMAGE", body)

    if kind == "AUDIO":
        if modalities.speech is None:
            return _shell(
                "AUDIO",
                f'<p class="mu-copy">{escape(NO_SPEECH_MESSAGE)}</p>',
            )
        transcript = block.transcript or ""
        transcript_html = (
            f'<div class="mu-sub">Transcript</div>'
            f'<blockquote class="mu-quote">{escape(transcript)}</blockquote>'
            if transcript
            else f'<p class="mu-copy">{escape(NO_SPEECH_MESSAGE)}</p>'
        )
        body = transcript_html + _evidence_block(
            "Transcript Sentiment",
            modalities.speech,
            "Faster-Whisper + Twitter-RoBERTa",
        )
        return _shell("AUDIO", body)

    if kind == "VIDEO":
        parts: list[str] = []
        assessment = block.final_temporal_assessment
        if assessment is not None:
            parts.append(_wellbeing_block(assessment))
            parts.append(_highlights_block(assessment))
            parts.append(_summary_block(assessment))
            parts.append(_poc_note())
        else:
            parts.append(
                '<p class="mu-copy">Temporal wellbeing assessment unavailable.</p>'
            )
            parts.append(_poc_note())
        return _shell("VIDEO", "".join(parts))

    evidence = modalities.text or modalities.visual or modalities.speech or block.overall
    body = _evidence_block("Overall Sentiment", evidence, routed.model_display_name or "—")
    return _shell(kind, body)
