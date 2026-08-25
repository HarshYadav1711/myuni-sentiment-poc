#!/usr/bin/env python3
"""MyUni Sentiment Intelligence — Streamlit client demo.

Visual design aligned to the product mockup (soft lavender background,
floating header icons, white workspace card). Unified input remains —
no Text / Image / Video modality tabs.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import MyUniSentimentPipeline
from src.routing.input_router import CapabilityStatus, InputType
from src.ui.display import (
    format_confidence_pct,
    format_probability_pct,
    format_score,
    label_color,
)

logger = logging.getLogger(__name__)

MAX_CHARS = 5000
MEDIA_TYPES = ["jpg", "jpeg", "png", "webp", "bmp", "mp4", "mov", "avi", "webm", "mkv"]


def _theme_css(mode: str) -> str:
    if mode == "dark":
        page_bg = (
            "radial-gradient(ellipse 70% 45% at 15% 10%, rgba(99,102,241,0.22) 0%, transparent 55%),"
            "radial-gradient(ellipse 60% 40% at 85% 15%, rgba(168,85,247,0.18) 0%, transparent 50%),"
            "radial-gradient(ellipse 80% 50% at 50% 100%, rgba(59,130,246,0.14) 0%, transparent 55%),"
            "linear-gradient(180deg, #12161f 0%, #0e1218 100%)"
        )
        page_fg = "#e8ecf3"
        brand = "#a5b4fc"
        title = "#f4f6fb"
        sub = "#9aa3b5"
        card_bg = "#171c26"
        card_border = "#2a3140"
        card_shadow = "0 24px 60px rgba(0,0,0,0.45)"
        panel = "#1c2230"
        muted = "#9aa3b5"
        accent = "#7c6cf0"
        accent_2 = "#4f7df3"
        privacy_bg = "#14301f"
        privacy_fg = "#9be4b5"
        privacy_border = "#245c3a"
        chip_bg = "#222b38"
        chip_fg = "#c5ced8"
        float_bg = "#1c2230"
        float_shadow = "0 10px 28px rgba(0,0,0,0.35)"
        input_bg = "#12161f"
        dash = "#3a4558"
    else:
        # Soft ethereal lavender / sky wash matching the mockup.
        page_bg = (
            "radial-gradient(ellipse 75% 50% at 12% 8%, rgba(196,181,253,0.55) 0%, transparent 55%),"
            "radial-gradient(ellipse 65% 45% at 88% 12%, rgba(165,180,252,0.50) 0%, transparent 52%),"
            "radial-gradient(ellipse 90% 55% at 50% 100%, rgba(191,219,254,0.65) 0%, transparent 58%),"
            "linear-gradient(180deg, #f7f5ff 0%, #f3f6fc 45%, #e8eef9 100%)"
        )
        page_fg = "#1c2430"
        brand = "#3b5bdb"
        title = "#152033"
        sub = "#6b7280"
        card_bg = "#ffffff"
        card_border = "#e8ecf4"
        card_shadow = "0 22px 56px rgba(70, 90, 150, 0.14)"
        panel = "#f8f9fc"
        muted = "#6b7280"
        accent = "#7c5cfc"
        accent_2 = "#4f7df3"
        privacy_bg = "#eaf8f1"
        privacy_fg = "#1f6b3a"
        privacy_border = "#bfe8cf"
        chip_bg = "#eef1f8"
        chip_fg = "#4b5568"
        float_bg = "#ffffff"
        float_shadow = "0 12px 30px rgba(80, 100, 160, 0.16)"
        input_bg = "#ffffff"
        dash = "#c5cde0"

    return f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

  :root {{
    --font-display: "Fraunces", "Iowan Old Style", "Palatino Linotype", serif;
    --font-ui: "Plus Jakarta Sans", "Segoe UI", sans-serif;
  }}

  html, body {{
    height: auto !important;
    overflow-y: auto !important;
    font-family: var(--font-ui) !important;
  }}
  [data-testid="stAppViewContainer"],
  .stApp {{
    background: {page_bg} !important;
    color: {page_fg};
    height: auto !important;
    min-height: 100vh;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    font-family: var(--font-ui) !important;
  }}
  .stApp p, .stApp label, .stApp li,
  [data-testid="stMarkdownContainer"],
  [data-testid="stWidgetLabel"],
  [data-testid="stCaptionContainer"] {{
    font-family: var(--font-ui) !important;
  }}
  section.main, [data-testid="stMain"] {{
    height: auto !important;
    overflow: visible !important;
  }}
  [data-testid="stMainBlockContainer"],
  .block-container {{
    padding-top: 2.6rem !important;
    padding-bottom: 2.2rem !important;
    max-width: 1040px;
    overflow: visible !important;
  }}
  [data-testid="stHorizontalBlock"],
  [data-testid="column"],
  [data-testid="stVerticalBlock"] {{
    overflow: visible !important;
  }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  div[data-testid="stSidebar"] {{ display: none; }}
  [data-testid="collapsedControl"] {{ display: none; }}

  div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {card_bg} !important;
    border: 1px solid {card_border} !important;
    border-radius: 22px !important;
    box-shadow: {card_shadow} !important;
    padding: 1rem 0.85rem 1.1rem 0.85rem !important;
  }}

  /* ---- Hero with floating icons ---- */
  .hero-wrap {{
    position: relative;
    text-align: center;
    padding: 1.1rem 0 1.35rem 0;
    margin: 0 auto 0.35rem auto;
    max-width: 920px;
    min-height: 168px;
  }}
  .float-card {{
    position: absolute;
    width: 52px;
    height: 52px;
    border-radius: 14px;
    background: {float_bg};
    box-shadow: {float_shadow};
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 2;
  }}
  .float-card svg {{ width: 24px; height: 24px; }}
  .float-tl {{ top: 8px; left: 6%; }}
  .float-bl {{ top: 88px; left: 14%; }}
  .float-tr {{ top: 4px; right: 8%; }}
  .float-br {{ top: 92px; right: 12%; }}
  .float-dot {{
    position: absolute;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    opacity: 0.55;
    z-index: 1;
  }}
  .dot-a {{ background: #a78bfa; top: 36px; left: 22%; }}
  .dot-b {{ background: #60a5fa; top: 18px; right: 24%; }}
  .dot-c {{ background: #c4b5fd; top: 120px; left: 28%; }}
  .dot-d {{ background: #93c5fd; top: 130px; right: 26%; }}

  .brand-row {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.55rem;
    margin: 0.15rem 0 0.45rem 0;
    position: relative;
    z-index: 3;
  }}
  .brand-mark {{
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: linear-gradient(145deg, #4f7df3 0%, #6d5efc 100%);
    color: #fff;
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 1.15rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    position: relative;
    box-shadow: 0 8px 18px rgba(79, 125, 243, 0.35);
  }}
  .brand-mark::before {{
    content: "";
    position: absolute;
    top: -7px;
    width: 22px;
    height: 10px;
    border-radius: 3px 3px 2px 2px;
    background: #3b5bdb;
    box-shadow: 8px 2px 0 -2px #3b5bdb, -8px 2px 0 -2px #3b5bdb;
  }}
  .brand-name {{
    font-family: var(--font-ui);
    font-weight: 750;
    font-size: 1.35rem;
    color: {brand};
    letter-spacing: -0.03em;
  }}
  h1.hero-title {{
    font-family: var(--font-display) !important;
    font-weight: 700;
    font-size: 2.45rem;
    line-height: 1.15;
    letter-spacing: -0.035em;
    color: {title};
    margin: 0.2rem 0 0.45rem 0;
    position: relative;
    z-index: 3;
  }}
  p.hero-sub {{
    font-family: var(--font-ui) !important;
    color: {sub};
    font-size: 1.02rem;
    font-weight: 450;
    margin: 0 auto;
    max-width: 520px;
    line-height: 1.55;
    position: relative;
    z-index: 3;
  }}

  .step-title {{
    font-family: var(--font-display) !important;
    font-weight: 650;
    font-size: 1.15rem;
    letter-spacing: -0.02em;
    color: {title};
    margin: 0 0 0.25rem 0;
  }}
  .step-help {{
    font-family: var(--font-ui) !important;
    color: {muted};
    font-size: 0.9rem;
    margin: 0 0 0.75rem 0;
    line-height: 1.45;
  }}
  .char-meta {{
    font-family: var(--font-ui) !important;
    color: {muted};
    font-size: 0.8rem;
    text-align: right;
    margin-top: -0.45rem;
    margin-bottom: 0.65rem;
    font-variant-numeric: tabular-nums;
  }}
  .chip-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin: 0.1rem 0 0.9rem 0;
  }}
  .chip {{
    font-family: var(--font-ui) !important;
    background: {chip_bg};
    color: {chip_fg};
    border-radius: 999px;
    padding: 0.28rem 0.7rem;
    font-size: 0.76rem;
    font-weight: 550;
  }}
  .panel-title {{
    font-family: var(--font-display) !important;
    font-weight: 650;
    font-size: 1.2rem;
    letter-spacing: -0.02em;
    color: {title};
    margin: 0 0 0.2rem 0;
  }}
  .panel-sub {{
    font-family: var(--font-ui) !important;
    color: {muted};
    font-size: 0.88rem;
    margin: 0 0 0.95rem 0;
  }}
  .preview-row {{
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
    padding: 0.55rem 0.15rem;
    margin-bottom: 0.35rem;
  }}
  .preview-icon {{
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 1rem;
  }}
  .preview-row strong {{
    font-family: var(--font-ui) !important;
    display: block;
    font-size: 0.92rem;
    font-weight: 650;
    color: {title};
  }}
  .preview-row span {{
    font-family: var(--font-ui) !important;
    display: block;
    font-size: 0.8rem;
    color: {muted};
    margin-top: 0.08rem;
  }}
  .result-badge {{
    font-family: var(--font-ui) !important;
    display: inline-block;
    border-radius: 999px;
    padding: 0.28rem 0.75rem;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }}
  .result-card {{
    background: {panel};
    border-radius: 14px;
    padding: 0.9rem 0.95rem;
    margin-bottom: 0.65rem;
    font-family: var(--font-ui) !important;
  }}
  .result-card .muted {{ color: {muted}; font-size: 0.8rem; }}
  .privacy-box {{
    margin-top: 0.95rem;
    background: {privacy_bg};
    color: {privacy_fg};
    border: 1px solid {privacy_border};
    border-radius: 14px;
    padding: 0.8rem 0.9rem;
    font-family: var(--font-ui) !important;
    font-size: 0.84rem;
    line-height: 1.45;
  }}
  .info-panel {{
    border: 1px solid {card_border};
    background: {panel};
    border-radius: 14px;
    padding: 0.85rem 0.95rem;
    color: {page_fg};
    font-family: var(--font-ui) !important;
    font-size: 0.9rem;
    line-height: 1.45;
  }}
  .app-footer {{
    text-align: center;
    color: {muted};
    font-family: var(--font-ui) !important;
    font-size: 0.78rem;
    margin-top: 1rem;
    line-height: 1.6;
  }}
  .beta-pill {{
    display: inline-block;
    margin-left: 0.35rem;
    padding: 0.1rem 0.45rem;
    border-radius: 999px;
    background: linear-gradient(90deg, {accent}, {accent_2});
    color: #fff;
    font-size: 0.68rem;
    font-weight: 700;
    vertical-align: middle;
  }}

  /* Soften Streamlit inputs toward mockup */
  [data-testid="stTextArea"] textarea {{
    background: {input_bg} !important;
    border-radius: 12px !important;
    font-family: var(--font-ui) !important;
    color: {page_fg} !important;
  }}
  [data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"] {{
    display: none !important;
  }}
  [data-testid="stFileUploaderDropzone"],
  [data-testid="stFileUploadDropzone"] {{
    border: 1.5px dashed {dash} !important;
    border-radius: 14px !important;
    background: {panel} !important;
  }}
  /* Upload button: force readable contrast (Streamlit secondary is too dark) */
  [data-testid="stFileUploaderDropzone"] button,
  [data-testid="stFileUploadDropzone"] button,
  [data-testid="stBaseButton-secondary"] {{
    background: linear-gradient(90deg, {accent} 0%, {accent_2} 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--font-ui) !important;
    font-weight: 650 !important;
  }}
  [data-testid="stFileUploaderDropzone"] button p,
  [data-testid="stFileUploaderDropzone"] button span,
  [data-testid="stFileUploaderDropzone"] button div,
  [data-testid="stFileUploadDropzone"] button p,
  [data-testid="stFileUploadDropzone"] button span,
  [data-testid="stFileUploadDropzone"] button div,
  [data-testid="stBaseButton-secondary"] p,
  [data-testid="stBaseButton-secondary"] span {{
    color: #ffffff !important;
  }}
  [data-testid="stFileUploaderDropzoneInstructions"] span {{
    color: {muted} !important;
    font-family: var(--font-ui) !important;
  }}
  [data-testid="stWidgetLabel"] p,
  [data-testid="stWidgetLabel"] span {{
    color: {muted} !important;
  }}
  [data-testid="stIconMaterial"] {{
    font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
  }}

  div.stButton > button[kind="primary"] {{
    font-family: var(--font-ui) !important;
    background: linear-gradient(90deg, {accent} 0%, {accent_2} 100%) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
    border-radius: 14px !important;
    min-height: 3rem !important;
    box-shadow: 0 10px 24px rgba(99, 102, 241, 0.28);
  }}
  div.stButton > button[kind="primary"]:hover {{
    filter: brightness(1.04);
    border: none !important;
    color: #fff !important;
  }}
  .dist-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.55rem;
    margin: 0.35rem 0 0.85rem 0;
  }}
  .dist-card {{
    background: {panel};
    border: 1px solid {card_border};
    border-radius: 12px;
    padding: 0.7rem 0.65rem;
    text-align: center;
  }}
  .dist-card .dist-label {{
    font-family: var(--font-ui) !important;
    font-size: 0.78rem;
    font-weight: 600;
    color: {muted} !important;
    margin-bottom: 0.25rem;
  }}
  .dist-card .dist-value {{
    font-family: var(--font-ui) !important;
    font-size: 1.25rem;
    font-weight: 750;
    color: {title} !important;
    letter-spacing: -0.02em;
  }}
  @media (max-width: 900px) {{
    .float-card, .float-dot {{ display: none; }}
    h1.hero-title {{ font-size: 1.9rem; }}
    .dist-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
"""


_FLOATING_HERO = """
<div class="hero-wrap">
  <div class="float-card float-tl" aria-hidden="true">
    <svg viewBox="0 0 24 24" fill="none"><path d="M5 6.5A2.5 2.5 0 0 1 7.5 4h9A2.5 2.5 0 0 1 19 6.5v6A2.5 2.5 0 0 1 16.5 15H10l-3.8 3.2c-.5.4-1.2.1-1.2-.5V6.5Z" stroke="#4F7DF3" stroke-width="1.8" stroke-linejoin="round"/><path d="M8.5 9h7M8.5 12h4.5" stroke="#94A3B8" stroke-width="1.6" stroke-linecap="round"/></svg>
  </div>
  <div class="float-card float-bl" aria-hidden="true">
    <svg viewBox="0 0 24 24" fill="none"><rect x="4" y="5.5" width="16" height="13" rx="2.2" stroke="#22C55E" stroke-width="1.8"/><circle cx="9" cy="10" r="1.4" fill="#22C55E"/><path d="M5.5 16.5 9.2 13l2.4 2.2 3.1-3.4 3.8 4.7" stroke="#22C55E" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
  </div>
  <div class="float-card float-tr" aria-hidden="true">
    <svg viewBox="0 0 24 24" fill="none"><rect x="4" y="5.5" width="16" height="13" rx="2.2" stroke="#8B5CF6" stroke-width="1.8"/><path d="M10.2 9.2v5.6l5-2.8-5-2.8Z" fill="#8B5CF6"/></svg>
  </div>
  <div class="float-card float-br" aria-hidden="true">
    <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="8" stroke="#F59E0B" stroke-width="1.8"/><circle cx="9.2" cy="10.5" r="1.1" fill="#F59E0B"/><circle cx="14.8" cy="10.5" r="1.1" fill="#F59E0B"/><path d="M8.8 14.2c1.1 1.3 2.1 1.9 3.2 1.9s2.1-.6 3.2-1.9" stroke="#F59E0B" stroke-width="1.6" stroke-linecap="round"/></svg>
  </div>
  <div class="float-dot dot-a"></div>
  <div class="float-dot dot-b"></div>
  <div class="float-dot dot-c"></div>
  <div class="float-dot dot-d"></div>

  <div class="brand-row">
    <div class="brand-mark">U</div>
    <div class="brand-name">MyUni</div>
  </div>
  <h1 class="hero-title">Sentiment Intelligence</h1>
  <p class="hero-sub">AI-powered sentiment and well-being analysis for a healthier campus community.</p>
</div>
"""


@st.cache_resource(show_spinner="Loading text analysis model (first run only)…")
def get_pipeline() -> MyUniSentimentPipeline:
    """Cache pipeline across Streamlit reruns; RoBERTa loads lazily on first text run."""
    return MyUniSentimentPipeline()


def _save_upload(upload, dest_dir: Path) -> Path:
    suffix = Path(upload.name).suffix or ".bin"
    path = dest_dir / f"upload_{uuid.uuid4().hex[:10]}{suffix}"
    path.write_bytes(upload.getvalue())
    return path


def _cleanup_temp_dir(path: Path) -> None:
    import shutil

    try:
        shutil.rmtree(path, ignore_errors=False)
    except OSError as exc:
        logger.warning("Failed to remove temporary upload dir %s: %s", path, exc)


def _render_idle_preview() -> None:
    st.markdown('<div class="panel-title">Analysis Preview</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="panel-sub">See how we analyze and understand your content.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="preview-row">
          <div class="preview-icon" style="background:#e8f8ef;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="#16a34a" stroke-width="1.8"/><circle cx="9" cy="10.5" r="1.1" fill="#16a34a"/><circle cx="15" cy="10.5" r="1.1" fill="#16a34a"/><path d="M8.5 14.2c1.1 1.4 2.2 2 3.5 2s2.4-.6 3.5-2" stroke="#16a34a" stroke-width="1.6" stroke-linecap="round"/></svg>
          </div>
          <div><strong>Overall Sentiment</strong><span>Positive, Neutral, or Negative</span></div>
        </div>
        <div class="preview-row">
          <div class="preview-icon" style="background:#efe9ff;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 4.5 13.6 9H18l-3.5 2.7L15.8 17 12 14.2 8.2 17l1.3-5.3L6 9h4.4L12 4.5Z" stroke="#7c3aed" stroke-width="1.5" stroke-linejoin="round"/></svg>
          </div>
          <div><strong>Emotion Detection</strong><span>Joy, Sadness, Anger, Fear, etc.</span></div>
        </div>
        <div class="preview-row">
          <div class="preview-icon" style="background:#e8f0ff;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 20s-7-4.4-7-9.2A4.2 4.2 0 0 1 12 7.2a4.2 4.2 0 0 1 7 3.6C19 15.6 12 20 12 20Z" stroke="#2563eb" stroke-width="1.7" stroke-linejoin="round"/></svg>
          </div>
          <div><strong>Well-being Indicators</strong><span>Mental well-being assessment</span></div>
        </div>
        <div class="preview-row">
          <div class="preview-icon" style="background:#fff4e8;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 4.5 21 19H3L12 4.5Z" stroke="#ea580c" stroke-width="1.7" stroke-linejoin="round"/><path d="M12 10v4.2M12 16.5h.01" stroke="#ea580c" stroke-width="1.7" stroke-linecap="round"/></svg>
          </div>
          <div><strong>Risk Assessment</strong><span>Identify potential concerns</span></div>
        </div>
        <div class="preview-row">
          <div class="preview-icon" style="background:#e8f4ff;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M5 7.2A2.2 2.2 0 0 1 7.2 5h9.6A2.2 2.2 0 0 1 19 7.2v5.1A2.2 2.2 0 0 1 16.8 14.5H10l-3.5 2.8c-.45.35-1.1.05-1.1-.5V7.2Z" stroke="#0284c7" stroke-width="1.7" stroke-linejoin="round"/></svg>
          </div>
          <div><strong>Contextual Insights</strong><span>Detailed analysis and suggestions</span></div>
        </div>
        <div class="privacy-box">
          <strong>Privacy First</strong><br>
          All analysis is confidential and follows strict privacy guidelines.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Live in this build: Overall Sentiment (English text). "
        "Other preview items are roadmap capabilities.",
    )


def _render_text_result_panel(routed: Any) -> None:
    result = routed.analysis
    # Prefer Twitter-RoBERTa modality evidence for label/distribution (truthful model output).
    # Fusion overall can rebuild soft probabilities that hide real negative mass.
    text_ev = result.analysis.modalities.text
    evidence = text_ev or result.analysis.overall
    color = label_color(evidence.label)
    probs = evidence.probabilities or {}

    st.markdown('<div class="panel-title">Analysis Results</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="panel-sub">Detected input: Text</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="result-card">
          <div class="muted">Overall Sentiment</div>
          <div style="margin-top:0.4rem">
            <span class="result-badge" style="background:{color}22;color:{color};border:1px solid {color}55;">
              {evidence.label}
            </span>
          </div>
          <div style="margin-top:0.7rem;font-size:0.9rem;color:inherit;">
            Confidence <strong>{format_confidence_pct(evidence.confidence)}</strong>
            &nbsp;·&nbsp;
            Score <strong>{format_score(evidence.score)}</strong>
          </div>
          <div class="muted" style="margin-top:0.45rem;">
            Model · {routed.model_display_name or "Twitter-RoBERTa"}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="muted" style="font-weight:650;margin:0.15rem 0 0.35rem 0;">'
        "Sentiment distribution</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="dist-grid">
          <div class="dist-card">
            <div class="dist-label">Positive</div>
            <div class="dist-value">{format_probability_pct(float(probs.get("positive", 0.0)))}</div>
          </div>
          <div class="dist-card">
            <div class="dist-label">Neutral</div>
            <div class="dist-value">{format_probability_pct(float(probs.get("neutral", 0.0)))}</div>
          </div>
          <div class="dist-card">
            <div class="dist-label">Negative</div>
            <div class="dist-value">{format_probability_pct(float(probs.get("negative", 0.0)))}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Technical details", expanded=False):
        st.code(routed.model_id or "cardiffnlp/twitter-roberta-base-sentiment-latest")
        st.caption(
            "Label = highest model probability. "
            "POC score = P(positive) − P(negative). Evaluation default only.",
        )

    st.markdown(
        """
        <div class="privacy-box">
          <strong>Privacy First</strong><br>
          All analysis is confidential and follows strict privacy guidelines.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_status_panel(payload: dict[str, Any]) -> None:
    status = payload.get("status")
    if status == CapabilityStatus.NOT_IMPLEMENTED:
        detected = payload.get("detected_input")
        label = detected.value.title() if detected else "Media"
        st.markdown('<div class="panel-title">Analysis Results</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="panel-sub">Detected input: {label}</p>',
            unsafe_allow_html=True,
        )
        message = (payload.get("message") or "").replace("\n", "<br>")
        st.markdown(f'<div class="info-panel">{message}</div>', unsafe_allow_html=True)
        return

    if status == CapabilityStatus.VALIDATION_ERROR:
        st.markdown('<div class="panel-title">Analysis Results</div>', unsafe_allow_html=True)
        st.warning(payload.get("message") or "Invalid input.")
        return

    if status == CapabilityStatus.OK and payload.get("routed") is not None:
        _render_text_result_panel(payload["routed"])
        return

    st.error(payload.get("message") or "Unable to complete analysis.")


def main() -> None:
    st.set_page_config(
        page_title="MyUni Sentiment Intelligence",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "light"
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    st.markdown(_theme_css(st.session_state.ui_theme), unsafe_allow_html=True)

    _, toggle_r = st.columns([5, 1])
    with toggle_r:
        night = st.toggle(
            "Night",
            value=st.session_state.ui_theme == "dark",
            key="theme_night_toggle",
            help="Day / night appearance",
        )
    new_theme = "dark" if night else "light"
    if new_theme != st.session_state.ui_theme:
        st.session_state.ui_theme = new_theme
        st.rerun()

    st.markdown(_FLOATING_HERO, unsafe_allow_html=True)

    with st.container(border=True):
        left, right = st.columns([1.15, 1], gap="large")

        with left:
            st.markdown('<div class="step-title">1. Enter your content</div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="step-help">Type or paste your social post, comment, or any text content. '
                "You can also upload media — input type is detected automatically.</p>",
                unsafe_allow_html=True,
            )

            text = st.text_area(
                "Share your thoughts...",
                height=150,
                key="draft_text",
                max_chars=MAX_CHARS,
                placeholder="Share your thoughts...",
                label_visibility="collapsed",
            )
            char_count = len(text or "")
            st.markdown(
                f'<div class="char-meta">{char_count} / {MAX_CHARS}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="chip-row">
                  <span class="chip">Supports English text</span>
                  <span class="chip">Max 5,000 characters</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            upload = st.file_uploader(
                "Or upload a file",
                type=MEDIA_TYPES,
                key="media_upload",
                help="Supported media is auto-detected. Image/video scoring is not enabled in this text-baseline build.",
            )
            if upload is not None and not (text and text.strip()):
                mime = (upload.type or "").lower()
                if mime.startswith("image/"):
                    st.image(upload, use_container_width=True)
                elif mime.startswith("video/"):
                    st.video(upload)

            st.markdown('<div class="step-title">2. Analyze</div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="step-help">Our AI will analyze the content and generate sentiment insights.</p>',
                unsafe_allow_html=True,
            )
            analyze = st.button("Analyze Now", type="primary", use_container_width=True)

            if analyze:
                tmp_root: Optional[Path] = None
                try:
                    media_path = None
                    mime_type = None
                    filename = None
                    if not (text and text.strip()) and upload is not None:
                        tmp_root = Path(tempfile.mkdtemp(prefix="myuni_st_"))
                        media_path = str(_save_upload(upload, tmp_root))
                        mime_type = upload.type
                        filename = upload.name

                    with st.spinner("Analyzing…"):
                        routed = get_pipeline().analyze(
                            text=text,
                            media_path=media_path,
                            mime_type=mime_type,
                            filename=filename,
                            user_id="DEMO-USER",
                        )

                    if routed.status == CapabilityStatus.OK and routed.detected_input == InputType.TEXT:
                        st.session_state.last_result = {
                            "status": CapabilityStatus.OK,
                            "routed": routed,
                        }
                    elif routed.status == CapabilityStatus.NOT_IMPLEMENTED:
                        st.session_state.last_result = {
                            "status": CapabilityStatus.NOT_IMPLEMENTED,
                            "detected_input": routed.detected_input,
                            "message": routed.message,
                        }
                    else:
                        st.session_state.last_result = {
                            "status": CapabilityStatus.VALIDATION_ERROR,
                            "message": routed.message or "Invalid input.",
                        }
                except Exception as exc:  # noqa: BLE001
                    st.session_state.last_result = {
                        "status": "error",
                        "message": f"Analysis failed: {exc}",
                    }
                finally:
                    if tmp_root is not None:
                        _cleanup_temp_dir(tmp_root)

        with right:
            if st.session_state.last_result is None:
                _render_idle_preview()
            else:
                _render_status_panel(st.session_state.last_result)

    st.markdown(
        """
        <div class="app-footer">
          MyUni Sentiment Intelligence <span class="beta-pill">Beta</span><br>
          © 2026 MyUni. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True,
    )


# Streamlit re-executes this file on each interaction; do not guard with
# ``if __name__ == "__main__"`` — that block is skipped under Streamlit.
main()
