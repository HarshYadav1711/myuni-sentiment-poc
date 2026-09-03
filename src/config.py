"""POC configuration for models, OCR, video sampling, and fusion weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


# Text sentiment (RoBERTa).
DEFAULT_TEXT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Exact Hugging Face checkpoint used for zero-shot visual concept scoring.
DEFAULT_VISUAL_MODEL = "google/siglip2-base-patch16-224"

# Zero-shot candidate prompts → sentiment labels (transparent, not a trained classifier).
DEFAULT_VISUAL_PROMPTS: dict[str, str] = {
    "positive": "a positive, joyful, pleasant or uplifting social media image",
    "neutral": "a neutral, ordinary or informational social media image",
    "negative": "a negative, sad, upsetting, unpleasant or distressing social media image",
}

# Facial-expression concept prompts (SigLIP zero-shot on face crops / face-gated frames).
DEFAULT_FACIAL_EXPRESSION_PROMPTS: dict[str, str] = {
    "positive": "a person or character with a happy smiling facial expression",
    "neutral": "a person or character with a calm neutral facial expression",
    "negative": "a person or character with a sad angry or upset facial expression",
}

# Face presence gate (positive≈face, negative≈no face, neutral≈ambiguous).
DEFAULT_FACE_GATE_PROMPTS: dict[str, str] = {
    "positive": "a close-up of a human or character face showing a facial expression",
    "neutral": "an ambiguous image that may or may not include a face",
    "negative": "a scene object or landscape with no visible face",
}

# Minimum SigLIP "face" probability to accept a Haar-miss full-frame expression score.
FACE_GATE_MIN_PROBABILITY = 0.45

# Minimum alphanumeric characters before OCR text is treated as meaningful.
OCR_MIN_ALNUM_CHARS = 3

# faster-whisper model size. base.en is CPU-friendly on ~16 GB RAM.
DEFAULT_WHISPER_MODEL = "base.en"
DEFAULT_WHISPER_COMPUTE_TYPE = "int8"
DEFAULT_ASR_LANGUAGE = "en"

# Video frame sampling. Baseline remains fixed ~1 FPS; scene_keyframe is optional.
DEFAULT_VIDEO_SAMPLE_FPS = 1.0
DEFAULT_VIDEO_MAX_FRAMES = 12
DEFAULT_VIDEO_MAX_OCR_FRAMES = 8
DEFAULT_VIDEO_SAMPLING_STRATEGY = "fixed_fps"

# Aliases matching the client multimodal milestone naming.
VIDEO_SAMPLE_FPS = DEFAULT_VIDEO_SAMPLE_FPS
MAX_VIDEO_FRAMES = DEFAULT_VIDEO_MAX_FRAMES

# ---------------------------------------------------------------------------
# Temporal context (Phase 1) — CPU-only structure over existing modality evidence.
# ---------------------------------------------------------------------------

# Fixed window length in seconds for temporal bucketing.
TEMPORAL_WINDOW_SECONDS = 5.0

# A window (or modality) is "meaningfully negative" when P(negative) >= this.
TEMPORAL_NEGATIVE_PROB_THRESHOLD = 0.45

# A window is "meaningfully positive" when P(positive) >= this.
TEMPORAL_POSITIVE_PROB_THRESHOLD = 0.45

# Minimum max(P(label)) across aggregated evidence for a window to be "usable".
TEMPORAL_MIN_USABLE_EVIDENCE = 0.30

# Sudden negative change: ΔP(negative) between consecutive usable windows.
TEMPORAL_SUDDEN_NEGATIVE_DELTA = 0.25

# Cross-modal conflict: each conflicting modality must have |score| >= this
# and confidence >= TEMPORAL_CROSS_MODAL_MIN_CONFIDENCE.
TEMPORAL_CROSS_MODAL_MIN_POLARITY = 0.35
TEMPORAL_CROSS_MODAL_MIN_CONFIDENCE = 0.40

# Trajectory: OLS slope of P(negative) vs *normalized* usable-timeline position
# in [0, 1] (first usable center → 0, last → 1). Slope ≈ ΔP(neg) over the
# observed usable timeline; |slope| below this → treated as flat (stable_*).
TEMPORAL_TRAJECTORY_SLOPE_THRESHOLD = 0.05

# Cap detailed events in default JSON (full list retained in technical details
# when include_all_events=True on the builder).
TEMPORAL_MAX_EVENTS_IN_OUTPUT = 48

# ---------------------------------------------------------------------------
# Temporal context reasoner (Phase 2) — text LLM over structured temporal evidence.
# ---------------------------------------------------------------------------

TEMPORAL_REASONER_ENABLED = True
TEMPORAL_REASONER_MODEL = "Qwen/Qwen3-1.7B"
# Explicit device only — never inferred from torch.cuda.is_available() (ZeroGPU).
TEMPORAL_REASONER_DEVICE = "cpu"
TEMPORAL_REASONER_MAX_NEW_TOKENS = 768
TEMPORAL_REASONER_TEMPERATURE = 0.0
TEMPORAL_REASONER_TOP_P = 1.0
TEMPORAL_REASONER_TOP_K = 0
TEMPORAL_REASONER_SEED = 0
TEMPORAL_REASONER_MAX_WINDOWS = 12
TEMPORAL_REASONER_MAX_EVIDENCE_ITEMS = 16
TEMPORAL_REASONER_MAX_RETRIES = 1
# Disable Qwen3 extended thinking for deterministic structured JSON POC.
TEMPORAL_REASONER_ENABLE_THINKING = False

# Evaluation-friendly Qwen non-thinking decoding profile for later comparisons.
TEMPORAL_REASONER_EVAL_TEMPERATURE = 0.7
TEMPORAL_REASONER_EVAL_TOP_P = 0.8
TEMPORAL_REASONER_EVAL_TOP_K = 20
TEMPORAL_REASONER_EVAL_SEED = 42
TEMPORAL_REASONER_EVAL_DO_SAMPLE = True

# Candidate model IDs for Phase 3B-A comparison (4B must not be downloaded locally).
TEMPORAL_REASONER_CANDIDATE_1_7B = "Qwen/Qwen3-1.7B"
TEMPORAL_REASONER_CANDIDATE_4B = "Qwen/Qwen3-4B-Instruct-2507"

# RoBERTa max sequence length; longer transcripts are chunked (not silently truncated).
TEXT_MAX_LENGTH = 512
TEXT_CHUNK_SIZE = 480
TEXT_CHUNK_STRIDE = 64

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FUSION_YAML = _REPO_ROOT / "config" / "fusion.yaml"


@dataclass(frozen=True)
class VideoSamplingConfig:
    """Configurable fixed-FPS frame sampling with pathological-count safeguards."""

    fps: float = DEFAULT_VIDEO_SAMPLE_FPS
    max_frames: int = DEFAULT_VIDEO_MAX_FRAMES
    max_ocr_frames: int = DEFAULT_VIDEO_MAX_OCR_FRAMES

    def effective_fps(self, duration_seconds: float) -> float:
        """Return FPS that keeps expected frame count within ``max_frames``."""
        if duration_seconds <= 0:
            return self.fps
        expected = duration_seconds * self.fps
        if expected <= self.max_frames:
            return self.fps
        return max(self.max_frames / duration_seconds, 1e-6)


@dataclass(frozen=True)
class TemporalConfig:
    """Deterministic temporal-context settings (CPU logic; no LLM).

    All thresholds are documented POC evaluation defaults — not clinical rules.
    """

    window_seconds: float = TEMPORAL_WINDOW_SECONDS
    negative_prob_threshold: float = TEMPORAL_NEGATIVE_PROB_THRESHOLD
    positive_prob_threshold: float = TEMPORAL_POSITIVE_PROB_THRESHOLD
    min_usable_evidence: float = TEMPORAL_MIN_USABLE_EVIDENCE
    sudden_negative_delta: float = TEMPORAL_SUDDEN_NEGATIVE_DELTA
    cross_modal_min_polarity: float = TEMPORAL_CROSS_MODAL_MIN_POLARITY
    cross_modal_min_confidence: float = TEMPORAL_CROSS_MODAL_MIN_CONFIDENCE
    trajectory_slope_threshold: float = TEMPORAL_TRAJECTORY_SLOPE_THRESHOLD
    max_events_in_output: int = TEMPORAL_MAX_EVENTS_IN_OUTPUT

    def __post_init__(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")


@dataclass(frozen=True)
class TemporalReasonerConfig:
    """Text LLM contextual reasoner settings (additive; never replaces fusion).

    Device must be explicit. Do not select CUDA from torch.cuda.is_available()
    because Hugging Face ZeroGPU deployments have special CUDA semantics.
    """

    enabled: bool = TEMPORAL_REASONER_ENABLED
    model_id: str = TEMPORAL_REASONER_MODEL
    device: str = TEMPORAL_REASONER_DEVICE
    max_new_tokens: int = TEMPORAL_REASONER_MAX_NEW_TOKENS
    temperature: float = TEMPORAL_REASONER_TEMPERATURE
    top_p: float = TEMPORAL_REASONER_TOP_P
    top_k: int = TEMPORAL_REASONER_TOP_K
    seed: int = TEMPORAL_REASONER_SEED
    max_windows: int = TEMPORAL_REASONER_MAX_WINDOWS
    max_evidence_items: int = TEMPORAL_REASONER_MAX_EVIDENCE_ITEMS
    max_retries: int = TEMPORAL_REASONER_MAX_RETRIES
    enable_thinking: bool = TEMPORAL_REASONER_ENABLE_THINKING
    # When None, do_sample is inferred from temperature/top_p/top_k.
    # Set explicitly True for evaluation sampling profiles so Transformers
    # does not ignore temperature/top_p/top_k under greedy decoding.
    do_sample: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if self.max_windows <= 0:
            raise ValueError("max_windows must be > 0")
        if self.max_evidence_items <= 0:
            raise ValueError("max_evidence_items must be > 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be in [0, 2]")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.do_sample is not None and not isinstance(self.do_sample, bool):
            raise ValueError("do_sample must be bool or None")


def evaluation_reasoner_config(
    model_id: str = TEMPORAL_REASONER_CANDIDATE_1_7B,
    *,
    device: str = TEMPORAL_REASONER_DEVICE,
    **overrides: Any,
) -> TemporalReasonerConfig:
    """Benchmark / evaluation decoding profile (explicit sampling).

    Uses fixed seed for reproducibility of *seeded sampling*. This does not
    make sampling deterministic across hardware/backends — seed is recorded.
    """
    base = dict(
        enabled=True,
        model_id=model_id,
        device=device,
        temperature=TEMPORAL_REASONER_EVAL_TEMPERATURE,
        top_p=TEMPORAL_REASONER_EVAL_TOP_P,
        top_k=TEMPORAL_REASONER_EVAL_TOP_K,
        seed=TEMPORAL_REASONER_EVAL_SEED,
        do_sample=TEMPORAL_REASONER_EVAL_DO_SAMPLE,
        enable_thinking=False,
        max_new_tokens=TEMPORAL_REASONER_MAX_NEW_TOKENS,
        max_retries=TEMPORAL_REASONER_MAX_RETRIES,
    )
    base.update(overrides)
    return TemporalReasonerConfig(**base)


@dataclass(frozen=True)
class FusionThresholds:
    """POC label thresholds on the fused score (not scientifically validated)."""

    positive_above: float = 0.15
    negative_below: float = -0.15


@dataclass(frozen=True)
class FusionConflictConfig:
    """POC conflict detection parameters."""

    min_confidence: float = 0.40
    min_polarity: float = 0.35
    disagreement_threshold: float = 0.90
    confidence_penalty: float = 0.50


@dataclass(frozen=True)
class FusionConfig:
    """POC-only late fusion settings loaded from ``config/fusion.yaml``.

    NOT the client business scoring methodology.
    """

    modality_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "text": 1.0,
            "visual": 1.0,
            "ocr": 0.8,
            "speech": 1.0,
        },
    )
    thresholds: FusionThresholds = field(default_factory=FusionThresholds)
    conflict: FusionConflictConfig = field(default_factory=FusionConflictConfig)
    note: str = "POC evaluation defaults only; not client scoring rules."
    source_path: Optional[str] = None

    @property
    def neutral_band(self) -> float:
        """Backward-compatible half-width around zero using symmetric thresholds."""
        return max(abs(self.thresholds.positive_above), abs(self.thresholds.negative_below))


def _as_float_map(raw: Any, fallback: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(raw, dict):
        return dict(fallback)
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out or dict(fallback)


def load_fusion_config(path: Optional[Path] = None) -> FusionConfig:
    """Load fusion settings from YAML; fall back to built-in POC defaults."""
    yaml_path = Path(path) if path is not None else DEFAULT_FUSION_YAML
    defaults = FusionConfig()

    if not yaml_path.is_file():
        return FusionConfig(source_path=str(yaml_path))

    try:
        import yaml
    except ImportError:
        return FusionConfig(
            note=defaults.note + " (PyYAML missing; using built-in defaults)",
            source_path=str(yaml_path),
        )

    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    weights = _as_float_map(data.get("modality_weights"), defaults.modality_weights)
    thr_raw = data.get("thresholds") or {}
    conflict_raw = data.get("conflict") or {}

    thresholds = FusionThresholds(
        positive_above=float(thr_raw.get("positive_above", defaults.thresholds.positive_above)),
        negative_below=float(thr_raw.get("negative_below", defaults.thresholds.negative_below)),
    )
    conflict = FusionConflictConfig(
        min_confidence=float(conflict_raw.get("min_confidence", defaults.conflict.min_confidence)),
        min_polarity=float(conflict_raw.get("min_polarity", defaults.conflict.min_polarity)),
        disagreement_threshold=float(
            conflict_raw.get("disagreement_threshold", defaults.conflict.disagreement_threshold),
        ),
        confidence_penalty=float(
            conflict_raw.get("confidence_penalty", defaults.conflict.confidence_penalty),
        ),
    )
    note = str(data.get("note") or defaults.note).strip()

    return FusionConfig(
        modality_weights=weights,
        thresholds=thresholds,
        conflict=conflict,
        note=note,
        source_path=str(yaml_path),
    )


DEFAULT_FUSION = load_fusion_config()
DEFAULT_VIDEO_SAMPLING = VideoSamplingConfig()
DEFAULT_TEMPORAL = TemporalConfig()
DEFAULT_TEMPORAL_REASONER = TemporalReasonerConfig()
