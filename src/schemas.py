"""Pydantic schemas for MyUni sentiment analysis POC inputs and outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SentimentLabel = Literal["positive", "neutral", "negative"]
ActivityType = Literal["text", "image", "video", "audio"]

# Reserved for future MyUni semantics (post / comment / story). Not enforced yet.
ContentKind = Literal["post", "comment", "story", "caption", "other"]

BatchRecordStatus = Literal["processed", "invalid", "unsupported", "failed", "skipped"]


# ---------------------------------------------------------------------------
# Activity input contract (batch / daily workflow)
# ---------------------------------------------------------------------------


class ActivityInput(BaseModel):
    """Validated MyUni activity record for ingestion.

    Batch validation policy (Milestone 2):
    - ``activity_id`` and ``user_id`` are required non-blank strings
    - ``text`` activities require usable (non-blank) ``text``
    - ``image`` / ``video`` activities require ``media_path``; caption ``text`` is optional
    - ``content_kind`` is optional reserved extensibility (post/comment/story), unused by MVP logic
    """

    activity_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    activity_type: ActivityType
    text: Optional[str] = None
    media_path: Optional[str] = None
    created_at: datetime
    metadata: Optional[dict[str, Any]] = None
    content_kind: Optional[ContentKind] = Field(
        default=None,
        description="Optional future semantic kind (post/comment/story); ignored by MVP routing.",
    )

    @field_validator("activity_id", "user_id", mode="before")
    @classmethod
    def _strip_required_ids(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("media_path", mode="before")
    @classmethod
    def _strip_optional_path(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("text", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @model_validator(mode="after")
    def _enforce_modality_rules(self) -> ActivityInput:
        if self.activity_type == "text":
            if not self.text:
                raise ValueError("text activities require non-blank text")
        elif self.activity_type in ("image", "video", "audio"):
            if not self.media_path:
                raise ValueError(
                    f"{self.activity_type} activities require media_path",
                )
        return self


# ---------------------------------------------------------------------------
# Analysis output (Milestone 1+)
# ---------------------------------------------------------------------------


class SentimentEvidence(BaseModel):
    """Single-modality (or overall) sentiment evidence."""

    label: SentimentLabel
    score: float = Field(
        ...,
        description="POC sentiment score approximately in [-1, +1].",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Optional[dict[str, float]] = None
    model: Optional[str] = None
    details: Optional[dict[str, Any]] = None


class FusionDiagnostics(BaseModel):
    """Deterministic explainable diagnostics for late fusion (no LLM)."""

    contributing_modalities: list[str] = Field(default_factory=list)
    configured_weights: dict[str, float] = Field(default_factory=dict)
    effective_weights: dict[str, float] = Field(default_factory=dict)
    modality_conflict: bool = False
    disagreement_score: float = 0.0
    thresholds: dict[str, float] = Field(default_factory=dict)
    explanation: str = ""
    note: str = "POC evaluation defaults only; not client scoring rules."
    source_path: Optional[str] = None


class ModalityBundle(BaseModel):
    """Per-modality evidence. Unused modalities are omitted (not null-filled)."""

    text: Optional[SentimentEvidence] = None  # caption / primary text
    visual: Optional[SentimentEvidence] = None
    ocr: Optional[SentimentEvidence] = None
    speech: Optional[SentimentEvidence] = None


class SpeechSegment(BaseModel):
    """Timed ASR segment from faster-whisper."""

    start: float
    end: float
    text: str


class SpeechWord(BaseModel):
    """Word-level timing from Faster-Whisper ``word_timestamps=True``.

    Used for temporal-window speech assignment. Not a replacement for
    top-level segments or the global full-transcript sentiment path.
    """

    start: float
    end: float
    text: str
    word_id: Optional[str] = Field(
        default=None,
        description="Stable internal id, e.g. word-0000.",
    )


SpeechAlignmentSource = Literal["word_timestamps", "segment_fallback"]


class SpeechAnalysisResult(BaseModel):
    """Structured speech-branch output (Milestone 4). Not full video fusion."""

    transcript: Optional[str] = None
    language: Optional[str] = None
    segments: list[SpeechSegment] = Field(default_factory=list)
    words: list[SpeechWord] = Field(
        default_factory=list,
        description=(
            "Optional word timings from one Whisper pass. Empty when "
            "word_timestamps were not requested or unavailable."
        ),
    )
    transcription_seconds: Optional[float] = None
    audio_duration_seconds: Optional[float] = None
    sentiment: Optional[SentimentEvidence] = None
    asr_model: str
    warnings: list[str] = Field(default_factory=list)
    details: Optional[dict[str, Any]] = None

    def model_dump_json_compatible(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class PocRuntimeInfo(BaseModel):
    """Configured models and sampling settings visible in analysis output."""

    models: dict[str, str] = Field(default_factory=dict)
    video_sampling: Optional[dict[str, Any]] = None
    fusion_source: Optional[str] = None
    note: str = "POC evaluation defaults only; not client scoring rules."


class AnalysisBlock(BaseModel):
    overall: SentimentEvidence
    modalities: ModalityBundle
    fusion: Optional[FusionDiagnostics] = None
    runtime: Optional[PocRuntimeInfo] = None
    warnings: list[str] = Field(default_factory=list)
    ocr_text: Optional[str] = Field(
        default=None,
        description="Normalized OCR string when extracted (may exist without ocr sentiment).",
    )
    transcript: Optional[str] = Field(
        default=None,
        description="ASR transcript when available (speech / video).",
    )
    video: Optional[VideoDiagnostics] = Field(
        default=None,
        description="Compact video sampling / processing diagnostics.",
    )
    temporal_context: Optional[TemporalContext] = Field(
        default=None,
        description="Parallel temporal timeline/features for VIDEO only (Phase 1).",
    )
    deterministic_context: Optional[DeterministicTemporalContext] = Field(
        default=None,
        description="Authoritative deterministic temporal context for VIDEO.",
    )
    temporal_reasoning: Optional[TemporalReasoningResult] = Field(
        default=None,
        description="Optional advisory LLM contextual interpretation for VIDEO.",
    )
    temporal_reasoner_diagnostics: Optional[TemporalReasonerDiagnostics] = Field(
        default=None,
        description="Non-client diagnostics for temporal reasoner execution.",
    )


class VideoFrameDebug(BaseModel):
    """Optional per-frame debug row (omitted from default JSON)."""

    index: int
    timestamp_seconds: Optional[float] = None
    visual_label: Optional[SentimentLabel] = None
    visual_score: Optional[float] = None
    visual_confidence: Optional[float] = None
    ocr_preview: Optional[str] = None
    error: Optional[str] = None


class VideoDiagnostics(BaseModel):
    """Compact video-level processing diagnostics (default output stays small)."""

    duration_seconds: Optional[float] = None
    sampling_strategy: str = "fixed_fps"
    sampling_fps: Optional[float] = None
    frames_extracted: int = 0
    frames_analyzed: int = 0
    frame_timestamps: list[float] = Field(default_factory=list)
    extraction_seconds: Optional[float] = None
    processing_seconds: Optional[float] = None
    has_audio: Optional[bool] = None
    scene_count: Optional[int] = None
    frame_debug: Optional[list[VideoFrameDebug]] = Field(
        default=None,
        description="Present only when debug=True on the video analyzer.",
    )


# ---------------------------------------------------------------------------
# Temporal context (Phase 1) — structured timeline over existing evidence
# ---------------------------------------------------------------------------


TrajectoryLabel = Literal[
    "stable_positive",
    "stable_neutral",
    "stable_negative",
    "increasing_negative",
    "decreasing_negative",
    "mixed",
    "insufficient_evidence",
]

AgreementLevel = Literal["high", "moderate", "low", "insufficient_evidence"]


class TemporalOcrEvidence(BaseModel):
    """OCR evidence inherited from a parent frame timestamp."""

    text: Optional[str] = None
    sentiment: Optional[SentimentEvidence] = None


class TemporalSpeechEvidence(BaseModel):
    """Speech evidence aligned by Faster-Whisper segment timestamps."""

    text: Optional[str] = None
    sentiment: Optional[SentimentEvidence] = None
    segment_start: Optional[float] = None
    segment_end: Optional[float] = None


class TemporalEvent(BaseModel):
    """Point-in-time multimodal evidence on the video timeline.

    Events are primarily frame-anchored. Missing modalities stay None —
    never coerced to neutral.
    """

    timestamp: float
    event_id: str
    visual: Optional[SentimentEvidence] = None
    ocr: Optional[TemporalOcrEvidence] = None
    speech: Optional[TemporalSpeechEvidence] = None


class TemporalWindow(BaseModel):
    """Fixed-duration temporal bucket preserving raw + aggregated evidence."""

    start: float
    end: float
    index: int
    visual_evidence: list[SentimentEvidence] = Field(default_factory=list)
    speech_segments: list[SpeechSegment] = Field(default_factory=list)
    speech_sentiments: list[SentimentEvidence] = Field(default_factory=list)
    ocr_texts: list[str] = Field(default_factory=list)
    ocr_sentiments: list[SentimentEvidence] = Field(default_factory=list)
    visual_probabilities: Optional[dict[str, float]] = None
    speech_probabilities: Optional[dict[str, float]] = None
    ocr_probabilities: Optional[dict[str, float]] = None
    available_modalities: list[str] = Field(default_factory=list)
    usable: bool = False
    dominant_label: Optional[SentimentLabel] = None
    negative_probability: Optional[float] = None


class StrongestNegativeWindow(BaseModel):
    """Window with the highest meaningful negative probability."""

    start: float
    end: float
    score: float
    index: int


class SuddenNegativeChange(BaseModel):
    """Significant negative-probability jump between consecutive usable windows."""

    detected: bool
    from_window: Optional[int] = None
    to_window: Optional[int] = None
    from_start: Optional[float] = None
    to_start: Optional[float] = None
    delta: Optional[float] = None


class CrossModalConflict(BaseModel):
    """Preserved contradiction between modalities in one window (not averaged away)."""

    window_index: int
    window_start: float
    window_end: float
    modalities: list[str]
    labels: dict[str, SentimentLabel]
    scores: dict[str, float]
    probabilities: dict[str, dict[str, float]] = Field(default_factory=dict)


class EvidenceCoverage(BaseModel):
    """Fraction of timeline windows containing each modality / usable evidence."""

    total_windows: int = 0
    usable_windows: int = 0
    visual_coverage: float = 0.0
    speech_coverage: float = 0.0
    ocr_coverage: float = 0.0
    overall_usable_coverage: float = 0.0


class TemporalFeatures(BaseModel):
    """Deterministic temporal features extracted from windowed evidence."""

    trajectory: TrajectoryLabel = "insufficient_evidence"
    negative_persistence: Optional[float] = Field(
        default=None,
        description="Fraction of usable windows that are meaningfully negative (0..1).",
    )
    longest_negative_run: Optional[int] = None
    longest_negative_run_seconds: Optional[float] = None
    strongest_negative_window: Optional[StrongestNegativeWindow] = None
    sudden_negative_change: SuddenNegativeChange = Field(
        default_factory=lambda: SuddenNegativeChange(detected=False),
    )
    cross_modal_agreement: AgreementLevel = "insufficient_evidence"
    cross_modal_conflicts: list[CrossModalConflict] = Field(default_factory=list)
    evidence_coverage: EvidenceCoverage = Field(default_factory=EvidenceCoverage)


class TemporalContext(BaseModel):
    """Parallel temporal-context payload attached to video analysis results."""

    window_seconds: float
    duration_seconds: Optional[float] = None
    events: list[TemporalEvent] = Field(default_factory=list)
    events_truncated: bool = False
    events_total: int = 0
    windows: list[TemporalWindow] = Field(default_factory=list)
    features: TemporalFeatures = Field(default_factory=TemporalFeatures)
    speech_alignment_source: Optional[SpeechAlignmentSource] = Field(
        default=None,
        description=(
            "How window speech text/sentiment was derived: "
            "word_timestamps (preferred) or segment_fallback."
        ),
    )
    speech_word_count: Optional[int] = Field(
        default=None,
        description="Count of usable Whisper word timings when word alignment ran.",
    )
    speech_words: list[SpeechWord] = Field(
        default_factory=list,
        description=(
            "Usable word timings used for temporal speech assignment "
            "(empty when segment_fallback or no speech)."
        ),
    )
    source_speech_segments: list[SpeechSegment] = Field(
        default_factory=list,
        description=(
            "Original Faster-Whisper top-level segments from the same ASR pass "
            "(retained for diagnostics; window speech may use word-local units)."
        ),
    )
    note: str = (
        "POC temporal context over existing modality evidence; "
        "CPU logic only; not a wellbeing / clinical score."
    )


# ---------------------------------------------------------------------------
# Temporal reasoning (Phase 2) — LLM contextual interpretation (not wellbeing)
# ---------------------------------------------------------------------------


ContextType = Literal[
    "personal_expression",
    "general_commentary",
    "humor_or_sarcasm",
    "quoted_or_reposted_content",
    "narrative_or_entertainment",
    "informational",
    "uncertain",
]

NegativeContentPattern = Literal[
    "isolated",
    "persistent",
    "increasing",
    "decreasing",
    "stable",
    "mixed",
    "unclear",
]

ReasonerStatus = Literal[
    "ok",
    "disabled",
    "reasoner_unavailable",
    "invalid_model_output",
]


class TemporalInterpretation(BaseModel):
    """How expressed content tone evolves across the observed timeline."""
    model_config = ConfigDict(extra="forbid")

    trajectory: str = Field(
        ...,
        description="Free-text or deterministic trajectory label restated by the reasoner.",
    )
    persistence: str = Field(
        ...,
        description="Whether negative expressed content is isolated/persistent/unclear.",
    )
    change_pattern: NegativeContentPattern = "unclear"


class CrossModalReasoningContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    consistency: AgreementLevel = "insufficient_evidence"
    conflicts_detected: bool = False
    description: str = ""


class ReasoningEvidenceReference(BaseModel):
    """LLM explanation anchored to a supplied deterministic evidence id."""
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    explanation: str


class ImportantTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: float
    end: float
    description: str
    evidence_ids: list[str] = Field(default_factory=list)


class TemporalReasoningResult(BaseModel):
    """Validated LLM contextual interpretation of structured temporal evidence.

    Not a wellbeing, clinical, or mental-health score. Confidence is model
    self-reported certainty about the *content interpretation* (0..1).
    """
    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    trajectory_explanation: str = ""
    cross_modal_context: Optional[CrossModalReasoningContext] = None
    important_transitions: list[ImportantTransition] = Field(default_factory=list)
    context_type: ContextType = "uncertain"
    evidence: list[ReasoningEvidenceReference] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    model: Optional[str] = None
    status: ReasonerStatus = "ok"
    details: Optional[dict[str, Any]] = None
    note: str = (
        "POC contextual interpretation of expressed content only; "
        "not a diagnosis, wellbeing score, or clinical risk assessment."
    )


class TemporalReasonerDiagnostics(BaseModel):
    """Non-client diagnostics for reasoner execution and validation."""

    prompt_construction_seconds: Optional[float] = None
    generation_seconds: Optional[float] = None
    parse_validation_seconds: Optional[float] = None
    total_reasoner_seconds: Optional[float] = None
    model_load_seconds: Optional[float] = None
    prompt_chars: Optional[int] = None
    prompt_windows_included: Optional[int] = None
    evidence_ids_supplied: list[str] = Field(default_factory=list)
    repair_attempted: bool = False
    raw_output_preview: Optional[str] = None
    generation_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description="Actual generate() kwargs used (sampling knobs, seed, etc.).",
    )
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None
    sampling_warning_detected: bool = Field(
        default=False,
        description="True if Transformers warned that sampling flags were ignored.",
    )


class DeterministicTemporalContext(BaseModel):
    """Authoritative temporal context from deterministic code only."""

    context: TemporalContext
    note: str = (
        "Authoritative deterministic temporal context. These facts are produced "
        "by code and must not be overridden by the LLM reasoner."
    )


class InputMetadata(BaseModel):
    text_length: Optional[int] = None
    text_preview: Optional[str] = None
    media_path: Optional[str] = None
    created_at: Optional[datetime] = None
    content_kind: Optional[ContentKind] = None
    extra: Optional[dict[str, Any]] = None


class ActivityAnalysisResult(BaseModel):
    """Standardized activity-level sentiment result."""

    activity_id: str
    user_id: Optional[str] = None
    activity_type: ActivityType
    input: InputMetadata
    analysis: AnalysisBlock
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def model_dump_json_compatible(self) -> dict[str, Any]:
        """Serialize with ISO timestamps for CLI / file output."""
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Batch processing outcomes
# ---------------------------------------------------------------------------


class BatchRecordOutcome(BaseModel):
    """Per-record result for JSONL batch ingestion."""

    line_number: int
    status: BatchRecordStatus
    activity_id: Optional[str] = None
    user_id: Optional[str] = None
    activity_type: Optional[str] = None
    error: Optional[str] = None
    note: Optional[str] = None
    result: Optional[ActivityAnalysisResult] = None

    def model_dump_json_compatible(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class BatchSummary(BaseModel):
    """Aggregate metrics for a batch run."""

    total: int = 0
    valid: int = 0
    invalid: int = 0
    processed: int = 0
    unsupported: int = 0
    failed: int = 0
    skipped: int = 0


class BatchProcessingResult(BaseModel):
    """Full batch response: summary + per-record outcomes."""

    source: str
    summary: BatchSummary
    records: list[BatchRecordOutcome]
    batch_id: Optional[str] = None

    def model_dump_json_compatible(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class DailyUserScore(BaseModel):
    """POC daily user-level sentiment aggregate — NOT the client business score."""

    user_id: str
    score_date: str
    activity_count: int = 0
    valid_analysis_count: int = 0
    mean_sentiment_score: Optional[float] = None
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0
    daily_sentiment_label: Optional[SentimentLabel] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    note: str = (
        "POC daily aggregate — mean of stored activity sentiment scores; "
        "NOT the future client business score"
    )

    def model_dump_json_compatible(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
