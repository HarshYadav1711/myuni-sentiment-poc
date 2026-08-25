"""Automatic input-type detection and capability-aware routing.

The Streamlit client uses this path so modality selection is never manual.
Existing CLI / batch ``analyze_activity`` paths are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from src.config import DEFAULT_TEXT_MODEL


class InputType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


class CapabilityStatus(str, Enum):
    OK = "ok"
    NOT_IMPLEMENTED = "not_implemented"
    VALIDATION_ERROR = "validation_error"


# Client text-baseline: only Twitter-RoBERTa text inference is enabled.
# Image/video analyzer modules may exist in the repo for future wiring, but
# they are not exposed through the unified client path until enabled here.
CLIENT_ENABLED_MODALITIES: frozenset[InputType] = frozenset({InputType.TEXT})

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".webm", ".mkv"})

_IMAGE_MIME_PREFIXES = ("image/",)
_VIDEO_MIME_PREFIXES = ("video/",)

_IMAGE_MIME_EXACT = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/x-ms-bmp",
    },
)
_VIDEO_MIME_EXACT = frozenset(
    {
        "video/mp4",
        "video/quicktime",
        "video/x-msvideo",
        "video/webm",
        "video/x-matroska",
        "video/avi",
        "application/mp4",
    },
)

_NOT_IMPLEMENTED_MESSAGES = {
    InputType.IMAGE: (
        "Image detected and routed successfully. "
        "Visual sentiment analysis is not enabled in the current text-baseline build."
    ),
    InputType.VIDEO: (
        "Video detected and routed successfully. "
        "Video sentiment analysis is not enabled in the current text-baseline build."
    ),
}


@dataclass(frozen=True)
class DetectedInput:
    """Normalized detection result before capability checks."""

    input_type: InputType
    text: Optional[str] = None
    media_path: Optional[str] = None
    mime_type: Optional[str] = None
    filename: Optional[str] = None


@dataclass
class RoutedAnalysisResult:
    """Unified client analyze response (implemented or capability-gated)."""

    status: CapabilityStatus
    detected_input: Optional[InputType]
    message: Optional[str] = None
    analysis: object = None  # ActivityAnalysisResult when status == ok
    model_display_name: Optional[str] = None
    model_id: Optional[str] = None


class InputRouter:
    """Detect modality from text and/or uploaded media signals."""

    @staticmethod
    def _normalize_mime(mime_type: Optional[str]) -> Optional[str]:
        if mime_type is None:
            return None
        cleaned = mime_type.strip().lower()
        return cleaned or None

    @staticmethod
    def _extension(filename: Optional[str], media_path: Optional[str]) -> Optional[str]:
        for candidate in (filename, media_path):
            if not candidate:
                continue
            suffix = Path(candidate).suffix.lower()
            if suffix:
                return suffix
        return None

    @classmethod
    def classify_media(
        cls,
        *,
        mime_type: Optional[str] = None,
        filename: Optional[str] = None,
        media_path: Optional[str] = None,
    ) -> Optional[InputType]:
        """Return IMAGE/VIDEO from MIME (primary) and extension (secondary), or None."""
        mime = cls._normalize_mime(mime_type)
        ext = cls._extension(filename, media_path)

        mime_kind: Optional[InputType] = None
        if mime:
            if mime in _IMAGE_MIME_EXACT or mime.startswith(_IMAGE_MIME_PREFIXES):
                mime_kind = InputType.IMAGE
            elif mime in _VIDEO_MIME_EXACT or mime.startswith(_VIDEO_MIME_PREFIXES):
                mime_kind = InputType.VIDEO

        ext_kind: Optional[InputType] = None
        if ext in _IMAGE_EXTENSIONS:
            ext_kind = InputType.IMAGE
        elif ext in _VIDEO_EXTENSIONS:
            ext_kind = InputType.VIDEO

        if mime_kind and ext_kind and mime_kind != ext_kind:
            raise ValueError(
                f"File type mismatch: MIME suggests {mime_kind.value}, "
                f"but extension suggests {ext_kind.value}. "
                "Please upload a supported image or video file.",
            )

        return mime_kind or ext_kind

    @classmethod
    def detect(
        cls,
        *,
        text: Optional[str] = None,
        media_path: Optional[str] = None,
        mime_type: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> DetectedInput:
        """Choose a single active input.

        Priority: non-blank text wins (media ignored). Otherwise media is required.
        """
        cleaned_text = text.strip() if isinstance(text, str) else None
        if cleaned_text:
            return DetectedInput(input_type=InputType.TEXT, text=cleaned_text)

        has_media = bool(media_path) or bool(filename) or bool(mime_type)
        if not has_media:
            raise ValueError(
                "Enter text or upload a supported media file to analyze.",
            )

        kind = cls.classify_media(
            mime_type=mime_type,
            filename=filename,
            media_path=media_path,
        )
        if kind is None:
            label = filename or media_path or mime_type or "upload"
            raise ValueError(
                f"Unsupported file type ({label}). "
                "Supported images: JPG, JPEG, PNG, WEBP. "
                "Supported videos: MP4, MOV, AVI, WEBM, MKV.",
            )

        return DetectedInput(
            input_type=kind,
            media_path=media_path,
            mime_type=cls._normalize_mime(mime_type),
            filename=filename,
        )

    @classmethod
    def is_enabled(
        cls,
        input_type: InputType,
        enabled: Optional[frozenset[InputType]] = None,
    ) -> bool:
        active = CLIENT_ENABLED_MODALITIES if enabled is None else enabled
        return input_type in active

    @classmethod
    def not_implemented_message(cls, input_type: InputType) -> str:
        return _NOT_IMPLEMENTED_MESSAGES.get(
            input_type,
            f"{input_type.value.title()} analysis is not enabled in this build.",
        )

    @classmethod
    def text_model_meta(cls) -> tuple[str, str]:
        """Return (display name, exact checkpoint id) for client results."""
        return "Twitter-RoBERTa", DEFAULT_TEXT_MODEL
