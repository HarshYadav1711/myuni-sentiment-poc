"""Central MyUni sentiment analysis pipeline."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from src.analyzers.audio import AudioAnalyzer
from src.analyzers.image import ImageAnalyzer
from src.analyzers.text import TextSentimentAnalyzer
from src.analyzers.video import VideoAnalyzer
from src.config import DEFAULT_FUSION, DEFAULT_VIDEO_SAMPLING, FusionConfig, VideoSamplingConfig
from src.fusion import fuse_modalities
from src.schemas import (
    ActivityAnalysisResult,
    ActivityInput,
    AnalysisBlock,
    InputMetadata,
    ModalityBundle,
    SpeechAnalysisResult,
)

logger = logging.getLogger(__name__)

_PREVIEW_LEN = 120


class MyUniSentimentPipeline:
    """Routes activities through modality analyzers and returns standardized results.

    Milestone 5: ``text``, ``image``, and ``video`` activities are analyzed.
    """

    def __init__(
        self,
        text_analyzer: Optional[TextSentimentAnalyzer] = None,
        image_analyzer: Optional[ImageAnalyzer] = None,
        audio_analyzer: Optional[AudioAnalyzer] = None,
        video_analyzer: Optional[VideoAnalyzer] = None,
        fusion_config: FusionConfig = DEFAULT_FUSION,
        video_sampling: VideoSamplingConfig = DEFAULT_VIDEO_SAMPLING,
        video_sampling_strategy: str = "fixed_fps",
        video_debug: bool = False,
    ) -> None:
        self._text_analyzer = text_analyzer or TextSentimentAnalyzer()
        self._image_analyzer = image_analyzer or ImageAnalyzer(
            text_analyzer=self._text_analyzer,
        )
        self._image_analyzer.set_text_analyzer(self._text_analyzer)
        self._audio_analyzer = audio_analyzer or AudioAnalyzer(
            text_analyzer=self._text_analyzer,
        )
        self._audio_analyzer.set_text_analyzer(self._text_analyzer)
        self._fusion_config = fusion_config
        self._video_analyzer = video_analyzer or VideoAnalyzer(
            image_analyzer=self._image_analyzer,
            audio_analyzer=self._audio_analyzer,
            text_analyzer=self._text_analyzer,
            sampling=video_sampling,
            sampling_strategy=video_sampling_strategy,
            fusion_config=fusion_config,
            debug=video_debug,
        )
        self._video_analyzer.set_text_analyzer(self._text_analyzer)

    @property
    def text_analyzer(self) -> TextSentimentAnalyzer:
        return self._text_analyzer

    @property
    def image_analyzer(self) -> ImageAnalyzer:
        return self._image_analyzer

    @property
    def audio_analyzer(self) -> AudioAnalyzer:
        return self._audio_analyzer

    @property
    def video_analyzer(self) -> VideoAnalyzer:
        return self._video_analyzer

    def analyze_speech(self, media_path: object) -> SpeechAnalysisResult:
        """Run the speech branch on an audio/video media path."""
        return self._audio_analyzer.analyze(media_path)  # type: ignore[arg-type]

    def analyze_text(
        self,
        text: object,
        *,
        user_id: Optional[str] = None,
        activity_id: Optional[str] = None,
    ) -> ActivityAnalysisResult:
        """Analyze a free-form text activity (CLI convenience path)."""
        cleaned = self._text_analyzer.validate_text(text)
        resolved_activity_id = activity_id or f"ACT-{uuid.uuid4().hex[:8].upper()}"
        return self._analyze_text_content(
            cleaned,
            activity_id=resolved_activity_id,
            user_id=user_id,
            created_at=None,
            content_kind=None,
            extra=None,
            media_path=None,
        )

    def analyze_activity(self, activity: ActivityInput) -> ActivityAnalysisResult:
        """Analyze a validated ActivityInput (text, image, or video)."""
        if activity.activity_type == "text":
            assert activity.text is not None
            cleaned = self._text_analyzer.validate_text(activity.text)
            return self._analyze_text_content(
                cleaned,
                activity_id=activity.activity_id,
                user_id=activity.user_id,
                created_at=activity.created_at,
                content_kind=activity.content_kind,
                extra=activity.metadata,
                media_path=activity.media_path,
            )

        if activity.activity_type == "image":
            return self._analyze_image_activity(activity)

        if activity.activity_type == "video":
            return self._analyze_video_activity(activity)

        raise NotImplementedError(
            f"activity_type={activity.activity_type!r} is not implemented yet",
        )

    def _analyze_text_content(
        self,
        cleaned: str,
        *,
        activity_id: str,
        user_id: Optional[str],
        created_at: object,
        content_kind: object,
        extra: object,
        media_path: Optional[str],
    ) -> ActivityAnalysisResult:
        evidence = self._text_analyzer.analyze(cleaned)
        preview = cleaned if len(cleaned) <= _PREVIEW_LEN else f"{cleaned[:_PREVIEW_LEN]}..."
        fusion = fuse_modalities({"text": evidence}, config=self._fusion_config)

        logger.info(
            "Analyzed text activity_id=%s user_id=%s label=%s score=%.3f",
            activity_id,
            user_id,
            fusion.overall.label,
            fusion.overall.score,
        )

        return ActivityAnalysisResult(
            activity_id=activity_id,
            user_id=user_id,
            activity_type="text",
            input=InputMetadata(
                text_length=len(cleaned),
                text_preview=preview,
                media_path=media_path,
                created_at=created_at,  # type: ignore[arg-type]
                content_kind=content_kind,  # type: ignore[arg-type]
                extra=extra,  # type: ignore[arg-type]
            ),
            analysis=AnalysisBlock(
                overall=fusion.overall,
                modalities=ModalityBundle(text=evidence),
                fusion=fusion.diagnostics,
            ),
        )

    def _analyze_image_activity(self, activity: ActivityInput) -> ActivityAnalysisResult:
        assert activity.media_path is not None
        media_path = activity.media_path

        caption_evidence = None
        caption_preview = None
        caption_length = None
        if activity.text:
            cleaned_caption = self._text_analyzer.validate_text(activity.text)
            caption_evidence = self._text_analyzer.analyze(cleaned_caption)
            caption_evidence = caption_evidence.model_copy(
                update={
                    "details": {
                        **(caption_evidence.details or {}),
                        "source": "caption",
                    },
                },
            )
            caption_length = len(cleaned_caption)
            caption_preview = (
                cleaned_caption
                if len(cleaned_caption) <= _PREVIEW_LEN
                else f"{cleaned_caption[:_PREVIEW_LEN]}..."
            )

        image_evidence = self._image_analyzer.analyze_path(media_path)

        modalities = ModalityBundle(
            text=caption_evidence,
            visual=image_evidence.visual,
            ocr=image_evidence.ocr_sentiment,
        )
        fusion = fuse_modalities(
            {
                "text": caption_evidence,
                "visual": image_evidence.visual,
                "ocr": image_evidence.ocr_sentiment,
            },
            config=self._fusion_config,
        )

        logger.info(
            "Analyzed image activity_id=%s user_id=%s overall=%s score=%.3f warnings=%s",
            activity.activity_id,
            activity.user_id,
            fusion.overall.label,
            fusion.overall.score,
            len(image_evidence.warnings),
        )

        return ActivityAnalysisResult(
            activity_id=activity.activity_id,
            user_id=activity.user_id,
            activity_type="image",
            input=InputMetadata(
                text_length=caption_length,
                text_preview=caption_preview,
                media_path=media_path,
                created_at=activity.created_at,
                content_kind=activity.content_kind,
                extra=activity.metadata,
            ),
            analysis=AnalysisBlock(
                overall=fusion.overall,
                modalities=modalities,
                fusion=fusion.diagnostics,
                warnings=list(image_evidence.warnings),
                ocr_text=image_evidence.ocr_text,
            ),
        )

    def _analyze_video_activity(self, activity: ActivityInput) -> ActivityAnalysisResult:
        assert activity.media_path is not None
        media_path = activity.media_path

        caption_evidence = None
        caption_preview = None
        caption_length = None
        if activity.text:
            cleaned_caption = self._text_analyzer.validate_text(activity.text)
            caption_evidence = self._text_analyzer.analyze(cleaned_caption)
            caption_evidence = caption_evidence.model_copy(
                update={
                    "details": {
                        **(caption_evidence.details or {}),
                        "source": "caption",
                    },
                },
            )
            caption_length = len(cleaned_caption)
            caption_preview = (
                cleaned_caption
                if len(cleaned_caption) <= _PREVIEW_LEN
                else f"{cleaned_caption[:_PREVIEW_LEN]}..."
            )

        bundle = self._video_analyzer.analyze(
            media_path,
            caption_sentiment=caption_evidence,
        )

        # Re-fuse so caption is always included even if VideoAnalyzer overall was computed.
        fusion = fuse_modalities(
            {
                "text": caption_evidence,
                "visual": bundle.visual,
                "ocr": bundle.ocr,
                "speech": bundle.speech,
            },
            config=self._fusion_config,
        )

        logger.info(
            "Analyzed video activity_id=%s user_id=%s overall=%s score=%.3f warnings=%s",
            activity.activity_id,
            activity.user_id,
            fusion.overall.label,
            fusion.overall.score,
            len(bundle.warnings),
        )

        return ActivityAnalysisResult(
            activity_id=activity.activity_id,
            user_id=activity.user_id,
            activity_type="video",
            input=InputMetadata(
                text_length=caption_length,
                text_preview=caption_preview,
                media_path=media_path,
                created_at=activity.created_at,
                content_kind=activity.content_kind,
                extra=activity.metadata,
            ),
            analysis=AnalysisBlock(
                overall=fusion.overall,
                modalities=ModalityBundle(
                    text=caption_evidence,
                    visual=bundle.visual,
                    ocr=bundle.ocr,
                    speech=bundle.speech,
                ),
                fusion=fusion.diagnostics,
                warnings=list(bundle.warnings),
                ocr_text=bundle.ocr_text,
                transcript=bundle.transcript,
                video=bundle.diagnostics,
            ),
        )
