"""Video activity analysis: fixed-FPS frames + OCR subset + speech + caption fusion inputs."""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

from src.analyzers.audio import AudioAnalyzer
from src.analyzers.image import ImageAnalyzer
from src.analyzers.ocr import is_meaningful_ocr_text
from src.analyzers.text import TextSentimentAnalyzer
from src.config import DEFAULT_FUSION, DEFAULT_VIDEO_SAMPLING, FusionConfig, VideoSamplingConfig
from src.fusion import aggregate_frame_visual_scores, fuse_modalities
from src.media.ffmpeg_utils import (
    FFmpegError,
    FFmpegNotFoundError,
    extract_frames_at_fps,
    probe_video,
)
from src.schemas import (
    SentimentEvidence,
    SpeechAnalysisResult,
    VideoDiagnostics,
    VideoFrameDebug,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclass
class VideoAnalysisBundle:
    """Pre-activity multimodal evidence for one video (pipeline wraps into ActivityAnalysisResult)."""

    visual: Optional[SentimentEvidence]
    ocr: Optional[SentimentEvidence]
    ocr_text: Optional[str]
    speech: Optional[SentimentEvidence]
    transcript: Optional[str]
    speech_result: Optional[SpeechAnalysisResult]
    diagnostics: VideoDiagnostics
    warnings: list[str] = field(default_factory=list)
    overall: Optional[SentimentEvidence] = None  # filled when caption provided at pipeline


def _ocr_frame_indices(n_frames: int, max_ocr: int) -> set[int]:
    if n_frames <= 0 or max_ocr <= 0:
        return set()
    if n_frames <= max_ocr:
        return set(range(n_frames))
    # Evenly spaced indices including first and last when possible.
    positions = {
        int(round(i * (n_frames - 1) / (max_ocr - 1)))
        for i in range(max_ocr)
    }
    return positions


class VideoAnalyzer:
    """Fixed-FPS video analyzer composing existing image/visual/OCR and audio branches.

    Does not use a native video VLM or scene detection.
    """

    def __init__(
        self,
        *,
        image_analyzer: Optional[ImageAnalyzer] = None,
        audio_analyzer: Optional[AudioAnalyzer] = None,
        text_analyzer: Optional[TextSentimentAnalyzer] = None,
        sampling: VideoSamplingConfig = DEFAULT_VIDEO_SAMPLING,
        fusion_config: FusionConfig = DEFAULT_FUSION,
        ffmpeg_path: Optional[str] = None,
        ffprobe_path: Optional[str] = None,
        debug: bool = False,
        preserve_temp: bool = False,
    ) -> None:
        self._image = image_analyzer or ImageAnalyzer(text_analyzer=text_analyzer)
        self._audio = audio_analyzer or AudioAnalyzer(text_analyzer=text_analyzer)
        self._text = text_analyzer
        self.sampling = sampling
        self.fusion_config = fusion_config
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.debug = debug
        self.preserve_temp = preserve_temp

    def set_text_analyzer(self, text_analyzer: TextSentimentAnalyzer) -> None:
        self._text = text_analyzer
        self._image.set_text_analyzer(text_analyzer)
        self._audio.set_text_analyzer(text_analyzer)

    @staticmethod
    def validate_media_path(media_path: PathLike) -> Path:
        path = Path(media_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        return path

    def analyze(
        self,
        media_path: PathLike,
        *,
        caption_sentiment: Optional[SentimentEvidence] = None,
    ) -> VideoAnalysisBundle:
        """Analyze a video file; return multimodal evidence + compact diagnostics."""
        source = self.validate_media_path(media_path)
        started = time.perf_counter()
        warnings: list[str] = []

        # Serious probe / decode failures propagate (not hidden).
        probe = probe_video(source, ffprobe_path=self.ffprobe_path)
        effective_fps = self.sampling.effective_fps(probe.duration_seconds)
        if effective_fps + 1e-9 < self.sampling.fps:
            warnings.append(
                f"Reduced sampling FPS from {self.sampling.fps} to {effective_fps:.4f} "
                f"to respect max_frames={self.sampling.max_frames}",
            )

        tmp_root: Optional[Path] = None
        frame_paths: list[Path] = []

        try:
            tmp_root = Path(tempfile.mkdtemp(prefix="myuni_video_"))
            frames_dir = tmp_root / "frames"
            try:
                frame_paths = extract_frames_at_fps(
                    source,
                    frames_dir,
                    fps=effective_fps,
                    ffmpeg_path=self.ffmpeg_path,
                )
            except FFmpegNotFoundError:
                raise
            except FFmpegError:
                raise

            # Hard cap after extraction in case FPS math undershoots.
            if len(frame_paths) > self.sampling.max_frames:
                warnings.append(
                    f"Truncated extracted frames from {len(frame_paths)} "
                    f"to max_frames={self.sampling.max_frames}",
                )
                frame_paths = frame_paths[: self.sampling.max_frames]

            timestamps = [
                round(i / effective_fps, 3) for i in range(len(frame_paths))
            ]
            ocr_indices = _ocr_frame_indices(
                len(frame_paths),
                self.sampling.max_ocr_frames,
            )

            frame_visuals: list[SentimentEvidence] = []
            ocr_sentiments: list[SentimentEvidence] = []
            ocr_texts: list[str] = []
            frame_debug: list[VideoFrameDebug] = []
            frames_analyzed = 0
            ocr_unavailable_noted = False

            for idx, frame_path in enumerate(frame_paths):
                ts = timestamps[idx] if idx < len(timestamps) else None
                try:
                    from src.analyzers.image import ImageModalityEvidence
                    from src.analyzers.visual import VisualSentimentAnalyzer

                    if idx in ocr_indices:
                        evidence = self._image.analyze_path(frame_path)
                    else:
                        image = VisualSentimentAnalyzer.load_image(frame_path)
                        visual = self._image._visual.analyze_image(image)
                        evidence = ImageModalityEvidence(
                            visual=visual,
                            ocr_text=None,
                            ocr_sentiment=None,
                            warnings=[],
                        )

                    frame_visuals.append(evidence.visual)
                    frames_analyzed += 1

                    for w in evidence.warnings:
                        if "OCR unavailable" in w:
                            if not ocr_unavailable_noted:
                                warnings.append(w)
                                ocr_unavailable_noted = True
                        elif "OCR returned no text" not in w and w not in warnings:
                            warnings.append(f"frame[{idx}]: {w}")

                    if evidence.ocr_text and is_meaningful_ocr_text(evidence.ocr_text):
                        ocr_texts.append(evidence.ocr_text)
                    if evidence.ocr_sentiment is not None:
                        ocr_sentiments.append(evidence.ocr_sentiment)

                    if self.debug:
                        frame_debug.append(
                            VideoFrameDebug(
                                index=idx,
                                timestamp_seconds=ts,
                                visual_label=evidence.visual.label,
                                visual_score=evidence.visual.score,
                                visual_confidence=evidence.visual.confidence,
                                ocr_preview=(evidence.ocr_text or "")[:80] or None,
                            ),
                        )
                except Exception as exc:  # noqa: BLE001 — one frame must not fail the video
                    msg = f"frame[{idx}] analysis failed: {exc}"
                    warnings.append(msg)
                    logger.warning("%s", msg)
                    if self.debug:
                        frame_debug.append(
                            VideoFrameDebug(
                                index=idx,
                                timestamp_seconds=ts,
                                error=str(exc),
                            ),
                        )

            visual_summary = aggregate_frame_visual_scores(
                frame_visuals,
                config=self.fusion_config,
            )
            if visual_summary is None:
                warnings.append("No frames successfully analyzed for visual sentiment")

            ocr_sentiment = self._aggregate_ocr(ocr_sentiments, ocr_texts, warnings)
            combined_ocr_text = " | ".join(dict.fromkeys(ocr_texts)) or None

            speech_sentiment: Optional[SentimentEvidence] = None
            transcript: Optional[str] = None
            speech_result: Optional[SpeechAnalysisResult] = None

            if not probe.has_audio:
                warnings.append("Video has no audio stream; speech modality skipped")
            else:
                try:
                    speech_result = self._audio.analyze(source)
                    transcript = speech_result.transcript
                    speech_sentiment = speech_result.sentiment
                    for w in speech_result.warnings:
                        if w not in warnings:
                            warnings.append(w)
                except FFmpegNotFoundError:
                    raise
                except FFmpegError as exc:
                    warnings.append(f"Speech/audio extraction failed: {exc}")
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"Speech analysis failed: {exc}")
                    logger.exception("Speech analysis failed for %s", source)

            processing_seconds = time.perf_counter() - started
            diagnostics = VideoDiagnostics(
                duration_seconds=probe.duration_seconds,
                sampling_fps=float(effective_fps),
                frames_extracted=len(frame_paths),
                frames_analyzed=frames_analyzed,
                frame_timestamps=timestamps,
                processing_seconds=float(processing_seconds),
                has_audio=probe.has_audio,
                frame_debug=frame_debug if self.debug else None,
            )

            overall_fusion = fuse_modalities(
                {
                    "text": caption_sentiment,
                    "visual": visual_summary,
                    "ocr": ocr_sentiment,
                    "speech": speech_sentiment,
                },
                config=self.fusion_config,
            )
            overall = overall_fusion.overall

            logger.info(
                "Video analysis complete path=%s frames=%s/%s overall=%s warnings=%s",
                source,
                frames_analyzed,
                len(frame_paths),
                overall.label,
                len(warnings),
            )

            return VideoAnalysisBundle(
                visual=visual_summary,
                ocr=ocr_sentiment,
                ocr_text=combined_ocr_text,
                speech=speech_sentiment,
                transcript=transcript,
                speech_result=speech_result,
                diagnostics=diagnostics,
                warnings=warnings,
                overall=overall,
            )
        finally:
            if tmp_root is not None:
                if self.preserve_temp:
                    logger.info(
                        "Preserving temporary video artifacts at %s (preserve_temp=True)",
                        tmp_root,
                    )
                else:
                    import shutil

                    try:
                        shutil.rmtree(tmp_root, ignore_errors=False)
                    except OSError as exc:
                        logger.warning("Failed to clean temporary video files: %s", exc)

    def _aggregate_ocr(
        self,
        ocr_sentiments: Sequence[SentimentEvidence],
        ocr_texts: Sequence[str],
        warnings: list[str],
    ) -> Optional[SentimentEvidence]:
        if ocr_sentiments:
            aggregated = aggregate_frame_visual_scores(
                list(ocr_sentiments),
                config=self.fusion_config,
            )
            if aggregated is not None:
                return aggregated.model_copy(
                    update={
                        "details": {
                            **(aggregated.details or {}),
                            "source": "ocr",
                            "method": "confidence-weighted average over OCR frame sentiments",
                        },
                    },
                )

        # Fallback: score joined OCR text once if sentiments were missing but text exists.
        joined = " ".join(dict.fromkeys(ocr_texts)).strip()
        if not joined or not is_meaningful_ocr_text(joined):
            return None
        if self._text is None:
            warnings.append("OCR text found but text sentiment analyzer is not configured")
            return None
        try:
            scored = self._text.analyze(joined)
            return scored.model_copy(
                update={
                    "details": {
                        **(scored.details or {}),
                        "source": "ocr",
                        "extracted_text_preview": joined[:200],
                    },
                },
            )
        except ValueError as exc:
            warnings.append(f"OCR text could not be scored: {exc}")
            return None
