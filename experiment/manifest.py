"""Experiment manifest schema and loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

PathLike = Union[str, Path]
ExperimentModality = Literal["text", "image", "video"]


class ExperimentSample(BaseModel):
    """One sample in a controlled POC experiment manifest."""

    sample_id: str = Field(..., min_length=1)
    modality: ExperimentModality
    text: Optional[str] = None
    path: Optional[str] = Field(
        default=None,
        description="Local media path for image/video samples.",
    )
    gold_label: Optional[str] = None
    user_id: Optional[str] = None
    notes: Optional[str] = None
    compare_sampling: Optional[bool] = Field(
        default=None,
        description="For video: run fixed_fps and scene_keyframe when true.",
    )

    @field_validator("sample_id", mode="before")
    @classmethod
    def _strip_id(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("gold_label", mode="before")
    @classmethod
    def _normalize_gold(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip().lower()
            return stripped or None
        return value

    @model_validator(mode="after")
    def _validate_modality_fields(self) -> ExperimentSample:
        if self.modality == "text":
            if not self.text or not str(self.text).strip():
                raise ValueError(f"sample_id={self.sample_id}: text modality requires non-blank text")
        elif self.modality in ("image", "video"):
            if not self.path or not str(self.path).strip():
                raise ValueError(f"sample_id={self.sample_id}: {self.modality} requires path")
        if self.gold_label is not None and self.gold_label not in ("positive", "neutral", "negative"):
            raise ValueError(
                f"sample_id={self.sample_id}: gold_label must be positive/neutral/negative",
            )
        return self


class ExperimentManifest(BaseModel):
    """Manifest describing a reproducible POC experiment run."""

    experiment_id: str = Field(..., min_length=1)
    name: Optional[str] = None
    description: Optional[str] = None
    video_compare_strategies: bool = Field(
        default=False,
        description="When true, video samples also run scene_keyframe vs fixed_fps.",
    )
    samples: list[ExperimentSample] = Field(..., min_length=1)

    @field_validator("experiment_id", mode="before")
    @classmethod
    def _strip_experiment_id(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


def load_manifest(path: PathLike) -> ExperimentManifest:
    """Load experiment manifest from JSON file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Experiment manifest not found: {file_path}")
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        # Allow bare list of samples with synthetic experiment_id.
        return ExperimentManifest(
            experiment_id=file_path.stem,
            samples=data,
        )
    return ExperimentManifest.model_validate(data)
