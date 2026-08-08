"""Internal media models with privacy-safe public serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResolvedSource(BaseModel):
    """A canonical local source; the real path is never serialized publicly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path = Field(exclude=True)
    source_kind: Literal["local", "download"]
    display_name: str
    cleanup_required: bool
    original_size: int = Field(ge=0)
    mime_type: str | None = None


class MediaRange(BaseModel):
    """An optional half-open time range requested by the caller."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_order(self) -> MediaRange:
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds <= self.start_seconds
        ):
            raise ValueError("end_seconds must follow start_seconds")
        return self


class MediaMetadata(BaseModel):
    """Bounded metadata obtained from trusted local inspection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    duration_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    frame_rate: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    mime_type: str
    has_audio: bool = False
    video_codec: str | None = None
    audio_codec: str | None = None
