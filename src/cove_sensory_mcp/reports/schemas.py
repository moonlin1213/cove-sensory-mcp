"""Strict provider-output models for normalized sensory observations."""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import uuid4

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from cove_sensory_mcp.models import Modality

_MAX_SUMMARY_LENGTH = 8_000
_MAX_ITEM_TEXT_LENGTH = 2_000
_MAX_SEGMENTS = 200
_MAX_WARNINGS = 200


def _round_timestamp(value: float) -> float:
    """Round a validated finite, non-negative timestamp to milliseconds."""
    return round(value, 3)


Timestamp = Annotated[
    float,
    Field(strict=True, ge=0, allow_inf_nan=False),
    AfterValidator(_round_timestamp),
]


class ObservationSegment(BaseModel):
    """One directly observed, time-bounded piece of sensory evidence."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    start_seconds: Timestamp
    end_seconds: Timestamp
    text: str = Field(max_length=_MAX_ITEM_TEXT_LENGTH)

    @model_validator(mode="after")
    def validate_time_range(self) -> ObservationSegment:
        """Reject reversed ranges while allowing adjacent and overlapping evidence."""
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must follow start_seconds")
        return self


class TranscriptSegment(BaseModel):
    """One bounded item of speech or lyric transcription."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    start_seconds: Timestamp
    end_seconds: Timestamp
    text: str = Field(max_length=_MAX_ITEM_TEXT_LENGTH)

    @model_validator(mode="after")
    def validate_time_range(self) -> TranscriptSegment:
        """Reject reversed transcript ranges without forbidding overlap."""
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must follow start_seconds")
        return self


class ObservationWarning(BaseModel):
    """A bounded, machine-readable caveat attached to one observation."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    code: str = Field(min_length=1, max_length=128)
    message: str = Field(max_length=_MAX_ITEM_TEXT_LENGTH)


class ObservationEnvelope(BaseModel):
    """One modality's normalized summary, evidence, transcript, and caveats."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    modality: Modality
    summary: str = Field(max_length=_MAX_SUMMARY_LENGTH)
    segments: list[ObservationSegment] = Field(default_factory=list, max_length=_MAX_SEGMENTS)
    transcript: list[TranscriptSegment] = Field(default_factory=list, max_length=_MAX_SEGMENTS)
    warnings: list[ObservationWarning] = Field(default_factory=list, max_length=_MAX_WARNINGS)
    confidence: Literal["low", "medium", "high"]


class ProviderObservationBatch(BaseModel):
    """A strict batch containing one envelope for each returned modality."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    observations: list[ObservationEnvelope] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def reject_duplicate_modalities(self) -> ProviderObservationBatch:
        """Keep the modality-to-envelope mapping unambiguous."""
        modalities = [observation.modality for observation in self.observations]
        if len(set(modalities)) != len(modalities):
            raise ValueError("observation modalities must be unique")
        return self

    def by_modality(self) -> dict[Modality, ObservationEnvelope]:
        """Return the unique observations indexed by their modality."""
        return {observation.modality: observation for observation in self.observations}


class ProviderUsage(BaseModel):
    """Actual Provider identity disclosed by a public sensory result."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    provider_id: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=256)
    fallback_used: bool = False


class Coverage(BaseModel):
    """Truthful visual and auditory coverage for a sensory result."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    visual: bool = False
    audio: bool = False
    visual_provider: str | None = None
    audio_provider: str | None = None


class TimelineEvidence(BaseModel):
    """Deterministically aligned evidence from overlapping source intervals."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    start_seconds: Timestamp
    end_seconds: Timestamp
    visual: list[str] = Field(default_factory=list, max_length=_MAX_SEGMENTS)
    audio: list[str] = Field(default_factory=list, max_length=_MAX_SEGMENTS)
    transcript: list[str] = Field(default_factory=list, max_length=_MAX_SEGMENTS)


class SensoryToolResult(BaseModel):
    """Common public result returned by all four production sensing tools."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    schema_version: Literal["1.0"] = "1.0"
    request_id: str = Field(default_factory=lambda: f"sense_{uuid4().hex}")
    modality: Literal["image", "video", "audio", "music"]
    status: Literal["completed", "partial", "error"]
    coverage: Coverage = Field(default_factory=Coverage)
    summary: str = Field(max_length=_MAX_SUMMARY_LENGTH)
    warnings: list[str] = Field(default_factory=list, max_length=_MAX_WARNINGS)
    providers: list[ProviderUsage] = Field(default_factory=list, max_length=5)
    requested_start_seconds: Timestamp | None = None
    requested_end_seconds: Timestamp | None = None
    segments: list[ObservationSegment] = Field(default_factory=list, max_length=_MAX_SEGMENTS)
    transcript: list[TranscriptSegment] = Field(default_factory=list, max_length=_MAX_SEGMENTS)


class ImageSensoryReport(SensoryToolResult):
    modality: Literal["image"] = "image"


class AudioSensoryReport(SensoryToolResult):
    modality: Literal["audio"] = "audio"


class MusicSensoryReport(SensoryToolResult):
    modality: Literal["music"] = "music"


class VideoSensoryReport(SensoryToolResult):
    modality: Literal["video"] = "video"
    visual_summary: str = Field(default="", max_length=_MAX_SUMMARY_LENGTH)
    audio_summary: str = Field(default="", max_length=_MAX_SUMMARY_LENGTH)
    timeline: list[TimelineEvidence] = Field(default_factory=list, max_length=500)
