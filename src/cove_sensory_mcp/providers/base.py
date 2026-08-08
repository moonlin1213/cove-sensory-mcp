"""Transport-neutral contracts shared by sensory Provider adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

from cove_sensory_mcp.models import DetailLevel, Modality, ProviderId
from cove_sensory_mcp.reports.schemas import ObservationEnvelope


class MediaKind(str, Enum):
    """The physical kind of one prepared local media file."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


@dataclass(frozen=True, slots=True)
class PreparedMedia:
    """One already-resolved local file ready for a Provider adapter."""

    path: Path
    mime_type: str
    media_kind: MediaKind
    duration_seconds: float | None

    def __post_init__(self) -> None:
        if not self.mime_type or len(self.mime_type) > 255:
            raise ValueError("mime_type must contain one bounded media type")
        if self.duration_seconds is not None and (
            isinstance(self.duration_seconds, bool)
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds < 0
        ):
            raise ValueError("duration_seconds must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ProviderMediaLimits:
    """Adapter media limits applied before any Provider transport call."""

    max_bytes: int
    accepted_mime_types: frozenset[str]
    max_duration_seconds: float | None
    max_image_edge: int | None

    def __post_init__(self) -> None:
        if isinstance(self.max_bytes, bool) or self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if not self.accepted_mime_types or len(self.accepted_mime_types) > 64:
            raise ValueError("accepted_mime_types must contain one to 64 values")
        if any(not value or len(value) > 255 for value in self.accepted_mime_types):
            raise ValueError("accepted_mime_types values must be bounded")
        if self.max_duration_seconds is not None and (
            isinstance(self.max_duration_seconds, bool)
            or not math.isfinite(self.max_duration_seconds)
            or self.max_duration_seconds <= 0
        ):
            raise ValueError("max_duration_seconds must be finite and positive")
        if self.max_image_edge is not None and (
            isinstance(self.max_image_edge, bool) or self.max_image_edge <= 0
        ):
            raise ValueError("max_image_edge must be positive")


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    """One transport-independent request passed to a sensory Provider."""

    media: PreparedMedia
    requested_modalities: frozenset[Modality]
    question: str
    detail: DetailLevel
    language: str
    start_seconds: float | None = None
    end_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ProviderCallResult:
    """Normalized observations and safe transport lifecycle metadata."""

    observations: dict[Modality, ObservationEnvelope]
    provider_id: ProviderId
    model: str
    remote_file_deleted: bool | None


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """One stable capability self-test outcome without Provider raw data."""

    provider_id: ProviderId
    modality: Modality
    verified: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderCandidate:
    """One configured and verified route entry in deterministic order."""

    provider_id: ProviderId
    expected_model: str
    modalities: frozenset[Modality]
    is_fallback: bool = False


@runtime_checkable
class SensoryProvider(Protocol):
    """The asynchronous behavior implemented by every Provider adapter."""

    async def sense(self, request: ProviderRequest) -> ProviderCallResult:
        """Return normalized observations for one prepared-media request."""
        ...
