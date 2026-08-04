"""Stable domain models shared by Cove Sensory MCP components."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Modality(str, Enum):
    """A sensory capability exposed by the public MCP schema."""

    IMAGE = "image"
    VIDEO_VISUAL = "video_visual"
    VIDEO_AUDIO = "video_audio"
    AUDIO = "audio"
    MUSIC = "music"


class DetailLevel(str, Enum):
    """The requested depth of a sensory observation."""

    AUTO = "auto"
    QUICK = "quick"
    DETAILED = "detailed"


class CapabilityStatus(BaseModel):
    """Configuration and verification state for one sensory modality."""

    model_config = ConfigDict(extra="forbid")

    modality: Modality
    enabled: bool
    verified: bool
    provider: str | None = None
    mode: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> CapabilityStatus:
        """Keep disabled, verified, and reason states mutually coherent."""
        if not self.enabled and self.verified:
            raise ValueError("a disabled capability cannot be verified")
        if not self.enabled and not (self.reason and self.reason.strip()):
            raise ValueError("a disabled capability requires a reason")
        if self.verified and self.reason is not None:
            raise ValueError("a verified capability cannot have a reason")
        return self


class ProviderRef(BaseModel):
    """A configured fallback-provider reference and its user authorization."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    authorized: bool = False


class RouteConfig(BaseModel):
    """One modality's primary Provider and explicitly authorized fallbacks."""

    model_config = ConfigDict(extra="forbid")

    primary: str = Field(min_length=1)
    fallbacks: list[ProviderRef] = Field(default_factory=list)


class SensoryStatus(BaseModel):
    """The public setup status returned by the ``sensory_status`` tool."""

    model_config = ConfigDict(extra="forbid")

    ready: bool
    version: str = Field(min_length=1)
    capabilities: dict[Modality, CapabilityStatus] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_capability_keys(self) -> SensoryStatus:
        """Ensure each public capability-map key identifies its own status."""
        for modality, capability in self.capabilities.items():
            if capability.modality is not modality:
                raise ValueError("capability key must match its modality")
        return self
