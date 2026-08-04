"""Strict, credential-reference-only YAML configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cove_sensory_mcp.models import RouteConfig


class ProviderConfig(BaseModel):
    """One provider's non-secret connection settings."""

    model_config = ConfigDict(extra="forbid")

    adapter: str = Field(min_length=1)
    base_url: str | None = None
    model: str = Field(min_length=1)
    credential_ref: str | None = Field(default=None, min_length=1)
    api_key_env: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_credential_reference(self) -> ProviderConfig:
        """Require exactly one non-secret credential lookup mechanism."""
        if self.credential_ref is None and self.api_key_env is None:
            raise ValueError("a provider requires credential_ref or api_key_env")
        if self.credential_ref is not None and self.api_key_env is not None:
            raise ValueError("a provider cannot use both credential_ref and api_key_env")
        return self


class RoutesConfig(BaseModel):
    """Explicit route choices for each supported sensory modality."""

    model_config = ConfigDict(extra="forbid")

    image: RouteConfig | None = None
    video_visual: RouteConfig | None = None
    video_audio: RouteConfig | None = None
    audio: RouteConfig | None = None
    music: RouteConfig | None = None


class LimitsConfig(BaseModel):
    """Reserved strict namespace for application-wide user limits."""

    model_config = ConfigDict(extra="forbid")


class AppConfig(BaseModel):
    """The complete versioned, non-secret local application configuration."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    routes: RoutesConfig = Field(default_factory=RoutesConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    allowed_media_roots: list[str] = Field(default_factory=list)
