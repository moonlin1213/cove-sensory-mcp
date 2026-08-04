"""Strict, credential-reference-only YAML configuration models."""

from __future__ import annotations

import ipaddress
import re
from typing import Literal
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cove_sensory_mcp.models import Modality, ProviderId, RouteConfig

_CREDENTIAL_PARAMETER_NAMES = frozenset(
    {
        "apikey",
        "token",
        "accesstoken",
        "refreshtoken",
        "idtoken",
        "secret",
        "clientsecret",
        "credential",
        "password",
        "passwd",
        "authorization",
        "auth",
        "key",
    }
)
_ENVIRONMENT_VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")


def _contains_credential_parameter(component: str) -> bool:
    """Return whether a URL query or fragment names a credential value."""
    return any(
        "".join(character for character in name.lower() if character.isalnum())
        in _CREDENTIAL_PARAMETER_NAMES
        for name, _ in parse_qsl(component.replace(";", "&"), keep_blank_values=True)
    )


def _is_valid_hostname(hostname: str) -> bool:
    """Validate an IP literal or an IDNA-compatible DNS hostname."""
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        try:
            ascii_hostname = hostname.encode("idna").decode("ascii").rstrip(".")
        except UnicodeError:
            return False
        labels = ascii_hostname.split(".")
        return bool(ascii_hostname) and len(ascii_hostname) <= 253 and all(
            label
            and len(label) <= 63
            and label[0].isalnum()
            and label[-1].isalnum()
            and all(character.isalnum() or character == "-" for character in label)
            for label in labels
        )
    return True


class ProviderConfig(BaseModel):
    """One provider's non-secret connection settings."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    adapter: str = Field(min_length=1)
    base_url: str | None = None
    model: str = Field(min_length=1)
    credential_ref: str | None = Field(default=None, min_length=1)
    api_key_env: str | None = None
    declared_capabilities: dict[Modality, bool] = Field(default_factory=dict)

    @field_validator("api_key_env")
    @classmethod
    def validate_environment_variable_name(cls, value: str | None) -> str | None:
        """Keep explicit environment references portable and bounded."""
        if value is not None and _ENVIRONMENT_VARIABLE_NAME.fullmatch(value) is None:
            raise ValueError("api_key_env must be a portable environment variable name")
        return value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        """Require a credential-free HTTPS endpoint with one valid host and port."""
        if value is None:
            return value
        try:
            parsed = urlsplit(value)
            hostname = parsed.hostname
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("base_url must be a valid HTTPS endpoint") from exc
        if (
            parsed.username is not None
            or parsed.password is not None
            or _contains_credential_parameter(parsed.query)
            or _contains_credential_parameter(parsed.fragment)
        ):
            raise ValueError("base_url must not contain credentials")
        authority = parsed.netloc.rsplit("@", maxsplit=1)[-1]
        if (
            parsed.scheme != "https"
            or hostname is None
            or not _is_valid_hostname(hostname)
            or authority.endswith(":")
            or parsed.query
            or parsed.fragment
            or any(
                character.isspace() or ord(character) < 32 or ord(character) == 127
                for character in value
            )
        ):
            raise ValueError("base_url must be a valid HTTPS endpoint")
        return value.rstrip("/")

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
    providers: dict[ProviderId, ProviderConfig] = Field(default_factory=dict)
    routes: RoutesConfig = Field(default_factory=RoutesConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    allowed_media_roots: list[str] = Field(default_factory=list)
