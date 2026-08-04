"""Strict, credential-reference-only YAML configuration models."""

from __future__ import annotations

import ipaddress
import re
from datetime import UTC, datetime
from typing import Annotated, Literal, TypeAlias
from urllib.parse import parse_qsl, urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

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
_MAX_CAPABILITIES = len(Modality)
_MAX_JOINT_CAPABILITIES = 26
_MAX_ADAPTER_OPTIONS = 32
_MAX_ADAPTER_OPTION_ITEMS = 32
_MAX_ADAPTER_OPTION_STRING = 2_048
_ADAPTER_OPTION_KEY_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")
_RESERVED_ADAPTER_OPTION_KEYS = frozenset(
    {
        "apikey",
        "apikeyenv",
        "auth",
        "authorization",
        "authorizationheader",
        "authorizationheaders",
        "authheader",
        "authheaders",
        "baseurl",
        "baseuri",
        "bearer",
        "credential",
        "credentialref",
        "endpoint",
        "env",
        "environment",
        "environmentvariable",
        "envsource",
        "header",
        "headers",
        "key",
        "keyring",
        "password",
        "passwd",
        "proxy",
        "secret",
        "token",
        "url",
    }
)
# Generic storage only proves a key/value is non-secret and bounded. Adapters that use
# known options such as endpoint_path or media_part_mode must validate their semantics.
_SAFE_ENDPOINT_OPTION_KEYS = frozenset({"endpointpath"})


def _normalize_config_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _validate_adapter_option_key(value: str) -> str:
    """Reject generic options that could shadow credentials or remote endpoints."""
    if _ADAPTER_OPTION_KEY_PATTERN.fullmatch(value) is None:
        raise ValueError("adapter option key is invalid")
    normalized = _normalize_config_name(value)
    if (
        normalized in _RESERVED_ADAPTER_OPTION_KEYS
        or "apikey" in normalized
        or "authorization" in normalized
        or "credential" in normalized
        or "headers" in normalized
        or "keyring" in normalized
        or "password" in normalized
        or "passwd" in normalized
        or "secret" in normalized
        or normalized.endswith(
            ("token", "key", "env", "environment", "envvar", "url", "uri")
        )
        or "baseurl" in normalized
        or "environmentvariable" in normalized
        or (
            "endpoint" in normalized
            and normalized not in _SAFE_ENDPOINT_OPTION_KEYS
        )
        or normalized.startswith(
            (
                "authentication",
                "authheader",
                "authkey",
                "authref",
                "authsource",
                "authtoken",
                "oauth",
                "proxy",
                "tokenenv",
                "tokenfile",
                "tokenref",
                "tokensource",
            )
        )
    ):
        raise ValueError("adapter option key is reserved")
    return value

AdapterOptionKey = Annotated[
    str,
    AfterValidator(_validate_adapter_option_key),
]
AdapterOptionString = Annotated[str, Field(max_length=_MAX_ADAPTER_OPTION_STRING)]
AdapterOptionInteger = Annotated[
    int,
    Field(strict=True, ge=-(2**63), le=2**63 - 1),
]
AdapterOptionFloat = Annotated[
    float,
    Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False),
]
AdapterOptionScalar: TypeAlias = (
    AdapterOptionString | StrictBool | AdapterOptionInteger | AdapterOptionFloat
)
AdapterOptionList = Annotated[
    list[AdapterOptionScalar],
    Field(max_length=_MAX_ADAPTER_OPTION_ITEMS),
]
AdapterOptionValue: TypeAlias = AdapterOptionScalar | AdapterOptionList
CapabilityMap = Annotated[
    dict[Modality, StrictBool],
    Field(max_length=_MAX_CAPABILITIES),
]
JointCapability = Annotated[
    frozenset[Modality],
    Field(min_length=2, max_length=_MAX_CAPABILITIES),
]
JointCapabilities = Annotated[
    list[JointCapability],
    Field(max_length=_MAX_JOINT_CAPABILITIES),
]
AdapterOptions = Annotated[
    dict[AdapterOptionKey, AdapterOptionValue],
    Field(max_length=_MAX_ADAPTER_OPTIONS),
]


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
    declared_capabilities: CapabilityMap = Field(default_factory=dict)
    verified_capabilities: CapabilityMap = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )
    verified_joint_capabilities: JointCapabilities = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
    )
    adapter_options: AdapterOptions = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )
    last_verified_at: datetime | None = None

    @field_validator("api_key_env")
    @classmethod
    def validate_environment_variable_name(cls, value: str | None) -> str | None:
        """Keep explicit environment references portable and bounded."""
        if value is not None and _ENVIRONMENT_VARIABLE_NAME.fullmatch(value) is None:
            raise ValueError("api_key_env must be a portable environment variable name")
        return value

    @field_validator("adapter_options", mode="before")
    @classmethod
    def validate_adapter_option_structure(cls, value: object) -> object:
        """Reject reserved keys and nested mappings before private input can surface."""
        if not isinstance(value, dict):
            return value
        for key, option in value.items():
            try:
                _validate_adapter_option_key(key)
            except (TypeError, ValueError):
                raise ValueError("adapter option key is invalid") from None
            if isinstance(option, dict):
                raise ValueError(  # noqa: TRY004 - Pydantic validation contract
                    "nested adapter option mappings are not allowed"
                )
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

    @field_validator("last_verified_at")
    @classmethod
    def normalize_verification_time(cls, value: datetime | None) -> datetime | None:
        """Require an unambiguous instant and persist it in UTC."""
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("last_verified_at must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_credential_reference(self) -> ProviderConfig:
        """Require exactly one non-secret credential lookup mechanism."""
        if self.credential_ref is None and self.api_key_env is None:
            raise ValueError("a provider requires credential_ref or api_key_env")
        if self.credential_ref is not None and self.api_key_env is not None:
            raise ValueError("a provider cannot use both credential_ref and api_key_env")
        declared = {
            modality for modality, enabled in self.declared_capabilities.items() if enabled
        }
        verified = {
            modality for modality, enabled in self.verified_capabilities.items() if enabled
        }
        if not verified <= declared:
            raise ValueError("verified capabilities must also be declared")
        if len(set(self.verified_joint_capabilities)) != len(
            self.verified_joint_capabilities
        ):
            raise ValueError("verified joint capabilities must be unique")
        if any(
            not joint <= declared or not joint <= verified
            for joint in self.verified_joint_capabilities
        ):
            raise ValueError(
                "verified joint capabilities must be individually declared and verified"
            )
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
