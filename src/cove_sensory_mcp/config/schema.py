"""Strict, credential-reference-only YAML configuration models."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Annotated, Literal
from urllib.parse import parse_qsl, unquote, urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails, PydanticCustomError

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
_HTTP_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}\Z")
_MAX_EXTRA_HEADERS = 16
_FORBIDDEN_EXTRA_HEADERS = frozenset(
    {
        "accept-encoding",
        "authorization",
        "connection",
        "content-encoding",
        "content-length",
        "content-type",
        "cookie",
        "expect",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "set-cookie",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_MAX_CAPABILITIES = len(Modality)
_MAX_JOINT_CAPABILITIES = 26
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
_MAX_INLINE_BYTES = 1_073_741_824
_MAX_OUTPUT_TOKENS = 1_000_000
_MAX_REQUEST_TIMEOUT_SECONDS = 3_600
_MAX_ENDPOINT_PATH_LENGTH = 1_024
_MAX_PROVIDER_MODEL_LENGTH = 256
_ADAPTER_OPTION_NAMES = frozenset(
    {
        "inline_max_bytes",
        "max_output_tokens",
        "temperature",
        "request_timeout_seconds",
        "endpoint_path",
        "media_part_mode",
        "extra_headers_env",
    }
)
_INVALID_ADAPTER_OPTIONS_MESSAGE = "Adapter options contain an unsupported field."
_INVALID_EXTRA_HEADERS_MESSAGE = "Adapter header environment references are invalid."
_INVALID_PROVIDER_MODEL_MESSAGE = "The provider model identifier is invalid."


def _private_adapter_validation_error(
    title: str,
    error_type: str,
    message: str,
) -> ValidationError:
    """Build one input-free adapter validation error safe for nested diagnostics."""
    return ValidationError.from_exception_data(
        title,
        [
            InitErrorDetails(
                type=PydanticCustomError(error_type, message),
                loc=(),
                input=None,
            )
        ],
        hide_input=True,
    )


def _invalid_extra_headers_error(title: str) -> ValidationError:
    return _private_adapter_validation_error(
        title,
        "adapter_extra_headers_invalid",
        _INVALID_EXTRA_HEADERS_MESSAGE,
    )


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


class AdapterOptions(BaseModel):
    """Closed, non-secret v1 adapter tuning options."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    inline_max_bytes: Annotated[
        int,
        Field(strict=True, gt=0, le=_MAX_INLINE_BYTES),
    ] | None = None
    max_output_tokens: Annotated[
        int,
        Field(strict=True, gt=0, le=_MAX_OUTPUT_TOKENS),
    ] | None = None
    temperature: Annotated[
        float,
        Field(strict=True, ge=0, le=2, allow_inf_nan=False),
    ] | None = None
    request_timeout_seconds: Annotated[
        float,
        Field(
            strict=True,
            gt=0,
            le=_MAX_REQUEST_TIMEOUT_SECONDS,
            allow_inf_nan=False,
        ),
    ] | None = None
    endpoint_path: StrictStr | None = Field(
        default=None,
        min_length=1,
        max_length=_MAX_ENDPOINT_PATH_LENGTH,
    )
    media_part_mode: Literal[
        "image_url_data_uri",
        "input_audio_base64",
        "video_url_data_uri",
        "anthropic_base64_media",
        "audio_url_data_uri",
    ] | None = None
    extra_headers_env: Annotated[
        dict[StrictStr, StrictStr],
        Field(max_length=_MAX_EXTRA_HEADERS),
    ] | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_unknown_options(cls, value: object) -> object:
        """Reject unknown keys without placing attacker-controlled data in errors."""
        if isinstance(value, Mapping):
            if any(key not in _ADAPTER_OPTION_NAMES for key in value):
                raise _private_adapter_validation_error(
                    cls.__name__,
                    "adapter_options_invalid",
                    _INVALID_ADAPTER_OPTIONS_MESSAGE,
                )
            extra_headers = value.get("extra_headers_env")
            if extra_headers is not None:
                if not isinstance(extra_headers, Mapping) or len(
                    extra_headers
                ) > _MAX_EXTRA_HEADERS:
                    raise _invalid_extra_headers_error(cls.__name__)
                normalized_names: set[str] = set()
                for header_name, environment_name in extra_headers.items():
                    if not isinstance(header_name, str) or not isinstance(
                        environment_name, str
                    ):
                        raise _invalid_extra_headers_error(cls.__name__)
                    normalized_name = header_name.lower()
                    if (
                        _HTTP_HEADER_NAME.fullmatch(header_name) is None
                        or normalized_name in _FORBIDDEN_EXTRA_HEADERS
                        or normalized_name in normalized_names
                        or _ENVIRONMENT_VARIABLE_NAME.fullmatch(environment_name)
                        is None
                    ):
                        raise _invalid_extra_headers_error(cls.__name__)
                    normalized_names.add(normalized_name)
        return value

    @field_validator("endpoint_path")
    @classmethod
    def validate_endpoint_path(cls, value: str | None) -> str | None:
        """Require a local relative-to-origin path without traversal or URL metadata."""
        if value is None:
            return None
        parsed = urlsplit(value)
        if (
            not value.startswith("/")
            or value.startswith("//")
            or parsed.scheme
            or parsed.netloc
            or parsed.query
            or parsed.fragment
            or "\\" in value
            or any(
                character.isspace() or ord(character) < 32 or ord(character) == 127
                for character in value
            )
        ):
            raise ValueError("endpoint_path must be a safe relative endpoint path")

        decoded = value
        for _ in range(len(value)):
            next_decoded = unquote(decoded)
            if next_decoded == decoded:
                break
            decoded = next_decoded
        decoded_parts = urlsplit(decoded)
        if (
            decoded.startswith("//")
            or decoded_parts.scheme
            or decoded_parts.netloc
            or decoded_parts.query
            or decoded_parts.fragment
            or "\\" in decoded
            or any(
                character.isspace() or ord(character) < 32 or ord(character) == 127
                for character in decoded
            )
        ):
            raise ValueError("endpoint_path must be a safe relative endpoint path")
        if any(
            segment.split(";", maxsplit=1)[0] in {".", ".."}
            for segment in decoded.split("/")
        ):
            raise ValueError("endpoint_path must not contain traversal")
        return value


def _adapter_options_are_empty(value: AdapterOptions) -> bool:
    return all(
        getattr(value, field_name) is None
        for field_name in AdapterOptions.model_fields
    )


class ProviderConfig(BaseModel):
    """One provider's non-secret connection settings."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    adapter: str = Field(min_length=1)
    base_url: str | None = None
    model: StrictStr
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
        default_factory=AdapterOptions,
        exclude_if=_adapter_options_are_empty,
    )
    last_verified_at: datetime | None = None

    @field_validator("model", mode="before")
    @classmethod
    def validate_model_identifier(cls, value: object) -> str:
        """Require one strict bounded model name without retaining rejected input."""
        if type(value) is not str or not value or len(value) > _MAX_PROVIDER_MODEL_LENGTH:
            raise _private_adapter_validation_error(
                cls.__name__,
                "provider_model_invalid",
                _INVALID_PROVIDER_MODEL_MESSAGE,
            )
        return value

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
