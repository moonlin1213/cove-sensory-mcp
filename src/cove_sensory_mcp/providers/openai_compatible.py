"""Bounded explicit-shape adapter for user-configured compatible Providers."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from enum import Enum
from typing import Any

import httpx

from cove_sensory_mcp.config.schema import ProviderConfig
from cove_sensory_mcp.config.secrets import SecretStore
from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import Modality, ProviderId
from cove_sensory_mcp.reports.normalize import normalize_provider_text
from cove_sensory_mcp.reports.prompts import build_sensory_prompt

from .base import MediaKind, ProviderCallResult, ProviderRequest

_DEFAULT_MAX_BYTES = 200 * 1024 * 1024
_DEFAULT_MAX_OUTPUT_TOKENS = 4_096
_DEFAULT_TEMPERATURE = 0.2
_DEFAULT_TIMEOUT_SECONDS = 120.0
_MAX_MODEL_LENGTH = 256
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_CONTENT_BLOCKS = 64
_MAX_TEXT_CHARS = 2_000_000
_MAX_HEADER_VALUE_LENGTH = 8_192
_MIME_TYPE = re.compile(r"(image|video|audio)/[A-Za-z0-9][A-Za-z0-9.+-]{0,126}\Z")

_CONFIG_MESSAGE = "The custom provider settings are invalid."
_AUTH_MESSAGE = "The provider credential was rejected."
_SAFETY_MESSAGE = "The provider rejected the request for safety reasons."
_TIMEOUT_MESSAGE = "The provider request timed out."
_UNAVAILABLE_MESSAGE = "The provider is temporarily unavailable."
_CAPABILITY_MESSAGE = "The provider cannot process the requested media modalities."
_RESPONSE_MESSAGE = "The provider returned an unsupported observation response."
_TOO_LARGE_MESSAGE = "The prepared media exceeds the provider size limit."
_HEADER_ENV_MESSAGE = "A required provider header environment variable is unavailable."


class _FailureKind(str, Enum):
    AUTH = "auth"
    SAFETY = "safety"
    TIMEOUT = "timeout"
    CAPABILITY = "capability"
    UNAVAILABLE = "unavailable"


class _ResponseReadRejected(Exception):
    """Internal size marker that deliberately retains no response bytes."""


def _config_error() -> SensoryError:
    return SensoryError(ErrorCode.CONFIG_INVALID, _CONFIG_MESSAGE)


def _response_error() -> SensoryError:
    return SensoryError(ErrorCode.PROVIDER_CAPABILITY_REJECTED, _RESPONSE_MESSAGE)


def _public_error(kind: _FailureKind) -> SensoryError:
    if kind is _FailureKind.AUTH:
        return SensoryError(ErrorCode.PROVIDER_AUTH_FAILED, _AUTH_MESSAGE)
    if kind is _FailureKind.SAFETY:
        return SensoryError(ErrorCode.PROVIDER_SAFETY_REJECTED, _SAFETY_MESSAGE)
    if kind is _FailureKind.TIMEOUT:
        return SensoryError(
            ErrorCode.PROVIDER_TIMEOUT,
            _TIMEOUT_MESSAGE,
            retryable=True,
        )
    if kind is _FailureKind.CAPABILITY:
        return SensoryError(
            ErrorCode.PROVIDER_CAPABILITY_REJECTED,
            _CAPABILITY_MESSAGE,
        )
    return SensoryError(
        ErrorCode.PROVIDER_UNAVAILABLE,
        _UNAVAILABLE_MESSAGE,
        retryable=True,
    )


def _header_environment_error() -> SensoryError:
    return SensoryError(ErrorCode.SETUP_REQUIRED, _HEADER_ENV_MESSAGE)


def _build_image_url_data_uri(
    encoded_media: str,
    mime_type: str,
    media_kind: MediaKind,
) -> dict[str, object]:
    del media_kind
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded_media}"},
    }


def _build_input_audio_base64(
    encoded_media: str,
    mime_type: str,
    media_kind: MediaKind,
) -> dict[str, object]:
    del media_kind
    return {
        "type": "input_audio",
        "input_audio": {
            "data": encoded_media,
            "format": mime_type.partition("/")[2],
        },
    }


def _build_video_url_data_uri(
    encoded_media: str,
    mime_type: str,
    media_kind: MediaKind,
) -> dict[str, object]:
    del media_kind
    return {
        "type": "video_url",
        "video_url": {"url": f"data:{mime_type};base64,{encoded_media}"},
    }


def _build_anthropic_base64_media(
    encoded_media: str,
    mime_type: str,
    media_kind: MediaKind,
) -> dict[str, object]:
    return {
        "type": "image" if media_kind is MediaKind.IMAGE else "video",
        "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": encoded_media,
        },
    }


_MEDIA_BUILDERS = {
    "image_url_data_uri": _build_image_url_data_uri,
    "input_audio_base64": _build_input_audio_base64,
    "video_url_data_uri": _build_video_url_data_uri,
    "anthropic_base64_media": _build_anthropic_base64_media,
}

_MODE_CAPABILITIES = {
    "image_url_data_uri": frozenset({Modality.IMAGE}),
    "input_audio_base64": frozenset(
        {Modality.VIDEO_AUDIO, Modality.AUDIO, Modality.MUSIC}
    ),
    "video_url_data_uri": frozenset(
        {Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO}
    ),
    "anthropic_base64_media": frozenset(
        {Modality.IMAGE, Modality.VIDEO_VISUAL}
    ),
}


def _request_is_compatible(
    request: ProviderRequest,
    mode: str,
    declared_capabilities: frozenset[Modality],
) -> bool:
    if len(request.requested_modalities) != 1:
        return False
    modality = next(iter(request.requested_modalities))
    if modality not in declared_capabilities or modality not in _MODE_CAPABILITIES[mode]:
        return False
    mime_match = _MIME_TYPE.fullmatch(request.media.mime_type)
    if mime_match is None:
        return False
    mime_kind = mime_match.group(1)
    if mode == "image_url_data_uri":
        return request.media.media_kind is MediaKind.IMAGE and mime_kind == "image"
    if mode == "input_audio_base64":
        return request.media.media_kind is MediaKind.AUDIO and mime_kind == "audio"
    if mode == "video_url_data_uri":
        return request.media.media_kind is MediaKind.VIDEO and mime_kind == "video"
    if request.media.media_kind is MediaKind.IMAGE:
        return modality is Modality.IMAGE and mime_kind == "image"
    return (
        request.media.media_kind is MediaKind.VIDEO
        and modality is Modality.VIDEO_VISUAL
        and mime_kind == "video"
    )


def _decode_payload(body: bytes) -> dict[str, Any] | None:
    try:
        payload: Any = json.loads(body)
    except (RecursionError, UnicodeDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _declared_content_length(response: httpx.Response) -> None:
    raw_length = response.headers.get("content-length")
    if raw_length is None:
        return
    if (
        not raw_length.isascii()
        or not raw_length.isdigit()
        or int(raw_length) > _MAX_RESPONSE_BYTES
    ):
        raise _ResponseReadRejected


def _validate_content_encoding(response: httpx.Response) -> None:
    encodings = response.headers.get_list("content-encoding")
    if not encodings:
        return
    if len(encodings) != 1 or encodings[0].strip().lower() != "identity":
        raise _ResponseReadRejected


async def _read_bounded_response(response: httpx.Response) -> bytes:
    _validate_content_encoding(response)
    _declared_content_length(response)
    body = bytearray()
    async for chunk in response.aiter_raw():
        if len(chunk) > _MAX_RESPONSE_BYTES - len(body):
            raise _ResponseReadRejected
        body.extend(chunk)
    return bytes(body)


def _structured_error_kind(error: dict[str, Any]) -> _FailureKind | None:
    code = error.get("code", error.get("type"))
    if isinstance(code, str):
        normalized = code.strip().lower()
        if normalized in {
            "authentication_error",
            "invalid_api_key",
            "invalid_authentication",
            "permission_denied",
            "unauthorized",
        }:
            return _FailureKind.AUTH
        if normalized in {
            "rate_limit_error",
            "rate_limit_exceeded",
            "too_many_requests",
        }:
            return _FailureKind.UNAVAILABLE
        if normalized in {"content_filter", "prohibited_content", "safety_error"}:
            return _FailureKind.SAFETY
        if normalized in {
            "invalid_media_type",
            "unsupported_media",
            "unsupported_media_type",
        }:
            return _FailureKind.CAPABILITY
    message = error.get("message")
    if isinstance(message, str):
        normalized_message = message.strip().lower()
        if normalized_message in {
            "authentication failed",
            "invalid api key",
            "unauthorized",
        }:
            return _FailureKind.AUTH
        if normalized_message in {"rate limit exceeded", "too many requests"}:
            return _FailureKind.UNAVAILABLE
        if normalized_message in {"content policy", "safety policy", "unsafe media"}:
            return _FailureKind.SAFETY
        if normalized_message in {
            "invalid media type",
            "unsupported media",
            "unsupported media type",
        }:
            return _FailureKind.CAPABILITY
    return None


def _error_envelope_kind(
    payload: dict[str, Any],
) -> tuple[bool, _FailureKind | None]:
    if "error" not in payload or payload["error"] is None:
        return False, None
    error = payload["error"]
    if not isinstance(error, dict):
        return True, None
    return True, _structured_error_kind(error)


def _classify_response(status_code: int) -> _FailureKind:
    if status_code in {401, 403}:
        return _FailureKind.AUTH
    if status_code in {408, 504}:
        return _FailureKind.TIMEOUT
    if status_code == 415:
        return _FailureKind.CAPABILITY
    return _FailureKind.UNAVAILABLE


def _bounded_text(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_TEXT_CHARS:
        return None
    return value


def _text_from_blocks(value: object) -> str | None:
    direct_text = _bounded_text(value)
    if direct_text is not None:
        return direct_text
    if not isinstance(value, list) or len(value) > _MAX_CONTENT_BLOCKS:
        return None
    parts: list[str] = []
    combined = 0
    for block in value:
        if not isinstance(block, dict) or block.get("type") not in {
            "text",
            "output_text",
        }:
            continue
        text = _bounded_text(block.get("text"))
        if text is None:
            return None
        separator = 1 if parts else 0
        if len(text) + separator > _MAX_TEXT_CHARS - combined:
            return None
        parts.append(text)
        combined += separator + len(text)
    return "\n".join(parts) if parts else None


def _extract_response_text(payload: dict[str, Any]) -> str:
    if "choices" in payload:
        choices = payload["choices"]
        if not isinstance(choices, list) or not (
            0 < len(choices) <= _MAX_CONTENT_BLOCKS
        ):
            raise _response_error()
        choice = choices[0]
        if not isinstance(choice, dict):
            raise _response_error()
        message = choice.get("message")
        if not isinstance(message, dict):
            raise _response_error()
        text = _text_from_blocks(message.get("content"))
        if text is None:
            raise _response_error()
        return text

    if "output" in payload:
        output = payload["output"]
        if not isinstance(output, list) or len(output) > _MAX_CONTENT_BLOCKS:
            raise _response_error()
        output_blocks: list[object] = []
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                content = item.get("content")
                if isinstance(content, list):
                    output_blocks.extend(content)
                    if len(output_blocks) > _MAX_CONTENT_BLOCKS:
                        raise _response_error()
        text = _text_from_blocks(output_blocks)
        if text is not None:
            return text
        raise _response_error()

    if "content" not in payload:
        raise _response_error()
    text = _text_from_blocks(payload["content"])
    if text is not None:
        return text
    raise _response_error()


class OpenAICompatibleProvider:
    """Send one prepared file through one of four explicit compatible wire shapes."""

    def __init__(
        self,
        *,
        provider_id: ProviderId,
        config: ProviderConfig,
        secret_store: SecretStore,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        mode = config.adapter_options.media_part_mode
        endpoint_path = config.adapter_options.endpoint_path
        declared = frozenset(
            modality
            for modality, enabled in config.declared_capabilities.items()
            if enabled
        )
        if (
            config.adapter != "openai-compatible"
            or config.base_url is None
            or endpoint_path is None
            or mode not in _MEDIA_BUILDERS
            or not declared
            or not declared <= _MODE_CAPABILITIES[mode]
            or len(config.model) > _MAX_MODEL_LENGTH
            or not config.model.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in config.model)
        ):
            raise _config_error()
        self._provider_id = provider_id
        self._config = config
        self._secret_store = secret_store
        self._mode = mode
        self._declared_capabilities = declared
        self._endpoint = f"{config.base_url}{endpoint_path}"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            follow_redirects=False,
            trust_env=False,
        )

    def _max_bytes(self) -> int:
        return self._config.adapter_options.inline_max_bytes or _DEFAULT_MAX_BYTES

    def _timeout(self) -> float:
        return (
            self._config.adapter_options.request_timeout_seconds
            or _DEFAULT_TIMEOUT_SECONDS
        )

    def _credential(self) -> str:
        reference = self._config.credential_ref or self._provider_id
        try:
            return self._secret_store.get(reference, self._config.api_key_env)
        except SensoryError:
            raise
        except Exception:  # noqa: BLE001 - privacy boundary discards store internals
            raise _public_error(_FailureKind.UNAVAILABLE) from None

    def _headers(self, credential: str) -> dict[str, str]:
        headers = {
            "accept-encoding": "identity",
            "authorization": f"Bearer {credential}",
            "content-type": "application/json",
        }
        for header_name, environment_name in (
            self._config.adapter_options.extra_headers_env or {}
        ).items():
            try:
                value = os.environ[environment_name]
            except KeyError:
                headers.clear()
                raise _header_environment_error() from None
            if (
                not value.strip()
                or len(value) > _MAX_HEADER_VALUE_LENGTH
                or any(ord(character) < 32 or ord(character) > 126 for character in value)
            ):
                headers.clear()
                value = ""
                raise _header_environment_error()
            headers[header_name] = value
            value = ""
        return headers

    def _prompt(self, request: ProviderRequest) -> str:
        return build_sensory_prompt(
            request.requested_modalities,
            request.question or None,
            request.detail,
            request.language,
            request.start_seconds,
            request.end_seconds,
        )

    async def sense(self, request: ProviderRequest) -> ProviderCallResult:
        """Return one normalized observation without retaining raw compatible data."""
        if not _request_is_compatible(
            request,
            self._mode,
            self._declared_capabilities,
        ):
            raise _public_error(_FailureKind.CAPABILITY)

        try:
            size = request.media.path.stat().st_size
        except OSError:
            raise _public_error(_FailureKind.UNAVAILABLE) from None
        if size <= 0:
            raise _response_error()
        if size > self._max_bytes():
            raise SensoryError(ErrorCode.MEDIA_TOO_LARGE, _TOO_LARGE_MESSAGE)
        try:
            media_bytes = request.media.path.read_bytes()
        except OSError:
            raise _public_error(_FailureKind.UNAVAILABLE) from None
        if not media_bytes:
            raise _response_error()
        if len(media_bytes) > self._max_bytes():
            media_bytes = b""
            raise SensoryError(ErrorCode.MEDIA_TOO_LARGE, _TOO_LARGE_MESSAGE)

        encoded_media = base64.b64encode(media_bytes).decode("ascii")
        media_bytes = b""
        prompt = self._prompt(request)
        media_part = _MEDIA_BUILDERS[self._mode](
            encoded_media,
            request.media.mime_type,
            request.media.media_kind,
        )
        encoded_media = ""
        payload: dict[str, object] = {
            "model": self._config.model,
            "max_tokens": (
                self._config.adapter_options.max_output_tokens
                or _DEFAULT_MAX_OUTPUT_TOKENS
            ),
            "temperature": (
                _DEFAULT_TEMPERATURE
                if self._config.adapter_options.temperature is None
                else self._config.adapter_options.temperature
            ),
            "messages": [
                {
                    "role": "user",
                    "content": [media_part, {"type": "text", "text": prompt}],
                }
            ],
        }
        media_part = {}
        credential = self._credential()
        headers = self._headers(credential)
        credential = ""
        response_status: int | None = None
        response_success = False
        response_body = b""
        try:
            async with self._client.stream(
                "POST",
                self._endpoint,
                headers=headers,
                json=payload,
                timeout=self._timeout(),
                follow_redirects=False,
            ) as response:
                response_status = response.status_code
                response_success = response.is_success
                if response.is_redirect:
                    raise _public_error(_FailureKind.UNAVAILABLE)
                try:
                    response_body = await _read_bounded_response(response)
                except _ResponseReadRejected:
                    if response_success:
                        raise _response_error() from None
                    raise _public_error(_FailureKind.UNAVAILABLE) from None
        except asyncio.CancelledError:
            raise
        except SensoryError:
            raise
        except httpx.TimeoutException:
            raise _public_error(_FailureKind.TIMEOUT) from None
        except httpx.RequestError:
            raise _public_error(_FailureKind.UNAVAILABLE) from None
        except Exception:  # noqa: BLE001 - privacy boundary discards transport internals
            raise _public_error(_FailureKind.UNAVAILABLE) from None
        finally:
            headers.clear()
            payload.clear()
            prompt = ""

        if response_status is None:
            raise _public_error(_FailureKind.UNAVAILABLE)
        response_payload = _decode_payload(response_body)
        response_body = b""
        if response_payload is not None:
            error_present, error_kind = _error_envelope_kind(response_payload)
            if error_present:
                if error_kind is not None:
                    raise _public_error(error_kind)
                if response_success:
                    raise _response_error()
                raise _public_error(_classify_response(response_status))
        if not response_success:
            raise _public_error(_classify_response(response_status))
        if response_payload is None:
            raise _response_error()
        response_text = _extract_response_text(response_payload)
        try:
            batch = normalize_provider_text(
                response_text,
                expected_modalities=request.requested_modalities,
                duration_seconds=request.media.duration_seconds,
            )
        except RecursionError:
            raise _response_error() from None
        response_text = ""
        return ProviderCallResult(
            observations=batch.by_modality(),
            provider_id=self._provider_id,
            model=self._config.model,
            remote_file_deleted=None,
        )

    async def aclose(self) -> None:
        """Close only a client created and owned by this adapter."""
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()
