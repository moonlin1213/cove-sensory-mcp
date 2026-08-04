"""Privacy-safe MiniMax-M3 adapter for direct image and native-video input."""

from __future__ import annotations

import asyncio
import base64
import json
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

MINIMAX_GLOBAL_BASE_URL = "https://api.minimax.io"
MINIMAX_CN_BASE_URL = "https://api.minimaxi.com"

_ANTHROPIC_MESSAGES_PATH = "/anthropic/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_BYTES = 200 * 1024 * 1024
_DEFAULT_MAX_OUTPUT_TOKENS = 4_096
_DEFAULT_TEMPERATURE = 0.2
_DEFAULT_TIMEOUT_SECONDS = 120.0
_MAX_MODEL_LENGTH = 256
_MAX_TEMPERATURE = 1.0
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_CONTENT_BLOCKS = 64
_MAX_TEXT_BLOCK_CHARS = 2_000_000
_MAX_COMBINED_TEXT_CHARS = 2_000_000

_CONFIG_MESSAGE = "The MiniMax provider settings are invalid."
_AUTH_MESSAGE = "The provider credential was rejected."
_SAFETY_MESSAGE = "The provider rejected the request for safety reasons."
_TIMEOUT_MESSAGE = "The provider request timed out."
_UNAVAILABLE_MESSAGE = "The provider is temporarily unavailable."
_CAPABILITY_MESSAGE = "The provider cannot process the requested media modalities."
_RESPONSE_MESSAGE = "The provider returned an unsupported observation response."
_TOO_LARGE_MESSAGE = "The prepared media exceeds the provider size limit."


class MiniMaxRegion(str, Enum):
    """Built-in MiniMax API regions available without a custom base URL."""

    GLOBAL = "global"
    CN = "cn"


class MiniMaxAuthMode(str, Enum):
    """Explicit credential headers accepted by supported MiniMax endpoints."""

    X_API_KEY = "x-api-key"
    BEARER = "bearer"


class _FailureKind(str, Enum):
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    SAFETY = "safety"
    UNSUPPORTED_MEDIA = "unsupported_media"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"


def _public_error(kind: _FailureKind) -> SensoryError:
    if kind is _FailureKind.AUTH:
        return SensoryError(ErrorCode.PROVIDER_AUTH_FAILED, _AUTH_MESSAGE)
    if kind is _FailureKind.SAFETY:
        return SensoryError(ErrorCode.PROVIDER_SAFETY_REJECTED, _SAFETY_MESSAGE)
    if kind is _FailureKind.UNSUPPORTED_MEDIA:
        return SensoryError(
            ErrorCode.PROVIDER_CAPABILITY_REJECTED,
            _CAPABILITY_MESSAGE,
        )
    if kind is _FailureKind.TIMEOUT:
        return SensoryError(
            ErrorCode.PROVIDER_TIMEOUT,
            _TIMEOUT_MESSAGE,
            retryable=True,
        )
    return SensoryError(
        ErrorCode.PROVIDER_UNAVAILABLE,
        _UNAVAILABLE_MESSAGE,
        retryable=True,
    )


def _config_error() -> SensoryError:
    return SensoryError(ErrorCode.CONFIG_INVALID, _CONFIG_MESSAGE)


def _response_error() -> SensoryError:
    return SensoryError(ErrorCode.PROVIDER_CAPABILITY_REJECTED, _RESPONSE_MESSAGE)


class _MalformedEnvelope(Exception):
    """Internal marker that deliberately retains no response data."""


class _ResponseReadRejected(Exception):
    """Internal marker for invalid or oversized streamed response bytes."""


def _structured_error_kind(payload: dict[str, Any]) -> _FailureKind | None:
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    error_code = error.get("code", error.get("type"))
    if isinstance(error_code, str):
        normalized_code = error_code.strip().lower()
        if normalized_code in {
            "invalid_media_type",
            "unsupported_media",
            "unsupported_media_type",
        }:
            return _FailureKind.UNSUPPORTED_MEDIA
        if normalized_code in {
            "content_filter",
            "prohibited_content",
            "safety_error",
        }:
            return _FailureKind.SAFETY
    message = error.get("message")
    if isinstance(message, str):
        normalized_message = message.strip().lower()
        if normalized_message in {
            "invalid media type",
            "unsupported media",
            "unsupported media type",
        }:
            return _FailureKind.UNSUPPORTED_MEDIA
        if normalized_message in {
            "content policy",
            "safety policy",
            "unsafe media",
        }:
            return _FailureKind.SAFETY
    return None


def _classify_response(
    status_code: int,
    payload: dict[str, Any] | None,
) -> _FailureKind:
    if status_code in {401, 403}:
        return _FailureKind.AUTH
    if status_code == 429:
        return _FailureKind.RATE_LIMIT
    if status_code in {408, 504}:
        return _FailureKind.TIMEOUT
    if status_code == 415:
        return _FailureKind.UNSUPPORTED_MEDIA
    if payload is not None:
        structured_kind = _structured_error_kind(payload)
        if structured_kind is not None:
            return structured_kind
    return _FailureKind.UNAVAILABLE


def _decode_response_payload(body: bytes) -> dict[str, Any] | None:
    try:
        payload: Any = json.loads(body)
    except (RecursionError, UnicodeDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _declared_content_length(response: httpx.Response) -> int | None:
    raw_length = response.headers.get("content-length")
    if raw_length is None:
        return None
    if not raw_length.isascii() or not raw_length.isdigit():
        raise _ResponseReadRejected
    content_length = int(raw_length)
    if content_length > _MAX_RESPONSE_BYTES:
        raise _ResponseReadRejected
    return content_length


async def _read_bounded_response(response: httpx.Response) -> bytes:
    _declared_content_length(response)
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(chunk) > _MAX_RESPONSE_BYTES - len(body):
            raise _ResponseReadRejected
        body.extend(chunk)
    return bytes(body)


def _base_response_code(payload: dict[str, Any]) -> int | None:
    if "base_resp" not in payload:
        return None
    base_response = payload["base_resp"]
    if not isinstance(base_response, dict):
        raise _MalformedEnvelope
    code = base_response.get("status_code")
    if isinstance(code, bool) or not isinstance(code, int):
        raise _MalformedEnvelope
    return code


def _base_response_failure(code: int) -> _FailureKind | None:
    if code == 0:
        return None
    if code == 1001:
        return _FailureKind.TIMEOUT
    if code == 1002:
        return _FailureKind.RATE_LIMIT
    if code in {1004, 2049}:
        return _FailureKind.AUTH
    if code in {1026, 1027}:
        return _FailureKind.SAFETY
    return _FailureKind.UNAVAILABLE


def _extract_response_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if not isinstance(content, list) or len(content) > _MAX_CONTENT_BLOCKS:
        raise _response_error()
    text_parts: list[str] = []
    combined_length = 0
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        text = part.get("text")
        if not isinstance(text, str):
            raise _response_error()
        if len(text) > _MAX_TEXT_BLOCK_CHARS:
            raise _response_error()
        separator_length = 1 if combined_length else 0
        if len(text) + separator_length > _MAX_COMBINED_TEXT_CHARS - combined_length:
            raise _response_error()
        combined_length += separator_length + len(text)
        if text.strip():
            text_parts.append(text)
    if not text_parts:
        raise _response_error()
    return "\n".join(text_parts)


def _media_type(request: ProviderRequest) -> str | None:
    if (
        request.media.media_kind is MediaKind.IMAGE
        and request.requested_modalities == frozenset({Modality.IMAGE})
        and request.media.mime_type.startswith("image/")
    ):
        return "image"
    if (
        request.media.media_kind is MediaKind.VIDEO
        and request.requested_modalities == frozenset({Modality.VIDEO_VISUAL})
        and request.media.mime_type.startswith("video/")
    ):
        return "video"
    return None


class MiniMaxM3Provider:
    """Send one prepared image or video directly to MiniMax-M3."""

    def __init__(
        self,
        *,
        provider_id: ProviderId,
        config: ProviderConfig,
        secret_store: SecretStore,
        client: httpx.AsyncClient | None = None,
        region: MiniMaxRegion = MiniMaxRegion.GLOBAL,
        auth_mode: MiniMaxAuthMode = MiniMaxAuthMode.X_API_KEY,
    ) -> None:
        self._validate_settings(config, region, auth_mode)
        self._provider_id = provider_id
        self._config = config
        self._secret_store = secret_store
        self._auth_mode = auth_mode
        regional_base_url = {
            MiniMaxRegion.GLOBAL: MINIMAX_GLOBAL_BASE_URL,
            MiniMaxRegion.CN: MINIMAX_CN_BASE_URL,
        }[region]
        self._base_url = config.base_url or regional_base_url
        self._endpoint = f"{self._base_url}{_ANTHROPIC_MESSAGES_PATH}"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(follow_redirects=False)

    @staticmethod
    def _validate_settings(
        config: ProviderConfig,
        region: MiniMaxRegion,
        auth_mode: MiniMaxAuthMode,
    ) -> None:
        if not isinstance(region, MiniMaxRegion) or not isinstance(
            auth_mode, MiniMaxAuthMode
        ):
            raise _config_error()
        model = config.model
        if (
            len(model) > _MAX_MODEL_LENGTH
            or not model.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in model)
        ):
            raise _config_error()
        temperature = config.adapter_options.temperature
        if temperature is not None and temperature > _MAX_TEMPERATURE:
            raise _config_error()

    def _credential(self) -> str:
        reference = self._config.credential_ref or self._provider_id
        try:
            return self._secret_store.get(reference, self._config.api_key_env)
        except SensoryError:
            raise
        except Exception:  # noqa: BLE001 - privacy boundary discards backend diagnostics
            raise _public_error(_FailureKind.UNAVAILABLE) from None

    def _headers(self, credential: str) -> dict[str, str]:
        headers = {
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        if self._auth_mode is MiniMaxAuthMode.X_API_KEY:
            headers["x-api-key"] = credential
        else:
            headers["authorization"] = f"Bearer {credential}"
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

    def _max_bytes(self) -> int:
        return self._config.adapter_options.inline_max_bytes or _DEFAULT_MAX_BYTES

    def _max_output_tokens(self) -> int:
        return (
            self._config.adapter_options.max_output_tokens
            or _DEFAULT_MAX_OUTPUT_TOKENS
        )

    def _temperature(self) -> float:
        temperature = self._config.adapter_options.temperature
        return _DEFAULT_TEMPERATURE if temperature is None else temperature

    def _timeout(self) -> float:
        return (
            self._config.adapter_options.request_timeout_seconds
            or _DEFAULT_TIMEOUT_SECONDS
        )

    async def sense(self, request: ProviderRequest) -> ProviderCallResult:
        """Return one normalized direct-visual observation without retaining raw data."""
        block_type = _media_type(request)
        if block_type is None:
            raise SensoryError(
                ErrorCode.PROVIDER_CAPABILITY_REJECTED,
                _CAPABILITY_MESSAGE,
            )

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
        payload: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": self._max_output_tokens(),
            "temperature": self._temperature(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": block_type,
                            "source": {
                                "type": "base64",
                                "media_type": request.media.mime_type,
                                "data": encoded_media,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        encoded_media = ""
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
        response_payload = _decode_response_payload(response_body)
        response_body = b""
        if response_payload is None:
            if response_success:
                raise _response_error()
            raise _public_error(_FailureKind.UNAVAILABLE)
        try:
            base_code = _base_response_code(response_payload)
        except _MalformedEnvelope:
            if response_success:
                raise _response_error() from None
            raise _public_error(_FailureKind.UNAVAILABLE) from None
        if base_code is not None:
            base_failure = _base_response_failure(base_code)
            if base_failure is not None:
                raise _public_error(base_failure)
        elif response_success:
            raise _response_error()
        if not response_success:
            raise _public_error(_classify_response(response_status, response_payload))

        response_text = _extract_response_text(response_payload)
        batch = normalize_provider_text(
            response_text,
            expected_modalities=request.requested_modalities,
            duration_seconds=request.media.duration_seconds,
        )
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
