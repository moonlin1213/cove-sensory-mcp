"""Privacy-safe Gemini adapter for native image, video, and audio inputs."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TypeVar

from cove_sensory_mcp.config.schema import ProviderConfig
from cove_sensory_mcp.config.secrets import SecretStore
from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import Modality, ProviderId
from cove_sensory_mcp.reports.normalize import normalize_provider_text
from cove_sensory_mcp.reports.prompts import build_sensory_prompt

from .base import MediaKind, ProviderCallResult, ProviderRequest

_LOGGER = logging.getLogger(__name__)
_DEFAULT_INLINE_MAX_BYTES = 20 * 1024 * 1024
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
_FILE_POLL_INTERVAL_SECONDS = 0.5

_AUTH_MESSAGE = "The provider credential was rejected."
_SAFETY_MESSAGE = "The provider rejected the request for safety reasons."
_TIMEOUT_MESSAGE = "The provider request timed out."
_UNAVAILABLE_MESSAGE = "The provider is temporarily unavailable."
_CAPABILITY_MESSAGE = "The provider cannot process the requested media modalities."
_TaskResult = TypeVar("_TaskResult")


@dataclass(frozen=True, slots=True)
class GeminiInlineMedia:
    """One in-request media part whose bytes never leave the call boundary in logs."""

    data: bytes
    mime_type: str


@dataclass(frozen=True, slots=True)
class GeminiUploadedMedia:
    """One Files API media reference used only as a generation input."""

    uri: str
    mime_type: str


@dataclass(frozen=True, slots=True)
class GeminiRemoteFile:
    """The minimum remote-file metadata required for use and cleanup."""

    name: str
    uri: str
    mime_type: str


@dataclass(frozen=True, slots=True)
class GeminiGeneration:
    """Sanitized generation output; raw SDK responses never cross this boundary."""

    text: str | None
    safety_rejected: bool = False


GeminiContent = str | GeminiInlineMedia | GeminiUploadedMedia


class GeminiFailureKind(str, Enum):
    """Stable failure categories emitted by the SDK wrapper."""

    AUTH = "auth"
    SAFETY = "safety"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"


class GeminiClientFailure(Exception):
    """An SDK-independent failure that deliberately retains no Provider exception."""

    def __init__(self, kind: GeminiFailureKind) -> None:
        super().__init__("Gemini client request failed.")
        self.kind = kind


class GeminiClient(Protocol):
    """Small async surface implemented by the official SDK wrapper and test fakes."""

    async def upload_file(self, *, path: Path, mime_type: str) -> GeminiRemoteFile:
        """Upload one local file and return its cleanup-safe identity."""
        ...

    async def wait_until_active(self, file: GeminiRemoteFile) -> GeminiRemoteFile:
        """Wait until an uploaded media file is usable for model inference."""
        ...

    async def generate_content(
        self,
        *,
        model: str,
        contents: tuple[GeminiContent, ...],
        max_output_tokens: int | None,
        temperature: float | None,
    ) -> GeminiGeneration:
        """Generate one sanitized text result from ordered content parts."""
        ...

    async def delete_file(self, *, name: str) -> None:
        """Delete one previously uploaded remote file."""
        ...

    async def aclose(self) -> None:
        """Close local SDK transport resources."""
        ...


class GeminiClientFactory(Protocol):
    """Construct a per-call client only after the credential has been resolved."""

    def __call__(self, *, api_key: str, base_url: str | None) -> GeminiClient:
        """Return a client without retaining the credential in adapter configuration."""
        ...


def _enum_name(value: object) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    if isinstance(value, str):
        return value.rsplit(".", maxsplit=1)[-1].upper()
    return ""


def _exception_status_code(exc: BaseException) -> int | None:
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _classify_sdk_exception(exc: BaseException) -> GeminiFailureKind:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return GeminiFailureKind.TIMEOUT
    if type(exc).__name__ in {
        "ConnectTimeout",
        "PoolTimeout",
        "ReadTimeout",
        "TimeoutException",
        "WriteTimeout",
    }:
        return GeminiFailureKind.TIMEOUT
    status_code = _exception_status_code(exc)
    if status_code in {401, 403}:
        return GeminiFailureKind.AUTH
    if status_code in {408, 504}:
        return GeminiFailureKind.TIMEOUT
    for attribute in ("status", "reason"):
        status = _enum_name(getattr(exc, attribute, None))
        if status in {"UNAUTHENTICATED", "PERMISSION_DENIED", "API_KEY_INVALID"}:
            return GeminiFailureKind.AUTH
        if status in {"SAFETY", "BLOCKED", "PROHIBITED_CONTENT"}:
            return GeminiFailureKind.SAFETY
        if status in {"DEADLINE_EXCEEDED", "TIMEOUT"}:
            return GeminiFailureKind.TIMEOUT
    return GeminiFailureKind.UNAVAILABLE


def _is_safety_rejection(response: object) -> bool:
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = _enum_name(getattr(feedback, "block_reason", None))
    if block_reason not in {"", "BLOCK_REASON_UNSPECIFIED", "UNSPECIFIED"}:
        return True
    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, list):
        return False
    safety_reasons = {
        "BLOCKLIST",
        "IMAGE_SAFETY",
        "PROHIBITED_CONTENT",
        "RECITATION",
        "SAFETY",
    }
    return any(
        _enum_name(getattr(candidate, "finish_reason", None)) in safety_reasons
        for candidate in candidates
    )


async def _await_owned_task(
    task: asyncio.Task[_TaskResult],
) -> tuple[_TaskResult | None, BaseException | None, bool]:
    """Await an owned task to terminal despite repeated caller cancellation."""
    interrupted = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                interrupted = True
        except BaseException:  # noqa: BLE001 - outcome is read only after terminal
            break
    if task.cancelled():
        return None, asyncio.CancelledError(), interrupted
    failure = task.exception()
    if failure is not None:
        return None, failure, interrupted
    return task.result(), None, interrupted


class _OfficialGoogleGenAIClient:
    """Compatibility wrapper for the bounded google-genai 2.x dependency."""

    def __init__(self, *, api_key: str, base_url: str | None) -> None:
        from google import genai
        from google.genai import types

        http_options = types.HttpOptions(base_url=base_url) if base_url else None
        self._client = genai.Client(api_key=api_key, http_options=http_options)
        self._types = types

    async def upload_file(self, *, path: Path, mime_type: str) -> GeminiRemoteFile:
        try:
            config = self._types.UploadFileConfig(mime_type=mime_type)
            uploaded = await self._client.aio.files.upload(
                file=path,
                config=config,
            )
            name = getattr(uploaded, "name", None)
            uri = getattr(uploaded, "uri", None)
            uploaded_mime = getattr(uploaded, "mime_type", None)
            if not isinstance(name, str) or not isinstance(uri, str):
                raise GeminiClientFailure(GeminiFailureKind.UNAVAILABLE)
            return GeminiRemoteFile(
                name=name,
                uri=uri,
                mime_type=uploaded_mime if isinstance(uploaded_mime, str) else mime_type,
            )
        except GeminiClientFailure:
            raise
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            kind = _classify_sdk_exception(exc)
        raise GeminiClientFailure(kind) from None

    async def wait_until_active(self, file: GeminiRemoteFile) -> GeminiRemoteFile:
        current = file
        while True:
            try:
                remote = await self._client.aio.files.get(name=current.name)
                state = _enum_name(getattr(remote, "state", None))
                if state == "ACTIVE":
                    name = getattr(remote, "name", current.name)
                    uri = getattr(remote, "uri", current.uri)
                    mime_type = getattr(remote, "mime_type", current.mime_type)
                    return GeminiRemoteFile(
                        name=name if isinstance(name, str) else current.name,
                        uri=uri if isinstance(uri, str) else current.uri,
                        mime_type=(
                            mime_type if isinstance(mime_type, str) else current.mime_type
                        ),
                    )
                if state == "FAILED":
                    raise GeminiClientFailure(GeminiFailureKind.UNAVAILABLE)
            except GeminiClientFailure:
                raise
            except BaseException as exc:
                if isinstance(exc, asyncio.CancelledError):
                    raise
                kind = _classify_sdk_exception(exc)
                raise GeminiClientFailure(kind) from None
            await asyncio.sleep(_FILE_POLL_INTERVAL_SECONDS)

    async def generate_content(
        self,
        *,
        model: str,
        contents: tuple[GeminiContent, ...],
        max_output_tokens: int | None,
        temperature: float | None,
    ) -> GeminiGeneration:
        parts: list[Any] = []
        for content in contents:
            if isinstance(content, str):
                parts.append(self._types.Part.from_text(text=content))
            elif isinstance(content, GeminiInlineMedia):
                parts.append(
                    self._types.Part.from_bytes(
                        data=content.data,
                        mime_type=content.mime_type,
                    )
                )
            else:
                parts.append(
                    self._types.Part.from_uri(
                        file_uri=content.uri,
                        mime_type=content.mime_type,
                    )
                )
        config = self._types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            response_mime_type="application/json",
        )
        try:
            response = await self._client.aio.models.generate_content(
                model=model,
                contents=[self._types.Content(role="user", parts=parts)],
                config=config,
            )
            safety_rejected = _is_safety_rejection(response)
            try:
                text = getattr(response, "text", None)
            except Exception:  # noqa: BLE001 - SDK property may raise arbitrary internals
                text = None
            return GeminiGeneration(
                text=text if isinstance(text, str) and text.strip() else None,
                safety_rejected=safety_rejected,
            )
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            kind = _classify_sdk_exception(exc)
        raise GeminiClientFailure(kind) from None

    async def delete_file(self, *, name: str) -> None:
        try:
            await self._client.aio.files.delete(name=name)
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            kind = _classify_sdk_exception(exc)
            raise GeminiClientFailure(kind) from None

    async def aclose(self) -> None:
        async_close = asyncio.create_task(self._client.aio.aclose())
        sync_close = asyncio.create_task(asyncio.to_thread(self._client.close))
        _, async_failure, async_interrupted = await _await_owned_task(async_close)
        _, sync_failure, sync_interrupted = await _await_owned_task(sync_close)
        if async_interrupted or sync_interrupted:
            raise asyncio.CancelledError
        if async_failure is not None or sync_failure is not None:
            raise GeminiClientFailure(GeminiFailureKind.UNAVAILABLE) from None


def _official_client_factory(*, api_key: str, base_url: str | None) -> GeminiClient:
    return _OfficialGoogleGenAIClient(api_key=api_key, base_url=base_url)


def _latency_bucket(started_at: float) -> str:
    elapsed = max(0.0, time.monotonic() - started_at)
    if elapsed < 1:
        return "under_1s"
    if elapsed < 5:
        return "1_to_5s"
    if elapsed < 30:
        return "5_to_30s"
    return "30s_or_more"


def _modality_label(request: ProviderRequest) -> str:
    return ",".join(sorted(item.value for item in request.requested_modalities))


def _safe_log(
    level: int,
    event: str,
    *,
    request_id: str,
    provider_id: ProviderId,
    model: str,
    modality: str,
    started_at: float,
    error_code: str,
) -> None:
    _LOGGER.log(
        level,
        event,
        extra={
            "request_id": request_id,
            "provider_id": provider_id,
            "model": model,
            "modality": modality,
            "latency_bucket": _latency_bucket(started_at),
            "error_code": error_code,
        },
    )


def _public_error(kind: GeminiFailureKind) -> SensoryError:
    if kind is GeminiFailureKind.AUTH:
        return SensoryError(ErrorCode.PROVIDER_AUTH_FAILED, _AUTH_MESSAGE)
    if kind is GeminiFailureKind.SAFETY:
        return SensoryError(ErrorCode.PROVIDER_SAFETY_REJECTED, _SAFETY_MESSAGE)
    if kind is GeminiFailureKind.TIMEOUT:
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


def _modalities_match_media(request: ProviderRequest) -> bool:
    allowed = {
        MediaKind.IMAGE: frozenset({Modality.IMAGE}),
        MediaKind.VIDEO: frozenset({Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO}),
        MediaKind.AUDIO: frozenset({Modality.AUDIO, Modality.MUSIC}),
    }[request.media.media_kind]
    return bool(request.requested_modalities) and request.requested_modalities <= allowed


async def _delete_remote_file(
    client: GeminiClient,
    remote_file: GeminiRemoteFile,
    *,
    timeout_seconds: float,
) -> bool:
    cleanup_task = asyncio.create_task(
        asyncio.wait_for(
            client.delete_file(name=remote_file.name),
            timeout=timeout_seconds,
        )
    )
    _, failure, interrupted = await _await_owned_task(cleanup_task)
    if interrupted:
        raise asyncio.CancelledError
    return failure is None


async def _close_client(client: GeminiClient) -> bool:
    close_task = asyncio.create_task(client.aclose())
    _, failure, interrupted = await _await_owned_task(close_task)
    if interrupted:
        raise asyncio.CancelledError
    return failure is None


class GeminiProvider:
    """Use one per-call official Gemini client to normalize native media reports."""

    def __init__(
        self,
        *,
        provider_id: ProviderId,
        config: ProviderConfig,
        secret_store: SecretStore,
        client_factory: GeminiClientFactory | None = None,
    ) -> None:
        self._provider_id = provider_id
        self._config = config
        self._secret_store = secret_store
        self._client_factory = client_factory or _official_client_factory

    def _credential(self) -> str:
        reference = self._config.credential_ref or self._provider_id
        return self._secret_store.get(reference, self._config.api_key_env)

    def _prompt(self, request: ProviderRequest) -> str:
        return build_sensory_prompt(
            request.requested_modalities,
            request.question or None,
            request.detail,
            request.language,
            request.start_seconds,
            request.end_seconds,
        )

    async def _prepare_media(
        self,
        client: GeminiClient,
        request: ProviderRequest,
    ) -> tuple[GeminiInlineMedia | None, GeminiRemoteFile | None]:
        inline_limit = (
            self._config.adapter_options.inline_max_bytes
            or _DEFAULT_INLINE_MAX_BYTES
        )
        if (
            request.media.media_kind is MediaKind.IMAGE
            and request.media.path.stat().st_size <= inline_limit
        ):
            inline = GeminiInlineMedia(
                data=request.media.path.read_bytes(),
                mime_type=request.media.mime_type,
            )
            return inline, None

        uploaded = await client.upload_file(
            path=request.media.path,
            mime_type=request.media.mime_type,
        )
        return None, uploaded

    def _ordered_contents(
        self,
        request: ProviderRequest,
        media: GeminiInlineMedia | GeminiUploadedMedia,
    ) -> tuple[GeminiContent, ...]:
        prompt = self._prompt(request)
        if request.media.media_kind in {MediaKind.IMAGE, MediaKind.VIDEO}:
            return media, prompt
        return prompt, media

    async def sense(self, request: ProviderRequest) -> ProviderCallResult:
        """Generate and strictly normalize one sensory report with upload cleanup."""
        started_at = time.monotonic()
        request_id = uuid.uuid4().hex
        modality = _modality_label(request)
        timeout_seconds = (
            self._config.adapter_options.request_timeout_seconds
            or _DEFAULT_REQUEST_TIMEOUT_SECONDS
        )
        client: GeminiClient | None = None
        remote_file: GeminiRemoteFile | None = None
        remote_file_deleted: bool | None = None
        observations = None
        pending_error: SensoryError | None = None

        try:
            if not _modalities_match_media(request):
                pending_error = SensoryError(
                    ErrorCode.PROVIDER_CAPABILITY_REJECTED,
                    _CAPABILITY_MESSAGE,
                )
            else:
                credential = self._credential()
                client = self._client_factory(
                    api_key=credential,
                    base_url=self._config.base_url,
                )
                try:
                    async with asyncio.timeout(timeout_seconds):
                        inline_media, remote_file = await self._prepare_media(
                            client,
                            request,
                        )
                        if remote_file is None:
                            if inline_media is None:
                                raise GeminiClientFailure(
                                    GeminiFailureKind.UNAVAILABLE
                                )
                            media: GeminiInlineMedia | GeminiUploadedMedia = inline_media
                        else:
                            active_file = await client.wait_until_active(remote_file)
                            media = GeminiUploadedMedia(
                                uri=active_file.uri,
                                mime_type=active_file.mime_type,
                            )
                        contents = self._ordered_contents(request, media)
                        generation = await client.generate_content(
                            model=self._config.model,
                            contents=contents,
                            max_output_tokens=(
                                self._config.adapter_options.max_output_tokens
                            ),
                            temperature=self._config.adapter_options.temperature,
                        )
                        if generation.safety_rejected:
                            raise GeminiClientFailure(GeminiFailureKind.SAFETY)
                        if generation.text is None:
                            raise GeminiClientFailure(GeminiFailureKind.UNAVAILABLE)
                        batch = normalize_provider_text(
                            generation.text,
                            expected_modalities=request.requested_modalities,
                            duration_seconds=request.media.duration_seconds,
                        )
                        observations = batch.by_modality()
                except TimeoutError:
                    pending_error = _public_error(GeminiFailureKind.TIMEOUT)
                except GeminiClientFailure as exc:
                    pending_error = _public_error(exc.kind)
                except SensoryError as exc:
                    pending_error = exc
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - privacy boundary discards SDK internals
                    pending_error = _public_error(GeminiFailureKind.UNAVAILABLE)
        except SensoryError as exc:
            pending_error = exc
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - factory/store internals are not retained
            pending_error = _public_error(GeminiFailureKind.UNAVAILABLE)
        finally:
            if client is not None:
                try:
                    if remote_file is not None:
                        remote_file_deleted = await _delete_remote_file(
                            client,
                            remote_file,
                            timeout_seconds=timeout_seconds,
                        )
                        if not remote_file_deleted:
                            _safe_log(
                                logging.WARNING,
                                "provider_remote_cleanup_failed",
                                request_id=request_id,
                                provider_id=self._provider_id,
                                model=self._config.model,
                                modality=modality,
                                started_at=started_at,
                                error_code="REMOTE_FILE_DELETE_FAILED",
                            )
                finally:
                    if not await _close_client(client):
                        _safe_log(
                            logging.WARNING,
                            "provider_client_close_failed",
                            request_id=request_id,
                            provider_id=self._provider_id,
                            model=self._config.model,
                            modality=modality,
                            started_at=started_at,
                            error_code="CLIENT_CLOSE_FAILED",
                        )

        if pending_error is not None:
            _safe_log(
                logging.WARNING,
                "provider_request_failed",
                request_id=request_id,
                provider_id=self._provider_id,
                model=self._config.model,
                modality=modality,
                started_at=started_at,
                error_code=pending_error.code.value,
            )
            raise pending_error
        if observations is None:
            raise _public_error(GeminiFailureKind.UNAVAILABLE)
        return ProviderCallResult(
            observations=observations,
            provider_id=self._provider_id,
            model=self._config.model,
            remote_file_deleted=remote_file_deleted,
        )
