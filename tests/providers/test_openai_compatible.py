from __future__ import annotations

import asyncio
import base64
import gzip
import json
import logging
import zlib
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import yaml
from pydantic import ValidationError

import cove_sensory_mcp.providers.openai_compatible as compatible_module
from cove_sensory_mcp.config.schema import AdapterOptions, AppConfig, ProviderConfig
from cove_sensory_mcp.config.secrets import MemorySecretStore
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import DetailLevel, Modality
from cove_sensory_mcp.providers.base import MediaKind, PreparedMedia, ProviderRequest
from cove_sensory_mcp.providers.openai_compatible import OpenAICompatibleProvider
from cove_sensory_mcp.reports.prompts import build_sensory_prompt

_PRIMARY_SECRET = "primary-custom-provider-secret"
_EXTRA_HEADER_ENV = "COVE_TEST_CUSTOM_HEADER"
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def _observation(modality: Modality, *, summary: str = "Direct evidence.") -> dict[str, object]:
    return {
        "modality": modality.value,
        "summary": summary,
        "segments": [],
        "transcript": [],
        "warnings": [],
        "confidence": "medium",
    }


def _report_text(modality: Modality, *, summary: str = "Direct evidence.") -> str:
    return json.dumps({"observations": [_observation(modality, summary=summary)]})


def _chat_response(modality: Modality, *, summary: str = "Direct evidence.") -> dict[str, object]:
    return {
        "id": "response-id",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": _report_text(modality, summary=summary),
                },
                "finish_reason": "stop",
            }
        ],
    }


def _config(
    mode: str,
    *,
    capabilities: frozenset[Modality] | None = None,
    base_url: str = "https://custom.example.test/api",
    endpoint_path: str = "/v1/chat/completions",
    max_bytes: int = 1_024,
    extra_headers_env: dict[str, str] | None = None,
) -> ProviderConfig:
    if capabilities is None:
        capabilities = {
            "image_url_data_uri": frozenset({Modality.IMAGE}),
            "input_audio_base64": frozenset(
                {Modality.VIDEO_AUDIO, Modality.AUDIO, Modality.MUSIC}
            ),
            "video_url_data_uri": frozenset(
                {Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO}
            ),
            "anthropic_base64_media": frozenset({Modality.IMAGE}),
            "audio_url_data_uri": frozenset(
                {Modality.VIDEO_AUDIO, Modality.AUDIO, Modality.MUSIC}
            ),
        }.get(mode, frozenset({Modality.IMAGE}))
    declared = {modality: True for modality in capabilities}
    return ProviderConfig(
        adapter="openai-compatible",
        base_url=base_url,
        model="custom-model",
        credential_ref="custom-provider-credential",
        declared_capabilities=declared,
        verified_capabilities=declared,
        adapter_options={
            "endpoint_path": endpoint_path,
            "media_part_mode": mode,
            "inline_max_bytes": max_bytes,
            "max_output_tokens": 768,
            "temperature": 0.2,
            "request_timeout_seconds": 9.0,
            "extra_headers_env": extra_headers_env or {},
        },
    )


def _request(
    path: Path,
    *,
    media_kind: MediaKind,
    mime_type: str,
    modality: Modality,
) -> ProviderRequest:
    return ProviderRequest(
        media=PreparedMedia(
            path=path,
            mime_type=mime_type,
            media_kind=media_kind,
            duration_seconds=None,
        ),
        requested_modalities=frozenset({modality}),
        question="",
        detail=DetailLevel.QUICK,
        language="en",
    )


def _store() -> MemorySecretStore:
    store = MemorySecretStore()
    store.set("custom-provider-credential", _PRIMARY_SECRET)
    return store


class StaticAsyncStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.body


def _client(
    handler: Callable[[httpx.Request], object],
    *,
    follow_redirects: bool = False,
) -> httpx.AsyncClient:
    async def streaming_handler(request: httpx.Request) -> object:
        response = handler(request)
        if isinstance(response, httpx.Response) and response.is_stream_consumed:
            return httpx.Response(
                response.status_code,
                headers=response.headers.raw,
                stream=StaticAsyncStream(response.content),
                extensions=response.extensions,
            )
        return response

    return httpx.AsyncClient(
        transport=httpx.MockTransport(cast(Any, streaming_handler)),
        follow_redirects=follow_redirects,
    )


def _provider(
    client: httpx.AsyncClient,
    config: ProviderConfig,
    *,
    store: MemorySecretStore | None = None,
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        provider_id="custom",
        config=config,
        secret_store=store or _store(),
        client=client,
    )


def _prompt(modality: Modality) -> str:
    return build_sensory_prompt(
        frozenset({modality}),
        None,
        DetailLevel.QUICK,
        "en",
        None,
        None,
    )


@pytest.mark.parametrize(
    ("mode", "media_kind", "mime_type", "modality", "expected_media_part"),
    [
        pytest.param(
            "image_url_data_uri",
            MediaKind.IMAGE,
            "image/jpeg",
            Modality.IMAGE,
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,bWVkaWEtYnl0ZXM="},
            },
            id="image-url-data-uri",
        ),
        pytest.param(
            "input_audio_base64",
            MediaKind.AUDIO,
            "audio/wav",
            Modality.AUDIO,
            {
                "type": "input_audio",
                "input_audio": {"data": "bWVkaWEtYnl0ZXM=", "format": "wav"},
            },
            id="input-audio-base64",
        ),
        pytest.param(
            "audio_url_data_uri",
            MediaKind.AUDIO,
            "audio/wav",
            Modality.AUDIO,
            {
                "type": "audio_url",
                "audio_url": {"url": "data:audio/wav;base64,bWVkaWEtYnl0ZXM="},
            },
            id="audio-url-data-uri-audio",
        ),
        pytest.param(
            "audio_url_data_uri",
            MediaKind.AUDIO,
            "audio/mpeg",
            Modality.MUSIC,
            {
                "type": "audio_url",
                "audio_url": {"url": "data:audio/mpeg;base64,bWVkaWEtYnl0ZXM="},
            },
            id="audio-url-data-uri-music",
        ),
        pytest.param(
            "audio_url_data_uri",
            MediaKind.AUDIO,
            "audio/wav",
            Modality.VIDEO_AUDIO,
            {
                "type": "audio_url",
                "audio_url": {"url": "data:audio/wav;base64,bWVkaWEtYnl0ZXM="},
            },
            id="audio-url-data-uri-video-audio",
        ),
        pytest.param(
            "video_url_data_uri",
            MediaKind.VIDEO,
            "video/mp4",
            Modality.VIDEO_VISUAL,
            {
                "type": "video_url",
                "video_url": {"url": "data:video/mp4;base64,bWVkaWEtYnl0ZXM="},
            },
            id="video-url-data-uri",
        ),
        pytest.param(
            "anthropic_base64_media",
            MediaKind.IMAGE,
            "image/png",
            Modality.IMAGE,
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "bWVkaWEtYnl0ZXM=",
                },
            },
            id="anthropic-base64-media",
        ),
    ],
)
@pytest.mark.asyncio
async def test_each_named_mode_emits_its_exact_bounded_request_shape(
    tmp_path: Path,
    mode: str,
    media_kind: MediaKind,
    mime_type: str,
    modality: Modality,
    expected_media_part: dict[str, object],
) -> None:
    """Changing a named builder's wire shape must not silently target another protocol."""
    media = tmp_path / "media.bin"
    media.write_bytes(b"media-bytes")
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_chat_response(modality))

    async with _client(handler) as client:
        result = await _provider(client, _config(mode)).sense(
            _request(
                media,
                media_kind=media_kind,
                mime_type=mime_type,
                modality=modality,
            )
        )

    assert len(captured) == 1
    assert captured[0].url == "https://custom.example.test/api/v1/chat/completions"
    assert captured[0].headers["authorization"] == f"Bearer {_PRIMARY_SECRET}"
    assert captured[0].headers["accept-encoding"] == "identity"
    assert json.loads(captured[0].content) == {
        "model": "custom-model",
        "max_tokens": 768,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [
                    expected_media_part,
                    {"type": "text", "text": _prompt(modality)},
                ],
            }
        ],
    }
    assert result.observations[modality].summary == "Direct evidence."
    assert result.remote_file_deleted is None


@pytest.mark.parametrize(
    ("mode", "capability"),
    [
        pytest.param("image_url_data_uri", Modality.VIDEO_VISUAL, id="image-claims-video"),
        pytest.param("input_audio_base64", Modality.IMAGE, id="audio-claims-image"),
        pytest.param("video_url_data_uri", Modality.AUDIO, id="video-claims-audio"),
        pytest.param("anthropic_base64_media", Modality.MUSIC, id="anthropic-claims-music"),
        pytest.param("audio_url_data_uri", Modality.IMAGE, id="audio-url-claims-image"),
    ],
)
def test_mode_rejects_incompatible_declared_capability(
    mode: str,
    capability: Modality,
) -> None:
    """A named media shape must not advertise a modality it cannot encode."""
    config = _config(mode, capabilities=frozenset({capability}))

    with pytest.raises(SensoryError) as caught:
        OpenAICompatibleProvider(
            provider_id="custom",
            config=config,
            secret_store=_store(),
        )

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert caught.value.cause is None


class FailingSecretStore:
    def get(self, ref: str, env_name: str | None = None) -> str:
        raise AssertionError("secret lookup must not happen")


@pytest.mark.asyncio
async def test_joint_or_mismatched_request_fails_before_secret_file_and_network(
    tmp_path: Path,
) -> None:
    """Widening a custom mode to joint input could send unauthorized media before routing."""
    transport_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal transport_calls
        transport_calls += 1
        raise AssertionError("network must not happen")

    request = ProviderRequest(
        media=PreparedMedia(
            path=tmp_path / "must-not-be-read.mp4",
            mime_type="video/mp4",
            media_kind=MediaKind.VIDEO,
            duration_seconds=1.0,
        ),
        requested_modalities=frozenset(
            {Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO}
        ),
        question="",
        detail=DetailLevel.QUICK,
        language="en",
    )
    config = _config(
        "video_url_data_uri",
        capabilities=frozenset({Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO}),
    )

    async with _client(handler) as client:
        provider = OpenAICompatibleProvider(
            provider_id="custom",
            config=config,
            secret_store=cast(Any, FailingSecretStore()),
            client=client,
        )
        with pytest.raises(SensoryError) as caught:
            await provider.sense(request)

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert transport_calls == 0


@pytest.mark.parametrize(
    ("headers", "private_marker"),
    [
        pytest.param({"Host": "SAFE_ENV"}, "Host", id="authority"),
        pytest.param({"Content-Length": "SAFE_ENV"}, "Content-Length", id="framing"),
        pytest.param({"Transfer-Encoding": "SAFE_ENV"}, "Transfer-Encoding", id="hop-by-hop"),
        pytest.param({"Connection": "SAFE_ENV"}, "Connection", id="connection"),
        pytest.param({"Cookie": "SAFE_ENV"}, "Cookie", id="cookie"),
        pytest.param({"Authorization": "SAFE_ENV"}, "Authorization", id="primary-auth-override"),
        pytest.param({"aCcEpT-EnCoDiNg": "SAFE_ENV"}, "aCcEpT-EnCoDiNg", id="compression-override"),
        pytest.param({"X Good": "SAFE_ENV"}, "X Good", id="whitespace"),
        pytest.param({"X-Good\r\nX-Evil": "SAFE_ENV"}, "X-Evil", id="crlf"),
        pytest.param({"X-Custom": "literal-secret-value"}, "literal-secret-value", id="literal-value"),
        pytest.param({"X-Custom": "1INVALID_ENV"}, "1INVALID_ENV", id="invalid-env-name"),
        pytest.param({"X-Test": "ENV_ONE", "x-test": "ENV_TWO"}, "ENV_TWO", id="case-duplicate"),
    ],
)
def test_extra_header_schema_rejects_unsafe_or_literal_mapping_without_echo(
    headers: dict[str, str],
    private_marker: str,
) -> None:
    """Weak header references could persist secrets or permit request smuggling."""
    with pytest.raises(ValidationError) as caught:
        AdapterOptions.model_validate({"extra_headers_env": headers})

    diagnostics = f"{caught.value}\n{caught.value.errors()!r}"
    assert private_marker not in diagnostics


def test_extra_header_env_mapping_is_bounded_yaml_safe_and_round_trips(tmp_path: Path) -> None:
    """Replacing env references with arbitrary objects or values must break persistence."""
    path = tmp_path / "config.yaml"
    provider = _config(
        "image_url_data_uri",
        extra_headers_env={"X-Tenant-Token": "COVE_CUSTOM_TENANT_TOKEN"},
    )
    expected = AppConfig(providers={"custom": provider})

    ConfigStore(path).save(expected)

    saved = path.read_text(encoding="utf-8")
    assert ConfigStore(path).load() == expected
    assert "X-Tenant-Token: COVE_CUSTOM_TENANT_TOKEN" in saved
    assert "!!python" not in saved


@pytest.mark.asyncio
async def test_extra_header_values_are_read_per_call_and_never_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Caching a resolved custom header would defeat rotation and retain a secret."""
    media = tmp_path / "rotating.jpg"
    media.write_bytes(b"image")
    first = "private-rotating-header-one"
    second = "private-rotating-header-two"
    seen: list[str] = []
    config = _config(
        "image_url_data_uri",
        extra_headers_env={"X-Tenant-Token": _EXTRA_HEADER_ENV},
    )
    caplog.set_level(logging.DEBUG)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["x-tenant-token"])
        return httpx.Response(200, json=_chat_response(Modality.IMAGE))

    async with _client(handler) as client:
        provider = _provider(client, config)
        monkeypatch.setenv(_EXTRA_HEADER_ENV, first)
        await provider.sense(
            _request(media, media_kind=MediaKind.IMAGE, mime_type="image/jpeg", modality=Modality.IMAGE)
        )
        monkeypatch.setenv(_EXTRA_HEADER_ENV, second)
        await provider.sense(
            _request(media, media_kind=MediaKind.IMAGE, mime_type="image/jpeg", modality=Modality.IMAGE)
        )

    serialized = yaml.safe_dump(config.model_dump(mode="json", exclude_none=True))
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert seen == [first, second]
    for private_value in (first, second):
        assert private_value not in serialized
        assert private_value not in log_text


@pytest.mark.asyncio
async def test_endpoint_path_preserves_valid_base_prefix_and_redirect_is_not_replayed(
    tmp_path: Path,
) -> None:
    """Using URL joining or client redirect defaults could escape the configured origin."""
    media = tmp_path / "redirect.jpg"
    media.write_bytes(b"image-private-marker")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            307,
            headers={"location": "https://attacker.example.test/steal"},
        )

    async with _client(handler, follow_redirects=True) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(
                client,
                _config(
                    "image_url_data_uri",
                    base_url="https://custom.example.test/prefix",
                    endpoint_path="/v1/responses",
                ),
            ).sense(
                _request(media, media_kind=MediaKind.IMAGE, mime_type="image/jpeg", modality=Modality.IMAGE)
            )

    assert caught.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert [str(request.url) for request in requests] == [
        "https://custom.example.test/prefix/v1/responses"
    ]


def test_constructor_requires_custom_adapter_base_endpoint_and_mode() -> None:
    """Defaulting any custom wire contract could send data to an unintended endpoint."""
    cases = [
        ProviderConfig(
            adapter="gemini",
            base_url="https://custom.example.test",
            model="custom-model",
            credential_ref="custom-provider-credential",
            declared_capabilities={Modality.IMAGE: True},
            adapter_options={
                "endpoint_path": "/v1/chat/completions",
                "media_part_mode": "image_url_data_uri",
            },
        ),
        ProviderConfig(
            adapter="openai-compatible",
            model="custom-model",
            credential_ref="custom-provider-credential",
            declared_capabilities={Modality.IMAGE: True},
            adapter_options={
                "endpoint_path": "/v1/chat/completions",
                "media_part_mode": "image_url_data_uri",
            },
        ),
        ProviderConfig(
            adapter="openai-compatible",
            base_url="https://custom.example.test",
            model="custom-model",
            credential_ref="custom-provider-credential",
            declared_capabilities={Modality.IMAGE: True},
            adapter_options={"media_part_mode": "image_url_data_uri"},
        ),
        ProviderConfig(
            adapter="openai-compatible",
            base_url="https://custom.example.test",
            model="custom-model",
            credential_ref="custom-provider-credential",
            declared_capabilities={Modality.IMAGE: True},
            adapter_options={"endpoint_path": "/v1/chat/completions"},
        ),
    ]

    for config in cases:
        with pytest.raises(SensoryError) as caught:
            OpenAICompatibleProvider(
                provider_id="custom",
                config=config,
                secret_store=_store(),
            )
        assert caught.value.code is ErrorCode.CONFIG_INVALID


@pytest.mark.asyncio
async def test_file_limit_is_enforced_before_read_and_accepted_file_is_read_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing stat/read bounds could allocate an unbounded base64 request body."""
    oversized = tmp_path / "oversized.jpg"
    oversized.write_bytes(b"12345")
    reads = 0
    real_read_bytes = Path.read_bytes

    def counted_read_bytes(path: Path) -> bytes:
        nonlocal reads
        reads += 1
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    async with _client(lambda request: httpx.Response(200, json=_chat_response(Modality.IMAGE))) as client:
        provider = _provider(client, _config("image_url_data_uri", max_bytes=4))
        with pytest.raises(SensoryError) as caught:
            await provider.sense(
                _request(oversized, media_kind=MediaKind.IMAGE, mime_type="image/jpeg", modality=Modality.IMAGE)
            )
    assert caught.value.code is ErrorCode.MEDIA_TOO_LARGE
    assert reads == 0

    accepted = tmp_path / "accepted.jpg"
    accepted.write_bytes(b"1234")
    async with _client(lambda request: httpx.Response(200, json=_chat_response(Modality.IMAGE))) as client:
        result = await _provider(client, _config("image_url_data_uri", max_bytes=4)).sense(
            _request(accepted, media_kind=MediaKind.IMAGE, mime_type="image/jpeg", modality=Modality.IMAGE)
        )
    assert result.observations[Modality.IMAGE].summary == "Direct evidence."
    assert reads == 1


class TrackingAsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.yielded = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.yielded += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class BlockingAsyncStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.gate = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        await self.gate.wait()
        yield b"unreachable"

    async def aclose(self) -> None:
        self.closed = True


class DelayedCloseAsyncStream(BlockingAsyncStream):
    def __init__(self) -> None:
        super().__init__()
        self.close_started = asyncio.Event()
        self.close_gate = asyncio.Event()
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        await self.close_gate.wait()
        self.closed = True


class FailingCloseAsyncStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.iterated = False
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.iterated = True
        yield self.body

    async def aclose(self) -> None:
        self.closed = True
        raise OSError("private-response-close-detail")


class RequestFailingCloseAsyncStream(FailingCloseAsyncStream):
    async def aclose(self) -> None:
        self.closed = True
        raise httpx.ReadError(
            "private-response-close-detail",
            request=httpx.Request("POST", "https://private.example.invalid"),
        )


class RejectIfIteratedStream(httpx.AsyncByteStream):
    def __init__(self, compressed: bytes) -> None:
        self.compressed = compressed
        self.iterated = False
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.iterated = True
        raise AssertionError("compressed response body must not be iterated")
        yield self.compressed  # pragma: no cover - keeps this an async generator

    async def aclose(self) -> None:
        self.closed = True


class RawOnlyResponse(httpx.Response):
    async def aiter_bytes(self, chunk_size: int | None = None) -> AsyncIterator[bytes]:
        del chunk_size
        raise AssertionError("adapter must use the raw response iterator")
        yield b"unreachable"  # pragma: no cover - keeps this an async generator


@pytest.mark.asyncio
async def test_identity_response_uses_raw_iterator_and_closes(tmp_path: Path) -> None:
    """Switching back to decoded iteration would reopen amplification before bounds."""
    media = tmp_path / "raw-only.jpg"
    media.write_bytes(b"image")
    body = json.dumps(_chat_response(Modality.IMAGE)).encode("utf-8")
    stream = TrackingAsyncStream([body])

    async with _client(
        lambda request: RawOnlyResponse(
            200,
            headers={"content-encoding": "identity"},
            stream=stream,
        )
    ) as client:
        result = await _provider(client, _config("image_url_data_uri")).sense(
            _request(
                media,
                media_kind=MediaKind.IMAGE,
                mime_type="image/jpeg",
                modality=Modality.IMAGE,
            )
        )

    assert result.observations[Modality.IMAGE].summary == "Direct evidence."
    assert stream.yielded == 1
    assert stream.closed is True


@pytest.mark.parametrize(
    ("content_encoding", "compressed"),
    [
        pytest.param(
            "gzip",
            gzip.compress(b"x" * (_MAX_RESPONSE_BYTES + 1)),
            id="gzip-amplification",
        ),
        pytest.param(
            "deflate",
            zlib.compress(b"x" * (_MAX_RESPONSE_BYTES + 1)),
            id="deflate-amplification",
        ),
    ],
)
@pytest.mark.asyncio
async def test_compressed_response_is_rejected_before_body_iteration(
    tmp_path: Path,
    content_encoding: str,
    compressed: bytes,
) -> None:
    """Automatic decompression must not allocate an expanded body before our cap."""
    media = tmp_path / "compressed.jpg"
    media.write_bytes(b"image")
    stream = RejectIfIteratedStream(compressed)

    async with _client(
        lambda request: httpx.Response(
            200,
            headers={"content-encoding": content_encoding},
            stream=stream,
        )
    ) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client, _config("image_url_data_uri")).sense(
                _request(
                    media,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modality=Modality.IMAGE,
                )
            )

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert caught.value.cause is None
    assert stream.iterated is False
    assert stream.closed is True


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param(
            [("content-encoding", "")],
            id="empty",
        ),
        pytest.param(
            [("content-encoding", "gzip, identity")],
            id="comma-list",
        ),
        pytest.param(
            [
                ("content-encoding", "identity"),
                ("content-encoding", "identity"),
            ],
            id="duplicate-identity",
        ),
    ],
)
@pytest.mark.asyncio
async def test_malformed_or_multiple_content_encoding_is_rejected_unread(
    tmp_path: Path,
    headers: list[tuple[str, str]],
) -> None:
    """Ambiguous encoding metadata must not select any decoder or body path."""
    media = tmp_path / "encoding.jpg"
    media.write_bytes(b"image")
    stream = RejectIfIteratedStream(b"private-compressed-marker")

    async with _client(
        lambda request: httpx.Response(200, headers=headers, stream=stream)
    ) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client, _config("image_url_data_uri")).sense(
                _request(
                    media,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modality=Modality.IMAGE,
                )
            )

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert "private-compressed-marker" not in str(caught.value)
    assert stream.iterated is False
    assert stream.closed is True


@pytest.mark.asyncio
async def test_response_bytes_are_stream_bounded_and_response_closes(tmp_path: Path) -> None:
    """An absent or lying Content-Length must not bypass the decoded response cap."""
    media = tmp_path / "bounded.jpg"
    media.write_bytes(b"image")
    stream = TrackingAsyncStream([b"x" * _MAX_RESPONSE_BYTES, b"overflow-tail"])

    async with _client(lambda request: httpx.Response(200, stream=stream)) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client, _config("image_url_data_uri")).sense(
                _request(media, media_kind=MediaKind.IMAGE, mime_type="image/jpeg", modality=Modality.IMAGE)
            )

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert stream.yielded == 2
    assert stream.closed is True


@pytest.mark.asyncio
async def test_streaming_cancellation_propagates_and_closes_response(tmp_path: Path) -> None:
    """Mapping cancellation as fallback-eligible failure could send media twice."""
    media = tmp_path / "cancel.jpg"
    media.write_bytes(b"image")
    stream = BlockingAsyncStream()

    async with _client(lambda request: httpx.Response(200, stream=stream)) as client:
        task = asyncio.create_task(
            _provider(client, _config("image_url_data_uri")).sense(
                _request(media, media_kind=MediaKind.IMAGE, mime_type="image/jpeg", modality=Modality.IMAGE)
            )
        )
        await stream.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert stream.closed is True


@pytest.mark.asyncio
async def test_response_close_survives_repeated_cancellation(tmp_path: Path) -> None:
    """Cancellation must wait for the independently owned response close task."""
    media = tmp_path / "repeated-cancel.jpg"
    media.write_bytes(b"image")
    stream = DelayedCloseAsyncStream()

    async with _client(lambda request: httpx.Response(200, stream=stream)) as client:
        task = asyncio.create_task(
            _provider(client, _config("image_url_data_uri")).sense(
                _request(
                    media,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modality=Modality.IMAGE,
                )
            )
        )
        await stream.started.wait()
        task.cancel()
        await stream.close_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        done_before_close = task.done()
        stream.close_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert done_before_close is False
    assert stream.close_calls == 1
    assert stream.closed is True


@pytest.mark.asyncio
async def test_cancellation_during_close_precedes_primary_response_error(
    tmp_path: Path,
) -> None:
    """A caller cancellation during cleanup must win after cleanup reaches terminal."""
    media = tmp_path / "primary-error-cancel.jpg"
    media.write_bytes(b"image")
    stream = DelayedCloseAsyncStream()
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop_contexts: list[dict[str, object]] = []
    loop.set_exception_handler(
        lambda active_loop, context: loop_contexts.append(context)
    )

    try:
        async with _client(
            lambda request: httpx.Response(
                200,
                headers={"content-encoding": "gzip"},
                stream=stream,
            )
        ) as client:
            task = asyncio.create_task(
                _provider(client, _config("image_url_data_uri")).sense(
                    _request(
                        media,
                        media_kind=MediaKind.IMAGE,
                        mime_type="image/jpeg",
                        modality=Modality.IMAGE,
                    )
                )
            )
            await stream.close_started.wait()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            done_before_close = task.done()
            stream.close_gate.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert done_before_close is False
    assert stream.close_calls == 1
    assert stream.closed is True
    assert loop_contexts == []


@pytest.mark.asyncio
async def test_response_close_failure_is_reported_without_masking_primary_error(
    tmp_path: Path,
) -> None:
    """Close failures matter on success but never replace an earlier response error."""
    media = tmp_path / "close-failure.jpg"
    media.write_bytes(b"image")
    valid_body = json.dumps(_chat_response(Modality.IMAGE)).encode("utf-8")
    success_stream = FailingCloseAsyncStream(valid_body)

    async with _client(
        lambda request: httpx.Response(200, stream=success_stream)
    ) as client:
        with pytest.raises(SensoryError) as close_caught:
            await _provider(client, _config("image_url_data_uri")).sense(
                _request(
                    media,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modality=Modality.IMAGE,
                )
            )

    assert close_caught.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert "private-response-close-detail" not in str(close_caught.value)
    assert success_stream.closed is True

    primary_stream = FailingCloseAsyncStream(b"private-compressed-marker")
    async with _client(
        lambda request: httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=primary_stream,
        )
    ) as client:
        with pytest.raises(SensoryError) as primary_caught:
            await _provider(client, _config("image_url_data_uri")).sense(
                _request(
                    media,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modality=Modality.IMAGE,
                )
            )

    assert primary_caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert "private-" not in str(primary_caught.value)
    assert primary_stream.iterated is False
    assert primary_stream.closed is True


@pytest.mark.parametrize(
    "content_lengths",
    [
        pytest.param(["9" * 100_000], id="huge-digits"),
        pytest.param(["8388608x"], id="malformed"),
        pytest.param(["1", "1"], id="duplicate"),
        pytest.param([str(_MAX_RESPONSE_BYTES + 1)], id="ordinary-over-cap"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_content_length_is_rejected_unread_and_closed(
    tmp_path: Path,
    content_lengths: list[str],
) -> None:
    """Even enormous numeric metadata must take the stable bounded rejection path."""
    media = tmp_path / "content-length.jpg"
    media.write_bytes(b"image")
    stream = RejectIfIteratedStream(b"private-response-marker")
    headers = [("content-length", value) for value in content_lengths]

    async with _client(
        lambda request: httpx.Response(200, headers=headers, stream=stream)
    ) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client, _config("image_url_data_uri")).sense(
                _request(
                    media,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modality=Modality.IMAGE,
                )
            )

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert caught.value.retryable is False
    assert caught.value.cause is None
    assert "private-response-marker" not in str(caught.value)
    assert stream.iterated is False
    assert stream.closed is True


@pytest.mark.parametrize(
    ("status", "body", "want_code", "want_retryable"),
    [
        pytest.param(401, {"error": {"code": "invalid_api_key"}}, ErrorCode.PROVIDER_AUTH_FAILED, False, id="auth"),
        pytest.param(429, {"error": {"code": "rate_limit_exceeded"}}, ErrorCode.PROVIDER_UNAVAILABLE, True, id="rate"),
        pytest.param(400, {"error": {"code": "content_filter"}}, ErrorCode.PROVIDER_SAFETY_REJECTED, False, id="safety"),
        pytest.param(415, {"error": {"code": "unsupported_media_type"}}, ErrorCode.PROVIDER_CAPABILITY_REJECTED, False, id="media"),
        pytest.param(500, b"private-provider-body", ErrorCode.PROVIDER_UNAVAILABLE, True, id="server-malformed"),
        pytest.param(200, b"private-provider-body", ErrorCode.PROVIDER_CAPABILITY_REJECTED, False, id="success-malformed"),
    ],
)
@pytest.mark.asyncio
async def test_http_and_malformed_failures_use_stable_private_errors(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    body: dict[str, object] | bytes,
    want_code: ErrorCode,
    want_retryable: bool,
) -> None:
    """Returning raw compatible-API errors would expose arbitrary Provider content."""
    monkeypatch.setattr(compatible_module, "_RETRY_BACKOFF_START", 0)
    media = tmp_path / "failure.jpg"
    media.write_bytes(b"private-media-marker")
    caplog.set_level(logging.DEBUG)
    response = httpx.Response(status, json=body) if isinstance(body, dict) else httpx.Response(status, content=body)

    async with _client(lambda request: response) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client, _config("image_url_data_uri")).sense(
                _request(media, media_kind=MediaKind.IMAGE, mime_type="image/jpeg", modality=Modality.IMAGE)
            )

    assert caught.value.code is want_code
    assert caught.value.retryable is want_retryable
    assert caught.value.cause is None
    diagnostics = f"{caught.value}\n" + "\n".join(record.getMessage() for record in caplog.records)
    for marker in ("private-provider-body", "private-media-marker", str(media)):
        assert marker not in diagnostics


@pytest.mark.parametrize(
    ("failure", "want_code"),
    [
        pytest.param(httpx.ReadTimeout("private-timeout-marker"), ErrorCode.PROVIDER_TIMEOUT, id="timeout"),
        pytest.param(httpx.ConnectError("private-network-marker"), ErrorCode.PROVIDER_UNAVAILABLE, id="network"),
        pytest.param(RuntimeError("private-runtime-marker"), ErrorCode.PROVIDER_UNAVAILABLE, id="unexpected"),
    ],
)
@pytest.mark.asyncio
async def test_transport_failures_are_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    want_code: ErrorCode,
) -> None:
    """Chaining transport diagnostics could expose headers, URLs, or local paths."""
    monkeypatch.setattr(compatible_module, "_RETRY_BACKOFF_START", 0)
    media = tmp_path / "transport.jpg"
    media.write_bytes(b"image")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise failure

    async with _client(handler) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client, _config("image_url_data_uri")).sense(
                _request(media, media_kind=MediaKind.IMAGE, mime_type="image/jpeg", modality=Modality.IMAGE)
            )

    assert caught.value.code is want_code
    assert calls == (1 if isinstance(failure, RuntimeError) else 2)
    assert caught.value.cause is None
    assert "private-" not in str(caught.value)


@pytest.mark.parametrize(
    ("error_code", "want_code", "want_retryable"),
    [
        pytest.param(
            "content_filter",
            ErrorCode.PROVIDER_SAFETY_REJECTED,
            False,
            id="safety",
        ),
        pytest.param(
            "unsupported_media_type",
            ErrorCode.PROVIDER_CAPABILITY_REJECTED,
            False,
            id="capability",
        ),
        pytest.param(
            "invalid_api_key",
            ErrorCode.PROVIDER_AUTH_FAILED,
            False,
            id="auth",
        ),
        pytest.param(
            "rate_limit_exceeded",
            ErrorCode.PROVIDER_UNAVAILABLE,
            True,
            id="rate",
        ),
    ],
)
@pytest.mark.asyncio
async def test_success_status_error_envelope_precedes_valid_report(
    tmp_path: Path,
    error_code: str,
    want_code: ErrorCode,
    want_retryable: bool,
) -> None:
    """A 2xx status must not let a Provider error masquerade as an observation."""
    media = tmp_path / "error-envelope.jpg"
    media.write_bytes(b"image")
    payload = _chat_response(Modality.IMAGE, summary="must be ignored")
    payload["error"] = {
        "code": error_code,
        "message": "private-recognized-error-detail",
    }

    async with _client(lambda request: httpx.Response(200, json=payload)) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client, _config("image_url_data_uri")).sense(
                _request(
                    media,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modality=Modality.IMAGE,
                )
            )

    assert caught.value.code is want_code
    assert caught.value.retryable is want_retryable
    assert caught.value.cause is None
    assert "must be ignored" not in str(caught.value)
    assert "private-recognized-error-detail" not in str(caught.value)


@pytest.mark.parametrize(
    "error_envelope",
    [
        pytest.param(
            {
                "code": "private-unknown-error-code",
                "message": "private-unknown-error-detail",
            },
            id="unknown-object",
        ),
        pytest.param("private-malformed-error-string", id="string"),
        pytest.param(["private-malformed-error-list"], id="list"),
    ],
)
@pytest.mark.asyncio
async def test_unknown_or_malformed_success_error_envelope_rejects_privately(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    error_envelope: object,
) -> None:
    """Malformed non-null errors must reject instead of falling through to choices."""
    media = tmp_path / "private-error.jpg"
    media.write_bytes(b"image")
    payload = _chat_response(Modality.IMAGE, summary="must not be accepted")
    payload["error"] = error_envelope
    caplog.set_level(logging.DEBUG)

    async with _client(lambda request: httpx.Response(200, json=payload)) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client, _config("image_url_data_uri")).sense(
                _request(
                    media,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modality=Modality.IMAGE,
                )
            )

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert caught.value.retryable is False
    assert caught.value.cause is None
    diagnostics = f"{caught.value}\n" + "\n".join(
        record.getMessage() for record in caplog.records
    )
    assert "private-" not in diagnostics
    assert "must not be accepted" not in diagnostics


@pytest.mark.asyncio
async def test_null_error_envelope_does_not_replace_valid_success(tmp_path: Path) -> None:
    """Only a present non-null error envelope changes compatible success handling."""
    media = tmp_path / "null-error.jpg"
    media.write_bytes(b"image")
    payload = _chat_response(Modality.IMAGE)
    payload["error"] = None

    async with _client(lambda request: httpx.Response(200, json=payload)) as client:
        result = await _provider(client, _config("image_url_data_uri")).sense(
            _request(
                media,
                media_kind=MediaKind.IMAGE,
                mime_type="image/jpeg",
                modality=Modality.IMAGE,
            )
        )

    assert result.observations[Modality.IMAGE].summary == "Direct evidence."


@pytest.mark.asyncio
async def test_only_documented_text_blocks_are_extracted_not_thinking(tmp_path: Path) -> None:
    """Recursive text harvesting could expose hidden reasoning as a sensory report."""
    media = tmp_path / "thinking.jpg"
    media.write_bytes(b"image")
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "reasoning", "text": _report_text(Modality.IMAGE, summary="private thinking")},
                        {"type": "text", "text": _report_text(Modality.IMAGE, summary="visible observation")},
                    ]
                }
            }
        ]
    }

    async with _client(lambda request: httpx.Response(200, json=payload)) as client:
        result = await _provider(client, _config("image_url_data_uri")).sense(
            _request(media, media_kind=MediaKind.IMAGE, mime_type="image/jpeg", modality=Modality.IMAGE)
        )

    assert result.observations[Modality.IMAGE].summary == "visible observation"


@pytest.mark.asyncio
async def test_present_oversized_choices_cannot_fall_through_to_another_envelope(
    tmp_path: Path,
) -> None:
    """Falling through a malformed named envelope would make its content bound optional."""
    media = tmp_path / "ambiguous.jpg"
    media.write_bytes(b"image")
    payload = {
        "choices": [{}] * 65,
        "content": [
            {"type": "text", "text": _report_text(Modality.IMAGE)}
        ],
    }

    async with _client(lambda request: httpx.Response(200, json=payload)) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client, _config("image_url_data_uri")).sense(
                _request(media, media_kind=MediaKind.IMAGE, mime_type="image/jpeg", modality=Modality.IMAGE)
            )

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED


@pytest.mark.asyncio
async def test_deep_report_json_is_mapped_without_raw_recursion_error(tmp_path: Path) -> None:
    """A bounded text block must not bypass the stable report-parse boundary by depth."""
    media = tmp_path / "deep.jpg"
    media.write_bytes(b"image")
    deep = '{"nested":' + "[" * 10_000 + '"private-deep-marker"' + "]" * 10_000 + "}"
    payload = {"choices": [{"message": {"content": deep}}]}

    async with _client(lambda request: httpx.Response(200, json=payload)) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client, _config("image_url_data_uri")).sense(
                _request(media, media_kind=MediaKind.IMAGE, mime_type="image/jpeg", modality=Modality.IMAGE)
            )

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert caught.value.cause is None
    assert "RecursionError" not in str(caught.value)
    assert "private-deep-marker" not in str(caught.value)


@pytest.mark.asyncio
async def test_borrowed_client_remains_open() -> None:
    """An injected client and its caller-controlled transport policy remain borrowed."""
    borrowed = _client(lambda request: httpx.Response(500))
    provider = _provider(borrowed, _config("image_url_data_uri"))
    await provider.aclose()
    assert borrowed.is_closed is False
    await borrowed.aclose()


@pytest.mark.parametrize(
    ("status", "body"),
    [
        pytest.param(408, b"request-timeout", id="408-text"),
        pytest.param(
            429,
            {"error": {"code": "rate_limit_exceeded", "message": "private"}},
            id="429-json",
        ),
        pytest.param(
            500,
            {"error": {"code": "internal_server_error", "message": "private"}},
            id="500-json",
        ),
        pytest.param(502, b"bad-gateway", id="502-text"),
        pytest.param(
            503,
            {"error": {"code": "internal_server_error", "message": "private"}},
            id="503-json",
        ),
        pytest.param(504, b"gateway-timeout", id="504-text"),
    ],
)
@pytest.mark.asyncio
async def test_transient_status_retries_once_then_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    body: dict[str, object] | bytes,
) -> None:
    """One bounded retry must cover both text and ordinary JSON error responses."""
    monkeypatch.setattr(compatible_module, "_RETRY_BACKOFF_START", 0)
    media = tmp_path / "retry.jpg"
    media.write_bytes(b"media-bytes")
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            if isinstance(body, dict):
                return httpx.Response(status, json=body)
            return httpx.Response(status, content=body)
        return httpx.Response(200, json=_chat_response(Modality.IMAGE))

    async with _client(handler) as client:
        result = await _provider(client, _config("image_url_data_uri")).sense(
            _request(media, media_kind=MediaKind.IMAGE, mime_type="image/jpeg", modality=Modality.IMAGE)
        )

    assert calls["count"] == 2
    assert result.observations[Modality.IMAGE].summary == "Direct evidence."


@pytest.mark.asyncio
async def test_transient_status_exhausts_one_retry_then_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent JSON 5xx must stop after one retry with a private error."""
    monkeypatch.setattr(compatible_module, "_RETRY_BACKOFF_START", 0)
    media = tmp_path / "retry-fail.jpg"
    media.write_bytes(b"media-bytes")
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(
            503,
            json={
                "error": {
                    "code": "internal_server_error",
                    "message": "private-still-cold",
                }
            },
        )

    async with _client(handler) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client, _config("image_url_data_uri")).sense(
                _request(media, media_kind=MediaKind.IMAGE, mime_type="image/jpeg", modality=Modality.IMAGE)
            )

    assert calls["count"] == 2
    assert caught.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert caught.value.retryable is True
    assert "private-still-cold" not in str(caught.value)


@pytest.mark.parametrize(
    ("error_code", "want_code"),
    [
        pytest.param("invalid_api_key", ErrorCode.PROVIDER_AUTH_FAILED, id="auth"),
        pytest.param(
            "content_filter",
            ErrorCode.PROVIDER_SAFETY_REJECTED,
            id="safety",
        ),
        pytest.param(
            "unsupported_media_type",
            ErrorCode.PROVIDER_CAPABILITY_REJECTED,
            id="capability",
        ),
    ],
)
@pytest.mark.asyncio
async def test_terminal_json_error_on_transient_status_is_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_code: str,
    want_code: ErrorCode,
) -> None:
    """A contradictory 503 must not override an explicit terminal Provider error."""
    monkeypatch.setattr(compatible_module, "_RETRY_BACKOFF_START", 0)
    media = tmp_path / "terminal.jpg"
    media.write_bytes(b"media")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": {"code": error_code}})

    async with _client(handler) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client, _config("image_url_data_uri")).sense(
                _request(
                    media,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modality=Modality.IMAGE,
                )
            )

    assert calls == 1
    assert caught.value.code is want_code


@pytest.mark.asyncio
async def test_retry_backoff_is_inside_the_overall_request_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short configured deadline must expire before a delayed second call starts."""
    monkeypatch.setattr(compatible_module, "_RETRY_BACKOFF_START", 0.05)
    media = tmp_path / "deadline.jpg"
    media.write_bytes(b"media")
    calls = 0
    config = _config("image_url_data_uri")
    options = config.adapter_options.model_copy(
        update={"request_timeout_seconds": 0.01}
    )
    config = config.model_copy(update={"adapter_options": options})

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, content=b"private-cold-start")

    async with _client(handler) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client, config).sense(
                _request(
                    media,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modality=Modality.IMAGE,
                )
            )

    assert calls == 1
    assert caught.value.code is ErrorCode.PROVIDER_TIMEOUT
    assert "private-cold-start" not in str(caught.value)


@pytest.mark.asyncio
async def test_cancellation_during_retry_backoff_starts_no_second_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller cancellation must interrupt backoff without resending private media."""
    monkeypatch.setattr(compatible_module, "_RETRY_BACKOFF_START", 60)
    media = tmp_path / "cancel-backoff.jpg"
    media.write_bytes(b"media")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, content=b"cold-start")

    async with _client(handler) as client:
        task = asyncio.create_task(
            _provider(client, _config("image_url_data_uri")).sense(
                _request(
                    media,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modality=Modality.IMAGE,
                )
            )
        )
        while calls == 0:
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert calls == 1


@pytest.mark.asyncio
async def test_response_close_request_error_does_not_repeat_paid_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response already received must not be resent only because close failed."""
    monkeypatch.setattr(compatible_module, "_RETRY_BACKOFF_START", 0)
    media = tmp_path / "close-request-error.jpg"
    media.write_bytes(b"media")
    calls = 0
    streams: list[RequestFailingCloseAsyncStream] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        stream = RequestFailingCloseAsyncStream(
            json.dumps(_chat_response(Modality.IMAGE)).encode("utf-8")
        )
        streams.append(stream)
        return httpx.Response(200, stream=stream)

    async with _client(handler) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client, _config("image_url_data_uri")).sense(
                _request(
                    media,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modality=Modality.IMAGE,
                )
            )

    assert calls == 1
    assert caught.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert all(stream.closed for stream in streams)
    assert "private-" not in str(caught.value)


@pytest.mark.asyncio
async def test_malformed_music_json_retries_once_with_correction_prompt(
    tmp_path: Path,
) -> None:
    """A long music report may retry formatting once without extra transport attempts."""
    media = tmp_path / "事件视界.MP3"
    media.write_bytes(b"audio")
    request_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_payloads.append(json.loads(request.content))
        if len(request_payloads) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": '{"observations":[{"modality":"music"'
                            }
                        }
                    ]
                },
            )
        return httpx.Response(200, json=_chat_response(Modality.MUSIC))

    async with _client(handler) as client:
        result = await _provider(client, _config("audio_url_data_uri")).sense(
            _request(
                media,
                media_kind=MediaKind.AUDIO,
                mime_type="audio/mpeg",
                modality=Modality.MUSIC,
            )
        )

    assert len(request_payloads) == 2
    first_prompt = request_payloads[0]["messages"][0]["content"][1]["text"]
    retry_prompt = request_payloads[1]["messages"][0]["content"][1]["text"]
    assert "FORMAT RETRY" not in first_prompt
    assert "FORMAT RETRY" in retry_prompt
    assert result.observations[Modality.MUSIC].summary == "Direct evidence."


@pytest.mark.asyncio
async def test_plain_string_lyrics_trigger_targeted_format_retry(
    tmp_path: Path,
) -> None:
    """A rejected string array must retry with the exact missing object contract."""
    media = tmp_path / "structured-lyrics.mp3"
    media.write_bytes(b"audio")
    request_payloads: list[dict[str, Any]] = []
    invalid_report = {
        "observations": [
            {
                **_observation(Modality.MUSIC),
                "transcript": ["private lyric one", "private lyric two"],
            }
        ]
    }
    valid_report = {
        "observations": [
            {
                **_observation(Modality.MUSIC),
                "transcript": [
                    {
                        "start_seconds": 0,
                        "end_seconds": 1,
                        "text": "structured lyric",
                    }
                ],
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        request_payloads.append(json.loads(request.content))
        report = invalid_report if len(request_payloads) == 1 else valid_report
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(report)}}
                ]
            },
        )

    async with _client(handler) as client:
        result = await _provider(client, _config("audio_url_data_uri")).sense(
            _request(
                media,
                media_kind=MediaKind.AUDIO,
                mime_type="audio/mpeg",
                modality=Modality.MUSIC,
            )
        )

    assert len(request_payloads) == 2
    first_prompt = request_payloads[0]["messages"][0]["content"][1]["text"]
    retry_prompt = request_payloads[1]["messages"][0]["content"][1]["text"]
    assert "FORMAT RETRY" not in first_prompt
    assert "FORMAT RETRY" in retry_prompt
    assert "JSON object" in retry_prompt
    assert "start_seconds" in retry_prompt
    assert "Never output a plain string" in retry_prompt
    observation = result.observations[Modality.MUSIC]
    assert [item.text for item in observation.transcript] == ["structured lyric"]
    assert observation.transcript[0].start_seconds == 0
    assert observation.transcript[0].end_seconds == 1


@pytest.mark.asyncio
async def test_format_and_transport_retries_share_one_two_call_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient first call leaves no third paid call for format correction."""
    monkeypatch.setattr(compatible_module, "_RETRY_BACKOFF_START", 0)
    media = tmp_path / "bounded-budget.mp3"
    media.write_bytes(b"audio")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, content=b"cold-start")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"observations":[{"modality":"music"'
                        }
                    }
                ]
            },
        )

    async with _client(handler) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client, _config("audio_url_data_uri")).sense(
                _request(
                    media,
                    media_kind=MediaKind.AUDIO,
                    mime_type="audio/mpeg",
                    modality=Modality.MUSIC,
                )
            )

    assert calls == 2
    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED


class DelayedCloseClient:
    def __init__(self) -> None:
        self.is_closed = False
        self.close_started = asyncio.Event()
        self.close_gate = asyncio.Event()
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        await self.close_gate.wait()
        self.is_closed = True


@pytest.mark.asyncio
async def test_owned_client_close_survives_repeated_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider must not return until its owned transport has closed."""
    client = DelayedCloseClient()
    monkeypatch.setattr(
        compatible_module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: cast(Any, client),
    )
    provider = OpenAICompatibleProvider(
        provider_id="custom",
        config=_config("image_url_data_uri"),
        secret_store=_store(),
    )

    task = asyncio.create_task(provider.aclose())
    await client.close_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    done_before_close = task.done()
    client.close_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert done_before_close is False
    assert client.close_calls == 1
    assert client.is_closed is True


@pytest.mark.asyncio
async def test_owned_client_ignores_ambient_proxies_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambient proxies must never receive Provider credentials, headers, or media."""
    media = tmp_path / "owned.jpg"
    media.write_bytes(b"owned-media-secret")
    proxy_marker = "ambient-proxy-secret"
    custom_header = "custom-header-secret"
    monkeypatch.setenv(
        "HTTPS_PROXY",
        f"https://proxy-user:{proxy_marker}@proxy.example.test:8443",
    )
    monkeypatch.setenv(
        "ALL_PROXY",
        f"socks5://proxy-user:{proxy_marker}@proxy.example.test:1080",
    )
    monkeypatch.setenv(_EXTRA_HEADER_ENV, custom_header)
    direct_requests: list[httpx.Request] = []
    constructor_kwargs: list[dict[str, object]] = []
    real_async_client = httpx.AsyncClient

    def direct_handler(request: httpx.Request) -> httpx.Response:
        direct_requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=StaticAsyncStream(
                json.dumps(_chat_response(Modality.IMAGE)).encode("utf-8")
            ),
        )

    def client_factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        constructor_kwargs.append(dict(kwargs))
        if kwargs.get("trust_env") is not False:
            raise RuntimeError("owned client inherited ambient proxy settings")
        return real_async_client(
            *args,
            transport=httpx.MockTransport(direct_handler),
            **kwargs,
        )

    monkeypatch.setattr(compatible_module.httpx, "AsyncClient", client_factory)
    owned = OpenAICompatibleProvider(
        provider_id="custom",
        config=_config(
            "image_url_data_uri",
            extra_headers_env={"X-Tenant-Token": _EXTRA_HEADER_ENV},
        ),
        secret_store=_store(),
    )
    result = await owned.sense(
        _request(
            media,
            media_kind=MediaKind.IMAGE,
            mime_type="image/jpeg",
            modality=Modality.IMAGE,
        )
    )
    await owned.aclose()
    await owned.aclose()

    assert result.observations[Modality.IMAGE].summary == "Direct evidence."
    assert constructor_kwargs == [{"follow_redirects": False, "trust_env": False}]
    assert len(direct_requests) == 1
    assert direct_requests[0].headers["authorization"] == f"Bearer {_PRIMARY_SECRET}"
    assert direct_requests[0].headers["x-tenant-token"] == custom_header
    assert base64.b64encode(b"owned-media-secret") in direct_requests[0].content
    assert cast(Any, owned)._client.is_closed is True
