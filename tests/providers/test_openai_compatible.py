from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import yaml
from pydantic import ValidationError

from cove_sensory_mcp.config.schema import AdapterOptions, AppConfig, ProviderConfig
from cove_sensory_mcp.config.secrets import MemorySecretStore
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import DetailLevel, Modality
from cove_sensory_mcp.providers.base import MediaKind, PreparedMedia, ProviderRequest
from cove_sensory_mcp.providers.openai_compatible import OpenAICompatibleProvider

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
            "input_audio_base64": frozenset({Modality.AUDIO, Modality.MUSIC}),
            "video_url_data_uri": frozenset({Modality.VIDEO_VISUAL}),
            "anthropic_base64_media": frozenset({Modality.IMAGE}),
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


def _client(
    handler: Callable[[httpx.Request], object],
    *,
    follow_redirects: bool = False,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(cast(Any, handler)),
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
    return "\n".join(
        [
            "Return only valid JSON matching the ProviderObservationBatch contract.",
            'The top-level object has exactly one key, "observations".',
            f'The observations array must contain modalities exactly: ["{modality.value}"].',
            "Return one observation per requested modality, with no duplicates or extra modalities.",
            (
                "Each observation has exactly these fields: modality, summary, segments, transcript, "
                "warnings, confidence."
            ),
            (
                "Each segment and transcript item has start_seconds, end_seconds, and text. Each "
                "warning has code and message. confidence is low, medium, or high."
            ),
            (
                "Report direct observations and concrete evidence. State uncertainty explicitly when "
                "the media does not support a confident observation."
            ),
            (
                "Do not invent identities, diagnoses, causal claims, events, dialogue, or lyrics not "
                "directly supported by the media."
            ),
            (
                'REQUEST_SCOPE: {"detail":"quick","language":"en","start_seconds":null,'
                '"end_seconds":null}'
            ),
        ]
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
    status: int,
    body: dict[str, object] | bytes,
    want_code: ErrorCode,
    want_retryable: bool,
) -> None:
    """Returning raw compatible-API errors would expose arbitrary Provider content."""
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
    failure: BaseException,
    want_code: ErrorCode,
) -> None:
    """Chaining transport diagnostics could expose headers, URLs, or local paths."""
    media = tmp_path / "transport.jpg"
    media.write_bytes(b"image")

    def handler(request: httpx.Request) -> httpx.Response:
        raise failure

    async with _client(handler) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client, _config("image_url_data_uri")).sense(
                _request(media, media_kind=MediaKind.IMAGE, mime_type="image/jpeg", modality=Modality.IMAGE)
            )

    assert caught.value.code is want_code
    assert caught.value.cause is None
    assert "private-" not in str(caught.value)


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
async def test_borrowed_client_remains_open_and_owned_client_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing borrowed clients breaks composition; leaking owned clients leaks sockets."""
    borrowed = _client(lambda request: httpx.Response(500))
    provider = _provider(borrowed, _config("image_url_data_uri"))
    await provider.aclose()
    assert borrowed.is_closed is False
    await borrowed.aclose()

    for environment_name in (
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "all_proxy",
        "https_proxy",
        "http_proxy",
    ):
        monkeypatch.delenv(environment_name, raising=False)
    owned = OpenAICompatibleProvider(
        provider_id="custom",
        config=_config("image_url_data_uri"),
        secret_store=_store(),
    )
    await owned.aclose()
    await owned.aclose()
    assert cast(Any, owned)._client.is_closed is True
