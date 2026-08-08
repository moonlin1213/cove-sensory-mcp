from __future__ import annotations

import asyncio
import base64
import gzip
import inspect
import json
import logging
import zlib
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import ValidationError

from cove_sensory_mcp.config.schema import AdapterOptions, ProviderConfig
from cove_sensory_mcp.config.secrets import MemorySecretStore
from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import DetailLevel, Modality
from cove_sensory_mcp.providers.base import MediaKind, PreparedMedia, ProviderRequest
from cove_sensory_mcp.providers.minimax_m3 import (
    MINIMAX_CN_BASE_URL,
    MINIMAX_GLOBAL_BASE_URL,
    MiniMaxAuthMode,
    MiniMaxM3Provider,
    MiniMaxRegion,
)

_SECRET = "test-minimax-secret-that-must-stay-private"
_MAX_RESPONSE_BYTES_TEST = 8 * 1024 * 1024
_MAX_CONTENT_BLOCKS_TEST = 64
_MAX_TEXT_BLOCK_CHARS_TEST = 2_000_000
_MAX_COMBINED_TEXT_CHARS_TEST = 2_000_000


def _observation(modality: Modality, *, summary: str = "Direct evidence.") -> dict[str, object]:
    return {
        "modality": modality.value,
        "summary": summary,
        "segments": [],
        "transcript": [],
        "warnings": [],
        "confidence": "medium",
    }


def _provider_response(modality: Modality, *, summary: str = "Direct evidence.") -> dict[str, object]:
    return {
        "id": "provider-response-id",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {"observations": [_observation(modality, summary=summary)]}
                ),
            }
        ],
        "model": "MiniMax-M3",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 18},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }


def _response_with_base_code(
    modality: Modality,
    status_code: object,
    *,
    status_msg: str = "private-provider-status-marker",
) -> dict[str, object]:
    payload = _provider_response(modality)
    payload["base_resp"] = {
        "status_code": status_code,
        "status_msg": status_msg,
    }
    return payload


class TrackingAsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.yielded_chunks = 0
        self.iterated = False
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.iterated = True
        for chunk in self._chunks:
            self.yielded_chunks += 1
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
        yield b"unreachable-after-cancel"

    async def aclose(self) -> None:
        self.closed = True


def _response_bytes(modality: Modality = Modality.IMAGE) -> bytes:
    return json.dumps(_provider_response(modality)).encode("utf-8")


def _config(
    *,
    base_url: str | None = None,
    model: str = "MiniMax-M3",
    max_bytes: int = 1_024,
    max_output_tokens: int = 768,
    temperature: float = 0.2,
    timeout: float = 9.0,
) -> ProviderConfig:
    return ProviderConfig(
        adapter="minimax-m3",
        base_url=base_url,
        model=model,
        credential_ref="minimax-test-credential",
        declared_capabilities={
            Modality.IMAGE: True,
            Modality.VIDEO_VISUAL: True,
        },
        verified_capabilities={
            Modality.IMAGE: True,
            Modality.VIDEO_VISUAL: True,
        },
        adapter_options=AdapterOptions(
            inline_max_bytes=max_bytes,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            request_timeout_seconds=timeout,
        ),
    )


def _request(
    path: Path,
    *,
    media_kind: MediaKind,
    mime_type: str,
    modalities: frozenset[Modality],
    duration_seconds: float | None = None,
) -> ProviderRequest:
    return ProviderRequest(
        media=PreparedMedia(
            path=path,
            mime_type=mime_type,
            media_kind=media_kind,
            duration_seconds=duration_seconds,
        ),
        requested_modalities=modalities,
        question="Focus on direct visual evidence.",
        detail=DetailLevel.DETAILED,
        language="en",
        start_seconds=0,
        end_seconds=duration_seconds,
    )


def _store() -> MemorySecretStore:
    store = MemorySecretStore()
    store.set("minimax-test-credential", _SECRET)
    return store


def _client(
    handler: Callable[[httpx.Request], object],
    *,
    follow_redirects: bool = False,
) -> httpx.AsyncClient:
    async def streaming_handler(request: httpx.Request) -> object:
        response = handler(request)
        if inspect.isawaitable(response):
            response = await response
        if isinstance(response, httpx.Response) and response.is_stream_consumed:
            return httpx.Response(
                response.status_code,
                headers=response.headers.raw,
                stream=TrackingAsyncStream([response.content]),
                extensions=response.extensions,
            )
        return response

    return httpx.AsyncClient(
        transport=httpx.MockTransport(cast(Any, streaming_handler)),
        follow_redirects=follow_redirects,
    )


def _provider(
    client: httpx.AsyncClient,
    *,
    config: ProviderConfig | None = None,
    region: MiniMaxRegion = MiniMaxRegion.GLOBAL,
    auth_mode: MiniMaxAuthMode = MiniMaxAuthMode.X_API_KEY,
    store: MemorySecretStore | None = None,
) -> MiniMaxM3Provider:
    return MiniMaxM3Provider(
        provider_id="minimax",
        config=config or _config(),
        secret_store=store or _store(),
        client=client,
        region=region,
        auth_mode=auth_mode,
    )


@pytest.mark.asyncio
async def test_image_uses_exact_anthropic_base64_media_payload(tmp_path: Path) -> None:
    """Changing the media block shape would make MiniMax treat the image as text."""
    image_bytes = b"encoded-test-bytes"
    image = tmp_path / "scene.jpg"
    image.write_bytes(image_bytes)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_provider_response(Modality.IMAGE))

    async with _client(handler) as client:
        result = await _provider(client).sense(
            _request(
                image,
                media_kind=MediaKind.IMAGE,
                mime_type="image/jpeg",
                modalities=frozenset({Modality.IMAGE}),
            )
        )

    payload = json.loads(captured[0].content)
    assert payload == {
        "model": "MiniMax-M3",
        "max_tokens": 768,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "ZW5jb2RlZC10ZXN0LWJ5dGVz",
                        },
                    },
                    {
                        "type": "text",
                        "text": cast(str, payload["messages"][0]["content"][1]["text"]),
                    },
                ],
            }
        ],
    }
    prompt = payload["messages"][0]["content"][1]["text"]
    assert 'exactly: ["image"]' in prompt
    assert captured[0].headers["accept-encoding"] == "identity"
    assert result.provider_id == "minimax"
    assert result.model == "MiniMax-M3"
    assert result.remote_file_deleted is None
    assert set(result.observations) == {Modality.IMAGE}
    assert result.observations[Modality.IMAGE].summary == "Direct evidence."


@pytest.mark.asyncio
async def test_video_uses_one_native_video_block_without_frame_sampling(tmp_path: Path) -> None:
    """Replacing native video with frame images would lose temporal evidence."""
    video_bytes = b"native-video-bytes"
    video = tmp_path / "clip.mp4"
    video.write_bytes(video_bytes)
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=_provider_response(Modality.VIDEO_VISUAL))

    async with _client(handler) as client:
        await _provider(client).sense(
            _request(
                video,
                media_kind=MediaKind.VIDEO,
                mime_type="video/mp4",
                modalities=frozenset({Modality.VIDEO_VISUAL}),
                duration_seconds=4,
            )
        )

    content = payloads[0]["messages"][0]["content"]
    assert content[0] == {
        "type": "video",
        "source": {
            "type": "base64",
            "media_type": "video/mp4",
            "data": base64.b64encode(video_bytes).decode("ascii"),
        },
    }
    assert [part["type"] for part in content] == ["video", "text"]
    assert len(payloads) == 1


@pytest.mark.parametrize(
    ("media_kind", "mime_type", "modalities"),
    [
        pytest.param(
            MediaKind.VIDEO,
            "video/mp4",
            frozenset({Modality.VIDEO_AUDIO}),
            id="video-audio",
        ),
        pytest.param(
            MediaKind.AUDIO,
            "audio/wav",
            frozenset({Modality.AUDIO}),
            id="audio",
        ),
        pytest.param(
            MediaKind.AUDIO,
            "audio/flac",
            frozenset({Modality.MUSIC}),
            id="music",
        ),
        pytest.param(
            MediaKind.VIDEO,
            "video/mp4",
            frozenset({Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO}),
            id="joint-video",
        ),
        pytest.param(
            MediaKind.IMAGE,
            "image/jpeg",
            frozenset({Modality.IMAGE, Modality.VIDEO_VISUAL}),
            id="joint-image-video",
        ),
    ],
)
@pytest.mark.asyncio
async def test_rejects_non_visual_and_joint_modality_sets_without_transport(
    tmp_path: Path,
    media_kind: MediaKind,
    mime_type: str,
    modalities: frozenset[Modality],
) -> None:
    """Letting hearing or joint requests through would misadvertise MiniMax as an ear."""
    media = tmp_path / "media.bin"
    media.write_bytes(b"media")
    transport_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal transport_calls
        transport_calls += 1
        return httpx.Response(500)

    async with _client(handler) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    media,
                    media_kind=media_kind,
                    mime_type=mime_type,
                    modalities=modalities,
                )
            )

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert transport_calls == 0


@pytest.mark.parametrize(
    ("region", "want_base_url"),
    [
        pytest.param(MiniMaxRegion.GLOBAL, MINIMAX_GLOBAL_BASE_URL, id="global"),
        pytest.param(MiniMaxRegion.CN, MINIMAX_CN_BASE_URL, id="cn"),
    ],
)
@pytest.mark.asyncio
async def test_regional_defaults_use_exact_anthropic_messages_endpoint(
    tmp_path: Path,
    region: MiniMaxRegion,
    want_base_url: str,
) -> None:
    """Using a native-v2 or Hailuo path would call a different MiniMax product."""
    image = tmp_path / "region.jpg"
    image.write_bytes(b"image")
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json=_provider_response(Modality.IMAGE))

    async with _client(handler) as client:
        await _provider(client, region=region).sense(
            _request(
                image,
                media_kind=MediaKind.IMAGE,
                mime_type="image/jpeg",
                modalities=frozenset({Modality.IMAGE}),
            )
        )

    assert urls == [f"{want_base_url}/anthropic/v1/messages"]


@pytest.mark.asyncio
async def test_custom_https_base_url_overrides_region_and_preserves_path(tmp_path: Path) -> None:
    """Ignoring a validated tenant path would send media to the wrong API origin."""
    image = tmp_path / "custom.jpg"
    image.write_bytes(b"image")
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json=_provider_response(Modality.IMAGE))

    config = _config(base_url="https://tenant.example.invalid/minimax")
    async with _client(handler) as client:
        await _provider(client, config=config, region=MiniMaxRegion.CN).sense(
            _request(
                image,
                media_kind=MediaKind.IMAGE,
                mime_type="image/jpeg",
                modalities=frozenset({Modality.IMAGE}),
            )
        )

    assert urls == [
        "https://tenant.example.invalid/minimax/anthropic/v1/messages"
    ]


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.example.invalid",
        "https://key:secret@api.example.invalid",
        "https://api.example.invalid?api_key=secret",
        "https://api.example.invalid/#fragment",
    ],
)
def test_provider_config_rejects_unsafe_custom_base_urls(base_url: str) -> None:
    """Allowing an unvalidated custom URL could leak media or credentials."""
    with pytest.raises(ValidationError):
        _config(base_url=base_url)


@pytest.mark.parametrize(
    ("auth_mode", "want_name", "want_value", "absent_name"),
    [
        pytest.param(
            MiniMaxAuthMode.X_API_KEY,
            "x-api-key",
            _SECRET,
            "authorization",
            id="x-api-key",
        ),
        pytest.param(
            MiniMaxAuthMode.BEARER,
            "authorization",
            f"Bearer {_SECRET}",
            "x-api-key",
            id="bearer",
        ),
    ],
)
@pytest.mark.asyncio
async def test_each_explicit_auth_mode_sends_only_its_authorized_header(
    tmp_path: Path,
    auth_mode: MiniMaxAuthMode,
    want_name: str,
    want_value: str,
    absent_name: str,
) -> None:
    """Mixing auth schemes would broaden credential exposure beyond the chosen mode."""
    image = tmp_path / "auth.jpg"
    image.write_bytes(b"image")
    headers: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        headers.append(request.headers)
        return httpx.Response(200, json=_provider_response(Modality.IMAGE))

    async with _client(handler) as client:
        await _provider(client, auth_mode=auth_mode).sense(
            _request(
                image,
                media_kind=MediaKind.IMAGE,
                mime_type="image/jpeg",
                modalities=frozenset({Modality.IMAGE}),
            )
        )

    assert headers[0][want_name] == want_value
    assert absent_name not in headers[0]
    assert headers[0]["anthropic-version"] == "2023-06-01"


class RotatingSecretStore(MemorySecretStore):
    def __init__(self) -> None:
        super().__init__()
        self.get_calls = 0

    def get(self, ref: str, env_name: str | None = None) -> str:
        self.get_calls += 1
        return super().get(ref, env_name)


class FailingSecretStore(MemorySecretStore):
    def get(self, ref: str, env_name: str | None = None) -> str:
        raise RuntimeError("credential backend exposed secret-value-marker")


@pytest.mark.asyncio
async def test_secret_is_looked_up_at_call_time_for_each_request(tmp_path: Path) -> None:
    """Caching a key in the adapter would ignore credential rotation and retain secrets."""
    image = tmp_path / "rotating.jpg"
    image.write_bytes(b"image")
    auth_values: list[str] = []
    store = RotatingSecretStore()
    store.set("minimax-test-credential", "first-secret-value")

    def handler(request: httpx.Request) -> httpx.Response:
        auth_values.append(request.headers["x-api-key"])
        return httpx.Response(200, json=_provider_response(Modality.IMAGE))

    async with _client(handler) as client:
        provider = _provider(client, store=store)
        request = _request(
            image,
            media_kind=MediaKind.IMAGE,
            mime_type="image/jpeg",
            modalities=frozenset({Modality.IMAGE}),
        )
        store.set("minimax-test-credential", "second-secret-value")
        await provider.sense(request)
        store.set("minimax-test-credential", "third-secret-value")
        await provider.sense(request)

    assert auth_values == ["second-secret-value", "third-secret-value"]
    assert store.get_calls == 2


@pytest.mark.asyncio
async def test_unexpected_secret_store_failure_is_sanitized(tmp_path: Path) -> None:
    """A broken credential backend must not leak its diagnostic through the adapter."""
    image = tmp_path / "secret-store.jpg"
    image.write_bytes(b"image")

    async with _client(lambda request: httpx.Response(500)) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client, store=FailingSecretStore()).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert caught.value.cause is None
    assert "secret-value-marker" not in str(caught.value)


@pytest.mark.asyncio
async def test_file_is_read_once_and_size_is_checked_before_base64(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated reads or encoding before the limit check would waste or expose oversized media."""
    image = tmp_path / "once.jpg"
    image.write_bytes(b"123456")
    original_read_bytes = Path.read_bytes
    read_count = 0
    encode_count = 0

    def tracked_read_bytes(path: Path) -> bytes:
        nonlocal read_count
        read_count += 1
        return original_read_bytes(path)

    original_b64encode = base64.b64encode

    def tracked_b64encode(data: bytes | bytearray) -> bytes:
        nonlocal encode_count
        encode_count += 1
        return original_b64encode(data)

    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)
    monkeypatch.setattr(base64, "b64encode", tracked_b64encode)

    async with _client(
        lambda request: httpx.Response(200, json=_provider_response(Modality.IMAGE))
    ) as client:
        await _provider(client).sense(
            _request(
                image,
                media_kind=MediaKind.IMAGE,
                mime_type="image/jpeg",
                modalities=frozenset({Modality.IMAGE}),
            )
        )

    assert read_count == 1
    assert encode_count == 1

    read_count = 0
    encode_count = 0
    calls = 0

    def should_not_call(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async with _client(should_not_call) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client, config=_config(max_bytes=5)).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is ErrorCode.MEDIA_TOO_LARGE
    assert read_count == 0
    assert encode_count == 0
    assert calls == 0


@pytest.mark.asyncio
async def test_configured_request_bounds_reach_exact_payload_and_timeout(tmp_path: Path) -> None:
    """Dropping bounded settings would permit unbounded output or slow calls."""
    image = tmp_path / "bounds.jpg"
    image.write_bytes(b"image")
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_provider_response(Modality.IMAGE))

    config = _config(
        model="MiniMax-M3-test",
        max_output_tokens=1_234,
        temperature=0.4,
        timeout=7.5,
    )
    async with _client(handler) as client:
        await _provider(client, config=config).sense(
            _request(
                image,
                media_kind=MediaKind.IMAGE,
                mime_type="image/jpeg",
                modalities=frozenset({Modality.IMAGE}),
            )
        )

    payload = json.loads(captured[0].content)
    assert payload["model"] == "MiniMax-M3-test"
    assert payload["max_tokens"] == 1_234
    assert payload["temperature"] == 0.4
    timeout = cast(dict[str, float], captured[0].extensions["timeout"])
    assert timeout == {"connect": 7.5, "read": 7.5, "write": 7.5, "pool": 7.5}


def test_adapter_rejects_temperature_outside_its_low_variance_bound() -> None:
    """Accepting high temperature would make observation output less deterministic."""
    config = _config(temperature=1.1)
    client = _client(lambda request: httpx.Response(500))
    with pytest.raises(SensoryError) as caught:
        _provider(client, config=config)
    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert "temperature" not in str(caught.value).lower()


@pytest.mark.parametrize(
    ("response", "want_code", "want_retryable"),
    [
        pytest.param(
            httpx.Response(401, json={"error": {"message": "secret rejected"}}),
            ErrorCode.PROVIDER_AUTH_FAILED,
            False,
            id="401-auth",
        ),
        pytest.param(
            httpx.Response(429, json={"error": {"message": "rate limit raw body"}}),
            ErrorCode.PROVIDER_UNAVAILABLE,
            True,
            id="429-rate-limit",
        ),
        pytest.param(
            httpx.Response(
                400,
                json={"error": {"type": "safety_error", "message": "unsafe media"}},
            ),
            ErrorCode.PROVIDER_SAFETY_REJECTED,
            False,
            id="safety",
        ),
        pytest.param(
            httpx.Response(
                400,
                json={
                    "type": "error",
                    "error": {
                        "type": "unsupported_media_type",
                        "message": "unsupported media type",
                    },
                },
            ),
            ErrorCode.PROVIDER_CAPABILITY_REJECTED,
            False,
            id="unsupported-media",
        ),
        pytest.param(
            httpx.Response(503, text="upstream unavailable raw body"),
            ErrorCode.PROVIDER_UNAVAILABLE,
            True,
            id="unavailable",
        ),
    ],
)
@pytest.mark.asyncio
async def test_http_failures_map_to_stable_sanitized_categories(
    tmp_path: Path,
    response: httpx.Response,
    want_code: ErrorCode,
    want_retryable: bool,
) -> None:
    """Returning Provider bodies would expose remote diagnostics and request data."""
    image = tmp_path / "category.jpg"
    image.write_bytes(b"image")
    raw_body = response.text

    async with _client(lambda request: response) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is want_code
    assert caught.value.retryable is want_retryable
    assert caught.value.cause is None
    assert raw_body not in str(caught.value)


@pytest.mark.parametrize(
    ("http_status", "base_code", "want_code", "want_retryable"),
    [
        pytest.param(200, 1001, ErrorCode.PROVIDER_TIMEOUT, True, id="timeout-1001"),
        pytest.param(
            200,
            1002,
            ErrorCode.PROVIDER_UNAVAILABLE,
            True,
            id="rate-limit-1002",
        ),
        pytest.param(
            503,
            1004,
            ErrorCode.PROVIDER_AUTH_FAILED,
            False,
            id="auth-1004-non-2xx",
        ),
        pytest.param(
            200,
            2049,
            ErrorCode.PROVIDER_AUTH_FAILED,
            False,
            id="auth-2049",
        ),
        pytest.param(
            200,
            1026,
            ErrorCode.PROVIDER_SAFETY_REJECTED,
            False,
            id="input-safety-1026",
        ),
        pytest.param(
            400,
            1027,
            ErrorCode.PROVIDER_SAFETY_REJECTED,
            False,
            id="output-safety-1027-non-2xx",
        ),
    ],
)
@pytest.mark.asyncio
async def test_base_response_codes_override_http_status_without_leaking_status_message(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    http_status: int,
    base_code: int,
    want_code: ErrorCode,
    want_retryable: bool,
) -> None:
    """Ignoring MiniMax's structured status would misclassify a valid HTTP envelope."""
    image = tmp_path / "base-code.jpg"
    image.write_bytes(b"image")
    private_status = f"private-status-{base_code}-{_SECRET}"
    caplog.set_level(logging.DEBUG)

    async with _client(
        lambda request: httpx.Response(
            http_status,
            json=_response_with_base_code(
                Modality.IMAGE,
                base_code,
                status_msg=private_status,
            ),
        )
    ) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is want_code
    assert caught.value.retryable is want_retryable
    assert caught.value.cause is None
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert private_status not in str(caught.value)
    assert private_status not in log_text


@pytest.mark.parametrize("http_status", [200, 400])
@pytest.mark.asyncio
async def test_unknown_nonzero_base_code_is_never_guessed_as_unsupported_media(
    tmp_path: Path,
    http_status: int,
) -> None:
    """An undocumented status must remain unavailable even if its message suggests media."""
    image = tmp_path / "unknown-base-code.jpg"
    image.write_bytes(b"image")

    async with _client(
        lambda request: httpx.Response(
            http_status,
            json=_response_with_base_code(
                Modality.IMAGE,
                9999,
                status_msg="unsupported media private body marker",
            ),
        )
    ) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert caught.value.retryable is True
    assert "unsupported media" not in str(caught.value)


@pytest.mark.parametrize(
    ("http_status", "base_resp", "want_code", "want_retryable"),
    [
        pytest.param(
            200,
            {"status_code": "1004", "status_msg": "invalid API key private"},
            ErrorCode.PROVIDER_CAPABILITY_REJECTED,
            False,
            id="string-code-success",
        ),
        pytest.param(
            400,
            {"status_code": "1026", "status_msg": "unsafe media private"},
            ErrorCode.PROVIDER_UNAVAILABLE,
            True,
            id="string-code-error",
        ),
        pytest.param(
            200,
            {"status_code": True, "status_msg": "boolean private"},
            ErrorCode.PROVIDER_CAPABILITY_REJECTED,
            False,
            id="boolean-code",
        ),
        pytest.param(
            200,
            {"status_msg": "missing code private"},
            ErrorCode.PROVIDER_CAPABILITY_REJECTED,
            False,
            id="missing-code",
        ),
        pytest.param(
            200,
            "not-an-object",
            ErrorCode.PROVIDER_CAPABILITY_REJECTED,
            False,
            id="non-object-base-response",
        ),
    ],
)
@pytest.mark.asyncio
async def test_malformed_base_response_fails_without_guessing_from_private_message(
    tmp_path: Path,
    http_status: int,
    base_resp: object,
    want_code: ErrorCode,
    want_retryable: bool,
) -> None:
    """Coercing a malformed status code could turn arbitrary text into an error category."""
    image = tmp_path / "malformed-base-response.jpg"
    image.write_bytes(b"image")
    payload = _provider_response(Modality.IMAGE)
    payload["base_resp"] = base_resp

    async with _client(
        lambda request: httpx.Response(http_status, json=payload)
    ) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is want_code
    assert caught.value.retryable is want_retryable
    assert "private" not in str(caught.value)


@pytest.mark.asyncio
async def test_streamed_response_without_content_length_is_bounded_and_closed(
    tmp_path: Path,
) -> None:
    """Chunked success responses must use the same bounded parser and release transport."""
    image = tmp_path / "chunked-success.jpg"
    image.write_bytes(b"image")
    body = _response_bytes()
    stream = TrackingAsyncStream([body[:17], body[17:]])

    async with _client(
        lambda request: httpx.Response(
            200,
            headers={"content-encoding": "identity"},
            stream=stream,
        )
    ) as client:
        result = await _provider(client).sense(
            _request(
                image,
                media_kind=MediaKind.IMAGE,
                mime_type="image/jpeg",
                modalities=frozenset({Modality.IMAGE}),
            )
        )

    assert result.observations[Modality.IMAGE].summary == "Direct evidence."
    assert stream.yielded_chunks == 2
    assert stream.closed is True


@pytest.mark.parametrize(
    "encoding_headers",
    [
        pytest.param([(b"content-encoding", b"")], id="empty"),
        pytest.param(
            [(b"content-encoding", b"identity, identity")],
            id="identity-list",
        ),
        pytest.param(
            [
                (b"content-encoding", b"identity"),
                (b"content-encoding", b"identity"),
            ],
            id="duplicate-identity-fields",
        ),
        pytest.param(
            [(b"content-encoding", b"identity, gzip")],
            id="mixed-list",
        ),
    ],
)
@pytest.mark.parametrize(
    ("http_status", "want_code", "want_retryable"),
    [
        (200, ErrorCode.PROVIDER_CAPABILITY_REJECTED, False),
        (503, ErrorCode.PROVIDER_UNAVAILABLE, True),
    ],
)
@pytest.mark.asyncio
async def test_non_exact_identity_content_encoding_is_rejected_before_iteration(
    tmp_path: Path,
    encoding_headers: list[tuple[bytes, bytes]],
    http_status: int,
    want_code: ErrorCode,
    want_retryable: bool,
) -> None:
    """Empty, repeated, or listed encodings must not enter the raw response reader."""
    image = tmp_path / "invalid-encoding.jpg"
    image.write_bytes(b"image")
    stream = TrackingAsyncStream([_response_bytes()])

    async with _client(
        lambda request: httpx.Response(
            http_status,
            headers=encoding_headers,
            stream=stream,
        )
    ) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is want_code
    assert caught.value.retryable is want_retryable
    assert stream.iterated is False
    assert stream.closed is True


@pytest.mark.parametrize(
    ("content_encoding", "compress"),
    [
        ("gzip", gzip.compress),
        ("deflate", zlib.compress),
    ],
)
@pytest.mark.parametrize(
    ("http_status", "want_code", "want_retryable"),
    [
        (200, ErrorCode.PROVIDER_CAPABILITY_REJECTED, False),
        (503, ErrorCode.PROVIDER_UNAVAILABLE, True),
    ],
)
@pytest.mark.asyncio
async def test_compressed_amplification_is_rejected_without_decompression(
    tmp_path: Path,
    content_encoding: str,
    compress: Callable[[bytes], bytes],
    http_status: int,
    want_code: ErrorCode,
    want_retryable: bool,
) -> None:
    """A tiny compressed body must not inflate beyond the raw eight-MiB boundary."""
    image = tmp_path / f"{content_encoding}-amplification.jpg"
    image.write_bytes(b"image")
    private_marker = b"private-decompressed-response-marker"
    expanded = private_marker + b"x" * _MAX_RESPONSE_BYTES_TEST
    compressed = compress(expanded)
    assert len(compressed) < _MAX_RESPONSE_BYTES_TEST
    stream = TrackingAsyncStream([compressed])

    async with _client(
        lambda request: httpx.Response(
            http_status,
            headers={
                "content-encoding": content_encoding,
                "content-length": str(len(compressed)),
            },
            stream=stream,
        )
    ) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is want_code
    assert caught.value.retryable is want_retryable
    assert private_marker.decode("ascii") not in str(caught.value)
    assert stream.iterated is False
    assert stream.closed is True


@pytest.mark.parametrize("content_length", ["invalid", "-1", "1, 2"])
@pytest.mark.asyncio
async def test_invalid_content_length_is_rejected_before_stream_iteration(
    tmp_path: Path,
    content_length: str,
) -> None:
    """Trusting a malformed length would disable the response precheck."""
    image = tmp_path / "invalid-length.jpg"
    image.write_bytes(b"image")
    stream = TrackingAsyncStream([_response_bytes()])

    async with _client(
        lambda request: httpx.Response(
            200,
            headers={"content-length": content_length},
            stream=stream,
        )
    ) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert stream.iterated is False
    assert stream.closed is True


@pytest.mark.asyncio
async def test_oversized_content_length_closes_without_reading_body(tmp_path: Path) -> None:
    """A declared oversized response must be refused before downloading any chunk."""
    image = tmp_path / "declared-oversized.jpg"
    image.write_bytes(b"image")
    stream = TrackingAsyncStream([b"private-body-marker"])

    async with _client(
        lambda request: httpx.Response(
            200,
            headers={
                "content-encoding": "identity",
                "content-length": str(_MAX_RESPONSE_BYTES_TEST + 1),
            },
            stream=stream,
        )
    ) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert "private-body-marker" not in str(caught.value)
    assert stream.iterated is False
    assert stream.closed is True


@pytest.mark.parametrize(
    ("http_status", "want_code", "want_retryable"),
    [
        pytest.param(
            200,
            ErrorCode.PROVIDER_CAPABILITY_REJECTED,
            False,
            id="oversized-success",
        ),
        pytest.param(
            503,
            ErrorCode.PROVIDER_UNAVAILABLE,
            True,
            id="oversized-error",
        ),
    ],
)
@pytest.mark.asyncio
async def test_chunked_response_stops_at_cumulative_byte_cap_and_closes(
    tmp_path: Path,
    http_status: int,
    want_code: ErrorCode,
    want_retryable: bool,
) -> None:
    """Reading past the cumulative cap would permit chunked response amplification."""
    image = tmp_path / "chunk-overflow.jpg"
    image.write_bytes(b"image")
    half = _MAX_RESPONSE_BYTES_TEST // 2
    stream = TrackingAsyncStream(
        [
            b"a" * half,
            b"b" * (half + 1),
            b"private-third-chunk-must-not-be-read",
        ]
    )

    async with _client(
        lambda request: httpx.Response(
            http_status,
            headers={"content-encoding": "identity"},
            stream=stream,
        )
    ) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is want_code
    assert caught.value.retryable is want_retryable
    assert "private-third-chunk" not in str(caught.value)
    assert stream.yielded_chunks == 2
    assert stream.closed is True


@pytest.mark.asyncio
async def test_lying_small_content_length_cannot_bypass_cumulative_cap(
    tmp_path: Path,
) -> None:
    """The streaming cap must override a server that declares a smaller response."""
    image = tmp_path / "lying-length.jpg"
    image.write_bytes(b"image")
    stream = TrackingAsyncStream(
        [
            b"a" * _MAX_RESPONSE_BYTES_TEST,
            b"overflow",
            b"private-unread-tail",
        ]
    )

    async with _client(
        lambda request: httpx.Response(
            200,
            headers={"content-length": "1"},
            stream=stream,
        )
    ) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert stream.yielded_chunks == 2
    assert stream.closed is True


@pytest.mark.asyncio
async def test_response_rejects_too_many_content_blocks(tmp_path: Path) -> None:
    """An unbounded content array would amplify parser work before normalization."""
    image = tmp_path / "many-blocks.jpg"
    image.write_bytes(b"image")
    payload = _provider_response(Modality.IMAGE)
    valid_text = cast(list[dict[str, str]], payload["content"])[0]["text"]
    payload["content"] = [
        {"type": "text", "text": valid_text},
        *[
            {"type": "text", "text": " "}
            for _ in range(_MAX_CONTENT_BLOCKS_TEST)
        ],
    ]

    async with _client(
        lambda request: httpx.Response(200, json=payload)
    ) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED


@pytest.mark.asyncio
async def test_response_rejects_oversized_text_block_before_normalization(
    tmp_path: Path,
) -> None:
    """Whitespace cannot be used to inflate one text block past its parser bound."""
    image = tmp_path / "large-block.jpg"
    image.write_bytes(b"image")
    payload = _provider_response(Modality.IMAGE)
    content = cast(list[dict[str, str]], payload["content"])
    content[0]["text"] += " " * (_MAX_TEXT_BLOCK_CHARS_TEST + 1)

    async with _client(
        lambda request: httpx.Response(200, json=payload)
    ) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED


@pytest.mark.asyncio
async def test_response_rejects_oversized_combined_text_before_normalization(
    tmp_path: Path,
) -> None:
    """Many individually valid text blocks must still obey one combined-text cap."""
    image = tmp_path / "combined-text.jpg"
    image.write_bytes(b"image")
    payload = _provider_response(Modality.IMAGE)
    valid_text = cast(list[dict[str, str]], payload["content"])[0]["text"]
    padding_size = _MAX_COMBINED_TEXT_CHARS_TEST // 2 + 1
    payload["content"] = [
        {"type": "text", "text": valid_text + " " * padding_size},
        {"type": "text", "text": " " * padding_size},
    ]

    async with _client(
        lambda request: httpx.Response(200, json=payload)
    ) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED


@pytest.mark.asyncio
async def test_streaming_response_is_closed_when_call_is_cancelled(tmp_path: Path) -> None:
    """Cancellation during download must release the active HTTP response promptly."""
    image = tmp_path / "stream-cancel.jpg"
    image.write_bytes(b"image")
    stream = BlockingAsyncStream()

    async with _client(
        lambda request: httpx.Response(200, stream=stream)
    ) as client:
        task = asyncio.create_task(
            _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )
        )
        await stream.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert stream.closed is True


@pytest.mark.parametrize(
    ("http_status", "want_code", "want_retryable"),
    [
        pytest.param(
            200,
            ErrorCode.PROVIDER_CAPABILITY_REJECTED,
            False,
            id="deep-success-envelope",
        ),
        pytest.param(
            500,
            ErrorCode.PROVIDER_UNAVAILABLE,
            True,
            id="deep-error-envelope",
        ),
    ],
)
@pytest.mark.asyncio
async def test_deeply_nested_bounded_json_cannot_escape_raw_recursion_error(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    http_status: int,
    want_code: ErrorCode,
    want_retryable: bool,
) -> None:
    """A byte-bounded body must also contain JSON decoder recursion failures."""
    image = tmp_path / "deep-json.jpg"
    image.write_bytes(b"image")
    private_marker = "private-deep-json-body-marker"
    depth = 10_000
    body = (
        b'{"nested":'
        + b"[" * depth
        + json.dumps(private_marker).encode("utf-8")
        + b"]" * depth
        + b"}"
    )
    stream = TrackingAsyncStream([body])
    caplog.set_level(logging.DEBUG)

    async with _client(
        lambda request: httpx.Response(http_status, stream=stream)
    ) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is want_code
    assert caught.value.retryable is want_retryable
    assert caught.value.cause is None
    public_text = str(caught.value)
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    for private_value in (private_marker, "RecursionError", "maximum recursion"):
        assert private_value not in public_text
        assert private_value not in log_text
    assert stream.closed is True


@pytest.mark.asyncio
async def test_recursive_report_text_uses_shared_raw_free_normalizer(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A valid MiniMax success envelope must contain recursive report failures."""
    image = tmp_path / "recursive-report.jpg"
    image.write_bytes(b"image")
    private_marker = "private-recursive-report-marker"
    depth = 10_000
    recursive_report = (
        '{"observations":'
        + "[" * depth
        + json.dumps(private_marker)
        + "]" * depth
        + "}"
    )
    payload = _provider_response(Modality.IMAGE)
    content = cast(list[dict[str, str]], payload["content"])
    content[0]["text"] = recursive_report
    stream = TrackingAsyncStream([json.dumps(payload).encode("utf-8")])
    caplog.set_level(logging.DEBUG)

    async with _client(
        lambda request: httpx.Response(200, stream=stream)
    ) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert caught.value.retryable is False
    assert caught.value.cause is None
    public_text = str(caught.value)
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    for forbidden in (
        private_marker,
        "RecursionError",
        "maximum recursion depth",
        "while decoding",
    ):
        assert forbidden not in public_text
        assert forbidden not in log_text
    assert stream.closed is True


@pytest.mark.parametrize(
    ("response", "want_code"),
    [
        pytest.param(httpx.Response(200, content=b""), ErrorCode.PROVIDER_CAPABILITY_REJECTED, id="empty"),
        pytest.param(httpx.Response(200, content=b"not-json"), ErrorCode.PROVIDER_CAPABILITY_REJECTED, id="malformed-json"),
        pytest.param(httpx.Response(200, json={"content": []}), ErrorCode.PROVIDER_CAPABILITY_REJECTED, id="empty-content"),
        pytest.param(
            httpx.Response(200, json={"content": [{"type": "text", "text": "not-report-json"}]}),
            ErrorCode.PROVIDER_CAPABILITY_REJECTED,
            id="malformed-report",
        ),
    ],
)
@pytest.mark.asyncio
async def test_empty_and_malformed_success_responses_are_rejected_safely(
    tmp_path: Path,
    response: httpx.Response,
    want_code: ErrorCode,
) -> None:
    """Treating an empty Provider response as an observation would fabricate evidence."""
    image = tmp_path / "malformed.jpg"
    image.write_bytes(b"image")

    async with _client(lambda request: response) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is want_code
    assert "not-report-json" not in str(caught.value)


@pytest.mark.parametrize("failure_type", [httpx.ReadTimeout, httpx.ConnectError])
@pytest.mark.asyncio
async def test_timeout_and_network_failures_map_without_transport_messages(
    tmp_path: Path,
    failure_type: type[httpx.RequestError],
) -> None:
    """Transport diagnostics can contain URLs and must not enter public exceptions."""
    image = tmp_path / "network.jpg"
    image.write_bytes(b"image")
    request = httpx.Request("POST", "https://sensitive.example.invalid/private/path")

    def handler(incoming: httpx.Request) -> httpx.Response:
        raise failure_type("raw network message with credential", request=request)

    async with _client(handler) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    expected = (
        ErrorCode.PROVIDER_TIMEOUT
        if failure_type is httpx.ReadTimeout
        else ErrorCode.PROVIDER_UNAVAILABLE
    )
    assert caught.value.code is expected
    assert caught.value.retryable is True
    assert caught.value.cause is None
    assert "credential" not in str(caught.value)
    assert "/private/path" not in str(caught.value)


@pytest.mark.asyncio
async def test_unexpected_transport_failure_is_sanitized(tmp_path: Path) -> None:
    """An unusual transport implementation must not pierce the public error boundary."""
    image = tmp_path / "unexpected-network.jpg"
    image.write_bytes(b"image")

    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("transport exposed raw-secret-marker")

    async with _client(handler) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert caught.value.cause is None
    assert "raw-secret-marker" not in str(caught.value)


@pytest.mark.parametrize("status_code", [307, 308])
@pytest.mark.asyncio
async def test_borrowed_redirecting_client_cannot_replay_secret_or_media_cross_origin(
    tmp_path: Path,
    status_code: int,
) -> None:
    """Per-request redirect control must override a borrowed client's unsafe default."""
    image = tmp_path / "redirect.jpg"
    media_marker = b"private-redirect-media-marker"
    image.write_bytes(media_marker)
    requests: list[httpx.Request] = []
    attacker_url = "https://attacker.example.invalid/collect"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "attacker.example.invalid":
            pytest.fail("redirect replay reached the attacker origin")
        return httpx.Response(
            status_code,
            headers={"location": attacker_url},
        )

    async with _client(handler, follow_redirects=True) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    assert caught.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert [str(request.url) for request in requests] == [
        f"{MINIMAX_GLOBAL_BASE_URL}/anthropic/v1/messages"
    ]
    public_text = str(caught.value)
    for private_value in (attacker_url, _SECRET, media_marker.decode("ascii")):
        assert private_value not in public_text


@pytest.mark.asyncio
async def test_cancellation_propagates_without_error_mapping(tmp_path: Path) -> None:
    """Mapping cancellation to fallback-eligible unavailability would send media elsewhere."""
    image = tmp_path / "cancel.jpg"
    image.write_bytes(b"image")
    started = asyncio.Event()
    gate = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        started.set()
        await gate.wait()
        return httpx.Response(200, json=_provider_response(Modality.IMAGE))

    async with _client(handler) as client:
        task = asyncio.create_task(
            _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_public_exceptions_and_logs_exclude_secret_body_and_path(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Logging caught Provider data would violate the privacy boundary."""
    private_marker = "private-media-name-marker"
    raw_marker = "raw-provider-response-marker"
    image = tmp_path / f"{private_marker}.jpg"
    image.write_bytes(b"private-media-bytes")
    caplog.set_level(logging.DEBUG)

    async with _client(
        lambda request: httpx.Response(
            400,
            text=f"{raw_marker} {_SECRET} {image}",
        )
    ) as client:
        with pytest.raises(SensoryError) as caught:
            await _provider(client).sense(
                _request(
                    image,
                    media_kind=MediaKind.IMAGE,
                    mime_type="image/jpeg",
                    modalities=frozenset({Modality.IMAGE}),
                )
            )

    public_text = str(caught.value)
    log_text = "\n".join(
        f"{record.getMessage()} {record.__dict__}" for record in caplog.records
    )
    for forbidden in (_SECRET, raw_marker, private_marker, str(image), "private-media-bytes"):
        assert forbidden not in public_text
        assert forbidden not in log_text


@pytest.mark.asyncio
async def test_injected_client_is_borrowed_and_not_closed_by_provider(tmp_path: Path) -> None:
    """Closing a borrowed client would break other adapters sharing its connection pool."""
    image = tmp_path / "borrowed.jpg"
    image.write_bytes(b"image")
    client = _client(
        lambda request: httpx.Response(200, json=_provider_response(Modality.IMAGE))
    )
    provider = _provider(client)

    await provider.sense(
        _request(
            image,
            media_kind=MediaKind.IMAGE,
            mime_type="image/jpeg",
            modalities=frozenset({Modality.IMAGE}),
        )
    )
    await provider.aclose()

    assert client.is_closed is False
    await client.aclose()


@pytest.mark.asyncio
async def test_provider_closes_the_async_client_it_constructs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leaking an owned client would retain network resources for the process lifetime."""
    owned = _client(lambda request: httpx.Response(500))

    def fake_async_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return owned

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    provider = MiniMaxM3Provider(
        provider_id="minimax",
        config=_config(),
        secret_store=_store(),
    )

    await provider.aclose()

    assert owned.is_closed is True


@pytest.mark.asyncio
async def test_owned_client_ignores_hostile_proxy_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambient HTTPS, all-protocol, or SOCKS proxies must not receive Provider media."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.invalid:8443")
    monkeypatch.setenv("ALL_PROXY", "http://all-proxy.example.invalid:8080")
    monkeypatch.setenv("SOCKS_PROXY", "socks5://socks.example.invalid:1080")
    owned = _client(lambda request: httpx.Response(500))
    construction_options: dict[str, object] = {}

    def fake_async_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        construction_options.update(kwargs)
        return owned

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    provider = MiniMaxM3Provider(
        provider_id="minimax",
        config=_config(),
        secret_store=_store(),
    )

    assert construction_options == {
        "follow_redirects": False,
        "trust_env": False,
    }
    await provider.aclose()
    assert owned.is_closed is True


@pytest.mark.asyncio
async def test_borrowed_client_is_not_reconstructed_under_hostile_proxy_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dependency-injected transport policy belongs to the caller and stays borrowed."""
    borrowed = _client(lambda request: httpx.Response(500))
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.invalid:8443")
    monkeypatch.setenv("ALL_PROXY", "socks5://all-proxy.example.invalid:1080")
    monkeypatch.setenv("SOCKS_PROXY", "socks5://socks.example.invalid:1080")

    def unexpected_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        pytest.fail("a borrowed MiniMax client was reconstructed")

    monkeypatch.setattr(httpx, "AsyncClient", unexpected_client)
    provider = MiniMaxM3Provider(
        provider_id="minimax",
        config=_config(),
        secret_store=_store(),
        client=borrowed,
    )

    await provider.aclose()
    assert borrowed.is_closed is False
    await borrowed.aclose()
