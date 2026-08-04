from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable
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
    }


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


def _client(handler: Callable[[httpx.Request], object]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(cast(Any, handler)))


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


@pytest.mark.parametrize(
    ("config", "message_part"),
    [
        pytest.param(_config(model="m" * 257), "model", id="model-length"),
        pytest.param(_config(temperature=1.1), "temperature", id="high-temperature"),
    ],
)
def test_adapter_rejects_values_outside_its_low_variance_bounds(
    config: ProviderConfig,
    message_part: str,
) -> None:
    """Accepting an unbounded model name or high temperature would violate adapter limits."""
    client = _client(lambda request: httpx.Response(500))
    with pytest.raises(SensoryError) as caught:
        _provider(client, config=config)
    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert message_part not in str(caught.value).lower()


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
                    "error": {
                        "code": "unsupported_media_type",
                        "message": "video media format unsupported",
                    }
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


@pytest.mark.asyncio
async def test_cross_origin_redirect_is_not_followed(tmp_path: Path) -> None:
    """Following a cross-origin redirect would disclose the key and encoded media."""
    image = tmp_path / "redirect.jpg"
    image.write_bytes(b"image")
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(
            307,
            headers={"location": "https://attacker.example.invalid/collect"},
        )

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
    assert urls == [f"{MINIMAX_GLOBAL_BASE_URL}/anthropic/v1/messages"]
    assert "attacker" not in str(caught.value)


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
