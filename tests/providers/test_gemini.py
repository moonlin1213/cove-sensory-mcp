from __future__ import annotations

import asyncio
import json
import logging
import secrets
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from google import genai as google_genai
from google.genai import types as google_types

from cove_sensory_mcp.config.schema import AdapterOptions, ProviderConfig
from cove_sensory_mcp.config.secrets import MemorySecretStore
from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import DetailLevel, Modality
from cove_sensory_mcp.providers.base import MediaKind, PreparedMedia, ProviderRequest
from cove_sensory_mcp.providers.gemini import (
    GeminiClient,
    GeminiClientFailure,
    GeminiFailureKind,
    GeminiGeneration,
    GeminiInlineMedia,
    GeminiProvider,
    GeminiRemoteFile,
    GeminiUploadedMedia,
    _OfficialGoogleGenAIClient,
)


def _observation(modality: Modality, summary: str | None = None) -> dict[str, object]:
    return {
        "modality": modality.value,
        "summary": summary or f"Observed {modality.value}.",
        "segments": [],
        "transcript": [],
        "warnings": [],
        "confidence": "medium",
    }


def _response_text(*modalities: Modality) -> str:
    return json.dumps(
        {"observations": [_observation(modality) for modality in modalities]}
    )


class FakeGeminiClient:
    def __init__(self, generation: GeminiGeneration) -> None:
        self.generation = generation
        self.generation_responses: list[GeminiGeneration] = []
        self.uploads: list[tuple[Path, str]] = []
        self.waited_for: list[GeminiRemoteFile] = []
        self.generate_calls: list[dict[str, object]] = []
        self.deleted_names: list[str] = []
        self.closed = False
        self.generate_started = asyncio.Event()
        self.generate_gate: asyncio.Event | None = None
        self.wait_started = asyncio.Event()
        self.wait_gate: asyncio.Event | None = None
        self.generate_error: BaseException | None = None
        self.delete_error: Exception | None = None
        self.delete_started = asyncio.Event()
        self.delete_gate: asyncio.Event | None = None
        self.delete_finished = False
        self.close_started = asyncio.Event()
        self.close_gate: asyncio.Event | None = None
        self.close_finished = False

    async def upload_file(self, *, path: Path, mime_type: str) -> GeminiRemoteFile:
        self.uploads.append((path, mime_type))
        return GeminiRemoteFile(
            name="files/test-upload",
            uri="https://files.example.invalid/test-upload",
            mime_type=mime_type,
        )

    async def wait_until_active(self, file: GeminiRemoteFile) -> GeminiRemoteFile:
        self.waited_for.append(file)
        self.wait_started.set()
        if self.wait_gate is not None:
            await self.wait_gate.wait()
        return file

    async def generate_content(
        self,
        *,
        model: str,
        contents: tuple[str | GeminiInlineMedia | GeminiUploadedMedia, ...],
        max_output_tokens: int | None,
        temperature: float | None,
    ) -> GeminiGeneration:
        self.generate_calls.append(
            {
                "model": model,
                "contents": contents,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
            }
        )
        self.generate_started.set()
        if self.generate_gate is not None:
            await self.generate_gate.wait()
        if self.generate_error is not None:
            raise self.generate_error
        if self.generation_responses:
            return self.generation_responses.pop(0)
        return self.generation

    async def delete_file(self, *, name: str) -> None:
        self.deleted_names.append(name)
        self.delete_started.set()
        if self.delete_gate is not None:
            await self.delete_gate.wait()
        if self.delete_error is not None:
            raise self.delete_error
        self.delete_finished = True

    async def aclose(self) -> None:
        self.close_started.set()
        if self.close_gate is not None:
            await self.close_gate.wait()
        self.closed = True
        self.close_finished = True


class FakeClientFactory:
    def __init__(self, client: FakeGeminiClient) -> None:
        self.client = client
        self.call_count = 0

    def __call__(self, *, api_key: str, base_url: str | None) -> GeminiClient:
        self.call_count += 1
        assert api_key
        assert base_url is None
        return self.client


def _config(
    *,
    inline_max_bytes: int = 64,
    timeout: float = 1,
    max_output_tokens: int | None = 512,
    temperature: float | None = 0.2,
) -> ProviderConfig:
    return ProviderConfig(
        adapter="gemini",
        model="gemini-test-model",
        credential_ref="gemini-test-credential",
        declared_capabilities={modality: True for modality in Modality},
        verified_capabilities={modality: True for modality in Modality},
        verified_joint_capabilities=[
            frozenset({Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO})
        ],
        adapter_options=AdapterOptions(
            inline_max_bytes=inline_max_bytes,
            request_timeout_seconds=timeout,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        ),
    )


def _provider(
    client: FakeGeminiClient,
    *,
    config: ProviderConfig | None = None,
) -> GeminiProvider:
    store = MemorySecretStore()
    store.set("gemini-test-credential", secrets.token_urlsafe(24))
    return GeminiProvider(
        provider_id="gemini",
        config=config or _config(),
        secret_store=store,
        client_factory=FakeClientFactory(client),
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
        question="Focus on direct sensory evidence.",
        detail=DetailLevel.AUTO,
        language="en",
    )


@pytest.mark.asyncio
async def test_small_image_uses_inline_bytes_without_remote_upload(tmp_path: Path) -> None:
    """Sending every image through Files API would retain small media remotely."""
    media_bytes = b"small-image-bytes"
    image = tmp_path / "small.png"
    image.write_bytes(media_bytes)
    client = FakeGeminiClient(GeminiGeneration(text=_response_text(Modality.IMAGE)))

    result = await _provider(client).sense(
        _request(
            image,
            media_kind=MediaKind.IMAGE,
            mime_type="image/png",
            modalities=frozenset({Modality.IMAGE}),
        )
    )

    contents = cast(tuple[object, ...], client.generate_calls[0]["contents"])
    assert contents[0] == GeminiInlineMedia(data=media_bytes, mime_type="image/png")
    assert isinstance(contents[1], str)
    assert client.uploads == []
    assert client.deleted_names == []
    assert result.remote_file_deleted is None
    assert result.provider_id == "gemini"
    assert result.model == "gemini-test-model"
    assert set(result.observations) == {Modality.IMAGE}
    assert result.observations[Modality.IMAGE].summary == "Observed image."


@pytest.mark.asyncio
async def test_video_uploads_once_and_places_prompt_after_media(tmp_path: Path) -> None:
    """Reversing video part order or re-uploading would violate Gemini video guidance."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    client = FakeGeminiClient(
        GeminiGeneration(text=_response_text(Modality.VIDEO_VISUAL))
    )

    result = await _provider(client).sense(
        _request(
            video,
            media_kind=MediaKind.VIDEO,
            mime_type="video/mp4",
            modalities=frozenset({Modality.VIDEO_VISUAL}),
            duration_seconds=8,
        )
    )

    contents = cast(tuple[object, ...], client.generate_calls[0]["contents"])
    assert contents[0] == GeminiUploadedMedia(
        uri="https://files.example.invalid/test-upload",
        mime_type="video/mp4",
    )
    assert isinstance(contents[1], str)
    assert 'exactly: ["video_visual"]' in contents[1]
    assert client.uploads == [(video, "video/mp4")]
    assert client.deleted_names == ["files/test-upload"]
    assert result.remote_file_deleted is True


@pytest.mark.parametrize(
    ("modality", "media_kind", "mime_type"),
    [
        pytest.param(Modality.IMAGE, MediaKind.IMAGE, "image/png", id="image"),
        pytest.param(
            Modality.VIDEO_VISUAL,
            MediaKind.VIDEO,
            "video/mp4",
            id="video-visual",
        ),
        pytest.param(
            Modality.VIDEO_AUDIO,
            MediaKind.VIDEO,
            "video/mp4",
            id="video-audio",
        ),
        pytest.param(
            Modality.VIDEO_AUDIO,
            MediaKind.AUDIO,
            "audio/wav",
            id="extracted-video-audio",
        ),
        pytest.param(Modality.AUDIO, MediaKind.AUDIO, "audio/wav", id="audio"),
        pytest.param(Modality.MUSIC, MediaKind.AUDIO, "audio/flac", id="music"),
    ],
)
@pytest.mark.asyncio
async def test_each_media_request_maps_to_its_exact_prompt_modality(
    tmp_path: Path,
    modality: Modality,
    media_kind: MediaKind,
    mime_type: str,
) -> None:
    """Using a physical media kind as the prompt modality would merge distinct reports."""
    media = tmp_path / "media.bin"
    media.write_bytes(b"media")
    client = FakeGeminiClient(GeminiGeneration(text=_response_text(modality)))

    await _provider(client).sense(
        _request(
            media,
            media_kind=media_kind,
            mime_type=mime_type,
            modalities=frozenset({modality}),
        )
    )

    contents = cast(tuple[object, ...], client.generate_calls[0]["contents"])
    prompt = next(item for item in contents if isinstance(item, str))
    assert f'exactly: ["{modality.value}"]' in prompt
    if media_kind is MediaKind.AUDIO:
        assert client.uploads == [(media, mime_type)]


@pytest.mark.asyncio
async def test_joint_video_returns_two_envelopes_from_one_upload(tmp_path: Path) -> None:
    """Splitting a verified joint request would upload the same private video twice."""
    video = tmp_path / "joint.mp4"
    video.write_bytes(b"joint-video")
    modalities = frozenset({Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO})
    client = FakeGeminiClient(
        GeminiGeneration(
            text=_response_text(Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO)
        )
    )

    result = await _provider(client).sense(
        _request(
            video,
            media_kind=MediaKind.VIDEO,
            mime_type="video/mp4",
            modalities=modalities,
            duration_seconds=12,
        )
    )

    assert client.uploads == [(video, "video/mp4")]
    assert len(client.generate_calls) == 1
    assert set(result.observations) == modalities
    assert result.observations[Modality.VIDEO_VISUAL].modality is Modality.VIDEO_VISUAL
    assert result.observations[Modality.VIDEO_AUDIO].modality is Modality.VIDEO_AUDIO
    assert result.remote_file_deleted is True


@pytest.mark.asyncio
async def test_uploaded_file_is_deleted_when_normalization_fails(tmp_path: Path) -> None:
    """Parsing before a cleanup finally block would leak an uploaded file on malformed JSON."""
    audio = tmp_path / "bad.wav"
    audio.write_bytes(b"audio")
    client = FakeGeminiClient(GeminiGeneration(text="not-json"))

    with pytest.raises(SensoryError) as caught:
        await _provider(client).sense(
            _request(
                audio,
                media_kind=MediaKind.AUDIO,
                mime_type="audio/wav",
                modalities=frozenset({Modality.AUDIO}),
            )
        )

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert len(client.generate_calls) == 2
    assert client.deleted_names == ["files/test-upload"]


@pytest.mark.asyncio
async def test_music_retries_malformed_json_once_with_zero_default_temperature(
    tmp_path: Path,
) -> None:
    """Structured music output should recover without inheriting Gemini's randomness."""
    audio = tmp_path / "music.wav"
    audio.write_bytes(b"music")
    client = FakeGeminiClient(GeminiGeneration(text=_response_text(Modality.MUSIC)))
    client.generation_responses = [
        GeminiGeneration(text="{malformed-json"),
        GeminiGeneration(text=_response_text(Modality.MUSIC)),
    ]

    result = await _provider(client, config=_config(temperature=None)).sense(
        _request(
            audio,
            media_kind=MediaKind.AUDIO,
            mime_type="audio/wav",
            modalities=frozenset({Modality.MUSIC}),
            duration_seconds=1,
        )
    )

    assert [call["temperature"] for call in client.generate_calls] == [0.0, 0.0]
    assert client.uploads == [(audio, "audio/wav")]
    assert client.deleted_names == ["files/test-upload"]
    assert result.observations[Modality.MUSIC].summary == "Observed music."


@pytest.mark.asyncio
async def test_uploaded_file_is_deleted_after_timeout(tmp_path: Path) -> None:
    """Timing out generation without shielded cleanup would retain uploaded media."""
    audio = tmp_path / "slow.wav"
    audio.write_bytes(b"audio")
    client = FakeGeminiClient(GeminiGeneration(text=_response_text(Modality.AUDIO)))
    client.generate_gate = asyncio.Event()
    provider = _provider(client, config=_config(timeout=0.01))

    with pytest.raises(SensoryError) as caught:
        await provider.sense(
            _request(
                audio,
                media_kind=MediaKind.AUDIO,
                mime_type="audio/wav",
                modalities=frozenset({Modality.AUDIO}),
            )
        )

    assert caught.value.code is ErrorCode.PROVIDER_TIMEOUT
    assert caught.value.retryable is True
    assert client.deleted_names == ["files/test-upload"]


@pytest.mark.asyncio
async def test_uploaded_file_is_deleted_when_processing_wait_times_out(
    tmp_path: Path,
) -> None:
    """Starting the cleanup scope after file processing would leak timed-out uploads."""
    video = tmp_path / "processing-timeout.mp4"
    video.write_bytes(b"video")
    client = FakeGeminiClient(
        GeminiGeneration(text=_response_text(Modality.VIDEO_VISUAL))
    )
    client.wait_gate = asyncio.Event()

    with pytest.raises(SensoryError) as caught:
        await _provider(client, config=_config(timeout=0.01)).sense(
            _request(
                video,
                media_kind=MediaKind.VIDEO,
                mime_type="video/mp4",
                modalities=frozenset({Modality.VIDEO_VISUAL}),
            )
        )

    assert caught.value.code is ErrorCode.PROVIDER_TIMEOUT
    assert client.deleted_names == ["files/test-upload"]


@pytest.mark.asyncio
async def test_uploaded_file_is_deleted_after_cancellation(tmp_path: Path) -> None:
    """Task cancellation must not bypass deletion of an upload already created."""
    video = tmp_path / "cancel.mp4"
    video.write_bytes(b"video")
    client = FakeGeminiClient(
        GeminiGeneration(text=_response_text(Modality.VIDEO_VISUAL))
    )
    client.generate_gate = asyncio.Event()
    task = asyncio.create_task(
        _provider(client).sense(
            _request(
                video,
                media_kind=MediaKind.VIDEO,
                mime_type="video/mp4",
                modalities=frozenset({Modality.VIDEO_VISUAL}),
            )
        )
    )
    await client.generate_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert client.deleted_names == ["files/test-upload"]


@pytest.mark.asyncio
async def test_uploaded_file_is_deleted_when_processing_wait_is_cancelled(
    tmp_path: Path,
) -> None:
    """Cancelling file processing after upload must still delete the remote object."""
    video = tmp_path / "processing-cancel.mp4"
    video.write_bytes(b"video")
    client = FakeGeminiClient(
        GeminiGeneration(text=_response_text(Modality.VIDEO_VISUAL))
    )
    client.wait_gate = asyncio.Event()
    task = asyncio.create_task(
        _provider(client).sense(
            _request(
                video,
                media_kind=MediaKind.VIDEO,
                mime_type="video/mp4",
                modalities=frozenset({Modality.VIDEO_VISUAL}),
            )
        )
    )
    await client.wait_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert client.deleted_names == ["files/test-upload"]


@pytest.mark.asyncio
async def test_remote_delete_reaches_terminal_state_before_two_cancellations_propagate(
    tmp_path: Path,
) -> None:
    """A second cancellation must not cancel the independently owned delete task."""
    video = tmp_path / "double-cancel.mp4"
    video.write_bytes(b"video")
    client = FakeGeminiClient(
        GeminiGeneration(text=_response_text(Modality.VIDEO_VISUAL))
    )
    client.delete_gate = asyncio.Event()
    task = asyncio.create_task(
        _provider(client).sense(
            _request(
                video,
                media_kind=MediaKind.VIDEO,
                mime_type="video/mp4",
                modalities=frozenset({Modality.VIDEO_VISUAL}),
            )
        )
    )
    await client.delete_started.wait()

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False
    client.delete_gate.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.delete_finished is True
    assert client.closed is True


@pytest.mark.asyncio
async def test_provider_close_reaches_terminal_state_before_cancellation_propagates(
    tmp_path: Path,
) -> None:
    """Cancellation during provider close must not abandon the injected client close."""
    image = tmp_path / "close-cancel.png"
    image.write_bytes(b"image")
    client = FakeGeminiClient(GeminiGeneration(text=_response_text(Modality.IMAGE)))
    client.close_gate = asyncio.Event()
    task = asyncio.create_task(
        _provider(client).sense(
            _request(
                image,
                media_kind=MediaKind.IMAGE,
                mime_type="image/png",
                modalities=frozenset({Modality.IMAGE}),
            )
        )
    )
    await client.close_started.wait()

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False
    client.close_gate.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.close_finished is True


@pytest.mark.parametrize(
    ("failure", "expected_code", "retryable"),
    [
        pytest.param(
            GeminiClientFailure(GeminiFailureKind.AUTH),
            ErrorCode.PROVIDER_AUTH_FAILED,
            False,
            id="authentication",
        ),
        pytest.param(
            GeminiClientFailure(GeminiFailureKind.SAFETY),
            ErrorCode.PROVIDER_SAFETY_REJECTED,
            False,
            id="safety",
        ),
        pytest.param(
            GeminiClientFailure(GeminiFailureKind.TIMEOUT),
            ErrorCode.PROVIDER_TIMEOUT,
            True,
            id="sdk-timeout",
        ),
    ],
)
@pytest.mark.asyncio
async def test_client_failure_maps_to_stable_public_error(
    tmp_path: Path,
    failure: GeminiClientFailure,
    expected_code: ErrorCode,
    retryable: bool,
) -> None:
    """Forwarding SDK categories would expose unstable provider errors to MCP callers."""
    audio = tmp_path / "failure.wav"
    audio.write_bytes(b"audio")
    client = FakeGeminiClient(GeminiGeneration(text=_response_text(Modality.AUDIO)))
    client.generate_error = failure

    with pytest.raises(SensoryError) as caught:
        await _provider(client).sense(
            _request(
                audio,
                media_kind=MediaKind.AUDIO,
                mime_type="audio/wav",
                modalities=frozenset({Modality.AUDIO}),
            )
        )

    assert caught.value.code is expected_code
    assert caught.value.retryable is retryable
    assert caught.value.cause is None
    assert caught.value.__context__ is None
    assert client.deleted_names == ["files/test-upload"]


@pytest.mark.asyncio
async def test_safety_refusal_without_text_maps_to_stable_error(tmp_path: Path) -> None:
    """Treating a blocked empty response as parse failure would lose the safety category."""
    video = tmp_path / "blocked.mp4"
    video.write_bytes(b"video")
    client = FakeGeminiClient(GeminiGeneration(text=None, safety_rejected=True))

    with pytest.raises(SensoryError) as caught:
        await _provider(client).sense(
            _request(
                video,
                media_kind=MediaKind.VIDEO,
                mime_type="video/mp4",
                modalities=frozenset({Modality.VIDEO_VISUAL}),
            )
        )

    assert caught.value.code is ErrorCode.PROVIDER_SAFETY_REJECTED
    assert client.deleted_names == ["files/test-upload"]


@pytest.mark.asyncio
async def test_delete_failure_does_not_mask_success(tmp_path: Path) -> None:
    """Raising a cleanup exception would discard an otherwise valid normalized result."""
    audio = tmp_path / "cleanup.wav"
    audio.write_bytes(b"audio")
    client = FakeGeminiClient(GeminiGeneration(text=_response_text(Modality.AUDIO)))
    client.delete_error = OSError("private cleanup detail")

    result = await _provider(client).sense(
        _request(
            audio,
            media_kind=MediaKind.AUDIO,
            mime_type="audio/wav",
            modalities=frozenset({Modality.AUDIO}),
        )
    )

    assert result.observations[Modality.AUDIO].summary == "Observed audio."
    assert result.remote_file_deleted is False


@pytest.mark.asyncio
async def test_delete_failure_does_not_replace_primary_error(tmp_path: Path) -> None:
    """Cleanup failure must not replace the stable error that caused the request to fail."""
    video = tmp_path / "primary.mp4"
    video.write_bytes(b"video")
    client = FakeGeminiClient(GeminiGeneration(text="not-json"))
    client.delete_error = OSError("private cleanup detail")

    with pytest.raises(SensoryError) as caught:
        await _provider(client).sense(
            _request(
                video,
                media_kind=MediaKind.VIDEO,
                mime_type="video/mp4",
                modalities=frozenset({Modality.VIDEO_VISUAL}),
            )
        )

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert "cleanup" not in str(caught.value).lower()


@pytest.mark.asyncio
async def test_tuning_options_are_forwarded_from_closed_adapter_options(
    tmp_path: Path,
) -> None:
    """Ignoring typed tuning fields would silently use uncontrolled SDK defaults."""
    image = tmp_path / "options.png"
    image.write_bytes(b"image")
    client = FakeGeminiClient(GeminiGeneration(text=_response_text(Modality.IMAGE)))

    await _provider(
        client,
        config=_config(max_output_tokens=321, temperature=0.7),
    ).sense(
        _request(
            image,
            media_kind=MediaKind.IMAGE,
            mime_type="image/png",
            modalities=frozenset({Modality.IMAGE}),
        )
    )

    assert client.generate_calls[0]["max_output_tokens"] == 321
    assert client.generate_calls[0]["temperature"] == 0.7


@pytest.mark.asyncio
async def test_logs_and_public_error_never_include_private_transport_data(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Logging an arbitrary SDK exception could expose keys, media, headers, or responses."""
    credential = secrets.token_urlsafe(24)
    media_marker = secrets.token_hex(24).encode()
    raw_marker = secrets.token_hex(24)
    header_marker = secrets.token_hex(24)
    image = tmp_path / "private.png"
    image.write_bytes(media_marker)
    client = FakeGeminiClient(GeminiGeneration(text=_response_text(Modality.IMAGE)))
    client.generate_error = RuntimeError(
        f"credential={credential}; header={header_marker}; response={raw_marker}"
    )
    store = MemorySecretStore()
    store.set("gemini-test-credential", credential)
    provider = GeminiProvider(
        provider_id="gemini",
        config=_config(),
        secret_store=store,
        client_factory=FakeClientFactory(client),
    )

    with caplog.at_level(logging.DEBUG), pytest.raises(SensoryError) as caught:
        await provider.sense(
            _request(
                image,
                media_kind=MediaKind.IMAGE,
                mime_type="image/png",
                modalities=frozenset({Modality.IMAGE}),
            )
        )

    rendered = "\n".join(record.getMessage() for record in caplog.records)
    public_text = str(caught.value)
    for private_value in (credential, media_marker.decode(), raw_marker, header_marker):
        assert private_value not in rendered
        assert private_value not in public_text
    failure_record = next(
        record
        for record in caplog.records
        if record.getMessage() == "provider_request_failed"
    )
    safe_record = cast(Any, failure_record)
    assert len(cast(str, safe_record.request_id)) == 32
    assert safe_record.provider_id == "gemini"
    assert safe_record.model == "gemini-test-model"
    assert safe_record.modality == "image"
    assert safe_record.latency_bucket in {
        "under_1s",
        "1_to_5s",
        "5_to_30s",
        "30s_or_more",
    }
    assert safe_record.error_code == ErrorCode.PROVIDER_UNAVAILABLE.value
    assert caught.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert caught.value.cause is None


@pytest.mark.asyncio
async def test_client_is_closed_after_each_call(tmp_path: Path) -> None:
    """Leaving one SDK client open per call would leak connection-pool resources."""
    image = tmp_path / "close.png"
    image.write_bytes(b"image")
    client = FakeGeminiClient(GeminiGeneration(text=_response_text(Modality.IMAGE)))

    await _provider(client).sense(
        _request(
            image,
            media_kind=MediaKind.IMAGE,
            mime_type="image/png",
            modalities=frozenset({Modality.IMAGE}),
        )
    )

    assert client.closed is True


class _SyncFilesForbidden:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def upload(self, **_kwargs: object) -> object:
        self.calls.append("upload")
        raise AssertionError("the synchronous Files API must not be used")

    def get(self, **_kwargs: object) -> object:
        self.calls.append("get")
        raise AssertionError("the synchronous Files API must not be used")

    def delete(self, **_kwargs: object) -> object:
        self.calls.append("delete")
        raise AssertionError("the synchronous Files API must not be used")


class _AsyncFilesSpy:
    def __init__(self) -> None:
        self.upload_calls: list[dict[str, object]] = []
        self.get_calls: list[dict[str, object]] = []
        self.delete_calls: list[dict[str, object]] = []
        self.upload_started = asyncio.Event()
        self.upload_gate: asyncio.Event | None = None
        self.upload_cancelled = False
        self.get_responses: list[object] = [
            SimpleNamespace(
                name="files/official-upload",
                uri="https://files.example.invalid/official-upload",
                mime_type="video/mp4",
                state=SimpleNamespace(name="ACTIVE"),
            )
        ]

    async def upload(self, **kwargs: object) -> object:
        self.upload_calls.append(kwargs)
        self.upload_started.set()
        try:
            if self.upload_gate is not None:
                await self.upload_gate.wait()
        except asyncio.CancelledError:
            self.upload_cancelled = True
            raise
        return SimpleNamespace(
            name="files/official-upload",
            uri="https://files.example.invalid/official-upload",
            mime_type="video/mp4",
        )

    async def get(self, **kwargs: object) -> object:
        self.get_calls.append(kwargs)
        return self.get_responses.pop(0)

    async def delete(self, **kwargs: object) -> object:
        self.delete_calls.append(kwargs)
        return SimpleNamespace()


class _AsyncModelsSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.response: object = SimpleNamespace(
            text="sanitized response text",
            prompt_feedback=SimpleNamespace(block_reason=None),
            candidates=[],
        )

    async def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


class _OfficialSDKClientSpy:
    def __init__(self) -> None:
        self.files = _SyncFilesForbidden()
        self.async_files = _AsyncFilesSpy()
        self.async_models = _AsyncModelsSpy()
        self.async_close_started = asyncio.Event()
        self.async_close_gate: asyncio.Event | None = None
        self.async_close_calls = 0
        self.async_close_error: Exception | None = None
        self.sync_close_calls = 0
        self.sync_close_error: Exception | None = None
        self.aio = SimpleNamespace(
            files=self.async_files,
            models=self.async_models,
            aclose=self._async_close,
        )

    async def _async_close(self) -> None:
        self.async_close_calls += 1
        self.async_close_started.set()
        if self.async_close_gate is not None:
            await self.async_close_gate.wait()
        if self.async_close_error is not None:
            raise self.async_close_error

    def close(self) -> None:
        self.sync_close_calls += 1
        if self.sync_close_error is not None:
            raise self.sync_close_error


async def _event_set_within(event: asyncio.Event, timeout: float = 0.1) -> bool:
    try:
        await asyncio.wait_for(event.wait(), timeout)
    except TimeoutError:
        return False
    return True


def _install_official_sdk_spies(
    monkeypatch: pytest.MonkeyPatch,
    sdk_client: _OfficialSDKClientSpy,
) -> dict[str, list[dict[str, object]]]:
    typed_calls: dict[str, list[dict[str, object]]] = {
        "http_options": [],
        "upload_config": [],
        "part_text": [],
        "part_bytes": [],
        "part_uri": [],
        "generation_config": [],
        "content": [],
        "client": [],
    }

    class PartFactory:
        @staticmethod
        def from_text(**kwargs: object) -> tuple[str, dict[str, object]]:
            typed_calls["part_text"].append(kwargs)
            return "text", kwargs

        @staticmethod
        def from_bytes(**kwargs: object) -> tuple[str, dict[str, object]]:
            typed_calls["part_bytes"].append(kwargs)
            return "bytes", kwargs

        @staticmethod
        def from_uri(**kwargs: object) -> tuple[str, dict[str, object]]:
            typed_calls["part_uri"].append(kwargs)
            return "uri", kwargs

    def typed_factory(name: str) -> Any:
        def factory(**kwargs: object) -> tuple[str, dict[str, object]]:
            typed_calls[name].append(kwargs)
            return name, kwargs

        return factory

    def client_factory(**kwargs: object) -> _OfficialSDKClientSpy:
        typed_calls["client"].append(kwargs)
        return sdk_client

    monkeypatch.setattr(google_genai, "Client", client_factory)
    monkeypatch.setattr(google_types, "HttpOptions", typed_factory("http_options"))
    monkeypatch.setattr(
        google_types,
        "UploadFileConfig",
        typed_factory("upload_config"),
    )
    monkeypatch.setattr(google_types, "Part", PartFactory)
    monkeypatch.setattr(
        google_types,
        "GenerateContentConfig",
        typed_factory("generation_config"),
    )
    monkeypatch.setattr(google_types, "Content", typed_factory("content"))
    return typed_calls


@pytest.mark.asyncio
async def test_official_wrapper_uses_async_files_with_typed_upload_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker-thread Files calls can finish after cancellation without a cleanup identity."""
    sdk_client = _OfficialSDKClientSpy()
    typed_calls = _install_official_sdk_spies(monkeypatch, sdk_client)
    credential = secrets.token_urlsafe(24)
    wrapper = _OfficialGoogleGenAIClient(
        api_key=credential,
        base_url="https://gateway.example.invalid",
    )
    media = tmp_path / "official.mp4"
    media.write_bytes(b"video")

    uploaded = await wrapper.upload_file(path=media, mime_type="video/mp4")
    active = await wrapper.wait_until_active(uploaded)
    await wrapper.delete_file(name=uploaded.name)

    assert typed_calls["client"] == [
        {
            "api_key": credential,
            "http_options": (
                "http_options",
                {"base_url": "https://gateway.example.invalid"},
            ),
        }
    ]
    assert typed_calls["upload_config"] == [{"mime_type": "video/mp4"}]
    assert sdk_client.async_files.upload_calls == [
        {"file": media, "config": ("upload_config", {"mime_type": "video/mp4"})}
    ]
    assert sdk_client.async_files.get_calls == [{"name": "files/official-upload"}]
    assert sdk_client.async_files.delete_calls == [
        {"name": "files/official-upload"}
    ]
    assert sdk_client.files.calls == []
    assert active == uploaded


@pytest.mark.asyncio
async def test_official_async_upload_propagates_cancellation_without_sync_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation must reach the SDK upload operation instead of abandoning a thread."""
    sdk_client = _OfficialSDKClientSpy()
    sdk_client.async_files.upload_gate = asyncio.Event()
    _install_official_sdk_spies(monkeypatch, sdk_client)
    wrapper = _OfficialGoogleGenAIClient(
        api_key=secrets.token_urlsafe(24),
        base_url=None,
    )
    media = tmp_path / "cancel-upload.wav"
    media.write_bytes(b"audio")
    task = asyncio.create_task(
        wrapper.upload_file(path=media, mime_type="audio/wav")
    )
    try:
        assert await _event_set_within(sdk_client.async_files.upload_started)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError, GeminiClientFailure):
            await task

    assert sdk_client.async_files.upload_cancelled is True
    assert sdk_client.files.calls == []


@pytest.mark.asyncio
async def test_official_wrapper_constructs_typed_parts_and_uses_async_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bypassing typed parts could silently change media encoding or request ordering."""
    sdk_client = _OfficialSDKClientSpy()
    typed_calls = _install_official_sdk_spies(monkeypatch, sdk_client)
    wrapper = _OfficialGoogleGenAIClient(
        api_key=secrets.token_urlsafe(24),
        base_url=None,
    )

    generation = await wrapper.generate_content(
        model="gemini-test-model",
        contents=(
            GeminiInlineMedia(data=b"image", mime_type="image/png"),
            GeminiUploadedMedia(
                uri="https://files.example.invalid/video",
                mime_type="video/mp4",
            ),
            "prompt",
        ),
        max_output_tokens=321,
        temperature=0.7,
    )

    assert typed_calls["part_bytes"] == [
        {"data": b"image", "mime_type": "image/png"}
    ]
    assert typed_calls["part_uri"] == [
        {
            "file_uri": "https://files.example.invalid/video",
            "mime_type": "video/mp4",
        }
    ]
    assert typed_calls["part_text"] == [{"text": "prompt"}]
    assert typed_calls["generation_config"] == [
        {
            "max_output_tokens": 321,
            "temperature": 0.7,
            "response_mime_type": "application/json",
        }
    ]
    assert len(sdk_client.async_models.calls) == 1
    assert sdk_client.async_models.calls[0]["model"] == "gemini-test-model"
    assert generation == GeminiGeneration(text="sanitized response text")


@pytest.mark.asyncio
async def test_official_wrapper_waits_for_explicit_active_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treating a missing or unknown state as active could generate from unready media."""
    sdk_client = _OfficialSDKClientSpy()
    sdk_client.async_files.get_responses = [
        SimpleNamespace(state=None),
        SimpleNamespace(state=SimpleNamespace(name="QUEUED")),
        SimpleNamespace(
            name="files/official-upload",
            uri="https://files.example.invalid/official-upload",
            mime_type="video/mp4",
            state=SimpleNamespace(name="ACTIVE"),
        ),
    ]
    _install_official_sdk_spies(monkeypatch, sdk_client)
    monkeypatch.setattr(
        "cove_sensory_mcp.providers.gemini._FILE_POLL_INTERVAL_SECONDS",
        0,
    )
    wrapper = _OfficialGoogleGenAIClient(
        api_key=secrets.token_urlsafe(24),
        base_url=None,
    )
    remote = GeminiRemoteFile(
        name="files/official-upload",
        uri="https://files.example.invalid/official-upload",
        mime_type="video/mp4",
    )

    active = await wrapper.wait_until_active(remote)

    assert active == remote
    assert sdk_client.async_files.get_calls == [
        {"name": "files/official-upload"},
        {"name": "files/official-upload"},
        {"name": "files/official-upload"},
    ]


@pytest.mark.asyncio
async def test_official_wrapper_rejects_explicit_failed_file_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Polling after an explicit Files failure would hide a terminal Provider rejection."""
    sdk_client = _OfficialSDKClientSpy()
    sdk_client.async_files.get_responses = [
        SimpleNamespace(state=SimpleNamespace(name="FAILED"))
    ]
    _install_official_sdk_spies(monkeypatch, sdk_client)
    wrapper = _OfficialGoogleGenAIClient(
        api_key=secrets.token_urlsafe(24),
        base_url=None,
    )
    remote = GeminiRemoteFile(
        name="files/official-upload",
        uri="https://files.example.invalid/official-upload",
        mime_type="video/mp4",
    )

    with pytest.raises(GeminiClientFailure) as caught:
        await wrapper.wait_until_active(remote)

    assert caught.value.kind is GeminiFailureKind.UNAVAILABLE
    assert len(sdk_client.async_files.get_calls) == 1


@pytest.mark.parametrize("failing_transport", ["async", "sync"])
@pytest.mark.asyncio
async def test_official_wrapper_closes_both_transports_when_either_close_fails(
    monkeypatch: pytest.MonkeyPatch,
    failing_transport: str,
) -> None:
    """Either close error must not prevent closure of the other SDK transport."""
    sdk_client = _OfficialSDKClientSpy()
    if failing_transport == "async":
        sdk_client.async_close_error = OSError("private async close detail")
    else:
        sdk_client.sync_close_error = OSError("private sync close detail")
    _install_official_sdk_spies(monkeypatch, sdk_client)
    wrapper = _OfficialGoogleGenAIClient(
        api_key=secrets.token_urlsafe(24),
        base_url=None,
    )

    with pytest.raises(GeminiClientFailure) as caught:
        await wrapper.aclose()

    assert caught.value.kind is GeminiFailureKind.UNAVAILABLE
    assert sdk_client.async_close_calls == 1
    assert sdk_client.sync_close_calls == 1


@pytest.mark.asyncio
async def test_official_wrapper_close_survives_two_cancellations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated cancellation must wait for both independently owned close operations."""
    sdk_client = _OfficialSDKClientSpy()
    sdk_client.async_close_gate = asyncio.Event()
    _install_official_sdk_spies(monkeypatch, sdk_client)
    wrapper = _OfficialGoogleGenAIClient(
        api_key=secrets.token_urlsafe(24),
        base_url=None,
    )
    task = asyncio.create_task(wrapper.aclose())
    await sdk_client.async_close_started.wait()

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False
    sdk_client.async_close_gate.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert sdk_client.async_close_calls == 1
    assert sdk_client.sync_close_calls == 1
