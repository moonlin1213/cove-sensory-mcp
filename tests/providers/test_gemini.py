from __future__ import annotations

import asyncio
import json
import logging
import secrets
from pathlib import Path
from typing import cast

import pytest

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
        return self.generation

    async def delete_file(self, *, name: str) -> None:
        self.deleted_names.append(name)
        if self.delete_error is not None:
            raise self.delete_error

    async def aclose(self) -> None:
        self.closed = True


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
    assert isinstance(contents[0], str)
    assert contents[1] == GeminiInlineMedia(data=media_bytes, mime_type="image/png")
    assert client.uploads == []
    assert client.deleted_names == []
    assert result.remote_file_deleted is None
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
    assert client.deleted_names == ["files/test-upload"]


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
