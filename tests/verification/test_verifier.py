"""Provider connectivity verification is atomic, exact, and privacy safe."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cove_sensory_mcp.config.schema import AppConfig, ProviderConfig
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.providers.base import (
    MediaKind,
    PreparedMedia,
    ProviderCallResult,
    ProviderRequest,
)
from cove_sensory_mcp.providers.registry import ProviderRegistry
from cove_sensory_mcp.reports.schemas import ObservationEnvelope
from cove_sensory_mcp.verification.assets import SelfTestAssetStore
from cove_sensory_mcp.verification.verifier import CapabilityVerifier

_VERIFIED_AT = datetime(2026, 8, 8, 4, 30, tzinfo=UTC)


class CountingConfigStore(ConfigStore):
    """Count verifier writes while retaining the real atomic YAML implementation."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.save_calls = 0

    def save(self, config: AppConfig) -> None:
        self.save_calls += 1
        super().save(config)

    def update(self, mutator: Callable[[AppConfig], AppConfig | None]) -> AppConfig:
        self.save_calls += 1
        return super().update(mutator)


def _provider_config(
    *,
    adapter: str = "gemini",
    verified: tuple[Modality, ...] = (),
) -> ProviderConfig:
    declared = {modality: True for modality in Modality}
    return ProviderConfig(
        adapter=adapter,
        model="test-model",
        credential_ref="test-credential",
        declared_capabilities=declared,
        verified_capabilities={modality: True for modality in verified},
        last_verified_at=_VERIFIED_AT if verified else None,
    )


def _observation(modality: Modality, summary: str) -> ObservationEnvelope:
    return ObservationEnvelope(
        modality=modality,
        summary=summary,
        segments=[],
        transcript=[],
        warnings=[],
        confidence="high",
    )


class RecordingProvider:
    """Record requests and return one normalized observation for each modality."""

    def __init__(
        self,
        summaries: dict[Modality, str | ObservationEnvelope | BaseException],
        *,
        before_return: Callable[[ProviderRequest], None] | None = None,
        returned_model: str = "test-model",
    ) -> None:
        self._summaries = summaries
        self._before_return = before_return
        self._returned_model = returned_model
        self.requests: list[ProviderRequest] = []

    async def sense(self, request: ProviderRequest) -> ProviderCallResult:
        self.requests.append(request)
        if self._before_return is not None:
            self._before_return(request)
        modality = next(iter(request.requested_modalities))
        outcome = self._summaries[modality]
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, ObservationEnvelope):
            observation = outcome
        else:
            observation = _observation(modality, outcome)
        return ProviderCallResult(
            observations={modality: observation},
            provider_id="vision",
            model=self._returned_model,
            remote_file_deleted=None,
        )


@pytest.fixture
def assets(tmp_path: Path) -> SelfTestAssetStore:
    """Inject tiny local fixtures; no packaged asset or network is consulted."""
    image = tmp_path / "tiny-image.png"
    video = tmp_path / "tiny-motion.mp4"
    audio = tmp_path / "tiny-audio.wav"
    image.write_bytes(b"image-fixture")
    video.write_bytes(b"video-fixture")
    audio.write_bytes(b"audio-fixture")
    return SelfTestAssetStore(
        {
            Modality.IMAGE: PreparedMedia(
                image,
                "image/png",
                MediaKind.IMAGE,
                None,
            ),
            Modality.VIDEO_VISUAL: PreparedMedia(
                video,
                "video/mp4",
                MediaKind.VIDEO,
                2.0,
            ),
            Modality.VIDEO_AUDIO: PreparedMedia(
                video,
                "video/mp4",
                MediaKind.VIDEO,
                2.0,
            ),
            Modality.AUDIO: PreparedMedia(
                audio,
                "audio/wav",
                MediaKind.AUDIO,
                1.0,
            ),
            Modality.MUSIC: PreparedMedia(
                audio,
                "audio/wav",
                MediaKind.AUDIO,
                1.0,
            ),
        },
        trusted_root=tmp_path,
    )


def _configured_store(
    tmp_path: Path, *, adapter: str = "gemini"
) -> CountingConfigStore:
    store = CountingConfigStore(tmp_path / "config.yaml")
    store.save(AppConfig(providers={"vision": _provider_config(adapter=adapter)}))
    store.save_calls = 0
    return store


def _verifier(
    store: ConfigStore,
    provider: RecordingProvider,
    assets: SelfTestAssetStore,
) -> CapabilityVerifier:
    return CapabilityVerifier(
        config_store=store,
        registry=ProviderRegistry({"vision": provider}),
        assets=assets,
        now=lambda: _VERIFIED_AT,
    )


@pytest.mark.asyncio
async def test_declared_image_and_video_are_verified_in_separate_provider_calls(
    tmp_path: Path,
    assets: SelfTestAssetStore,
) -> None:
    """Combining modalities would fail providers that verify each capability separately."""
    store = _configured_store(tmp_path)
    provider = RecordingProvider(
        {
            Modality.IMAGE: "A blue triangle is centered on a white background.",
            Modality.VIDEO_VISUAL: "A red ball moves from left to right.",
        }
    )

    results = await _verifier(store, provider, assets).verify(
        "vision",
        [Modality.IMAGE, Modality.VIDEO_VISUAL],
    )

    assert [(item.modality, item.verified, item.reason) for item in results] == [
        (Modality.IMAGE, True, None),
        (Modality.VIDEO_VISUAL, True, None),
    ]
    assert [request.requested_modalities for request in provider.requests] == [
        frozenset({Modality.IMAGE}),
        frozenset({Modality.VIDEO_VISUAL}),
    ]
    assert provider.requests[0].media.path.name == "tiny-image.png"
    assert provider.requests[1].media.path.name == "tiny-motion.mp4"
    assert all("blue" not in request.question.lower() for request in provider.requests)


@pytest.mark.asyncio
async def test_provider_receives_canonical_trusted_asset_not_cwd_same_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    canonical = trusted / "fixture.png"
    canonical.write_bytes(b"trusted")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "fixture.png").write_bytes(b"untrusted")
    monkeypatch.chdir(cwd)
    store = _configured_store(tmp_path)
    provider = RecordingProvider({Modality.IMAGE: "A blue triangle is visible."})
    relative_assets = SelfTestAssetStore(
        {
            Modality.IMAGE: PreparedMedia(
                Path("fixture.png"),
                "IMAGE/PNG",
                MediaKind.IMAGE,
                None,
            )
        },
        trusted_root=trusted,
    )

    result = await _verifier(store, provider, relative_assets).verify(
        "vision", [Modality.IMAGE]
    )

    assert result[0].verified is True
    requested_media = provider.requests[0].media
    assert requested_media.path == canonical.resolve(strict=True)
    assert requested_media.path.read_bytes() == b"trusted"
    assert requested_media.mime_type == "image/png"


@pytest.mark.parametrize(
    ("modality", "summary"),
    [
        (Modality.IMAGE, "I cannot confidently name the pictured object."),
        (Modality.VIDEO_VISUAL, "A generic clip is available."),
        (Modality.VIDEO_AUDIO, "Sound is present, described in another style."),
        (Modality.AUDIO, "Three short signals may be audible."),
        (Modality.MUSIC, "听到一段音乐，具体描述风格因模型而异。"),
    ],
)
@pytest.mark.asyncio
async def test_successful_structured_response_verifies_without_scoring_description(
    tmp_path: Path,
    assets: SelfTestAssetStore,
    modality: Modality,
    summary: str,
) -> None:
    """Model wording, language, and confidence must not decide connectivity."""
    store = _configured_store(tmp_path)

    result = await _verifier(
        store,
        RecordingProvider({modality: summary}),
        assets,
    ).verify("vision", [modality])

    assert [(item.verified, item.reason) for item in result] == [(True, None)]


@pytest.mark.asyncio
async def test_undeclared_modality_makes_no_provider_call_or_config_write(
    tmp_path: Path,
    assets: SelfTestAssetStore,
) -> None:
    store = CountingConfigStore(tmp_path / "config.yaml")
    provider_config = _provider_config()
    provider_config.declared_capabilities[Modality.MUSIC] = False
    store.save(AppConfig(providers={"vision": provider_config}))
    store.save_calls = 0
    provider = RecordingProvider({Modality.MUSIC: "A piano scale rises upward."})

    with pytest.raises(SensoryError) as caught:
        await _verifier(store, provider, assets).verify("vision", [Modality.MUSIC])

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert provider.requests == []
    assert store.save_calls == 0


@pytest.mark.asyncio
async def test_batch_updates_requested_capabilities_once_and_preserves_unrelated_state(
    tmp_path: Path,
    assets: SelfTestAssetStore,
) -> None:
    """Writing per call could expose partial verification and erase unrelated successes."""
    store = CountingConfigStore(tmp_path / "config.yaml")
    store.save(
        AppConfig(
            providers={
                "vision": _provider_config(
                    verified=(Modality.VIDEO_VISUAL, Modality.AUDIO)
                )
            }
        )
    )
    store.save_calls = 0
    provider = RecordingProvider(
        {
            Modality.IMAGE: "The test image contains a blue triangle.",
            Modality.VIDEO_VISUAL: "Generic moving imagery.",
        }
    )

    results = await _verifier(store, provider, assets).verify(
        "vision",
        [Modality.IMAGE, Modality.VIDEO_VISUAL],
    )

    assert [item.verified for item in results] == [True, True]
    assert store.save_calls == 1
    saved = store.load().providers["vision"]
    assert saved.verified_capabilities == {
        Modality.IMAGE: True,
        Modality.VIDEO_VISUAL: True,
        Modality.AUDIO: True,
    }
    assert saved.last_verified_at == _VERIFIED_AT


@pytest.mark.asyncio
async def test_verifier_reloads_before_atomic_write_to_preserve_unrelated_changes(
    tmp_path: Path,
    assets: SelfTestAssetStore,
) -> None:
    """Saving the start snapshot would lose configuration changed during provider I/O."""
    store = _configured_store(tmp_path)
    external = ConfigStore(store.path)
    changed = False

    def add_unrelated_provider(_: ProviderRequest) -> None:
        nonlocal changed
        if changed:
            return
        changed = True
        latest = external.load()
        latest.providers["unrelated"] = ProviderConfig(
            adapter="gemini",
            model="other-model",
            credential_ref="other-credential",
            declared_capabilities={Modality.AUDIO: True},
        )
        external.save(latest)

    provider = RecordingProvider(
        {Modality.IMAGE: "A white card displays a blue triangle."},
        before_return=add_unrelated_provider,
    )

    await _verifier(store, provider, assets).verify("vision", [Modality.IMAGE])

    assert store.save_calls == 1
    assert set(store.load().providers) == {"vision", "unrelated"}


@pytest.mark.asyncio
async def test_provider_identity_conflict_during_remote_call_aborts_without_overwrite(
    tmp_path: Path,
    assets: SelfTestAssetStore,
) -> None:
    """A changed model must not receive verification earned by an earlier model."""
    store = _configured_store(tmp_path)
    external = ConfigStore(store.path)

    def change_model(_: ProviderRequest) -> None:
        def mutate(config: AppConfig) -> None:
            config.providers["vision"].model = "late-model"
            config.allowed_media_roots.append("late-setting")

        external.update(mutate)

    provider = RecordingProvider(
        {Modality.IMAGE: "A blue triangle is visible."},
        before_return=change_model,
    )

    with pytest.raises(SensoryError) as caught:
        await _verifier(store, provider, assets).verify("vision", [Modality.IMAGE])

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    saved = store.load()
    assert saved.providers["vision"].model == "late-model"
    assert saved.providers["vision"].verified_capabilities == {}
    assert saved.allowed_media_roots == ["late-setting"]


@pytest.mark.asyncio
async def test_cancellation_and_local_asset_errors_never_partially_write(
    tmp_path: Path,
    assets: SelfTestAssetStore,
) -> None:
    """Cancellation or missing local fixtures must leave prior verification untouched."""
    store = _configured_store(tmp_path)
    cancelling = RecordingProvider({Modality.IMAGE: asyncio.CancelledError()})

    with pytest.raises(asyncio.CancelledError):
        await _verifier(store, cancelling, assets).verify("vision", [Modality.IMAGE])

    assert store.save_calls == 0

    missing_assets = SelfTestAssetStore({}, trusted_root=tmp_path)
    missing_provider = RecordingProvider({Modality.IMAGE: "blue triangle"})
    with pytest.raises(SensoryError) as caught:
        await _verifier(
            store,
            missing_provider,
            missing_assets,
        ).verify("vision", [Modality.IMAGE])

    assert caught.value.code is ErrorCode.SOURCE_NOT_FOUND
    assert missing_provider.requests == []
    assert store.save_calls == 0


@pytest.mark.asyncio
async def test_provider_configuration_error_aborts_batch_without_any_write(
    tmp_path: Path,
    assets: SelfTestAssetStore,
) -> None:
    """A later configuration fault must not commit an earlier successful response."""
    store = _configured_store(tmp_path)
    provider = RecordingProvider(
        {
            Modality.IMAGE: "A blue triangle is visible.",
            Modality.VIDEO_VISUAL: SensoryError(
                ErrorCode.CONFIG_INVALID,
                "private-provider-config-marker",
            ),
        }
    )

    with pytest.raises(SensoryError) as caught:
        await _verifier(store, provider, assets).verify(
            "vision",
            [Modality.IMAGE, Modality.VIDEO_VISUAL],
        )

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert "private-provider-config-marker" not in str(caught.value)
    assert store.save_calls == 0
    assert store.load().providers["vision"].verified_capabilities == {}


@pytest.mark.asyncio
async def test_failures_return_only_stable_reasons_without_paths_or_provider_text(
    tmp_path: Path,
    assets: SelfTestAssetStore,
) -> None:
    """Self-test output must not reveal a fixture path or raw provider diagnostics."""
    store = _configured_store(tmp_path)
    marker = "private-provider-body-marker"
    provider = RecordingProvider(
        {
            Modality.IMAGE: SensoryError(
                ErrorCode.PROVIDER_SAFETY_REJECTED,
                marker,
            )
        },
        returned_model="wrong-private-model",
    )

    results = await _verifier(store, provider, assets).verify(
        "vision", [Modality.IMAGE]
    )

    assert results[0].reason == ErrorCode.PROVIDER_SAFETY_REJECTED.value
    public = repr(results)
    assert marker not in public
    assert str(tmp_path) not in public
    assert "wrong-private-model" not in public


@pytest.mark.asyncio
async def test_mismatched_provider_model_fails_through_the_exact_executor_boundary(
    tmp_path: Path,
    assets: SelfTestAssetStore,
) -> None:
    """Trusting adapter metadata would verify a model other than the configured one."""
    store = _configured_store(tmp_path)
    provider = RecordingProvider(
        {Modality.IMAGE: "A blue triangle."},
        returned_model="wrong-model",
    )

    results = await _verifier(store, provider, assets).verify(
        "vision", [Modality.IMAGE]
    )

    assert len(results) == 1
    assert results[0].verified is False
    assert results[0].reason == ErrorCode.PROVIDER_CAPABILITY_REJECTED.value
