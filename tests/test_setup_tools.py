"""Public setup handlers remain safe before provider adapters exist."""

from __future__ import annotations

import socket
from dataclasses import fields

import pytest

from cove_sensory_mcp.config.schema import AppConfig, ProviderConfig
from cove_sensory_mcp.config.secrets import MemorySecretStore
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.models import Modality, RouteConfig
from cove_sensory_mcp.providers.base import (
    MediaKind,
    PreparedMedia,
    ProviderCallResult,
    ProviderRequest,
)
from cove_sensory_mcp.providers.registry import ProviderRegistry
from cove_sensory_mcp.reports.schemas import ObservationEnvelope
from cove_sensory_mcp.services import AppServices
from cove_sensory_mcp.tools.setup import (
    sensory_self_test,
    sensory_setup_guide,
    sensory_status,
)
from cove_sensory_mcp.verification.assets import SelfTestAssetStore


@pytest.fixture
def services(tmp_path) -> AppServices:
    """Provide isolated, non-persistent dependencies for setup handler tests."""
    return AppServices(
        config_store=ConfigStore(tmp_path / "config.yaml"),
        secret_store=MemorySecretStore(),
    )


def test_app_services_keeps_only_config_and_secret_boundaries() -> None:
    """Adding provider runtime state here would change the foundation composition contract."""
    assert [service_field.name for service_field in fields(AppServices)] == [
        "config_store",
        "secret_store",
    ]


@pytest.mark.asyncio
async def test_empty_install_reports_setup_required(services: AppServices) -> None:
    """Enabling a fresh install by default would falsely advertise perception."""
    result = await sensory_status(services)

    assert result["ready"] is False
    assert result["capabilities"]["image"]["enabled"] is False
    assert result["capabilities"]["video_audio"]["verified"] is False

    guide = await sensory_setup_guide(services)

    assert guide["command"] == "cove-sensory-mcp configure"
    assert "API Key" in guide["security_notice"]
    assert guide["provider_options"] == [
        {
            "id": "gemini",
            "name": "Gemini",
            "default_capabilities": [
                "image and video visual understanding",
                "video audio, audio, and music understanding",
            ],
        },
        {
            "id": "minimax-m3",
            "name": "MiniMax-M3",
            "default_capabilities": [
                "image and native-video visual understanding",
                "not a default ear; video audio, audio, and music require a verified hearing provider",
            ],
        },
        {
            "id": "openai-compatible",
            "name": "OpenAI-compatible",
            "default_capabilities": [
                "capabilities depend on the endpoint and model",
                "verify each selected modality before use",
            ],
        },
    ]


@pytest.mark.asyncio
async def test_foundation_self_test_requires_setup_without_network(
    services: AppServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing the foundation verifier with a provider call must remain visible."""

    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "the foundation self-test must not open a network connection"
        )

    monkeypatch.setattr(socket, "create_connection", fail_network)

    result = await sensory_self_test(services, [Modality.IMAGE, Modality.AUDIO])

    assert result == {
        "status": "error",
        "error": {
            "code": "SETUP_REQUIRED",
            "message": "No verified provider is configured.",
            "retryable": False,
            "setup_command": "cove-sensory-mcp configure",
        },
    }


@pytest.mark.asyncio
async def test_setup_guide_returns_fresh_json_data_for_each_request(
    services: AppServices,
) -> None:
    """Sharing mutable guide data would let one caller alter another caller's response."""
    first = await sensory_setup_guide(services)
    first["provider_options"][0]["default_capabilities"].append("incorrect mutation")

    second = await sensory_setup_guide(services)

    assert second["provider_options"][0]["default_capabilities"] == [
        "image and video visual understanding",
        "video audio, audio, and music understanding",
    ]


class StubVerifier:
    """A deterministic verifier double at the provider boundary."""

    def __init__(self) -> None:
        self.modalities: list[Modality] | None = None
        self.calls = 0

    async def verify(self, modalities: list[Modality]) -> dict[str, object]:
        self.calls += 1
        self.modalities = modalities
        return {"status": "ok", "verified": [modality.value for modality in modalities]}


@pytest.mark.asyncio
async def test_self_test_returns_the_injected_verifiers_public_result(
    services: AppServices,
) -> None:
    """Bypassing a future verifier would make provider self-tests impossible to add."""
    verifier = StubVerifier()

    result = await sensory_self_test(
        services, [Modality.VIDEO_VISUAL], verifier=verifier
    )

    assert result == {"status": "ok", "verified": ["video_visual"]}
    assert verifier.modalities == [Modality.VIDEO_VISUAL]


@pytest.mark.parametrize(
    "modalities",
    [
        [],
        [Modality.IMAGE, Modality.IMAGE],
        ["image"],
        [{}],
        None,
    ],
)
@pytest.mark.asyncio
async def test_self_test_rejects_empty_duplicate_and_non_enum_modalities_before_verifier(
    services: AppServices,
    modalities: object,
) -> None:
    verifier = StubVerifier()

    result = await sensory_self_test(services, modalities, verifier=verifier)  # type: ignore[arg-type]

    assert result == {
        "status": "error",
        "error": {
            "code": "CONFIG_INVALID",
            "message": "The self-test modalities are invalid.",
            "retryable": False,
        },
    }
    assert verifier.calls == 0
    assert services.config_store.load() == AppConfig()


class ReusedResultVerifier:
    """A verifier that intentionally returns one object for every call."""

    def __init__(self) -> None:
        self.result: dict[str, object] = {"status": "ok", "verified": ["image"]}

    async def verify(self, modalities: list[Modality]) -> dict[str, object]:
        del modalities
        return self.result


@pytest.mark.asyncio
async def test_self_test_returns_an_isolated_copy_of_a_reused_verifier_result(
    services: AppServices,
) -> None:
    """Returning a verifier's dict directly would let callers corrupt later responses."""
    verifier = ReusedResultVerifier()
    first = await sensory_self_test(services, [Modality.IMAGE], verifier=verifier)
    first["verified"].append("incorrect mutation")

    second = await sensory_self_test(services, [Modality.IMAGE], verifier=verifier)

    assert second == {"status": "ok", "verified": ["image"]}
    assert verifier.result == {"status": "ok", "verified": ["image"]}


class InvalidResultVerifier:
    """A verifier returning a non-JSON value must not escape through the public tool."""

    async def verify(self, modalities: list[Modality]) -> dict[str, object]:
        del modalities
        return {"status": "ok", "unsafe": object()}


@pytest.mark.asyncio
async def test_self_test_replaces_non_json_verifier_output_with_a_public_error(
    services: AppServices,
) -> None:
    """Serializing arbitrary verifier objects could reveal implementation details."""
    result = await sensory_self_test(
        services, [Modality.IMAGE], verifier=InvalidResultVerifier()
    )

    assert result == {
        "status": "error",
        "error": {
            "code": "CONFIG_INVALID",
            "message": "The self-test result is invalid.",
            "retryable": False,
        },
    }


@pytest.mark.asyncio
async def test_status_advertises_only_routes_whose_provider_modality_is_verified(
    services: AppServices,
) -> None:
    """A stale or merely declared route must never advertise working perception."""
    services.config_store.save(
        AppConfig(
            providers={
                "vision": ProviderConfig(
                    adapter="gemini",
                    model="test-model",
                    credential_ref="test-ref",
                    declared_capabilities={
                        Modality.IMAGE: True,
                        Modality.VIDEO_VISUAL: True,
                    },
                    verified_capabilities={Modality.IMAGE: True},
                )
            },
            routes={
                "image": RouteConfig(primary="vision"),
                "video_visual": RouteConfig(primary="vision"),
            },
        )
    )

    result = await sensory_status(services)

    assert result["ready"] is True
    assert result["capabilities"]["image"] == {
        "modality": "image",
        "enabled": True,
        "verified": True,
        "provider": "vision",
        "mode": None,
        "reason": None,
    }
    assert result["capabilities"]["video_visual"]["enabled"] is False
    assert result["capabilities"]["video_visual"]["verified"] is False
    assert result["capabilities"]["video_visual"]["provider"] is None


class PublicSemanticProvider:
    """Return asset-specific image facts through the real verifier boundary."""

    async def sense(self, request: ProviderRequest) -> ProviderCallResult:
        modality = next(iter(request.requested_modalities))
        observation = ObservationEnvelope(
            modality=modality,
            summary="A blue triangle appears on a white field.",
            segments=[],
            transcript=[],
            warnings=[],
            confidence="high",
        )
        return ProviderCallResult(
            observations={modality: observation},
            provider_id="vision",
            model="test-model",
            remote_file_deleted=None,
        )


@pytest.mark.asyncio
async def test_self_test_invokes_capability_verifier_and_returns_stable_public_results(
    services: AppServices,
    tmp_path,
) -> None:
    """Leaving the foundation placeholder wired would never verify configured providers."""
    image = tmp_path / "self-test-image.png"
    image.write_bytes(b"tiny-image")
    services.config_store.save(
        AppConfig(
            providers={
                "vision": ProviderConfig(
                    adapter="gemini",
                    model="test-model",
                    credential_ref="test-ref",
                    declared_capabilities={Modality.IMAGE: True},
                )
            }
        )
    )

    result = await sensory_self_test(
        services,
        [Modality.IMAGE],
        registry=ProviderRegistry({"vision": PublicSemanticProvider()}),
        assets=SelfTestAssetStore(
            {
                Modality.IMAGE: PreparedMedia(
                    image,
                    "image/png",
                    MediaKind.IMAGE,
                    None,
                )
            },
            trusted_root=tmp_path,
        ),
    )

    assert result == {
        "status": "ok",
        "results": [
            {
                "provider": "vision",
                "modality": "image",
                "verified": True,
                "reason": None,
            }
        ],
    }
    assert services.config_store.load().providers["vision"].verified_capabilities == {
        Modality.IMAGE: True
    }
