"""Public setup handlers remain safe before provider adapters exist."""

from __future__ import annotations

import socket

import pytest

from cove_sensory_mcp.config.secrets import MemorySecretStore
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.services import AppServices
from cove_sensory_mcp.tools.setup import (
    sensory_self_test,
    sensory_setup_guide,
    sensory_status,
)


@pytest.fixture
def services(tmp_path) -> AppServices:
    """Provide isolated, non-persistent dependencies for setup handler tests."""
    return AppServices(
        config_store=ConfigStore(tmp_path / "config.yaml"),
        secret_store=MemorySecretStore(),
    )


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
        raise AssertionError("the foundation self-test must not open a network connection")

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
async def test_setup_guide_returns_fresh_json_data_for_each_request(services: AppServices) -> None:
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

    async def verify(self, modalities: list[Modality]) -> dict[str, object]:
        self.modalities = modalities
        return {"status": "ok", "verified": [modality.value for modality in modalities]}


@pytest.mark.asyncio
async def test_self_test_returns_the_injected_verifiers_public_result(
    services: AppServices,
) -> None:
    """Bypassing a future verifier would make provider self-tests impossible to add."""
    verifier = StubVerifier()

    result = await sensory_self_test(services, [Modality.VIDEO_VISUAL], verifier=verifier)

    assert result == {"status": "ok", "verified": ["video_visual"]}
    assert verifier.modalities == [Modality.VIDEO_VISUAL]
