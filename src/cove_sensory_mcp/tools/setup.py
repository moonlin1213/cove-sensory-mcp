"""Read-only setup and verification handlers with safe public responses."""

from __future__ import annotations

from copy import deepcopy

from cove_sensory_mcp import __version__
from cove_sensory_mcp.errors import SensoryError, error_result
from cove_sensory_mcp.models import CapabilityStatus, Modality, SensoryStatus
from cove_sensory_mcp.services import AppServices, SelfTestVerifier

_SETUP_COMMAND = "cove-sensory-mcp configure"
_NOT_VERIFIED_REASON = "No verified provider is configured."
_PROVIDER_OPTIONS: list[dict[str, object]] = [
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


def _foundation_status() -> SensoryStatus:
    """Return the truthful capability state before provider verification exists."""
    return SensoryStatus(
        ready=False,
        version=__version__,
        capabilities={
            modality: CapabilityStatus(
                modality=modality,
                enabled=False,
                verified=False,
                reason=_NOT_VERIFIED_REASON,
            )
            for modality in Modality
        },
    )


async def sensory_status(services: AppServices) -> dict[str, object]:
    """Return local configuration status without resolving credentials or using a provider."""
    try:
        services.config_store.load()
    except SensoryError as error:
        return error_result(error)
    return _foundation_status().model_dump(mode="json")


async def sensory_setup_guide(services: AppServices) -> dict[str, object]:
    """Return local-only setup options while keeping API keys outside tool inputs."""
    del services
    return {
        "command": _SETUP_COMMAND,
        "provider_options": deepcopy(_PROVIDER_OPTIONS),
        "security_notice": (
            "Enter an API Key only in the local cove-sensory-mcp configure wizard. "
            "Never send API Keys in chat messages or tool arguments."
        ),
    }


async def sensory_self_test(
    services: AppServices,
    modalities: list[Modality],
    *,
    verifier: SelfTestVerifier | None = None,
) -> dict[str, object]:
    """Delegate a requested self-test to an injected verifier implementation."""
    return await (verifier or services.self_test_verifier).verify(modalities)
