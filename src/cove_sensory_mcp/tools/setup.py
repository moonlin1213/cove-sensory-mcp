"""Privacy-safe setup, status, and capability-verification handlers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from typing import Protocol

from cove_sensory_mcp import __version__
from cove_sensory_mcp.config.schema import AppConfig, ProviderConfig
from cove_sensory_mcp.errors import ErrorCode, SensoryError, error_result
from cove_sensory_mcp.models import CapabilityStatus, Modality, SensoryStatus
from cove_sensory_mcp.providers.base import SensoryProvider, VerificationResult
from cove_sensory_mcp.providers.gemini import GeminiProvider
from cove_sensory_mcp.providers.minimax_m3 import (
    MINIMAX_CN_BASE_URL,
    MiniMaxM3Provider,
    MiniMaxRegion,
)
from cove_sensory_mcp.providers.openai_compatible import OpenAICompatibleProvider
from cove_sensory_mcp.providers.registry import ProviderRegistry
from cove_sensory_mcp.services import AppServices
from cove_sensory_mcp.verification.assets import SelfTestAssetStore

_SETUP_COMMAND = "cove-sensory-mcp configure"
_NOT_VERIFIED_REASON = "No verified provider is configured."
_INVALID_MODALITIES_MESSAGE = "The self-test modalities are invalid."
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


class SelfTestVerifier(Protocol):
    """Produce a public self-test result for requested modalities."""

    async def verify(self, modalities: list[Modality]) -> dict[str, object]:
        """Return a JSON-compatible verification result without private data."""


def _public_result(result: dict[str, object]) -> dict[str, object]:
    """Validate and copy a verifier result before exposing it to an MCP client."""
    try:
        copied = json.loads(json.dumps(result, allow_nan=False))
    except (TypeError, ValueError):
        return error_result(
            SensoryError(ErrorCode.CONFIG_INVALID, "The self-test result is invalid.")
        )
    if not isinstance(copied, dict):
        return error_result(
            SensoryError(ErrorCode.CONFIG_INVALID, "The self-test result is invalid.")
        )
    return copied


def _status_from_config(config: AppConfig) -> SensoryStatus:
    """Advertise only routes backed by the route Provider's verified capability."""
    capabilities: dict[Modality, CapabilityStatus] = {}
    for modality in Modality:
        route = getattr(config.routes, modality.value)
        provider_id: str | None = None
        mode: str | None = None
        if route is not None:
            provider = config.providers.get(route.primary)
            if provider is not None and provider.verified_capabilities.get(
                modality, False
            ):
                provider_id = route.primary
                if (
                    provider.adapter == "minimax-m3"
                    and modality is Modality.VIDEO_VISUAL
                ):
                    mode = "native_video"
        verified = provider_id is not None
        capabilities[modality] = CapabilityStatus(
            modality=modality,
            enabled=verified,
            verified=verified,
            provider=provider_id,
            mode=mode,
            reason=None if verified else _NOT_VERIFIED_REASON,
        )
    return SensoryStatus(
        ready=any(capability.verified for capability in capabilities.values()),
        version=__version__,
        capabilities=capabilities,
    )


async def sensory_status(services: AppServices) -> dict[str, object]:
    """Return local configuration status without resolving credentials or using a provider."""
    try:
        config = services.config_store.load()
    except SensoryError as error:
        return error_result(error)
    return _status_from_config(config).model_dump(mode="json")


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
    registry: ProviderRegistry | None = None,
    assets: SelfTestAssetStore | None = None,
) -> dict[str, object]:
    """Delegate a requested self-test to an injected verifier implementation."""
    if (
        type(modalities) is not list
        or not modalities
        or len(modalities) > len(Modality)
        or any(type(modality) is not Modality for modality in modalities)
        or len(set(modalities)) != len(modalities)
    ):
        return error_result(
            SensoryError(ErrorCode.CONFIG_INVALID, _INVALID_MODALITIES_MESSAGE)
        )
    if verifier is not None:
        return _public_result(await verifier.verify(modalities))
    try:
        return _public_result(
            await _run_configured_self_test(
                services,
                modalities,
                registry=registry,
                assets=assets,
            )
        )
    except SensoryError as error:
        return error_result(error)
    except Exception:  # noqa: BLE001 - public self-test boundary discards internals
        return error_result(
            SensoryError(
                ErrorCode.CONFIG_INVALID, "The self-test configuration is invalid."
            )
        )


def _selected_providers(
    config: AppConfig,
    modalities: Sequence[Modality],
) -> dict[str, list[Modality]]:
    if not config.providers:
        raise SensoryError(
            ErrorCode.SETUP_REQUIRED,
            _NOT_VERIFIED_REASON,
            setup_command=_SETUP_COMMAND,
        )
    selected: dict[str, list[Modality]] = {}
    for modality in modalities:
        route = getattr(config.routes, modality.value)
        if route is not None:
            provider_id = route.primary
        else:
            declared = sorted(
                provider_id
                for provider_id, provider in config.providers.items()
                if provider.declared_capabilities.get(modality, False)
            )
            if len(declared) != 1:
                raise SensoryError(
                    ErrorCode.CAPABILITY_NOT_CONFIGURED,
                    "The requested self-test capability is not configured.",
                )
            provider_id = declared[0]
        provider = config.providers.get(provider_id)
        if provider is None or not provider.declared_capabilities.get(modality, False):
            raise SensoryError(
                ErrorCode.CAPABILITY_NOT_CONFIGURED,
                "The requested self-test capability is not configured.",
            )
        selected.setdefault(provider_id, []).append(modality)
    return selected


def _provider_adapter(
    provider_id: str,
    config: ProviderConfig,
    services: AppServices,
) -> SensoryProvider:
    if config.adapter == "gemini":
        return GeminiProvider(
            provider_id=provider_id,
            config=config,
            secret_store=services.secret_store,
        )
    if config.adapter == "minimax-m3":
        region = (
            MiniMaxRegion.CN
            if config.base_url == MINIMAX_CN_BASE_URL
            else MiniMaxRegion.GLOBAL
        )
        return MiniMaxM3Provider(
            provider_id=provider_id,
            config=config,
            secret_store=services.secret_store,
            region=region,
        )
    if config.adapter == "openai-compatible":
        return OpenAICompatibleProvider(
            provider_id=provider_id,
            config=config,
            secret_store=services.secret_store,
        )
    raise SensoryError(ErrorCode.CONFIG_INVALID, "The provider adapter is invalid.")


async def _close_adapters(adapters: Sequence[SensoryProvider]) -> None:
    for adapter in reversed(adapters):
        close = getattr(adapter, "aclose", None)
        if close is not None:
            await close()


def _public_verification_results(
    results: Sequence[VerificationResult],
) -> dict[str, object]:
    passed = sum(result.verified for result in results)
    status = "ok" if passed == len(results) else "partial" if passed else "error"
    return {
        "status": status,
        "results": [
            {
                "provider": result.provider_id,
                "modality": result.modality.value,
                "verified": result.verified,
                "reason": result.reason,
            }
            for result in results
        ],
    }


async def _run_configured_self_test(
    services: AppServices,
    modalities: list[Modality],
    *,
    registry: ProviderRegistry | None,
    assets: SelfTestAssetStore | None,
) -> dict[str, object]:
    config = services.config_store.load()
    selected = _selected_providers(config, modalities)
    owned_adapters: list[SensoryProvider] = []
    active_registry = registry
    if active_registry is None:
        provider_map: dict[str, SensoryProvider] = {}
        for provider_id in selected:
            adapter = _provider_adapter(
                provider_id, config.providers[provider_id], services
            )
            provider_map[provider_id] = adapter
            owned_adapters.append(adapter)
        active_registry = ProviderRegistry(provider_map)
    active_assets = assets or SelfTestAssetStore.packaged()
    results: list[VerificationResult] = []
    try:
        for provider_id, provider_modalities in selected.items():
            capability_verifier = services.capability_verifier(
                active_registry,
                assets=active_assets,
            )
            results.extend(
                await capability_verifier.verify(provider_id, provider_modalities)
            )
    finally:
        if owned_adapters:
            await _close_adapters(owned_adapters)
    return _public_verification_results(results)


async def verify_provider_capabilities(
    services: AppServices,
    provider_id: str,
    modalities: list[Modality],
    *,
    registry: ProviderRegistry | None = None,
    assets: SelfTestAssetStore | None = None,
) -> dict[str, object]:
    """Verify one explicitly selected Provider for the local configure wizard."""
    config = services.config_store.load()
    try:
        provider_config = config.providers[provider_id]
    except KeyError:
        return error_result(
            SensoryError(
                ErrorCode.CONFIG_INVALID, "The provider configuration is invalid."
            )
        )
    owned: list[SensoryProvider] = []
    active_registry = registry
    try:
        if active_registry is None:
            adapter = _provider_adapter(provider_id, provider_config, services)
            owned.append(adapter)
            active_registry = ProviderRegistry({provider_id: adapter})
        verifier = services.capability_verifier(active_registry, assets=assets)
        results = await verifier.verify(provider_id, modalities)
        return _public_verification_results(results)
    except SensoryError as error:
        return error_result(error)
    except Exception:  # noqa: BLE001 - wizard result must never expose internals
        return error_result(
            SensoryError(
                ErrorCode.CONFIG_INVALID, "The self-test configuration is invalid."
            )
        )
    finally:
        if owned:
            await _close_adapters(owned)
