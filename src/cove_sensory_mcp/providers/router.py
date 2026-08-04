"""Deterministic routing over configured and verified Provider capabilities."""

from __future__ import annotations

from cove_sensory_mcp.config.schema import AppConfig, ProviderConfig
from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import Modality, ProviderId, RouteConfig

from .base import ProviderCandidate

_CONFIG_ERROR_MESSAGE = "The provider configuration is invalid."
_NOT_CONFIGURED_MESSAGE = "The requested sensory capability is not configured."
_NOT_VERIFIED_MESSAGE = "The configured provider capability is not verified."


def _config_error() -> SensoryError:
    return SensoryError(ErrorCode.CONFIG_INVALID, _CONFIG_ERROR_MESSAGE)


def _not_verified_error() -> SensoryError:
    return SensoryError(ErrorCode.CAPABILITY_NOT_VERIFIED, _NOT_VERIFIED_MESSAGE)


class ProviderRouter:
    """Select only explicit, verified primary and authorized fallback routes."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def _route(self, modality: Modality) -> RouteConfig | None:
        return getattr(self._config.routes, modality.value)

    def _configured_provider(self, provider_id: ProviderId) -> ProviderConfig:
        provider = self._config.providers.get(provider_id)
        if provider is None:
            raise _config_error()
        return provider

    def _verified_provider(
        self,
        provider_id: ProviderId,
        modality: Modality,
    ) -> ProviderConfig:
        provider = self._configured_provider(provider_id)
        if not provider.verified_capabilities.get(modality, False):
            raise _not_verified_error()
        return provider

    def candidates(self, modality: Modality) -> list[ProviderCandidate]:
        """Return the verified primary followed by authorized verified fallbacks."""
        route = self._route(modality)
        if route is None:
            raise SensoryError(
                ErrorCode.CAPABILITY_NOT_CONFIGURED,
                _NOT_CONFIGURED_MESSAGE,
            )

        self._verified_provider(route.primary, modality)
        candidates = [
            ProviderCandidate(
                provider_id=route.primary,
                modalities=frozenset({modality}),
            )
        ]
        seen = {route.primary}
        for fallback in route.fallbacks:
            self._configured_provider(fallback.provider)
            if fallback.provider in seen:
                raise _config_error()
            seen.add(fallback.provider)
            if not fallback.authorized:
                continue
            self._verified_provider(fallback.provider, modality)
            candidates.append(
                ProviderCandidate(
                    provider_id=fallback.provider,
                    modalities=frozenset({modality}),
                    is_fallback=True,
                )
            )
        return candidates

    def joint_candidate(
        self,
        modalities: frozenset[Modality],
    ) -> ProviderCandidate | None:
        """Return one shared primary only for an exact verified joint capability."""
        if len(modalities) < 2:
            return None

        routes: list[tuple[Modality, RouteConfig]] = []
        for modality in sorted(modalities, key=lambda item: item.value):
            route = self._route(modality)
            if route is None:
                return None
            routes.append((modality, route))

        primary = routes[0][1].primary
        if any(route.primary != primary for _, route in routes[1:]):
            return None
        provider = self._verified_provider(primary, routes[0][0])
        for modality, _ in routes[1:]:
            self._verified_provider(primary, modality)
        if modalities not in provider.verified_joint_capabilities:
            return None
        return ProviderCandidate(provider_id=primary, modalities=modalities)
