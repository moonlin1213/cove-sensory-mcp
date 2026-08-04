"""Injected registry of already-constructed sensory Provider adapters."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import TypeAdapter, ValidationError

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import ProviderId

from .base import SensoryProvider

_PROVIDER_ID_ADAPTER = TypeAdapter(ProviderId)
_CONFIG_ERROR_MESSAGE = "The provider configuration is invalid."


def _config_error() -> SensoryError:
    return SensoryError(ErrorCode.CONFIG_INVALID, _CONFIG_ERROR_MESSAGE)


class ProviderRegistry:
    """Resolve adapters supplied by the composition root, without global state."""

    def __init__(self, providers: Mapping[ProviderId, SensoryProvider]) -> None:
        validated: dict[ProviderId, SensoryProvider] = {}
        for raw_provider_id, provider in providers.items():
            try:
                provider_id = _PROVIDER_ID_ADAPTER.validate_python(raw_provider_id)
            except ValidationError:
                raise _config_error() from None
            validated[provider_id] = provider
        self._providers = validated

    def get(self, provider_id: str) -> SensoryProvider:
        """Return one injected adapter or a privacy-safe configuration error."""
        try:
            validated_id = _PROVIDER_ID_ADAPTER.validate_python(provider_id)
            return self._providers[validated_id]
        except (KeyError, ValidationError):
            raise _config_error() from None
