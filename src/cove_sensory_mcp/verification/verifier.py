"""Capability verification over exact Provider execution boundaries."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from cove_sensory_mcp.config.schema import AppConfig, ProviderConfig
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import DetailLevel, Modality, RouteConfig
from cove_sensory_mcp.providers.base import ProviderRequest, VerificationResult
from cove_sensory_mcp.providers.executor import ProviderExecutor
from cove_sensory_mcp.providers.registry import ProviderRegistry
from cove_sensory_mcp.providers.router import ProviderRouter

from .assets import SelfTestAssetStore

_CONFIG_MESSAGE = "The provider verification configuration is invalid."
_ABORT_MESSAGE = "The capability self-test could not run safely."
_FOCUS = "Direct observations from the capability test media."

_BATCH_ABORT_CODES = frozenset(
    {
        ErrorCode.SETUP_REQUIRED,
        ErrorCode.CONFIG_INVALID,
        ErrorCode.CAPABILITY_NOT_CONFIGURED,
        ErrorCode.CAPABILITY_NOT_VERIFIED,
        ErrorCode.PATH_NOT_ALLOWED,
        ErrorCode.SOURCE_NOT_FOUND,
        ErrorCode.UNSUPPORTED_MEDIA_TYPE,
        ErrorCode.MEDIA_TOO_LARGE,
        ErrorCode.LONG_MEDIA_CONFIRMATION_REQUIRED,
        ErrorCode.MEDIA_RUNTIME_REQUIRED,
        ErrorCode.DOWNLOAD_BLOCKED,
        ErrorCode.DOWNLOAD_FAILED,
        ErrorCode.TEMP_CLEANUP_FAILED,
    }
)


def _config_error() -> SensoryError:
    return SensoryError(ErrorCode.CONFIG_INVALID, _CONFIG_MESSAGE)


def _provider_identity(config: ProviderConfig) -> dict[str, object]:
    """Return fields that must not change while remote verification is running."""
    return config.model_dump(
        mode="python",
        exclude={
            "verified_capabilities",
            "verified_joint_capabilities",
            "last_verified_at",
        },
    )


def _verification_overlay(
    config: AppConfig,
    provider_id: str,
    modality: Modality,
) -> AppConfig:
    """Authorize exactly one in-memory route without persisting a pre-verification claim."""
    overlay = config.model_copy(deep=True)
    provider = overlay.providers[provider_id]
    provider.verified_capabilities[modality] = True
    setattr(overlay.routes, modality.value, RouteConfig(primary=provider_id))
    return AppConfig.model_validate(overlay.model_dump(mode="python"))


class CapabilityVerifier:
    """Verify declared modalities separately, then commit one complete result batch."""

    def __init__(
        self,
        *,
        config_store: ConfigStore,
        registry: ProviderRegistry,
        assets: SelfTestAssetStore,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._config_store = config_store
        self._registry = registry
        self._assets = assets
        self._now = now or (lambda: datetime.now(UTC))

    async def verify(
        self,
        provider_id: str,
        modalities: Sequence[Modality],
    ) -> list[VerificationResult]:
        """Persist success when each Provider call returns a valid observation envelope."""
        requested = tuple(modalities)
        if (
            not requested
            or len(set(requested)) != len(requested)
            or any(type(modality) is not Modality for modality in requested)
        ):
            raise _config_error()

        starting = self._config_store.load()
        try:
            provider_config = starting.providers[provider_id]
        except (KeyError, TypeError):
            raise _config_error() from None
        declared = {
            modality
            for modality, enabled in provider_config.declared_capabilities.items()
            if enabled
        }
        if not set(requested) <= declared:
            raise _config_error()
        identity = _provider_identity(provider_config)

        results: list[VerificationResult] = []
        for modality in requested:
            media = self._assets.get(modality)
            overlay = _verification_overlay(starting, provider_id, modality)
            executor = ProviderExecutor(
                router=ProviderRouter(overlay),
                registry=self._registry,
            )
            requested_set = frozenset({modality})
            executed = await executor.sense(
                requested_set,
                ProviderRequest(
                    media=media,
                    requested_modalities=requested_set,
                    question=_FOCUS,
                    detail=DetailLevel.QUICK,
                    language="en",
                ),
            )
            abort_code = next(
                (code for code in executed.failures if code in _BATCH_ABORT_CODES),
                None,
            )
            if abort_code is not None:
                raise SensoryError(abort_code, _ABORT_MESSAGE)
            verified = modality in executed.observations
            if verified:
                reason = None
            elif executed.failures:
                reason = executed.failures[0].value
            else:
                reason = ErrorCode.PROVIDER_CAPABILITY_REJECTED.value
            results.append(
                VerificationResult(
                    provider_id=provider_id,
                    modality=modality,
                    verified=verified,
                    reason=reason,
                )
            )

        verified_at = self._now()
        if verified_at.tzinfo is None or verified_at.utcoffset() is None:
            raise _config_error()

        def merge_results(latest: AppConfig) -> None:
            try:
                latest_provider = latest.providers[provider_id]
            except KeyError:
                raise _config_error() from None
            if _provider_identity(latest_provider) != identity:
                raise _config_error()
            for result in results:
                if result.verified:
                    latest_provider.verified_capabilities[result.modality] = True
                else:
                    latest_provider.verified_capabilities.pop(result.modality, None)
            verified_modalities = {
                modality
                for modality, enabled in latest_provider.verified_capabilities.items()
                if enabled
            }
            latest_provider.verified_joint_capabilities = [
                joint
                for joint in latest_provider.verified_joint_capabilities
                if joint <= verified_modalities
            ]
            latest_provider.last_verified_at = verified_at.astimezone(UTC)

        self._config_store.update(merge_results)
        return results
