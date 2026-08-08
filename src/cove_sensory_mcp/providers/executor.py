"""Bounded execution over verified routes and explicitly authorized fallbacks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import Modality, ProviderId
from cove_sensory_mcp.reports.schemas import ObservationEnvelope

from .base import ProviderCallResult, ProviderCandidate, ProviderRequest
from .registry import ProviderRegistry
from .router import ProviderRouter

_CONFIG_ERROR_MESSAGE = "The provider execution configuration is invalid."
_JOINT_NOT_VERIFIED_MESSAGE = "The requested joint capability is not verified."

_FALLBACK_ERROR_CODES = frozenset(
    {
        ErrorCode.PROVIDER_TIMEOUT,
        ErrorCode.PROVIDER_UNAVAILABLE,
        ErrorCode.PROVIDER_SAFETY_REJECTED,
    }
)


@dataclass(frozen=True, slots=True)
class ExecutedObservation:
    """One successful observation or a stable, raw-free failure history."""

    observations: dict[Modality, ObservationEnvelope]
    requested_provider: ProviderId
    used_provider: ProviderId | None
    fallback_used: bool
    failures: tuple[ErrorCode, ...]


def _config_error() -> SensoryError:
    return SensoryError(ErrorCode.CONFIG_INVALID, _CONFIG_ERROR_MESSAGE)


def _joint_not_verified_error() -> SensoryError:
    return SensoryError(
        ErrorCode.CAPABILITY_NOT_VERIFIED,
        _JOINT_NOT_VERIFIED_MESSAGE,
    )


def _failure_result(
    requested_provider: ProviderId,
    failures: list[ErrorCode],
) -> ExecutedObservation:
    return ExecutedObservation(
        observations={},
        requested_provider=requested_provider,
        used_provider=None,
        fallback_used=False,
        failures=tuple(failures),
    )


def _result_is_exact(
    result: ProviderCallResult,
    candidate: ProviderCandidate,
    requested_modalities: frozenset[Modality],
) -> bool:
    """Validate identity and the full modality map at the adapter boundary."""
    if result.provider_id != candidate.provider_id:
        return False
    if not isinstance(result.observations, dict):
        return False
    if frozenset(result.observations) != requested_modalities:
        return False
    return all(
        isinstance(observation, ObservationEnvelope)
        and observation.modality is modality
        for modality, observation in result.observations.items()
    )


def _validate_candidates(
    candidates: list[ProviderCandidate],
    requested_modalities: frozenset[Modality],
) -> None:
    if not candidates:
        raise _config_error()
    seen: set[ProviderId] = set()
    for index, candidate in enumerate(candidates):
        if (
            candidate.modalities != requested_modalities
            or candidate.provider_id in seen
            or candidate.is_fallback is (index == 0)
        ):
            raise _config_error()
        seen.add(candidate.provider_id)


def _can_fallback(error: SensoryError) -> bool:
    if error.code is ErrorCode.PROVIDER_SAFETY_REJECTED:
        return True
    return error.retryable and error.code in _FALLBACK_ERROR_CODES


class ProviderExecutor:
    """Invoke each verified route candidate at most once and never invent a route."""

    def __init__(
        self,
        *,
        router: ProviderRouter,
        registry: ProviderRegistry,
    ) -> None:
        self._router = router
        self._registry = registry

    def _candidates(
        self,
        requested_modalities: frozenset[Modality],
    ) -> list[ProviderCandidate]:
        if not requested_modalities:
            raise _config_error()
        if len(requested_modalities) == 1:
            modality = next(iter(requested_modalities))
            candidates = self._router.candidates(modality)
        else:
            joint = self._router.joint_candidate(requested_modalities)
            if joint is None:
                raise _joint_not_verified_error()
            candidates = [joint]
        _validate_candidates(candidates, requested_modalities)
        return candidates

    async def sense(
        self,
        requested_modalities: frozenset[Modality],
        request: ProviderRequest,
    ) -> ExecutedObservation:
        """Execute one route, allowing only bounded and authorized fallback calls."""
        if request.requested_modalities != requested_modalities:
            raise _config_error()

        candidates = self._candidates(requested_modalities)
        requested_provider = candidates[0].provider_id
        failures: list[ErrorCode] = []

        for index, candidate in enumerate(candidates):
            provider = self._registry.get(candidate.provider_id)
            try:
                result = await provider.sense(request)
            except asyncio.CancelledError:
                raise
            except SensoryError as error:
                failures.append(error.code)
                if not _can_fallback(error):
                    return _failure_result(requested_provider, failures)
                if index + 1 >= len(candidates):
                    if len(candidates) == 1:
                        failures.append(ErrorCode.FALLBACK_NOT_AUTHORIZED)
                    return _failure_result(requested_provider, failures)
                continue
            except Exception:  # noqa: BLE001 - sanitize an invalid adapter boundary
                failures.append(ErrorCode.PROVIDER_UNAVAILABLE)
                return _failure_result(requested_provider, failures)

            if not _result_is_exact(result, candidate, requested_modalities):
                failures.append(ErrorCode.PROVIDER_CAPABILITY_REJECTED)
                return _failure_result(requested_provider, failures)

            return ExecutedObservation(
                observations=dict(result.observations),
                requested_provider=requested_provider,
                used_provider=candidate.provider_id,
                fallback_used=candidate.is_fallback,
                failures=tuple(failures),
            )

        return _failure_result(requested_provider, failures)
