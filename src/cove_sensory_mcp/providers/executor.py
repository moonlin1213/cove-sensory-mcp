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
    requested_model: str
    used_provider: ProviderId | None
    used_model: str | None
    fallback_used: bool
    failures: tuple[ErrorCode, ...]


@dataclass(frozen=True, slots=True)
class _ExecutionCandidate:
    """Executor-owned immutable copy of one authorized routing decision."""

    provider_id: ProviderId
    expected_model: str
    modalities: frozenset[Modality]
    is_fallback: bool


def _config_error() -> SensoryError:
    return SensoryError(ErrorCode.CONFIG_INVALID, _CONFIG_ERROR_MESSAGE)


def _joint_not_verified_error() -> SensoryError:
    return SensoryError(
        ErrorCode.CAPABILITY_NOT_VERIFIED,
        _JOINT_NOT_VERIFIED_MESSAGE,
    )


def _failure_result(
    requested_provider: ProviderId,
    requested_model: str,
    failures: list[ErrorCode],
) -> ExecutedObservation:
    return ExecutedObservation(
        observations={},
        requested_provider=requested_provider,
        requested_model=requested_model,
        used_provider=None,
        used_model=None,
        fallback_used=False,
        failures=tuple(failures),
    )


def _result_is_exact(
    result: object,
    candidate: _ExecutionCandidate,
    requested_modalities: frozenset[Modality],
) -> dict[Modality, ObservationEnvelope] | None:
    """Validate and copy one exact adapter result inside a sanitized boundary."""
    if type(result) is not ProviderCallResult:
        return None
    if (
        type(result.provider_id) is not str
        or result.provider_id != candidate.provider_id
        or type(result.model) is not str
        or result.model != candidate.expected_model
        or type(result.observations) is not dict
    ):
        return None
    if frozenset(result.observations) != requested_modalities:
        return None
    copied: dict[Modality, ObservationEnvelope] = {}
    for modality, observation in result.observations.items():
        if (
            type(modality) is not Modality
            or type(observation) is not ObservationEnvelope
            or observation.modality is not modality
        ):
            return None
        copied[modality] = observation
    return copied


def _snapshot_candidate(candidate: object) -> _ExecutionCandidate:
    """Copy only the exact immutable route shape emitted by ProviderRouter."""
    if type(candidate) is not ProviderCandidate:
        raise _config_error()
    if (
        type(candidate.provider_id) is not str
        or not candidate.provider_id
        or len(candidate.provider_id) > 64
        or type(candidate.expected_model) is not str
        or not candidate.expected_model
        or len(candidate.expected_model) > 256
        or type(candidate.modalities) is not frozenset
        or not candidate.modalities
        or any(type(modality) is not Modality for modality in candidate.modalities)
        or type(candidate.is_fallback) is not bool
    ):
        raise _config_error()
    return _ExecutionCandidate(
        provider_id=candidate.provider_id,
        expected_model=candidate.expected_model,
        modalities=frozenset(tuple(candidate.modalities)),
        is_fallback=candidate.is_fallback,
    )


def _validate_candidates(
    candidates: tuple[_ExecutionCandidate, ...],
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
    ) -> tuple[_ExecutionCandidate, ...]:
        if not requested_modalities:
            raise _config_error()
        try:
            if len(requested_modalities) == 1:
                modality = next(iter(requested_modalities))
                routed = self._router.candidates(modality)
                if type(routed) is not list:
                    raise _config_error()
                candidates = tuple(
                    _snapshot_candidate(candidate) for candidate in tuple(routed)
                )
            else:
                joint = self._router.joint_candidate(requested_modalities)
                if joint is None:
                    raise _joint_not_verified_error()
                candidates = (_snapshot_candidate(joint),)
        except SensoryError:
            raise
        except Exception:  # noqa: BLE001 - sanitize a malformed router boundary
            raise _config_error() from None
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
        requested_model = candidates[0].expected_model
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
                    return _failure_result(
                        requested_provider,
                        requested_model,
                        failures,
                    )
                if index + 1 >= len(candidates):
                    if len(candidates) == 1:
                        failures.append(ErrorCode.FALLBACK_NOT_AUTHORIZED)
                    return _failure_result(
                        requested_provider,
                        requested_model,
                        failures,
                    )
                continue
            except Exception:  # noqa: BLE001 - sanitize an invalid adapter boundary
                failures.append(ErrorCode.PROVIDER_UNAVAILABLE)
                return _failure_result(
                    requested_provider,
                    requested_model,
                    failures,
                )

            try:
                observations = _result_is_exact(
                    result,
                    candidate,
                    requested_modalities,
                )
            except Exception:  # noqa: BLE001 - sanitize malformed result properties
                observations = None
            if observations is None:
                failures.append(ErrorCode.PROVIDER_CAPABILITY_REJECTED)
                return _failure_result(
                    requested_provider,
                    requested_model,
                    failures,
                )

            return ExecutedObservation(
                observations=observations,
                requested_provider=requested_provider,
                requested_model=requested_model,
                used_provider=candidate.provider_id,
                used_model=candidate.expected_model,
                fallback_used=candidate.is_fallback,
                failures=tuple(failures),
            )

        return _failure_result(requested_provider, requested_model, failures)
