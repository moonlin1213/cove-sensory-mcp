from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn, cast

import pytest

from cove_sensory_mcp.config.schema import AppConfig, ProviderConfig, RoutesConfig
from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import DetailLevel, Modality, ProviderRef, RouteConfig
from cove_sensory_mcp.providers.base import (
    MediaKind,
    PreparedMedia,
    ProviderCallResult,
    ProviderCandidate,
    ProviderRequest,
    SensoryProvider,
)
from cove_sensory_mcp.providers.executor import ProviderExecutor
from cove_sensory_mcp.providers.registry import ProviderRegistry
from cove_sensory_mcp.providers.router import ProviderRouter
from cove_sensory_mcp.reports.schemas import ObservationEnvelope


def _provider_config(
    *modalities: Modality,
    joint: tuple[frozenset[Modality], ...] = (),
    model: str = "test-model",
) -> ProviderConfig:
    capabilities = {modality: True for modality in modalities}
    return ProviderConfig(
        adapter="test-adapter",
        model=model,
        credential_ref="test-credential",
        declared_capabilities=capabilities,
        verified_capabilities=capabilities,
        verified_joint_capabilities=list(joint),
        last_verified_at=datetime(2026, 8, 4, tzinfo=UTC),
    )


def _request(modalities: frozenset[Modality]) -> ProviderRequest:
    return ProviderRequest(
        media=PreparedMedia(
            path=Path("prepared-media.mp4"),
            mime_type="video/mp4",
            media_kind=MediaKind.VIDEO,
            duration_seconds=8.0,
        ),
        requested_modalities=modalities,
        question="What is directly observable?",
        detail=DetailLevel.QUICK,
        language="en",
    )


def _observation(modality: Modality, summary: str) -> ObservationEnvelope:
    return ObservationEnvelope(
        modality=modality,
        summary=summary,
        segments=[],
        transcript=[],
        warnings=[],
        confidence="medium",
    )


def _success(
    provider_id: str,
    modalities: frozenset[Modality],
    *,
    model: str = "test-model",
) -> ProviderCallResult:
    return ProviderCallResult(
        observations={
            modality: _observation(modality, f"{modality.value} evidence")
            for modality in modalities
        },
        provider_id=provider_id,
        model=model,
        remote_file_deleted=None,
    )


class ScriptedProvider:
    def __init__(
        self,
        outcomes: Sequence[object],
    ) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[ProviderRequest] = []

    async def sense(self, request: ProviderRequest) -> ProviderCallResult:
        self.requests.append(request)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return cast(ProviderCallResult, outcome)


def _single_config(
    *,
    fallback_authorized: bool | None = None,
) -> AppConfig:
    providers = {"primary": _provider_config(Modality.VIDEO_VISUAL)}
    fallbacks: list[ProviderRef] = []
    if fallback_authorized is not None:
        providers["fallback"] = _provider_config(
            Modality.VIDEO_VISUAL,
            model="fallback-model",
        )
        fallbacks.append(
            ProviderRef(provider="fallback", authorized=fallback_authorized)
        )
    return AppConfig(
        providers=providers,
        routes=RoutesConfig(
            video_visual=RouteConfig(primary="primary", fallbacks=fallbacks)
        ),
    )


def _executor(
    config: AppConfig,
    **providers: ScriptedProvider,
) -> ProviderExecutor:
    return ProviderExecutor(
        router=ProviderRouter(config),
        registry=ProviderRegistry(
            {
                provider_id: cast(SensoryProvider, provider)
                for provider_id, provider in providers.items()
            }
        ),
    )


@pytest.mark.asyncio
async def test_primary_success_returns_exact_execution_metadata() -> None:
    """Using fallback metadata on primary success would misreport where media was sent."""
    modalities = frozenset({Modality.VIDEO_VISUAL})
    primary = ScriptedProvider([_success("primary", modalities)])
    fallback = ScriptedProvider(
        [_success("fallback", modalities, model="fallback-model")]
    )

    result = await _executor(
        _single_config(fallback_authorized=True),
        primary=primary,
        fallback=fallback,
    ).sense(modalities, _request(modalities))

    assert result.observations == _success("primary", modalities).observations
    assert result.requested_provider == "primary"
    assert result.requested_model == "test-model"
    assert result.used_provider == "primary"
    assert result.used_model == "test-model"
    assert result.fallback_used is False
    assert result.failures == ()
    assert len(primary.requests) == 1
    assert fallback.requests == []


@pytest.mark.asyncio
async def test_retryable_failure_uses_only_the_ordered_authorized_fallback() -> None:
    """Skipping route order could send media to an unchosen Provider."""
    modalities = frozenset({Modality.VIDEO_VISUAL})
    primary = ScriptedProvider(
        [
            SensoryError(
                ErrorCode.PROVIDER_TIMEOUT,
                "private primary timeout detail",
                retryable=True,
            )
        ]
    )
    fallback = ScriptedProvider(
        [_success("fallback", modalities, model="fallback-model")]
    )

    result = await _executor(
        _single_config(fallback_authorized=True),
        primary=primary,
        fallback=fallback,
    ).sense(modalities, _request(modalities))

    assert result.requested_provider == "primary"
    assert result.requested_model == "test-model"
    assert result.used_provider == "fallback"
    assert result.used_model == "fallback-model"
    assert result.fallback_used is True
    assert result.failures == (ErrorCode.PROVIDER_TIMEOUT,)
    assert len(primary.requests) == len(fallback.requests) == 1


@pytest.mark.asyncio
async def test_safety_rejection_uses_an_explicitly_authorized_fallback() -> None:
    """Treating safety as terminal despite authorization would break the documented route."""
    modalities = frozenset({Modality.VIDEO_VISUAL})
    primary = ScriptedProvider(
        [
            SensoryError(
                ErrorCode.PROVIDER_SAFETY_REJECTED,
                "private safety body",
            )
        ]
    )
    fallback = ScriptedProvider(
        [_success("fallback", modalities, model="fallback-model")]
    )

    result = await _executor(
        _single_config(fallback_authorized=True),
        primary=primary,
        fallback=fallback,
    ).sense(modalities, _request(modalities))

    assert result.used_provider == "fallback"
    assert result.used_model == "fallback-model"
    assert result.fallback_used is True
    assert result.failures == (ErrorCode.PROVIDER_SAFETY_REJECTED,)
    assert len(fallback.requests) == 1


@pytest.mark.asyncio
async def test_unauthorized_fallback_is_never_invoked_and_is_reported() -> None:
    """Ignoring an authorization decline could disclose the user's media cross-Provider."""
    modalities = frozenset({Modality.VIDEO_VISUAL})
    primary = ScriptedProvider(
        [
            SensoryError(
                ErrorCode.PROVIDER_UNAVAILABLE,
                "private upstream marker",
                retryable=True,
            )
        ]
    )
    fallback = ScriptedProvider(
        [_success("fallback", modalities, model="fallback-model")]
    )

    result = await _executor(
        _single_config(fallback_authorized=False),
        primary=primary,
        fallback=fallback,
    ).sense(modalities, _request(modalities))

    assert result.observations == {}
    assert result.requested_provider == "primary"
    assert result.requested_model == "test-model"
    assert result.used_provider is None
    assert result.used_model is None
    assert result.fallback_used is False
    assert result.failures == (
        ErrorCode.PROVIDER_UNAVAILABLE,
        ErrorCode.FALLBACK_NOT_AUTHORIZED,
    )
    assert fallback.requests == []


@pytest.mark.asyncio
async def test_all_authorized_providers_fail_with_codes_only() -> None:
    """Retaining exception messages would expose raw Provider diagnostics publicly."""
    modalities = frozenset({Modality.VIDEO_VISUAL})
    private_markers = ("primary-secret-message", "fallback-secret-message")
    primary = ScriptedProvider(
        [
            SensoryError(
                ErrorCode.PROVIDER_TIMEOUT,
                private_markers[0],
                retryable=True,
            )
        ]
    )
    fallback = ScriptedProvider(
        [
            SensoryError(
                ErrorCode.PROVIDER_UNAVAILABLE,
                private_markers[1],
                retryable=True,
            )
        ]
    )

    result = await _executor(
        _single_config(fallback_authorized=True),
        primary=primary,
        fallback=fallback,
    ).sense(modalities, _request(modalities))

    assert result.observations == {}
    assert result.used_provider is None
    assert result.used_model is None
    assert result.fallback_used is False
    assert result.failures == (
        ErrorCode.PROVIDER_TIMEOUT,
        ErrorCode.PROVIDER_UNAVAILABLE,
    )
    assert all(marker not in repr(result) for marker in private_markers)


@pytest.mark.asyncio
async def test_cancellation_propagates_without_starting_fallback() -> None:
    """Catching cancellation as a Provider failure could start an unwanted paid call."""
    modalities = frozenset({Modality.VIDEO_VISUAL})
    primary = ScriptedProvider([asyncio.CancelledError()])
    fallback = ScriptedProvider(
        [_success("fallback", modalities, model="fallback-model")]
    )

    with pytest.raises(asyncio.CancelledError):
        await _executor(
            _single_config(fallback_authorized=True),
            primary=primary,
            fallback=fallback,
        ).sense(modalities, _request(modalities))

    assert len(primary.requests) == 1
    assert fallback.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            SensoryError(ErrorCode.CONFIG_INVALID, "private config marker"),
            id="config-invalid",
        ),
        pytest.param(
            SensoryError(ErrorCode.MEDIA_TOO_LARGE, "private local marker"),
            id="invalid-local-media",
        ),
        pytest.param(
            SensoryError(ErrorCode.PROVIDER_AUTH_FAILED, "private auth marker"),
            id="provider-auth",
        ),
        pytest.param(
            SensoryError(
                ErrorCode.PROVIDER_CAPABILITY_REJECTED,
                "private capability marker",
            ),
            id="provider-capability",
        ),
    ],
)
async def test_terminal_errors_never_start_fallback(error: SensoryError) -> None:
    """Falling back on terminal errors would resend invalid or unauthorized requests."""
    modalities = frozenset({Modality.VIDEO_VISUAL})
    primary = ScriptedProvider([error])
    fallback = ScriptedProvider(
        [_success("fallback", modalities, model="fallback-model")]
    )

    result = await _executor(
        _single_config(fallback_authorized=True),
        primary=primary,
        fallback=fallback,
    ).sense(modalities, _request(modalities))

    assert result.failures == (error.code,)
    assert result.observations == {}
    assert fallback.requests == []


@pytest.mark.asyncio
async def test_executor_does_not_add_an_implicit_provider_retry() -> None:
    """Retrying an adapter call here would exceed the one transport retry boundary."""
    modalities = frozenset({Modality.VIDEO_VISUAL})
    primary = ScriptedProvider(
        [
            SensoryError(
                ErrorCode.PROVIDER_TIMEOUT,
                "private timeout",
                retryable=True,
            ),
            _success("primary", modalities),
        ]
    )

    result = await _executor(
        _single_config(),
        primary=primary,
    ).sense(modalities, _request(modalities))

    assert len(primary.requests) == 1
    assert result.failures == (
        ErrorCode.PROVIDER_TIMEOUT,
        ErrorCode.FALLBACK_NOT_AUTHORIZED,
    )


@pytest.mark.asyncio
async def test_verified_joint_request_uses_one_provider_call_for_exact_modalities() -> None:
    """Splitting a verified joint request would duplicate uploads and paid Provider calls."""
    modalities = frozenset({Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO})
    config = AppConfig(
        providers={
            "joint": _provider_config(*modalities, joint=(modalities,)),
        },
        routes=RoutesConfig(
            video_visual=RouteConfig(primary="joint"),
            video_audio=RouteConfig(primary="joint"),
        ),
    )
    joint = ScriptedProvider([_success("joint", modalities)])

    result = await _executor(config, joint=joint).sense(
        modalities,
        _request(modalities),
    )

    assert set(result.observations) == modalities
    assert result.requested_provider == result.used_provider == "joint"
    assert result.requested_model == result.used_model == "test-model"
    assert result.fallback_used is False
    assert len(joint.requests) == 1
    assert joint.requests[0].requested_modalities == modalities


@pytest.mark.asyncio
async def test_missing_joint_candidate_is_rejected_without_combining_providers() -> None:
    """Combining eye and ear routes would invent an unauthorized joint Provider path."""
    modalities = frozenset({Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO})
    config = AppConfig(
        providers={
            "eye": _provider_config(Modality.VIDEO_VISUAL),
            "ear": _provider_config(Modality.VIDEO_AUDIO),
        },
        routes=RoutesConfig(
            video_visual=RouteConfig(primary="eye"),
            video_audio=RouteConfig(primary="ear"),
        ),
    )
    eye = ScriptedProvider([_success("eye", frozenset({Modality.VIDEO_VISUAL}))])
    ear = ScriptedProvider([_success("ear", frozenset({Modality.VIDEO_AUDIO}))])

    with pytest.raises(SensoryError) as caught:
        await _executor(config, eye=eye, ear=ear).sense(
            modalities,
            _request(modalities),
        )

    assert caught.value.code is ErrorCode.CAPABILITY_NOT_VERIFIED
    assert eye.requests == []
    assert ear.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "returned_modalities",
    [
        pytest.param(frozenset(), id="missing"),
        pytest.param(
            frozenset({Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO}),
            id="extra",
        ),
    ],
)
async def test_provider_result_must_exactly_cover_requested_modalities(
    returned_modalities: frozenset[Modality],
) -> None:
    """Accepting a partial or extra map could mislabel evidence from the wrong modality."""
    modalities = frozenset({Modality.VIDEO_VISUAL})
    primary = ScriptedProvider([_success("primary", returned_modalities)])

    result = await _executor(
        _single_config(),
        primary=primary,
    ).sense(modalities, _request(modalities))

    assert result.observations == {}
    assert result.failures == (ErrorCode.PROVIDER_CAPABILITY_REJECTED,)
    assert result.used_provider is None


@pytest.mark.asyncio
async def test_envelope_modality_must_match_its_map_key() -> None:
    """Trusting only map keys could return a mislabeled Provider envelope."""
    modalities = frozenset({Modality.VIDEO_VISUAL})
    malformed = ProviderCallResult(
        observations={
            Modality.VIDEO_VISUAL: _observation(Modality.VIDEO_AUDIO, "wrong")
        },
        provider_id="primary",
        model="test-model",
        remote_file_deleted=None,
    )

    result = await _executor(
        _single_config(),
        primary=ScriptedProvider([malformed]),
    ).sense(modalities, _request(modalities))

    assert result.failures == (ErrorCode.PROVIDER_CAPABILITY_REJECTED,)
    assert result.observations == {}


@pytest.mark.asyncio
async def test_provider_result_identity_mismatch_is_rejected() -> None:
    """Trusting adapter-supplied identity could misreport which Provider received media."""
    modalities = frozenset({Modality.VIDEO_VISUAL})
    primary = ScriptedProvider([_success("different-provider", modalities)])

    result = await _executor(
        _single_config(),
        primary=primary,
    ).sense(modalities, _request(modalities))

    assert result.failures == (ErrorCode.PROVIDER_CAPABILITY_REJECTED,)
    assert result.used_provider is None


@pytest.mark.asyncio
async def test_request_modalities_must_match_executor_modalities_before_call() -> None:
    """Allowing two modality sets could route under one authorization and send another."""
    requested = frozenset({Modality.VIDEO_VISUAL})
    primary = ScriptedProvider([_success("primary", requested)])

    with pytest.raises(SensoryError) as caught:
        await _executor(_single_config(), primary=primary).sense(
            requested,
            _request(frozenset({Modality.VIDEO_AUDIO})),
        )

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert primary.requests == []


class DuplicateCandidateRouter:
    def candidates(self, modality: Modality) -> list[ProviderCandidate]:
        del modality
        return [
            ProviderCandidate(
                provider_id="primary",
                expected_model="test-model",
                modalities=frozenset({Modality.VIDEO_VISUAL}),
            ),
            ProviderCandidate(
                provider_id="primary",
                expected_model="test-model",
                modalities=frozenset({Modality.VIDEO_VISUAL}),
                is_fallback=True,
            ),
        ]

    def joint_candidate(
        self,
        modalities: frozenset[Modality],
    ) -> NoReturn:
        del modalities
        raise AssertionError("joint_candidate must not be used")


@pytest.mark.asyncio
async def test_duplicate_provider_candidates_are_rejected_before_any_call() -> None:
    """Invoking a duplicate candidate would resend media to one Provider twice."""
    modalities = frozenset({Modality.VIDEO_VISUAL})
    primary = ScriptedProvider([_success("primary", modalities)])
    executor = ProviderExecutor(
        router=cast(ProviderRouter, DuplicateCandidateRouter()),
        registry=ProviderRegistry({"primary": cast(SensoryProvider, primary)}),
    )

    with pytest.raises(SensoryError) as caught:
        await executor.sense(modalities, _request(modalities))

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert primary.requests == []


@pytest.mark.asyncio
async def test_unexpected_provider_exception_is_sanitized_without_fallback() -> None:
    """Propagating an arbitrary exception could leak its raw message to the MCP client."""
    modalities = frozenset({Modality.VIDEO_VISUAL})
    marker = "private-runtime-diagnostic"
    result = await _executor(
        _single_config(fallback_authorized=True),
        primary=ScriptedProvider([RuntimeError(marker)]),
        fallback=ScriptedProvider([_success("fallback", modalities)]),
    ).sense(modalities, _request(modalities))

    assert result.failures == (ErrorCode.PROVIDER_UNAVAILABLE,)
    assert marker not in repr(result)


class SharedCandidateRouter:
    def __init__(self) -> None:
        self.entries = [
            ProviderCandidate(
                provider_id="primary",
                expected_model="test-model",
                modalities=frozenset({Modality.VIDEO_VISUAL}),
            ),
            ProviderCandidate(
                provider_id="fallback",
                expected_model="fallback-model",
                modalities=frozenset({Modality.VIDEO_VISUAL}),
                is_fallback=True,
            ),
        ]

    def candidates(self, modality: Modality) -> list[ProviderCandidate]:
        assert modality is Modality.VIDEO_VISUAL
        return self.entries

    def joint_candidate(
        self,
        modalities: frozenset[Modality],
    ) -> NoReturn:
        del modalities
        raise AssertionError("joint_candidate must not be used")


class MutatingPrimary:
    def __init__(self, router: SharedCandidateRouter) -> None:
        self._router = router
        self.requests: list[ProviderRequest] = []

    async def sense(self, request: ProviderRequest) -> ProviderCallResult:
        self.requests.append(request)
        authorized_fallback = self._router.entries[1]
        intruder = ProviderCandidate(
            provider_id="intruder",
            expected_model="intruder-model",
            modalities=frozenset({Modality.VIDEO_VISUAL}),
            is_fallback=True,
        )
        self._router.entries[:] = [
            authorized_fallback,
            intruder,
        ]
        raise SensoryError(
            ErrorCode.PROVIDER_TIMEOUT,
            "private timeout",
            retryable=True,
        )


@pytest.mark.asyncio
async def test_route_sequence_is_snapshotted_before_the_first_provider_await() -> None:
    """Mutating the Router list during a call must not inject or delete a Provider."""
    modalities = frozenset({Modality.VIDEO_VISUAL})
    router = SharedCandidateRouter()
    primary = MutatingPrimary(router)
    fallback = ScriptedProvider(
        [_success("fallback", modalities, model="fallback-model")]
    )
    intruder = ScriptedProvider(
        [_success("intruder", modalities, model="intruder-model")]
    )
    executor = ProviderExecutor(
        router=cast(ProviderRouter, router),
        registry=ProviderRegistry(
            {
                "primary": cast(SensoryProvider, primary),
                "fallback": cast(SensoryProvider, fallback),
                "intruder": cast(SensoryProvider, intruder),
            }
        ),
    )

    result = await executor.sense(modalities, _request(modalities))

    assert result.used_provider == "fallback"
    assert result.used_model == "fallback-model"
    assert result.fallback_used is True
    assert len(primary.requests) == len(fallback.requests) == 1
    assert intruder.requests == []


class MutableModalitiesRouter:
    def candidates(self, modality: Modality) -> list[ProviderCandidate]:
        return [
            ProviderCandidate(
                provider_id="primary",
                expected_model="test-model",
                modalities=cast(frozenset[Modality], {modality}),
            )
        ]

    def joint_candidate(
        self,
        modalities: frozenset[Modality],
    ) -> NoReturn:
        del modalities
        raise AssertionError("joint_candidate must not be used")


class NonBooleanFallbackRouter:
    def candidates(self, modality: Modality) -> list[ProviderCandidate]:
        return [
            ProviderCandidate(
                provider_id="primary",
                expected_model="test-model",
                modalities=frozenset({modality}),
                is_fallback=cast(bool, 0),
            )
        ]

    def joint_candidate(
        self,
        modalities: frozenset[Modality],
    ) -> NoReturn:
        del modalities
        raise AssertionError("joint_candidate must not be used")


@pytest.mark.asyncio
async def test_mutable_candidate_modalities_are_rejected_before_any_call() -> None:
    """Retaining a nested mutable modality set would invalidate the route snapshot."""
    modalities = frozenset({Modality.VIDEO_VISUAL})
    primary = ScriptedProvider([_success("primary", modalities)])
    executor = ProviderExecutor(
        router=cast(ProviderRouter, MutableModalitiesRouter()),
        registry=ProviderRegistry({"primary": cast(SensoryProvider, primary)}),
    )

    with pytest.raises(SensoryError) as caught:
        await executor.sense(modalities, _request(modalities))

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert primary.requests == []


@pytest.mark.asyncio
async def test_non_boolean_candidate_fallback_flag_is_rejected_before_call() -> None:
    """Accepting integer truthiness could alter the immutable authorization snapshot."""
    modalities = frozenset({Modality.VIDEO_VISUAL})
    primary = ScriptedProvider([_success("primary", modalities)])
    executor = ProviderExecutor(
        router=cast(ProviderRouter, NonBooleanFallbackRouter()),
        registry=ProviderRegistry({"primary": cast(SensoryProvider, primary)}),
    )

    with pytest.raises(SensoryError) as caught:
        await executor.sense(modalities, _request(modalities))

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert primary.requests == []


class RaisingResult:
    provider_id = "primary"

    @property
    def observations(self) -> NoReturn:
        raise RuntimeError("private-result-property")


class RaisingObservationDict(dict[Modality, ObservationEnvelope]):
    def __iter__(self) -> NoReturn:
        raise RuntimeError("private-observation-iterator")


class ModelNameSubclass(str):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "private_marker"),
    [
        pytest.param(None, "", id="none"),
        pytest.param(object(), "", id="plain-object"),
        pytest.param(RaisingResult(), "private-result-property", id="raising-property"),
        pytest.param(
            ProviderCallResult(
                observations=RaisingObservationDict(),
                provider_id="primary",
                model="test-model",
                remote_file_deleted=None,
            ),
            "private-observation-iterator",
            id="malicious-dict-subclass",
        ),
    ],
)
async def test_malformed_provider_result_is_terminal_and_raw_free(
    result: object,
    private_marker: str,
) -> None:
    """Malformed return objects must not escape their properties or start fallback."""
    modalities = frozenset({Modality.VIDEO_VISUAL})
    fallback = ScriptedProvider(
        [_success("fallback", modalities, model="fallback-model")]
    )

    executed = await _executor(
        _single_config(fallback_authorized=True),
        primary=ScriptedProvider([result]),
        fallback=fallback,
    ).sense(modalities, _request(modalities))

    assert executed.failures == (ErrorCode.PROVIDER_CAPABILITY_REJECTED,)
    assert executed.observations == {}
    assert executed.used_provider is None
    assert executed.used_model is None
    if private_marker:
        assert private_marker not in repr(executed)
    assert fallback.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "returned_model",
    [
        pytest.param("wrong-model", id="wrong"),
        pytest.param("", id="empty"),
        pytest.param("x" * 257, id="overlong"),
        pytest.param(ModelNameSubclass("test-model"), id="str-subclass"),
    ],
)
async def test_result_model_must_exactly_match_the_routed_model(
    returned_model: str,
) -> None:
    """Accepting inconsistent model metadata would misreport which model saw media."""
    modalities = frozenset({Modality.VIDEO_VISUAL})
    fallback = ScriptedProvider(
        [_success("fallback", modalities, model="fallback-model")]
    )

    result = await _executor(
        _single_config(fallback_authorized=True),
        primary=ScriptedProvider(
            [_success("primary", modalities, model=returned_model)]
        ),
        fallback=fallback,
    ).sense(modalities, _request(modalities))

    assert result.failures == (ErrorCode.PROVIDER_CAPABILITY_REJECTED,)
    assert result.used_provider is None
    assert result.used_model is None
    assert fallback.requests == []


@pytest.mark.asyncio
async def test_control_base_exception_from_provider_is_not_sanitized() -> None:
    """Catching control BaseExceptions would prevent immediate process-level control."""
    modalities = frozenset({Modality.VIDEO_VISUAL})

    with pytest.raises(SystemExit):
        await _executor(
            _single_config(fallback_authorized=True),
            primary=ScriptedProvider([SystemExit(7)]),
            fallback=ScriptedProvider(
                [_success("fallback", modalities, model="fallback-model")]
            ),
        ).sense(modalities, _request(modalities))
