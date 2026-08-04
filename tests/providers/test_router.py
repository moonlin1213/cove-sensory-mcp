from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest
import yaml
from pydantic import ValidationError

from cove_sensory_mcp.config.schema import AppConfig, ProviderConfig, RoutesConfig
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import Modality, ProviderRef, RouteConfig
from cove_sensory_mcp.providers.base import SensoryProvider
from cove_sensory_mcp.providers.registry import ProviderRegistry
from cove_sensory_mcp.providers.router import ProviderRouter


def _provider(
    *verified_modalities: Modality,
    joint: tuple[frozenset[Modality], ...] = (),
) -> ProviderConfig:
    return ProviderConfig(
        adapter="test-adapter",
        model="test-model",
        credential_ref="test-credential",
        declared_capabilities={modality: True for modality in verified_modalities},
        verified_capabilities={modality: True for modality in verified_modalities},
        verified_joint_capabilities=list(joint),
        adapter_options={"timeout_seconds": 30, "region": "test"},
        last_verified_at=datetime(2026, 8, 4, tzinfo=UTC),
    )


def test_unverified_primary_is_rejected_without_echoing_provider_id() -> None:
    """Skipping the verification gate could send private media to an untested Provider."""
    private_id = "private-primary"
    config = AppConfig(
        providers={private_id: _provider()},
        routes=RoutesConfig(image=RouteConfig(primary=private_id)),
    )

    with pytest.raises(SensoryError) as exc_info:
        ProviderRouter(config).candidates(Modality.IMAGE)

    assert exc_info.value.code is ErrorCode.CAPABILITY_NOT_VERIFIED
    assert str(exc_info.value) == "The configured provider capability is not verified."
    assert private_id not in str(exc_info.value)


def test_verified_primary_is_returned_first() -> None:
    """Dropping the primary-first branch would silently change the user's route choice."""
    config = AppConfig(
        providers={"primary": _provider(Modality.IMAGE)},
        routes=RoutesConfig(image=RouteConfig(primary="primary")),
    )

    candidates = ProviderRouter(config).candidates(Modality.IMAGE)

    assert [candidate.provider_id for candidate in candidates] == ["primary"]
    assert candidates[0].modalities == frozenset({Modality.IMAGE})
    assert candidates[0].is_fallback is False


def test_unauthorized_fallback_is_not_returned() -> None:
    """Ignoring authorization could send media to a Provider the user did not approve."""
    config = AppConfig(
        providers={
            "primary": _provider(Modality.IMAGE),
            "fallback": _provider(Modality.IMAGE),
        },
        routes=RoutesConfig(
            image=RouteConfig(
                primary="primary",
                fallbacks=[ProviderRef(provider="fallback", authorized=False)],
            )
        ),
    )

    candidates = ProviderRouter(config).candidates(Modality.IMAGE)

    assert [candidate.provider_id for candidate in candidates] == ["primary"]


def test_authorized_fallback_is_returned_after_primary() -> None:
    """Reordering a route could invoke fallback before the configured primary."""
    config = AppConfig(
        providers={
            "primary": _provider(Modality.IMAGE),
            "fallback-one": _provider(Modality.IMAGE),
            "fallback-two": _provider(Modality.IMAGE),
        },
        routes=RoutesConfig(
            image=RouteConfig(
                primary="primary",
                fallbacks=[
                    ProviderRef(provider="fallback-one", authorized=True),
                    ProviderRef(provider="fallback-two", authorized=True),
                ],
            )
        ),
    )

    candidates = ProviderRouter(config).candidates(Modality.IMAGE)

    assert [candidate.provider_id for candidate in candidates] == [
        "primary",
        "fallback-one",
        "fallback-two",
    ]
    assert [candidate.is_fallback for candidate in candidates] == [False, True, True]


def test_unknown_provider_id_is_config_invalid_without_identifier_echo() -> None:
    """Accepting a dangling route could defer a deterministic config error until execution."""
    private_id = "private-missing-provider"
    config = AppConfig(routes=RoutesConfig(image=RouteConfig(primary=private_id)))

    with pytest.raises(SensoryError) as exc_info:
        ProviderRouter(config).candidates(Modality.IMAGE)

    assert exc_info.value.code is ErrorCode.CONFIG_INVALID
    assert str(exc_info.value) == "The provider configuration is invalid."
    assert private_id not in str(exc_info.value)


def test_minimax_visual_verification_does_not_authorize_video_audio() -> None:
    """Conflating video sight with hearing could route audio to an incapable Provider."""
    config = AppConfig(
        providers={"minimax-m3": _provider(Modality.VIDEO_VISUAL)},
        routes=RoutesConfig(
            video_audio=RouteConfig(primary="minimax-m3"),
        ),
    )

    with pytest.raises(SensoryError) as exc_info:
        ProviderRouter(config).candidates(Modality.VIDEO_AUDIO)

    assert exc_info.value.code is ErrorCode.CAPABILITY_NOT_VERIFIED


def test_joint_candidate_requires_same_primary_and_exact_verified_set() -> None:
    """Removing exact-set verification could manufacture unsupported joint requests."""
    exact = frozenset({Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO})
    config = AppConfig(
        providers={
            "gemini": _provider(
                Modality.VIDEO_VISUAL,
                Modality.VIDEO_AUDIO,
                joint=(exact,),
            )
        },
        routes=RoutesConfig(
            video_visual=RouteConfig(primary="gemini"),
            video_audio=RouteConfig(primary="gemini"),
        ),
    )

    candidate = ProviderRouter(config).joint_candidate(exact)

    assert candidate is not None
    assert candidate.provider_id == "gemini"
    assert candidate.modalities == exact
    assert candidate.is_fallback is False


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            AppConfig(
                providers={
                    "gemini": _provider(Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO)
                },
                routes=RoutesConfig(
                    video_visual=RouteConfig(primary="gemini"),
                    video_audio=RouteConfig(primary="gemini"),
                ),
            ),
            id="joint-set-not-verified",
        ),
        pytest.param(
            AppConfig(
                providers={
                    "eye": _provider(Modality.VIDEO_VISUAL),
                    "ear": _provider(Modality.VIDEO_AUDIO),
                },
                routes=RoutesConfig(
                    video_visual=RouteConfig(primary="eye"),
                    video_audio=RouteConfig(primary="ear"),
                ),
            ),
            id="different-primary-providers",
        ),
    ],
)
def test_joint_candidate_is_not_manufactured(config: AppConfig) -> None:
    """Combining separate routes into one candidate would invent a joint Provider path."""
    modalities = frozenset({Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO})

    assert ProviderRouter(config).joint_candidate(modalities) is None


def test_split_primary_joint_route_returns_none_before_verification() -> None:
    """Verifying split routes first could raise instead of declining a joint route."""
    modalities = frozenset({Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO})
    config = AppConfig(
        providers={
            "verified-eye": _provider(Modality.VIDEO_VISUAL),
            "unverified-ear": _provider(),
        },
        routes=RoutesConfig(
            video_visual=RouteConfig(primary="verified-eye"),
            video_audio=RouteConfig(primary="unverified-ear"),
        ),
    )

    assert ProviderRouter(config).joint_candidate(modalities) is None


def test_registry_returns_only_the_injected_adapter() -> None:
    """Reading global state in get() could replace the adapter injected by the caller."""
    injected = cast(SensoryProvider, object())
    registry = ProviderRegistry({"gemini": injected})

    assert registry.get("gemini") is injected


def test_registry_unknown_id_is_stable_config_error_without_echo() -> None:
    """Leaking a caller-supplied lookup key could expose a private Provider identifier."""
    private_id = "private-unknown"
    registry = ProviderRegistry({})

    with pytest.raises(SensoryError) as exc_info:
        registry.get(private_id)

    assert exc_info.value.code is ErrorCode.CONFIG_INVALID
    assert str(exc_info.value) == "The provider configuration is invalid."
    assert private_id not in str(exc_info.value)


@pytest.mark.parametrize(
    "provider_kwargs",
    [
        pytest.param(
            {"adapter_options": {f"key-{index}": index for index in range(33)}},
            id="too-many-adapter-options",
        ),
        pytest.param(
            {"adapter_options": {"nested": {"value": "x" * 2049}}},
            id="oversized-nested-adapter-option",
        ),
        pytest.param(
            {
                "verified_joint_capabilities": [
                    frozenset({Modality.IMAGE, Modality.AUDIO}) for _ in range(27)
                ]
            },
            id="too-many-joint-capabilities",
        ),
    ],
)
def test_provider_capability_configuration_is_bounded(
    provider_kwargs: dict[str, object],
) -> None:
    """Removing config bounds could allow an unbounded local YAML structure."""
    with pytest.raises(ValidationError):
        ProviderConfig(
            adapter="test-adapter",
            model="test-model",
            credential_ref="test-credential",
            **provider_kwargs,
        )


@pytest.mark.parametrize(
    ("adapter_options", "private_fragment"),
    [
        pytest.param({"api_key": "plaintext-secret"}, "plaintext-secret", id="api-key"),
        pytest.param(
            {"api-key": "plaintext-secret"},
            "plaintext-secret",
            id="punctuation-alias",
        ),
        pytest.param(
            {"Authorization": "Bearer private-token"},
            "private-token",
            id="authorization",
        ),
        pytest.param(
            {"headers": {"Authorization": "Bearer private-token"}},
            "private-token",
            id="headers-map",
        ),
        pytest.param(
            {"nested": {"safe_name": "private-nested-value"}},
            "private-nested-value",
            id="nested-map",
        ),
        pytest.param(
            {"base_url": "https://private-endpoint.test"},
            "private-endpoint.test",
            id="base-url-shadow",
        ),
        pytest.param(
            {"url": "https://private-endpoint.test"},
            "private-endpoint.test",
            id="url-shadow",
        ),
        pytest.param(
            {"endpoint": "https://private-endpoint.test"},
            "private-endpoint.test",
            id="endpoint-shadow",
        ),
        pytest.param(
            {"api_key_env": "PRIVATE_API_KEY"},
            "PRIVATE_API_KEY",
            id="api-key-env-shadow",
        ),
        pytest.param(
            {"credential_ref": "private-credential-ref"},
            "private-credential-ref",
            id="credential-ref-shadow",
        ),
        pytest.param(
            {"keyring_service": "private-keyring-service"},
            "private-keyring-service",
            id="keyring-shadow",
        ),
        pytest.param(
            {"accessToken": "private-access-token"},
            "private-access-token",
            id="camel-case-token",
        ),
        pytest.param(
            {"secret-source": "private-secret-source"},
            "private-secret-source",
            id="secret-source",
        ),
        pytest.param(
            {"bearer": "private-bearer-token"},
            "private-bearer-token",
            id="bearer-alias",
        ),
        pytest.param(
            {"private_key": "private-signing-key"},
            "private-signing-key",
            id="private-key-alias",
        ),
        pytest.param(
            {"base_uri": "https://private-endpoint.test"},
            "private-endpoint.test",
            id="base-uri-shadow",
        ),
        pytest.param(
            {"environment_variable": "PRIVATE_API_KEY"},
            "PRIVATE_API_KEY",
            id="environment-variable-source",
        ),
        pytest.param(
            {"proxy": "https://private-proxy.test"},
            "private-proxy.test",
            id="proxy-endpoint-shadow",
        ),
    ],
)
def test_config_store_rejects_unsafe_adapter_options_without_echo(
    tmp_path: Path,
    adapter_options: dict[str, object],
    private_fragment: str,
) -> None:
    """Allowing secret or endpoint shadows could persist credentials in local YAML."""
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "providers": {
                    "custom": {
                        "adapter": "openai-compatible",
                        "model": "test-model",
                        "credential_ref": "safe-reference",
                        "adapter_options": adapter_options,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SensoryError) as exc_info:
        ConfigStore(path).load()

    assert exc_info.value.code is ErrorCode.CONFIG_INVALID
    assert str(exc_info.value) == "The configuration file is invalid."
    assert private_fragment not in str(exc_info.value)


def test_unsafe_adapter_option_validation_hides_private_input() -> None:
    """Pydantic diagnostics must not echo a rejected plaintext option value."""
    private = "private-plaintext-credential"

    with pytest.raises(ValidationError) as exc_info:
        ProviderConfig(
            adapter="openai-compatible",
            model="test-model",
            credential_ref="safe-reference",
            adapter_options={"api-key": private},
        )

    assert private not in str(exc_info.value)


def test_safe_scalar_and_list_adapter_options_remain_portable(tmp_path: Path) -> None:
    """Overbroad credential filtering must not reject bounded non-secret tuning options."""
    path = tmp_path / "config.yaml"
    provider = ProviderConfig(
        adapter="openai-compatible",
        model="test-model",
        credential_ref="safe-reference",
        adapter_options={
            "max_output_tokens": 4_096,
            "temperature": 0.2,
            "retry_status_codes": [429, 503],
            "feature_flags": [True, False],
            "media_part_mode": "video_url_data_uri",
            "endpoint_path": "/v1/responses",
        },
    )
    expected = AppConfig(providers={"custom": provider})

    ConfigStore(path).save(expected)

    assert ConfigStore(path).load() == expected
    assert "!!python" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "provider_kwargs",
    [
        pytest.param(
            {"verified_capabilities": {Modality.IMAGE: True}},
            id="verified-capability-not-declared",
        ),
        pytest.param(
            {
                "declared_capabilities": {Modality.IMAGE: True},
                "verified_capabilities": {Modality.IMAGE: True},
                "verified_joint_capabilities": [frozenset({Modality.IMAGE})],
            },
            id="single-modality-joint-set",
        ),
        pytest.param(
            {
                "declared_capabilities": {
                    Modality.IMAGE: True,
                    Modality.AUDIO: True,
                },
                "verified_capabilities": {Modality.IMAGE: True},
                "verified_joint_capabilities": [
                    frozenset({Modality.IMAGE, Modality.AUDIO})
                ],
            },
            id="joint-set-not-individually-verified",
        ),
        pytest.param(
            {
                "declared_capabilities": {
                    Modality.IMAGE: True,
                    Modality.AUDIO: True,
                },
                "verified_capabilities": {
                    Modality.IMAGE: True,
                    Modality.AUDIO: True,
                },
                "verified_joint_capabilities": [
                    frozenset({Modality.IMAGE, Modality.AUDIO}),
                    frozenset({Modality.AUDIO, Modality.IMAGE}),
                ],
            },
            id="duplicate-joint-set",
        ),
        pytest.param(
            {
                "last_verified_at": datetime(  # noqa: DTZ001 - deliberate invalid fixture
                    2026, 8, 4, 10, 0
                )
            },
            id="naive-verification-time",
        ),
    ],
)
def test_provider_verification_state_rejects_incoherent_values(
    provider_kwargs: dict[str, object],
) -> None:
    """Weakening state coherence could advertise a capability without valid proof."""
    with pytest.raises(ValidationError):
        ProviderConfig(
            adapter="test-adapter",
            model="test-model",
            credential_ref="test-credential",
            **provider_kwargs,
        )


def test_provider_verification_time_is_normalized_to_utc() -> None:
    """Keeping arbitrary offsets could make persisted verification ordering ambiguous."""
    config = ProviderConfig(
        adapter="test-adapter",
        model="test-model",
        credential_ref="test-credential",
        last_verified_at=datetime(
            2026,
            8,
            4,
            18,
            30,
            tzinfo=timezone(timedelta(hours=8)),
        ),
    )

    assert config.last_verified_at == datetime(2026, 8, 4, 10, 30, tzinfo=UTC)
