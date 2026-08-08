from __future__ import annotations

import threading
from pathlib import Path

import pytest
import yaml
from filelock import Timeout
from pydantic import ValidationError

from cove_sensory_mcp.config.schema import AppConfig, ProviderConfig, RoutesConfig
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import Modality, RouteConfig


def test_load_returns_version_one_config_when_file_is_missing(tmp_path: Path) -> None:
    """Changing the missing-file branch must not make a fresh install unusable."""
    config = ConfigStore(tmp_path / "config.yaml").load()

    assert config == AppConfig(version=1)


def test_save_and_load_round_trip_unicode_provider_and_windows_path(
    tmp_path: Path,
) -> None:
    """Changing YAML serialization must not corrupt Unicode or Windows-style paths."""
    path = tmp_path / "config.yaml"
    expected = AppConfig(
        providers={
            "星河视觉": ProviderConfig(
                adapter="openai-compatible",
                base_url="https://api.example.test/v1",
                model="vision-1",
                credential_ref="test-credential-reference",
            )
        },
        routes=RoutesConfig(),
        allowed_media_roots=[r"C:\\Media Library\\感官"],
    )

    store = ConfigStore(path)
    store.save(expected)

    assert store.load() == expected


def test_save_persists_credential_reference_without_secret_value(
    tmp_path: Path,
) -> None:
    """Replacing reference-only storage with plaintext credential storage must fail."""
    path = tmp_path / "config.yaml"
    fake_secret = "test-secret-never-persisted"
    config = AppConfig(
        providers={
            "gemini": ProviderConfig(
                adapter="gemini",
                model="gemini-test",
                credential_ref="gemini-main",
            )
        }
    )

    ConfigStore(path).save(config)

    saved = path.read_text(encoding="utf-8")
    assert "credential_ref: gemini-main" in saved
    assert fake_secret not in saved
    assert "api_key:" not in saved


def test_save_replaces_destination_with_temporary_file_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing atomic replacement with a direct destination write must fail."""
    path = tmp_path / "config.yaml"
    path.write_text("version: 1\n", encoding="utf-8")
    replaced: list[tuple[Path, Path]] = []
    real_replace = Path.replace

    def recording_replace(source: Path, destination: Path) -> Path:
        replaced.append((source, destination))
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "replace", recording_replace)

    ConfigStore(path).save(AppConfig())

    assert replaced == [(tmp_path / "config.yaml.tmp", path)]
    assert (
        path.read_text(encoding="utf-8")
        == "version: 1\nproviders: {}\nroutes: {}\nlimits: {}\nallowed_media_roots: []\n"
    )


def test_two_store_instances_merge_racing_provider_route_and_settings_updates(
    tmp_path: Path,
) -> None:
    """Separate processes must not lose late unrelated configuration updates."""
    path = tmp_path / "config.yaml"
    first = ConfigStore(path)
    second = ConfigStore(path)
    first.save(AppConfig())
    start = threading.Barrier(3)
    failures: list[BaseException] = []

    def add_provider_and_route() -> None:
        try:
            start.wait()

            def mutate(config: AppConfig) -> None:
                config.providers["vision"] = ProviderConfig(
                    adapter="gemini",
                    model="test-model",
                    credential_ref="test-ref",
                    declared_capabilities={Modality.IMAGE: True},
                    verified_capabilities={Modality.IMAGE: True},
                )
                config.routes.image = RouteConfig(primary="vision")

            first.update(mutate)
        except (
            AssertionError,
            SensoryError,
            threading.BrokenBarrierError,
        ) as error:  # pragma: no cover - asserted below
            failures.append(error)

    def add_unrelated_setting() -> None:
        try:
            start.wait()
            second.update(
                lambda config: config.allowed_media_roots.append("D:/Shared Media")
            )
        except (
            AssertionError,
            SensoryError,
            threading.BrokenBarrierError,
        ) as error:  # pragma: no cover - asserted below
            failures.append(error)

    threads = [
        threading.Thread(target=add_provider_and_route),
        threading.Thread(target=add_unrelated_setting),
    ]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert not failures
    assert all(not thread.is_alive() for thread in threads)
    saved = ConfigStore(path).load()
    assert set(saved.providers) == {"vision"}
    assert saved.routes.image == RouteConfig(primary="vision")
    assert saved.allowed_media_roots == ["D:/Shared Media"]


def test_update_mutator_conflict_aborts_without_overwriting_latest_config(
    tmp_path: Path,
) -> None:
    """A caller's explicit conflict check must leave the latest file untouched."""
    store = ConfigStore(tmp_path / "config.yaml")
    store.save(AppConfig(allowed_media_roots=["late-setting"]))

    def reject_conflict(config: AppConfig) -> None:
        assert config.allowed_media_roots == ["late-setting"]
        raise SensoryError(ErrorCode.CONFIG_INVALID, "The configuration changed.")

    with pytest.raises(SensoryError) as caught:
        store.update(reject_conflict)

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert store.load().allowed_media_roots == ["late-setting"]


def test_lock_and_io_failures_are_sanitized_without_private_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "private-config.yaml"
    store = ConfigStore(path)

    def fail_replace(source: Path, destination: Path) -> Path:
        del source, destination
        raise OSError(f"private path: {path}")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(SensoryError) as caught:
        store.save(AppConfig())

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert str(caught.value) == "The configuration file could not be updated safely."
    assert str(tmp_path) not in str(caught.value)


def test_lock_timeout_is_sanitized_without_lock_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "private-config.yaml"
    store = ConfigStore(path)

    def fail_lock(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise Timeout(str(path.with_suffix(".yaml.lock")))

    monkeypatch.setattr(
        "cove_sensory_mcp.config.store.FileLock.acquire",
        fail_lock,
    )
    with pytest.raises(SensoryError) as caught:
        store.update(lambda config: config)

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert str(caught.value) == "The configuration file could not be updated safely."
    assert str(tmp_path) not in str(caught.value)


def test_malformed_yaml_raises_config_invalid_without_overwriting_file(
    tmp_path: Path,
) -> None:
    """Changing invalid-load handling must not erase the user's recoverable file."""
    path = tmp_path / "config.yaml"
    original = "providers: [not: valid\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(SensoryError) as exc_info:
        ConfigStore(path).load()

    assert exc_info.value.code is ErrorCode.CONFIG_INVALID
    assert str(exc_info.value) == "The configuration file is invalid."
    assert str(tmp_path) not in str(exc_info.value)
    assert path.read_text(encoding="utf-8") == original


def test_unknown_top_level_field_is_rejected(tmp_path: Path) -> None:
    """Changing top-level schema strictness must not accept misspelled settings."""
    path = tmp_path / "config.yaml"
    path.write_text("version: 1\nunexpected_setting: true\n", encoding="utf-8")

    with pytest.raises(SensoryError) as exc_info:
        ConfigStore(path).load()

    assert exc_info.value.code is ErrorCode.CONFIG_INVALID


def test_provider_schema_rejects_plaintext_api_key() -> None:
    """Adding a plaintext api_key field must fail instead of widening the credential boundary."""
    with pytest.raises(ValidationError):
        ProviderConfig(
            adapter="gemini", model="gemini-test", api_key="test-secret-never-persisted"
        )


@pytest.mark.parametrize(
    "environment_name",
    [
        pytest.param("", id="blank"),
        pytest.param("1PRIVATE_KEY", id="leading-digit"),
        pytest.param("PRIVATE-KEY", id="hyphen"),
        pytest.param("PRIVATE.KEY", id="dot"),
        pytest.param("PRIVATE\nKEY", id="multiline"),
        pytest.param("P" * 129, id="oversized"),
    ],
)
def test_provider_schema_rejects_nonportable_environment_name_without_echo(
    environment_name: str,
) -> None:
    """Relaxing environment-name validation could create platform-specific secret lookup."""
    with pytest.raises(ValidationError) as exc_info:
        ProviderConfig(
            adapter="gemini",
            model="gemini-test",
            api_key_env=environment_name,
        )

    if environment_name:
        assert environment_name not in str(exc_info.value)


@pytest.mark.parametrize(
    "provider_id",
    [
        pytest.param("", id="blank"),
        pytest.param("   ", id="whitespace"),
        pytest.param("private\nprovider", id="multiline"),
        pytest.param("private\x00provider", id="control-character"),
        pytest.param("private/provider", id="invalid-character"),
        pytest.param("p" * 65, id="oversized"),
    ],
)
def test_load_rejects_invalid_provider_map_identifier_without_echo(
    tmp_path: Path, provider_id: str
) -> None:
    """Weakening provider-map key validation could expose unsafe identifiers in CLI output."""
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "providers": {
                    provider_id: {
                        "adapter": "gemini",
                        "model": "gemini-test",
                        "credential_ref": "private-reference",
                    }
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(SensoryError) as exc_info:
        ConfigStore(path).load()

    assert exc_info.value.code is ErrorCode.CONFIG_INVALID
    assert str(exc_info.value) == "The configuration file is invalid."
    if provider_id:
        assert provider_id not in str(exc_info.value)


@pytest.mark.parametrize(
    ("route", "private_identifier"),
    [
        pytest.param(
            {"primary": "private/provider", "fallbacks": []},
            "private/provider",
            id="primary",
        ),
        pytest.param(
            {
                "primary": "gemini",
                "fallbacks": [{"provider": "private\nprovider", "authorized": True}],
            },
            "private\nprovider",
            id="fallback",
        ),
    ],
)
def test_load_rejects_invalid_route_provider_identifier_without_echo(
    tmp_path: Path, route: dict[str, object], private_identifier: str
) -> None:
    """Bypassing the shared identifier type in routes could reintroduce printable raw keys."""
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "providers": {
                    "gemini": {
                        "adapter": "gemini",
                        "model": "gemini-test",
                        "credential_ref": "private-reference",
                    }
                },
                "routes": {"image": route},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SensoryError) as exc_info:
        ConfigStore(path).load()

    assert exc_info.value.code is ErrorCode.CONFIG_INVALID
    assert str(exc_info.value) == "The configuration file is invalid."
    assert private_identifier not in str(exc_info.value)


@pytest.mark.parametrize(
    "base_url",
    [
        pytest.param("not-a-url", id="non-url"),
        pytest.param("http://api.example.test/v1", id="non-https"),
        pytest.param("https:///v1", id="missing-hostname"),
        pytest.param("https://api.example.test:not-a-port/v1", id="malformed-port"),
        pytest.param("https://api.example.test:70000/v1", id="out-of-range-port"),
        pytest.param("https://user@api.example.test/v1", id="userinfo"),
        pytest.param("https://api.example.test/v1?mode=fast", id="query"),
        pytest.param("https://api.example.test/v1#regional", id="fragment"),
    ],
)
def test_load_rejects_invalid_provider_endpoint_without_echo(
    tmp_path: Path, base_url: str
) -> None:
    """Loading an endpoint outside the HTTPS origin contract could misroute later media."""
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "providers": {
                    "custom": {
                        "adapter": "openai-compatible",
                        "base_url": base_url,
                        "model": "vision-1",
                        "credential_ref": "custom-main",
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
    assert base_url not in str(exc_info.value)


@pytest.mark.parametrize(
    "provider",
    [
        pytest.param(
            {
                "adapter": "gemini",
                "model": "gemini-test",
                "credential_ref": "gemini-main",
            },
            id="built-in-without-endpoint",
        ),
        pytest.param(
            {
                "adapter": "openai-compatible",
                "base_url": "https://api.example.test:8443/v1",
                "model": "vision-1",
                "credential_ref": "custom-main",
            },
            id="custom-https-endpoint",
        ),
    ],
)
def test_load_accepts_provider_with_valid_endpoint_contract(
    tmp_path: Path, provider: dict[str, object]
) -> None:
    """Overtightening the endpoint invariant could reject built-ins or valid custom origins."""
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"version": 1, "providers": {"provider-one": provider}}),
        encoding="utf-8",
    )

    loaded = ConfigStore(path).load()

    assert loaded.providers["provider-one"].base_url == provider.get("base_url")


@pytest.mark.parametrize(
    "base_url",
    [
        "https://test-user:test-secret-never-persisted@example.test/v1",
        "https://example.test/v1?api_key=test-secret-never-persisted",
        "https://example.test/v1#access_token=test-secret-never-persisted",
        "https://example.test/v1?mode=fast;api_key=plaintext-secret",
        "https://example.test/v1#view=compact;access_token=plaintext-secret",
    ],
)
def test_provider_schema_rejects_secret_bearing_base_url_before_save(
    tmp_path: Path, base_url: str
) -> None:
    """Removing endpoint credential checks must not let secrets reach the config file."""
    path = tmp_path / "config.yaml"

    with pytest.raises(ValidationError) as exc_info:
        ConfigStore(path).save(
            AppConfig(
                providers={
                    "custom": ProviderConfig(
                        adapter="openai-compatible",
                        base_url=base_url,
                        model="vision-1",
                        credential_ref="custom-main",
                    )
                }
            )
        )

    assert "base_url must not contain credentials" in str(exc_info.value)
    assert base_url not in str(exc_info.value)
    assert not path.exists()
