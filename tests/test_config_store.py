from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from cove_sensory_mcp.config.schema import AppConfig, ProviderConfig, RoutesConfig
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.errors import ErrorCode, SensoryError


def test_load_returns_version_one_config_when_file_is_missing(tmp_path: Path) -> None:
    """Changing the missing-file branch must not make a fresh install unusable."""
    config = ConfigStore(tmp_path / "config.yaml").load()

    assert config == AppConfig(version=1)


def test_save_and_load_round_trip_unicode_provider_and_windows_path(tmp_path: Path) -> None:
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


def test_save_persists_credential_reference_without_secret_value(tmp_path: Path) -> None:
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
    assert path.read_text(encoding="utf-8") == "version: 1\nproviders: {}\nroutes: {}\nlimits: {}\nallowed_media_roots: []\n"


def test_malformed_yaml_raises_config_invalid_without_overwriting_file(tmp_path: Path) -> None:
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
        ProviderConfig(adapter="gemini", model="gemini-test", api_key="test-secret-never-persisted")


@pytest.mark.parametrize(
    "base_url",
    [
        "https://test-user:test-secret-never-persisted@example.test/v1",
        "https://example.test/v1?api_key=test-secret-never-persisted",
        "https://example.test/v1#access_token=test-secret-never-persisted",
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
