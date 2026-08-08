from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import Self

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

    assert len(replaced) == 1
    temporary, destination = replaced[0]
    assert temporary.parent == tmp_path
    assert temporary.name.startswith(".config.yaml.")
    assert temporary.name.endswith(".tmp")
    assert destination == path
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


@pytest.mark.parametrize("operation", ["load", "save", "update"])
@pytest.mark.parametrize("use_peer", [False, True], ids=["same-store", "peer-store"])
def test_same_thread_lock_reentry_is_rejected_immediately_without_file_mutation(
    tmp_path: Path,
    operation: str,
    use_peer: bool,
) -> None:
    path = tmp_path / "config.yaml"
    store = ConfigStore(path)
    store.save(AppConfig(allowed_media_roots=["original-setting"]))
    original = path.read_bytes()
    target = ConfigStore(path) if use_peer else store
    elapsed = 10.0

    def reenter(_: AppConfig) -> None:
        nonlocal elapsed
        started = time.monotonic()
        try:
            if operation == "load":
                target.load()
            elif operation == "save":
                target.save(AppConfig(allowed_media_roots=["private-overwrite"]))
            else:
                target.update(
                    lambda config: config.allowed_media_roots.append(
                        "private-overwrite"
                    )
                )
        finally:
            elapsed = time.monotonic() - started

    with pytest.raises(SensoryError) as caught:
        store.update(reenter)

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert str(caught.value) == "A nested configuration transaction is not allowed."
    assert elapsed < 0.5
    assert path.read_bytes() == original
    assert ConfigStore(path).load().allowed_media_roots == ["original-setting"]


def test_cross_thread_updates_remain_serialized(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    first = ConfigStore(path)
    second = ConfigStore(path)
    first.save(AppConfig())
    state_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def mutate(config: AppConfig) -> None:
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        config.allowed_media_roots.append(threading.current_thread().name)
        with state_lock:
            active -= 1

    threads = [
        threading.Thread(target=first.update, args=(mutate,), name="first"),
        threading.Thread(target=second.update, args=(mutate,), name="second"),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert maximum_active == 1
    assert sorted(first.load().allowed_media_roots) == ["first", "second"]


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
    assert list(tmp_path.glob(".private-config.yaml.*.tmp")) == []


def test_validation_failure_creates_no_temporary_yaml(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    store = ConfigStore(path)
    store.save(AppConfig())

    def invalidate(config: AppConfig) -> None:
        config.version = 2  # type: ignore[assignment]
        config.allowed_media_roots.append("private-setting")

    with pytest.raises(SensoryError) as caught:
        store.update(invalidate)

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert list(tmp_path.glob(".config.yaml.*.tmp")) == []
    assert "private-setting" not in path.read_text(encoding="utf-8")


def test_temp_write_failure_is_sanitized_and_cleans_unique_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "config.yaml"
    store = ConfigStore(path)
    real_named_temporary_file = tempfile.NamedTemporaryFile

    class FailingTemporaryFile:
        def __init__(self, **kwargs: object) -> None:
            self._wrapped = real_named_temporary_file(**kwargs)
            self.name = self._wrapped.name

        def __enter__(self) -> Self:
            self._wrapped.__enter__()
            return self

        def write(self, payload: str) -> None:
            del payload
            raise OSError("private-write-marker")

        def __exit__(self, *args: object) -> object:
            return self._wrapped.__exit__(*args)

    monkeypatch.setattr(
        "cove_sensory_mcp.config.store.tempfile.NamedTemporaryFile",
        FailingTemporaryFile,
    )

    with pytest.raises(SensoryError) as caught:
        store.save(AppConfig(allowed_media_roots=["private-yaml-marker"]))

    assert str(caught.value) == "The configuration file could not be updated safely."
    assert list(tmp_path.glob(".config.yaml.*.tmp")) == []
    assert not path.exists()


def test_temp_cleanup_failure_does_not_mask_primary_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "config.yaml"
    store = ConfigStore(path)

    def fail_replace(source: Path, destination: Path) -> Path:
        del source, destination
        raise OSError("primary-private-marker")

    def fail_cleanup(source: Path, *, missing_ok: bool = False) -> None:
        del source, missing_ok
        raise OSError("cleanup-private-marker")

    monkeypatch.setattr(Path, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)

    with pytest.raises(SensoryError) as caught:
        store.save(AppConfig(allowed_media_roots=["private-yaml-marker"]))

    assert str(caught.value) == "The configuration file could not be updated safely."
    assert isinstance(caught.value.__cause__, OSError)
    assert str(caught.value.__cause__) == "primary-private-marker"
    public = str(caught.value)
    assert "private" not in public
    leftovers = list(tmp_path.glob(".config.yaml.*.tmp"))
    assert len(leftovers) == 1
    assert leftovers[0].read_bytes() == b""


def test_concurrent_stores_use_distinct_temporary_files_without_collisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "config.yaml"
    first = ConfigStore(path)
    second = ConfigStore(path)
    sources: list[Path] = []
    real_replace = Path.replace

    def record_replace(source: Path, destination: Path) -> Path:
        sources.append(source)
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "replace", record_replace)
    threads = [
        threading.Thread(
            target=first.save,
            args=(AppConfig(allowed_media_roots=["first"]),),
        ),
        threading.Thread(
            target=second.save,
            args=(AppConfig(allowed_media_roots=["second"]),),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert len(sources) == 2
    assert len({source.name for source in sources}) == 2
    assert all(source.parent == tmp_path for source in sources)
    assert list(tmp_path.glob(".config.yaml.*.tmp")) == []


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
