"""Tests for the non-config credential storage and redaction boundary."""

from __future__ import annotations

from typing import Any

import keyring
import pytest

from cove_sensory_mcp.config.secrets import (
    KeyringSecretStore,
    MemorySecretStore,
    redact_text,
)
from cove_sensory_mcp.errors import ErrorCode, SensoryError


def test_environment_reference_wins_without_persisting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falling through to another store would violate explicit environment configuration."""
    monkeypatch.setenv("TEST_GEMINI_KEY", "gemini-secret-value")
    store = MemorySecretStore()

    assert store.get("gemini-main", env_name="TEST_GEMINI_KEY") == "gemini-secret-value"
    assert store.values == {}


def test_missing_environment_reference_does_not_fall_through_to_keyring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falling back from an explicit environment source could contact unavailable keyring."""
    environment_name = "TEST_MISSING_API_KEY"
    monkeypatch.delenv(environment_name, raising=False)

    def unexpected_keyring_read(service_name: str, username: str) -> str | None:
        raise AssertionError("explicit environment mode must not consult keyring")

    monkeypatch.setattr(keyring, "get_password", unexpected_keyring_read)

    with pytest.raises(SensoryError) as caught:
        KeyringSecretStore().get("gemini", env_name=environment_name)

    assert caught.value.code is ErrorCode.SETUP_REQUIRED
    assert environment_name not in str(caught.value)


def test_missing_secret_raises_public_error() -> None:
    """Exposing the missing reference would disclose private configuration details."""
    store = MemorySecretStore()

    with pytest.raises(SensoryError) as caught:
        store.get("gemini-main")

    assert caught.value.code is ErrorCode.SETUP_REQUIRED
    assert "gemini-main" not in str(caught.value)


def test_redaction_removes_all_registered_secrets() -> None:
    """Leaving either registered token in public text would breach the privacy boundary."""
    text = redact_text("Bearer secret-one and secret-two", ["secret-one", "secret-two"])

    assert text == "Bearer [REDACTED] and [REDACTED]"


def test_redaction_handles_overlapping_registered_secrets() -> None:
    """Replacing the shorter token first would leave a suffix of the longer token exposed."""
    text = redact_text("Bearer token-secret-value", ["token", "token-secret-value"])

    assert text == "Bearer [REDACTED]"


@pytest.mark.parametrize("reference", ["", " \t", "gemini\nmain", "gemini\rmain"])
def test_set_rejects_blank_or_newline_bearing_reference(reference: str) -> None:
    """Passing an unsafe reference to a backend could expose or corrupt its namespace."""
    with pytest.raises(SensoryError) as caught:
        MemorySecretStore().set(reference, "valid-secret-value")

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    if reference:
        assert reference not in str(caught.value)


@pytest.mark.parametrize("secret", ["", " \t", "short", "seven77"])
def test_set_rejects_blank_or_short_secret(secret: str) -> None:
    """Persisting an unusable credential would create an opaque setup failure later."""
    with pytest.raises(SensoryError) as caught:
        MemorySecretStore().set("gemini-main", secret)

    assert caught.value.code is ErrorCode.SETUP_REQUIRED
    if secret:
        assert secret not in str(caught.value)


def test_keyring_store_uses_project_service_name(fake_keyring: Any) -> None:
    """Changing the service namespace would make saved credentials unavailable to the app."""
    store = KeyringSecretStore()
    secret = "keyring-secret-value"

    store.set("gemini-main", secret)

    assert fake_keyring.values == {("cove-sensory-mcp", "gemini-main"): secret}
    assert store.get("gemini-main") == secret


def test_secret_store_exists_distinguishes_free_and_occupied_references(
    fake_keyring: Any,
) -> None:
    """Without a non-destructive occupancy check, configure can overwrite an existing key."""
    keyring_store = KeyringSecretStore()
    memory_store = MemorySecretStore()

    assert keyring_store.exists("gemini-main") is False
    assert memory_store.exists("gemini-main") is False

    keyring_store.set("gemini-main", "keyring-secret-value")
    memory_store.set("gemini-main", "memory-secret-value")

    assert keyring_store.exists("gemini-main") is True
    assert memory_store.exists("gemini-main") is True


def test_secret_store_delete_removes_only_the_requested_reference(fake_keyring: Any) -> None:
    """Deleting a different key during config rollback would destroy unrelated credentials."""
    store = KeyringSecretStore()
    store.set("gemini-main", "gemini-secret-value")
    store.set("minimax-main", "minimax-secret-value")

    store.delete("gemini-main")

    assert fake_keyring.values == {
        ("cove-sensory-mcp", "minimax-main"): "minimax-secret-value"
    }


def test_memory_secret_store_delete_removes_only_the_requested_reference() -> None:
    """A test backend that cannot model rollback would leave the CLI behavior untested."""
    store = MemorySecretStore()
    store.set("gemini-main", "gemini-secret-value")
    store.set("minimax-main", "minimax-secret-value")

    store.delete("gemini-main")

    assert store.values == {"minimax-main": "minimax-secret-value"}


def test_keyring_backend_failure_has_public_environment_setup_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Formatting a keyring failure would leak backend detail into a public error."""
    backend_error = RuntimeError("backend details with keyring-secret-value")

    def fail_get_password(service_name: str, username: str) -> str | None:
        raise backend_error

    monkeypatch.setattr(keyring, "get_password", fail_get_password)

    with pytest.raises(SensoryError) as caught:
        KeyringSecretStore().get("gemini-main")

    public_message = str(caught.value)
    captured = capsys.readouterr()
    assert caught.value.code is ErrorCode.SETUP_REQUIRED
    assert "environment variable" in public_message.lower()
    assert "gemini-main" not in public_message
    assert "keyring-secret-value" not in public_message
    assert "backend details" not in public_message
    assert caught.value.cause is None
    assert caught.value.__cause__ is backend_error
    assert "keyring-secret-value" not in captured.out
    assert "keyring-secret-value" not in captured.err


def test_keyring_write_failure_retains_backend_error_only_as_chained_cause(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Storing the raw backend error on the public exception would make it serializable."""
    backend_error = RuntimeError("write backend details with keyring-secret-value")

    def fail_set_password(service_name: str, username: str, password: str) -> None:
        raise backend_error

    monkeypatch.setattr(keyring, "set_password", fail_set_password)

    with pytest.raises(SensoryError) as caught:
        KeyringSecretStore().set("gemini-main", "keyring-secret-value")

    public_message = str(caught.value)
    captured = capsys.readouterr()
    assert caught.value.code is ErrorCode.SETUP_REQUIRED
    assert "environment variable" in public_message.lower()
    assert "gemini-main" not in public_message
    assert "keyring-secret-value" not in public_message
    assert "write backend details" not in public_message
    assert caught.value.cause is None
    assert caught.value.__cause__ is backend_error
    assert "keyring-secret-value" not in captured.out
    assert "keyring-secret-value" not in captured.err
