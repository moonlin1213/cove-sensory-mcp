"""Credential storage and redaction that keep secrets outside application config."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Protocol

import keyring

from cove_sensory_mcp.errors import ErrorCode, SensoryError

_KEYRING_SERVICE_NAME = "cove-sensory-mcp"
_INVALID_REFERENCE_MESSAGE = "The credential reference is invalid."
_INVALID_SECRET_MESSAGE = "The provider credential is invalid."
_MISSING_SECRET_MESSAGE = (
    "A provider credential is required. Set an environment variable or configure a credential."
)
_KEYRING_UNAVAILABLE_MESSAGE = (
    "Credential storage is unavailable. Set an environment variable for this provider."
)


class SecretStore(Protocol):
    """Interface for non-config credential storage."""

    def set(self, ref: str, secret: str) -> None:
        """Store one validated secret under its non-secret reference."""

    def get(self, ref: str, env_name: str | None = None) -> str:
        """Return an environment override or the secret stored for ``ref``."""


def _validate_ref(ref: str) -> None:
    """Reject references that could be blank or span multiple backend entries."""
    if not ref.strip() or "\n" in ref or "\r" in ref:
        raise SensoryError(ErrorCode.CONFIG_INVALID, _INVALID_REFERENCE_MESSAGE)


def _validate_secret(secret: str) -> None:
    """Reject absent and implausibly short credential values."""
    if not secret.strip() or len(secret) < 8:
        raise SensoryError(ErrorCode.SETUP_REQUIRED, _INVALID_SECRET_MESSAGE)


def _missing_secret_error() -> SensoryError:
    """Return the public error used when no credential source contains a value."""
    return SensoryError(ErrorCode.SETUP_REQUIRED, _MISSING_SECRET_MESSAGE)


class KeyringSecretStore:
    """Use the operating system's credential store for production secrets."""

    def set(self, ref: str, secret: str) -> None:
        """Store a validated secret in the project keyring namespace."""
        _validate_ref(ref)
        _validate_secret(secret)
        try:
            keyring.set_password(_KEYRING_SERVICE_NAME, ref, secret)
        except Exception as exc:
            raise SensoryError(
                ErrorCode.SETUP_REQUIRED,
                _KEYRING_UNAVAILABLE_MESSAGE,
                cause=exc,
            ) from exc

    def get(self, ref: str, env_name: str | None = None) -> str:
        """Return the explicit environment override before consulting the keyring."""
        _validate_ref(ref)
        if env_name is not None and env_name in os.environ:
            secret = os.environ[env_name]
            _validate_secret(secret)
            return secret
        try:
            secret = keyring.get_password(_KEYRING_SERVICE_NAME, ref)
        except Exception as exc:
            raise SensoryError(
                ErrorCode.SETUP_REQUIRED,
                _KEYRING_UNAVAILABLE_MESSAGE,
                cause=exc,
            ) from exc
        if secret is None:
            raise _missing_secret_error()
        _validate_secret(secret)
        return secret


class MemorySecretStore:
    """Test-only in-memory implementation of :class:`SecretStore`."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def set(self, ref: str, secret: str) -> None:
        """Store a validated test secret without touching persistent storage."""
        _validate_ref(ref)
        _validate_secret(secret)
        self.values[ref] = secret

    def get(self, ref: str, env_name: str | None = None) -> str:
        """Return the explicit environment override before consulting test memory."""
        _validate_ref(ref)
        if env_name is not None and env_name in os.environ:
            secret = os.environ[env_name]
            _validate_secret(secret)
            return secret
        try:
            secret = self.values[ref]
        except KeyError as exc:
            raise _missing_secret_error() from exc
        _validate_secret(secret)
        return secret


def redact_text(value: str, secrets: Iterable[str]) -> str:
    """Replace every supplied non-empty secret value with a fixed public marker."""
    for secret in sorted({secret for secret in secrets if secret}, key=len, reverse=True):
        value = value.replace(secret, "[REDACTED]")
    return value
