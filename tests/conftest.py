"""Shared test doubles for credential-storage boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field

import keyring
import pytest


@dataclass
class FakeKeyring:
    """In-memory keyring double that prevents tests from reaching the OS store."""

    values: dict[tuple[str, str], str] = field(default_factory=dict)

    def get_password(self, service_name: str, username: str) -> str | None:
        """Return a stored fake credential without any operating-system side effect."""
        return self.values.get((service_name, username))

    def set_password(self, service_name: str, username: str, password: str) -> None:
        """Store a fake credential without any operating-system side effect."""
        self.values[(service_name, username)] = password

    def delete_password(self, service_name: str, username: str) -> None:
        """Delete a fake credential without any operating-system side effect."""
        del self.values[(service_name, username)]


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> FakeKeyring:
    """Replace keyring I/O so tests never access a machine credential store."""
    store = FakeKeyring()
    monkeypatch.setattr(keyring, "get_password", store.get_password)
    monkeypatch.setattr(keyring, "set_password", store.set_password)
    monkeypatch.setattr(keyring, "delete_password", store.delete_password)
    return store
