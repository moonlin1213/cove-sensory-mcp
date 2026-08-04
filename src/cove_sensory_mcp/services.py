"""Application dependencies for the Cove Sensory MCP entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from cove_sensory_mcp.config.secrets import SecretStore
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.errors import ErrorCode, SensoryError, error_result
from cove_sensory_mcp.models import Modality


class SelfTestVerifier(Protocol):
    """Verify requested modalities without exposing provider internals to tools."""

    async def verify(self, modalities: list[Modality]) -> dict[str, object]:
        """Return a JSON-compatible public verification result."""


class FoundationSelfTestVerifier:
    """Honest placeholder used until a provider verification implementation exists."""

    async def verify(self, modalities: list[Modality]) -> dict[str, object]:
        """Report that no provider capability has been verified, without I/O."""
        del modalities
        return error_result(
            SensoryError(
                ErrorCode.SETUP_REQUIRED,
                "No verified provider is configured.",
                setup_command="cove-sensory-mcp configure",
            )
        )


@dataclass(slots=True)
class AppServices:
    """The application-owned dependencies shared by command and MCP handlers."""

    config_store: ConfigStore
    secret_store: SecretStore
    self_test_verifier: SelfTestVerifier = field(default_factory=FoundationSelfTestVerifier)
