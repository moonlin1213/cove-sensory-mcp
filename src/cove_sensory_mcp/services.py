"""Application dependencies for the Cove Sensory MCP entry points."""

from __future__ import annotations

from dataclasses import dataclass

from cove_sensory_mcp.config.secrets import SecretStore
from cove_sensory_mcp.config.store import ConfigStore


@dataclass(slots=True)
class AppServices:
    """The application-owned dependencies shared by command and MCP handlers."""

    config_store: ConfigStore
    secret_store: SecretStore
