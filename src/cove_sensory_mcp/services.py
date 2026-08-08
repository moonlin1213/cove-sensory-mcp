"""Application dependencies for the Cove Sensory MCP entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cove_sensory_mcp.config.secrets import SecretStore
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from cove_sensory_mcp.verification.assets import SelfTestAssetStore
    from cove_sensory_mcp.verification.verifier import CapabilityVerifier


@dataclass(slots=True)
class AppServices:
    """The application-owned dependencies shared by command and MCP handlers."""

    config_store: ConfigStore
    secret_store: SecretStore

    def capability_verifier(
        self,
        registry: ProviderRegistry,
        *,
        assets: SelfTestAssetStore | None = None,
    ) -> CapabilityVerifier:
        """Compose verification lazily so setup imports do not create a module cycle."""
        from cove_sensory_mcp.verification.assets import SelfTestAssetStore
        from cove_sensory_mcp.verification.verifier import CapabilityVerifier

        active_assets = assets or SelfTestAssetStore.packaged()
        return CapabilityVerifier(
            config_store=self.config_store,
            registry=registry,
            assets=active_assets,
        )
