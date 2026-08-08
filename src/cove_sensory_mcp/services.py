"""Application dependencies for the Cove Sensory MCP entry points."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from cove_sensory_mcp.config.secrets import SecretStore
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from cove_sensory_mcp.reports.schemas import SensoryToolResult
    from cove_sensory_mcp.verification.assets import SelfTestAssetStore
    from cove_sensory_mcp.verification.verifier import CapabilityVerifier


@dataclass
class AppServices:
    """The application-owned dependencies shared by command and MCP handlers."""

    config_store: ConfigStore
    secret_store: SecretStore
    sensing_coordinators: InitVar[Mapping[str, Any] | None] = None

    def __post_init__(self, sensing_coordinators: Mapping[str, Any] | None) -> None:
        self._sensing_coordinators = sensing_coordinators

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

    async def sense(self, kind: str, input: Any) -> SensoryToolResult:
        """Run one injected or locally composed coordinator and close owned adapters."""
        if self._sensing_coordinators is not None:
            coordinator = self._sensing_coordinators.get(kind)
            if coordinator is None:
                from cove_sensory_mcp.errors import ErrorCode, SensoryError

                raise SensoryError(
                    ErrorCode.SETUP_REQUIRED,
                    "The sensory coordinator is not configured.",
                )
            return await coordinator.sense(input)

        coordinator, adapters = self._compose_sensing(kind, input)
        try:
            return await coordinator.sense(input)
        finally:
            for adapter in reversed(adapters):
                close = getattr(adapter, "aclose", None)
                if close is not None:
                    await close()

    def _compose_sensing(self, kind: str, input: Any) -> tuple[Any, list[Any]]:
        """Build request-scoped media and Provider dependencies from verified config."""
        from cove_sensory_mcp.coordinators import (
            AudioCoordinator,
            ImageCoordinator,
            MusicCoordinator,
            VideoCoordinator,
        )
        from cove_sensory_mcp.errors import ErrorCode, SensoryError
        from cove_sensory_mcp.media.jobs import JobWorkspace
        from cove_sensory_mcp.media.resolver import SourceResolver
        from cove_sensory_mcp.media.runtime import MediaRuntime
        from cove_sensory_mcp.models import Modality
        from cove_sensory_mcp.providers.base import ProviderMediaLimits, SensoryProvider
        from cove_sensory_mcp.providers.executor import ProviderExecutor
        from cove_sensory_mcp.providers.gemini import GeminiProvider
        from cove_sensory_mcp.providers.minimax_m3 import (
            MINIMAX_CN_BASE_URL,
            MiniMaxM3Provider,
            MiniMaxRegion,
        )
        from cove_sensory_mcp.providers.openai_compatible import (
            OpenAICompatibleProvider,
        )
        from cove_sensory_mcp.providers.registry import ProviderRegistry
        from cove_sensory_mcp.providers.router import ProviderRouter

        config = self.config_store.load()
        router = ProviderRouter(config)
        expected_overrides = {
            "image": ((Modality.IMAGE, getattr(input, "provider", None)),),
            "audio": (),
            "music": (),
            "video": (
                (Modality.VIDEO_VISUAL, getattr(input, "visual_provider", None)),
                (Modality.VIDEO_AUDIO, getattr(input, "audio_provider", None)),
            ),
        }
        if kind not in expected_overrides:
            raise SensoryError(
                ErrorCode.CONFIG_INVALID, "The sensory modality is invalid."
            )
        for modality, override in expected_overrides[kind]:
            if (
                override is not None
                and router.candidates(modality)[0].provider_id != override
            ):
                raise SensoryError(
                    ErrorCode.CAPABILITY_NOT_CONFIGURED,
                    "The requested Provider override is not the verified route.",
                )

        adapters: list[Any] = []
        provider_map: dict[str, Any] = {}
        for provider_id, provider in config.providers.items():
            adapter: SensoryProvider
            if provider.adapter == "gemini":
                adapter = GeminiProvider(
                    provider_id=provider_id,
                    config=provider,
                    secret_store=self.secret_store,
                )
            elif provider.adapter == "minimax-m3":
                region = (
                    MiniMaxRegion.CN
                    if provider.base_url == MINIMAX_CN_BASE_URL
                    else MiniMaxRegion.GLOBAL
                )
                adapter = MiniMaxM3Provider(
                    provider_id=provider_id,
                    config=provider,
                    secret_store=self.secret_store,
                    region=region,
                )
            elif provider.adapter == "openai-compatible":
                adapter = OpenAICompatibleProvider(
                    provider_id=provider_id,
                    config=provider,
                    secret_store=self.secret_store,
                )
            else:
                raise SensoryError(
                    ErrorCode.CONFIG_INVALID, "The provider adapter is invalid."
                )
            provider_map[provider_id] = adapter
            adapters.append(adapter)
        executor = ProviderExecutor(
            router=router, registry=ProviderRegistry(provider_map)
        )
        resolver = SourceResolver(config.allowed_media_roots)
        workspace_factory = lambda: JobWorkspace.create(
            self.config_store.jobs_dir, lambda: datetime.now(UTC)
        )
        image_limits = ProviderMediaLimits(
            20 * 1024 * 1024,
            frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"}),
            None,
            4096,
        )
        audio_limits = ProviderMediaLimits(
            100 * 1024 * 1024,
            frozenset({"audio/wav", "audio/mpeg", "audio/flac", "audio/mp4"}),
            3600,
            None,
        )
        video_limits = ProviderMediaLimits(
            500 * 1024 * 1024,
            frozenset({"video/mp4", "video/webm", "video/quicktime"}),
            3600,
            None,
        )
        if kind == "image":
            return ImageCoordinator(
                resolver=resolver,
                workspace_factory=workspace_factory,
                executor=executor,
                provider_limits=image_limits,
            ), adapters
        runtime = MediaRuntime.discover(None, None, os.environ.get("PATH"))
        if kind == "audio":
            return AudioCoordinator(
                resolver=resolver,
                workspace_factory=workspace_factory,
                executor=executor,
                provider_limits=audio_limits,
                runtime=runtime,
            ), adapters
        if kind == "music":
            return MusicCoordinator(
                resolver=resolver,
                workspace_factory=workspace_factory,
                executor=executor,
                provider_limits=audio_limits,
                runtime=runtime,
            ), adapters
        joint = (
            router.joint_candidate(
                frozenset({Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO})
            )
            is not None
        )
        return VideoCoordinator(
            resolver=resolver,
            workspace_factory=workspace_factory,
            executor=executor,
            visual_limits=video_limits,
            audio_limits=audio_limits,
            runtime=runtime,
            joint_video=joint,
        ), adapters
