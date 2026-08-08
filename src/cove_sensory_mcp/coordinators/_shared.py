"""Small injected contracts shared by sensory coordinators."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, Protocol

from cove_sensory_mcp.media.jobs import JobWorkspace
from cove_sensory_mcp.media.types import ResolvedSource
from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.providers.base import ProviderRequest
from cove_sensory_mcp.providers.executor import ExecutedObservation
from cove_sensory_mcp.reports.schemas import ProviderUsage


class SourceResolverProtocol(Protocol):
    async def resolve(self, source: str, job_dir: Path) -> ResolvedSource: ...


class ExecutorProtocol(Protocol):
    async def sense(
        self, modalities: frozenset[Modality], request: ProviderRequest
    ) -> ExecutedObservation: ...


WorkspaceFactory = Callable[[], AbstractAsyncContextManager[JobWorkspace]]
Processor = Callable[..., Any]


def provider_usage(result: ExecutedObservation) -> list[ProviderUsage]:
    if result.used_provider is None or result.used_model is None:
        return []
    return [
        ProviderUsage(
            provider_id=result.used_provider,
            model=result.used_model,
            fallback_used=result.fallback_used,
        )
    ]


def failure_warnings(result: ExecutedObservation) -> list[str]:
    return [code.value for code in result.failures]
