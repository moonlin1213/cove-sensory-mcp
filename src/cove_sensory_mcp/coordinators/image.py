"""Image resolution, preparation, Provider execution, and report assembly."""

from __future__ import annotations

from typing import Any

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.media.image import prepare_image
from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.providers.base import ProviderMediaLimits, ProviderRequest
from cove_sensory_mcp.reports.schemas import Coverage, ImageSensoryReport

from ._shared import (
    ExecutorProtocol,
    Processor,
    SourceResolverProtocol,
    WorkspaceFactory,
    failure_warnings,
    provider_usage,
)


class ImageCoordinator:
    def __init__(
        self,
        *,
        resolver: SourceResolverProtocol,
        workspace_factory: WorkspaceFactory,
        executor: ExecutorProtocol,
        provider_limits: ProviderMediaLimits,
        prepare: Processor = prepare_image,
    ) -> None:
        self._resolver = resolver
        self._workspace_factory = workspace_factory
        self._executor = executor
        self._limits = provider_limits
        self._prepare = prepare

    async def sense(self, input: Any) -> ImageSensoryReport:
        observations = []
        usage = []
        warnings: list[str] = []
        async with self._workspace_factory() as workspace:
            source = await self._resolver.resolve(input.source, workspace.path)
            prepared = await self._prepare(
                source, input.question, input.detail, self._limits, workspace
            )
            for media in prepared:
                request = ProviderRequest(
                    media=media,
                    requested_modalities=frozenset({Modality.IMAGE}),
                    question=input.question,
                    detail=input.detail,
                    language=input.language,
                )
                executed = await self._executor.sense(
                    frozenset({Modality.IMAGE}), request
                )
                observation = executed.observations.get(Modality.IMAGE)
                if observation is not None:
                    observations.append(observation)
                usage.extend(provider_usage(executed))
                warnings.extend(failure_warnings(executed))
        if not observations:
            raise SensoryError(
                ErrorCode.PROVIDER_UNAVAILABLE,
                "The image Provider returned no usable observation.",
                retryable=True,
            )
        unique_usage = {
            (item.provider_id, item.model, item.fallback_used): item for item in usage
        }
        return ImageSensoryReport(
            status="completed",
            coverage=Coverage(
                visual=True, visual_provider=usage[-1].provider_id if usage else None
            ),
            summary=" ".join(item.summary for item in observations),
            warnings=warnings
            + [warning.code for item in observations for warning in item.warnings],
            providers=list(unique_usage.values()),
            segments=[segment for item in observations for segment in item.segments],
            transcript=[
                segment for item in observations for segment in item.transcript
            ],
        )
