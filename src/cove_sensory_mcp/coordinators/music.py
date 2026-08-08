"""Music-specific hearing coordination without local stem analysis."""

from __future__ import annotations

from typing import Any

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.media.audio import prepare_audio
from cove_sensory_mcp.media.inspect import inspect_media
from cove_sensory_mcp.media.types import MediaRange
from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.providers.base import ProviderMediaLimits, ProviderRequest
from cove_sensory_mcp.reports.schemas import Coverage, MusicSensoryReport

from ._shared import (
    ExecutorProtocol,
    Processor,
    SourceResolverProtocol,
    WorkspaceFactory,
    failure_warnings,
    provider_usage,
)


class MusicCoordinator:
    def __init__(
        self,
        *,
        resolver: SourceResolverProtocol,
        workspace_factory: WorkspaceFactory,
        executor: ExecutorProtocol,
        provider_limits: ProviderMediaLimits,
        runtime: object,
        inspect: Processor = inspect_media,
        prepare: Processor = prepare_audio,
    ) -> None:
        self._resolver, self._workspace_factory, self._executor = (
            resolver,
            workspace_factory,
            executor,
        )
        self._limits, self._runtime, self._inspect, self._prepare = (
            provider_limits,
            runtime,
            inspect,
            prepare,
        )

    async def sense(self, input: Any) -> MusicSensoryReport:
        media_range = MediaRange(
            start_seconds=input.start_seconds, end_seconds=input.end_seconds
        )
        async with self._workspace_factory() as workspace:
            source = await self._resolver.resolve(input.source, workspace.path)
            metadata = await self._inspect(source.path, self._runtime)
            media = await self._prepare(
                source, metadata, media_range, self._limits, workspace, self._runtime
            )
            question = (
                "Observe musical structure, instrumentation, rhythm, texture, vocals, dynamics, and key moments. "
                + input.question
            )
            if input.include_lyrics_transcript:
                question += " Include a timestamped lyrics transcript when audible."
            executed = await self._executor.sense(
                frozenset({Modality.MUSIC}),
                ProviderRequest(
                    media,
                    frozenset({Modality.MUSIC}),
                    question,
                    input.detail,
                    input.language,
                    input.start_seconds,
                    input.end_seconds,
                ),
            )
        observation = executed.observations.get(Modality.MUSIC)
        if observation is None:
            raise SensoryError(
                ErrorCode.PROVIDER_UNAVAILABLE,
                "The music Provider returned no usable observation.",
                retryable=True,
            )
        usage = provider_usage(executed)
        return MusicSensoryReport(
            status="completed",
            coverage=Coverage(
                audio=True, audio_provider=usage[0].provider_id if usage else None
            ),
            summary=observation.summary,
            warnings=failure_warnings(executed)
            + [item.code for item in observation.warnings],
            providers=usage,
            requested_start_seconds=input.start_seconds,
            requested_end_seconds=input.end_seconds,
            segments=observation.segments,
            transcript=observation.transcript,
        )
