"""Parallel or verified-joint video eye/ear coordination with partial success."""

from __future__ import annotations

import asyncio
from typing import Any

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.media.inspect import inspect_media
from cove_sensory_mcp.media.types import MediaRange
from cove_sensory_mcp.media.video import extract_video_audio, prepare_video
from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.providers.base import ProviderMediaLimits, ProviderRequest
from cove_sensory_mcp.providers.executor import ExecutedObservation
from cove_sensory_mcp.reports.merge import merge_video_observations
from cove_sensory_mcp.reports.schemas import Coverage, ProviderUsage, VideoSensoryReport

from ._shared import (
    ExecutorProtocol,
    Processor,
    SourceResolverProtocol,
    WorkspaceFactory,
    failure_warnings,
    provider_usage,
)


class VideoCoordinator:
    def __init__(
        self,
        *,
        resolver: SourceResolverProtocol,
        workspace_factory: WorkspaceFactory,
        executor: ExecutorProtocol,
        visual_limits: ProviderMediaLimits,
        audio_limits: ProviderMediaLimits,
        runtime: object,
        inspect: Processor = inspect_media,
        prepare: Processor = prepare_video,
        extract_audio: Processor = extract_video_audio,
        joint_video: bool = False,
        max_unbounded_duration: float = 600,
    ) -> None:
        self._resolver, self._workspace_factory, self._executor = (
            resolver,
            workspace_factory,
            executor,
        )
        self._visual_limits, self._audio_limits, self._runtime = (
            visual_limits,
            audio_limits,
            runtime,
        )
        self._inspect, self._prepare, self._extract = inspect, prepare, extract_audio
        self._joint, self._max_unbounded = joint_video, max_unbounded_duration

    async def _execute(
        self, modalities: frozenset[Modality], request: ProviderRequest
    ) -> ExecutedObservation | None:
        try:
            return await self._executor.sense(modalities, request)
        except SensoryError:
            return None

    async def sense(self, input: Any) -> VideoSensoryReport:
        media_range = MediaRange(
            start_seconds=input.start_seconds, end_seconds=input.end_seconds
        )
        warnings: list[str] = []
        async with self._workspace_factory() as workspace:
            source = await self._resolver.resolve(input.source, workspace.path)
            metadata = await self._inspect(source.path, self._runtime)
            if (
                metadata.duration_seconds
                and metadata.duration_seconds > self._max_unbounded
                and input.start_seconds is None
                and input.end_seconds is None
            ):
                raise SensoryError(
                    ErrorCode.LONG_MEDIA_CONFIRMATION_REQUIRED,
                    "Choose a time range for this long video.",
                )
            video = await self._prepare(
                source,
                metadata,
                media_range,
                self._visual_limits,
                workspace,
                self._runtime,
            )
            if self._joint and input.include_audio and metadata.has_audio:
                modalities = frozenset({Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO})
                joint = await self._execute(
                    modalities,
                    ProviderRequest(
                        video,
                        modalities,
                        input.question,
                        input.detail,
                        input.language,
                        input.start_seconds,
                        input.end_seconds,
                    ),
                )
                visual_result = audio_result = joint
            else:
                audio_media = (
                    await self._extract(
                        source, metadata, media_range, workspace, self._runtime
                    )
                    if input.include_audio
                    else None
                )
                visual_request = ProviderRequest(
                    video,
                    frozenset({Modality.VIDEO_VISUAL}),
                    input.question,
                    input.detail,
                    input.language,
                    input.start_seconds,
                    input.end_seconds,
                )
                async with asyncio.TaskGroup() as group:
                    visual_task = group.create_task(
                        self._execute(
                            frozenset({Modality.VIDEO_VISUAL}), visual_request
                        )
                    )
                    audio_task = None
                    if audio_media is not None:
                        audio_request = ProviderRequest(
                            audio_media,
                            frozenset({Modality.VIDEO_AUDIO}),
                            input.question,
                            input.detail,
                            input.language,
                            input.start_seconds,
                            input.end_seconds,
                        )
                        audio_task = group.create_task(
                            self._execute(
                                frozenset({Modality.VIDEO_AUDIO}), audio_request
                            )
                        )
                visual_result = visual_task.result()
                audio_result = audio_task.result() if audio_task is not None else None
                if input.include_audio and audio_media is None:
                    warnings.append("NO_AUDIO_TRACK")
        visual = (
            visual_result.observations.get(Modality.VIDEO_VISUAL)
            if visual_result
            else None
        )
        audio = (
            audio_result.observations.get(Modality.VIDEO_AUDIO)
            if audio_result
            else None
        )
        if visual is None and audio is None:
            raise SensoryError(
                ErrorCode.PROVIDER_UNAVAILABLE,
                "Both video perception channels failed.",
                retryable=True,
            )
        report = merge_video_observations(visual, audio, metadata.duration_seconds or 0)
        usage: list[ProviderUsage] = []
        for result in (visual_result, audio_result):
            if result is not None:
                usage.extend(provider_usage(result))
                warnings.extend(failure_warnings(result))
        unique = {
            (item.provider_id, item.model, item.fallback_used): item for item in usage
        }
        visual_provider = (
            provider_usage(visual_result)[0].provider_id
            if visual_result and provider_usage(visual_result)
            else None
        )
        audio_provider = (
            provider_usage(audio_result)[0].provider_id
            if audio_result and provider_usage(audio_result)
            else None
        )
        return report.model_copy(
            update={
                "coverage": Coverage(
                    visual=visual is not None,
                    audio=audio is not None,
                    visual_provider=visual_provider,
                    audio_provider=audio_provider,
                ),
                "providers": list(unique.values()),
                "warnings": report.warnings + warnings,
                "requested_start_seconds": input.start_seconds,
                "requested_end_seconds": input.end_seconds,
            }
        )
