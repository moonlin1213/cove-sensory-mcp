from __future__ import annotations

from pathlib import Path

import pytest

from cove_sensory_mcp.coordinators.video import VideoCoordinator
from cove_sensory_mcp.media.types import MediaMetadata
from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.providers.base import (
    MediaKind,
    PreparedMedia,
    ProviderMediaLimits,
)

from .conftest import Executor, Resolver, workspace_factory


@pytest.mark.asyncio
async def test_video_preserves_visual_partial_when_audio_fails(
    tmp_path: Path, request_factory
) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"source")
    executor = Executor({Modality.VIDEO_AUDIO})

    async def inspect(path, runtime):
        return MediaMetadata(
            duration_seconds=10,
            width=100,
            height=100,
            mime_type="video/mp4",
            has_audio=True,
        )

    async def prepare(resolved, metadata, media_range, limits, workspace, runtime):
        return PreparedMedia(source, "video/mp4", MediaKind.VIDEO, 10)

    async def extract(resolved, metadata, media_range, workspace, runtime):
        return PreparedMedia(source, "audio/wav", MediaKind.AUDIO, 10)

    result = await VideoCoordinator(
        resolver=Resolver(source),
        workspace_factory=workspace_factory(tmp_path / "jobs"),
        executor=executor,
        visual_limits=ProviderMediaLimits(1000, frozenset({"video/mp4"}), 30, None),
        audio_limits=ProviderMediaLimits(1000, frozenset({"audio/wav"}), 30, None),
        runtime=object(),
        inspect=inspect,
        prepare=prepare,
        extract_audio=extract,
    ).sense(request_factory(str(source)))
    assert result.status == "partial"
    assert result.coverage.visual is True and result.coverage.audio is False


@pytest.mark.asyncio
async def test_joint_video_uses_one_provider_request(
    tmp_path: Path, request_factory
) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"source")
    executor = Executor()

    async def inspect(path, runtime):
        return MediaMetadata(
            duration_seconds=10,
            width=100,
            height=100,
            mime_type="video/mp4",
            has_audio=True,
        )

    async def prepare(resolved, metadata, media_range, limits, workspace, runtime):
        return PreparedMedia(source, "video/mp4", MediaKind.VIDEO, 10)

    async def unexpected(*args):
        raise AssertionError("audio extraction must not run")

    result = await VideoCoordinator(
        resolver=Resolver(source),
        workspace_factory=workspace_factory(tmp_path / "jobs"),
        executor=executor,
        visual_limits=ProviderMediaLimits(1000, frozenset({"video/mp4"}), 30, None),
        audio_limits=ProviderMediaLimits(1000, frozenset({"audio/wav"}), 30, None),
        runtime=object(),
        inspect=inspect,
        prepare=prepare,
        extract_audio=unexpected,
        joint_video=True,
    ).sense(request_factory(str(source)))
    assert executor.calls == [frozenset({Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO})]
    assert result.status == "completed"
