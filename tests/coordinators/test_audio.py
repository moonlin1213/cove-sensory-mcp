from __future__ import annotations

from pathlib import Path

import pytest

from cove_sensory_mcp.coordinators.audio import AudioCoordinator
from cove_sensory_mcp.media.types import MediaMetadata
from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.providers.base import (
    MediaKind,
    PreparedMedia,
    ProviderMediaLimits,
)

from .conftest import Executor, Resolver, workspace_factory


@pytest.mark.asyncio
async def test_audio_coordinator_uses_audio_modality_and_transcript_prompt(
    tmp_path: Path, request_factory
) -> None:
    source = tmp_path / "sound.wav"
    source.write_bytes(b"source")
    executor = Executor()

    async def inspect(path, runtime):
        return MediaMetadata(duration_seconds=10, mime_type="audio/wav", has_audio=True)

    async def prepare(resolved, metadata, media_range, limits, workspace, runtime):
        return PreparedMedia(source, "audio/wav", MediaKind.AUDIO, 10)

    result = await AudioCoordinator(
        resolver=Resolver(source),
        workspace_factory=workspace_factory(tmp_path / "jobs"),
        executor=executor,
        provider_limits=ProviderMediaLimits(1000, frozenset({"audio/wav"}), 30, None),
        runtime=object(),
        inspect=inspect,
        prepare=prepare,
    ).sense(request_factory(str(source)))
    assert executor.calls[0] == frozenset({Modality.AUDIO})
    assert result.modality == "audio"
