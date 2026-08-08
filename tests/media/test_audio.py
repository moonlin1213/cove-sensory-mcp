from __future__ import annotations

from pathlib import Path

import pytest

from cove_sensory_mcp.media.audio import prepare_audio
from cove_sensory_mcp.media.types import MediaMetadata, MediaRange, ResolvedSource
from cove_sensory_mcp.providers.base import ProviderMediaLimits


class Workspace:
    def __init__(self, path: Path) -> None:
        self.path = path

    def new_artifact(self, name: str, suffix: str) -> Path:
        return self.path / f"{name}{suffix}"


class Runtime:
    def __init__(self) -> None:
        self.args: list[str] = []

    async def run_ffmpeg(self, args: list[str], timeout: float, cancel_event=None):
        self.args = args
        Path(args[-1]).write_bytes(b"audio")


@pytest.mark.asyncio
async def test_audio_range_is_normalized_without_mutating_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.flac"
    path.write_bytes(b"original")
    source = ResolvedSource(
        path=path,
        source_kind="local",
        display_name="source.flac",
        cleanup_required=False,
        original_size=8,
    )
    metadata = MediaMetadata(
        duration_seconds=30, mime_type="audio/flac", has_audio=True
    )
    runtime = Runtime()
    result = await prepare_audio(
        source,
        metadata,
        MediaRange(start_seconds=2, end_seconds=8),
        ProviderMediaLimits(1000, frozenset({"audio/wav"}), 10, None),
        Workspace(tmp_path),
        runtime,
    )
    assert result.mime_type == "audio/wav"
    assert path.read_bytes() == b"original"
    assert "-ss" in runtime.args and "-to" in runtime.args
