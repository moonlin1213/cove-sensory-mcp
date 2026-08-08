from __future__ import annotations

from pathlib import Path

import pytest

from cove_sensory_mcp.media.types import MediaMetadata, MediaRange, ResolvedSource
from cove_sensory_mcp.media.video import extract_video_audio, prepare_video
from cove_sensory_mcp.providers.base import ProviderMediaLimits


class Workspace:
    def __init__(self, path: Path) -> None:
        self.path = path

    def new_artifact(self, name: str, suffix: str) -> Path:
        return self.path / f"{name}{suffix}"


class Runtime:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def run_ffmpeg(self, args: list[str], timeout: float, cancel_event=None):
        self.calls.append(args)
        Path(args[-1]).write_bytes(b"prepared")


def _source(tmp_path: Path) -> ResolvedSource:
    path = tmp_path / "source.mp4"
    path.write_bytes(b"source")
    return ResolvedSource(
        path=path,
        source_kind="local",
        display_name="source.mp4",
        cleanup_required=False,
        original_size=6,
    )


@pytest.mark.asyncio
async def test_video_clip_and_audio_extraction_use_workspace_outputs(
    tmp_path: Path,
) -> None:
    runtime = Runtime()
    metadata = MediaMetadata(
        duration_seconds=100,
        width=1920,
        height=1080,
        frame_rate=30,
        mime_type="video/mp4",
        has_audio=True,
    )
    limits = ProviderMediaLimits(1000, frozenset({"video/mp4"}), 60, None)
    prepared = await prepare_video(
        _source(tmp_path),
        metadata,
        MediaRange(start_seconds=10, end_seconds=20),
        limits,
        Workspace(tmp_path),
        runtime,
    )
    audio = await extract_video_audio(
        _source(tmp_path),
        metadata,
        MediaRange(start_seconds=10, end_seconds=20),
        Workspace(tmp_path),
        runtime,
    )
    assert prepared.path.parent == tmp_path
    assert audio is not None and audio.path.parent == tmp_path
    assert any("-ss" in call and "-to" in call for call in runtime.calls)
    assert any("-ac" in call and "1" in call for call in runtime.calls)


@pytest.mark.asyncio
async def test_video_without_audio_returns_none(tmp_path: Path) -> None:
    metadata = MediaMetadata(
        duration_seconds=10,
        width=100,
        height=100,
        frame_rate=24,
        mime_type="video/mp4",
        has_audio=False,
    )
    assert (
        await extract_video_audio(
            _source(tmp_path), metadata, MediaRange(), Workspace(tmp_path), Runtime()
        )
        is None
    )
