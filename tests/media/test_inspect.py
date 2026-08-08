from __future__ import annotations

from pathlib import Path

import pytest

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.media.inspect import inspect_media
from cove_sensory_mcp.media.runtime import ProcessResult


class FakeRuntime:
    def __init__(self, output: str) -> None:
        self.output = output

    async def run_ffprobe(self, args: list[str], timeout: float = 15) -> ProcessResult:
        return ProcessResult(0, self.output, "")


@pytest.mark.asyncio
async def test_ffprobe_json_becomes_media_metadata(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    metadata = await inspect_media(
        source,
        FakeRuntime(
            '{"format":{"duration":"12.5","format_name":"mp4"},'
            '"streams":[{"codec_type":"video","codec_name":"h264","width":1920,'
            '"height":1080,"avg_frame_rate":"30000/1001"},'
            '{"codec_type":"audio","codec_name":"aac"}]}'
        ),
    )
    assert metadata.duration_seconds == 12.5
    assert metadata.width == 1920
    assert metadata.has_audio is True


@pytest.mark.asyncio
async def test_invalid_probe_json_is_unsupported(tmp_path: Path) -> None:
    source = tmp_path / "bad.bin"
    source.write_bytes(b"x")
    with pytest.raises(SensoryError) as caught:
        await inspect_media(source, FakeRuntime("not-json"))
    assert caught.value.code is ErrorCode.UNSUPPORTED_MEDIA_TYPE
