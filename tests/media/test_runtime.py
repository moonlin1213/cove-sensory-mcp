from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.media.runtime import MediaRuntime


def _executable(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return path


def test_runtime_discovery_prefers_bundled_then_configured(tmp_path: Path) -> None:
    if os.name == "nt":
        bundled = Path(sys.executable)
        configured = tmp_path / "unused-configured.exe"
        configured.write_bytes(b"not reached")
    else:
        bundled = _executable(tmp_path / "ffmpeg-bundled", "echo ok\n")
        configured = _executable(tmp_path / "ffmpeg-configured", "echo ok\n")
    runtime = MediaRuntime.discover(bundled, configured, "")
    assert runtime.ffmpeg_path == bundled


def test_runtime_discovery_maps_missing_to_stable_error(tmp_path: Path) -> None:
    with pytest.raises(SensoryError) as caught:
        MediaRuntime.discover(None, tmp_path / "missing", "")
    assert caught.value.code is ErrorCode.MEDIA_RUNTIME_REQUIRED


@pytest.mark.asyncio
async def test_process_arguments_preserve_spaces_and_non_ascii(tmp_path: Path) -> None:
    binary = Path(sys.executable)
    runtime = MediaRuntime(binary, binary)
    result = await runtime.run_ffmpeg(
        [
            "-c",
            "import sys; sys.stdout.buffer.write(sys.argv[1].encode('utf-8'))",
            "路径 with spaces",
        ],
        timeout=2,
    )
    assert result.returncode == 0
    assert result.stdout == "路径 with spaces"


@pytest.mark.asyncio
async def test_timeout_terminates_child(tmp_path: Path) -> None:
    binary = Path(sys.executable)
    runtime = MediaRuntime(binary, binary)
    with pytest.raises(SensoryError) as caught:
        await runtime.run_ffmpeg(
            ["-c", "import time; time.sleep(30)"], timeout=0.05
        )
    assert caught.value.code is ErrorCode.PROVIDER_TIMEOUT
