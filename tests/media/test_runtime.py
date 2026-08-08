from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.media.runtime import MediaRuntime


def _executable(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return path


def _discovery_executable(path: Path) -> Path:
    if os.name == "nt":
        executable = path.with_suffix(".exe")
        shutil.copy2(sys.executable, executable)
        return executable
    return _executable(path, "echo ok\n")


def test_runtime_discovery_prefers_bundled_then_configured(tmp_path: Path) -> None:
    bundled = _discovery_executable(tmp_path / "ffmpeg-bundled")
    configured = _discovery_executable(tmp_path / "ffmpeg-configured")
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
        ["-c", "import sys; print(sys.argv[1], end='')", "路径 with spaces"],
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
