from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.media.runtime import MediaRuntime


def test_runtime_discovery_prefers_bundled_then_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundled = tmp_path / "ffmpeg-bundled"
    configured = tmp_path / "ffmpeg-configured"
    bundled.write_bytes(b"test fixture")
    configured.write_bytes(b"test fixture")
    bundled.chmod(bundled.stat().st_mode | 0o111)
    configured.chmod(configured.stat().st_mode | 0o111)
    probes: list[list[str]] = []

    def probe(args: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        probes.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("cove_sensory_mcp.media.runtime.subprocess.run", probe)
    runtime = MediaRuntime.discover(bundled, configured, "")
    assert runtime.ffmpeg_path == bundled
    assert probes == [[str(bundled), "-version"]]


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
