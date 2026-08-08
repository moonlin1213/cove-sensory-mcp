"""Cross-platform FFmpeg discovery and cancellation-safe subprocess execution."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cove_sensory_mcp.errors import ErrorCode, SensoryError

_DIAGNOSTIC_LIMIT = 8_192


@dataclass(frozen=True, slots=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str


class MediaRuntime:
    def __init__(self, ffmpeg_path: Path, ffprobe_path: Path) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

    @classmethod
    def discover(
        cls,
        bundled: Path | None,
        configured: Path | None,
        path_env: str | None,
    ) -> MediaRuntime:
        candidates = [bundled, configured]
        found = shutil.which("ffmpeg", path=path_env) if path_env else None
        candidates.append(Path(found) if found else None)
        for candidate in candidates:
            if candidate is None or not candidate.is_file():
                continue
            if os.name != "nt" and not os.access(candidate, os.X_OK):
                continue
            try:
                probe = subprocess.run(
                    [str(candidate), "-version"],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if probe.returncode != 0:
                continue
            ffprobe = candidate.with_name(candidate.name.replace("ffmpeg", "ffprobe"))
            if not ffprobe.is_file():
                ffprobe = candidate
            return cls(candidate, ffprobe)
        raise SensoryError(
            ErrorCode.MEDIA_RUNTIME_REQUIRED,
            "A working FFmpeg runtime is required for this media.",
            setup_command="cove-sensory-mcp doctor",
        )

    async def run_ffmpeg(
        self,
        args: list[str],
        timeout: float,
        cancel_event: asyncio.Event | None = None,
    ) -> ProcessResult:
        return await self._run(self.ffmpeg_path, args, timeout, cancel_event)

    async def run_ffprobe(
        self,
        args: list[str],
        timeout: float = 15,
    ) -> ProcessResult:
        return await self._run(self.ffprobe_path, args, timeout, None)

    async def _run(
        self,
        executable: Path,
        args: list[str],
        timeout: float,
        cancel_event: asyncio.Event | None,
    ) -> ProcessResult:
        process = await asyncio.create_subprocess_exec(
            str(executable),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        communicate = asyncio.create_task(process.communicate())
        cancellation = (
            asyncio.create_task(cancel_event.wait())
            if cancel_event is not None
            else None
        )
        try:
            waiters: set[asyncio.Task[object]] = {communicate}  # type: ignore[arg-type]
            if cancellation is not None:
                waiters.add(cancellation)  # type: ignore[arg-type]
            done, _ = await asyncio.wait(
                waiters,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if communicate not in done:
                await self._stop(process)
                communicate.cancel()
                raise SensoryError(
                    ErrorCode.PROVIDER_TIMEOUT,
                    "Media processing timed out or was cancelled.",
                    retryable=True,
                )
            stdout, stderr = await communicate
        except asyncio.CancelledError:
            await self._stop(process)
            communicate.cancel()
            raise
        finally:
            if cancellation is not None:
                cancellation.cancel()
        return ProcessResult(
            process.returncode or 0,
            stdout.decode("utf-8", errors="replace")[:_DIAGNOSTIC_LIMIT],
            stderr.decode("utf-8", errors="replace")[:_DIAGNOSTIC_LIMIT],
        )

    @staticmethod
    async def _stop(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=1)
        except TimeoutError:
            process.kill()
            await process.wait()
