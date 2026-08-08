"""Provider-bounded native video preparation and audio extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.providers.base import (
    MediaKind,
    PreparedMedia,
    ProviderMediaLimits,
)

from .types import MediaMetadata, MediaRange, ResolvedSource


class ArtifactWorkspace(Protocol):
    path: Path

    def new_artifact(self, name: str, suffix: str) -> Path: ...


class FfmpegRuntime(Protocol):
    async def run_ffmpeg(
        self, args: list[str], timeout: float, cancel_event: object | None = None
    ) -> object: ...


def _validate_range(media_range: MediaRange, duration: float | None) -> None:
    if duration is None:
        return
    if media_range.start_seconds is not None and media_range.start_seconds >= duration:
        raise SensoryError(
            ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            "The requested range is outside the media.",
        )
    if media_range.end_seconds is not None and media_range.end_seconds > duration:
        raise SensoryError(
            ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            "The requested range is outside the media.",
        )


def _range_args(media_range: MediaRange) -> list[str]:
    args: list[str] = []
    if media_range.start_seconds is not None:
        args += ["-ss", str(media_range.start_seconds)]
    if media_range.end_seconds is not None:
        args += ["-to", str(media_range.end_seconds)]
    return args


async def prepare_video(
    source: ResolvedSource,
    metadata: MediaMetadata,
    media_range: MediaRange,
    provider_limits: ProviderMediaLimits,
    workspace: ArtifactWorkspace,
    runtime: FfmpegRuntime,
) -> PreparedMedia:
    _validate_range(media_range, metadata.duration_seconds)
    selected_duration = (media_range.end_seconds or metadata.duration_seconds or 0) - (
        media_range.start_seconds or 0
    )
    needs_processing = (
        media_range.start_seconds is not None
        or media_range.end_seconds is not None
        or metadata.mime_type not in provider_limits.accepted_mime_types
        or source.original_size > provider_limits.max_bytes
        or (
            provider_limits.max_duration_seconds is not None
            and selected_duration > provider_limits.max_duration_seconds
        )
    )
    if not needs_processing:
        return PreparedMedia(
            source.path, metadata.mime_type, MediaKind.VIDEO, metadata.duration_seconds
        )
    if (
        provider_limits.max_duration_seconds is not None
        and selected_duration > provider_limits.max_duration_seconds
    ):
        raise SensoryError(
            ErrorCode.MEDIA_TOO_LARGE,
            "The selected video range exceeds provider limits.",
        )
    output = workspace.new_artifact("video", ".mp4")
    args = [
        *_range_args(media_range),
        "-i",
        str(source.path),
        "-map_metadata",
        "-1",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-y",
        str(output),
    ]
    await runtime.run_ffmpeg(args, timeout=120)
    if not output.is_file() or output.stat().st_size > provider_limits.max_bytes:
        raise SensoryError(
            ErrorCode.MEDIA_TOO_LARGE, "The prepared video exceeds provider limits."
        )
    return PreparedMedia(output, "video/mp4", MediaKind.VIDEO, selected_duration)


async def extract_video_audio(
    source: ResolvedSource,
    metadata: MediaMetadata,
    media_range: MediaRange,
    workspace: ArtifactWorkspace,
    runtime: FfmpegRuntime,
) -> PreparedMedia | None:
    _validate_range(media_range, metadata.duration_seconds)
    if not metadata.has_audio:
        return None
    output = workspace.new_artifact("video_audio", ".wav")
    args = [
        *_range_args(media_range),
        "-i",
        str(source.path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-map_metadata",
        "-1",
        "-y",
        str(output),
    ]
    await runtime.run_ffmpeg(args, timeout=120)
    duration = (media_range.end_seconds or metadata.duration_seconds or 0) - (
        media_range.start_seconds or 0
    )
    return PreparedMedia(output, "audio/wav", MediaKind.AUDIO, duration)
