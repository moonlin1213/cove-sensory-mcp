"""Bounded audio clipping and normalization for hearing Providers."""

from __future__ import annotations

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.providers.base import (
    MediaKind,
    PreparedMedia,
    ProviderMediaLimits,
)

from .types import MediaMetadata, MediaRange, ResolvedSource
from .video import ArtifactWorkspace, FfmpegRuntime, _range_args, _validate_range


async def prepare_audio(
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
    if (
        provider_limits.max_duration_seconds is not None
        and selected_duration > provider_limits.max_duration_seconds
    ):
        raise SensoryError(
            ErrorCode.MEDIA_TOO_LARGE,
            "The selected audio range exceeds provider limits.",
        )
    if (
        media_range.start_seconds is None
        and media_range.end_seconds is None
        and metadata.mime_type in provider_limits.accepted_mime_types
        and source.original_size <= provider_limits.max_bytes
    ):
        return PreparedMedia(
            source.path, metadata.mime_type, MediaKind.AUDIO, metadata.duration_seconds
        )
    output = workspace.new_artifact("audio", ".wav")
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
    if not output.is_file() or output.stat().st_size > provider_limits.max_bytes:
        raise SensoryError(
            ErrorCode.MEDIA_TOO_LARGE, "The prepared audio exceeds provider limits."
        )
    return PreparedMedia(output, "audio/wav", MediaKind.AUDIO, selected_duration)
