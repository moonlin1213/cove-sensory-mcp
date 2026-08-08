"""FFprobe-backed media metadata inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from cove_sensory_mcp.errors import ErrorCode, SensoryError

from .runtime import ProcessResult
from .types import MediaMetadata


class ProbeRuntime(Protocol):
    async def run_ffprobe(
        self, args: list[str], timeout: float = 15
    ) -> ProcessResult: ...


def _frame_rate(value: object) -> float | None:
    try:
        numerator, denominator = str(value).split("/", 1)
        return float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        return None


async def inspect_media(path: Path, runtime: ProbeRuntime) -> MediaMetadata:
    result = await runtime.run_ffprobe(
        ["-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]
    )
    try:
        if result.returncode != 0:
            raise ValueError("probe failed")
        payload = json.loads(result.stdout)
        streams = payload["streams"]
        media_format = payload.get("format", {})
        video = next(
            (item for item in streams if item.get("codec_type") == "video"), None
        )
        audio = next(
            (item for item in streams if item.get("codec_type") == "audio"), None
        )
        duration = (
            float(media_format["duration"])
            if media_format.get("duration") is not None
            else None
        )
        if video is not None:
            mime = (
                "video/mp4"
                if "mp4" in media_format.get("format_name", "")
                else "video/unknown"
            )
        elif audio is not None:
            name = media_format.get("format_name", "")
            mime = (
                "audio/wav"
                if "wav" in name
                else ("audio/mpeg" if "mp3" in name else f"audio/{name.split(',')[0]}")
            )
        else:
            raise ValueError("no media streams")
        return MediaMetadata(
            duration_seconds=duration,
            width=video.get("width") if video else None,
            height=video.get("height") if video else None,
            frame_rate=_frame_rate(video.get("avg_frame_rate")) if video else None,
            mime_type=mime,
            has_audio=audio is not None,
            video_codec=video.get("codec_name") if video else None,
            audio_codec=audio.get("codec_name") if audio else None,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise SensoryError(
            ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            "The media metadata is invalid or unsupported.",
        ) from None
