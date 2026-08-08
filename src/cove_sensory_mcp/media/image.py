"""Bounded EXIF-aware image preparation without source mutation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageOps, UnidentifiedImageError

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import DetailLevel
from cove_sensory_mcp.providers.base import (
    MediaKind,
    PreparedMedia,
    ProviderMediaLimits,
)

from .types import ResolvedSource

_MAX_PIXELS = 80_000_000
_OCR_MARKERS = ("ocr", "text", "文字", "字幕", "document")


class ArtifactWorkspace(Protocol):
    path: Path

    def new_artifact(self, name: str, suffix: str) -> Path: ...


def _output_format(
    limits: ProviderMediaLimits, has_alpha: bool
) -> tuple[str, str, str]:
    if has_alpha and "image/png" in limits.accepted_mime_types:
        return "PNG", ".png", "image/png"
    if "image/jpeg" in limits.accepted_mime_types:
        return "JPEG", ".jpg", "image/jpeg"
    if "image/png" in limits.accepted_mime_types:
        return "PNG", ".png", "image/png"
    raise SensoryError(
        ErrorCode.UNSUPPORTED_MEDIA_TYPE,
        "The provider accepts no prepared image format.",
    )


def _fit(image: Image.Image, edge: int | None) -> Image.Image:
    if edge is None or max(image.size) <= edge:
        return image.copy()
    result = image.copy()
    result.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    return result


def _save(
    image: Image.Image,
    workspace: ArtifactWorkspace,
    limits: ProviderMediaLimits,
    index: int,
) -> PreparedMedia:
    has_alpha = image.mode in {"RGBA", "LA"} or "transparency" in image.info
    file_format, suffix, mime = _output_format(limits, has_alpha)
    output = workspace.new_artifact("image", suffix)
    prepared = image
    if file_format == "JPEG":
        if has_alpha:
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (245, 245, 245))
            background.paste(rgba, mask=rgba.getchannel("A"))
            prepared = background
        else:
            prepared = image.convert("RGB")
        for quality in (90, 82, 74, 66, 58):
            prepared.save(output, format="JPEG", quality=quality, optimize=True)
            if output.stat().st_size <= limits.max_bytes:
                break
    else:
        prepared.save(output, format="PNG", optimize=True)
    if output.stat().st_size > limits.max_bytes:
        output.unlink(missing_ok=True)
        raise SensoryError(
            ErrorCode.MEDIA_TOO_LARGE, "The prepared image exceeds provider limits."
        )
    return PreparedMedia(output, mime, MediaKind.IMAGE, None, False, index)


async def prepare_image(
    source: ResolvedSource,
    focus: str,
    detail: DetailLevel,
    provider_limits: ProviderMediaLimits,
    workspace: ArtifactWorkspace,
) -> list[PreparedMedia]:
    """Correct orientation, bound pixels/bytes, and tile OCR-focused long images."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source.path) as opened:
                animated = bool(getattr(opened, "is_animated", False))
                width, height = opened.size
                if width * height > _MAX_PIXELS:
                    raise SensoryError(
                        ErrorCode.MEDIA_TOO_LARGE,
                        "The image has too many decoded pixels.",
                    )
                if animated and (
                    opened.get_format_mimetype() in provider_limits.accepted_mime_types
                ):
                    return [
                        PreparedMedia(
                            source.path,
                            opened.get_format_mimetype() or "image/gif",
                            MediaKind.IMAGE,
                            None,
                            True,
                            0,
                        )
                    ]
                opened.seek(0)
                image = ImageOps.exif_transpose(opened).copy()
    except SensoryError:
        raise
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        raise SensoryError(
            ErrorCode.UNSUPPORTED_MEDIA_TYPE, "The image is invalid or unsupported."
        ) from None

    wants_tiles = (
        image.height > image.width * 3
        and detail is DetailLevel.DETAILED
        and any(marker in focus.lower() for marker in _OCR_MARKERS)
    )
    images: list[Image.Image]
    edge = provider_limits.max_image_edge
    if wants_tiles and edge is not None:
        tile_height = max(edge, image.width)
        overlap = max(1, tile_height // 10)
        images = []
        top = 0
        while top < image.height:
            bottom = min(image.height, top + tile_height)
            images.append(image.crop((0, top, image.width, bottom)))
            if bottom == image.height:
                break
            top = bottom - overlap
    else:
        images = [image]
    return [
        _save(_fit(item, edge), workspace, provider_limits, index)
        for index, item in enumerate(images)
    ]
