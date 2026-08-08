"""Local, non-private media fixtures used only for capability self-tests."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.providers.base import MediaKind, PreparedMedia

_MISSING_ASSET_MESSAGE = "A required self-test media asset is unavailable."

_EXPECTED_MEDIA: dict[Modality, tuple[MediaKind, str]] = {
    Modality.IMAGE: (MediaKind.IMAGE, "image/"),
    Modality.VIDEO_VISUAL: (MediaKind.VIDEO, "video/"),
    Modality.VIDEO_AUDIO: (MediaKind.VIDEO, "video/"),
    Modality.AUDIO: (MediaKind.AUDIO, "audio/"),
    Modality.MUSIC: (MediaKind.AUDIO, "audio/"),
}


class SelfTestAssetStore:
    """Resolve injected or packaged tiny fixtures without exposing their paths."""

    def __init__(
        self,
        assets: Mapping[Modality, PreparedMedia],
        *,
        trusted_root: Path,
    ) -> None:
        self._assets = dict(assets)
        self._trusted_root = trusted_root.resolve(strict=False)

    @classmethod
    def packaged(cls) -> SelfTestAssetStore:
        """Describe the stable package locations supplied by the media/release milestone."""
        root = Path(__file__).resolve().parent / "self_test_media"
        video = root / "motion.mp4"
        audio = root / "tones.wav"
        return cls(
            {
                Modality.IMAGE: PreparedMedia(
                    root / "shape.png",
                    "image/png",
                    MediaKind.IMAGE,
                    None,
                ),
                Modality.VIDEO_VISUAL: PreparedMedia(
                    video,
                    "video/mp4",
                    MediaKind.VIDEO,
                    2.0,
                ),
                Modality.VIDEO_AUDIO: PreparedMedia(
                    video,
                    "video/mp4",
                    MediaKind.VIDEO,
                    2.0,
                ),
                Modality.AUDIO: PreparedMedia(
                    audio,
                    "audio/wav",
                    MediaKind.AUDIO,
                    2.0,
                ),
                Modality.MUSIC: PreparedMedia(
                    root / "scale.wav",
                    "audio/wav",
                    MediaKind.AUDIO,
                    2.0,
                ),
            },
            trusted_root=root,
        )

    def get(self, modality: Modality) -> PreparedMedia:
        """Return one existing prepared fixture or a stable path-free error."""
        try:
            media = self._assets[modality]
        except (KeyError, TypeError):
            raise SensoryError(
                ErrorCode.SOURCE_NOT_FOUND,
                _MISSING_ASSET_MESSAGE,
            ) from None
        try:
            expected_kind, mime_prefix = _EXPECTED_MEDIA[modality]
            if (
                type(media) is not PreparedMedia
                or type(media.media_kind) is not MediaKind
                or media.media_kind is not expected_kind
                or type(media.mime_type) is not str
                or not media.mime_type.lower().startswith(mime_prefix)
                or len(media.mime_type) <= len(mime_prefix)
            ):
                raise ValueError
            candidate = Path(media.path)
            if not candidate.is_absolute():
                candidate = self._trusted_root / candidate
            candidate = Path(os.path.abspath(candidate))
            relative = candidate.relative_to(self._trusted_root)
            current = self._trusted_root
            for part in relative.parts:
                current = current / part
                if current.is_symlink():
                    raise ValueError
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self._trusted_root)
            if resolved != candidate or not stat.S_ISREG(candidate.lstat().st_mode):
                raise ValueError
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            raise SensoryError(
                ErrorCode.SOURCE_NOT_FOUND,
                _MISSING_ASSET_MESSAGE,
            ) from None
        return media
