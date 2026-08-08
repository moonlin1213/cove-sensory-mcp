"""Local, non-private media fixtures used only for capability self-tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.providers.base import MediaKind, PreparedMedia

_MISSING_ASSET_MESSAGE = "A required self-test media asset is unavailable."


class SelfTestAssetStore:
    """Resolve injected or packaged tiny fixtures without exposing their paths."""

    def __init__(self, assets: Mapping[Modality, PreparedMedia]) -> None:
        self._assets = dict(assets)

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
            }
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
        if not media.path.is_file():
            raise SensoryError(
                ErrorCode.SOURCE_NOT_FOUND,
                _MISSING_ASSET_MESSAGE,
            )
        return media
