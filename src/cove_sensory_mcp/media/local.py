"""Canonical containment checks for user-authorized local media roots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from cove_sensory_mcp.errors import ErrorCode, SensoryError

from .types import ResolvedSource


def _path_is_within(candidate: Path, root: Path) -> bool:
    candidate_text = os.path.normcase(str(candidate))
    root_text = os.path.normcase(str(root))
    try:
        return os.path.commonpath((candidate_text, root_text)) == root_text
    except ValueError:
        return False


class LocalSourceResolver:
    """Resolve only absolute regular files beneath canonical configured roots."""

    def __init__(self, allowed_roots: Iterable[str | Path]) -> None:
        self._roots: tuple[Path, ...] = tuple(
            Path(root).expanduser().resolve(strict=True) for root in allowed_roots
        )
        if any(not root.is_dir() for root in self._roots):
            raise SensoryError(
                ErrorCode.CONFIG_INVALID, "An allowed media root is invalid."
            )

    def resolve(self, raw: str) -> ResolvedSource:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            raise SensoryError(
                ErrorCode.PATH_NOT_ALLOWED, "The local media path is not allowed."
            )
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError):
            raise SensoryError(
                ErrorCode.SOURCE_NOT_FOUND, "The media source was not found."
            ) from None
        if not resolved.is_file():
            raise SensoryError(
                ErrorCode.UNSUPPORTED_MEDIA_TYPE,
                "The media source is not a supported regular file.",
            )
        if not any(_path_is_within(resolved, root) for root in self._roots):
            raise SensoryError(
                ErrorCode.PATH_NOT_ALLOWED, "The local media path is not allowed."
            )
        return ResolvedSource(
            path=resolved,
            source_kind="local",
            display_name=resolved.name,
            cleanup_required=False,
            original_size=resolved.stat().st_size,
        )
