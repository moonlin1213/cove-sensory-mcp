from __future__ import annotations

from pathlib import Path

import pytest

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.media.local import LocalSourceResolver


def test_authorized_local_file_resolves_without_public_path(tmp_path: Path) -> None:
    root = tmp_path / "media"
    root.mkdir()
    source = root / "photo.jpg"
    source.write_bytes(b"image")

    resolved = LocalSourceResolver([root]).resolve(str(source))

    assert resolved.path == source.resolve()
    assert resolved.display_name == "photo.jpg"
    assert resolved.original_size == 5
    assert "path" not in resolved.model_dump()


@pytest.mark.parametrize("raw", ["photo.jpg", "../photo.jpg"])
def test_relative_paths_are_rejected(tmp_path: Path, raw: str) -> None:
    with pytest.raises(SensoryError) as caught:
        LocalSourceResolver([tmp_path]).resolve(raw)
    assert caught.value.code is ErrorCode.PATH_NOT_ALLOWED


def test_symlink_escape_is_rejected_but_internal_symlink_is_allowed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "inside.jpg"
    inside.write_bytes(b"ok")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"no")
    internal_link = root / "internal.jpg"
    escape_link = root / "escape.jpg"
    internal_link.symlink_to(inside)
    escape_link.symlink_to(outside)

    assert LocalSourceResolver([root]).resolve(str(internal_link)).path == inside.resolve()
    with pytest.raises(SensoryError) as caught:
        LocalSourceResolver([root]).resolve(str(escape_link))
    assert caught.value.code is ErrorCode.PATH_NOT_ALLOWED


def test_missing_and_directory_sources_have_stable_errors(tmp_path: Path) -> None:
    resolver = LocalSourceResolver([tmp_path])
    with pytest.raises(SensoryError) as missing:
        resolver.resolve(str(tmp_path / "missing.jpg"))
    assert missing.value.code is ErrorCode.SOURCE_NOT_FOUND
    with pytest.raises(SensoryError) as directory:
        resolver.resolve(str(tmp_path))
    assert directory.value.code is ErrorCode.UNSUPPORTED_MEDIA_TYPE
