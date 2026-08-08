"""Self-test assets stay inside one trusted, type-safe local root."""

from __future__ import annotations

from pathlib import Path

import pytest

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.providers.base import MediaKind, PreparedMedia
from cove_sensory_mcp.verification.assets import SelfTestAssetStore


def _media(path: Path, mime_type: str, kind: MediaKind) -> PreparedMedia:
    return PreparedMedia(path, mime_type, kind, None)


def _assert_private_asset_error(
    store: SelfTestAssetStore,
    modality: Modality,
    *private_fragments: str,
) -> None:
    with pytest.raises(SensoryError) as caught:
        store.get(modality)

    assert caught.value.code is ErrorCode.SOURCE_NOT_FOUND
    public = str(caught.value)
    assert public == "A required self-test media asset is unavailable."
    assert all(fragment not in public for fragment in private_fragments)


def test_asset_store_rejects_missing_and_resolved_escape_without_path_leak(
    tmp_path: Path,
) -> None:
    trusted = tmp_path / "trusted-private-root"
    trusted.mkdir()
    missing = trusted / "private-missing.png"
    outside = tmp_path / "private-outside.png"
    outside.write_bytes(b"image")

    _assert_private_asset_error(
        SelfTestAssetStore(
            {Modality.IMAGE: _media(missing, "image/png", MediaKind.IMAGE)},
            trusted_root=trusted,
        ),
        Modality.IMAGE,
        str(tmp_path),
        "private-missing",
    )
    _assert_private_asset_error(
        SelfTestAssetStore(
            {Modality.IMAGE: _media(outside, "image/png", MediaKind.IMAGE)},
            trusted_root=trusted,
        ),
        Modality.IMAGE,
        str(tmp_path),
        "private-outside",
    )


def test_asset_store_rejects_non_regular_file(tmp_path: Path) -> None:
    private_directory = tmp_path / "private-directory.png"
    private_directory.mkdir()
    store = SelfTestAssetStore(
        {
            Modality.IMAGE: _media(
                private_directory,
                "image/png",
                MediaKind.IMAGE,
            )
        },
        trusted_root=tmp_path,
    )

    _assert_private_asset_error(
        store,
        Modality.IMAGE,
        str(tmp_path),
        "private-directory",
    )


@pytest.mark.parametrize("symlink_position", ["direct", "parent"])
def test_asset_store_rejects_direct_and_ancestor_symlinks_inside_root(
    tmp_path: Path,
    symlink_position: str,
) -> None:
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    outside = tmp_path / "private-target"
    outside.mkdir()
    target = outside / "private-image.png"
    target.write_bytes(b"image")
    try:
        if symlink_position == "direct":
            candidate = trusted / "fixture.png"
            candidate.symlink_to(target)
        else:
            linked_parent = trusted / "fixtures"
            linked_parent.symlink_to(outside, target_is_directory=True)
            candidate = linked_parent / target.name
    except OSError:
        pytest.skip("this platform does not permit test symlink creation")

    _assert_private_asset_error(
        SelfTestAssetStore(
            {Modality.IMAGE: _media(candidate, "image/png", MediaKind.IMAGE)},
            trusted_root=trusted,
        ),
        Modality.IMAGE,
        str(tmp_path),
        "private-target",
    )


@pytest.mark.parametrize(
    ("modality", "mime_type", "kind"),
    [
        (Modality.IMAGE, "video/mp4", MediaKind.VIDEO),
        (Modality.IMAGE, "video/mp4", MediaKind.IMAGE),
        (Modality.VIDEO_VISUAL, "image/png", MediaKind.IMAGE),
        (Modality.VIDEO_AUDIO, "audio/wav", MediaKind.VIDEO),
        (Modality.AUDIO, "video/mp4", MediaKind.AUDIO),
        (Modality.MUSIC, "audio/wav", MediaKind.IMAGE),
    ],
)
def test_asset_store_rejects_modality_kind_and_mime_mismatch(
    tmp_path: Path,
    modality: Modality,
    mime_type: str,
    kind: MediaKind,
) -> None:
    asset = tmp_path / "fixture.bin"
    asset.write_bytes(b"fixture")
    store = SelfTestAssetStore(
        {modality: _media(asset, mime_type, kind)},
        trusted_root=tmp_path,
    )

    _assert_private_asset_error(store, modality, str(tmp_path))


def test_asset_store_accepts_exact_kind_and_mime_inside_trusted_root(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "fixture.wav"
    asset.write_bytes(b"fixture")
    media = _media(asset, "audio/wav", MediaKind.AUDIO)
    store = SelfTestAssetStore(
        {Modality.MUSIC: media},
        trusted_root=tmp_path,
    )

    assert store.get(Modality.MUSIC) == media


def test_relative_asset_is_returned_as_canonical_root_path_with_normalized_mime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    canonical = trusted / "fixture.png"
    canonical.write_bytes(b"trusted-image")
    untrusted_cwd = tmp_path / "cwd"
    untrusted_cwd.mkdir()
    (untrusted_cwd / "fixture.png").write_bytes(b"wrong-image")
    monkeypatch.chdir(untrusted_cwd)
    original = _media(
        Path("fixture.png"),
        "IMAGE/PNG",
        MediaKind.IMAGE,
    )
    store = SelfTestAssetStore(
        {Modality.IMAGE: original},
        trusted_root=trusted,
    )

    normalized = store.get(Modality.IMAGE)

    assert normalized is not original
    assert normalized.path == canonical.resolve(strict=True)
    assert normalized.path.is_absolute()
    assert normalized.path.read_bytes() == b"trusted-image"
    assert normalized.mime_type == "image/png"


def test_trusted_root_alias_uses_resolved_containment_and_canonical_result(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    asset = real_root / "fixture.png"
    asset.write_bytes(b"fixture")
    alias_root = tmp_path / "root-alias"
    try:
        alias_root.symlink_to(real_root, target_is_directory=True)
    except OSError:
        pytest.skip("this platform does not permit test symlink creation")
    store = SelfTestAssetStore(
        {
            Modality.IMAGE: _media(
                alias_root / "fixture.png",
                "image/png",
                MediaKind.IMAGE,
            )
        },
        trusted_root=alias_root,
    )

    normalized = store.get(Modality.IMAGE)

    assert normalized.path == asset.resolve(strict=True)
