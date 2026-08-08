from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from cove_sensory_mcp.media.runtime_install import (
    RuntimeManifest,
    install_media_runtime,
)


def _manifest(path: Path, archive: bytes, *, runtime_path: str = "bin/ffmpeg") -> None:
    payload = {
        "targets": {
            "test-x64": {
                "version": "1.2.3", "url": "https://example.test/ffmpeg-1.2.3.zip",
                "sha256": hashlib.sha256(archive).hexdigest(), "ffmpeg_path": runtime_path,
                "ffprobe_path": "bin/ffprobe", "license": "LGPL-2.1-or-later",
                "source_url": "https://ffmpeg.org/releases/ffmpeg-1.2.3.tar.xz",
                "distributor_url": "https://example.test/builds", "notice_files": ["LICENSE.txt"]
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_public_manifest_names_all_targets_but_refuses_unaudited_download() -> None:
    manifest = RuntimeManifest.load(Path("packaging/media-runtime-manifest.json"))
    assert manifest.pending == {"macos-arm64", "macos-x64", "windows-x64"}
    with pytest.raises(ValueError, match="provenance is pending"):
        manifest.require("macos-arm64")


def test_installer_requires_confirmation_before_download(tmp_path: Path) -> None:
    archive_path = tmp_path / "runtime.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("bin/ffmpeg", b"fake")
        archive.writestr("bin/ffprobe", b"fake")
        archive.writestr("LICENSE.txt", b"LGPL")
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, archive_path.read_bytes())
    called = False

    def download(_url: str, _target: Path) -> None:
        nonlocal called
        called = True

    with pytest.raises(PermissionError):
        install_media_runtime("test-x64", tmp_path / "install", download, lambda _: False, manifest_path=manifest)
    assert called is False


def test_manifest_rejects_archive_path_traversal(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, b"archive", runtime_path="../ffmpeg")
    with pytest.raises(ValueError, match="relative"):
        RuntimeManifest.load(manifest)
