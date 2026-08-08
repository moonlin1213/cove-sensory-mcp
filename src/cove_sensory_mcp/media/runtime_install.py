"""Auditable, confirmation-gated installation of an optional FFmpeg runtime."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_LICENSES = {"LGPL-2.1-or-later", "GPL-3.0-or-later"}


@dataclass(frozen=True, slots=True)
class RuntimeEntry:
    platform_tag: str
    version: str
    url: str
    sha256: str
    ffmpeg_path: str
    ffprobe_path: str
    license: str
    source_url: str
    distributor_url: str
    notice_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InstalledRuntime:
    version: str
    root: Path
    ffmpeg: Path
    ffprobe: Path
    sha256: str


class RuntimeManifest:
    def __init__(self, entries: dict[str, RuntimeEntry], pending: frozenset[str]) -> None:
        self.entries = entries
        self.pending = pending

    @classmethod
    def load(cls, path: Path) -> RuntimeManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        targets = payload.get("targets")
        if not isinstance(targets, dict):
            raise TypeError("runtime manifest has no targets")
        entries: dict[str, RuntimeEntry] = {}
        pending: set[str] = set()
        for tag, raw in targets.items():
            if not isinstance(tag, str) or not isinstance(raw, dict):
                raise TypeError("invalid runtime target")
            if raw.get("status") == "provenance-pending":
                pending.add(tag)
                continue
            entry = _parse_entry(tag, raw)
            entries[tag] = entry
        return cls(entries, frozenset(pending))

    def require(self, platform_tag: str) -> RuntimeEntry:
        if platform_tag in self.pending:
            raise ValueError("runtime provenance is pending for this platform")
        try:
            return self.entries[platform_tag]
        except KeyError:
            raise ValueError("unsupported runtime platform") from None


def _relative(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("invalid runtime path")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("runtime path must be relative")
    return path.as_posix()


def _parse_entry(tag: str, raw: dict[str, Any]) -> RuntimeEntry:
    urls = (raw.get("url"), raw.get("source_url"), raw.get("distributor_url"))
    if any(not isinstance(url, str) or not url.startswith("https://") for url in urls):
        raise ValueError("runtime URLs must use HTTPS")
    if "latest" in str(raw.get("url", "")).lower():
        raise ValueError("runtime URL must identify a fixed archive")
    checksum = raw.get("sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("runtime checksum must be SHA-256")
    int(checksum, 16)
    license_id = raw.get("license")
    if license_id not in _LICENSES:
        raise ValueError("unreviewed runtime license")
    notices = raw.get("notice_files")
    if not isinstance(notices, list) or not notices:
        raise ValueError("runtime notices are required")
    return RuntimeEntry(
        platform_tag=tag,
        version=str(raw["version"]),
        url=str(raw["url"]),
        sha256=checksum.lower(),
        ffmpeg_path=_relative(raw["ffmpeg_path"]),
        ffprobe_path=_relative(raw["ffprobe_path"]),
        license=license_id,
        source_url=str(raw["source_url"]),
        distributor_url=str(raw["distributor_url"]),
        notice_files=tuple(_relative(item) for item in notices),
    )


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            members = source.infolist()
            names = [member.filename for member in members]
            for member in members:
                if member.is_dir():
                    continue
                mode = member.external_attr >> 16
                if mode and not (mode & 0o100000):
                    raise ValueError("runtime archive contains a special entry")
            _validate_names(names)
            source.extractall(destination)
        return
    with tarfile.open(archive, "r:*") as source:
        tar_members = source.getmembers()
        if any(not (item.isfile() or item.isdir()) for item in tar_members):
            raise ValueError("runtime archive contains links or special entries")
        _validate_names([item.name for item in tar_members])
        source.extractall(destination, filter="data")


def _validate_names(names: list[str]) -> None:
    for name in names:
        path = PurePosixPath(name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("runtime archive traversal")


def _probe(executable: Path, version: str) -> None:
    result = subprocess.run(
        [str(executable), "-version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode or version not in result.stdout:
        raise ValueError("runtime version probe failed")


def install_media_runtime(
    platform_tag: str,
    destination: Path,
    downloader: Callable[[str, Path], None],
    confirmer: Callable[[str], bool],
    *,
    manifest_path: Path,
) -> InstalledRuntime:
    entry = RuntimeManifest.load(manifest_path).require(platform_tag)
    active = destination / entry.version
    marker = active / ".cove-runtime.json"
    if marker.is_file():
        current = json.loads(marker.read_text(encoding="utf-8"))
        if current.get("sha256") == entry.sha256:
            ffmpeg = active / entry.ffmpeg_path
            ffprobe = active / entry.ffprobe_path
            _probe(ffmpeg, entry.version)
            _probe(ffprobe, entry.version)
            return InstalledRuntime(entry.version, active, ffmpeg, ffprobe, entry.sha256)
    if not confirmer(
        f"Download FFmpeg {entry.version} ({entry.license}, SHA-256 {entry.sha256})?"
    ):
        raise PermissionError("runtime download declined")
    destination.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".runtime-", dir=destination))
    archive = scratch / "download"
    staging = scratch / "extract"
    try:
        downloader(entry.url, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != entry.sha256:
            raise ValueError("runtime checksum mismatch")
        _safe_extract(archive, staging)
        ffmpeg = staging / entry.ffmpeg_path
        ffprobe = staging / entry.ffprobe_path
        _probe(ffmpeg, entry.version)
        _probe(ffprobe, entry.version)
        marker_payload = {"version": entry.version, "sha256": entry.sha256}
        (staging / ".cove-runtime.json").write_text(
            json.dumps(marker_payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        if active.exists():
            shutil.rmtree(active)
        os.replace(staging, active)
        return InstalledRuntime(
            entry.version,
            active,
            active / entry.ffmpeg_path,
            active / entry.ffprobe_path,
            entry.sha256,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
