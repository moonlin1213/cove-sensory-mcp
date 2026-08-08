#!/usr/bin/env python3
"""Create a reproducible archive from one native PyInstaller onedir build."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import platform
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

TAGS = {"macos-arm64", "macos-x64", "windows-x64"}


def native_platform_tag() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        return "macos-arm64" if machine in {"arm64", "aarch64"} else "macos-x64"
    if system == "Windows" and machine in {"amd64", "x86_64"}:
        return "windows-x64"
    raise ValueError("this host is not a standalone release target")


def _files(source: Path) -> list[Path]:
    forbidden = {".env", ".git", "config.yaml", "credentials", "cache"}
    files = [path for path in source.rglob("*") if path.is_file()]
    if any(forbidden.intersection(path.parts) for path in files):
        raise ValueError("standalone tree contains local or development data")
    return sorted(files, key=lambda item: item.relative_to(source).as_posix())


def build_archive(platform_tag: str, source: Path, output: Path) -> Path:
    if platform_tag not in TAGS or platform_tag != native_platform_tag():
        raise ValueError("platform tag does not match the native build job")
    if not source.is_dir():
        raise ValueError("PyInstaller onedir build is missing")
    output.mkdir(parents=True, exist_ok=True)
    stem = f"cove-sensory-mcp-0.1.0-{platform_tag}"
    if platform_tag == "windows-x64":
        archive = output / f"{stem}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
            for path in _files(source):
                relative = PurePosixPath("cove-sensory-mcp") / path.relative_to(source).as_posix()
                info = zipfile.ZipInfo(str(relative), (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                target.writestr(info, path.read_bytes())
    else:
        archive = output / f"{stem}.tar.gz"
        with tempfile.NamedTemporaryFile(dir=output, delete=False) as scratch:
            temp_name = Path(scratch.name)
        try:
            with temp_name.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped, tarfile.open(fileobj=zipped, mode="w") as target:
                for path in _files(source):
                    relative = PurePosixPath("cove-sensory-mcp") / path.relative_to(source).as_posix()
                    info = target.gettarinfo(str(path), arcname=str(relative))
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        target.addfile(info, handle)
            os.replace(temp_name, archive)
        finally:
            temp_name.unlink(missing_ok=True)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{digest}  {archive.name}\n", encoding="ascii"
    )
    return archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform-tag", choices=sorted(TAGS), required=True)
    parser.add_argument("--source", type=Path, default=Path("dist/cove-sensory-mcp"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_archive(args.platform_tag, args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
