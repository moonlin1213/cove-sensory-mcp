#!/usr/bin/env python3
"""Verify one standalone release archive without trusting archive paths."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from privacy_scan import scan

TOOLS = {
    "sensory_status", "sensory_setup_guide", "sensory_self_test", "sense_image",
    "sense_video", "sense_audio", "sense_music",
}


def _members(archive: Path) -> list[str]:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            names = [member.filename for member in source.infolist()]
    else:
        with tarfile.open(archive, "r:*") as source:
            names = [member.name for member in source.getmembers()]
    for name in names:
        path = PurePosixPath(name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("archive traversal")
    return names


def _read_member(archive: Path, name: str) -> bytes:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            return source.read(name)
    with tarfile.open(archive, "r:*") as source:
        handle = source.extractfile(name)
        if handle is None:
            raise ValueError("release member is unreadable")
        return handle.read()


def _verify_assets(archive: Path, names: list[str]) -> None:
    manifests = [name for name in names if name.endswith("/assets/self_test/manifest.json")]
    if len(manifests) != 1:
        raise ValueError("self-test manifest missing")
    manifest_name = manifests[0]
    root = manifest_name.removesuffix("manifest.json")
    payload = json.loads(_read_member(archive, manifest_name))
    assets = payload.get("assets")
    if not isinstance(assets, list) or len(assets) != 4:
        raise ValueError("self-test manifest is invalid")
    for item in assets:
        name = root + str(item["name"])
        if name not in names:
            raise ValueError("self-test asset missing")
        if hashlib.sha256(_read_member(archive, name)).hexdigest() != item["sha256"]:
            raise ValueError("self-test asset checksum mismatch")


def _extract(archive: Path, root: Path) -> None:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            source.extractall(root)
    else:
        with tarfile.open(archive, "r:*") as source:
            source.extractall(root, filter="data")


def _checksum(archive: Path) -> None:
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError("checksum sidecar missing")
    expected = sidecar.read_text(encoding="ascii").split()[0].lower()
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if expected != actual:
        raise ValueError("archive checksum mismatch")


async def _mcp_tools(executable: Path) -> set[str]:
    server = StdioServerParameters(command=str(executable), args=["serve"], env={})
    async with (
        stdio_client(server) as (reader, writer),
        ClientSession(reader, writer) as session,
    ):
        await session.initialize()
        listed = await session.list_tools()
        return {tool.name for tool in listed.tools}


def verify(archive: Path, platform_tag: str, *, execute: bool = True) -> None:
    if platform_tag not in {"macos-arm64", "macos-x64", "windows-x64"}:
        raise ValueError("unsupported platform tag")
    _checksum(archive)
    names = _members(archive)
    required = {"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "SBOM.spdx.json"}
    for item in required:
        if not any(name.endswith("/" + item) for name in names):
            raise ValueError(f"required release file missing: {item}")
    executable_name = "cove-sensory-mcp.exe" if platform_tag == "windows-x64" else "cove-sensory-mcp"
    candidates = [name for name in names if name.endswith("/" + executable_name)]
    if len(candidates) != 1:
        raise ValueError("expected standalone executable missing")
    _verify_assets(archive, names)
    privacy = scan(archive)
    if privacy:
        raise ValueError(f"privacy scan failed: {privacy[0].rule}")
    if not execute:
        return
    with tempfile.TemporaryDirectory(prefix="cove-release-verify-") as scratch:
        root = Path(scratch)
        _extract(archive, root)
        executable = root / candidates[0]
        if os.name != "nt":
            executable.chmod(executable.stat().st_mode | 0o100)
        version = subprocess.run([str(executable), "--version"], capture_output=True, text=True, timeout=20, check=False)
        if version.returncode or "0.1.1" not in version.stdout:
            raise ValueError("version smoke test failed")
        doctor = subprocess.run([str(executable), "doctor"], capture_output=True, text=True, timeout=30, check=False)
        if doctor.returncode not in {0, 1} or "Config:" not in doctor.stdout:
            raise ValueError("doctor smoke test failed")
        if asyncio.run(asyncio.wait_for(_mcp_tools(executable), timeout=30)) != TOOLS:
            raise ValueError("MCP tool inventory mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--platform-tag", required=True)
    parser.add_argument("--no-execute", action="store_true", help="Use only when inspecting a foreign-platform artifact.")
    args = parser.parse_args()
    try:
        verify(args.archive, args.platform_tag, execute=not args.no_execute)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"release verification failed: {exc}")
        return 1
    print("release verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
