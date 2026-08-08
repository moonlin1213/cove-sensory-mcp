#!/usr/bin/env python3
"""Bounded privacy scan for source trees and release archives."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_TEXT = 2_000_000
SKIP_PARTS = {".git", ".venv", ".uv-cache", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".superpowers", "__pycache__", "dist", "build"}
BLOCKED_NAMES = {".env", ".env.local", "credentials.json", "keyring.dump"}
BLOCKED_SUFFIXES = {".sqlite", ".sqlite3", ".db"}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".json", ".yaml", ".yml", ".txt", ".sh", ".ps1", ".spec"}
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".wav", ".mp3", ".m4a", ".mp4", ".mov", ".mkv"}


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    rule: str
    line: int | None = None

    def render(self) -> str:
        location = f":{self.line}" if self.line is not None else ""
        return f"{self.path}{location}: {self.rule}"


RULES = (
    ("PRIVATE_UNIX_USER_PATH", re.compile(r"/Users/(?!alice(?:/|\b))[A-Za-z0-9._-]+/", re.IGNORECASE)),
    ("PRIVATE_WINDOWS_USER_PATH", re.compile(r"C:\\Users\\(?!Alice(?:\\|\b))[A-Za-z0-9._-]+\\", re.IGNORECASE)),
    ("API_KEY", re.compile(r"(?:AIza[0-9A-Za-z_-]{30,}|sk-[A-Za-z0-9]{20,})")),
    ("AUTHORIZATION_VALUE", re.compile(r"(?:Authorization\s*[:=]\s*[\"']?|Bearer\s+)(?!\[REDACTED\]|\{|private-|test-|fake-)[A-Za-z0-9._~+/-]{12,}", re.IGNORECASE)),
)


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts


def _scan_file(name: str, data: bytes, denylist: tuple[str, ...], approved_media: set[str]) -> list[Finding]:
    path = PurePosixPath(name)
    findings: list[Finding] = []
    if path.name.lower() in BLOCKED_NAMES or path.suffix.lower() in BLOCKED_SUFFIXES:
        findings.append(Finding(name, "FORBIDDEN_DATA_FILE"))
        return findings
    if "tests" in path.parts:
        return findings
    packaged_self_test = "/assets/self_test/" in "/" + name
    if path.suffix.lower() in MEDIA_SUFFIXES and name not in approved_media and not packaged_self_test:
        findings.append(Finding(name, "UNAPPROVED_MEDIA"))
    if path.suffix.lower() not in TEXT_SUFFIXES or len(data) > MAX_TEXT:
        return findings
    text = data.decode("utf-8", errors="replace")
    markdown_example = path.suffix.lower() == ".md"
    for number, line in enumerate(text.splitlines(), start=1):
        for rule, pattern in RULES:
            if markdown_example and rule in {"PRIVATE_UNIX_USER_PATH", "PRIVATE_WINDOWS_USER_PATH"}:
                scrubbed = line.replace("/Users/alice/", "").replace("C:\\Users\\Alice\\", "")
            else:
                scrubbed = line
            if pattern.search(scrubbed):
                findings.append(Finding(name, rule, number))
        for denied in denylist:
            if denied and denied.casefold() in line.casefold():
                findings.append(Finding(name, "PRIVATE_IDENTIFIER", number))
    return findings


def _approved_assets(root: Path) -> set[str]:
    manifest = root / "src/cove_sensory_mcp/assets/self_test/manifest.json"
    if not manifest.is_file():
        return set()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    prefix = "src/cove_sensory_mcp/assets/self_test/"
    return {prefix + item["name"] for item in payload.get("assets", [])}


def scan(target: Path, denylist: tuple[str, ...] = ()) -> list[Finding]:
    findings: list[Finding] = []
    if target.is_dir():
        approved = _approved_assets(target)
        for path in sorted(target.rglob("*")):
            if not path.is_file() or SKIP_PARTS.intersection(path.relative_to(target).parts):
                continue
            name = path.relative_to(target).as_posix()
            findings.extend(_scan_file(name, path.read_bytes()[: MAX_TEXT + 1], denylist, approved))
        return findings
    if zipfile.is_zipfile(target):
        with zipfile.ZipFile(target) as archive:
            for member in archive.infolist():
                if not _safe_name(member.filename):
                    findings.append(Finding(member.filename, "ARCHIVE_TRAVERSAL"))
                    continue
                if member.is_dir():
                    continue
                findings.extend(_scan_file(member.filename, archive.read(member)[: MAX_TEXT + 1], denylist, set()))
        return findings
    with tarfile.open(target, "r:*") as archive:
        for member in archive.getmembers():
            if not _safe_name(member.name):
                findings.append(Finding(member.name, "ARCHIVE_TRAVERSAL"))
                continue
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            data = handle.read(MAX_TEXT + 1) if handle else b""
            findings.extend(_scan_file(member.name, data, denylist, set()))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("--denylist", type=Path)
    args = parser.parse_args()
    denylist = ()
    if args.denylist:
        denylist = tuple(line.strip() for line in args.denylist.read_text(encoding="utf-8").splitlines() if line.strip())
    findings = scan(args.target, denylist)
    for finding in findings:
        print(finding.render())
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
