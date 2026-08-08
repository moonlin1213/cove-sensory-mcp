"""Dispatch caller sources to authorized local or guarded direct-URL resolution."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from .local import LocalSourceResolver
from .network import MediaDownloader, NetworkLimits
from .types import ResolvedSource


class SourceResolver:
    def __init__(
        self,
        allowed_roots: list[str | Path],
        *,
        downloader: MediaDownloader | None = None,
        network_limits: NetworkLimits | None = None,
    ) -> None:
        self._local = LocalSourceResolver(allowed_roots)
        self._downloader = downloader or MediaDownloader()
        self._limits = network_limits or NetworkLimits(100 * 1024 * 1024, 30, 3)

    async def resolve(self, source: str, job_dir: Path) -> ResolvedSource:
        if urlsplit(source).scheme.lower() in {"http", "https"}:
            return await self._downloader.download(source, job_dir, self._limits)
        return self._local.resolve(source)
