"""Strictly scoped transient job directories and stale cleanup."""

from __future__ import annotations

import re
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Self

from cove_sensory_mcp.errors import ErrorCode, SensoryError

Clock = Callable[[], datetime]
_JOB_PATTERN = re.compile(r"job_[a-f0-9]{32}\Z")
_ARTIFACT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_SUFFIX_PATTERN = re.compile(r"\.[A-Za-z0-9]{1,12}\Z")


@dataclass(frozen=True, slots=True)
class CleanupReport:
    removed: int
    failures: tuple[str, ...] = ()


def _delete_job(root: Path, candidate: Path) -> None:
    resolved_root = root.resolve(strict=True)
    resolved_candidate = candidate.resolve(strict=True)
    if (
        resolved_candidate.parent != resolved_root
        or _JOB_PATTERN.fullmatch(candidate.name) is None
    ):
        raise SensoryError(
            ErrorCode.TEMP_CLEANUP_FAILED,
            "A temporary job directory could not be cleaned safely.",
        )
    shutil.rmtree(resolved_candidate)


class JobWorkspace:
    """Own every derivative for one request and delete it on every exit path."""

    def __init__(self, jobs_root: Path, path: Path, clock: Clock) -> None:
        self._root = jobs_root
        self.path = path
        self._clock = clock

    @classmethod
    def create(cls, jobs_root: Path, clock: Clock) -> JobWorkspace:
        root = jobs_root.resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"job_{uuid.uuid4().hex}"
        path.mkdir(mode=0o700)
        return cls(root, path, clock)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        try:
            if self.path.exists():
                _delete_job(self._root, self.path)
        except (OSError, SensoryError) as cleanup_error:
            if exc is None:
                raise SensoryError(
                    ErrorCode.TEMP_CLEANUP_FAILED,
                    "Temporary media cleanup failed; run doctor.",
                    cause=cleanup_error,
                ) from None
        return False

    def new_artifact(self, name: str, suffix: str) -> Path:
        if (
            _ARTIFACT_PATTERN.fullmatch(name) is None
            or _SUFFIX_PATTERN.fullmatch(suffix) is None
        ):
            raise ValueError("artifact name or suffix is invalid")
        artifact = self.path / f"{name}_{uuid.uuid4().hex}{suffix.lower()}"
        if artifact.parent != self.path:
            raise ValueError("artifact must remain in its workspace")
        return artifact


def cleanup_stale_jobs(
    jobs_root: Path,
    older_than: timedelta,
    clock: Clock,
) -> CleanupReport:
    """Delete only strict stale job entries and report bounded anonymous failures."""
    if older_than.total_seconds() < 0:
        raise ValueError("older_than must be non-negative")
    root = jobs_root.resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    cutoff = clock().timestamp() - older_than.total_seconds()
    removed = 0
    failures: list[str] = []
    for candidate in root.iterdir():
        if _JOB_PATTERN.fullmatch(candidate.name) is None:
            continue
        try:
            if candidate.stat().st_mtime >= cutoff:
                continue
            _delete_job(root, candidate)
            removed += 1
        except (OSError, SensoryError):
            if len(failures) < 20:
                failures.append(ErrorCode.TEMP_CLEANUP_FAILED.value)
    return CleanupReport(removed=removed, failures=tuple(failures))
