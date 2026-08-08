from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.media.jobs import JobWorkspace, cleanup_stale_jobs


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, RuntimeError("boom"), SensoryError(ErrorCode.CONFIG_INVALID, "bad")])
async def test_workspace_cleans_after_success_and_errors(tmp_path: Path, failure: Exception | None) -> None:
    created: Path | None = None
    with pytest.raises(type(failure)) if failure else _does_not_raise():
        async with JobWorkspace.create(tmp_path, lambda: datetime.now(UTC)) as workspace:
            created = workspace.path
            workspace.new_artifact("clip", ".mp4").write_bytes(b"derived")
            assert re.fullmatch(r"job_[a-f0-9]{32}", workspace.path.name)
            if failure:
                raise failure
    assert created is not None and not created.exists()


class _does_not_raise:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> bool:
        return False


@pytest.mark.asyncio
async def test_workspace_cleans_after_cancellation(tmp_path: Path) -> None:
    entered = asyncio.Event()
    path: Path | None = None

    async def work() -> None:
        nonlocal path
        async with JobWorkspace.create(tmp_path, lambda: datetime.now(UTC)) as workspace:
            path = workspace.path
            entered.set()
            await asyncio.sleep(60)

    task = asyncio.create_task(work())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert path is not None and not path.exists()


def test_artifact_names_reject_traversal(tmp_path: Path) -> None:
    workspace = JobWorkspace.create(tmp_path, lambda: datetime.now(UTC))
    for name in ("../escape", "a/b", r"a\b"):
        with pytest.raises(ValueError):
            workspace.new_artifact(name, ".tmp")


def test_stale_cleanup_only_removes_strict_job_names(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    stale = tmp_path / ("job_" + "a" * 32)
    unrelated = tmp_path / "keep-me"
    stale.mkdir()
    unrelated.mkdir()
    old_timestamp = (now - timedelta(days=2)).timestamp()
    import os

    os.utime(stale, (old_timestamp, old_timestamp))
    os.utime(unrelated, (old_timestamp, old_timestamp))

    report = cleanup_stale_jobs(tmp_path, timedelta(days=1), lambda: now)

    assert report.removed == 1
    assert not stale.exists()
    assert unrelated.exists()
