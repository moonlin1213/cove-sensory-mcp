from __future__ import annotations

from pathlib import Path

import pytest

from cove_sensory_mcp.coordinators.image import ImageCoordinator
from cove_sensory_mcp.providers.base import (
    MediaKind,
    PreparedMedia,
    ProviderMediaLimits,
)

from .conftest import Executor, Resolver, workspace_factory


@pytest.mark.asyncio
async def test_image_coordinator_resolves_prepares_calls_provider_and_cleans(
    tmp_path: Path, request_factory
) -> None:
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"source")
    jobs = tmp_path / "jobs"
    executor = Executor()

    async def prepare(resolved, focus, detail, limits, workspace):
        output = workspace.new_artifact("image", ".jpg")
        output.write_bytes(b"prepared")
        return [PreparedMedia(output, "image/jpeg", MediaKind.IMAGE, None)]

    result = await ImageCoordinator(
        resolver=Resolver(source),
        workspace_factory=workspace_factory(jobs),
        executor=executor,
        provider_limits=ProviderMediaLimits(1000, frozenset({"image/jpeg"}), None, 100),
        prepare=prepare,
    ).sense(request_factory(str(source)))
    assert result.status == "completed" and result.coverage.visual is True
    assert result.providers[0].provider_id == "provider"
    assert list(jobs.iterdir()) == []
