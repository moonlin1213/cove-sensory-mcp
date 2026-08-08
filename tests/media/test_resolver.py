from __future__ import annotations

from pathlib import Path

import pytest

from cove_sensory_mcp.media.resolver import SourceResolver


@pytest.mark.asyncio
async def test_source_resolver_dispatches_local_source(tmp_path: Path) -> None:
    source = tmp_path / "a.jpg"
    source.write_bytes(b"x")
    result = await SourceResolver([tmp_path]).resolve(str(source), tmp_path / "job")
    assert result.source_kind == "local"
