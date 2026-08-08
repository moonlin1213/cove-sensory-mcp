from __future__ import annotations

from pathlib import Path

import pytest
from mcp import Client
from pydantic import ValidationError

from cove_sensory_mcp.config.secrets import MemorySecretStore
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.reports.schemas import Coverage, ImageSensoryReport
from cove_sensory_mcp.server import create_server
from cove_sensory_mcp.services import AppServices
from cove_sensory_mcp.tools.inputs import SenseImageInput, SenseVideoInput


@pytest.mark.parametrize(
    "payload",
    [
        {"source": ""},
        {"source": "x", "detail": "deep"},
        {"source": "x", "provider": "bad\nprovider"},
        {"source": "x", "api_key": "secret"},
    ],
)
def test_image_input_rejects_unsafe_or_unknown_fields(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        SenseImageInput.model_validate(payload)


def test_video_input_rejects_invalid_ranges() -> None:
    with pytest.raises(ValidationError):
        SenseVideoInput(source="x", start_seconds=5, end_seconds=5)
    with pytest.raises(ValidationError):
        SenseVideoInput(source="x", start_seconds=-1)


class ImageCoordinator:
    async def sense(self, input: SenseImageInput) -> ImageSensoryReport:
        return ImageSensoryReport(
            request_id="sense_public",
            status="completed",
            coverage=Coverage(visual=True, visual_provider="eye"),
            summary="a visible object",
        )


@pytest.mark.asyncio
async def test_image_tool_returns_matching_text_and_structured_content(
    tmp_path: Path,
) -> None:
    services = AppServices(
        ConfigStore(tmp_path / "config.yaml"),
        MemorySecretStore(),
        sensing_coordinators={"image": ImageCoordinator()},
    )
    async with Client(create_server(services), mode="legacy") as client:
        result = await client.call_tool(
            "sense_image", {"source": "/private/source.jpg"}
        )
    assert result.is_error is False
    assert result.structured_content["summary"] == "a visible object"
    assert "a visible object" in result.content[0].text
    assert "/private/source.jpg" not in result.content[0].text


@pytest.mark.asyncio
async def test_sensing_tool_annotations_disclose_provider_transmission(
    tmp_path: Path,
) -> None:
    services = AppServices(
        ConfigStore(tmp_path / "config.yaml"),
        MemorySecretStore(),
        sensing_coordinators={},
    )
    tools = {tool.name: tool for tool in await create_server(services).list_tools()}
    for name in ("sense_image", "sense_video", "sense_audio", "sense_music"):
        tool = tools[name]
        assert "provider" in tool.description.lower()
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False
        assert tool.annotations.open_world_hint is True
