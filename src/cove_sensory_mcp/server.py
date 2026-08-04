"""Official Python MCP server composition and protocol-clean stdio launch."""

from __future__ import annotations

import logging
import sys
from typing import Literal

from mcp.server.mcpserver import MCPServer as FastMCP
from mcp.types import ToolAnnotations

from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.services import AppServices
from cove_sensory_mcp.tools.setup import (
    sensory_self_test,
    sensory_setup_guide,
    sensory_status,
)

RequestedModality = Literal["image", "video_visual", "video_audio", "audio", "music"]

_SAFE_ANNOTATIONS = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
_STATUS_DESCRIPTION = (
    "Inspect local configuration status; read-only setup tools never accept credentials."
)
_GUIDE_DESCRIPTION = (
    "Inspect local configuration setup options; read-only setup tools never accept credentials."
)
_SELF_TEST_DESCRIPTION = (
    "Inspect local configuration readiness; read-only setup tools never accept credentials."
)


def create_server(services: AppServices) -> FastMCP[None]:
    """Bind the foundation setup handlers to the official Python MCP server."""
    server: FastMCP[None] = FastMCP("cove-sensory-mcp")

    @server.tool(
        name="sensory_status",
        description=_STATUS_DESCRIPTION,
        annotations=_SAFE_ANNOTATIONS,
    )
    async def status_tool() -> dict[str, object]:
        return await sensory_status(services)

    @server.tool(
        name="sensory_setup_guide",
        description=_GUIDE_DESCRIPTION,
        annotations=_SAFE_ANNOTATIONS,
    )
    async def setup_guide_tool() -> dict[str, object]:
        return await sensory_setup_guide(services)

    @server.tool(
        name="sensory_self_test",
        description=_SELF_TEST_DESCRIPTION,
        annotations=_SAFE_ANNOTATIONS,
    )
    async def self_test_tool(modalities: list[RequestedModality]) -> dict[str, object]:
        return await sensory_self_test(services, [Modality(modality) for modality in modalities])

    return server


def run_stdio(services: AppServices) -> None:
    """Run the MCP loop with every ordinary diagnostic directed to stderr."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )
    logging.getLogger(__name__).info("Starting cove-sensory-mcp stdio server")
    create_server(services).run(transport="stdio")
