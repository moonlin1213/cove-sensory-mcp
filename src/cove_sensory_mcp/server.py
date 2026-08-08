"""Official Python MCP server composition and protocol-clean stdio launch."""

from __future__ import annotations

import json
import logging
import sys
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer as FastMCP
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, InputRequiredResult, TextContent, ToolAnnotations
from pydantic import AfterValidator, Field

from cove_sensory_mcp.errors import ErrorCode, SensoryError, error_result
from cove_sensory_mcp.models import DetailLevel, Modality, ProviderId
from cove_sensory_mcp.services import AppServices
from cove_sensory_mcp.tools.audio import sense_audio
from cove_sensory_mcp.tools.image import sense_image
from cove_sensory_mcp.tools.inputs import (
    SenseAudioInput,
    SenseImageInput,
    SenseMusicInput,
    SenseVideoInput,
)
from cove_sensory_mcp.tools.music import sense_music
from cove_sensory_mcp.tools.setup import (
    sensory_self_test,
    sensory_setup_guide,
    sensory_status,
)
from cove_sensory_mcp.tools.video import sense_video

RequestedModality = Literal["image", "video_visual", "video_audio", "audio", "music"]


def _require_unique_modalities(
    modalities: list[RequestedModality],
) -> list[RequestedModality]:
    if len(set(modalities)) != len(modalities):
        raise ValueError("modalities must be unique")
    return modalities


RequestedModalities = Annotated[
    list[RequestedModality],
    Field(
        min_length=1,
        max_length=len(Modality),
        json_schema_extra={"uniqueItems": True},
    ),
    AfterValidator(_require_unique_modalities),
]

_SAFE_ANNOTATIONS = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
_SELF_TEST_ANNOTATIONS = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=True,
)
_SENSING_ANNOTATIONS = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=True,
)
_SENSING_DESCRIPTION = (
    "Read caller-authorized media and send it to the user's configured sensory Provider; "
    "the original media is never modified or stored permanently."
)
_STATUS_DESCRIPTION = "Inspect local configuration status; read-only setup tools never accept credentials."
_GUIDE_DESCRIPTION = "Inspect local configuration setup options; read-only setup tools never accept credentials."
_SELF_TEST_DESCRIPTION = (
    "Inspect local configuration readiness by sending tiny test media to the configured "
    "Provider; this may use a small amount of Provider quota and update verified state. "
    "This tool never accepts credentials."
)
_INVALID_ARGUMENTS_MESSAGE = "The tool arguments are invalid."


def _invalid_arguments_result() -> CallToolResult:
    payload = error_result(
        SensoryError(ErrorCode.CONFIG_INVALID, _INVALID_ARGUMENTS_MESSAGE)
    )
    return CallToolResult(
        content=[
            TextContent(
                text=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            )
        ],
        structured_content=payload,
        is_error=True,
    )


class _PrivacySafeFastMCP(FastMCP[None]):
    """Translate SDK tool failures before their raw details reach an MCP client."""

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Context[None, Any] | None = None,
    ) -> CallToolResult | InputRequiredResult:
        try:
            return await super().call_tool(name, arguments, context)
        except ToolError:
            return _invalid_arguments_result()


def create_server(services: AppServices) -> FastMCP[None]:
    """Bind the foundation setup handlers to the official Python MCP server."""
    server: FastMCP[None] = _PrivacySafeFastMCP("cove-sensory-mcp")

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
        annotations=_SELF_TEST_ANNOTATIONS,
    )
    async def self_test_tool(modalities: RequestedModalities) -> dict[str, object]:
        return await sensory_self_test(
            services, [Modality(modality) for modality in modalities]
        )

    @server.tool(
        name="sense_image",
        description=_SENSING_DESCRIPTION,
        annotations=_SENSING_ANNOTATIONS,
    )
    async def image_tool(
        source: str,
        question: str = "",
        detail: DetailLevel = DetailLevel.AUTO,
        language: str = "zh-CN",
        provider: ProviderId | None = None,
    ) -> CallToolResult:
        return await sense_image(
            services,
            SenseImageInput(
                source=source,
                question=question,
                detail=detail,
                language=language,
                provider=provider,
            ),
        )

    @server.tool(
        name="sense_video",
        description=_SENSING_DESCRIPTION,
        annotations=_SENSING_ANNOTATIONS,
    )
    async def video_tool(
        source: str,
        question: str = "",
        start_seconds: float | None = None,
        end_seconds: float | None = None,
        detail: DetailLevel = DetailLevel.AUTO,
        include_audio: bool = True,
        language: str = "zh-CN",
        visual_provider: ProviderId | None = None,
        audio_provider: ProviderId | None = None,
    ) -> CallToolResult:
        return await sense_video(
            services,
            SenseVideoInput(
                source=source,
                question=question,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                detail=detail,
                include_audio=include_audio,
                language=language,
                visual_provider=visual_provider,
                audio_provider=audio_provider,
            ),
        )

    @server.tool(
        name="sense_audio",
        description=_SENSING_DESCRIPTION,
        annotations=_SENSING_ANNOTATIONS,
    )
    async def audio_tool(
        source: str,
        question: str = "",
        start_seconds: float | None = None,
        end_seconds: float | None = None,
        detail: DetailLevel = DetailLevel.AUTO,
        include_transcript: bool = True,
        language: str = "zh-CN",
    ) -> CallToolResult:
        return await sense_audio(
            services,
            SenseAudioInput(
                source=source,
                question=question,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                detail=detail,
                include_transcript=include_transcript,
                language=language,
            ),
        )

    @server.tool(
        name="sense_music",
        description=_SENSING_DESCRIPTION,
        annotations=_SENSING_ANNOTATIONS,
    )
    async def music_tool(
        source: str,
        question: str = "",
        start_seconds: float | None = None,
        end_seconds: float | None = None,
        detail: DetailLevel = DetailLevel.AUTO,
        include_lyrics_transcript: bool = False,
        language: str = "zh-CN",
    ) -> CallToolResult:
        return await sense_music(
            services,
            SenseMusicInput(
                source=source,
                question=question,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                detail=detail,
                include_lyrics_transcript=include_lyrics_transcript,
                language=language,
            ),
        )

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
