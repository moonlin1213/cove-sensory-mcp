from __future__ import annotations

from mcp.types import CallToolResult

from cove_sensory_mcp.errors import SensoryError
from cove_sensory_mcp.services import AppServices

from ._sensing import failure_result, success_result
from .inputs import SenseMusicInput


async def sense_music(services: AppServices, input: SenseMusicInput) -> CallToolResult:
    try:
        return success_result(await services.sense("music", input))
    except SensoryError as error:
        return failure_result(error)
