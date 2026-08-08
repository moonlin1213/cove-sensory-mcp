from __future__ import annotations

from mcp.types import CallToolResult

from cove_sensory_mcp.errors import SensoryError
from cove_sensory_mcp.services import AppServices

from ._sensing import failure_result, success_result
from .inputs import SenseImageInput


async def sense_image(services: AppServices, input: SenseImageInput) -> CallToolResult:
    try:
        return success_result(await services.sense("image", input))
    except SensoryError as error:
        return failure_result(error)
