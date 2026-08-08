"""Shared public result conversion for sensing handlers."""

from __future__ import annotations

import json

from mcp.types import CallToolResult, TextContent

from cove_sensory_mcp.errors import SensoryError, error_result
from cove_sensory_mcp.reports.render import render_sensory_text
from cove_sensory_mcp.reports.schemas import SensoryToolResult


def success_result(report: SensoryToolResult) -> CallToolResult:
    structured = report.model_dump(mode="json")
    return CallToolResult(
        content=[TextContent(text=render_sensory_text(report))],
        structured_content=structured,
        is_error=False,
    )


def failure_result(error: SensoryError) -> CallToolResult:
    structured = error_result(error)
    return CallToolResult(
        content=[
            TextContent(
                text=json.dumps(structured, ensure_ascii=False, separators=(",", ":"))
            )
        ],
        structured_content=structured,
        is_error=True,
    )
