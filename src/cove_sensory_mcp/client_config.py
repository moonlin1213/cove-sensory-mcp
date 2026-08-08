"""Secret-free local stdio MCP client configuration rendering."""

from __future__ import annotations

import json
from pathlib import Path

CLIENTS = ("generic", "codex", "claude-desktop", "claude-code")


def _command(executable: Path) -> str:
    return str(executable.resolve(strict=False))


def render_client_config(client: str, executable: Path) -> str:
    command = _command(executable)
    if client == "codex":
        escaped = command.replace("\\", "\\\\").replace('"', '\\"')
        return (
            "[mcp_servers.cove-sensory]\n"
            f'command = "{escaped}"\n'
            'args = ["serve"]\n'
        )
    payload = {
        "mcpServers": {
            "cove-sensory": {"command": command, "args": ["serve"]}
        }
    }
    if client in {"generic", "claude-desktop", "claude-code"}:
        return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    raise ValueError(f"unsupported client: {client}")
