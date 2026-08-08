from __future__ import annotations

import json
from pathlib import Path

import pytest

from cove_sensory_mcp.client_config import render_client_config


def test_generic_and_claude_are_valid_secret_free_json() -> None:
    rendered = render_client_config("generic", Path("/Applications/Cove Sensory/bin/cove"))
    config = json.loads(rendered)
    server = config["mcpServers"]["cove-sensory"]
    assert server["args"] == ["serve"]
    assert "Cove Sensory" in server["command"]
    assert "api" not in rendered.lower() and "env" not in rendered.lower()
    assert json.loads(render_client_config("claude-desktop", Path("/usr/bin/cove")))


def test_codex_toml_escapes_windows_paths() -> None:
    rendered = render_client_config("codex", Path(r"C:\Program Files\Cove\cove.exe"))
    assert 'args = ["serve"]' in rendered
    assert "\\\\" in rendered


def test_unknown_client_is_bounded() -> None:
    with pytest.raises(ValueError, match="unsupported client"):
        render_client_config("unknown", Path("cove"))
