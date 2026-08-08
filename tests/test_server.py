"""FastMCP registration and stdio transport contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from mcp import Client

from cove_sensory_mcp.config.secrets import MemorySecretStore
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.server import create_server
from cove_sensory_mcp.services import AppServices


@pytest.fixture
def services(tmp_path) -> AppServices:
    """Provide isolated dependencies without persistent credential access."""
    return AppServices(
        config_store=ConfigStore(tmp_path / "config.yaml"),
        secret_store=MemorySecretStore(),
    )


@pytest.mark.asyncio
async def test_foundation_server_lists_exact_setup_tools(services: AppServices) -> None:
    """Adding, omitting, or renaming a foundation tool changes the public MCP surface."""
    server = create_server(services)

    names = sorted(tool.name for tool in await server.list_tools())

    assert server.name == "cove-sensory-mcp"
    assert names == ["sensory_self_test", "sensory_setup_guide", "sensory_status"]


@pytest.mark.asyncio
async def test_setup_tools_advertise_local_read_only_safety(
    services: AppServices,
) -> None:
    """Unsafe descriptions or mutation hints could invite credential-bearing calls."""
    tools = await create_server(services).list_tools()

    safe_tools = [tool for tool in tools if tool.name != "sensory_self_test"]
    for tool in safe_tools:
        assert "inspect local configuration" in tool.description.lower()
        assert "never accept credentials" in tool.description.lower()
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False
        assert tool.annotations.idempotent_hint is True
        assert tool.annotations.open_world_hint is False

    self_test = next(tool for tool in tools if tool.name == "sensory_self_test")
    description = self_test.description.lower()
    assert "tiny test media" in description
    assert "provider quota" in description
    assert self_test.annotations is not None
    assert self_test.annotations.read_only_hint is False
    assert self_test.annotations.destructive_hint is True
    assert self_test.annotations.idempotent_hint is False
    assert self_test.annotations.open_world_hint is True


@pytest.mark.asyncio
async def test_setup_tools_expose_only_explicit_public_inputs(
    services: AppServices,
) -> None:
    """Leaking service or verifier parameters would expose internal composition over MCP."""
    tools = {tool.name: tool for tool in await create_server(services).list_tools()}

    assert tools["sensory_status"].input_schema["properties"] == {}
    assert tools["sensory_setup_guide"].input_schema["properties"] == {}
    assert tools["sensory_self_test"].input_schema["required"] == ["modalities"]
    assert tools["sensory_self_test"].input_schema["properties"] == {
        "modalities": {
            "items": {
                "enum": ["image", "video_visual", "video_audio", "audio", "music"],
                "type": "string",
            },
            "title": "Modalities",
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "uniqueItems": True,
        }
    }


@pytest.mark.parametrize("modalities", [[], ["image", "image"]])
@pytest.mark.asyncio
async def test_empty_and_duplicate_mcp_modalities_return_stable_invalid_arguments(
    services: AppServices,
    modalities: list[str],
) -> None:
    async with Client(create_server(services), mode="legacy") as client:
        result = await client.call_tool(
            "sensory_self_test",
            {"modalities": modalities},
        )

    assert result.is_error is True
    assert result.structured_content == {
        "status": "error",
        "error": {
            "code": "CONFIG_INVALID",
            "message": "The tool arguments are invalid.",
            "retryable": False,
        },
    }
    assert services.config_store.load().providers == {}


@pytest.mark.parametrize(
    ("invalid_modality", "private_fragment"),
    [
        ("sk-review-secret-123456789", "review-secret"),
        (r"C:\Users\private-review\secret.txt", "private-review"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_self_test_modalities_return_only_a_bounded_public_error(
    services: AppServices,
    invalid_modality: str,
    private_fragment: str,
) -> None:
    """SDK argument validation must not reflect credentials, paths, or raw internals."""
    async with Client(create_server(services), mode="legacy") as client:
        result = await client.call_tool(
            "sensory_self_test",
            {"modalities": [invalid_modality]},
        )

    assert result.is_error is True
    assert result.structured_content == {
        "status": "error",
        "error": {
            "code": "CONFIG_INVALID",
            "message": "The tool arguments are invalid.",
            "retryable": False,
        },
    }
    public_wire = result.model_dump_json(by_alias=True)
    assert len(public_wire) < 800
    assert private_fragment not in public_wire
    for raw_detail in (
        "validation error",
        "self_test_toolArguments",
        "literal_error",
        "input_value",
        "pydantic.dev",
        "ValidationError",
    ):
        assert raw_detail not in public_wire


def test_serve_stdout_contains_only_mcp_json(tmp_path) -> None:
    """A plain-text diagnostic on stdout would corrupt the line-framed MCP transport."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "protocol-clean-test", "version": "1.0.0"},
        },
    }
    environment = os.environ.copy()
    environment["HOME"] = str(tmp_path)

    completed = subprocess.run(
        [sys.executable, "-m", "cove_sensory_mcp", "serve"],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert lines
    messages = [json.loads(line) for line in lines]
    assert messages[0]["id"] == 1
    assert messages[0]["result"]["serverInfo"]["name"] == "cove-sensory-mcp"
    assert "Starting cove-sensory-mcp stdio server" in completed.stderr
    assert "Starting cove-sensory-mcp stdio server" not in completed.stdout
