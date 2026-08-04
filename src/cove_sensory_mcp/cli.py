from __future__ import annotations

import argparse
import os
import platform
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config.paths import AppPaths
from .config.secrets import KeyringSecretStore
from .config.store import ConfigStore
from .server import run_stdio
from .services import AppServices


def _optional_environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _build_services() -> AppServices:
    paths = AppPaths.for_system(
        platform.system(),
        home=Path.home(),
        roaming=_optional_environment_path("APPDATA"),
        local=_optional_environment_path("LOCALAPPDATA"),
    )
    return AppServices(
        config_store=ConfigStore(paths.config_file),
        secret_store=KeyringSecretStore(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cove-sensory-mcp")
    parser.add_argument("--version", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="Run the local MCP server over stdio.")
    args = parser.parse_args(argv)
    if args.version:
        print(f"cove-sensory-mcp {__version__}")
        return 0
    if args.command == "serve":
        run_stdio(_build_services())
        return 0
    parser.print_help()
    return 0


def entrypoint() -> None:
    raise SystemExit(main())
