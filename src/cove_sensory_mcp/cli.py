from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import __version__


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cove-sensory-mcp")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)
    if args.version:
        print(f"cove-sensory-mcp {__version__}")
        return 0
    parser.print_help()
    return 0


def entrypoint() -> None:
    raise SystemExit(main())
