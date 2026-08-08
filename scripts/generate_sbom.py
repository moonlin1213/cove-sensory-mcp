#!/usr/bin/env python3
"""Generate a deterministic, lock-derived SPDX 2.3 package inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path


def generate(lockfile: Path, output: Path) -> None:
    lock = tomllib.loads(lockfile.read_text(encoding="utf-8"))
    packages = []
    for item in sorted(lock.get("package", []), key=lambda value: (value["name"], value["version"])):
        identity = f"{item['name']}-{item['version']}"
        packages.append(
            {
                "SPDXID": "SPDXRef-Package-" + hashlib.sha256(identity.encode()).hexdigest()[:16],
                "name": item["name"],
                "versionInfo": item["version"],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "copyrightText": "NOASSERTION",
            }
        )
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "cove-sensory-mcp-0.1.0",
        "documentNamespace": "https://example.invalid/cove-sensory-mcp/0.1.0/sbom",
        "creationInfo": {"created": "2026-08-04T00:00:00Z", "creators": ["Tool: cove-sensory-mcp-generate-sbom"]},
        "packages": packages,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lockfile", type=Path, default=Path("uv.lock"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.lockfile, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
