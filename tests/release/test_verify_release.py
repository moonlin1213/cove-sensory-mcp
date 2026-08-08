from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def _verifier():
    import sys
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("verify_release", ROOT / "scripts/verify_release.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _archive(tmp_path: Path, missing: str | None = None) -> Path:
    tree = tmp_path / "tree/cove-sensory-mcp"
    (tree / "assets/self_test").mkdir(parents=True)
    assets = []
    for index, name in enumerate(("shape.png", "motion.mp4", "speech.wav", "music.wav")):
        data = f"asset-{index}".encode()
        (tree / "assets/self_test" / name).write_bytes(data)
        assets.append({"name": name, "sha256": hashlib.sha256(data).hexdigest()})
    (tree / "assets/self_test/manifest.json").write_text(json.dumps({"assets": assets}))
    names = ["LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "SBOM.spdx.json", "cove-sensory-mcp"]
    for name in names:
        if name != missing:
            path = tree / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture", encoding="utf-8")
    archive = tmp_path / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        target.add(tree, arcname="cove-sensory-mcp")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".gz.sha256").write_text(f"{digest}  {archive.name}\n")
    return archive


def test_release_verifier_checks_required_inventory_without_execution(tmp_path: Path) -> None:
    verifier = _verifier()
    verifier.verify(_archive(tmp_path), "macos-arm64", execute=False)


def test_release_verifier_fails_a_missing_sbom(tmp_path: Path) -> None:
    verifier = _verifier()
    with pytest.raises(ValueError, match="SBOM"):
        verifier.verify(_archive(tmp_path, "SBOM.spdx.json"), "macos-arm64", execute=False)
