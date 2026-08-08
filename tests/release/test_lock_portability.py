from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_pillow_lock_contains_cross_platform_install_artifacts() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    pillow = next(package for package in lock["package"] if package["name"] == "pillow")

    assert pillow.get("sdist"), "Pillow must retain a source distribution in uv.lock"
    wheel_urls = [wheel["url"].lower() for wheel in pillow.get("wheels", [])]
    assert any("manylinux" in url or "musllinux" in url for url in wheel_urls)
    assert any("macosx" in url for url in wheel_urls)
    assert any("win_amd64" in url for url in wheel_urls)


def test_readme_installs_the_mcp_from_its_public_source_checkout() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "git clone https://github.com/moonlin1213/cove-sensory-mcp.git" in readme
    assert "uv sync --locked" in readme
    assert "uv run cove-sensory-mcp doctor" in readme
    assert "uv run cove-sensory-mcp print-config --client generic" in readme
    assert "uvx cove-sensory-mcp doctor" not in readme
