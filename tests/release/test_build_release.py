from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def _builder():
    spec = importlib.util.spec_from_file_location("build_release", ROOT / "scripts/build_release.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spec_uses_onedir_assets_metadata_and_keyring_without_private_data() -> None:
    source = (ROOT / "packaging/cove_sensory_mcp.spec").read_text(encoding="utf-8")
    assert "COLLECT(" in source and "assets/self_test" in source
    assert "copy_metadata" in source and "keyring.backends" in source
    assert ".env" not in source and "config.yaml" not in source


def test_builder_rejects_non_native_tag_and_is_reproducible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _builder()
    source = tmp_path / "app"
    source.mkdir()
    (source / "cove-sensory-mcp").write_bytes(b"executable")
    monkeypatch.setattr(module, "native_platform_tag", lambda: "macos-arm64")
    with pytest.raises(ValueError, match="native"):
        module.build_archive("windows-x64", source, tmp_path / "wrong")
    first = module.build_archive("macos-arm64", source, tmp_path / "one")
    second = module.build_archive("macos-arm64", source, tmp_path / "two")
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
