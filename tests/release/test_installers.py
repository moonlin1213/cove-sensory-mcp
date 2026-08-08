from __future__ import annotations

import hashlib
import os
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_installers_are_current_user_scoped_confirm_path_and_preserve_data() -> None:
    mac = (ROOT / "scripts/install-macos.sh").read_text(encoding="utf-8")
    windows = (ROOT / "scripts/install-windows.ps1").read_text(encoding="utf-8")
    assert "sudo" not in mac and "Program Files" not in windows
    assert "${HOME}/Library/Application Support" in mac
    assert "$env:LOCALAPPDATA" in windows
    assert "--confirm-path" in mac and "[switch]$ConfirmPath" in windows
    assert "--remove-data" in mac and "[switch]$RemoveData" in windows
    assert "rollback" in mac.lower() and "rollback" in windows.lower()


def test_macos_installer_verifies_and_activates_inside_injected_home(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    tree = tmp_path / "tree/cove-sensory-mcp"
    tree.mkdir(parents=True)
    executable = tree / "cove-sensory-mcp"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    machine = subprocess.check_output(["uname", "-m"], text=True).strip()
    tag = "macos-arm64" if machine in {"arm64", "aarch64"} else "macos-x64"
    archive = tmp_path / f"cove-sensory-mcp-0.1.0-{tag}.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        target.add(tree, arcname="cove-sensory-mcp")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/install-macos.sh"), "--archive", str(archive), "--sha256", digest],
        env={**os.environ, "HOME": str(home)}, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (home / "Library/Application Support/cove-sensory-mcp/bin/current/cove-sensory-mcp").is_file()
