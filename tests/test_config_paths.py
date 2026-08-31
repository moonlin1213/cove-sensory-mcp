from __future__ import annotations

from pathlib import Path

import pytest

from cove_sensory_mcp.config.paths import AppPaths
from cove_sensory_mcp.errors import ErrorCode, SensoryError


def test_macos_paths_use_library_directories(tmp_path: Path) -> None:
    """Changing the macOS branch must not silently select POSIX locations."""
    paths = AppPaths.for_system("Darwin", home=tmp_path, roaming=None, local=None)

    assert paths.config_file == tmp_path / "Library/Application Support/cove-sensory-mcp/config.yaml"
    assert paths.jobs_dir == tmp_path / "Library/Caches/cove-sensory-mcp/jobs"


def test_windows_paths_use_roaming_and_local(tmp_path: Path) -> None:
    """Changing the Windows branch must keep config and cache in their correct roots."""
    roaming = tmp_path / "Roaming"
    local = tmp_path / "Local"

    paths = AppPaths.for_system("Windows", home=tmp_path, roaming=roaming, local=local)

    assert paths.config_file == roaming / "CoveSensoryMCP/config.yaml"
    assert paths.jobs_dir == local / "CoveSensoryMCP/Cache/jobs"


def test_linux_paths_use_xdg_directories(tmp_path: Path) -> None:
    """Linux users may relocate persistent config and transient cache independently."""
    config_home = tmp_path / "xdg-config"
    cache_home = tmp_path / "xdg-cache"

    paths = AppPaths.for_system(
        "Linux",
        home=tmp_path,
        roaming=None,
        local=None,
        xdg_config=config_home,
        xdg_cache=cache_home,
    )

    assert paths.config_file == config_home / "cove-sensory-mcp/config.yaml"
    assert paths.jobs_dir == cache_home / "cove-sensory-mcp/jobs"


def test_linux_paths_fall_back_to_home_when_xdg_is_unset(tmp_path: Path) -> None:
    """A minimal Linux environment must work without optional XDG variables."""
    paths = AppPaths.for_system(
        "Linux",
        home=tmp_path,
        roaming=None,
        local=None,
        xdg_config=None,
        xdg_cache=None,
    )

    assert paths.config_file == tmp_path / ".config/cove-sensory-mcp/config.yaml"
    assert paths.jobs_dir == tmp_path / ".cache/cove-sensory-mcp/jobs"


def test_linux_paths_ignore_relative_xdg_directories(tmp_path: Path) -> None:
    """Relative XDG roots must not make storage depend on the process CWD."""
    paths = AppPaths.for_system(
        "Linux",
        home=tmp_path,
        roaming=None,
        local=None,
        xdg_config=Path("relative-config"),
        xdg_cache=Path("relative-cache"),
    )

    assert paths.config_file == tmp_path / ".config/cove-sensory-mcp/config.yaml"
    assert paths.jobs_dir == tmp_path / ".cache/cove-sensory-mcp/jobs"


def test_unsupported_platform_reports_a_public_configuration_error(tmp_path: Path) -> None:
    """Changing the fallback branch must not permit an unsupported platform."""
    with pytest.raises(SensoryError) as exc_info:
        AppPaths.for_system("Plan9", home=tmp_path, roaming=None, local=None)

    assert exc_info.value.code is ErrorCode.CONFIG_INVALID
    assert str(exc_info.value) == "Unsupported operating system."
    assert str(tmp_path) not in str(exc_info.value)
