"""Platform-specific storage locations with no implicit POSIX fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cove_sensory_mcp.errors import ErrorCode, SensoryError


@dataclass(frozen=True, slots=True)
class AppPaths:
    """The local configuration and transient-job locations for one system."""

    config_file: Path
    jobs_dir: Path

    @classmethod
    def for_system(
        cls,
        system: str,
        home: Path,
        roaming: Path | None,
        local: Path | None,
        xdg_config: Path | None = None,
        xdg_cache: Path | None = None,
    ) -> AppPaths:
        """Return the public storage contract for a supported operating system."""
        if system == "Darwin":
            return cls(
                config_file=home / "Library/Application Support/cove-sensory-mcp/config.yaml",
                jobs_dir=home / "Library/Caches/cove-sensory-mcp/jobs",
            )
        if system == "Windows":
            if roaming is None or local is None:
                raise SensoryError(
                    ErrorCode.CONFIG_INVALID,
                    "Windows application directories are unavailable.",
                )
            return cls(
                config_file=roaming / "CoveSensoryMCP/config.yaml",
                jobs_dir=local / "CoveSensoryMCP/Cache/jobs",
            )
        if system == "Linux":
            config_home = (
                xdg_config
                if xdg_config is not None and xdg_config.is_absolute()
                else home / ".config"
            )
            cache_home = (
                xdg_cache
                if xdg_cache is not None and xdg_cache.is_absolute()
                else home / ".cache"
            )
            return cls(
                config_file=config_home / "cove-sensory-mcp/config.yaml",
                jobs_dir=cache_home / "cove-sensory-mcp/jobs",
            )
        raise SensoryError(ErrorCode.CONFIG_INVALID, "Unsupported operating system.")
