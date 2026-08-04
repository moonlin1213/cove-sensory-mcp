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
        raise SensoryError(ErrorCode.CONFIG_INVALID, "Unsupported operating system.")
