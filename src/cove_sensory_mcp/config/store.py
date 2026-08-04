"""Safe YAML persistence for strict, non-secret application configuration."""

from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from cove_sensory_mcp.errors import ErrorCode, SensoryError

from .schema import AppConfig


class ConfigStore:
    """Load and atomically replace one configuration file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """Return the configuration location without loading it."""
        return self._path

    def load(self) -> AppConfig:
        """Load valid YAML or return the fresh-install configuration."""
        if not self._path.exists():
            return AppConfig()
        try:
            document = yaml.safe_load(self._path.read_text(encoding="utf-8"))
            return AppConfig.model_validate(document)
        except (OSError, UnicodeError, ValidationError, yaml.YAMLError) as exc:
            raise SensoryError(
                ErrorCode.CONFIG_INVALID,
                "The configuration file is invalid.",
            ) from exc

    def save(self, config: AppConfig) -> None:
        """Persist a validated config by replacing the destination in its directory."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = yaml.safe_dump(
            config.model_dump(mode="json", exclude_none=True),
            allow_unicode=True,
            sort_keys=False,
        )
        temporary = self._path.with_suffix(".yaml.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self._path)
