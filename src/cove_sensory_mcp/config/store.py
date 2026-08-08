"""Safe YAML persistence for strict, non-secret application configuration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from filelock import FileLock, Timeout
from pydantic import ValidationError

from cove_sensory_mcp.errors import ErrorCode, SensoryError

from .schema import AppConfig

_INVALID_CONFIG_MESSAGE = "The configuration file is invalid."
_UPDATE_FAILED_MESSAGE = "The configuration file could not be updated safely."

ConfigMutator = Callable[[AppConfig], AppConfig | None]


class ConfigStore:
    """Load and atomically replace one configuration file."""

    def __init__(self, path: Path, *, jobs_dir: Path | None = None) -> None:
        self._path = path
        self._jobs_dir = jobs_dir if jobs_dir is not None else path.parent / "jobs"
        self._lock_path = path.with_suffix(f"{path.suffix}.lock")

    @property
    def path(self) -> Path:
        """Return the configuration location without loading it."""
        return self._path

    @property
    def jobs_dir(self) -> Path:
        """Return the resolved root reserved for transient sensory jobs."""
        return self._jobs_dir

    def load(self) -> AppConfig:
        """Load valid YAML or return the fresh-install configuration."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(self._lock_path), timeout=5):
                return self._load_unlocked()
        except SensoryError:
            raise
        except (OSError, Timeout) as exc:
            raise SensoryError(
                ErrorCode.CONFIG_INVALID,
                _INVALID_CONFIG_MESSAGE,
            ) from exc

    def save(self, config: AppConfig) -> None:
        """Persist a validated config under the cross-process transaction lock."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(self._lock_path), timeout=5):
                validated = AppConfig.model_validate(config.model_dump(mode="python"))
                self._save_unlocked(validated)
        except SensoryError:
            raise
        except (OSError, Timeout, UnicodeError, ValidationError, yaml.YAMLError) as exc:
            raise SensoryError(
                ErrorCode.CONFIG_INVALID,
                _UPDATE_FAILED_MESSAGE,
            ) from exc

    def update(self, mutator: ConfigMutator) -> AppConfig:
        """Load, mutate, validate, and replace one latest snapshot under one short lock."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(self._lock_path), timeout=5):
                working = self._load_unlocked().model_copy(deep=True)
                replacement = mutator(working)
                candidate = working if replacement is None else replacement
                validated = AppConfig.model_validate(
                    candidate.model_dump(mode="python")
                )
                self._save_unlocked(validated)
                return validated.model_copy(deep=True)
        except SensoryError:
            raise
        except (OSError, Timeout, UnicodeError, ValidationError, yaml.YAMLError) as exc:
            raise SensoryError(
                ErrorCode.CONFIG_INVALID,
                _UPDATE_FAILED_MESSAGE,
            ) from exc

    def _load_unlocked(self) -> AppConfig:
        """Load while the caller owns the configuration lock."""
        if not self._path.exists():
            return AppConfig()
        try:
            document = yaml.safe_load(self._path.read_text(encoding="utf-8"))
            return AppConfig.model_validate(document)
        except (OSError, UnicodeError, ValidationError, yaml.YAMLError) as exc:
            raise SensoryError(
                ErrorCode.CONFIG_INVALID,
                _INVALID_CONFIG_MESSAGE,
            ) from exc

    def _save_unlocked(self, config: AppConfig) -> None:
        """Replace the destination while the caller owns the configuration lock."""
        payload = yaml.safe_dump(
            config.model_dump(mode="json", exclude_none=True),
            allow_unicode=True,
            sort_keys=False,
        )
        temporary = self._path.with_suffix(".yaml.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self._path)
