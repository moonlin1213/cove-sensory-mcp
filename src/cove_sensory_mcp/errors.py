"""Privacy-safe public error contracts for Cove Sensory MCP."""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """Stable, machine-readable error codes exposed by the MCP server."""

    SETUP_REQUIRED = "SETUP_REQUIRED"
    CAPABILITY_NOT_CONFIGURED = "CAPABILITY_NOT_CONFIGURED"
    CAPABILITY_NOT_VERIFIED = "CAPABILITY_NOT_VERIFIED"
    PATH_NOT_ALLOWED = "PATH_NOT_ALLOWED"
    SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    MEDIA_TOO_LARGE = "MEDIA_TOO_LARGE"
    LONG_MEDIA_CONFIRMATION_REQUIRED = "LONG_MEDIA_CONFIRMATION_REQUIRED"
    MEDIA_RUNTIME_REQUIRED = "MEDIA_RUNTIME_REQUIRED"
    DOWNLOAD_BLOCKED = "DOWNLOAD_BLOCKED"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    PROVIDER_AUTH_FAILED = "PROVIDER_AUTH_FAILED"
    PROVIDER_CAPABILITY_REJECTED = "PROVIDER_CAPABILITY_REJECTED"
    PROVIDER_SAFETY_REJECTED = "PROVIDER_SAFETY_REJECTED"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    FALLBACK_NOT_AUTHORIZED = "FALLBACK_NOT_AUTHORIZED"
    PARTIAL_PERCEPTION = "PARTIAL_PERCEPTION"
    TEMP_CLEANUP_FAILED = "TEMP_CLEANUP_FAILED"


class SensoryError(Exception):
    """An error whose public fields are safe to return to an MCP client."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
        setup_command: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.setup_command = setup_command
        self.cause = cause

    def __str__(self) -> str:
        """Return only the deliberate public message, never internal context."""
        return self.message


def error_result(error: SensoryError) -> dict[str, object]:
    """Convert a :class:`SensoryError` into the stable public error envelope."""
    payload: dict[str, object] = {
        "code": error.code.value,
        "message": error.message,
        "retryable": error.retryable,
    }
    if error.setup_command is not None:
        payload["setup_command"] = error.setup_command
    return {"status": "error", "error": payload}
