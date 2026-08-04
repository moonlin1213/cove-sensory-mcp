from __future__ import annotations

from cove_sensory_mcp.errors import ErrorCode, SensoryError, error_result


def test_error_result_has_stable_public_shape() -> None:
    result = error_result(
        SensoryError(
            ErrorCode.SETUP_REQUIRED,
            "No provider is configured.",
            retryable=False,
            setup_command="cove-sensory-mcp configure",
        )
    )

    assert result == {
        "status": "error",
        "error": {
            "code": "SETUP_REQUIRED",
            "message": "No provider is configured.",
            "retryable": False,
            "setup_command": "cove-sensory-mcp configure",
        },
    }


def test_error_string_returns_only_public_message() -> None:
    error = SensoryError(
        ErrorCode.PROVIDER_TIMEOUT,
        "The provider did not respond in time.",
        cause=RuntimeError("internal transport detail"),
    )

    assert str(error) == "The provider did not respond in time."
    assert error_result(error) == {
        "status": "error",
        "error": {
            "code": "PROVIDER_TIMEOUT",
            "message": "The provider did not respond in time.",
            "retryable": False,
        },
    }


def test_error_code_members_match_the_public_contract() -> None:
    assert [code.value for code in ErrorCode] == [
        "SETUP_REQUIRED",
        "CAPABILITY_NOT_CONFIGURED",
        "CAPABILITY_NOT_VERIFIED",
        "PATH_NOT_ALLOWED",
        "SOURCE_NOT_FOUND",
        "UNSUPPORTED_MEDIA_TYPE",
        "MEDIA_TOO_LARGE",
        "LONG_MEDIA_CONFIRMATION_REQUIRED",
        "MEDIA_RUNTIME_REQUIRED",
        "DOWNLOAD_BLOCKED",
        "DOWNLOAD_FAILED",
        "PROVIDER_AUTH_FAILED",
        "PROVIDER_CAPABILITY_REJECTED",
        "PROVIDER_SAFETY_REJECTED",
        "PROVIDER_TIMEOUT",
        "PROVIDER_UNAVAILABLE",
        "FALLBACK_NOT_AUTHORIZED",
        "PARTIAL_PERCEPTION",
        "TEMP_CLEANUP_FAILED",
    ]
