from __future__ import annotations

import pytest
from pydantic import ValidationError

from cove_sensory_mcp.models import (
    CapabilityStatus,
    DetailLevel,
    Modality,
    ProviderRef,
    RouteConfig,
    SensoryStatus,
)


def test_disabled_capability_requires_reason() -> None:
    capability = CapabilityStatus(
        modality=Modality.AUDIO,
        enabled=False,
        verified=False,
        reason="No hearing provider configured",
    )

    assert capability.reason == "No hearing provider configured"


def test_disabled_capability_cannot_be_verified() -> None:
    with pytest.raises(ValidationError, match="cannot be verified"):
        CapabilityStatus(
            modality=Modality.AUDIO,
            enabled=False,
            verified=True,
            reason="No hearing provider configured",
        )


def test_disabled_capability_requires_a_nonempty_reason() -> None:
    with pytest.raises(ValidationError, match="requires a reason"):
        CapabilityStatus(
            modality=Modality.IMAGE,
            enabled=False,
            verified=False,
        )


def test_public_enum_values_match_the_tool_schema() -> None:
    assert [modality.value for modality in Modality] == [
        "image",
        "video_visual",
        "video_audio",
        "audio",
        "music",
    ]
    assert [detail.value for detail in DetailLevel] == ["auto", "quick", "detailed"]


def test_status_serializes_capabilities_by_modality() -> None:
    status = SensoryStatus(
        ready=True,
        version="1.0.0",
        capabilities={
            Modality.IMAGE: CapabilityStatus(
                modality=Modality.IMAGE,
                enabled=True,
                verified=True,
                provider="minimax-m3",
                mode="native_video",
            )
        },
    )

    assert status.model_dump(mode="json") == {
        "ready": True,
        "version": "1.0.0",
        "capabilities": {
            "image": {
                "modality": "image",
                "enabled": True,
                "verified": True,
                "provider": "minimax-m3",
                "mode": "native_video",
                "reason": None,
            }
        },
    }


def test_route_fallback_keeps_its_explicit_authorization() -> None:
    route = RouteConfig(
        primary="minimax",
        fallbacks=[ProviderRef(provider="gemini", authorized=True)],
    )

    assert route.model_dump() == {
        "primary": "minimax",
        "fallbacks": [{"provider": "gemini", "authorized": True}],
    }
