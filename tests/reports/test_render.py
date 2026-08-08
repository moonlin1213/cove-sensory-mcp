from __future__ import annotations

from cove_sensory_mcp.reports.render import render_sensory_text
from cove_sensory_mcp.reports.schemas import Coverage, ProviderUsage, VideoSensoryReport


def test_render_contains_same_public_evidence_without_private_sources() -> None:
    report = VideoSensoryReport(
        request_id="sense_test",
        status="partial",
        coverage=Coverage(visual=True, audio=False, visual_provider="eye"),
        summary="视觉：a",
        visual_summary="a",
        audio_summary="",
        providers=[ProviderUsage(provider_id="eye", model="vision-v1")],
        warnings=["NO_AUDIO_TRACK"],
        requested_start_seconds=30,
        requested_end_seconds=60,
    )
    text = render_sensory_text(report)
    assert "视频" in text and "30.000–60.000" in text
    assert "视觉：a" in text
    assert "eye" in text and "vision-v1" in text
    assert "NO_AUDIO_TRACK" in text
    assert "/Users/" not in text and "https://" not in text
