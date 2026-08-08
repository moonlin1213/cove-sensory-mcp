from __future__ import annotations

from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.reports.merge import merge_video_observations
from cove_sensory_mcp.reports.schemas import ObservationEnvelope, ObservationSegment


def _observation(modality: Modality, summary: str, segments: list[tuple[float, float, str]]) -> ObservationEnvelope:
    return ObservationEnvelope(
        modality=modality,
        summary=summary,
        segments=[ObservationSegment(start_seconds=start, end_seconds=end, text=text) for start, end, text in segments],
        confidence="medium",
    )


def test_merge_uses_union_boundaries_without_inventing_causality() -> None:
    visual = _observation(Modality.VIDEO_VISUAL, "visual summary", [(30, 45, "person enters"), (46, 62, "door closes")])
    audio = _observation(Modality.VIDEO_AUDIO, "audio summary", [(30, 38, "music"), (39, 53, "speech"), (54, 62, "silence")])
    report = merge_video_observations(visual, audio, 62)
    boundaries = [(item.start_seconds, item.end_seconds) for item in report.timeline]
    assert boundaries == sorted(boundaries)
    assert (30, 38) in boundaries and (54, 62) in boundaries
    assert report.summary == "视觉：visual summary 听觉：audio summary"
    assert "because" not in report.summary.lower()


def test_partial_merge_preserves_successful_side() -> None:
    visual = _observation(Modality.VIDEO_VISUAL, "visible", [])
    report = merge_video_observations(visual, None, 10)
    assert report.status == "partial"
    assert report.coverage.visual is True
    assert report.coverage.audio is False
    assert report.visual_summary == "visible"
