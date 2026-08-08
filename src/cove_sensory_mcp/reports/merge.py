"""Deterministic visual/audio interval union without new inference."""

from __future__ import annotations

from itertools import pairwise

from cove_sensory_mcp.reports.schemas import (
    Coverage,
    ObservationEnvelope,
    TimelineEvidence,
    VideoSensoryReport,
)


def _overlaps(start: float, end: float, item_start: float, item_end: float) -> bool:
    return item_start < end and item_end > start


def merge_video_observations(
    visual: ObservationEnvelope | None,
    audio: ObservationEnvelope | None,
    duration: float,
) -> VideoSensoryReport:
    """Align existing evidence by boundary union; never synthesize causal prose."""
    boundaries: set[float] = set()
    for observation in (visual, audio):
        if observation is None:
            continue
        for segment in observation.segments:
            boundaries.add(max(0, min(duration, segment.start_seconds)))
            boundaries.add(max(0, min(duration, segment.end_seconds)))
        for transcript in observation.transcript:
            boundaries.add(max(0, min(duration, transcript.start_seconds)))
            boundaries.add(max(0, min(duration, transcript.end_seconds)))
    ordered = sorted(boundaries)
    timeline: list[TimelineEvidence] = []
    for start, end in pairwise(ordered):
        if end <= start:
            continue
        visual_items = [
            item.text
            for item in visual.segments if _overlaps(start, end, item.start_seconds, item.end_seconds)
        ] if visual else []
        audio_items = [
            item.text
            for item in audio.segments if _overlaps(start, end, item.start_seconds, item.end_seconds)
        ] if audio else []
        transcripts = [
            item.text
            for item in audio.transcript if _overlaps(start, end, item.start_seconds, item.end_seconds)
        ] if audio else []
        if visual_items or audio_items or transcripts:
            timeline.append(
                TimelineEvidence(
                    start_seconds=start,
                    end_seconds=end,
                    visual=visual_items,
                    audio=audio_items,
                    transcript=transcripts,
                )
            )
    pieces: list[str] = []
    if visual is not None:
        pieces.append(f"视觉：{visual.summary}")
    if audio is not None:
        pieces.append(f"听觉：{audio.summary}")
    warnings = [warning.code for observation in (visual, audio) if observation for warning in observation.warnings]
    return VideoSensoryReport(
        status="completed" if visual is not None and audio is not None else "partial",
        coverage=Coverage(visual=visual is not None, audio=audio is not None),
        summary=" ".join(pieces),
        visual_summary=visual.summary if visual else "",
        audio_summary=audio.summary if audio else "",
        warnings=warnings,
        timeline=timeline,
    )
