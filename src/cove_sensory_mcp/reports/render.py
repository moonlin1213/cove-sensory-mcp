"""Compact text rendering from the same structured sensory evidence."""

from __future__ import annotations

from .schemas import SensoryToolResult, VideoSensoryReport

_LABELS = {"image": "图片", "video": "视频", "audio": "音频", "music": "音乐"}


def render_sensory_text(report: SensoryToolResult) -> str:
    """Render only fields already present in the validated public report."""
    start = report.requested_start_seconds
    end = report.requested_end_seconds
    range_text = ""
    if start is not None or end is not None:
        range_text = f"｜{(start or 0):.3f}–{end:.3f}" if end is not None else f"｜{(start or 0):.3f}–结束"
    lines = [f"【感官观察｜{_LABELS[report.modality]}{range_text}】", "", report.summary]
    if isinstance(report, VideoSensoryReport):
        if report.visual_summary:
            lines.append(f"视觉：{report.visual_summary}")
        if report.audio_summary:
            lines.append(f"听觉：{report.audio_summary}")
        if report.timeline:
            lines.extend(("", "关键时间点："))
            for item in report.timeline:
                evidence: list[str] = []
                if item.visual:
                    evidence.append("画面" + "；".join(item.visual))
                if item.audio:
                    evidence.append("声音" + "；".join(item.audio))
                if item.transcript:
                    evidence.append("转写" + "；".join(item.transcript))
                lines.append(f"- {item.start_seconds:.3f}–{item.end_seconds:.3f}：{'；'.join(evidence)}")
    elif report.segments:
        lines.extend(("", "关键时间点："))
        lines.extend(
            f"- {item.start_seconds:.3f}–{item.end_seconds:.3f}：{item.text}"
            for item in report.segments
        )
    if report.providers:
        lines.extend(("", "Provider："))
        lines.extend(f"- {item.provider_id} / {item.model}" for item in report.providers)
    if report.warnings:
        lines.extend(("", "不确定项：" + "；".join(report.warnings)))
    return "\n".join(lines)
