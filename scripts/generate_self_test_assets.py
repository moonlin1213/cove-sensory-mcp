#!/usr/bin/env python3
"""Generate tiny, deterministic, project-owned capability-test media."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import subprocess
import tempfile
import wave
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageDraw, PngImagePlugin

RATE = 16_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(path: Path, seconds: float, sample: Callable[[float], float]) -> None:
    count = int(RATE * seconds)
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(RATE)
        frames = bytearray()
        for index in range(count):
            value = max(-1.0, min(1.0, float(sample(index / RATE))))
            frames.extend(struct.pack("<h", round(value * 22_000)))
        target.writeframes(frames)


def _beep_sample(t: float) -> float:
    """Three separated 880 Hz beeps for the hearing self-test."""
    for start in (0.15, 0.75, 1.35):
        local = t - start
        if 0 <= local < 0.25:
            envelope = min(local / 0.015, 1.0) * min((0.25 - local) / 0.025, 1.0)
            return 0.55 * envelope * math.sin(2 * math.pi * 880 * local)
    return 0.0


def _bell_sample(t: float) -> float:
    """Two separated synthetic bell chimes for the video-hearing self-test."""
    for start in (0.20, 1.10):
        local = t - start
        if 0 <= local < 0.45:
            envelope = min(local / 0.008, 1.0) * math.exp(-5.5 * local)
            return envelope * (
                0.42 * math.sin(2 * math.pi * 660 * local)
                + 0.24 * math.sin(2 * math.pi * 990 * local)
                + 0.14 * math.sin(2 * math.pi * 1_430 * local)
            )
    return 0.0


def _music_sample(t: float) -> float:
    """Four piano-like notes forming one unambiguous ascending scale."""
    notes = (261.626, 329.628, 391.995, 523.251)
    note_index = min(3, int(t / 0.75))
    local = t - note_index * 0.75
    tone = notes[note_index]
    envelope = min(local / 0.012, 1.0) * math.exp(-1.8 * local)
    return envelope * (
        0.48 * math.sin(2 * math.pi * tone * local)
        + 0.16 * math.sin(2 * math.pi * tone * 2 * local)
        + 0.08 * math.sin(2 * math.pi * tone * 3 * local)
    )


def _draw_shape() -> Image.Image:
    image = Image.new("RGB", (320, 180), (248, 248, 244))
    draw = ImageDraw.Draw(image)
    draw.polygon(((160, 45), (105, 135), (215, 135)), fill=(33, 102, 214))
    return image


def _draw_motion_frame(index: int) -> Image.Image:
    image = Image.new("RGB", (320, 180), (248, 248, 244))
    draw = ImageDraw.Draw(image)
    x = 42 + round((320 - 84) * index / 47)
    draw.ellipse((x - 24, 66, x + 24, 114), fill=(210, 52, 52))
    return image


def generate(output: Path, ffmpeg: str = "ffmpeg") -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    shape = output / "shape.png"
    pnginfo = PngImagePlugin.PngInfo()
    _draw_shape().save(shape, format="PNG", optimize=False, pnginfo=pnginfo)

    speech = output / "speech.wav"
    music = output / "music.wav"
    _write_wav(speech, 2.0, _beep_sample)
    _write_wav(music, 3.0, _music_sample)

    motion = output / "motion.mp4"
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise SystemExit(
            "A verified ffmpeg executable is required to generate motion.mp4"
        )
    with tempfile.TemporaryDirectory(prefix="cove-self-test-") as scratch:
        frames = Path(scratch)
        bell = frames / "bell.wav"
        _write_wav(bell, 2.0, _bell_sample)
        for index in range(48):
            _draw_motion_frame(index).save(
                frames / f"frame-{index:03d}.png", format="PNG", optimize=False
            )
        subprocess.run(
            [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                "24",
                "-i",
                str(frames / "frame-%03d.png"),
                "-i",
                str(bell),
                "-t",
                "2",
                "-map_metadata",
                "-1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "medium",
                "-crf",
                "28",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(motion),
            ],
            check=True,
        )

    descriptions = {
        "shape.png": ("image", "image/png", [["blue", "triangle"]]),
        "motion.mp4": (
            "video_visual",
            "video/mp4",
            [["red", "ball", "left", "right"], ["bell", "twice"]],
        ),
        "speech.wav": ("audio", "audio/wav", [["tone", "beep", "three"]]),
        "music.wav": ("music", "audio/wav", [["piano", "ascending", "scale"]]),
    }
    assets = [
        {
            "name": name,
            "modality": modality,
            "mime_type": mime,
            "sha256": _sha256(output / name),
            "expected_keyword_groups": groups,
            "license": "Apache-2.0",
        }
        for name, (modality, mime, groups) in descriptions.items()
    ]
    manifest: dict[str, object] = {"schema_version": 1, "assets": assets}
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    generate(args.output, args.ffmpeg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
