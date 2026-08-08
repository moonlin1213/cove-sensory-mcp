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


def _speech_sample(t: float) -> float:
    """Small formant-style synthetic phrase followed by an 880 Hz confirmation tone."""
    if t >= 1.55:
        return 0.45 * math.sin(2 * math.pi * 880 * t)
    syllable = min(4, int(t / 0.31))
    local = t - syllable * 0.31
    if local > 0.24:
        return 0.0
    fundamental = (120, 145, 165, 195, 220)[syllable]
    envelope = min(local / 0.025, 1.0) * min((0.24 - local) / 0.04, 1.0)
    return envelope * (
        0.42 * math.sin(2 * math.pi * fundamental * t)
        + 0.18 * math.sin(2 * math.pi * fundamental * 3 * t)
        + 0.12 * math.sin(2 * math.pi * fundamental * 7 * t)
    )


def _music_sample(t: float) -> float:
    beat = int(t * 4) if t < 1.5 else int((t - 1.5) * 8)
    pulse = 1.0 if (t * (4 if t < 1.5 else 8)) % 1 < 0.22 else 0.45
    notes = (440.0, 554.365, 659.255, 554.365)
    tone = notes[beat % len(notes)]
    return pulse * (
        0.38 * math.sin(2 * math.pi * tone * t)
        + 0.22 * math.sin(2 * math.pi * 330 * t)
    )


def _draw_frame(index: int, *, text: bool) -> Image.Image:
    image = Image.new("RGB", (320, 180), (248, 248, 244))
    draw = ImageDraw.Draw(image)
    x = 42 + round((320 - 84) * index / 47)
    draw.ellipse((x - 24, 66, x + 24, 114), fill=(33, 102, 214))
    if text:
        draw.text((135, 20), "COVE 3", fill=(20, 20, 20), stroke_width=1)
    return image


def generate(output: Path, ffmpeg: str = "ffmpeg") -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    shape = output / "shape.png"
    pnginfo = PngImagePlugin.PngInfo()
    _draw_frame(0, text=True).save(shape, format="PNG", optimize=False, pnginfo=pnginfo)

    speech = output / "speech.wav"
    music = output / "music.wav"
    _write_wav(speech, 2.0, _speech_sample)
    _write_wav(music, 3.0, _music_sample)

    motion = output / "motion.mp4"
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise SystemExit("A verified ffmpeg executable is required to generate motion.mp4")
    with tempfile.TemporaryDirectory(prefix="cove-self-test-") as scratch:
        frames = Path(scratch)
        for index in range(48):
            _draw_frame(index, text=index >= 24).save(
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
                str(speech),
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
        "shape.png": ("image", "image/png", [["blue", "circle"], ["COVE", "3"]]),
        "motion.mp4": (
            "video_visual",
            "video/mp4",
            [["blue", "circle", "left", "right"], ["3"]],
        ),
        "speech.wav": ("audio", "audio/wav", [["tone", "880"]]),
        "music.wav": ("music", "audio/wav", [["tone", "rhythm", "change"]]),
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
