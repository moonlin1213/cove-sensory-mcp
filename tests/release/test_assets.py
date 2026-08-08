from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest
from PIL import Image

from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.verification.assets import SelfTestAssetStore

ROOT = Path(__file__).parents[2]
ASSETS = ROOT / "src/cove_sensory_mcp/assets/self_test"


def _samples(path: Path) -> tuple[int, list[int]]:
    with wave.open(str(path)) as source:
        rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    return rate, list(struct.unpack(f"<{len(frames) // 2}h", frames))


def _event_count(samples: list[int], rate: int) -> int:
    size = rate // 20
    active = [
        math.sqrt(sum(value * value for value in samples[start : start + size]) / size)
        > 2_000
        for start in range(0, len(samples) - size + 1, size)
    ]
    return sum(
        state and (index == 0 or not active[index - 1])
        for index, state in enumerate(active)
    )


def _tone_power(samples: list[int], rate: int, frequency: float) -> float:
    coefficient = 2 * math.cos(2 * math.pi * frequency / rate)
    previous = 0.0
    previous_two = 0.0
    for sample in samples:
        current = sample + coefficient * previous - previous_two
        previous_two, previous = previous, current
    return previous_two**2 + previous**2 - coefficient * previous * previous_two


def _is_red(pixel: tuple[int, int, int]) -> bool:
    return pixel[0] > 160 and pixel[1] < 100 and pixel[2] < 100


def test_packaged_assets_match_manifest_and_are_usable() -> None:
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    assert {item["name"] for item in manifest["assets"]} == {
        "shape.png",
        "motion.mp4",
        "speech.wav",
        "music.wav",
    }
    for item in manifest["assets"]:
        path = ASSETS / item["name"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        assert item["license"] == "Apache-2.0"
    image = Image.open(ASSETS / "shape.png")
    assert image.size == (320, 180) and not image.getexif()
    with wave.open(str(ASSETS / "speech.wav")) as audio:
        assert audio.getframerate() == 16_000 and audio.getnframes() == 32_000
    with wave.open(str(ASSETS / "music.wav")) as music:
        assert music.getnframes() == 48_000
    assert (
        SelfTestAssetStore.packaged().get(Modality.IMAGE).path
        == (ASSETS / "shape.png").resolve()
    )


def test_packaged_assets_match_the_verifiers_observable_facts() -> None:
    """Changing an asset without its verifier facts must fail before release."""
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        item["name"]: item["expected_keyword_groups"] for item in manifest["assets"]
    }
    assert expected == {
        "shape.png": [["blue", "triangle"]],
        "motion.mp4": [["red", "ball", "left", "right"], ["bell", "twice"]],
        "speech.wav": [["tone", "beep", "three"]],
        "music.wav": [["piano", "ascending", "scale"]],
    }

    image = Image.open(ASSETS / "shape.png").convert("RGB")
    blue = [
        (x, y)
        for y in range(image.height)
        for x in range(image.width)
        if image.getpixel((x, y)) == (33, 102, 214)
    ]
    assert 145 <= sum(x for x, _ in blue) / len(blue) <= 175
    row_widths = [sum(1 for x, y in blue if y == row) for row in range(image.height)]
    occupied = [width for width in row_widths if width]
    assert occupied[0] < occupied[len(occupied) // 2] < occupied[-1]

    rate, beeps = _samples(ASSETS / "speech.wav")
    assert _event_count(beeps, rate) == 3

    rate, music = _samples(ASSETS / "music.wav")
    notes = (261.626, 329.628, 391.995, 523.251)
    winners: list[int] = []
    for index in range(4):
        start = round((index * 0.75 + 0.10) * rate)
        stop = round((index * 0.75 + 0.60) * rate)
        powers = [_tone_power(music[start:stop], rate, note) for note in notes]
        winners.append(powers.index(max(powers)))
    assert winners == [0, 1, 2, 3]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="Inspecting the packaged self-test video requires FFmpeg.",
)
def test_packaged_video_contains_the_verified_red_ball_and_two_bells() -> None:
    """Changing the video or embedded audio independently must fail before release."""
    video = ASSETS / "motion.mp4"
    centers: list[float] = []
    for timestamp in (0.05, 1.90):
        frame = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                str(timestamp),
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ],
            check=True,
            capture_output=True,
        ).stdout
        image = Image.open(io.BytesIO(frame)).convert("RGB")
        red_x = [
            x
            for y in range(image.height)
            for x in range(image.width)
            if _is_red(image.getpixel((x, y)))
        ]
        assert red_x
        centers.append(sum(red_x) / len(red_x))
    assert centers[0] < 80 and centers[1] > 240

    pcm = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-vn",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    samples = list(struct.unpack(f"<{len(pcm) // 2}h", pcm))
    assert _event_count(samples, 16_000) == 2


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="Regenerating the committed test video requires a user-supplied FFmpeg.",
)
def test_generator_is_reproducible_with_same_runtime(tmp_path: Path) -> None:
    first, second = tmp_path / "one", tmp_path / "two"
    for output in (first, second):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/generate_self_test_assets.py"),
                "--output",
                str(output),
            ],
            check=True,
        )
    one = json.loads((first / "manifest.json").read_text())
    two = json.loads((second / "manifest.json").read_text())
    assert one == two
