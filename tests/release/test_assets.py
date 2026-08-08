from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import wave
from pathlib import Path

from PIL import Image

from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.verification.assets import SelfTestAssetStore


ROOT = Path(__file__).parents[2]
ASSETS = ROOT / "src/cove_sensory_mcp/assets/self_test"


def test_packaged_assets_match_manifest_and_are_usable() -> None:
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    assert {item["name"] for item in manifest["assets"]} == {
        "shape.png", "motion.mp4", "speech.wav", "music.wav"
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
    assert SelfTestAssetStore.packaged().get(Modality.IMAGE).path == (
        ASSETS / "shape.png"
    ).resolve()


def test_generator_is_reproducible_with_same_runtime(tmp_path: Path) -> None:
    first, second = tmp_path / "one", tmp_path / "two"
    for output in (first, second):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/generate_self_test_assets.py"), "--output", str(output)],
            check=True,
        )
    one = json.loads((first / "manifest.json").read_text())
    two = json.loads((second / "manifest.json").read_text())
    assert one == two
