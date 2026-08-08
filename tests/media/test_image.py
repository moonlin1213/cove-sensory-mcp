from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from cove_sensory_mcp.media.image import prepare_image
from cove_sensory_mcp.media.types import ResolvedSource
from cove_sensory_mcp.models import DetailLevel
from cove_sensory_mcp.providers.base import ProviderMediaLimits


class Workspace:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.count = 0

    def new_artifact(self, name: str, suffix: str) -> Path:
        self.count += 1
        return self.path / f"{name}_{self.count}{suffix}"


def _source(path: Path) -> ResolvedSource:
    return ResolvedSource(
        path=path,
        source_kind="local",
        display_name=path.name,
        cleanup_required=False,
        original_size=path.stat().st_size,
    )


@pytest.mark.asyncio
async def test_exif_rotation_and_downscale_do_not_modify_source(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    image = Image.new("RGB", (80, 40), "red")
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)
    before = path.read_bytes()
    outputs = await prepare_image(
        _source(path),
        "details",
        DetailLevel.AUTO,
        ProviderMediaLimits(1_000_000, frozenset({"image/jpeg"}), None, 50),
        Workspace(tmp_path),
    )
    with Image.open(outputs[0].path) as prepared:
        assert prepared.width <= 50 and prepared.height <= 50
        assert prepared.height > prepared.width
    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_transparency_is_preserved_for_png_and_flattened_for_jpeg(
    tmp_path: Path,
) -> None:
    path = tmp_path / "transparent.png"
    Image.new("RGBA", (20, 20), (255, 0, 0, 0)).save(path)
    png = await prepare_image(
        _source(path),
        "",
        DetailLevel.AUTO,
        ProviderMediaLimits(1_000_000, frozenset({"image/png"}), None, 100),
        Workspace(tmp_path),
    )
    jpg = await prepare_image(
        _source(path),
        "",
        DetailLevel.AUTO,
        ProviderMediaLimits(1_000_000, frozenset({"image/jpeg"}), None, 100),
        Workspace(tmp_path),
    )
    with Image.open(png[0].path) as value:
        assert value.mode == "RGBA"
    with Image.open(jpg[0].path) as value:
        assert value.mode == "RGB"


@pytest.mark.asyncio
async def test_tall_ocr_image_is_split_into_ordered_overlapping_tiles(
    tmp_path: Path,
) -> None:
    path = tmp_path / "long.png"
    Image.new("RGB", (100, 1000), "white").save(path)
    outputs = await prepare_image(
        _source(path),
        "OCR text",
        DetailLevel.DETAILED,
        ProviderMediaLimits(1_000_000, frozenset({"image/png"}), None, 200),
        Workspace(tmp_path),
    )
    assert len(outputs) > 1
    assert [item.sequence_index for item in outputs] == list(range(len(outputs)))
    assert all(item.path.parent == tmp_path for item in outputs)
