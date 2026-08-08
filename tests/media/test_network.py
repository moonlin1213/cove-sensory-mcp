from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.media.network import (
    MediaDownloader,
    NetworkLimits,
    NetworkPolicy,
)


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/example.jpg",
        "ftp://example.com/file.mp3",
        "http://localhost/file.jpg",
        "http://127.0.0.1/file.jpg",
        "http://[::1]/file.jpg",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.1/file.jpg",
        "http://172.16.0.1/file.jpg",
        "http://192.168.0.1/file.jpg",
        "https://user:password@example.com/file.jpg",
    ],
)
def test_network_policy_blocks_unsafe_targets(url: str) -> None:
    with pytest.raises(SensoryError) as caught:
        NetworkPolicy(lambda _: ["93.184.216.34"]).validate_url(url)
    assert caught.value.code is ErrorCode.DOWNLOAD_BLOCKED


def test_dns_resolution_to_private_address_is_blocked() -> None:
    with pytest.raises(SensoryError) as caught:
        NetworkPolicy(lambda _: ["10.0.0.5"]).validate_url("https://public.example/a.jpg")
    assert caught.value.code is ErrorCode.DOWNLOAD_BLOCKED


@pytest.mark.asyncio
async def test_download_revalidates_redirect_and_cleans_partial(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    downloader = MediaDownloader(
        NetworkPolicy(lambda _: ["93.184.216.34"]),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(SensoryError) as caught:
        await downloader.download(
            "https://public.example/a.jpg", tmp_path, NetworkLimits(1024, 2, 2)
        )
    assert caught.value.code is ErrorCode.DOWNLOAD_BLOCKED
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("headers", "body"),
    [
        ({"content-type": "image/jpeg", "content-length": "999"}, b"\xff\xd8\xffok"),
        ({"content-type": "image/jpeg"}, b"MZ" + b"x" * 20),
        ({"content-type": "application/octet-stream"}, b"\xff\xd8\xffok"),
    ],
)
@pytest.mark.asyncio
async def test_download_rejects_size_and_media_mismatch(
    tmp_path: Path, headers: dict[str, str], body: bytes
) -> None:
    downloader = MediaDownloader(
        NetworkPolicy(lambda _: ["93.184.216.34"]),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, headers=headers, content=body))
        ),
    )
    with pytest.raises(SensoryError):
        await downloader.download(
            "https://public.example/a.jpg", tmp_path, NetworkLimits(16, 2, 1)
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_valid_public_image_download_is_scoped(tmp_path: Path) -> None:
    downloader = MediaDownloader(
        NetworkPolicy(lambda _: ["93.184.216.34"]),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    headers={"content-type": "image/jpeg"},
                    content=b"\xff\xd8\xffvalid-image",
                )
            )
        ),
    )
    result = await downloader.download(
        "https://public.example/photo.jpg", tmp_path, NetworkLimits(1024, 2, 1)
    )
    assert result.path.parent == tmp_path.resolve()
    assert result.mime_type == "image/jpeg"
    assert result.cleanup_required is True
