import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.processing.image_search_client import BraveImageClient


class FakeResponse:
    def __init__(self, status=200, json_data=None, content=b"", headers=None):
        self.status = status
        self._json_data = json_data or {}
        self._content = content
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._json_data

    async def read(self):
        return self._content


class FakeSession:
    def __init__(self):
        self.closed = False
        self.get_calls = []

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        if "images/search" in url:
            return FakeResponse(
                status=200,
                json_data={"results": [{"url": "https://example.com/test.jpg"}]},
            )

        return FakeResponse(
            status=200,
            content=b"image-bytes",
            headers={"Content-Type": "image/jpeg"},
        )

    async def close(self):
        self.closed = True


def _mock_config(tmp_path):
    config = Mock()
    config.paths = Mock()
    config.paths.cache_folder = tmp_path
    return config


@pytest.mark.asyncio
async def test_reuses_single_session_with_context_manager(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    created_sessions = []

    def _session_factory(*args, **kwargs):
        session = FakeSession()
        created_sessions.append(session)
        return session

    with patch(
        "src.processing.image_search_client.get_config",
        return_value=_mock_config(tmp_path),
    ):
        with patch(
            "src.processing.image_search_client.aiohttp.ClientSession",
            side_effect=_session_factory,
        ):
            async with BraveImageClient() as client:
                await client.search_images("query", count=1)
                await client.download_image("https://example.com/test.jpg", "query")

    assert len(created_sessions) == 1
    assert len(created_sessions[0].get_calls) == 2
    assert created_sessions[0].closed is True


@pytest.mark.asyncio
async def test_get_broll_images_concurrent_mapping(tmp_path):
    with patch(
        "src.processing.image_search_client.get_config",
        return_value=_mock_config(tmp_path),
    ):
        client = BraveImageClient(max_concurrent_downloads=4)

    async def _fake_search(query, count=1):
        return [{"url": f"https://example.com/{query}.jpg"}]

    async def _fake_download(url, query):
        await asyncio.sleep(0.01)
        if query == "q2":
            return None
        return Path(f"/tmp/{query}.jpg")

    client.search_images = _fake_search
    client.download_image = _fake_download

    result = await client.get_broll_images(["q1", "q2", "q3"], max_per_query=1)

    assert result == {
        "q1": [Path("/tmp/q1.jpg")],
        "q3": [Path("/tmp/q3.jpg")],
    }


@pytest.mark.asyncio
async def test_get_broll_images_respects_semaphore_limit(tmp_path):
    with patch(
        "src.processing.image_search_client.get_config",
        return_value=_mock_config(tmp_path),
    ):
        client = BraveImageClient(max_concurrent_downloads=2)

    async def _fake_search(query, count=1):
        return [{"url": f"https://example.com/{query}.jpg"}]

    active_downloads = 0
    max_active_downloads = 0

    async def _fake_download(url, query):
        nonlocal active_downloads, max_active_downloads
        active_downloads += 1
        max_active_downloads = max(max_active_downloads, active_downloads)
        await asyncio.sleep(0.03)
        active_downloads -= 1
        return Path(f"/tmp/{query}.jpg")

    client.search_images = _fake_search
    client.download_image = _fake_download

    queries = [f"q{i}" for i in range(6)]
    result = await client.get_broll_images(queries, max_per_query=1)

    assert len(result) == 6
    assert max_active_downloads == 2
