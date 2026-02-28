"""
Brave Image Search Client for B-Roll retrieval.
Downloads relevant images for video segments based on AI search queries.
"""

import logging
import os
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Optional, Dict, Any
import hashlib

from src.config.settings import get_config


class BraveImageClient:
    """
    Client for searching and downloading images from Brave Search API.
    Used for dynamic B-roll injection based on AI-generated search queries.
    """

    DEFAULT_MAX_CONCURRENT_DOWNLOADS = 6

    def __init__(
        self, max_concurrent_downloads: int = DEFAULT_MAX_CONCURRENT_DOWNLOADS
    ):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        self.base_url = "https://api.search.brave.com/res/v1/images/search"
        self.max_concurrent_downloads = max(1, max_concurrent_downloads)

        self._session: Optional[aiohttp.ClientSession] = None
        self._search_timeout = aiohttp.ClientTimeout(total=15)
        self._download_timeout = aiohttp.ClientTimeout(total=30)

        # Cache directory for downloaded images
        self.cache_dir = self.config.paths.cache_folder / "broll_images"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"BraveImageClient initialized. Cache: {self.cache_dir}")

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _get_cache_path(self, query: str) -> Path:
        """Generate cache filename from query"""
        # Create a hash of the query for the filename
        query_hash = hashlib.md5(query.encode()).hexdigest()[:12]
        return self.cache_dir / f"{query_hash}.jpg"

    async def search_images(self, query: str, count: int = 3) -> List[Dict[str, Any]]:
        """
        Search for images using Brave API.

        Args:
            query: Search query string
            count: Number of results to return

        Returns:
            List of image result dictionaries with URL, title, etc.
        """
        if not self.api_key:
            self.logger.warning("BRAVE_SEARCH_API_KEY not set")
            return []

        headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}

        try:
            session = await self._ensure_session()
            async with session.get(
                self.base_url,
                headers=headers,
                params={"q": query, "count": count},
                timeout=self._search_timeout,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("results", [])
                    self.logger.info(f"Found {len(results)} images for query: {query}")
                    return results
                else:
                    self.logger.warning(f"Brave image search failed: {response.status}")
                    return []
        except Exception as e:
            self.logger.error(f"Image search error for '{query}': {e}")
            return []

    async def download_image(self, url: str, query: str) -> Optional[Path]:
        """
        Download an image from URL and save to cache.

        Args:
            url: Image URL to download
            query: Search query (for cache naming)

        Returns:
            Path to downloaded image, or None if failed
        """
        cache_path = self._get_cache_path(query)

        # Return cached version if exists
        if cache_path.exists():
            self.logger.debug(f"Using cached image for: {query}")
            return cache_path

        try:
            session = await self._ensure_session()
            async with session.get(url, timeout=self._download_timeout) as response:
                if response.status == 200:
                    content = await response.read()

                    # Determine file extension from content-type
                    content_type = response.headers.get("Content-Type", "image/jpeg")
                    ext = ".jpg"
                    if "png" in content_type:
                        ext = ".png"
                    elif "webp" in content_type:
                        ext = ".webp"

                    # Update cache path with correct extension
                    cache_path = cache_path.with_suffix(ext)

                    # Write to cache
                    with open(cache_path, "wb") as f:
                        f.write(content)

                    self.logger.info(f"Downloaded image: {cache_path.name}")
                    return cache_path
                else:
                    self.logger.warning(f"Image download failed: {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Download error for '{url}': {e}")
            return None

    async def get_broll_images(
        self, queries: List[str], max_per_query: int = 1
    ) -> Dict[str, List[Path]]:
        """
        Get B-roll images for multiple queries.

        Args:
            queries: List of search queries
            max_per_query: Maximum images to download per query

        Returns:
            Dictionary mapping query to list of image paths
        """
        results: Dict[str, List[Path]] = {}
        pending_downloads: List[tuple[str, str]] = []

        for query in queries:
            if not query:
                continue

            search_results = await self.search_images(query, count=max_per_query)
            for result in search_results[:max_per_query]:
                image_url = result.get("url")
                if image_url:
                    pending_downloads.append((query, image_url))

        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)

        async def _download_with_limit(
            query: str, image_url: str
        ) -> tuple[str, Optional[Path]]:
            async with semaphore:
                downloaded_path = await self.download_image(image_url, query)
                return query, downloaded_path

        download_tasks = [
            _download_with_limit(query, image_url)
            for query, image_url in pending_downloads
        ]
        download_results = await asyncio.gather(*download_tasks, return_exceptions=True)

        for item in download_results:
            if isinstance(item, BaseException):
                self.logger.error(f"B-roll download task failed: {item}")
                continue
            if not isinstance(item, tuple):
                continue

            query, downloaded_path = item
            if downloaded_path:
                if query not in results:
                    results[query] = []
                results[query].append(downloaded_path)

        for query, query_images in results.items():
            self.logger.info(f"B-roll for '{query}': {len(query_images)} images")

        return results

    def clear_cache(self) -> int:
        """Clear the B-roll image cache"""
        count = 0
        if self.cache_dir.exists():
            for file in self.cache_dir.glob("*"):
                if file.is_file():
                    file.unlink()
                    count += 1
        self.logger.info(f"Cleared {count} images from B-roll cache")
        return count
