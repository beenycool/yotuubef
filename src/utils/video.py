"""Video utility functions for async video metadata extraction."""

import asyncio
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


async def extract_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Extracts video metadata asynchronously, with a fallback."""
    try:
        from moviepy import VideoFileClip

        def _get_metadata():
            with VideoFileClip(str(video_path)) as clip:
                return {
                    "duration": clip.duration,
                    "fps": clip.fps,
                    "size": clip.size,
                    "has_audio": clip.audio is not None,
                }

        return await asyncio.to_thread(_get_metadata)
    except Exception as e:
        logger.warning(f"Video metadata extraction failed for {video_path}: {e}")
        return {"duration": 60, "fps": 30, "size": (1920, 1080), "has_audio": True}
