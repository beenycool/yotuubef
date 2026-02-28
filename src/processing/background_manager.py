import logging
import os
import random
from pathlib import Path
from typing import Optional, Union

from moviepy import VideoFileClip

from src.processing.video_processor_fixes import MoviePyCompat


class BackgroundManager:
    def __init__(self, bg_folder: Optional[Union[str, Path]] = None):
        self.logger = logging.getLogger(__name__)
        configured_bg = bg_folder or os.getenv("BACKGROUND_FOLDER")
        self.bg_folder = (
            Path(configured_bg)
            if configured_bg
            else Path("data/backgrounds/minecraft/")
        )

    def get_sliced_background(self, target_duration: float) -> VideoFileClip:
        """Grab a random Minecraft clip chunk matching target duration."""
        if target_duration <= 0:
            raise ValueError("target_duration must be positive")

        bg_files = list(self.bg_folder.glob("*.mp4"))
        if not bg_files:
            raise FileNotFoundError(
                "No Minecraft videos found in data/backgrounds/minecraft/"
            )

        chosen_video = random.choice(bg_files)
        clip = VideoFileClip(str(chosen_video))

        if clip.duration <= target_duration:
            return MoviePyCompat.resize(clip, (1080, 1920))

        max_start = max(0.0, clip.duration - target_duration - 1.0)
        start_time = random.uniform(0, max_start) if max_start > 0 else 0.0
        subclip = MoviePyCompat.subclip(clip, start_time, start_time + target_duration)

        w, h = subclip.size
        target_w = int(h * (9 / 16))

        if w > target_w:
            x_center = w / 2
            x1 = max(0, int(x_center - (target_w / 2)))
            x2 = min(w, int(x_center + (target_w / 2)))
            if x2 > x1:
                subclip = MoviePyCompat.crop(subclip, x1=x1, x2=x2)

        return MoviePyCompat.resize(subclip, (1080, 1920))
