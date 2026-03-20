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
        self.base_folder = (
            Path(configured_bg) if configured_bg else Path("data/backgrounds/")
        )

    def get_sliced_background(
        self, target_duration: float, subreddit: str = "general", text_content: str = ""
    ) -> VideoFileClip:
        """Grab a random clip chunk matching target duration, routed by niche."""
        if target_duration <= 0:
            raise ValueError("target_duration must be positive")

        sub_lower = subreddit.lower()
        text_lower = text_content.lower()
        folder_name = "minecraft"

        # Hybrid Routing Logic
        if sub_lower in ("eve", "eveonline") or any(
            k in text_lower for k in ("eve online", "ccp games", "isk")
        ):
            folder_name = "eve_online"
        elif sub_lower in ("speedrun", "speedrun_drama", "summoningsalt") or any(
            k in text_lower
            for k in ("speedrun", "world record", "frame perfect", "glitchless")
        ):
            folder_name = "speedrunning"
        elif sub_lower in (
            "lostmedia",
            "defunctland",
            "internetmysteries",
        ) or any(
            k in text_lower for k in ("lost media", "unfound", "creepy", "abandoned")
        ):
            folder_name = "creepy_static"

        target_folder = self.base_folder / folder_name

        target_folder.mkdir(parents=True, exist_ok=True)
        bg_files = list(target_folder.glob("*.mp4"))

        if not bg_files:
            self.logger.warning(
                f"No backgrounds found in {target_folder}, falling back to minecraft."
            )
            target_folder = self.base_folder / "minecraft"
            target_folder.mkdir(parents=True, exist_ok=True)
            bg_files = list(target_folder.glob("*.mp4"))

        if not bg_files:
            raise FileNotFoundError(f"No background videos found in {self.base_folder}")

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
