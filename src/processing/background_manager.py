import logging
import random
from pathlib import Path

from moviepy import VideoFileClip

from src.processing.video_processor_fixes import MoviePyCompat


class BackgroundManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.bg_folder = Path("data/backgrounds/minecraft/")

    def get_sliced_background(self, target_duration: float) -> VideoFileClip:
        """Grab a random Minecraft clip chunk matching target duration."""
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
        x_center = w / 2
        subclip = MoviePyCompat.crop(
            subclip,
            x1=int(x_center - (target_w / 2)),
            x2=int(x_center + (target_w / 2)),
        )

        return MoviePyCompat.resize(subclip, (1080, 1920))
