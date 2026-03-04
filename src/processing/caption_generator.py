"""
Word-Level Caption Generator using Faster-Whisper.
Generates dynamic, word-by-word captions that sync perfectly with TTS audio.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

from src.config.settings import get_config

# Try to import faster-whisper
try:
    from faster_whisper import WhisperModel

    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None


class CaptionGenerator:
    """
    Generates word-level dynamic captions using Faster-Whisper.
    Creates Hormozi-style fast-paced captions that highlight exactly when each word is spoken.
    """

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        # Model settings - tiny.en is fast, base.en is more accurate
        self.model_size = getattr(self.config.api, "whisper_model_size", "tiny.en")
        use_gpu_env = os.getenv("USE_GPU", "").strip().lower()
        if use_gpu_env in {"true", "false"}:
            self.device = "cuda" if use_gpu_env == "true" else "cpu"
        else:
            cuda_available = bool(torch and torch.cuda.is_available())
            self.device = "cuda" if cuda_available else "cpu"

        self.model = None
        if FASTER_WHISPER_AVAILABLE:
            try:
                self.model = WhisperModel(self.model_size, device=self.device)
                self.logger.info(
                    f"Faster-Whisper loaded: {self.model_size} on {self.device}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to load Faster-Whisper: {e}")
        else:
            self.logger.warning("Faster-Whisper not installed - word captions disabled")

    def transcribe_with_timestamps(self, audio_path: Path) -> List[Dict[str, Any]]:
        """
        Transcribe audio and get word-level timestamps.

        Args:
            audio_path: Path to the audio file (TTS output)

        Returns:
            List of word dictionaries with 'word', 'start', 'end' keys
        """
        if not self.model:
            self.logger.warning("Whisper model not available")
            return []

        try:
            segments, info = self.model.transcribe(
                str(audio_path), word_timestamps=True, language="en"
            )

            words = []
            for segment in segments:
                if hasattr(segment, "words") and segment.words:
                    for word in segment.words:
                        words.append(
                            {
                                "word": word.word.strip(),
                                "start": word.start,
                                "end": word.end,
                                "probability": getattr(word, "probability", 1.0),
                            }
                        )

            self.logger.info(f"Transcribed {len(words)} words from {audio_path.name}")
            return words

        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return []

    def generate_word_captions(
        self,
        video_clip,
        audio_path: Path,
        font_size: int = 90,
        font: str = "Impact",
        text_color: str = "white",
        stroke_color: str = "black",
        stroke_width: int = 5,
        highlight_color: str = "yellow",
    ) -> Optional[Any]:
        """
        Generate word-level caption clips for video.

        Args:
            video_clip: The video to add captions to
            audio_path: Path to TTS audio for transcription
            font_size: Font size for captions
            font: Font name
            text_color: Main text color
            stroke_color: Text outline color
            stroke_width: Text outline width
            highlight_color: Color for highlighted words

        Returns:
            CompositeVideoClip with captions, or None if failed
        """
        if not self.model:
            self.logger.warning("Cannot generate captions - Whisper not available")
            return None

        caption_clips = []
        composite_clip = None
        try:
            from moviepy import TextClip, CompositeVideoClip

            # Transcribe audio
            words = self.transcribe_with_timestamps(audio_path)
            if not words:
                self.logger.warning("No words transcribed from audio")
                return None

            for word_data in words:
                word_text = word_data["word"].upper()
                start_time = word_data["start"]
                end_time = word_data["end"]

                from src.processing.video_processor_fixes import MoviePyCompat

                txt_clip = MoviePyCompat.create_text_clip(
                    word_text,
                    font=font,
                    font_size=font_size,
                    color=text_color,
                    stroke_color=stroke_color,
                    stroke_width=stroke_width
                )

                if txt_clip is None:
                    continue

                txt_clip = MoviePyCompat.with_position(txt_clip, ("center", "center"))
                txt_clip = MoviePyCompat.with_start(txt_clip, start_time)
                txt_clip = MoviePyCompat.with_duration(txt_clip, end_time - start_time)

                # Add fade for smoother transitions
                fade_duration = min(0.1, (end_time - start_time) * 0.2)
                if fade_duration > 0.02:
                    txt_clip = MoviePyCompat.crossfadein(txt_clip, fade_duration)
                    txt_clip = MoviePyCompat.crossfadeout(txt_clip, fade_duration)

                caption_clips.append(txt_clip)

            if caption_clips:
                self.logger.info(f"Generated {len(caption_clips)} word captions")
                composite_clip = CompositeVideoClip([video_clip] + caption_clips)
                return composite_clip
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error generating word captions: {e}")
            for clip in caption_clips:
                try:
                    clip.close()
                except Exception:
                    pass
            if composite_clip is not None:
                try:
                    composite_clip.close()
                except Exception:
                    pass
            return None

    def get_word_count_and_duration(self, audio_path: Path) -> Dict[str, Any]:
        """Get quick stats about the audio"""
        words = self.transcribe_with_timestamps(audio_path)
        if not words:
            return {"word_count": 0, "duration": 0, "words_per_minute": 0}

        duration = words[-1]["end"] if words else 0
        word_count = len(words)
        wpm = (word_count / duration * 60) if duration > 0 else 0

        return {
            "word_count": word_count,
            "duration": duration,
            "words_per_minute": round(wpm, 1),
        }
