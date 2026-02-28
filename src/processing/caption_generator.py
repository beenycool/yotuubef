"""
Word-Level Caption Generator using Faster-Whisper.
Generates dynamic, word-by-word captions that sync perfectly with TTS audio.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

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
        self.device = "cuda" if os.getenv("USE_GPU", "").lower() == "true" else "cpu"

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

        try:
            from moviepy import TextClip, CompositeVideoClip

            # Transcribe audio
            words = self.transcribe_with_timestamps(audio_path)
            if not words:
                self.logger.warning("No words transcribed from audio")
                return None

            caption_clips = []

            for word_data in words:
                word_text = word_data["word"].upper()
                start_time = word_data["start"]
                end_time = word_data["end"]

                # Create bold, heavily stroked text clip for single word
                txt_clip = TextClip(
                    word_text,
                    font=font,
                    fontsize=font_size,
                    color=text_color,
                    stroke_color=stroke_color,
                    stroke_width=stroke_width,
                    method="label",
                    align="center",
                )

                # Position center-bottom with padding
                txt_clip = txt_clip.set_position(("center", "center"))

                # Set exact timing from Whisper
                txt_clip = txt_clip.set_start(start_time).set_end(end_time)

                # Add fade for smoother transitions
                fade_duration = min(0.1, (end_time - start_time) * 0.2)
                if fade_duration > 0.02:
                    txt_clip = txt_clip.crossfadein(fade_duration)
                    txt_clip = txt_clip.crossfadeout(fade_duration)

                caption_clips.append(txt_clip)

            if caption_clips:
                self.logger.info(f"Generated {len(caption_clips)} word captions")
                return CompositeVideoClip([video_clip] + caption_clips)
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error generating word captions: {e}")
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
