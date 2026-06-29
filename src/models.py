"""
Pydantic models for robust data validation and type checking.
Replaces dataclasses with validated models for AI analysis results.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


class PositionType(str, Enum):
    """Valid text overlay positions"""

    CENTER = "center"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


class TextStyle(str, Enum):
    """Valid text overlay styles"""

    DEFAULT = "default"
    BOLD = "bold"
    HIGHLIGHT = "highlight"
    DRAMATIC = "dramatic"


class EmotionType(str, Enum):
    """Valid emotion types for TTS"""

    EXCITED = "excited"
    CALM = "calm"
    DRAMATIC = "dramatic"
    NEUTRAL = "neutral"


class PacingType(str, Enum):
    """Valid pacing types for TTS"""

    SLOW = "slow"
    NORMAL = "normal"
    FAST = "fast"


class TextOverlay(BaseModel):
    """Text overlay information with validation"""

    text: str = Field(
        ..., min_length=1, max_length=200, description="Text content (1-200 characters)"
    )
    timestamp_seconds: float = Field(
        ..., ge=0, description="Timestamp in seconds (must be non-negative)"
    )
    duration: float = Field(
        ..., ge=0.1, le=10.0, description="Display duration in seconds (0.1-10.0)"
    )
    position: PositionType = Field(
        default=PositionType.CENTER, description="Text position on screen"
    )
    style: TextStyle = Field(default=TextStyle.DEFAULT, description="Text style")

    @field_validator("text")
    @classmethod
    def validate_text_content(cls, v):
        if not v.strip():
            raise ValueError("Text content cannot be empty or whitespace only")
        return v.strip()

    model_config = ConfigDict(use_enum_values=True)


class NarrativeSegment(BaseModel):
    """TTS narrative segment with validation"""

    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Narrative text (1-500 characters)",
    )
    time_seconds: float = Field(
        ..., ge=0, description="Start time in seconds (must be non-negative)"
    )
    intended_duration_seconds: float = Field(
        ..., ge=0.1, le=30.0, description="Intended duration (0.1-30.0 seconds)"
    )
    emotion: EmotionType = Field(
        default=EmotionType.NEUTRAL, description="Emotion for TTS"
    )
    pacing: PacingType = Field(default=PacingType.NORMAL, description="Pacing for TTS")
    expression_cue: Optional[str] = Field(
        default=None,
        description="Expression/delivery cue for TTS, e.g. 'whispered, intense, conspiratorial'",
    )
    b_roll_search_query: Optional[str] = Field(
        default=None, description="B-roll image search query for this segment"
    )

    @field_validator("text")
    @classmethod
    def validate_narrative_text(cls, v):
        if not v.strip():
            raise ValueError("Narrative text cannot be empty")
        return v.strip()

    model_config = ConfigDict(use_enum_values=True)


class AudioDuckingConfig(BaseModel):
    """Configuration for intelligent audio ducking during narration"""

    duck_during_narration: bool = Field(
        default=True, description="Enable ducking during TTS"
    )
    duck_volume: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Volume during ducking"
    )
    fade_duration: float = Field(
        default=0.5, ge=0.1, le=2.0, description="Fade duration in seconds"
    )
    smart_detection: bool = Field(
        default=True, description="Use AI to detect speech moments"
    )
    preserve_music_dynamics: bool = Field(
        default=True, description="Preserve music rhythm during ducking"
    )
