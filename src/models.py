"""
Pydantic models for robust data validation and type checking.
Replaces dataclasses with validated models for AI analysis results.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
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


class EffectType(str, Enum):
    """Valid visual effect types"""
    ZOOM = "zoom"
    HIGHLIGHT = "highlight"
    TEXT_OVERLAY = "text_overlay"
    COLOR_GRADE = "color_grade"


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


class VisualCue(BaseModel):
    """Visual cue for video enhancement with validation"""
    timestamp_seconds: float = Field(..., ge=0, description="Timestamp in seconds (must be non-negative)")
    description: str = Field(..., min_length=1, description="Description of the visual cue")
    effect_type: EffectType = Field(..., description="Type of effect to apply")
    intensity: float = Field(default=1.0, ge=0.1, le=2.0, description="Effect intensity (0.1-2.0)")
    duration: float = Field(default=1.0, ge=0.1, le=10.0, description="Effect duration in seconds (0.1-10.0)")

    @field_validator('timestamp_seconds')
    @classmethod
    def validate_timestamp(cls, v):
        if v < 0:
            raise ValueError('Timestamp must be non-negative')
        return v

    model_config = ConfigDict(use_enum_values=True)


class TextOverlay(BaseModel):
    """Text overlay information with validation"""
    text: str = Field(..., min_length=1, max_length=200, description="Text content (1-200 characters)")
    timestamp_seconds: float = Field(..., ge=0, description="Timestamp in seconds (must be non-negative)")
    duration: float = Field(..., ge=0.1, le=10.0, description="Display duration in seconds (0.1-10.0)")
    position: PositionType = Field(default=PositionType.CENTER, description="Text position on screen")
    style: TextStyle = Field(default=TextStyle.DEFAULT, description="Text style")

    @field_validator('text')
    @classmethod
    def validate_text_content(cls, v):
        if not v.strip():
            raise ValueError('Text content cannot be empty or whitespace only')
        return v.strip()

    model_config = ConfigDict(use_enum_values=True)


class NarrativeSegment(BaseModel):
    """TTS narrative segment with validation"""
    text: str = Field(..., min_length=1, max_length=500, description="Narrative text (1-500 characters)")
    time_seconds: float = Field(..., ge=0, description="Start time in seconds (must be non-negative)")
    intended_duration_seconds: float = Field(..., ge=0.1, le=30.0, description="Intended duration (0.1-30.0 seconds)")
    emotion: EmotionType = Field(default=EmotionType.NEUTRAL, description="Emotion for TTS")
    pacing: PacingType = Field(default=PacingType.NORMAL, description="Pacing for TTS")

    @field_validator('text')
    @classmethod
    def validate_narrative_text(cls, v):
        if not v.strip():
            raise ValueError('Narrative text cannot be empty')
        return v.strip()

    model_config = ConfigDict(use_enum_values=True)


class FocusPoint(BaseModel):
    """Key focus point in video with validation"""
    x: float = Field(..., ge=0.0, le=1.0, description="X coordinate (0.0-1.0)")
    y: float = Field(..., ge=0.0, le=1.0, description="Y coordinate (0.0-1.0)")
    timestamp_seconds: float = Field(..., ge=0, description="Timestamp in seconds")
    description: str = Field(..., min_length=1, description="Description of the focus point")


class SpeedEffect(BaseModel):
    """Speed effect configuration with validation"""
    start_seconds: float = Field(..., ge=0, description="Start time in seconds")
    end_seconds: float = Field(..., ge=0, description="End time in seconds")
    speed_factor: float = Field(..., ge=0.1, le=5.0, description="Speed multiplier (0.1-5.0)")
    effect_type: str = Field(default="speed_change", description="Type of speed effect")

    @field_validator('end_seconds')
    @classmethod
    def validate_time_range(cls, v, info):
        if hasattr(info, 'data') and 'start_seconds' in info.data and v <= info.data['start_seconds']:
            raise ValueError('End time must be greater than start time')
        return v


class SoundEffect(BaseModel):
    """Sound effect configuration with validation"""
    timestamp_seconds: float = Field(..., ge=0, description="Timestamp for sound effect")
    effect_name: str = Field(..., min_length=1, description="Name of the sound effect")
    volume: float = Field(default=0.7, ge=0.0, le=1.0, description="Volume level (0.0-1.0)")


class VideoSegment(BaseModel):
    """Video segment information with validation"""
    start_seconds: float = Field(..., ge=0, description="Start time in seconds")
    end_seconds: float = Field(..., ge=0, description="End time in seconds")
    reason: str = Field(..., min_length=1, description="Reason for this segment selection")

    @field_validator('end_seconds')
    @classmethod
    def validate_segment_range(cls, v, info):
        if hasattr(info, 'data') and 'start_seconds' in info.data and v <= info.data['start_seconds']:
            raise ValueError('End time must be greater than start time')
        return v


class HookMoment(BaseModel):
    """Visual hook moment with validation"""
    timestamp_seconds: float = Field(..., ge=0, description="Timestamp of hook moment")
    description: str = Field(..., min_length=1, description="Description of the hook moment")


class AudioHook(BaseModel):
    """Audio hook configuration with validation"""
    type: str = Field(..., min_length=1, description="Type of audio hook")
    sound_name: str = Field(..., min_length=1, description="Name of the sound")
    timestamp_seconds: float = Field(..., ge=0, description="Timestamp for audio hook")


class ThumbnailInfo(BaseModel):
    """Thumbnail information with validation"""
    timestamp_seconds: float = Field(..., ge=0, description="Timestamp for thumbnail")
    reason: str = Field(..., min_length=1, description="Reason for thumbnail selection")
    headline_text: str = Field(default="", description="Headline text for thumbnail")


class CallToAction(BaseModel):
    """Call to action configuration with validation"""
    text: str = Field(..., min_length=1, max_length=100, description="CTA text (1-100 characters)")
    type: str = Field(..., min_length=1, description="Type of call to action")


class VideoAnalysis(BaseModel):
    """Complete video analysis result from AI with comprehensive validation"""
    
    # Basic information
    suggested_title: str = Field(..., min_length=1, max_length=100, description="Video title (1-100 characters)")
    summary_for_description: str = Field(..., min_length=1, max_length=500, description="Description (1-500 characters)")
    mood: str = Field(..., min_length=1, description="Overall mood of the video")
    has_clear_narrative: bool = Field(..., description="Whether video has clear narrative")
    original_audio_is_key: bool = Field(..., description="Whether original audio is important")
    
    # Hook content
    hook_text: str = Field(..., min_length=1, max_length=200, description="Hook text (1-200 characters)")
    hook_variations: List[str] = Field(..., min_length=1, max_length=10, description="Alternative hooks (1-10 items)")
    visual_hook_moment: HookMoment = Field(..., description="Most engaging visual moment")
    audio_hook: AudioHook = Field(..., description="Audio hook configuration")
    
    # Video segments
    best_segment: VideoSegment = Field(..., description="Best video segment")
    segments: List[VideoSegment] = Field(..., min_length=1, description="All video segments")
    
    # Enhancement elements
    key_focus_points: List[FocusPoint] = Field(default_factory=list, description="Key focus points in video")
    text_overlays: List[TextOverlay] = Field(default_factory=list, description="Text overlays to add")
    narrative_script_segments: List[NarrativeSegment] = Field(default_factory=list, description="TTS segments")
    visual_cues: List[VisualCue] = Field(default_factory=list, description="Visual enhancement cues")
    speed_effects: List[SpeedEffect] = Field(default_factory=list, description="Speed effects to apply")
    
    # Audio and effects
    music_genres: List[str] = Field(..., min_length=1, description="Suitable music genres")
    sound_effects: List[SoundEffect] = Field(default_factory=list, description="Sound effects to add")
    hashtags: List[str] = Field(..., min_length=1, max_length=30, description="Relevant hashtags (1-30 items)")
    original_duration: float = Field(default=0.0, ge=0, description="Original video duration")
    tts_pacing: PacingType = Field(default=PacingType.NORMAL, description="TTS pacing")
    emotional_keywords: List[str] = Field(default_factory=list, description="Emotional keywords")
    
    # Engagement and metadata
    thumbnail_info: ThumbnailInfo = Field(..., description="Thumbnail configuration")
    call_to_action: CallToAction = Field(..., description="Call to action")
    retention_tactics: List[str] = Field(default_factory=list, description="Viewer retention tactics")
    
    # Content safety
    is_explicitly_age_restricted: bool = Field(default=False, description="Whether content is age-restricted")
    fallback: bool = Field(default=False, description="Whether this is fallback analysis")

    @field_validator('hook_variations')
    @classmethod
    def validate_hook_variations(cls, v):
        # Ensure all hook variations are non-empty and unique
        cleaned = [hook.strip() for hook in v if hook.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError('Hook variations must be unique')
        return cleaned

    @field_validator('hashtags')
    @classmethod
    def validate_hashtags(cls, v):
        # Ensure hashtags start with # and are properly formatted
        validated = []
        for tag in v:
            tag = tag.strip()
            if not tag:
                continue
            if not tag.startswith('#'):
                tag = f'#{tag}'
            
            # Split by # and process each part separately to handle cases like "#special!@#"
            parts = tag.split('#')
            if len(parts) >= 2:  # Should have at least ['', 'content'] after split
                # Take the first non-empty part after #
                content = parts[1] if len(parts) > 1 else ''
                # Remove spaces and special characters except underscore
                clean_content = ''.join(c for c in content if c.isalnum() or c == '_')
                if clean_content:  # Must have content after cleaning
                    validated.append(f'#{clean_content}')
        
        if not validated:
            raise ValueError('At least one valid hashtag is required')
        return validated

    @model_validator(mode='after')
    def validate_analysis_consistency(self):
        """Validate consistency across the entire analysis"""
        
        # Ensure visual cues don't exceed reasonable bounds
        if len(self.visual_cues) > 20:
            raise ValueError('Too many visual cues (maximum 20)')
        
        # Ensure text overlays don't overlap too much
        if len(self.text_overlays) > 15:
            raise ValueError('Too many text overlays (maximum 15)')
        
        # Validate that segments make sense
        if self.segments:
            total_duration = max([seg.end_seconds for seg in self.segments])
            if total_duration > 300:  # 5 minutes max
                raise ValueError('Total video duration seems unreasonable (>5 minutes)')
        
        return self

    def get_fallback_defaults(self) -> Dict[str, Any]:
        """Get fallback default values for missing or invalid data"""
        return {
            'suggested_title': 'Amazing Video Content',
            'summary_for_description': 'Check out this incredible video!',
            'mood': 'exciting',
            'has_clear_narrative': False,
            'original_audio_is_key': True,
            'hook_text': 'Watch this amazing moment!',
            'hook_variations': ['Incredible!', 'Must see!', 'Amazing!'],
            'music_genres': ['upbeat'],
            'hashtags': ['#shorts', '#viral'],
            'tts_pacing': PacingType.NORMAL,
            'fallback': True
        }

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        extra="forbid"  # Don't allow extra fields
    )