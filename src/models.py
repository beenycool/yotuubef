"""
Pydantic models for robust data validation and type checking.
Replaces dataclasses with validated models for AI analysis results.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum
from datetime import datetime


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


class CameraMovement(BaseModel):
    """AI-suggested camera movement for dynamic effects"""
    start_time: float = Field(..., ge=0, description="Start time in seconds")
    end_time: float = Field(..., ge=0, description="End time in seconds")
    movement_type: str = Field(..., description="Type of movement (pan, zoom, pan_zoom)")
    start_position: Tuple[float, float] = Field(..., description="Starting position (x, y) as ratios 0-1")
    end_position: Tuple[float, float] = Field(..., description="Ending position (x, y) as ratios 0-1")
    zoom_factor: float = Field(default=1.0, ge=0.5, le=3.0, description="Zoom factor (0.5-3.0)")
    easing: str = Field(default="ease_in_out", description="Easing function for smooth movement")
    intensity: float = Field(default=1.0, ge=0.1, le=2.0, description="Movement intensity")

    @field_validator('end_time')
    @classmethod
    def validate_time_range(cls, v, info):
        if hasattr(info, 'data') and 'start_time' in info.data and v <= info.data['start_time']:
            raise ValueError('End time must be greater than start time')
        return v


class AudioDuckingConfig(BaseModel):
    """Configuration for intelligent audio ducking during narration"""
    duck_during_narration: bool = Field(default=True, description="Enable ducking during TTS")
    duck_volume: float = Field(default=0.3, ge=0.0, le=1.0, description="Volume during ducking")
    fade_duration: float = Field(default=0.5, ge=0.1, le=2.0, description="Fade duration in seconds")
    smart_detection: bool = Field(default=True, description="Use AI to detect speech moments")
    preserve_music_dynamics: bool = Field(default=True, description="Preserve music rhythm during ducking")


class ThumbnailVariant(BaseModel):
    """A/B test thumbnail variant configuration"""
    variant_id: str = Field(..., description="Unique variant identifier")
    headline_text: str = Field(..., description="Headline text for this variant")
    timestamp_seconds: float = Field(..., ge=0, description="Frame timestamp for thumbnail")
    text_style: str = Field(default="bold", description="Text styling approach")
    color_scheme: str = Field(default="high_contrast", description="Color scheme for text")
    emotional_tone: str = Field(default="exciting", description="Emotional tone of design")
    click_through_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Measured CTR")
    performance_score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Overall performance score")


class EnhancementOptimization(BaseModel):
    """Performance-based enhancement optimization settings"""
    effect_type: str = Field(..., description="Type of enhancement effect")
    current_intensity: float = Field(..., ge=0.0, le=2.0, description="Current intensity setting")
    performance_score: float = Field(..., ge=0.0, le=100.0, description="Performance score (0-100)")
    usage_frequency: float = Field(..., ge=0.0, le=1.0, description="How often this effect is used")
    retention_impact: float = Field(default=0.0, description="Impact on viewer retention")
    engagement_impact: float = Field(default=0.0, description="Impact on engagement metrics")
    recommended_adjustment: float = Field(default=0.0, description="Recommended intensity adjustment")


class CommentEngagement(BaseModel):
    """AI-analyzed comment for engagement boosting"""
    comment_id: str = Field(..., description="YouTube comment ID")
    comment_text: str = Field(..., description="Comment content")
    engagement_score: float = Field(..., ge=0.0, le=100.0, description="AI-calculated engagement potential")
    sentiment: str = Field(..., description="Comment sentiment (positive, negative, neutral)")
    reply_suggestion: Optional[str] = Field(default=None, description="AI-suggested reply")
    should_pin: bool = Field(default=False, description="Whether to pin this comment")
    interaction_type: str = Field(default="like", description="Recommended interaction type")


class PerformanceMetrics(BaseModel):
    """Enhanced performance tracking for optimization"""
    video_id: str = Field(..., description="Video identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")
    
    # Core metrics
    views: int = Field(default=0, ge=0, description="View count")
    likes: int = Field(default=0, ge=0, description="Like count")
    comments: int = Field(default=0, ge=0, description="Comment count")
    shares: int = Field(default=0, ge=0, description="Share count")
    
    # Advanced metrics
    watch_time_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Average watch percentage")
    retention_curve: List[float] = Field(default_factory=list, description="Retention at each 10% of video")
    click_through_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Thumbnail CTR")
    audience_retention_peaks: List[float] = Field(default_factory=list, description="Timestamps of retention peaks")
    
    # Enhancement correlation
    active_enhancements: List[str] = Field(default_factory=list, description="Enhancements used in this video")
    enhancement_performance: Dict[str, float] = Field(default_factory=dict, description="Per-enhancement performance scores")


class SystemPerformanceMetrics(BaseModel):
    """System-wide performance metrics for orchestrator statistics"""
    total_videos_processed: int = Field(default=0, ge=0, description="Total videos processed by the system")
    average_processing_time: float = Field(default=0.0, ge=0.0, description="Average processing time in seconds")
    success_rate: float = Field(default=0.0, ge=0.0, le=100.0, description="Success rate percentage")
    enhancement_usage: Dict[str, int] = Field(default_factory=dict, description="Count of each enhancement used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")
    
    model_config = ConfigDict(use_enum_values=True)


# Enhanced VideoAnalysis with new AI-powered features
class VideoAnalysisEnhanced(VideoAnalysis):
    """Enhanced video analysis with AI-powered cinematic and engagement features"""
    
    # AI-powered cinematic editing
    camera_movements: List[CameraMovement] = Field(default_factory=list, description="AI-suggested camera movements")
    dynamic_focus_points: List[FocusPoint] = Field(default_factory=list, description="Dynamic focus points for attention")
    cinematic_transitions: List[Dict[str, Any]] = Field(default_factory=list, description="Transition effects between segments")
    
    # Advanced audio processing
    audio_ducking_config: AudioDuckingConfig = Field(default_factory=AudioDuckingConfig, description="Audio ducking settings")
    voice_enhancement_params: Dict[str, float] = Field(default_factory=dict, description="Voice processing parameters")
    background_audio_zones: List[Dict[str, Any]] = Field(default_factory=list, description="Background audio management zones")
    
    # Engagement-driven thumbnails
    thumbnail_variants: List[ThumbnailVariant] = Field(default_factory=list, description="A/B test thumbnail variants")
    optimal_thumbnail_elements: Dict[str, Any] = Field(default_factory=dict, description="AI-optimized thumbnail elements")
    
    # Performance optimization
    enhancement_recommendations: List[EnhancementOptimization] = Field(default_factory=list, description="Performance-based recommendations")
    predicted_performance: Dict[str, float] = Field(default_factory=dict, description="AI-predicted performance metrics")
    
    # Channel management
    comment_engagement_targets: List[CommentEngagement] = Field(default_factory=list, description="Comments to engage with")
    auto_response_triggers: List[str] = Field(default_factory=list, description="Trigger phrases for auto-responses")
    
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        extra="forbid"
    )


# Long-Form Video Generation Models

class VideoFormat(str, Enum):
    """Video format types"""
    SHORTS = "shorts"
    LONG_FORM = "long_form"


class ContentStructureType(str, Enum):
    """Content structure types for long-form videos"""
    INTRO = "intro"
    BODY = "body"
    CONCLUSION = "conclusion"


class NicheCategory(str, Enum):
    """Niche categories for targeted content"""
    TECHNOLOGY = "technology"
    EDUCATION = "education"
    ENTERTAINMENT = "entertainment"
    LIFESTYLE = "lifestyle"
    BUSINESS = "business"
    SCIENCE = "science"
    HEALTH = "health"
    GAMING = "gaming"
    COOKING = "cooking"
    TRAVEL = "travel"
    FITNESS = "fitness"
    FINANCE = "finance"


class ContentSection(BaseModel):
    """Structured content section for long-form videos"""
    section_type: ContentStructureType = Field(..., description="Type of content section")
    title: str = Field(..., min_length=1, max_length=100, description="Section title")
    content: str = Field(..., min_length=10, max_length=2000, description="Section content")
    duration_seconds: float = Field(..., ge=5.0, le=300.0, description="Target duration in seconds")
    key_points: List[str] = Field(default_factory=list, description="Key points to cover")
    visual_suggestions: List[str] = Field(default_factory=list, description="Visual content suggestions")
    
    model_config = ConfigDict(use_enum_values=True)


class NicheTopicConfig(BaseModel):
    """Configuration for niche topic targeting"""
    category: NicheCategory = Field(..., description="Primary niche category")
    target_audience: str = Field(..., min_length=1, description="Target audience description")
    expertise_level: str = Field(default="beginner", description="Content expertise level")
    tone: str = Field(default="informative", description="Content tone")
    keywords: List[str] = Field(default_factory=list, description="Target keywords")
    
    model_config = ConfigDict(use_enum_values=True)


class LongFormVideoStructure(BaseModel):
    """Complete structure for long-form video content"""
    title: str = Field(..., min_length=1, max_length=100, description="Video title")
    description: str = Field(..., min_length=10, max_length=1000, description="Video description")
    niche_config: NicheTopicConfig = Field(..., description="Niche targeting configuration")
    
    intro_section: ContentSection = Field(..., description="Introduction section")
    body_sections: List[ContentSection] = Field(..., min_length=1, description="Body sections")
    conclusion_section: ContentSection = Field(..., description="Conclusion section")
    
    total_duration_seconds: float = Field(..., ge=60.0, le=3600.0, description="Total video duration")
    hashtags: List[str] = Field(default_factory=list, description="Relevant hashtags")
    
    def get_total_sections(self) -> int:
        """Get total number of sections"""
        return 2 + len(self.body_sections)  # intro + body_sections + conclusion
    
    def get_estimated_duration(self) -> float:
        """Get estimated total duration from sections"""
        total = self.intro_section.duration_seconds + self.conclusion_section.duration_seconds
        total += sum(section.duration_seconds for section in self.body_sections)
        return total
    
    @model_validator(mode='after')
    def validate_structure(self):
        """Validate the video structure"""
        if len(self.body_sections) > 10:
            raise ValueError("Too many body sections (maximum 10)")
        
        estimated_duration = self.get_estimated_duration()
        if abs(estimated_duration - self.total_duration_seconds) > 60:
            raise ValueError(f"Duration mismatch: estimated {estimated_duration}s vs target {self.total_duration_seconds}s")
        
        return self
    
    model_config = ConfigDict(use_enum_values=True)


class LongFormVideoAnalysis(BaseModel):
    """Analysis results for long-form video generation"""
    video_format: VideoFormat = Field(default=VideoFormat.LONG_FORM, description="Video format type")
    video_structure: LongFormVideoStructure = Field(..., description="Complete video structure")
    
    # Enhanced narration
    detailed_narration: List[NarrativeSegment] = Field(default_factory=list, description="Detailed narration segments")
    section_transitions: List[str] = Field(default_factory=list, description="Transition phrases between sections")
    
    # Visual elements (reuse existing models)
    visual_cues: List[VisualCue] = Field(default_factory=list, description="Visual enhancement cues")
    text_overlays: List[TextOverlay] = Field(default_factory=list, description="Text overlays")
    
    # Audience targeting
    target_audience_analysis: Dict[str, Any] = Field(default_factory=dict, description="Audience analysis data")
    engagement_hooks: List[str] = Field(default_factory=list, description="Engagement hooks throughout video")
    
    model_config = ConfigDict(use_enum_values=True)