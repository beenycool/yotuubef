"""
Pydantic models for robust data validation and type checking.
Replaces dataclasses with validated models for AI analysis results.
"""

from typing import Optional, Dict, List, Any
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


class CallToAction(BaseModel):
    text: str = ""
    type: str = "subscribe"


class HookMoment(BaseModel):
    timestamp_seconds: float = 0.0
    description: str = ""


class AudioHook(BaseModel):
    type: str = "dramatic"
    sound_name: str = "whoosh"
    timestamp_seconds: float = 0.0


class VideoSegment(BaseModel):
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    reason: str = ""


class ThumbnailInfo(BaseModel):
    timestamp_seconds: float = 0.0
    reason: str = ""
    headline_text: str = ""


class CameraMovement(BaseModel):
    start_time: float = Field(0.0, ge=0)
    end_time: float = Field(0.0, ge=0)
    movement_type: str = "pan"
    start_position: tuple = (0.5, 0.5)
    end_position: tuple = (0.5, 0.5)
    zoom_factor: float = 1.0
    easing: str = "linear"
    intensity: float = Field(0.5, ge=0, le=1.0)


class FocusPoint(BaseModel):
    x: float = Field(0.5, ge=0, le=1)
    y: float = Field(0.5, ge=0, le=1)
    timestamp_seconds: float = 0.0
    description: str = ""


class SpeedEffect(BaseModel):
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    speed_factor: float = 1.0
    effect_type: str = "slow_motion"


class ThumbnailVariant(BaseModel):
    variant_id: str = ""
    headline_text: str = ""
    timestamp_seconds: float = 0.0
    text_style: str = "bold_impact"
    color_scheme: str = "high_contrast"
    emotional_tone: str = "exciting"
    performance_score: Optional[float] = None


class PerformanceMetrics(BaseModel):
    video_id: str = ""
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    watch_time_percentage: float = 0.0
    click_through_rate: float = 0.0


class CommentEngagement(BaseModel):
    comment_id: str = ""
    engagement_score: float = 0.0
    sentiment: str = "neutral"
    toxicity_score: float = 0.0


class EnhancementOptimization(BaseModel):
    last_optimization_date: Optional[str] = None
    parameter_history: Dict[str, Any] = Field(default_factory=dict)
    performance_trends: Dict[str, Any] = Field(default_factory=dict)


class VideoAnalysis(BaseModel):
    suggested_title: str = ""
    summary_for_description: str = ""
    mood: str = "exciting"
    has_clear_narrative: bool = True
    original_audio_is_key: bool = False
    hook_text: str = ""
    hook_variations: List[str] = Field(default_factory=list)
    visual_hook_moment: Optional[HookMoment] = None
    audio_hook: Optional[AudioHook] = None
    best_segment: Optional[VideoSegment] = None
    segments: List[VideoSegment] = Field(default_factory=list)
    text_overlays: List[TextOverlay] = Field(default_factory=list)
    narrative_script_segments: List[NarrativeSegment] = Field(default_factory=list)
    thumbnail_info: Optional[ThumbnailInfo] = None
    call_to_action: Optional[CallToAction] = None
    music_genres: List[str] = Field(default_factory=list)
    hashtags: List[str] = Field(default_factory=list)
    audio_ducking_config: Optional[AudioDuckingConfig] = None
    camera_movements: List[CameraMovement] = Field(default_factory=list)
    speed_effects: List[SpeedEffect] = Field(default_factory=list)
    dynamic_focus_points: List[FocusPoint] = Field(default_factory=list)
    cinematic_transitions: List[Dict[str, Any]] = Field(default_factory=list)
    key_focus_points: List[FocusPoint] = Field(default_factory=list)
    original_duration: Optional[float] = None

    model_config = ConfigDict(use_enum_values=True)


class VideoAnalysisEnhanced(VideoAnalysis):
    model_config = ConfigDict(use_enum_values=True)


class ScriptSegment(BaseModel):
    """Validated script segment from AI generation."""

    time_seconds: float = Field(default=0.0, ge=0)
    intended_duration_seconds: float = Field(default=6.0, ge=0.1, le=30.0)
    narration: str = Field(..., min_length=1, max_length=500)
    expression_cue: Optional[str] = None
    visual_asset_path: Optional[str] = None
    visual_directive: Optional[str] = None
    text_overlay: Optional[str] = None
    evidence_refs: List[str] = Field(default_factory=list)
    pace: str = Field(default="fast", pattern=r"^(fast|normal|slow)$")
    emotion: str = Field(
        default="dramatic", pattern=r"^(excited|calm|dramatic|neutral)$"
    )


class ScriptSchema(BaseModel):
    """Validated AI generation output for SCRIPTING phase."""

    phase: str = Field(default="SCRIPTING", pattern=r"^SCRIPTING$")
    title: str = Field(..., min_length=1, max_length=200)
    hook: str = Field(..., min_length=1, max_length=500)
    loop_bridge: str = Field(default="", max_length=500)
    segments: List[ScriptSegment] = Field(..., min_length=1)
    sources_to_check: List[str] = Field(default_factory=list)
    hashtags: List[str] = Field(default_factory=list)


class IdeaAngle(BaseModel):
    id: str = Field(default="A1", pattern=r"^A\d+$")
    title: str = Field(..., min_length=1, max_length=200)
    hook: str = Field(..., min_length=1, max_length=500)
    viability_score: int = Field(default=50, ge=0, le=100)
    source_urls: List[str] = Field(default_factory=list)


class IdeaGenerationSchema(BaseModel):
    """Validated AI generation output for IDEA_GENERATION phase."""

    phase: str = Field(default="IDEA_GENERATION", pattern=r"^IDEA_GENERATION$")
    angles: List[IdeaAngle] = Field(..., min_length=1, max_length=10)
    gemini_deep_research_prompt: str = Field(..., min_length=20)
    next_phase: str = Field(default="WAIT_FOR_GEMINI_REPORT")


class ScriptJudgeResult(BaseModel):
    """Result from the LLM Judge evaluating script quality."""

    score: int = Field(..., ge=1, le=10)
    hook_under_3_seconds: bool = True
    sounds_natural: bool = True
    sentences_too_long: bool = False
    has_forbidden_phrases: bool = False
    feedback: str = Field(default="", max_length=1000)
    passes_quality_bar: bool = Field(default=True)


class EvidencePlanItem(BaseModel):
    claim: str = Field(..., min_length=1)
    evidence_needed: List[str] = Field(default_factory=list)
    priority: str = Field(default="medium", pattern=r"^(high|medium|low)$")


class EvidenceGatheringSchema(BaseModel):
    """Validated AI generation output for EVIDENCE_GATHERING phase."""

    phase: str = Field(default="EVIDENCE_GATHERING", pattern=r"^EVIDENCE_GATHERING$")
    evidence_plan: List[EvidencePlanItem] = Field(..., min_length=1)
    media_queries: List[str] = Field(..., min_length=1)
    next_phase: str = Field(default="SCRIPTING")


PHASE_SCHEMA_MAP = {
    "SCRIPTING": ScriptSchema,
    "IDEA_GENERATION": IdeaGenerationSchema,
    "EVIDENCE_GATHERING": EvidenceGatheringSchema,
}
