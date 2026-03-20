import pytest
from pydantic import ValidationError
from src.models import (
    PositionType,
    TextStyle,
    EffectType,
    EmotionType,
    PacingType,
    VisualCue,
    TextOverlay,
    NarrativeSegment,
    SpeedEffect,
    VideoSegment,
    HookMoment,
    AudioHook,
    ThumbnailInfo,
    CallToAction,
    VideoAnalysis,
    CameraMovement,
    AudioDuckingConfig,
    VideoAnalysisEnhanced,
    SoundEffect,
)


def test_visual_cue_valid():
    cue = VisualCue(
        timestamp_seconds=1.5,
        description="Explosion effect",
        effect_type=EffectType.ZOOM,
        intensity=1.5,
        duration=2.0,
    )
    assert cue.timestamp_seconds == 1.5
    assert cue.description == "Explosion effect"


def test_visual_cue_invalid_duration():
    with pytest.raises(ValidationError):
        VisualCue(
            timestamp_seconds=1.5,
            description="Explosion effect",
            effect_type=EffectType.ZOOM,
            intensity=1.5,
            duration=20.0,  # Exceeds max 10.0
        )


def test_text_overlay_valid():
    overlay = TextOverlay(
        text="Hello World",
        timestamp_seconds=5.0,
        duration=3.0,
        position=PositionType.BOTTOM,
        style=TextStyle.BOLD,
    )
    assert overlay.text == "Hello World"


def test_text_overlay_invalid_text():
    with pytest.raises(ValidationError):
        TextOverlay(
            text="   ",  # Empty text
            timestamp_seconds=5.0,
            duration=3.0,
        )


def test_narrative_segment_valid():
    seg = NarrativeSegment(
        text="This is a story",
        time_seconds=0.0,
        intended_duration_seconds=5.0,
        emotion=EmotionType.EXCITED,
        pacing=PacingType.FAST,
    )
    assert seg.text == "This is a story"


def test_narrative_segment_invalid_duration():
    with pytest.raises(ValidationError):
        NarrativeSegment(
            text="This is a story",
            time_seconds=0.0,
            intended_duration_seconds=40.0,  # Exceeds max 30.0
        )


def test_speed_effect_valid():
    effect = SpeedEffect(start_seconds=1.0, end_seconds=5.0, speed_factor=2.0)
    assert effect.start_seconds == 1.0
    assert effect.end_seconds == 5.0


@pytest.mark.parametrize(
    "start_seconds, end_seconds",
    [
        pytest.param(5.0, 1.0, id="end_less_than_start"),
        pytest.param(5.0, 5.0, id="end_equal_to_start"),
    ],
)
def test_speed_effect_invalid_time_range(start_seconds, end_seconds):
    # End time less than or equal to start time should fail
    with pytest.raises(ValidationError) as excinfo:
        SpeedEffect(
            start_seconds=start_seconds, end_seconds=end_seconds, speed_factor=1.5
        )
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("end_seconds",)
    assert "End time must be greater than start time" in errors[0]["msg"]


def test_speed_effect_invalid_start_time_does_not_break_validator():
    with pytest.raises(ValidationError) as excinfo:
        SpeedEffect(start_seconds=-1.0, end_seconds=5.0, speed_factor=1.5)
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("start_seconds",)
    assert errors[0]["type"] == "greater_than_equal"


def test_video_segment_valid():
    segment = VideoSegment(start_seconds=10.0, end_seconds=20.0, reason="Action scene")
    assert segment.start_seconds == 10.0
    assert segment.end_seconds == 20.0


def test_video_segment_invalid_time_range():
    with pytest.raises(ValidationError):
        VideoSegment(start_seconds=20.0, end_seconds=10.0, reason="Action scene")


def test_video_analysis_valid():
    analysis = VideoAnalysis(
        suggested_title="Test Video",
        summary_for_description="Test Description",
        mood="happy",
        has_clear_narrative=True,
        original_audio_is_key=False,
        hook_text="Look at this!",
        hook_variations=["Hook 1", "Hook 2"],
        visual_hook_moment=HookMoment(timestamp_seconds=2.0, description="Jump"),
        audio_hook=AudioHook(type="sfx", sound_name="whoosh", timestamp_seconds=2.0),
        best_segment=VideoSegment(start_seconds=10.0, end_seconds=20.0, reason="best"),
        segments=[VideoSegment(start_seconds=10.0, end_seconds=20.0, reason="part 1")],
        music_genres=["pop"],
        hashtags=["#test", "cool"],  # Should auto-add # to cool
        thumbnail_info=ThumbnailInfo(
            timestamp_seconds=5.0, reason="face", headline_text="Wow"
        ),
        call_to_action=CallToAction(text="Subscribe", type="text"),
    )
    assert analysis.suggested_title == "Test Video"
    assert "#cool" in analysis.hashtags


def test_video_analysis_invalid_hook_variations():
    with pytest.raises(ValidationError):
        VideoAnalysis(
            suggested_title="Test Video",
            summary_for_description="Test Description",
            mood="happy",
            has_clear_narrative=True,
            original_audio_is_key=False,
            hook_text="Look at this!",
            hook_variations=["Hook 1", "Hook 1"],  # Duplicate
            visual_hook_moment=HookMoment(timestamp_seconds=2.0, description="Jump"),
            audio_hook=AudioHook(
                type="sfx", sound_name="whoosh", timestamp_seconds=2.0
            ),
            best_segment=VideoSegment(
                start_seconds=10.0, end_seconds=20.0, reason="best"
            ),
            segments=[
                VideoSegment(start_seconds=10.0, end_seconds=20.0, reason="part 1")
            ],
            music_genres=["pop"],
            hashtags=["#test"],
            thumbnail_info=ThumbnailInfo(
                timestamp_seconds=5.0, reason="face", headline_text="Wow"
            ),
            call_to_action=CallToAction(text="Subscribe", type="text"),
        )


def test_video_analysis_invalid_hashtags():
    with pytest.raises(ValidationError):
        VideoAnalysis(
            suggested_title="Test Video",
            summary_for_description="Test Description",
            mood="happy",
            has_clear_narrative=True,
            original_audio_is_key=False,
            hook_text="Look at this!",
            hook_variations=["Hook 1", "Hook 2"],
            visual_hook_moment=HookMoment(timestamp_seconds=2.0, description="Jump"),
            audio_hook=AudioHook(
                type="sfx", sound_name="whoosh", timestamp_seconds=2.0
            ),
            best_segment=VideoSegment(
                start_seconds=10.0, end_seconds=20.0, reason="best"
            ),
            segments=[
                VideoSegment(start_seconds=10.0, end_seconds=20.0, reason="part 1")
            ],
            music_genres=["pop"],
            hashtags=["   ", "!@#"],  # Invalid hashtags
            thumbnail_info=ThumbnailInfo(
                timestamp_seconds=5.0, reason="face", headline_text="Wow"
            ),
            call_to_action=CallToAction(text="Subscribe", type="text"),
        )


def test_video_analysis_consistency():
    with pytest.raises(ValidationError):
        VideoAnalysis(
            suggested_title="Test Video",
            summary_for_description="Test Description",
            mood="happy",
            has_clear_narrative=True,
            original_audio_is_key=False,
            hook_text="Look at this!",
            hook_variations=["Hook 1", "Hook 2"],
            visual_hook_moment=HookMoment(timestamp_seconds=2.0, description="Jump"),
            audio_hook=AudioHook(
                type="sfx", sound_name="whoosh", timestamp_seconds=2.0
            ),
            best_segment=VideoSegment(
                start_seconds=10.0, end_seconds=20.0, reason="best"
            ),
            segments=[
                VideoSegment(start_seconds=10.0, end_seconds=400.0, reason="too long")
            ],  # > 300
            music_genres=["pop"],
            hashtags=["#test"],
            thumbnail_info=ThumbnailInfo(
                timestamp_seconds=5.0, reason="face", headline_text="Wow"
            ),
            call_to_action=CallToAction(text="Subscribe", type="text"),
        )


def test_video_analysis_fallback_defaults():
    analysis = VideoAnalysis.model_construct()
    defaults = analysis.get_fallback_defaults()
    assert defaults["suggested_title"] == "Amazing Video Content"
    assert defaults["fallback"] is True


def test_camera_movement_valid():
    movement = CameraMovement(
        start_time=1.0,
        end_time=5.0,
        movement_type="pan",
        start_position=(0.0, 0.0),
        end_position=(1.0, 1.0),
    )
    assert movement.start_time == 1.0
    assert movement.end_time == 5.0


def test_camera_movement_invalid_time_range():
    with pytest.raises(ValidationError):
        CameraMovement(
            start_time=5.0,
            end_time=1.0,
            movement_type="pan",
            start_position=(0.0, 0.0),
            end_position=(1.0, 1.0),
        )


def test_video_analysis_enhanced_valid():
    analysis = VideoAnalysisEnhanced(
        suggested_title="Enhanced Video",
        summary_for_description="Test Description",
        mood="happy",
        has_clear_narrative=True,
        original_audio_is_key=False,
        hook_text="Look at this!",
        hook_variations=["Hook 1", "Hook 2"],
        visual_hook_moment=HookMoment(timestamp_seconds=2.0, description="Jump"),
        audio_hook=AudioHook(type="sfx", sound_name="whoosh", timestamp_seconds=2.0),
        best_segment=VideoSegment(start_seconds=10.0, end_seconds=20.0, reason="best"),
        segments=[VideoSegment(start_seconds=10.0, end_seconds=20.0, reason="part 1")],
        music_genres=["pop"],
        hashtags=["#test"],
        thumbnail_info=ThumbnailInfo(
            timestamp_seconds=5.0, reason="face", headline_text="Wow"
        ),
        call_to_action=CallToAction(text="Subscribe", type="text"),
        audio_ducking_config=AudioDuckingConfig(
            duck_during_narration=True, duck_volume=0.5
        ),
    )
    assert analysis.suggested_title == "Enhanced Video"
    assert analysis.audio_ducking_config.duck_volume == 0.5


def test_text_overlay_valid_text():
    overlay = TextOverlay(text="Hello World", timestamp_seconds=1.0, duration=5.0)
    assert overlay.text == "Hello World"


def test_text_overlay_strips_whitespace():
    overlay = TextOverlay(text="  Hello World  ", timestamp_seconds=1.0, duration=5.0)
    assert overlay.text == "Hello World"


def test_text_overlay_empty_string():
    with pytest.raises(ValidationError) as exc_info:
        TextOverlay(text="", timestamp_seconds=1.0, duration=5.0)
    assert "String should have at least 1 character" in str(exc_info.value)


def test_text_overlay_whitespace_only():
    with pytest.raises(ValidationError) as exc_info:
        TextOverlay(text="   ", timestamp_seconds=1.0, duration=5.0)
    assert "Text content cannot be empty or whitespace only" in str(exc_info.value)


def test_text_overlay_too_long():
    long_text = "a" * 201
    with pytest.raises(ValidationError) as exc_info:
        TextOverlay(text=long_text, timestamp_seconds=1.0, duration=5.0)
    assert "String should have at most 200 characters" in str(exc_info.value)


def test_sound_effect_valid():
    effect = SoundEffect(timestamp_seconds=2.0, effect_name="whoosh")
    assert effect.timestamp_seconds == 2.0
    assert effect.effect_name == "whoosh"
    assert effect.volume == 0.7  # default


def test_sound_effect_invalid_timestamp():
    with pytest.raises(ValidationError) as excinfo:
        SoundEffect(timestamp_seconds=-1.0, effect_name="whoosh")
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("timestamp_seconds",)
    assert errors[0]["type"] == "greater_than_equal"
