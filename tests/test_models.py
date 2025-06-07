"""
Test suite for Pydantic models and data validation.
Tests the robustness of the new models against various edge cases.
"""

import pytest
from typing import Dict, Any
from pydantic import ValidationError

from src.models import (
    VideoAnalysis, TextOverlay, NarrativeSegment, VisualCue,
    HookMoment, AudioHook, VideoSegment, ThumbnailInfo, CallToAction,
    FocusPoint, SpeedEffect, SoundEffect, PositionType, TextStyle,
    EffectType, EmotionType, PacingType
)


class TestTextOverlay:
    """Test TextOverlay model validation"""
    
    def test_valid_text_overlay(self):
        """Test creation of valid text overlay"""
        overlay = TextOverlay(
            text="Amazing moment!",
            timestamp_seconds=5.0,
            duration=2.0,
            position="center",
            style="dramatic"
        )
        assert overlay.text == "Amazing moment!"
        assert overlay.timestamp_seconds == 5.0
        assert overlay.duration == 2.0
        assert overlay.position == PositionType.CENTER
        assert overlay.style == TextStyle.DRAMATIC
    
    def test_text_overlay_validation_errors(self):
        """Test text overlay validation failures"""
        # Empty text
        with pytest.raises(ValidationError, match="at least 1 character"):
            TextOverlay(text="", timestamp_seconds=0, duration=1)
        
        # Negative timestamp
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            TextOverlay(text="Test", timestamp_seconds=-1, duration=1)
        
        # Duration too long
        with pytest.raises(ValidationError, match="less than or equal to 10"):
            TextOverlay(text="Test", timestamp_seconds=0, duration=15)
        
        # Text too long
        with pytest.raises(ValidationError, match="at most 200 characters"):
            TextOverlay(text="x" * 201, timestamp_seconds=0, duration=1)
    
    def test_text_overlay_whitespace_handling(self):
        """Test that whitespace is properly handled"""
        overlay = TextOverlay(
            text="  Trimmed text  ",
            timestamp_seconds=0,
            duration=1
        )
        assert overlay.text == "Trimmed text"
    
    def test_text_overlay_enum_validation(self):
        """Test enum validation for position and style"""
        # Valid enums
        overlay = TextOverlay(
            text="Test",
            timestamp_seconds=0,
            duration=1,
            position="top",
            style="highlight"
        )
        assert overlay.position == PositionType.TOP
        assert overlay.style == TextStyle.HIGHLIGHT
        
        # Invalid enum values should use defaults or raise errors
        with pytest.raises(ValidationError):
            TextOverlay(
                text="Test",
                timestamp_seconds=0,
                duration=1,
                position="invalid_position"
            )


class TestNarrativeSegment:
    """Test NarrativeSegment model validation"""
    
    def test_valid_narrative_segment(self):
        """Test creation of valid narrative segment"""
        segment = NarrativeSegment(
            text="This is an exciting moment!",
            time_seconds=3.0,
            intended_duration_seconds=4.0,
            emotion="excited",
            pacing="fast"
        )
        assert segment.text == "This is an exciting moment!"
        assert segment.time_seconds == 3.0
        assert segment.intended_duration_seconds == 4.0
        assert segment.emotion == EmotionType.EXCITED
        assert segment.pacing == PacingType.FAST
    
    def test_narrative_segment_validation_errors(self):
        """Test narrative segment validation failures"""
        # Text too long
        with pytest.raises(ValidationError):
            NarrativeSegment(
                text="x" * 501,
                time_seconds=0,
                intended_duration_seconds=1
            )
        
        # Negative time
        with pytest.raises(ValidationError):
            NarrativeSegment(
                text="Test",
                time_seconds=-1,
                intended_duration_seconds=1
            )
        
        # Duration too long
        with pytest.raises(ValidationError):
            NarrativeSegment(
                text="Test",
                time_seconds=0,
                intended_duration_seconds=31
            )


class TestVideoAnalysis:
    """Test VideoAnalysis model validation"""
    
    def test_valid_video_analysis(self):
        """Test creation of valid video analysis"""
        analysis = VideoAnalysis(
            suggested_title="Amazing Video",
            summary_for_description="This is a great video",
            mood="exciting",
            has_clear_narrative=True,
            original_audio_is_key=False,
            hook_text="Watch this!",
            hook_variations=["Amazing!", "Incredible!"],
            visual_hook_moment=HookMoment(timestamp_seconds=0.0, description="Start"),
            audio_hook=AudioHook(type="sound", sound_name="whoosh", timestamp_seconds=0.0),
            best_segment=VideoSegment(start_seconds=0, end_seconds=30, reason="Best part"),
            segments=[VideoSegment(start_seconds=0, end_seconds=60, reason="Full video")],
            music_genres=["upbeat"],
            hashtags=["#amazing", "#video"],
            thumbnail_info=ThumbnailInfo(timestamp_seconds=5.0, reason="Good frame"),
            call_to_action=CallToAction(text="Subscribe!", type="subscribe")
        )
        
        assert analysis.suggested_title == "Amazing Video"
        assert len(analysis.hook_variations) == 2
        assert analysis.hashtags[0] == "#amazing"
    
    def test_video_analysis_title_length_validation(self):
        """Test title length validation"""
        # Title too long
        with pytest.raises(ValidationError):
            VideoAnalysis(
                suggested_title="x" * 101,  # Too long
                summary_for_description="Description",
                mood="neutral",
                has_clear_narrative=False,
                original_audio_is_key=True,
                hook_text="Hook",
                hook_variations=["Hook1"],
                visual_hook_moment=HookMoment(timestamp_seconds=0.0, description="Start"),
                audio_hook=AudioHook(type="sound", sound_name="sound", timestamp_seconds=0.0),
                best_segment=VideoSegment(start_seconds=0, end_seconds=30, reason="Best"),
                segments=[VideoSegment(start_seconds=0, end_seconds=60, reason="Full")],
                music_genres=["background"],
                hashtags=["#test"],
                thumbnail_info=ThumbnailInfo(timestamp_seconds=0.0, reason="Default"),
                call_to_action=CallToAction(text="Action", type="general")
            )
    
    def test_hashtag_validation_and_cleaning(self):
        """Test hashtag validation and automatic cleaning"""
        analysis = VideoAnalysis(
            suggested_title="Test Video",
            summary_for_description="Test description",
            mood="neutral",
            has_clear_narrative=False,
            original_audio_is_key=True,
            hook_text="Hook",
            hook_variations=["Hook1"],
            visual_hook_moment=HookMoment(timestamp_seconds=0.0, description="Start"),
            audio_hook=AudioHook(type="sound", sound_name="sound", timestamp_seconds=0.0),
            best_segment=VideoSegment(start_seconds=0, end_seconds=30, reason="Best"),
            segments=[VideoSegment(start_seconds=0, end_seconds=60, reason="Full")],
            music_genres=["background"],
            hashtags=["amazing", "#viral", "test with spaces", "#special!@#", ""],
            thumbnail_info=ThumbnailInfo(timestamp_seconds=0.0, reason="Default"),
            call_to_action=CallToAction(text="Action", type="general")
        )
        
        # Should clean up hashtags automatically
        expected_hashtags = ["#amazing", "#viral", "#testwithspaces", "#special"]
        assert len(analysis.hashtags) == len(expected_hashtags)
        for tag in expected_hashtags:
            assert tag in analysis.hashtags
    
    def test_hook_variations_uniqueness(self):
        """Test that hook variations must be unique"""
        with pytest.raises(ValidationError, match="must be unique"):
            VideoAnalysis(
                suggested_title="Test",
                summary_for_description="Test",
                mood="neutral",
                has_clear_narrative=False,
                original_audio_is_key=True,
                hook_text="Hook",
                hook_variations=["Same", "Same", "Different"],  # Duplicates
                visual_hook_moment=HookMoment(timestamp_seconds=0.0, description="Start"),
                audio_hook=AudioHook(type="sound", sound_name="sound", timestamp_seconds=0.0),
                best_segment=VideoSegment(start_seconds=0, end_seconds=30, reason="Best"),
                segments=[VideoSegment(start_seconds=0, end_seconds=60, reason="Full")],
                music_genres=["background"],
                hashtags=["#test"],
                thumbnail_info=ThumbnailInfo(timestamp_seconds=0.0, reason="Default"),
                call_to_action=CallToAction(text="Action", type="general")
            )
    
    def test_video_analysis_consistency_validation(self):
        """Test root validator for analysis consistency"""
        # Too many visual cues
        with pytest.raises(ValidationError, match="Too many visual cues"):
            visual_cues = [
                VisualCue(timestamp_seconds=i, description=f"Cue {i}", effect_type="zoom")
                for i in range(25)  # Over limit of 20
            ]
            
            VideoAnalysis(
                suggested_title="Test",
                summary_for_description="Test",
                mood="neutral",
                has_clear_narrative=False,
                original_audio_is_key=True,
                hook_text="Hook",
                hook_variations=["Hook1"],
                visual_hook_moment=HookMoment(timestamp_seconds=0.0, description="Start"),
                audio_hook=AudioHook(type="sound", sound_name="sound", timestamp_seconds=0.0),
                best_segment=VideoSegment(start_seconds=0, end_seconds=30, reason="Best"),
                segments=[VideoSegment(start_seconds=0, end_seconds=60, reason="Full")],
                visual_cues=visual_cues,  # Too many
                music_genres=["background"],
                hashtags=["#test"],
                thumbnail_info=ThumbnailInfo(timestamp_seconds=0.0, reason="Default"),
                call_to_action=CallToAction(text="Action", type="general")
            )


class TestSpeedEffect:
    """Test SpeedEffect model validation"""
    
    def test_valid_speed_effect(self):
        """Test creation of valid speed effect"""
        effect = SpeedEffect(
            start_seconds=10.0,
            end_seconds=15.0,
            speed_factor=0.5
        )
        assert effect.start_seconds == 10.0
        assert effect.end_seconds == 15.0
        assert effect.speed_factor == 0.5
    
    def test_speed_effect_time_validation(self):
        """Test that end time must be greater than start time"""
        with pytest.raises(ValidationError, match="greater than start time"):
            SpeedEffect(
                start_seconds=15.0,
                end_seconds=10.0,  # End before start
                speed_factor=1.0
            )
    
    def test_speed_factor_bounds(self):
        """Test speed factor bounds"""
        # Too slow
        with pytest.raises(ValidationError):
            SpeedEffect(
                start_seconds=0,
                end_seconds=1,
                speed_factor=0.05  # Below minimum
            )
        
        # Too fast
        with pytest.raises(ValidationError):
            SpeedEffect(
                start_seconds=0,
                end_seconds=1,
                speed_factor=6.0  # Above maximum
            )


class TestFocusPoint:
    """Test FocusPoint model validation"""
    
    def test_valid_focus_point(self):
        """Test creation of valid focus point"""
        point = FocusPoint(
            x=0.5,
            y=0.3,
            timestamp_seconds=5.0,
            description="Center focus"
        )
        assert point.x == 0.5
        assert point.y == 0.3
        assert point.timestamp_seconds == 5.0
        assert point.description == "Center focus"
    
    def test_focus_point_coordinate_bounds(self):
        """Test coordinate bounds (0.0 to 1.0)"""
        # Valid bounds
        point = FocusPoint(x=0.0, y=1.0, timestamp_seconds=0, description="Corner")
        assert point.x == 0.0
        assert point.y == 1.0
        
        # Invalid bounds
        with pytest.raises(ValidationError):
            FocusPoint(x=-0.1, y=0.5, timestamp_seconds=0, description="Invalid")
        
        with pytest.raises(ValidationError):
            FocusPoint(x=0.5, y=1.1, timestamp_seconds=0, description="Invalid")


def test_fallback_defaults():
    """Test the fallback defaults functionality"""
    analysis = VideoAnalysis(
        suggested_title="Test",
        summary_for_description="Test",
        mood="neutral",
        has_clear_narrative=False,
        original_audio_is_key=True,
        hook_text="Hook",
        hook_variations=["Hook1"],
        visual_hook_moment=HookMoment(timestamp_seconds=0.0, description="Start"),
        audio_hook=AudioHook(type="sound", sound_name="sound", timestamp_seconds=0.0),
        best_segment=VideoSegment(start_seconds=0, end_seconds=30, reason="Best"),
        segments=[VideoSegment(start_seconds=0, end_seconds=60, reason="Full")],
        music_genres=["background"],
        hashtags=["#test"],
        thumbnail_info=ThumbnailInfo(timestamp_seconds=0.0, reason="Default"),
        call_to_action=CallToAction(text="Action", type="general")
    )
    
    fallback_defaults = analysis.get_fallback_defaults()
    
    assert "suggested_title" in fallback_defaults
    assert "hook_text" in fallback_defaults
    assert "hashtags" in fallback_defaults
    assert fallback_defaults["fallback"] is True


if __name__ == "__main__":
    pytest.main([__file__])