"""
Tests for utility functions in src/utils.py
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from src.utils import (
    select_and_validate_segments,
    validate_file_paths,
    get_safe_filename,
    calculate_video_metrics,
    validate_analysis_completeness,
    format_duration
)
from src.models import (
    VideoAnalysis, VideoSegment, TextOverlay, NarrativeSegment,
    VisualCue, SpeedEffect, HookMoment, AudioHook, ThumbnailInfo, CallToAction
)


class TestSegmentValidation:
    """Test segment validation functionality"""
    
    def test_select_valid_segments(self):
        """Test selection of valid segments"""
        # Create test analysis with valid segments
        best_segment = VideoSegment(start_seconds=5.0, end_seconds=35.0, reason="Best moment")
        segments = [
            VideoSegment(start_seconds=0.0, end_seconds=30.0, reason="Opening"),
            VideoSegment(start_seconds=10.0, end_seconds=50.0, reason="Middle section")
        ]
        
        analysis = self._create_minimal_analysis()
        analysis.best_segment = best_segment
        analysis.segments = segments
        
        config = {
            'video': {
                'max_short_duration_seconds': 60,
                'min_short_duration_seconds': 15
            }
        }
        
        valid_segments = select_and_validate_segments(analysis, config)
        
        assert len(valid_segments) > 0
        assert all(seg['duration_seconds'] >= 15 for seg in valid_segments)
        assert all(seg['duration_seconds'] <= 60 for seg in valid_segments)
        assert all(seg['start_seconds'] >= 0 for seg in valid_segments)
        assert all(seg['end_seconds'] > seg['start_seconds'] for seg in valid_segments)
    
    def test_reject_invalid_segments(self):
        """Test rejection of invalid segments"""
        # Create analysis with only the best segment being too short
        best_segment = VideoSegment(start_seconds=5.0, end_seconds=10.0, reason="Too short")  # 5 seconds
        segments = [
            VideoSegment(start_seconds=0.0, end_seconds=5.0, reason="Too short")  # Valid structure but too short
        ]
        
        analysis = self._create_minimal_analysis()
        analysis.best_segment = best_segment
        analysis.segments = segments
        
        config = {
            'video': {
                'max_short_duration_seconds': 60,
                'min_short_duration_seconds': 15
            }
        }
        
        valid_segments = select_and_validate_segments(analysis, config)
        
        # Should have no valid segments due to all being too short
        assert len(valid_segments) == 0
    
    def test_truncate_long_segments(self):
        """Test truncation of segments that are too long"""
        best_segment = VideoSegment(start_seconds=0.0, end_seconds=120.0, reason="Too long")
        
        analysis = self._create_minimal_analysis()
        analysis.best_segment = best_segment
        analysis.segments = [best_segment]
        
        config = {
            'video': {
                'max_short_duration_seconds': 60,
                'min_short_duration_seconds': 15
            }
        }
        
        valid_segments = select_and_validate_segments(analysis, config)
        
        assert len(valid_segments) == 1
        assert valid_segments[0]['duration_seconds'] == 60
        assert valid_segments[0]['end_seconds'] == 60
        assert 'truncated' in valid_segments[0]['reason']
    
    def _create_minimal_analysis(self) -> VideoAnalysis:
        """Create minimal valid VideoAnalysis for testing"""
        return VideoAnalysis(
            suggested_title="Test Video",
            summary_for_description="Test description",
            mood="exciting",
            has_clear_narrative=True,
            original_audio_is_key=False,
            hook_text="Amazing!",
            hook_variations=["Wow!", "Incredible!"],
            visual_hook_moment=HookMoment(timestamp_seconds=1.0, description="Opening"),
            audio_hook=AudioHook(type="sound", sound_name="whoosh", timestamp_seconds=0.0),
            best_segment=VideoSegment(start_seconds=0.0, end_seconds=30.0, reason="Test"),
            segments=[VideoSegment(start_seconds=0.0, end_seconds=30.0, reason="Test")],
            music_genres=["upbeat"],
            hashtags=["#test"],
            thumbnail_info=ThumbnailInfo(timestamp_seconds=5.0, reason="Test thumbnail"),
            call_to_action=CallToAction(text="Subscribe!", type="subscribe")
        )


class TestFileValidation:
    """Test file path validation"""
    
    def test_valid_file_paths(self, tmp_path):
        """Test validation of valid file paths"""
        input_file = tmp_path / "input.mp4"
        input_file.write_text("test content")
        
        output_file = tmp_path / "output" / "result.mp4"
        
        is_valid, error_msg = validate_file_paths(input_file, output_file)
        
        assert is_valid
        assert error_msg == ""
        assert output_file.parent.exists()  # Should create output directory
    
    def test_nonexistent_input_file(self, tmp_path):
        """Test validation with nonexistent input file"""
        input_file = tmp_path / "nonexistent.mp4"
        output_file = tmp_path / "output.mp4"
        
        is_valid, error_msg = validate_file_paths(input_file, output_file)
        
        assert not is_valid
        assert "does not exist" in error_msg
    
    def test_input_is_directory(self, tmp_path):
        """Test validation when input path is a directory"""
        input_dir = tmp_path / "input_dir"
        input_dir.mkdir()
        output_file = tmp_path / "output.mp4"
        
        is_valid, error_msg = validate_file_paths(input_dir, output_file)
        
        assert not is_valid
        assert "not a file" in error_msg


class TestSafeFilename:
    """Test safe filename generation"""
    
    def test_basic_filename(self):
        """Test basic filename sanitization"""
        title = "Amazing Video Title"
        result = get_safe_filename(title)
        assert result == "Amazing_Video_Title"
    
    def test_unsafe_characters(self):
        """Test removal of unsafe characters"""
        title = 'Video<>:"/\\|?*Title'
        result = get_safe_filename(title)
        assert result == "VideoTitle"
    
    def test_length_limiting(self):
        """Test filename length limiting"""
        long_title = "A" * 100
        result = get_safe_filename(long_title, max_length=20)
        assert len(result) == 20
    
    def test_empty_title(self):
        """Test handling of empty title"""
        result = get_safe_filename("")
        assert result == "video"
        
        result = get_safe_filename("   ")
        assert result == "video"


class TestVideoMetrics:
    """Test video metrics calculation"""
    
    def test_simple_metrics(self):
        """Test basic metrics calculation"""
        analysis = self._create_test_analysis_with_effects()
        metrics = calculate_video_metrics(analysis)
        
        assert metrics['total_segments'] == 2
        assert metrics['has_text_overlays'] == True
        assert metrics['has_narrative'] == True
        assert metrics['has_visual_effects'] == True
        assert metrics['has_speed_effects'] == True
        assert metrics['complexity_score'] > 0
        assert metrics['processing_priority'] in ['high', 'medium', 'low']
    
    def test_complexity_scoring(self):
        """Test complexity score calculation"""
        # Simple analysis
        simple_analysis = self._create_minimal_analysis()
        simple_metrics = calculate_video_metrics(simple_analysis)
        
        # Complex analysis
        complex_analysis = self._create_test_analysis_with_effects()
        complex_metrics = calculate_video_metrics(complex_analysis)
        
        assert complex_metrics['complexity_score'] > simple_metrics['complexity_score']
    
    def _create_test_analysis_with_effects(self):
        """Create analysis with various effects for testing"""
        analysis = self._create_minimal_analysis()
        
        # Add effects to increase complexity
        analysis.text_overlays = [
            TextOverlay(text="Amazing!", timestamp_seconds=1.0, duration=2.0)
        ]
        analysis.narrative_script_segments = [
            NarrativeSegment(text="This is amazing!", time_seconds=2.0, intended_duration_seconds=3.0)
        ]
        analysis.visual_cues = [
            VisualCue(timestamp_seconds=3.0, description="Zoom", effect_type="zoom")
        ]
        analysis.speed_effects = [
            SpeedEffect(start_seconds=5.0, end_seconds=10.0, speed_factor=0.5)
        ]
        
        return analysis
    
    def _create_minimal_analysis(self):
        """Create minimal analysis for testing"""
        return VideoAnalysis(
            suggested_title="Test",
            summary_for_description="Test",
            mood="neutral",
            has_clear_narrative=False,
            original_audio_is_key=True,
            hook_text="Hook",
            hook_variations=["Hook1"],
            visual_hook_moment=HookMoment(timestamp_seconds=0.0, description="Hook"),
            audio_hook=AudioHook(type="sound", sound_name="beep", timestamp_seconds=0.0),
            best_segment=VideoSegment(start_seconds=0.0, end_seconds=30.0, reason="Test"),
            segments=[
                VideoSegment(start_seconds=0.0, end_seconds=30.0, reason="First"),
                VideoSegment(start_seconds=30.0, end_seconds=60.0, reason="Second")
            ],
            music_genres=["ambient"],
            hashtags=["#test"],
            thumbnail_info=ThumbnailInfo(timestamp_seconds=0.0, reason="Test"),
            call_to_action=CallToAction(text="Like!", type="like")
        )


class TestAnalysisValidation:
    """Test analysis completeness validation"""
    
    def test_complete_analysis(self):
        """Test validation of complete analysis"""
        analysis = self._create_complete_analysis()
        is_complete, missing_items = validate_analysis_completeness(analysis)
        
        assert is_complete
        assert len(missing_items) == 0
    
    def test_incomplete_analysis(self):
        """Test validation of incomplete analysis with minimal valid structure"""
        # Create analysis that passes Pydantic validation but is logically incomplete
        analysis = VideoAnalysis(
            suggested_title="   ",  # Only whitespace - will be caught by our validator
            summary_for_description="Test",
            mood="neutral",  # Valid but we'll test if this gets caught as incomplete
            has_clear_narrative=False,
            original_audio_is_key=False,  # No audio content
            hook_text="Hook",
            hook_variations=["Hook1"],
            visual_hook_moment=HookMoment(timestamp_seconds=0.0, description="Hook"),
            audio_hook=AudioHook(type="sound", sound_name="beep", timestamp_seconds=0.0),
            best_segment=VideoSegment(start_seconds=0.0, end_seconds=30.0, reason="Test"),
            segments=[VideoSegment(start_seconds=0.0, end_seconds=30.0, reason="Minimal")],  # Minimal segments
            music_genres=["ambient"],
            hashtags=["#test"],  # Minimal hashtags
            thumbnail_info=ThumbnailInfo(timestamp_seconds=0.0, reason="Test"),
            call_to_action=CallToAction(text="Like!", type="like")
        )
        
        # Test that the analysis is structurally valid but logically incomplete
        is_complete, missing_items = validate_analysis_completeness(analysis)
        
        # Should detect missing audio content (no TTS and original_audio_is_key=False)
        assert not is_complete
        assert "audio_content" in missing_items
    
    def _create_complete_analysis(self):
        """Create complete analysis for testing"""
        return VideoAnalysis(
            suggested_title="Complete Test Video",
            summary_for_description="Complete test description",
            mood="exciting",
            has_clear_narrative=True,
            original_audio_is_key=True,
            hook_text="Amazing hook!",
            hook_variations=["Hook1", "Hook2"],
            visual_hook_moment=HookMoment(timestamp_seconds=1.0, description="Hook moment"),
            audio_hook=AudioHook(type="sound", sound_name="whoosh", timestamp_seconds=0.0),
            best_segment=VideoSegment(start_seconds=0.0, end_seconds=30.0, reason="Best part"),
            segments=[
                VideoSegment(start_seconds=0.0, end_seconds=30.0, reason="First segment"),
                VideoSegment(start_seconds=30.0, end_seconds=60.0, reason="Second segment")
            ],
            music_genres=["upbeat", "energetic"],
            hashtags=["#test", "#video", "#amazing"],
            thumbnail_info=ThumbnailInfo(timestamp_seconds=5.0, reason="Best moment"),
            call_to_action=CallToAction(text="Subscribe for more!", type="subscribe")
        )


class TestDurationFormatting:
    """Test duration formatting utility"""
    
    def test_seconds_format(self):
        """Test formatting of durations under 60 seconds"""
        assert format_duration(30) == "30s"
        assert format_duration(45.7) == "46s"
    
    def test_minutes_format(self):
        """Test formatting of durations over 60 seconds"""
        assert format_duration(90) == "1:30"
        assert format_duration(125) == "2:05"
        assert format_duration(3661) == "61:01"
    
    def test_zero_duration(self):
        """Test formatting of zero duration"""
        assert format_duration(0) == "0s"