"""
Integration tests for the enhanced VideoProcessor with Pydantic models,
improved error handling, and temporary file management.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from moviepy.editor import VideoFileClip, ColorClip

from src.processing.video_processor import VideoProcessor, TemporaryFileManager
from src.models import (
    VideoAnalysis, TextOverlay, NarrativeSegment, VisualCue,
    HookMoment, AudioHook, VideoSegment, ThumbnailInfo, CallToAction
)


class TestTemporaryFileManager:
    """Test the TemporaryFileManager functionality"""
    
    def test_temp_file_creation_and_cleanup(self):
        """Test temporary file creation and cleanup"""
        with TemporaryFileManager() as temp_manager:
            # Create temporary files
            temp_file1 = temp_manager.create_temp_file(suffix=".mp4")
            temp_file2 = temp_manager.create_temp_file(suffix=".mp3")
            
            # Create the files to simulate actual usage
            temp_file1.touch()
            temp_file2.touch()
            
            assert temp_file1.exists()
            assert temp_file2.exists()
            assert len(temp_manager.temp_files) == 2
        
        # Files should be cleaned up after context exit
        assert not temp_file1.exists()
        assert not temp_file2.exists()
    
    def test_temp_directory_creation_and_cleanup(self):
        """Test temporary directory creation and cleanup"""
        with TemporaryFileManager() as temp_manager:
            temp_dir = temp_manager.create_temp_dir()
            
            # Create some content in the directory
            test_file = temp_dir / "test.txt"
            test_file.write_text("test content")
            
            assert temp_dir.exists()
            assert test_file.exists()
            assert len(temp_manager.temp_dirs) == 1
        
        # Directory should be cleaned up after context exit
        assert not temp_dir.exists()
    
    def test_manual_file_registration(self):
        """Test manual file registration for cleanup"""
        with TemporaryFileManager() as temp_manager:
            # Create a file manually
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                temp_path = Path(tmp.name)
            
            # Register it for cleanup
            temp_manager.register_file(temp_path)
            
            assert temp_path.exists()
            assert temp_path in temp_manager.temp_files
        
        # Should be cleaned up
        assert not temp_path.exists()


class TestVideoProcessorValidation:
    """Test VideoProcessor input validation"""
    
    @pytest.fixture
    def processor(self):
        """Create a VideoProcessor instance for testing"""
        return VideoProcessor()
    
    @pytest.fixture
    def valid_analysis(self):
        """Create a valid VideoAnalysis for testing"""
        return VideoAnalysis(
            suggested_title="Test Video",
            summary_for_description="A test video for validation",
            mood="neutral",
            has_clear_narrative=False,
            original_audio_is_key=True,
            hook_text="Test hook",
            hook_variations=["Hook 1", "Hook 2"],
            visual_hook_moment=HookMoment(timestamp_seconds=0.0, description="Start"),
            audio_hook=AudioHook(type="sound", sound_name="test", timestamp_seconds=0.0),
            best_segment=VideoSegment(start_seconds=0, end_seconds=30, reason="Test segment"),
            segments=[VideoSegment(start_seconds=0, end_seconds=60, reason="Full video")],
            music_genres=["background"],
            hashtags=["#test"],
            thumbnail_info=ThumbnailInfo(timestamp_seconds=0.0, reason="Test thumbnail"),
            call_to_action=CallToAction(text="Test CTA", type="general")
        )
    
    def test_validate_inputs_missing_video(self, processor, valid_analysis, tmp_path):
        """Test validation with missing video file"""
        non_existent_video = tmp_path / "missing.mp4"
        output_path = tmp_path / "output.mp4"
        
        result = processor._validate_inputs(non_existent_video, output_path, valid_analysis)
        assert result is False
    
    def test_validate_inputs_none_analysis(self, processor, tmp_path):
        """Test validation with None analysis"""
        video_path = tmp_path / "test.mp4"
        video_path.touch()  # Create empty file
        output_path = tmp_path / "output.mp4"
        
        result = processor._validate_inputs(video_path, output_path, None)
        assert result is False
    
    def test_validate_inputs_valid(self, processor, valid_analysis, tmp_path):
        """Test validation with valid inputs"""
        video_path = tmp_path / "test.mp4"
        video_path.touch()  # Create empty file
        output_path = tmp_path / "output.mp4"
        
        result = processor._validate_inputs(video_path, output_path, valid_analysis)
        assert result is True
        assert output_path.parent.exists()  # Output directory should be created


class TestVideoProcessorErrorHandling:
    """Test error handling in VideoProcessor"""
    
    @pytest.fixture
    def processor(self):
        return VideoProcessor()
    
    @pytest.fixture
    def mock_video_clip(self):
        """Create a mock video clip"""
        clip = Mock(spec=VideoFileClip)
        clip.duration = 60.0
        clip.size = (1920, 1080)
        clip.audio = Mock()
        return clip
    
    def test_load_and_prepare_video_success(self, processor, tmp_path):
        """Test successful video loading"""
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        
        with patch('src.processing.video_processor.VideoFileClip') as mock_video_clip:
            mock_clip = Mock()
            mock_video_clip.return_value = mock_clip
            
            with patch.object(processor, '_prepare_video_clip', return_value=mock_clip):
                resource_manager = Mock()
                result = processor._load_and_prepare_video(video_path, resource_manager)
                
                assert result == mock_clip
                resource_manager.register_clip.assert_called()
    
    def test_load_and_prepare_video_failure(self, processor, tmp_path):
        """Test video loading failure"""
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        
        with patch('src.processing.video_processor.VideoFileClip', side_effect=Exception("Load failed")):
            resource_manager = Mock()
            result = processor._load_and_prepare_video(video_path, resource_manager)
            
            assert result is None
    
    def test_apply_visual_effects_with_failures(self, processor, mock_video_clip):
        """Test visual effects with some failures"""
        # Create a mock analysis with visual cues
        analysis = Mock()
        analysis.visual_cues = [Mock()]
        analysis.speed_effects = [Mock()]
        
        resource_manager = Mock()
        
        # Mock effects to raise exceptions
        processor.effects.add_visual_cues = Mock(side_effect=Exception("Visual cues failed"))
        processor.effects.apply_speed_effects = Mock(side_effect=Exception("Speed effects failed"))
        processor.effects.apply_subtle_zoom = Mock(return_value=mock_video_clip)
        processor.effects.apply_color_grading = Mock(return_value=mock_video_clip)
        
        result = processor._apply_visual_effects(mock_video_clip, analysis, resource_manager)
        
        # Should return original clip despite failures
        assert result == mock_video_clip
    
    def test_process_audio_with_fallback(self, processor, mock_video_clip):
        """Test audio processing with fallback to original audio"""
        analysis = Mock()
        analysis.narrative_script_segments = []
        
        resource_manager = Mock()
        temp_manager = Mock()
        
        # Mock audio processor to fail
        processor.audio_processor.process_audio = Mock(side_effect=Exception("Audio failed"))
        
        result = processor._process_audio(
            mock_video_clip, analysis, None, resource_manager, temp_manager
        )
        
        # Should fallback to original clip
        assert result == mock_video_clip
    
    def test_write_video_with_retry_success_on_second_attempt(self, processor, tmp_path):
        """Test video writing success on retry"""
        output_path = tmp_path / "output.mp4"
        mock_clip = Mock()
        
        temp_manager = TemporaryFileManager()
        
        # Mock _write_video to fail first time, succeed second time
        call_count = 0
        def mock_write_video(clip, path):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First attempt failed")
            else:
                # Create the file to simulate success
                path.touch()
        
        processor._write_video = Mock(side_effect=mock_write_video)
        
        with patch('shutil.move') as mock_move:
            result = processor._write_video_with_retry(mock_clip, output_path, temp_manager)
        
        assert result is True
        assert processor._write_video.call_count == 2  # Should retry once
        mock_move.assert_called_once()
    
    def test_write_video_with_retry_all_attempts_fail(self, processor, tmp_path):
        """Test video writing when all attempts fail"""
        output_path = tmp_path / "output.mp4"
        mock_clip = Mock()
        
        temp_manager = TemporaryFileManager()
        
        # Mock _write_video to always fail
        processor._write_video = Mock(side_effect=Exception("Always fails"))
        
        result = processor._write_video_with_retry(mock_clip, output_path, temp_manager, max_retries=1)
        
        assert result is False
        assert processor._write_video.call_count == 2  # Original + 1 retry


class TestVideoProcessorIntegration:
    """Integration tests for the complete video processing pipeline"""
    
    @pytest.fixture
    def processor(self):
        return VideoProcessor()
    
    @pytest.fixture
    def valid_analysis(self):
        return VideoAnalysis(
            suggested_title="Integration Test Video",
            summary_for_description="Testing the complete pipeline",
            mood="exciting",
            has_clear_narrative=True,
            original_audio_is_key=False,
            hook_text="Amazing test!",
            hook_variations=["Incredible!", "Must see!"],
            visual_hook_moment=HookMoment(timestamp_seconds=1.0, description="Hook moment"),
            audio_hook=AudioHook(type="sound", sound_name="impact", timestamp_seconds=0.5),
            best_segment=VideoSegment(start_seconds=5, end_seconds=25, reason="Best action"),
            segments=[
                VideoSegment(start_seconds=0, end_seconds=30, reason="Full clip")
            ],
            text_overlays=[
                TextOverlay(
                    text="AMAZING!",
                    timestamp_seconds=3.0,
                    duration=2.0,
                    position="center",
                    style="dramatic"
                )
            ],
            narrative_script_segments=[
                NarrativeSegment(
                    text="This is an incredible moment!",
                    time_seconds=2.0,
                    intended_duration_seconds=3.0,
                    emotion="excited",
                    pacing="normal"
                )
            ],
            visual_cues=[
                VisualCue(
                    timestamp_seconds=4.0,
                    description="Zoom effect",
                    effect_type="zoom",
                    intensity=1.2,
                    duration=1.5
                )
            ],
            music_genres=["upbeat", "energetic"],
            hashtags=["#amazing", "#test", "#integration"],
            thumbnail_info=ThumbnailInfo(
                timestamp_seconds=10.0,
                reason="Most engaging frame",
                headline_text="WOW!"
            ),
            call_to_action=CallToAction(
                text="Subscribe for more!",
                type="subscribe"
            )
        )
    
    @patch('src.processing.video_processor.VideoFileClip')
    @patch('moviepy.editor.ColorClip')
    def test_process_video_complete_pipeline(self, mock_color_clip, mock_video_clip, 
                                           processor, valid_analysis, tmp_path):
        """Test the complete video processing pipeline"""
        # Setup
        video_path = tmp_path / "input.mp4"
        video_path.touch()
        output_path = tmp_path / "output.mp4"
        
        # Mock video clip
        mock_clip = Mock()
        mock_clip.duration = 30.0
        mock_clip.size = (1920, 1080)
        mock_clip.audio = Mock()
        mock_video_clip.return_value = mock_clip
        
        # Mock all the processing methods to return the clip
        processor._prepare_video_clip = Mock(return_value=mock_clip)
        processor.effects.add_visual_cues = Mock(return_value=mock_clip)
        processor.effects.apply_speed_effects = Mock(return_value=mock_clip)
        processor.effects.apply_subtle_zoom = Mock(return_value=mock_clip)
        processor.effects.apply_color_grading = Mock(return_value=mock_clip)
        processor.text_processor.add_text_overlays = Mock(return_value=mock_clip)
        processor.audio_processor.process_audio = Mock(return_value=mock_clip.audio)
        
        # Mock video writing
        def mock_write_video(clip, path):
            path.touch()
        processor._write_video = Mock(side_effect=mock_write_video)
        
        # Execute
        result = processor.process_video(video_path, output_path, valid_analysis)
        
        # Verify
        assert result is True
        assert output_path.exists()
        
        # Verify all processing steps were called
        processor._prepare_video_clip.assert_called_once()
        processor.effects.add_visual_cues.assert_called_once()
        processor.text_processor.add_text_overlays.assert_called_once()
        processor.audio_processor.process_audio.assert_called_once()
    
    def test_process_video_with_invalid_analysis(self, processor, tmp_path):
        """Test processing with invalid analysis data"""
        video_path = tmp_path / "input.mp4"
        video_path.touch()
        output_path = tmp_path / "output.mp4"
        
        # Create analysis with validation errors
        try:
            invalid_analysis = VideoAnalysis(
                suggested_title="",  # Empty title should fail validation
                summary_for_description="Test",
                mood="test",
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
        except Exception:
            # If Pydantic validation prevents creation, that's expected
            invalid_analysis = None
        
        result = processor.process_video(video_path, output_path, invalid_analysis)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__])