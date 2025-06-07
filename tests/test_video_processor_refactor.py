"""
Tests for the refactored VideoProcessor class focusing on the new modular structure.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.processing.video_processor import VideoProcessor, ResourceManager, TemporaryFileManager
from src.models import (
    VideoAnalysis, VideoSegment, TextOverlay, NarrativeSegment,
    HookMoment, AudioHook, ThumbnailInfo, CallToAction
)


class TestVideoProcessorModularStructure:
    """Test the modular structure of the refactored VideoProcessor"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = VideoProcessor()
        self.test_analysis = self._create_test_analysis()
    
    def _create_test_analysis(self):
        """Create test analysis object"""
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
            best_segment=VideoSegment(start_seconds=0.0, end_seconds=30.0, reason="Best part"),
            segments=[VideoSegment(start_seconds=0.0, end_seconds=30.0, reason="Test segment")],
            music_genres=["upbeat"],
            hashtags=["#test"],
            thumbnail_info=ThumbnailInfo(timestamp_seconds=5.0, reason="Test thumbnail"),
            call_to_action=CallToAction(text="Subscribe!", type="subscribe"),
            text_overlays=[
                TextOverlay(text="Amazing!", timestamp_seconds=2.0, duration=1.5)
            ],
            narrative_script_segments=[
                NarrativeSegment(text="This is incredible!", time_seconds=1.0, intended_duration_seconds=2.0)
            ]
        )


class TestLoadSourceClip:
    """Test the _load_source_clip method (Stage 1)"""
    
    def setup_method(self):
        self.processor = VideoProcessor()
        self.test_analysis = self._create_test_analysis()
    
    @patch('src.processing.video_processor.VideoFileClip')
    def test_successful_clip_loading(self, mock_video_clip):
        """Test successful video clip loading"""
        # Setup mocks
        mock_clip = Mock()
        mock_clip.duration = 60.0
        mock_video_clip.return_value = mock_clip
        
        mock_resource_manager = Mock()
        
        # Mock validation and preparation methods
        with patch.object(self.processor, '_validate_inputs', return_value=True), \
             patch.object(self.processor, '_load_and_prepare_video', return_value=mock_clip):
            
            result = self.processor._load_source_clip(
                Path("test.mp4"), 
                Path("output.mp4"), 
                self.test_analysis, 
                mock_resource_manager
            )
        
        assert result == mock_clip
        mock_resource_manager.register_clip.assert_called()
    
    def test_validation_failure(self):
        """Test handling of validation failure"""
        mock_resource_manager = Mock()
        
        with patch.object(self.processor, '_validate_inputs', return_value=False):
            result = self.processor._load_source_clip(
                Path("test.mp4"),
                Path("output.mp4"),
                self.test_analysis,
                mock_resource_manager
            )
        
        assert result is None
    
    def test_clip_loading_failure(self):
        """Test handling of clip loading failure"""
        mock_resource_manager = Mock()
        
        with patch.object(self.processor, '_validate_inputs', return_value=True), \
             patch.object(self.processor, '_load_and_prepare_video', return_value=None):
            
            result = self.processor._load_source_clip(
                Path("test.mp4"),
                Path("output.mp4"),
                self.test_analysis,
                mock_resource_manager
            )
        
        assert result is None
    
    def _create_test_analysis(self):
        """Create test analysis object"""
        return VideoAnalysis(
            suggested_title="Test Video",
            summary_for_description="Test description",
            mood="exciting",
            has_clear_narrative=True,
            original_audio_is_key=False,
            hook_text="Amazing!",
            hook_variations=["Wow!"],
            visual_hook_moment=HookMoment(timestamp_seconds=1.0, description="Opening"),
            audio_hook=AudioHook(type="sound", sound_name="whoosh", timestamp_seconds=0.0),
            best_segment=VideoSegment(start_seconds=0.0, end_seconds=30.0, reason="Best part"),
            segments=[VideoSegment(start_seconds=0.0, end_seconds=30.0, reason="Test segment")],
            music_genres=["upbeat"],
            hashtags=["#test"],
            thumbnail_info=ThumbnailInfo(timestamp_seconds=5.0, reason="Test thumbnail"),
            call_to_action=CallToAction(text="Subscribe!", type="subscribe")
        )


class TestSynthesizeAudio:
    """Test the _synthesize_audio method (Stage 2)"""
    
    def setup_method(self):
        self.processor = VideoProcessor()
        self.test_analysis = self._create_test_analysis_with_audio()
    
    def test_audio_synthesis_with_original_audio(self):
        """Test audio synthesis when original audio is key"""
        mock_source_audio = Mock()
        mock_source_audio.duration = 30.0
        mock_resource_manager = Mock()
        mock_temp_manager = Mock()
        
        # Analysis with original audio as key
        self.test_analysis.original_audio_is_key = True
        
        with patch.object(self.processor, '_process_tts_segments', return_value=[]), \
             patch.object(self.processor, '_add_background_music', return_value=None):
            
            result = self.processor._synthesize_audio(
                mock_source_audio,
                self.test_analysis,
                None,
                mock_resource_manager,
                mock_temp_manager
            )
        
        assert result == mock_source_audio
    
    def test_audio_synthesis_with_tts(self):
        """Test audio synthesis with TTS narration"""
        mock_source_audio = Mock()
        mock_source_audio.duration = 30.0
        mock_resource_manager = Mock()
        mock_temp_manager = Mock()
        
        mock_tts_clip = Mock()
        
        # Analysis without original audio but with TTS
        self.test_analysis.original_audio_is_key = False
        
        with patch.object(self.processor, '_process_tts_segments', return_value=[mock_tts_clip]), \
             patch.object(self.processor, '_add_background_music', return_value=None), \
             patch('src.processing.video_processor.CompositeAudioClip') as mock_composite:
            
            mock_composite_result = Mock()
            mock_composite.return_value = mock_composite_result
            
            result = self.processor._synthesize_audio(
                mock_source_audio,
                self.test_analysis,
                None,
                mock_resource_manager,
                mock_temp_manager
            )
        
        mock_composite.assert_called_once()
        assert result == mock_composite_result
    
    def test_audio_synthesis_fallback_on_error(self):
        """Test audio synthesis fallback behavior on error"""
        mock_source_audio = Mock()
        mock_resource_manager = Mock()
        mock_temp_manager = Mock()
        
        with patch.object(self.processor, '_process_tts_segments', side_effect=Exception("TTS failed")):
            result = self.processor._synthesize_audio(
                mock_source_audio,
                self.test_analysis,
                None,
                mock_resource_manager,
                mock_temp_manager
            )
        
        # Should return source audio as fallback
        assert result == mock_source_audio
    
    def _create_test_analysis_with_audio(self):
        """Create test analysis with audio elements"""
        return VideoAnalysis(
            suggested_title="Test Video",
            summary_for_description="Test description",
            mood="exciting",
            has_clear_narrative=True,
            original_audio_is_key=False,
            hook_text="Amazing!",
            hook_variations=["Wow!"],
            visual_hook_moment=HookMoment(timestamp_seconds=1.0, description="Opening"),
            audio_hook=AudioHook(type="sound", sound_name="whoosh", timestamp_seconds=0.0),
            best_segment=VideoSegment(start_seconds=0.0, end_seconds=30.0, reason="Best part"),
            segments=[VideoSegment(start_seconds=0.0, end_seconds=30.0, reason="Test segment")],
            music_genres=["upbeat"],
            hashtags=["#test"],
            thumbnail_info=ThumbnailInfo(timestamp_seconds=5.0, reason="Test thumbnail"),
            call_to_action=CallToAction(text="Subscribe!", type="subscribe"),
            narrative_script_segments=[
                NarrativeSegment(text="This is incredible!", time_seconds=1.0, intended_duration_seconds=2.0)
            ]
        )


class TestComposeVisuals:
    """Test the _compose_visuals method (Stage 3)"""
    
    def setup_method(self):
        self.processor = VideoProcessor()
        self.test_analysis = self._create_test_analysis_with_visuals()
    
    def test_visual_composition_success(self):
        """Test successful visual composition"""
        mock_source_clip = Mock()
        mock_resource_manager = Mock()
        mock_enhanced_clip = Mock()
        
        with patch.object(self.processor, '_apply_visual_effects', return_value=mock_enhanced_clip), \
             patch.object(self.processor, '_add_text_overlays', return_value=mock_enhanced_clip), \
             patch.object(self.processor, '_apply_duration_constraints', return_value=mock_enhanced_clip):
            
            result = self.processor._compose_visuals(
                mock_source_clip,
                self.test_analysis,
                mock_resource_manager
            )
        
        assert result == mock_enhanced_clip
    
    def test_visual_composition_error_fallback(self):
        """Test visual composition with error fallback"""
        mock_source_clip = Mock()
        mock_resource_manager = Mock()
        
        with patch.object(self.processor, '_apply_visual_effects', side_effect=Exception("Visual effects failed")):
            result = self.processor._compose_visuals(
                mock_source_clip,
                self.test_analysis,
                mock_resource_manager
            )
        
        # Should return source clip as fallback
        assert result == mock_source_clip
    
    def _create_test_analysis_with_visuals(self):
        """Create test analysis with visual effects"""
        return VideoAnalysis(
            suggested_title="Test Video",
            summary_for_description="Test description",
            mood="exciting",
            has_clear_narrative=True,
            original_audio_is_key=False,
            hook_text="Amazing!",
            hook_variations=["Wow!"],
            visual_hook_moment=HookMoment(timestamp_seconds=1.0, description="Opening"),
            audio_hook=AudioHook(type="sound", sound_name="whoosh", timestamp_seconds=0.0),
            best_segment=VideoSegment(start_seconds=0.0, end_seconds=30.0, reason="Best part"),
            segments=[VideoSegment(start_seconds=0.0, end_seconds=30.0, reason="Test segment")],
            music_genres=["upbeat"],
            hashtags=["#test"],
            thumbnail_info=ThumbnailInfo(timestamp_seconds=5.0, reason="Test thumbnail"),
            call_to_action=CallToAction(text="Subscribe!", type="subscribe"),
            text_overlays=[
                TextOverlay(text="Amazing!", timestamp_seconds=2.0, duration=1.5)
            ]
        )


class TestRenderVideo:
    """Test the _render_video method (Stage 4)"""
    
    def setup_method(self):
        self.processor = VideoProcessor()
    
    def test_successful_rendering(self):
        """Test successful video rendering"""
        mock_final_clip = Mock()
        mock_temp_manager = Mock()
        output_path = Path("output.mp4")
        
        with patch.object(self.processor, '_write_video_with_retry', return_value=True):
            result = self.processor._render_video(
                mock_final_clip,
                output_path,
                mock_temp_manager
            )
        
        assert result is True
    
    def test_rendering_failure(self):
        """Test handling of rendering failure"""
        mock_final_clip = Mock()
        mock_temp_manager = Mock()
        output_path = Path("output.mp4")
        
        with patch.object(self.processor, '_write_video_with_retry', return_value=False):
            result = self.processor._render_video(
                mock_final_clip,
                output_path,
                mock_temp_manager
            )
        
        assert result is False
    
    def test_rendering_exception(self):
        """Test handling of rendering exception"""
        mock_final_clip = Mock()
        mock_temp_manager = Mock()
        output_path = Path("output.mp4")
        
        with patch.object(self.processor, '_write_video_with_retry', side_effect=Exception("Render failed")):
            result = self.processor._render_video(
                mock_final_clip,
                output_path,
                mock_temp_manager
            )
        
        assert result is False


class TestResourceManagement:
    """Test resource management in VideoProcessor"""
    
    def test_resource_manager_context(self):
        """Test ResourceManager context manager"""
        with ResourceManager() as rm:
            assert rm is not None
            assert hasattr(rm, 'active_clips')
            assert hasattr(rm, 'register_clip')
            assert hasattr(rm, 'cleanup_clips')
    
    def test_temporary_file_manager_context(self):
        """Test TemporaryFileManager context manager"""
        with TemporaryFileManager() as tm:
            assert tm is not None
            assert hasattr(tm, 'temp_files')
            assert hasattr(tm, 'temp_dirs')
            assert hasattr(tm, 'cleanup')
    
    def test_cleanup_method(self):
        """Test cleanup method execution"""
        processor = VideoProcessor()
        
        with patch('gc.collect') as mock_gc:
            processor._cleanup()
            mock_gc.assert_called_once()


class TestIntegrationWithNewStructure:
    """Test integration of the new modular structure"""
    
    def setup_method(self):
        self.processor = VideoProcessor()
        self.test_analysis = self._create_full_test_analysis()
    
    @patch('src.processing.video_processor.ResourceManager')
    @patch('src.processing.video_processor.TemporaryFileManager')
    def test_full_pipeline_integration(self, mock_temp_manager, mock_resource_manager):
        """Test the full processing pipeline with new structure"""
        # Setup mocks
        mock_rm = Mock()
        mock_tm = Mock()
        mock_resource_manager.return_value.__enter__ = Mock(return_value=mock_rm)
        mock_resource_manager.return_value.__exit__ = Mock(return_value=None)
        mock_temp_manager.return_value.__enter__ = Mock(return_value=mock_tm)
        mock_temp_manager.return_value.__exit__ = Mock(return_value=None)
        
        mock_video_clip = Mock()
        mock_audio_clip = Mock()
        mock_final_clip = Mock()
        
        # Mock all the stage methods
        with patch.object(self.processor, '_load_source_clip', return_value=mock_video_clip), \
             patch.object(self.processor, '_synthesize_audio', return_value=mock_audio_clip), \
             patch.object(self.processor, '_compose_visuals', return_value=mock_final_clip), \
             patch.object(self.processor, '_render_video', return_value=True), \
             patch.object(self.processor, '_cleanup'):
            
            # Mock the set_audio method
            mock_final_clip.set_audio.return_value = mock_final_clip
            
            result = self.processor.process_video(
                Path("input.mp4"),
                Path("output.mp4"),
                self.test_analysis
            )
        
        assert result is True
        
        # Verify all stages were called
        self.processor._load_source_clip.assert_called_once()
        self.processor._synthesize_audio.assert_called_once()
        self.processor._compose_visuals.assert_called_once()
        self.processor._render_video.assert_called_once()
        self.processor._cleanup.assert_called_once()
    
    def _create_full_test_analysis(self):
        """Create comprehensive test analysis"""
        return VideoAnalysis(
            suggested_title="Full Test Video",
            summary_for_description="Complete test description",
            mood="exciting",
            has_clear_narrative=True,
            original_audio_is_key=False,
            hook_text="Amazing content!",
            hook_variations=["Incredible!", "Must see!", "Wow!"],
            visual_hook_moment=HookMoment(timestamp_seconds=1.0, description="Opening moment"),
            audio_hook=AudioHook(type="sound_effect", sound_name="whoosh", timestamp_seconds=0.0),
            best_segment=VideoSegment(start_seconds=5.0, end_seconds=35.0, reason="Best content"),
            segments=[
                VideoSegment(start_seconds=0.0, end_seconds=30.0, reason="Opening"),
                VideoSegment(start_seconds=10.0, end_seconds=40.0, reason="Middle")
            ],
            music_genres=["upbeat", "energetic"],
            hashtags=["#test", "#video", "#amazing"],
            thumbnail_info=ThumbnailInfo(timestamp_seconds=15.0, reason="Best visual moment"),
            call_to_action=CallToAction(text="Subscribe for more amazing content!", type="subscribe"),
            text_overlays=[
                TextOverlay(text="Amazing!", timestamp_seconds=2.0, duration=1.5, position="center"),
                TextOverlay(text="Subscribe!", timestamp_seconds=25.0, duration=2.0, position="bottom")
            ],
            narrative_script_segments=[
                NarrativeSegment(text="Welcome to this incredible video!", time_seconds=1.0, intended_duration_seconds=3.0),
                NarrativeSegment(text="Don't forget to like and subscribe!", time_seconds=27.0, intended_duration_seconds=3.0)
            ]
        )