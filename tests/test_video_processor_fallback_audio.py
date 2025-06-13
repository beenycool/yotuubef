"""
Test suite for video processor fallback audio functionality.
This test verifies that the _create_fallback_audio method correctly handles
MoviePy API calls without raising AttributeError exceptions.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import logging

# Configure test logging
logging.basicConfig(level=logging.DEBUG)

from src.processing.video_processor import VideoProcessor
from src.config.settings import get_config, ConfigManager


class MockResourceManager:
    """Mock resource manager for testing"""
    def __init__(self):
        self.clips = []
    
    def register_clip(self, clip):
        """Register a clip for cleanup"""
        self.clips.append(clip)
    
    def cleanup(self):
        """Cleanup all registered clips"""
        for clip in self.clips:
            try:
                if hasattr(clip, 'close'):
                    clip.close()
            except:
                pass
        self.clips.clear()


class MockAudioFileClip:
    """Mock AudioFileClip that simulates the real MoviePy AudioFileClip behavior"""
    
    def __init__(self, path, duration=30.0):
        self.path = path
        self.duration = duration
        self.fps = 44100
        self._closed = False
    
    def with_volume_scaled(self, factor):
        """Mock the correct volume scaling method"""
        new_clip = MockAudioFileClip(self.path, self.duration)
        new_clip._volume_factor = factor
        return new_clip
    
    def volumex(self, factor):
        """Mock the alternative volume method"""
        new_clip = MockAudioFileClip(self.path, self.duration)
        new_clip._volume_factor = factor
        return new_clip
    
    def close(self):
        """Mock close method"""
        self._closed = True
    
    def copy(self):
        """Mock copy method"""
        return MockAudioFileClip(self.path, self.duration)


class TestVideoProcessorFallbackAudio:
    """Test cases for the _create_fallback_audio method"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_music_file(self, temp_dir):
        """Create a mock music file"""
        music_file = temp_dir / "test_music.mp3"
        music_file.write_bytes(b"fake audio data")
        return music_file
    
    @pytest.fixture
    def video_processor(self, temp_dir):
        """Create a video processor instance with mock configuration"""
        with patch('src.processing.video_processor.get_config') as mock_get_config:
            # Mock the configuration manager
            mock_config = Mock()
            mock_config.audio = Mock()
            mock_config.audio.background_music_volume = 0.7
            mock_config.paths = Mock()
            mock_config.paths.music_folder = temp_dir
            mock_get_config.return_value = mock_config
            
            processor = VideoProcessor()
            processor.config = mock_config
            processor.logger = logging.getLogger(__name__)
            return processor
    
    @pytest.fixture
    def resource_manager(self):
        """Create a mock resource manager"""
        return MockResourceManager()
    
    def test_fallback_audio_with_shorter_music(self, video_processor, mock_music_file, resource_manager):
        """Test fallback audio creation when music is shorter than video duration"""
        video_duration = 60.0  # 60 seconds
        music_duration = 20.0  # 20 seconds - shorter than video
        
        # Mock the music directory to contain our test file
        video_processor.config.paths.music_folder = mock_music_file.parent
        
        # Mock the imports that happen inside the method
        with patch('moviepy.audio.io.AudioFileClip.AudioFileClip') as mock_audio_clip_class:
            with patch('moviepy.editor.concatenate_audioclips') as mock_concatenate:
                with patch('src.processing.video_processor_fixes.MoviePyCompat') as mock_compat:
                    
                    # Setup mocks
                    mock_clip = MockAudioFileClip(str(mock_music_file), music_duration)
                    mock_audio_clip_class.return_value = mock_clip
                    
                    # Mock concatenated clip (longer than original)
                    concatenated_clip = MockAudioFileClip(str(mock_music_file), 60.0)
                    mock_concatenate.return_value = concatenated_clip
                    
                    # Mock subclip to return properly trimmed clip
                    trimmed_clip = MockAudioFileClip(str(mock_music_file), video_duration)
                    mock_compat.subclip.return_value = trimmed_clip
                    
                    # Test the method
                    result = video_processor._create_fallback_audio(video_duration, resource_manager)
                    
                    # Verify the result
                    assert result is not None
                    assert mock_audio_clip_class.called
                    assert mock_concatenate.called
                    
                    # Verify concatenate was called with multiple clips (looping)
                    call_args = mock_concatenate.call_args[0][0]
                    assert len(call_args) >= 3  # Should have 3 copies (20*3=60 >= 60)
    
    def test_fallback_audio_with_longer_music(self, video_processor, mock_music_file, resource_manager):
        """Test fallback audio creation when music is longer than video duration"""
        video_duration = 30.0  # 30 seconds
        music_duration = 60.0  # 60 seconds - longer than video
        
        # Mock the music directory to contain our test file
        video_processor.config.paths.music_folder = mock_music_file.parent
        
        with patch('src.processing.video_processor.AudioFileClip') as mock_audio_clip_class:
            with patch('src.processing.video_processor.MoviePyCompat') as mock_compat:
                
                # Setup mocks
                mock_clip = MockAudioFileClip(str(mock_music_file), music_duration)
                mock_audio_clip_class.return_value = mock_clip
                
                # Mock subclip to return properly trimmed clip
                trimmed_clip = MockAudioFileClip(str(mock_music_file), video_duration)
                mock_compat.subclip.return_value = trimmed_clip
                
                # Test the method
                result = video_processor._create_fallback_audio(video_duration, resource_manager)
                
                # Verify the result
                assert result is not None
                assert mock_audio_clip_class.called
                assert mock_compat.subclip.called
                
                # Verify subclip was called with correct parameters
                mock_compat.subclip.assert_called_with(mock_clip, 0, video_duration)
    
    def test_fallback_audio_volume_scaling(self, video_processor, mock_music_file, resource_manager):
        """Test that volume scaling uses the correct MoviePy methods"""
        video_duration = 30.0
        music_duration = 30.0  # Same duration
        
        # Mock the music directory to contain our test file
        video_processor.config.paths.music_folder = mock_music_file.parent
        
        with patch('src.processing.video_processor.AudioFileClip') as mock_audio_clip_class:
            
            # Setup mock that supports with_volume_scaled
            mock_clip = MockAudioFileClip(str(mock_music_file), music_duration)
            mock_audio_clip_class.return_value = mock_clip
            
            # Test the method
            result = video_processor._create_fallback_audio(video_duration, resource_manager)
            
            # Verify the result
            assert result is not None
            assert hasattr(result, '_volume_factor')
            assert result._volume_factor == video_processor.config.audio.background_music_volume
    
    def test_fallback_audio_volume_scaling_fallback(self, video_processor, mock_music_file, resource_manager):
        """Test volume scaling fallback when with_volume_scaled fails"""
        video_duration = 30.0
        music_duration = 30.0
        
        # Mock the music directory to contain our test file
        video_processor.config.paths.music_folder = mock_music_file.parent
        
        with patch('src.processing.video_processor.AudioFileClip') as mock_audio_clip_class:
            
            # Setup mock that fails with_volume_scaled but supports volumex
            mock_clip = Mock()
            mock_clip.duration = music_duration
            mock_clip.with_volume_scaled.side_effect = Exception("Method not available")
            
            # Mock successful volumex fallback
            volumex_result = MockAudioFileClip(str(mock_music_file), music_duration)
            mock_clip.volumex.return_value = volumex_result
            
            mock_audio_clip_class.return_value = mock_clip
            
            # Test the method
            result = video_processor._create_fallback_audio(video_duration, resource_manager)
            
            # Verify the result
            assert result is not None
            assert mock_clip.volumex.called
    
    def test_fallback_audio_no_music_files(self, video_processor, temp_dir, resource_manager):
        """Test fallback audio when no music files are available"""
        # Create empty music directory
        empty_music_dir = temp_dir / "empty_music"
        empty_music_dir.mkdir()
        video_processor.config.paths.music_folder = empty_music_dir
        
        video_duration = 30.0
        
        # Test the method
        result = video_processor._create_fallback_audio(video_duration, resource_manager)
        
        # Should return None when no music files are found
        assert result is None
    
    def test_fallback_audio_exception_handling(self, video_processor, mock_music_file, resource_manager):
        """Test that exceptions in fallback audio creation are properly handled"""
        video_duration = 30.0
        
        # Mock the music directory to contain our test file
        video_processor.config.paths.music_folder = mock_music_file.parent
        
        with patch('src.processing.video_processor.AudioFileClip') as mock_audio_clip_class:
            # Make AudioFileClip raise an exception
            mock_audio_clip_class.side_effect = Exception("Failed to load audio file")
            
            # Test the method - should not raise exception
            result = video_processor._create_fallback_audio(video_duration, resource_manager)
            
            # Should return None when an exception occurs
            assert result is None
    
    def test_resource_manager_registration(self, video_processor, mock_music_file, resource_manager):
        """Test that audio clips are properly registered with resource manager"""
        video_duration = 60.0
        music_duration = 20.0
        
        # Mock the music directory to contain our test file
        video_processor.config.paths.music_folder = mock_music_file.parent
        
        with patch('src.processing.video_processor.AudioFileClip') as mock_audio_clip_class:
            with patch('src.processing.video_processor.concatenate_audioclips') as mock_concatenate:
                with patch('src.processing.video_processor.MoviePyCompat') as mock_compat:
                    
                    # Setup mocks
                    mock_clip = MockAudioFileClip(str(mock_music_file), music_duration)
                    concatenated_clip = MockAudioFileClip(str(mock_music_file), 60.0)
                    trimmed_clip = MockAudioFileClip(str(mock_music_file), video_duration)
                    final_clip = MockAudioFileClip(str(mock_music_file), video_duration)
                    
                    mock_audio_clip_class.return_value = mock_clip
                    mock_concatenate.return_value = concatenated_clip
                    mock_compat.subclip.return_value = trimmed_clip
                    
                    # Mock volume scaling to return final clip
                    trimmed_clip.with_volume_scaled = Mock(return_value=final_clip)
                    
                    # Test the method
                    result = video_processor._create_fallback_audio(video_duration, resource_manager)
                    
                    # Verify clips were registered with resource manager
                    assert len(resource_manager.clips) >= 3  # Original, concatenated, final
                    assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])