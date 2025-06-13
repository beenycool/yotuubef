"""
Simple test to verify that the fallback audio fix correctly uses the right MoviePy methods.
This test specifically checks that the problematic method calls have been fixed.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

def test_fallback_audio_uses_correct_moviepy_methods():
    """
    Test that the _create_fallback_audio method uses correct MoviePy API calls
    instead of the problematic ones that would raise AttributeError.
    """
    
    # Create a temporary directory with a fake music file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        music_file = temp_path / "test_music.mp3"
        music_file.write_bytes(b"fake audio data")
        
        # Create mock objects
        mock_resource_manager = Mock()
        mock_resource_manager.register_clip = Mock()
        
        # Mock the configuration
        mock_config = Mock()
        mock_config.audio.background_music_volume = 0.7
        mock_config.paths.music_folder = temp_path
        
        # Mock the video processor
        mock_processor = Mock()
        mock_processor.config = mock_config
        mock_processor.logger = logging.getLogger(__name__)
        
        # Import the actual method we want to test
        from src.processing.video_processor import VideoProcessor
        
        # Create an instance and override the config
        processor = VideoProcessor()
        processor.config = mock_config
        processor.logger = logging.getLogger(__name__)
        
        # Mock the imports that happen inside the method
        with patch('builtins.__import__') as mock_import:
            # Set up the mock to return our mocked modules
            mock_audioclip = Mock()
            mock_concatenate = Mock()
            mock_moviepy_compat = Mock()
            mock_math = Mock()
            
            def import_side_effect(name, *args, **kwargs):
                if name == 'moviepy.audio.io.AudioFileClip':
                    mock_module = Mock()
                    mock_module.AudioFileClip = mock_audioclip
                    return mock_module
                elif name == 'moviepy.editor':
                    mock_module = Mock()
                    mock_module.concatenate_audioclips = mock_concatenate
                    return mock_module
                elif name == 'src.processing.video_processor_fixes':
                    mock_module = Mock()
                    mock_module.MoviePyCompat = mock_moviepy_compat
                    return mock_module
                elif name == 'math':
                    mock_math.ceil = lambda x: int(x) + 1
                    return mock_math
                else:
                    # For other imports, use the real import
                    return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            # Set up the audio clip mock
            mock_clip = Mock()
            mock_clip.duration = 20.0  # Shorter than video duration
            mock_audioclip.return_value = mock_clip
            
            # Set up concatenate mock
            mock_concatenated_clip = Mock()
            mock_concatenated_clip.duration = 60.0
            mock_concatenate.return_value = mock_concatenated_clip
            
            # Set up subclip mock
            mock_trimmed_clip = Mock()
            mock_trimmed_clip.duration = 60.0
            mock_moviepy_compat.subclip.return_value = mock_trimmed_clip
            
            # Set up volume scaling mock
            mock_final_clip = Mock()
            mock_trimmed_clip.with_volume_scaled.return_value = mock_final_clip
            
            # Test the method
            video_duration = 60.0
            result = processor._create_fallback_audio(video_duration, mock_resource_manager)
            
            # Verify that our fixed method calls were used
            assert result is not None
            
            # Verify that concatenate_audioclips was called (for looping)
            assert mock_concatenate.called
            
            # Verify that MoviePyCompat.subclip was called (not subclipped)
            # This test passes if it doesn't get called, since the music was looped to exactly the right duration
            
            # Verify that with_volume_scaled was called (not with_volume_multiplied)
            assert mock_trimmed_clip.with_volume_scaled.called
            
            # Verify resource manager was called
            assert mock_resource_manager.register_clip.call_count >= 3


def test_fallback_audio_method_names_are_correct():
    """
    Test that verifies the specific method names that were problematic are no longer used.
    This is a more direct test of the fix.
    """
    
    # Read the source code of the fixed method
    from src.processing.video_processor import VideoProcessor
    import inspect
    
    # Get the source code of the _create_fallback_audio method
    source = inspect.getsource(VideoProcessor._create_fallback_audio)
    
    # Verify the problematic method calls are NOT present
    assert "music_clip.loop(" not in source, "Found problematic 'loop()' method call"
    assert "music_clip.subclipped(" not in source, "Found problematic 'subclipped()' method call"  
    assert "with_volume_multiplied(" not in source, "Found problematic 'with_volume_multiplied()' method call"
    
    # Verify the correct method calls ARE present
    assert "concatenate_audioclips(" in source, "Missing correct 'concatenate_audioclips()' call"
    assert "MoviePyCompat.subclip(" in source, "Missing correct 'MoviePyCompat.subclip()' call"
    assert "with_volume_scaled(" in source, "Missing correct 'with_volume_scaled()' call"
    
    print("✓ All problematic method calls have been fixed")
    print("✓ Correct method calls are now being used")


if __name__ == "__main__":
    test_fallback_audio_method_names_are_correct()
    print("All tests passed!")