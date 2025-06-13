"""
Test MoviePy audio compatibility fix for with_audio/set_audio methods.
"""
import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.processing.video_processor_fixes import MoviePyCompat


class TestMoviePyAudioCompatibility(unittest.TestCase):
    """Test MoviePy audio compatibility layer for with_audio/set_audio methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_video_clip = Mock()
        self.mock_audio_clip = Mock()
        
    def test_with_audio_moviepy_2x(self):
        """Test with_audio compatibility for MoviePy 2.x (has with_audio method)."""
        # Mock MoviePy 2.x clip with with_audio method
        self.mock_video_clip.with_audio = Mock(return_value="result_clip")
        del self.mock_video_clip.set_audio  # Ensure set_audio doesn't exist
        
        result = MoviePyCompat.with_audio(self.mock_video_clip, self.mock_audio_clip)
        
        self.mock_video_clip.with_audio.assert_called_once_with(self.mock_audio_clip)
        self.assertEqual(result, "result_clip")
        
    def test_set_audio_moviepy_1x(self):
        """Test set_audio compatibility for MoviePy 1.x (has set_audio method)."""
        # Mock MoviePy 1.x clip with set_audio method
        self.mock_video_clip.set_audio = Mock(return_value="result_clip")
        del self.mock_video_clip.with_audio  # Ensure with_audio doesn't exist
        
        result = MoviePyCompat.with_audio(self.mock_video_clip, self.mock_audio_clip)
        
        self.mock_video_clip.set_audio.assert_called_once_with(self.mock_audio_clip)
        self.assertEqual(result, "result_clip")
        
    def test_both_methods_exist_prefers_with_audio(self):
        """Test that with_audio is preferred when both methods exist."""
        # Mock clip with both methods
        self.mock_video_clip.with_audio = Mock(return_value="with_audio_result")
        self.mock_video_clip.set_audio = Mock(return_value="set_audio_result")
        
        result = MoviePyCompat.with_audio(self.mock_video_clip, self.mock_audio_clip)
        
        # Should call with_audio, not set_audio
        self.mock_video_clip.with_audio.assert_called_once_with(self.mock_audio_clip)
        self.mock_video_clip.set_audio.assert_not_called()
        self.assertEqual(result, "with_audio_result")
        
    def test_neither_method_exists_raises_error(self):
        """Test that AttributeError is raised when neither method exists."""
        # Mock clip with neither method
        del self.mock_video_clip.with_audio
        del self.mock_video_clip.set_audio
        
        with self.assertRaises(AttributeError) as cm:
            MoviePyCompat.with_audio(self.mock_video_clip, self.mock_audio_clip)
            
        self.assertIn("Neither 'with_audio' nor 'set_audio' method found", str(cm.exception))
        
    @patch('logging.getLogger')
    def test_exception_handling_returns_original_clip(self, mock_logger):
        """Test that exceptions are handled gracefully by returning original clip."""
        # Mock clip that raises exception when calling with_audio
        self.mock_video_clip.with_audio = Mock(side_effect=Exception("Test exception"))
        del self.mock_video_clip.set_audio
        
        result = MoviePyCompat.with_audio(self.mock_video_clip, self.mock_audio_clip)
        
        # Should return original clip and log warning
        self.assertEqual(result, self.mock_video_clip)
        mock_logger.return_value.warning.assert_called_once()
        
    def test_with_audio_method_exception_fallback_to_set_audio(self):
        """Test fallback to set_audio when with_audio method raises exception."""
        # Mock clip where with_audio raises exception but set_audio works
        self.mock_video_clip.with_audio = Mock(side_effect=Exception("with_audio failed"))
        self.mock_video_clip.set_audio = Mock(return_value="set_audio_result")
        
        result = MoviePyCompat.with_audio(self.mock_video_clip, self.mock_audio_clip)
        
        # Should try with_audio first, then fall back to set_audio
        self.mock_video_clip.with_audio.assert_called_once_with(self.mock_audio_clip)
        self.mock_video_clip.set_audio.assert_called_once_with(self.mock_audio_clip)
        self.assertEqual(result, "set_audio_result")


class TestMoviePyCompatibilityIntegration(unittest.TestCase):
    """Integration tests for MoviePy compatibility in video processing."""
    
    def test_mock_moviepy_1x_scenario(self):
        """Test scenario simulating MoviePy 1.x environment."""
        # Create mock clips that simulate MoviePy 1.x behavior
        video_clip = Mock()
        audio_clip = Mock()
        
        # MoviePy 1.x has set_audio but not with_audio
        video_clip.set_audio = Mock(return_value="combined_clip")
        if hasattr(video_clip, 'with_audio'):
            del video_clip.with_audio
            
        result = MoviePyCompat.with_audio(video_clip, audio_clip)
        
        video_clip.set_audio.assert_called_once_with(audio_clip)
        self.assertEqual(result, "combined_clip")
        
    def test_mock_moviepy_2x_scenario(self):
        """Test scenario simulating MoviePy 2.x environment."""
        # Create mock clips that simulate MoviePy 2.x behavior
        video_clip = Mock()
        audio_clip = Mock()
        
        # MoviePy 2.x has with_audio
        video_clip.with_audio = Mock(return_value="combined_clip")
        
        result = MoviePyCompat.with_audio(video_clip, audio_clip)
        
        video_clip.with_audio.assert_called_once_with(audio_clip)
        self.assertEqual(result, "combined_clip")


if __name__ == '__main__':
    unittest.main()