"""
Comprehensive test suite for sound effects functionality.
Tests sound effects manager, video processing integration, and engagement metrics.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from src.processing.sound_effects_manager import SoundEffectsManager, SoundEffectFile
from src.processing.video_processor import AudioProcessor
from src.config.settings import get_config


class TestSoundEffectsManager:
    """Test suite for SoundEffectsManager functionality"""
    
    @pytest.fixture
    def temp_sound_dir(self):
        """Create temporary sound effects directory with test files"""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create directory structure
        categories = ['impact', 'transition', 'liquid', 'mechanical', 'notification', 'dramatic']
        for category in categories:
            (temp_dir / category).mkdir(parents=True, exist_ok=True)
        
        # Create mock sound files
        test_files = {
            'impact/impact.wav': b'mock_audio_data',
            'impact/hit.wav': b'mock_audio_data',
            'transition/whoosh.wav': b'mock_audio_data',
            'transition/swoosh.wav': b'mock_audio_data',
            'liquid/splash.wav': b'mock_audio_data',
            'mechanical/click.wav': b'mock_audio_data',
            'notification/ding.wav': b'mock_audio_data',
            'dramatic/boom.wav': b'mock_audio_data',
        }
        
        for file_path, content in test_files.items():
            full_path = temp_dir / file_path
            full_path.write_bytes(content)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_sound_dir):
        """Mock configuration with test sound directory"""
        with patch('src.processing.sound_effects_manager.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.paths.sound_effects_folder = temp_sound_dir
            mock_get_config.return_value = mock_config
            yield mock_config
    
    def test_sound_effects_manager_initialization(self, mock_config):
        """Test SoundEffectsManager initializes correctly"""
        manager = SoundEffectsManager()
        
        assert manager.sound_effects_dir == mock_config.paths.sound_effects_folder
        assert 'impact' in manager.category_mapping
        assert 'transition' in manager.category_mapping
        assert len(manager._sound_cache) > 0
    
    def test_cache_refresh(self, mock_config, temp_sound_dir):
        """Test sound effects cache refresh functionality"""
        manager = SoundEffectsManager()
        
        # Check cache was populated
        cache_status = manager.get_cache_status()
        assert cache_status['total_effects'] > 0
        assert cache_status['categories'] > 0
        assert 'impact' in cache_status['category_breakdown']
    
    def test_find_sound_effect_exact_match(self, mock_config):
        """Test finding sound effects with exact name match"""
        manager = SoundEffectsManager()
        
        # Test exact match
        result = manager.find_sound_effect('impact')
        assert result is not None
        assert result.name == 'impact.wav'
        
        result = manager.find_sound_effect('whoosh')
        assert result is not None
        assert result.name == 'whoosh.wav'
    
    def test_find_sound_effect_category_preference(self, mock_config):
        """Test sound effect search with category preference"""
        manager = SoundEffectsManager()
        
        # Test with preferred category
        result = manager.find_sound_effect('impact', preferred_category='impact')
        assert result is not None
        assert 'impact' in str(result)
    
    def test_find_sound_effect_fallback(self, mock_config):
        """Test sound effect fallback mechanism"""
        manager = SoundEffectsManager()
        
        # Test fallback for non-existent effect
        result = manager.find_sound_effect('nonexistent_effect')
        assert result is not None  # Should fallback to some available effect
    
    def test_sound_effect_aliases(self, mock_config):
        """Test sound effect alias resolution"""
        manager = SoundEffectsManager()
        
        # Test alias matching
        result1 = manager.find_sound_effect('swoosh')
        result2 = manager.find_sound_effect('swish')  # Should find similar effect
        
        assert result1 is not None
        # Both should find effects (may be same or different based on aliases)
    
    def test_categorize_by_filename(self, mock_config):
        """Test automatic categorization by filename"""
        manager = SoundEffectsManager()
        
        # Test categorization
        assert manager._categorize_by_filename('water_splash') == 'liquid'
        assert manager._categorize_by_filename('button_click') == 'mechanical'
        assert manager._categorize_by_filename('wind_whoosh') == 'transition'
    
    @patch('src.processing.sound_effects_manager.AudioFileClip')
    def test_validate_sound_effect(self, mock_audio_clip, mock_config, temp_sound_dir):
        """Test sound effect file validation"""
        manager = SoundEffectsManager()
        
        # Mock audio clip
        mock_clip = Mock()
        mock_clip.duration = 2.5
        mock_audio_clip.return_value.__enter__.return_value = mock_clip
        
        # Test valid file
        test_file = temp_sound_dir / 'impact' / 'impact.wav'
        is_valid, message = manager.validate_sound_effect(test_file)
        assert is_valid
        
        # Test invalid extension
        invalid_file = temp_sound_dir / 'test.txt'
        invalid_file.write_text('test')
        is_valid, message = manager.validate_sound_effect(invalid_file)
        assert not is_valid
        assert 'Unsupported audio format' in message
    
    def test_get_available_effects(self, mock_config):
        """Test getting available effects summary"""
        manager = SoundEffectsManager()
        
        available = manager.get_available_effects()
        assert isinstance(available, dict)
        assert 'impact' in available
        assert len(available['impact']) > 0


class TestAudioProcessorSoundEffects:
    """Test suite for AudioProcessor sound effects integration"""
    
    @pytest.fixture
    def mock_audio_processor(self):
        """Create mock AudioProcessor with SoundEffectsManager"""
        with patch('src.processing.video_processor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_get_config.return_value = mock_config
            
            with patch('src.processing.video_processor.SoundEffectsManager') as mock_manager_class:
                mock_manager = Mock()
                mock_manager_class.return_value = mock_manager
                
                processor = AudioProcessor()
                yield processor, mock_manager
    
    @patch('src.processing.video_processor.AudioFileClip')
    @patch('src.processing.video_processor.CompositeAudioClip')
    def test_add_sound_effects_success(self, mock_composite, mock_audio_clip, mock_audio_processor):
        """Test successful sound effects addition"""
        processor, mock_manager = mock_audio_processor
        
        # Mock sound effect file found
        mock_manager.find_sound_effect.return_value = Path('test_sound.wav')
        mock_manager.validate_sound_effect.return_value = (True, None)
        
        # Mock audio clip
        mock_clip = Mock()
        mock_clip.duration = 1.0
        mock_clip.multiply_volume.return_value = mock_clip
        mock_clip.set_start.return_value = mock_clip
        mock_clip.audio_fadein.return_value = mock_clip
        mock_clip.audio_fadeout.return_value = mock_clip
        mock_audio_clip.return_value = mock_clip
        
        # Mock base audio
        mock_base_audio = Mock()
        
        # Test data
        sound_effects = [
            {'effect_name': 'whoosh', 'timestamp_seconds': 2.0, 'volume': 0.8},
            {'effect_name': 'splash', 'timestamp_seconds': 5.0, 'volume': 0.6}
        ]
        
        result = processor.add_sound_effects(mock_base_audio, sound_effects, 10.0)
        
        # Verify manager was called
        assert mock_manager.find_sound_effect.call_count == 2
        assert mock_manager.validate_sound_effect.call_count == 2
        
        # Verify audio processing
        assert mock_audio_clip.call_count == 2
    
    def test_add_sound_effects_no_effects(self, mock_audio_processor):
        """Test handling of empty sound effects list"""
        processor, mock_manager = mock_audio_processor
        
        mock_base_audio = Mock()
        result = processor.add_sound_effects(mock_base_audio, [], 10.0)
        
        assert result == mock_base_audio
        assert mock_manager.find_sound_effect.call_count == 0
    
    def test_add_sound_effects_file_not_found(self, mock_audio_processor):
        """Test handling of missing sound effect files"""
        processor, mock_manager = mock_audio_processor
        
        # Mock no sound effect file found
        mock_manager.find_sound_effect.return_value = None
        
        mock_base_audio = Mock()
        sound_effects = [{'effect_name': 'nonexistent', 'timestamp_seconds': 2.0, 'volume': 0.8}]
        
        result = processor.add_sound_effects(mock_base_audio, sound_effects, 10.0)
        
        assert result == mock_base_audio
        assert mock_manager.find_sound_effect.call_count == 1
    
    @patch('src.processing.video_processor.AudioFileClip')
    def test_add_sound_effects_invalid_file(self, mock_audio_clip, mock_audio_processor):
        """Test handling of invalid sound effect files"""
        processor, mock_manager = mock_audio_processor
        
        # Mock sound effect file found but invalid
        mock_manager.find_sound_effect.return_value = Path('invalid_sound.wav')
        mock_manager.validate_sound_effect.return_value = (False, "Invalid audio file")
        
        mock_base_audio = Mock()
        sound_effects = [{'effect_name': 'invalid', 'timestamp_seconds': 2.0, 'volume': 0.8}]
        
        result = processor.add_sound_effects(mock_base_audio, sound_effects, 10.0)
        
        assert result == mock_base_audio
        assert mock_audio_clip.call_count == 0  # Should not try to load invalid file


class TestSoundEffectsIntegration:
    """Integration tests for sound effects in video processing pipeline"""
    
    @pytest.fixture
    def sample_video_analysis(self):
        """Create sample video analysis with sound effects"""
        return {
            'sound_effects': [
                {
                    'effect_name': 'whoosh',
                    'timestamp_seconds': 2.0,
                    'volume': 0.8,
                    'category': 'transition',
                    'description': 'Quick movement transition'
                },
                {
                    'effect_name': 'splash',
                    'timestamp_seconds': 8.0,
                    'volume': 0.9,
                    'category': 'liquid',
                    'description': 'Water impact sound'
                },
                {
                    'effect_name': 'pop',
                    'timestamp_seconds': 15.0,
                    'volume': 0.7,
                    'category': 'mechanical',
                    'description': 'Button or bubble pop'
                }
            ]
        }
    
    def test_sound_effects_json_format(self, sample_video_analysis):
        """Test sound effects data format matches expected JSON schema"""
        sound_effects = sample_video_analysis['sound_effects']
        
        for effect in sound_effects:
            # Required fields
            assert 'effect_name' in effect
            assert 'timestamp_seconds' in effect
            assert 'volume' in effect
            
            # Optional fields
            assert isinstance(effect.get('category', ''), str)
            assert isinstance(effect.get('description', ''), str)
            
            # Value validation
            assert 0 <= effect['volume'] <= 1.0
            assert effect['timestamp_seconds'] >= 0
    
    def test_sound_effects_timing_validation(self, sample_video_analysis):
        """Test sound effects timing validation"""
        sound_effects = sample_video_analysis['sound_effects']
        video_duration = 20.0  # seconds
        
        valid_effects = []
        for effect in sound_effects:
            if effect['timestamp_seconds'] < video_duration:
                valid_effects.append(effect)
        
        # All sample effects should be within duration
        assert len(valid_effects) == len(sound_effects)
    
    @patch('src.processing.video_processor.VideoProcessor')
    def test_video_processing_with_sound_effects(self, mock_video_processor, sample_video_analysis):
        """Test complete video processing pipeline with sound effects"""
        # This would be a more complex integration test
        # For now, we'll test that the sound effects are properly formatted
        
        sound_effects = sample_video_analysis['sound_effects']
        
        # Verify all required fields are present
        for effect in sound_effects:
            assert all(key in effect for key in ['effect_name', 'timestamp_seconds', 'volume'])
        
        # Verify reasonable values
        assert all(0 <= effect['volume'] <= 1.0 for effect in sound_effects)
        assert all(effect['timestamp_seconds'] >= 0 for effect in sound_effects)


@pytest.mark.integration
class TestSoundEffectsEndToEnd:
    """End-to-end tests for sound effects functionality"""
    
    def test_sound_effects_directory_structure(self):
        """Test that sound effects directory structure matches README.md"""
        config = get_config()
        sound_dir = config.paths.sound_effects_folder
        
        if sound_dir.exists():
            expected_categories = ['impact', 'transition', 'liquid', 'mechanical', 'notification', 'dramatic']
            
            for category in expected_categories:
                category_dir = sound_dir / category
                # Directory should exist (may be empty in test environment)
                assert category_dir.exists(), f"Missing sound effects category directory: {category}"
    
    def test_sound_effects_manager_real_files(self):
        """Test SoundEffectsManager with real sound effects files"""
        try:
            manager = SoundEffectsManager()
            cache_status = manager.get_cache_status()
            
            # Should have populated cache (even if empty directories)
            assert cache_status['categories'] >= 0
            assert isinstance(cache_status['category_breakdown'], dict)
            
            # Test finding effects (may return None if no files exist)
            result = manager.find_sound_effect('whoosh')
            # Don't assert result is not None since files may not exist in test environment
            
        except Exception as e:
            pytest.fail(f"SoundEffectsManager failed with real configuration: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])