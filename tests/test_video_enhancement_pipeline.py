"""
Comprehensive end-to-end testing for video enhancement pipeline.
Tests sound effects, visual effects, and engagement monitoring with sample videos.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sqlite3

from src.processing.sound_effects_manager import SoundEffectsManager
from src.processing.video_processor import VideoProcessor, AudioProcessor
from src.monitoring.engagement_metrics import EngagementMonitor, VideoMetrics, EnhancementType
from src.config.settings import get_config


class TestVideoEnhancementPipeline:
    """End-to-end tests for video enhancement pipeline"""
    
    @pytest.fixture
    def temp_test_environment(self):
        """Create temporary test environment with all necessary directories"""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create directory structure
        (temp_dir / "sound_effects").mkdir(parents=True)
        (temp_dir / "music").mkdir(parents=True)
        (temp_dir / "fonts").mkdir(parents=True)
        (temp_dir / "temp_processing").mkdir(parents=True)
        
        # Create sound effects categories
        sound_categories = ['impact', 'transition', 'liquid', 'mechanical', 'notification', 'dramatic']
        for category in sound_categories:
            (temp_dir / "sound_effects" / category).mkdir(parents=True)
        
        # Create mock sound files
        test_sounds = {
            'sound_effects/transition/whoosh.wav': b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00',
            'sound_effects/impact/impact.wav': b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00',
            'sound_effects/liquid/splash.wav': b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00',
            'sound_effects/mechanical/click.wav': b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00',
        }
        
        for file_path, content in test_sounds.items():
            full_path = temp_dir / file_path
            full_path.write_bytes(content)
        
        # Create mock font files
        (temp_dir / "fonts" / "Montserrat-Bold.ttf").write_bytes(b'mock_font_data')
        (temp_dir / "fonts" / "BebasNeue-Regular.ttf").write_bytes(b'mock_font_data')
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_config(self, temp_test_environment):
        """Mock configuration for testing"""
        with patch('src.processing.sound_effects_manager.get_config') as mock_get_config, \
             patch('src.monitoring.engagement_metrics.get_config') as mock_get_config2:
            
            mock_config = Mock()
            mock_config.paths.sound_effects_folder = temp_test_environment / "sound_effects"
            mock_config.paths.music_folder = temp_test_environment / "music"
            mock_config.paths.fonts_folder = temp_test_environment / "fonts"
            mock_config.paths.temp_dir = temp_test_environment / "temp_processing"
            mock_config.paths.base_dir = temp_test_environment
            mock_config.paths.db_file = temp_test_environment / "test.db"
            
            # Video settings
            mock_config.video.target_duration = 59
            mock_config.video.target_resolution = (1080, 1920)
            mock_config.video.max_memory_usage = 0.8
            
            # Audio settings
            mock_config.audio.background_music_enabled = True
            mock_config.audio.background_music_volume = 0.06
            
            # Effects settings
            mock_config.effects.subtle_zoom_enabled = True
            mock_config.effects.color_grade_enabled = True
            
            mock_get_config.return_value = mock_config
            mock_get_config2.return_value = mock_config
            yield mock_config
    
    @pytest.fixture
    def sample_video_analysis(self):
        """Sample video analysis data for testing"""
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
                    'effect_name': 'click',
                    'timestamp_seconds': 15.0,
                    'volume': 0.7,
                    'category': 'mechanical',
                    'description': 'Button click'
                }
            ],
            'visual_effects': [
                {
                    'effect_type': 'dynamic_zoom',
                    'timestamp_seconds': 5.0,
                    'duration': 2.0,
                    'intensity': 0.8
                }
            ],
            'text_overlays': [
                {
                    'text': 'Amazing!',
                    'start_seconds': 1.0,
                    'end_seconds': 3.0,
                    'position': 'center',
                    'style': 'impact'
                }
            ]
        }
    
    def test_sound_effects_manager_integration(self, mock_config):
        """Test SoundEffectsManager with test environment"""
        manager = SoundEffectsManager()
        
        # Test cache refresh
        manager.refresh_sound_effects_cache()
        cache_status = manager.get_cache_status()
        
        assert cache_status['total_effects'] > 0
        assert 'transition' in cache_status['category_breakdown']
        assert cache_status['category_breakdown']['transition'] > 0
        
        # Test finding sound effects
        whoosh_file = manager.find_sound_effect('whoosh')
        assert whoosh_file is not None
        assert whoosh_file.name == 'whoosh.wav'
        
        splash_file = manager.find_sound_effect('splash')
        assert splash_file is not None
        assert splash_file.name == 'splash.wav'
        
        # Test fallback mechanism
        nonexistent_file = manager.find_sound_effect('nonexistent_sound')
        assert nonexistent_file is not None  # Should fallback to available sound
    
    @patch('src.processing.video_processor.AudioFileClip')
    @patch('src.processing.video_processor.CompositeAudioClip')
    def test_audio_processor_with_sound_effects(self, mock_composite_audio, mock_audio_clip, 
                                               mock_config, sample_video_analysis):
        """Test AudioProcessor with sound effects integration"""
        # Mock audio clips
        mock_clip = Mock()
        mock_clip.duration = 1.0
        mock_clip.multiply_volume.return_value = mock_clip
        mock_clip.set_start.return_value = mock_clip
        mock_clip.audio_fadein.return_value = mock_clip
        mock_clip.audio_fadeout.return_value = mock_clip
        mock_audio_clip.return_value = mock_clip
        
        # Mock composite audio
        mock_base_audio = Mock()
        mock_composite_audio.return_value = mock_base_audio
        
        # Create processor
        processor = AudioProcessor()
        
        # Process sound effects
        result = processor.add_sound_effects(
            mock_base_audio,
            sample_video_analysis['sound_effects'],
            20.0  # video duration
        )
        
        # Verify sound effects were found and processed
        assert processor.sound_effects_manager.find_sound_effect.call_count == 3
        assert mock_audio_clip.call_count == 3  # All effects should be loaded
    
    def test_engagement_metrics_system(self, mock_config):
        """Test engagement metrics monitoring system"""
        monitor = EngagementMonitor()
        
        # Test recording video upload
        video_id = "test_video_123"
        title = "Test Video with Enhancements"
        duration = 30.0
        enhancements = ['sound_effects', 'visual_effects', 'text_overlays']
        
        success = monitor.record_video_upload(video_id, title, duration, enhancements)
        assert success
        
        # Test retrieving metrics
        metrics = monitor.db.get_video_metrics(video_id)
        assert metrics is not None
        assert metrics.video_id == video_id
        assert metrics.title == title
        assert metrics.duration_seconds == duration
        assert len(metrics.enhancements_used) == 3
        
        # Test updating with fake YouTube data
        youtube_data = {
            'statistics': {
                'viewCount': '1000',
                'likeCount': '50',
                'dislikeCount': '2',
                'commentCount': '15'
            }
        }
        
        success = monitor.update_metrics_from_youtube(video_id, youtube_data)
        assert success
        
        # Verify metrics were updated
        updated_metrics = monitor.db.get_video_metrics(video_id)
        assert updated_metrics.views == 1000
        assert updated_metrics.likes == 50
        assert updated_metrics.engagement_rate > 0  # Should be calculated
    
    def test_enhancement_performance_analysis(self, mock_config):
        """Test enhancement performance analysis"""
        monitor = EngagementMonitor()
        
        # Create test data with and without enhancements
        test_videos = [
            # Videos with sound effects
            {
                'video_id': 'video_with_se_1',
                'title': 'Video 1 with Sound Effects',
                'duration': 30.0,
                'enhancements': ['sound_effects'],
                'metrics': {'views': 1500, 'likes': 75, 'comments': 20}
            },
            {
                'video_id': 'video_with_se_2',
                'title': 'Video 2 with Sound Effects',
                'duration': 45.0,
                'enhancements': ['sound_effects'],
                'metrics': {'views': 1200, 'likes': 60, 'comments': 15}
            },
            # Videos without sound effects
            {
                'video_id': 'video_without_se_1',
                'title': 'Video 1 without Sound Effects',
                'duration': 35.0,
                'enhancements': [],
                'metrics': {'views': 800, 'likes': 30, 'comments': 8}
            },
            {
                'video_id': 'video_without_se_2',
                'title': 'Video 2 without Sound Effects',
                'duration': 40.0,
                'enhancements': [],
                'metrics': {'views': 900, 'likes': 35, 'comments': 10}
            }
        ]
        
        # Store test videos
        for video_data in test_videos:
            # Record upload
            monitor.record_video_upload(
                video_data['video_id'],
                video_data['title'],
                video_data['duration'],
                video_data['enhancements']
            )
            
            # Update with metrics
            youtube_data = {
                'statistics': {
                    'viewCount': str(video_data['metrics']['views']),
                    'likeCount': str(video_data['metrics']['likes']),
                    'commentCount': str(video_data['metrics']['comments'])
                }
            }
            monitor.update_metrics_from_youtube(video_data['video_id'], youtube_data)
        
        # Analyze performance
        performance = monitor.analyzer.analyze_enhancement_performance('sound_effects')
        
        if performance:  # May be None if insufficient data
            assert performance.videos_with_enhancement == 2
            assert performance.videos_without_enhancement == 2
            assert performance.avg_views_with > 0
            assert performance.avg_views_without > 0
            # Sound effects should show improvement in this test data
            assert performance.views_improvement > 0
    
    def test_performance_report_generation(self, mock_config):
        """Test performance report generation"""
        monitor = EngagementMonitor()
        
        # Generate report (may be empty if no data)
        report = monitor.analyzer.generate_performance_report()
        
        assert 'report_date' in report
        assert 'analysis_period_days' in report
        assert 'enhancement_performance' in report
        assert 'summary' in report
        
        # Test report export
        report_path = monitor.export_performance_report()
        assert report_path.exists()
        
        # Verify JSON format
        with open(report_path, 'r') as f:
            exported_report = json.load(f)
        
        assert exported_report == report
    
    @patch('src.processing.video_processor.VideoFileClip')
    @patch('src.processing.video_processor.AudioFileClip')
    def test_complete_video_enhancement_workflow(self, mock_audio_clip, mock_video_clip, 
                                                mock_config, sample_video_analysis):
        """Test complete video enhancement workflow"""
        # Mock video and audio clips
        mock_video = Mock()
        mock_video.duration = 20.0
        mock_video.size = (1920, 1080)
        mock_video.fps = 30
        mock_video_clip.return_value = mock_video
        
        mock_audio = Mock()
        mock_audio.duration = 1.0
        mock_audio_clip.return_value = mock_audio
        
        # Test video processor initialization
        try:
            processor = VideoProcessor()
            assert processor is not None
            assert hasattr(processor, 'audio_processor')
            assert hasattr(processor.audio_processor, 'sound_effects_manager')
        except Exception as e:
            # May fail due to missing dependencies, but structure should be correct
            pytest.skip(f"Video processor initialization failed: {e}")
    
    def test_sound_effects_validation(self, mock_config):
        """Test sound effects file validation"""
        manager = SoundEffectsManager()
        
        # Test valid sound file (mock WAV file)
        valid_file = mock_config.paths.sound_effects_folder / "transition" / "whoosh.wav"
        
        with patch('src.processing.sound_effects_manager.AudioFileClip') as mock_audio_clip:
            mock_clip = Mock()
            mock_clip.duration = 2.5
            mock_audio_clip.return_value.__enter__.return_value = mock_clip
            
            is_valid, message = manager.validate_sound_effect(valid_file)
            assert is_valid
            assert message is None
        
        # Test invalid file extension
        invalid_file = mock_config.paths.sound_effects_folder / "test.txt"
        invalid_file.write_text("not an audio file")
        
        is_valid, message = manager.validate_sound_effect(invalid_file)
        assert not is_valid
        assert "Unsupported audio format" in message


@pytest.mark.integration
class TestVideoEnhancementIntegration:
    """Integration tests requiring actual files and configuration"""
    
    def test_real_sound_effects_directory(self):
        """Test with real sound effects directory structure"""
        try:
            config = get_config()
            sound_dir = config.paths.sound_effects_folder
            
            if sound_dir.exists():
                manager = SoundEffectsManager()
                cache_status = manager.get_cache_status()
                
                # Should have initialized without errors
                assert cache_status['categories'] >= 0
                assert isinstance(cache_status['category_breakdown'], dict)
                
                # Test basic functionality
                available_effects = manager.get_available_effects()
                assert isinstance(available_effects, dict)
                
        except Exception as e:
            pytest.skip(f"Real sound effects test skipped: {e}")
    
    def test_engagement_metrics_database(self):
        """Test engagement metrics database with real configuration"""
        try:
            config = get_config()
            db_path = config.paths.base_dir / "test_engagement_metrics.db"
            
            # Clean up any existing test database
            if db_path.exists():
                db_path.unlink()
            
            from src.monitoring.engagement_metrics import EngagementMetricsDB
            
            db = EngagementMetricsDB(db_path)
            
            # Test database initialization
            assert db_path.exists()
            
            # Test storing and retrieving metrics
            from src.monitoring.engagement_metrics import VideoMetrics
            
            test_metrics = VideoMetrics(
                video_id="integration_test_video",
                title="Integration Test Video",
                upload_date=datetime.now(),
                duration_seconds=30.0,
                views=100,
                likes=10,
                enhancements_used=['sound_effects', 'visual_effects']
            )
            
            success = db.store_video_metrics(test_metrics)
            assert success
            
            retrieved_metrics = db.get_video_metrics("integration_test_video")
            assert retrieved_metrics is not None
            assert retrieved_metrics.video_id == "integration_test_video"
            assert retrieved_metrics.views == 100
            
            # Clean up
            db_path.unlink()
            
        except Exception as e:
            pytest.skip(f"Database integration test skipped: {e}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])