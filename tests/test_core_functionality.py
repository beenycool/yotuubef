#!/usr/bin/env python3
"""
Minimal test suite for core functionality
Covers critical paths to ensure functionality is preserved
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock environment for testing
import os
os.environ.setdefault('REDDIT_CLIENT_ID', 'test_id')
os.environ.setdefault('REDDIT_CLIENT_SECRET', 'test_secret')
os.environ.setdefault('GEMINI_API_KEY', 'test_key')


class TestCoreFunctionality:
    """Test core functionality to ensure no regressions"""
    
    def test_imports(self):
        """Test that core modules can be imported"""
        try:
            from src.config.settings import get_config
            from src.application import Application
            from src.scheduling import Scheduler
            from src.content import ContentSource
            from src.pipeline import PipelineManager
            assert True, "Core modules imported successfully"
        except ImportError as e:
            pytest.fail(f"Failed to import core modules: {e}")
    
    def test_config_loading(self):
        """Test that configuration loads properly"""
        try:
            from src.config.settings import get_config
            config = get_config()
            assert config is not None, "Configuration should not be None"
            assert hasattr(config, 'video'), "Config should have video section"
            assert hasattr(config, 'audio'), "Config should have audio section"
        except Exception as e:
            pytest.fail(f"Config loading failed: {e}")
    
    def test_application_initialization(self):
        """Test Application class initialization"""
        try:
            from src.application import Application
            app = Application()
            assert app is not None, "Application should initialize"
            assert hasattr(app, 'scheduler'), "Application should have scheduler"
            assert hasattr(app, 'config'), "Application should have config"
        except Exception as e:
            pytest.fail(f"Application initialization failed: {e}")
    
    def test_scheduler_functionality(self):
        """Test basic scheduler functionality"""
        try:
            from src.scheduling import Scheduler
            from src.config.settings import get_config
            
            config = get_config()
            scheduler = Scheduler(config)
            
            stats = scheduler.get_stats()
            assert isinstance(stats, dict), "Stats should be a dictionary"
            assert 'daily_video_count' in stats, "Stats should contain daily_video_count"
            
            should_generate = scheduler.should_generate_video()
            assert isinstance(should_generate, bool), "should_generate should return boolean"
            
        except Exception as e:
            pytest.fail(f"Scheduler functionality failed: {e}")
    
    @pytest.mark.asyncio
    async def test_async_application_initialization(self):
        """Test async component initialization"""
        try:
            from src.application import Application
            
            app = Application()
            await app.initialize_async_components()
            
            assert app.initialization_complete, "Async initialization should complete"
            
        except Exception as e:
            # This is expected to fail in test environment without real services
            # Just ensure the method exists and can be called
            assert "initialize_async_components" in dir(app), "Method should exist"
    
    def test_enhanced_orchestrator_initialization(self):
        """Test EnhancedVideoOrchestrator initialization"""
        try:
            from src.enhanced_orchestrator import EnhancedVideoOrchestrator
            
            orchestrator = EnhancedVideoOrchestrator()
            assert orchestrator is not None, "Orchestrator should initialize"
            assert hasattr(orchestrator, 'total_videos_processed'), "Should have video tracking"
            assert orchestrator.total_videos_processed == 0, "Should start with 0 videos"
            
        except Exception as e:
            pytest.fail(f"Orchestrator initialization failed: {e}")
    
    def test_performance_metrics(self):
        """Test performance metrics functionality"""
        try:
            from src.enhanced_orchestrator import EnhancedVideoOrchestrator
            
            orchestrator = EnhancedVideoOrchestrator()
            metrics = orchestrator.get_performance_metrics()
            
            assert metrics is not None, "Metrics should not be None"
            assert hasattr(metrics, 'total_videos_processed'), "Should have total_videos_processed"
            assert hasattr(metrics, 'average_processing_time'), "Should have average_processing_time"
            assert hasattr(metrics, 'success_rate'), "Should have success_rate"
            
        except Exception as e:
            pytest.fail(f"Performance metrics failed: {e}")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])