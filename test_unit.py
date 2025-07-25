#!/usr/bin/env python3
"""
Unit tests for the new architectural components
"""

import os
import sys
import asyncio
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set test environment variables
os.environ.setdefault('REDDIT_CLIENT_ID', 'test_id')  
os.environ.setdefault('REDDIT_CLIENT_SECRET', 'test_secret')
os.environ.setdefault('GEMINI_API_KEY', 'test_key')


class TestScheduler(unittest.TestCase):
    """Test the Scheduler class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from src.config.settings import get_config
        from src.scheduling import Scheduler
        
        self.config = get_config()
        self.scheduler = Scheduler(self.config)
        
    def test_scheduler_initialization(self):
        """Test scheduler initializes with correct defaults"""
        self.assertEqual(self.scheduler.daily_video_count, 0)
        self.assertEqual(self.scheduler.min_videos_per_day, 3)
        self.assertEqual(self.scheduler.max_videos_per_day, 8)
        self.assertIsInstance(self.scheduler.optimal_posting_times, list)
        
    def test_should_generate_video_under_minimum(self):
        """Test video generation when under minimum daily count"""
        # When under minimum, should always return True
        self.scheduler.daily_video_count = 1  # Under minimum of 3
        self.assertTrue(self.scheduler.should_generate_video())
        
    def test_should_generate_video_over_maximum(self):
        """Test video generation when over maximum daily count"""
        # When over maximum, should return False
        self.scheduler.daily_video_count = 10  # Over maximum of 8
        self.assertFalse(self.scheduler.should_generate_video())
        
    def test_increment_daily_count(self):
        """Test incrementing daily video count"""
        initial_count = self.scheduler.daily_video_count
        self.scheduler.increment_daily_count()
        self.assertEqual(self.scheduler.daily_video_count, initial_count + 1)
        
    def test_get_stats(self):
        """Test getting scheduler statistics"""
        stats = self.scheduler.get_stats()
        
        required_keys = [
            'daily_video_count', 'max_videos_per_day', 'min_videos_per_day',
            'last_reset_date', 'next_scheduled_time', 'optimal_posting_times'
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
            
    def test_get_next_scheduled_time(self):
        """Test getting next scheduled time"""
        next_time = self.scheduler.get_next_scheduled_time()
        self.assertIsInstance(next_time, datetime)
        
        # Should be in the future
        self.assertGreater(next_time, datetime.now())


class TestContentSource(unittest.TestCase):
    """Test the ContentSource class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from src.content import ContentSource
        self.content_source = ContentSource()
        
    def test_content_source_initialization(self):
        """Test content source initializes correctly"""
        self.assertIsInstance(self.content_source.subreddits, list)
        self.assertGreater(len(self.content_source.subreddits), 0)
        self.assertGreater(self.content_source.min_content_score, 0)
        
    def test_is_suitable_content(self):
        """Test content suitability filtering"""
        # Good content
        good_post = {
            'title': 'This is a good title for testing',
            'score': 500,
            'selftext': 'Good content text',
            'is_video': False
        }
        self.assertTrue(self.content_source._is_suitable_content(good_post))
        
        # Low score content
        low_score_post = {
            'title': 'Low score post',
            'score': 50,  # Below threshold
            'selftext': 'Content text'
        }
        self.assertFalse(self.content_source._is_suitable_content(low_score_post))
        
        # Too short title
        short_title_post = {
            'title': 'Short',  # Too short
            'score': 500,
            'selftext': 'Content text'
        }
        self.assertFalse(self.content_source._is_suitable_content(short_title_post))
        
    def test_generate_sample_posts(self):
        """Test sample post generation"""
        posts = self.content_source._generate_sample_posts('test_subreddit', 3)
        
        self.assertEqual(len(posts), 3)
        for post in posts:
            self.assertIn('title', post)
            self.assertIn('score', post)
            self.assertIn('subreddit', post)
            self.assertEqual(post['subreddit'], 'test_subreddit')
            
    def test_find_and_analyze_content(self):
        """Test finding and analyzing content"""
        async def run_test():
            content = await self.content_source.find_and_analyze_content(max_items=2)
            
            self.assertIsInstance(content, list)
            self.assertLessEqual(len(content), 2)
            
            if content:
                item = content[0]
                self.assertIn('content', item)
                self.assertIn('analysis', item)
                self.assertIn('source', item)
                
        asyncio.run(run_test())


class TestPipelineManager(unittest.TestCase):
    """Test the PipelineManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from src.pipeline import PipelineManager
        self.pipeline = PipelineManager()
        
    def test_pipeline_manager_initialization(self):
        """Test pipeline manager initializes correctly"""
        self.assertIsInstance(self.pipeline.pipeline_stages, list)
        self.assertGreater(len(self.pipeline.pipeline_stages), 0)
        self.assertIn('content_analysis', self.pipeline.pipeline_stages)
        self.assertIn('video_generation', self.pipeline.pipeline_stages)
        
    def test_pipeline_stats(self):
        """Test getting pipeline statistics"""
        stats = self.pipeline.get_pipeline_stats()
        
        required_keys = [
            'total_processed', 'successful_completions', 'failed_tasks',
            'success_rate_percent', 'stage_performance', 'pipeline_stages'
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
            
    def test_prepare_stage_input(self):
        """Test preparing stage input data"""
        context = {
            'content': {'title': 'Test'},
            'pipeline_id': 'test_123',
            'stage_results': {
                'content_analysis': {'analysis': {'score': 0.8}}
            }
        }
        
        # Test video generation stage input
        input_data = self.pipeline._prepare_stage_input('video_generation', context)
        
        self.assertIn('content', input_data)
        self.assertIn('pipeline_id', input_data)
        self.assertIn('stage', input_data)
        self.assertIn('analysis', input_data)
        self.assertEqual(input_data['stage'], 'video_generation')
        
    def test_process_content_through_pipeline(self):
        """Test processing content through pipeline"""
        async def run_test():
            test_content = {
                'content': {
                    'title': 'Test Content',
                    'selftext': 'Test content for pipeline'
                },
                'analysis': {
                    'overall_score': 0.8
                }
            }
            
            result = await self.pipeline.process_content_through_pipeline(test_content)
            
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            
            if result['success']:
                self.assertIn('stage_results', result)
                self.assertIn('processing_time', result)
                
        asyncio.run(run_test())


class TestApplication(unittest.TestCase):
    """Test the Application class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from src.application import Application
        self.app = Application()
        
    def test_application_initialization(self):
        """Test application initializes correctly"""
        self.assertIsNotNone(self.app.config)
        self.assertIsNotNone(self.app.scheduler)
        self.assertFalse(self.app.running)
        self.assertFalse(self.app.initialization_complete)
        
    def test_scheduler_integration(self):
        """Test scheduler integration in application"""
        stats = self.app.scheduler.get_stats()
        self.assertIsInstance(stats, dict)
        
        should_generate = self.app.scheduler.should_generate_video()
        self.assertIsInstance(should_generate, bool)
        
    def test_stop_method(self):
        """Test application stop method"""
        self.app.running = True
        self.app.stop()
        self.assertFalse(self.app.running)


def run_tests():
    """Run all unit tests"""
    print("🧪 Running Unit Tests for Architectural Components")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [TestScheduler, TestContentSource, TestPipelineManager, TestApplication]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All Unit Tests Passed!")
        print(f"🎉 Ran {result.testsRun} tests successfully")
    else:
        print(f"❌ {len(result.failures)} test(s) failed")
        print(f"❌ {len(result.errors)} test(s) had errors")
        
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())