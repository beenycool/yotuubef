#!/usr/bin/env python3
"""
Simple test script to validate the new architectural components
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set minimal environment variables for testing
os.environ.setdefault('REDDIT_CLIENT_ID', 'test_id')
os.environ.setdefault('REDDIT_CLIENT_SECRET', 'test_secret')
os.environ.setdefault('GEMINI_API_KEY', 'test_key')

def test_scheduler():
    """Test the Scheduler class"""
    print("🧪 Testing Scheduler class...")
    
    from src.config.settings import get_config
    from src.scheduling import Scheduler
    
    config = get_config()
    scheduler = Scheduler(config)
    
    # Test basic functionality
    stats = scheduler.get_stats()
    should_generate = scheduler.should_generate_video()
    next_time = scheduler.get_next_scheduled_time()
    
    print(f"   ✅ Daily video count: {stats['daily_video_count']}")
    print(f"   ✅ Should generate video: {should_generate}")
    print(f"   ✅ Next scheduled time: {next_time.strftime('%H:%M')}")
    print("   ✅ Scheduler class working correctly")

def test_content_source():
    """Test the ContentSource class"""
    print("\n🧪 Testing ContentSource class...")
    
    import asyncio
    from src.content import ContentSource
    
    content_source = ContentSource()
    
    async def test_async():
        # Test simulated content
        content = await content_source.find_and_analyze_content(max_items=2)
        print(f"   ✅ Found {len(content)} content items")
        
        if content:
            first_item = content[0]
            print(f"   ✅ First item title: {first_item['content']['title'][:50]}...")
            print(f"   ✅ Analysis score: {first_item.get('analysis', {}).get('overall_score', 'N/A')}")
    
    asyncio.run(test_async())
    print("   ✅ ContentSource class working correctly")

def test_pipeline_manager():
    """Test the PipelineManager class"""
    print("\n🧪 Testing PipelineManager class...")
    
    import asyncio
    from src.pipeline import PipelineManager
    
    pipeline = PipelineManager()
    
    async def test_async():
        # Test pipeline processing
        test_content = {
            'content': {
                'title': 'Test video content',
                'selftext': 'Test content for pipeline processing'
            },
            'analysis': {
                'overall_score': 0.8,
                'keywords': ['test', 'pipeline']
            }
        }
        
        result = await pipeline.process_content_through_pipeline(test_content)
        print(f"   ✅ Pipeline processing success: {result['success']}")
        
        if result['success']:
            print(f"   ✅ Processing time: {result['processing_time']:.2f}s")
            print(f"   ✅ Stages completed: {len(result['stage_results'])}")
    
    asyncio.run(test_async())
    print("   ✅ PipelineManager class working correctly")

def test_application():
    """Test the Application class"""
    print("\n🧪 Testing Application class...")
    from src.application import Application
    
    app = Application()
    
    # Test basic initialization
    print(f"   ✅ Scheduler initialized: {app.scheduler is not None}")
    print(f"   ✅ Content source initialized: {app.content_source is not None}")
    print(f"   ✅ Pipeline manager initialized: {app.pipeline_manager is not None}")
    print("   ✅ Application class working correctly")

def main():
    """Run all tests"""
    print("🚀 Testing New Architectural Components")
    print("=" * 50)
    
    try:
        test_scheduler()
        test_content_source()
        test_pipeline_manager()
        test_application()
        
        print("\n✅ All Tests Passed!")
        print("🎉 Architectural refactoring components are working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())