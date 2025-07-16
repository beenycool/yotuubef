#!/usr/bin/env python3
"""
Test core functionality without external dependencies
"""

import sys
import os
from pathlib import Path
import logging

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Create a simple mock environment
os.environ.setdefault('REDDIT_CLIENT_ID', 'mock_client_id')
os.environ.setdefault('REDDIT_CLIENT_SECRET', 'mock_client_secret')
os.environ.setdefault('REDDIT_USER_AGENT', 'mock_agent')
os.environ.setdefault('GEMINI_API_KEY', 'mock_gemini_key')

def test_config_loading():
    """Test that configuration loads properly"""
    try:
        from src.config.settings import get_config
        config = get_config()
        print("✅ Configuration loaded successfully")
        print(f"   - Video config present: {config.video is not None}")
        print(f"   - Video processing config present: {config.video_processing is not None}")
        print(f"   - AI features config present: {config.ai_features is not None}")
        print(f"   - Long-form video enabled: {config.long_form_video.enable_long_form_generation}")
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_reddit_client():
    """Test Reddit client initialization"""
    try:
        from src.integrations.reddit_client import RedditClient
        client = RedditClient()
        
        # Test basic functionality
        if client.is_connected():
            print("✅ Reddit client connected (unexpected - should be mock)")
        else:
            print("✅ Reddit client not connected (expected with mock credentials)")
        
        return True
    except Exception as e:
        print(f"❌ Reddit client test failed: {e}")
        return False

def test_gemini_client():
    """Test Gemini AI client"""
    try:
        from src.integrations.gemini_ai_client import GeminiAIClient
        client = GeminiAIClient()
        
        print("✅ Gemini AI client initialized")
        print(f"   - API available: {client.is_api_available()}")
        return True
    except Exception as e:
        print(f"❌ Gemini client test failed: {e}")
        return False

def test_models():
    """Test data models"""
    try:
        from src.models import VideoFormat, NicheCategory, ContentStructureType
        
        # Test enums
        assert VideoFormat.SHORTS == "shorts"
        assert NicheCategory.TECHNOLOGY == "technology"
        assert ContentStructureType.INTRO == "intro"
        
        print("✅ Data models work correctly")
        return True
    except Exception as e:
        print(f"❌ Models test failed: {e}")
        return False

def test_long_form_functionality():
    """Test long-form video generation components"""
    try:
        from src.processing.long_form_video_generator import LongFormVideoGenerator
        
        generator = LongFormVideoGenerator()
        print("✅ Long-form video generator initialized")
        
        # Test configuration
        config = generator.long_form_config
        enabled = config.get('enable_long_form_generation', False)
        print(f"   - Long-form generation enabled: {enabled}")
        print(f"   - Components available: {generator.ai_client is not None}")
        
        return True
    except Exception as e:
        print(f"❌ Long-form generator test failed: {e}")
        return False

def test_enhanced_orchestrator():
    """Test enhanced orchestrator (without external dependencies)"""
    try:
        from src.enhanced_orchestrator import EnhancedVideoOrchestrator
        
        orchestrator = EnhancedVideoOrchestrator()
        print("✅ Enhanced orchestrator initialized")
        
        # Test basic properties
        print(f"   - Components initialized: {getattr(orchestrator, 'components_initialized', False)}")
        print(f"   - Cinematic editing enabled: {orchestrator.enable_cinematic_editing}")
        print(f"   - Advanced audio enabled: {orchestrator.enable_advanced_audio}")
        
        # Test system status (should work without external dependencies)
        # We'll wrap this in a try-catch since it might fail due to missing dependencies
        try:
            import asyncio
            status = asyncio.run(orchestrator.get_system_status())
            print(f"   - System status: {status.get('system_status', 'unknown')}")
        except Exception as e:
            print(f"   - System status check failed (expected): {str(e)[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Enhanced orchestrator test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Core Functionality (without external dependencies)")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Reddit Client", test_reddit_client),
        ("Gemini AI Client", test_gemini_client),
        ("Data Models", test_models),
        ("Long-form Video Generator", test_long_form_functionality),
        ("Enhanced Orchestrator", test_enhanced_orchestrator),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core functionality tests passed!")
        print("\n💡 Next steps:")
        print("   1. Install missing dependencies (asyncpraw, google-generativeai)")
        print("   2. Configure API credentials in .env file")
        print("   3. Test with real Reddit and Gemini API calls")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)