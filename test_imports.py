#!/usr/bin/env python3
"""
Test imports individually to identify what's causing numpy dependency issues
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Create a simple mock environment
os.environ.setdefault('REDDIT_CLIENT_ID', 'mock_client_id')
os.environ.setdefault('REDDIT_CLIENT_SECRET', 'mock_client_secret')
os.environ.setdefault('REDDIT_USER_AGENT', 'mock_agent')
os.environ.setdefault('GEMINI_API_KEY', 'mock_gemini_key')

def test_import(module_name, desc):
    """Test importing a single module"""
    try:
        exec(f"from {module_name} import *")
        print(f"✅ {desc}")
        return True
    except Exception as e:
        print(f"❌ {desc}: {e}")
        return False

def main():
    """Test individual imports"""
    print("🧪 Testing Individual Component Imports")
    print("=" * 50)
    
    tests = [
        ("src.config.settings", "Configuration"),
        ("src.models", "Data Models"),
        ("src.integrations.ai_client", "AI Client"),
        ("src.integrations.gemini_ai_client", "Gemini AI Client"),
        ("src.integrations.reddit_client", "Reddit Client"),
        ("src.integrations.youtube_client", "YouTube Client"),
        ("src.integrations.tts_service", "TTS Service"),
        ("src.processing.video_processor", "Video Processor"),
        ("src.processing.cinematic_editor", "Cinematic Editor"),
        ("src.processing.advanced_audio_processor", "Advanced Audio Processor"),
        ("src.processing.enhanced_thumbnail_generator", "Enhanced Thumbnail Generator"),
        ("src.processing.enhancement_optimizer", "Enhancement Optimizer"),
        ("src.management.channel_manager", "Channel Manager"),
        ("src.monitoring.engagement_metrics", "Engagement Monitor"),
        ("src.utils.gpu_memory_manager", "GPU Memory Manager"),
        ("src.processing.long_form_video_generator", "Long-form Video Generator"),
        ("src.enhanced_orchestrator", "Enhanced Orchestrator"),
    ]
    
    passed = 0
    for module_name, desc in tests:
        if test_import(module_name, desc):
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(tests)} imports successful")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)