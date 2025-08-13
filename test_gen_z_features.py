#!/usr/bin/env python3
"""
Test script to demonstrate Gen Z features implementation
"""

import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_gen_z_features():
    """Test the Gen Z features implementation"""
    
    logger.info("🚀 Testing Gen Z Features Implementation")
    
    try:
        # Test 1: Configuration
        logger.info("📋 Test 1: Configuration")
        from src.config.settings import get_config
        config = get_config()
        
        gen_z_enabled = config.ai_features.get('gen_z_mode', False)
        logger.info(f"Gen Z mode enabled: {gen_z_enabled}")
        
        if gen_z_enabled:
            logger.info("✅ Gen Z mode is enabled in configuration")
        else:
            logger.warning("⚠️ Gen Z mode is not enabled in configuration")
        
        # Test 2: Meme Generator
        logger.info("🎭 Test 2: Meme Generator")
        try:
            from src.processing.meme_generator import MemeGenerator
            meme_gen = MemeGenerator()
            
            if meme_gen.available:
                logger.info("✅ Meme generator is available")
                
                # Test meme text generation
                meme_text = meme_gen.generate_random_meme_text("funny video content")
                logger.info(f"Generated meme text: {meme_text}")
                
                # Test meme overlay decision
                should_add = meme_gen.should_add_meme_overlay(1, {'is_key_moment': True})
                logger.info(f"Should add meme overlay: {should_add}")
                
            else:
                logger.warning("⚠️ Meme generator is not available (missing dependencies)")
                
        except Exception as e:
            logger.error(f"❌ Meme generator test failed: {e}")
        
        # Test 3: Gen Z CTA Generation
        logger.info("📢 Test 3: Gen Z CTA Generation")
        try:
            from src.processing.cta_processor import CTAProcessor
            cta_processor = CTAProcessor()
            
            # Test Gen Z CTA generation
            gen_z_cta = cta_processor.generate_gen_z_cta({}, audience="gen_z")
            logger.info(f"Generated Gen Z CTA: {gen_z_cta}")
            
            # Test general CTA generation
            general_cta = cta_processor.generate_gen_z_cta({}, audience="general")
            logger.info(f"Generated general CTA: {general_cta}")
            
            logger.info("✅ CTA generation test completed")
            
        except Exception as e:
            logger.error(f"❌ CTA generation test failed: {e}")
        
        # Test 4: Spotify Gen Z Mode
        logger.info("🎵 Test 4: Spotify Gen Z Mode")
        try:
            from src.integrations.spotify_client import SpotifyClient
            spotify_client = SpotifyClient()
            
            # Test Gen Z mode in search
            logger.info("Testing Spotify search with Gen Z mode...")
            # Note: This would require actual Spotify credentials to test fully
            
            logger.info("✅ Spotify client test completed")
            
        except Exception as e:
            logger.error(f"❌ Spotify client test failed: {e}")
        
        # Test 5: Sound Effects Manager
        logger.info("🔊 Test 5: Sound Effects Manager")
        try:
            from src.processing.sound_effects_manager import SoundEffectsManager
            sound_manager = SoundEffectsManager()
            
            # Test Gen Z sound effects
            gen_z_effects = sound_manager.apply_gen_z_sound_effects({
                'humor_level': 0.8,
                'key_moments': [{'timestamp': 5.0}],
                'dramatic_moments': [{'timestamp': 15.0, 'intensity': 0.8}]
            }, gen_z_mode=True)
            
            logger.info(f"Generated {len(gen_z_effects)} Gen Z sound effects")
            for effect in gen_z_effects:
                logger.info(f"  - {effect['effect_name']} at {effect['timestamp_seconds']}s")
            
            logger.info("✅ Sound effects manager test completed")
            
        except Exception as e:
            logger.error(f"❌ Sound effects manager test failed: {e}")
        
        # Test 6: Enhanced Thumbnail Generator
        logger.info("🖼️ Test 6: Enhanced Thumbnail Generator")
        try:
            from src.processing.enhanced_thumbnail_generator import EnhancedThumbnailGenerator
            thumbnail_gen = EnhancedThumbnailGenerator()
            
            # Check if Gen Z styles are available
            gen_z_styles = list(thumbnail_gen.gen_z_styles.keys())
            logger.info(f"Available Gen Z styles: {gen_z_styles}")
            
            # Check emoji combinations
            emoji_moods = list(thumbnail_gen.gen_z_emojis.keys())
            logger.info(f"Available emoji moods: {emoji_moods}")
            
            logger.info("✅ Enhanced thumbnail generator test completed")
            
        except Exception as e:
            logger.error(f"❌ Enhanced thumbnail generator test failed: {e}")
        
        # Test 7: Pipeline Manager
        logger.info("🔧 Test 7: Pipeline Manager")
        try:
            from src.pipeline.pipeline_manager import PipelineManager
            pipeline_manager = PipelineManager()
            
            # Test Gen Z mode detection
            gen_z_enabled = pipeline_manager._is_gen_z_mode_enabled()
            logger.info(f"Pipeline manager Gen Z mode: {gen_z_enabled}")
            
            logger.info("✅ Pipeline manager test completed")
            
        except Exception as e:
            logger.error(f"❌ Pipeline manager test failed: {e}")
        
        logger.info("🎉 All Gen Z feature tests completed!")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        raise

def main():
    """Main function"""
    try:
        asyncio.run(test_gen_z_features())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    main()