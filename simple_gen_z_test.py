#!/usr/bin/env python3
"""
Simple test script for Gen Z features (no external dependencies)
"""

import sys
import os

def test_config_file():
    """Test if the config.yaml file has Gen Z settings"""
    print("🔧 Testing Configuration File...")
    
    try:
        with open('config.yaml', 'r') as f:
            content = f.read()
            
        # Check for Gen Z mode
        if 'gen_z_mode: true' in content:
            print("✅ Gen Z mode is enabled in config.yaml")
        else:
            print("⚠️ Gen Z mode is not enabled in config.yaml")
            
        # Check for Gen Z features
        gen_z_features = [
            'gen_z_pacing: true',
            'gen_z_trends: true',
            'enable_meme_overlays: true',
            'enable_trending_audio: true',
            'enable_interactive_ctas: true',
            'enable_vibrant_thumbnails: true'
        ]
        
        found_features = []
        for feature in gen_z_features:
            if feature in content:
                found_features.append(feature)
        
        print(f"✅ Found {len(found_features)} Gen Z features in config:")
        for feature in found_features:
            print(f"   - {feature}")
            
    except Exception as e:
        print(f"❌ Error reading config.yaml: {e}")

def test_file_structure():
    """Test if Gen Z feature files exist"""
    print("\n📁 Testing File Structure...")
    
    gen_z_files = [
        'src/processing/meme_generator.py',
        'src/processing/cta_processor.py',
        'src/processing/sound_effects_manager.py',
        'src/processing/enhanced_thumbnail_generator.py',
        'src/processing/cinematic_editor.py',
        'src/integrations/spotify_client.py',
        'src/integrations/gemini_ai_client.py',
        'src/pipeline/pipeline_manager.py'
    ]
    
    existing_files = []
    missing_files = []
    
    for file_path in gen_z_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    print(f"✅ Found {len(existing_files)} Gen Z feature files:")
    for file_path in existing_files:
        print(f"   - {file_path}")
    
    if missing_files:
        print(f"⚠️ Missing {len(missing_files)} files:")
        for file_path in missing_files:
            print(f"   - {file_path}")

def test_code_quality():
    """Test basic code quality of Gen Z features"""
    print("\n🔍 Testing Code Quality...")
    
    try:
        # Test meme generator
        with open('src/processing/meme_generator.py', 'r') as f:
            meme_content = f.read()
            
        if 'class MemeGenerator:' in meme_content:
            print("✅ MemeGenerator class found")
        else:
            print("❌ MemeGenerator class not found")
            
        if 'add_meme_overlay' in meme_content:
            print("✅ add_meme_overlay method found")
        else:
            print("❌ add_meme_overlay method not found")
            
        # Test CTA processor
        with open('src/processing/cta_processor.py', 'r') as f:
            cta_content = f.read()
            
        if 'generate_gen_z_cta' in cta_content:
            print("✅ generate_gen_z_cta method found")
        else:
            print("❌ generate_gen_z_cta method not found")
            
        # Test sound effects manager
        with open('src/processing/sound_effects_manager.py', 'r') as f:
            sound_content = f.read()
            
        if 'apply_gen_z_sound_effects' in sound_content:
            print("✅ apply_gen_z_sound_effects method found")
        else:
            print("❌ apply_gen_z_sound_effects method not found")
            
        # Test thumbnail generator
        with open('src/processing/enhanced_thumbnail_generator.py', 'r') as f:
            thumb_content = f.read()
            
        if 'generate_gen_z_thumbnails' in thumb_content:
            print("✅ generate_gen_z_thumbnails method found")
        else:
            print("❌ generate_gen_z_thumbnails method not found")
            
    except Exception as e:
        print(f"❌ Error testing code quality: {e}")

def test_documentation():
    """Test if documentation exists"""
    print("\n📚 Testing Documentation...")
    
    docs = [
        'GEN_Z_FEATURES.md',
        'README.md'
    ]
    
    for doc in docs:
        if os.path.exists(doc):
            print(f"✅ {doc} exists")
            
            # Check if it contains Gen Z content
            try:
                with open(doc, 'r') as f:
                    content = f.read()
                    if 'Gen Z' in content or 'gen_z' in content:
                        print(f"   - Contains Gen Z content")
                    else:
                        print(f"   - No Gen Z content found")
            except Exception as e:
                print(f"   - Error reading content: {e}")
        else:
            print(f"❌ {doc} missing")

def main():
    """Main test function"""
    print("🚀 Simple Gen Z Features Test")
    print("=" * 40)
    
    test_config_file()
    test_file_structure()
    test_code_quality()
    test_documentation()
    
    print("\n" + "=" * 40)
    print("🎉 Simple Gen Z test completed!")
    print("\nTo run the full test suite, ensure all dependencies are installed:")
    print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()