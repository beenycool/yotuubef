#!/usr/bin/env python3
"""
Quick Setup and Usage Guide for YouTube Video Generator Improvements
"""

import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check for required dependencies"""
    print("🔍 Checking Dependencies...")
    
    required_deps = [
        'pydantic', 'python-dotenv', 'pyyaml', 'aiofiles', 'praw'
    ]
    
    optional_deps = [
        'numpy', 'opencv-python', 'moviepy', 'psutil', 'asyncpraw', 
        'google-generativeai', 'elevenlabs', 'yt-dlp'
    ]
    
    missing_required = []
    missing_optional = []
    
    for dep in required_deps:
        try:
            if dep == 'python-dotenv':
                __import__('dotenv')
            elif dep == 'pyyaml':
                __import__('yaml')
            else:
                __import__(dep.replace('-', '_'))
            print(f"✅ {dep}")
        except ImportError:
            missing_required.append(dep)
            print(f"❌ {dep} (REQUIRED)")
    
    for dep in optional_deps:
        try:
            if dep == 'opencv-python':
                __import__('cv2')
            else:
                __import__(dep.replace('-', '_'))
            print(f"✅ {dep}")
        except ImportError:
            missing_optional.append(dep)
            print(f"⚠️  {dep} (OPTIONAL)")
    
    if missing_required:
        print(f"\n❌ Missing required dependencies: {', '.join(missing_required)}")
        print("Install with: pip install " + ' '.join(missing_required))
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional dependencies: {', '.join(missing_optional)}")
        print("Install with: pip install " + ' '.join(missing_optional))
        print("(Optional dependencies enable additional features)")
    
    return True


def run_tests():
    """Run all available tests"""
    print("\n🧪 Running Tests...")
    
    test_files = [
        'test_core_functionality.py',
        'test_imports.py'
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\n📋 Running {test_file}...")
            try:
                result = subprocess.run([sys.executable, test_file], 
                                     capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ {test_file} passed")
                else:
                    print(f"❌ {test_file} failed")
                    print("Error output:")
                    print(result.stdout)
            except Exception as e:
                print(f"❌ Failed to run {test_file}: {e}")
        else:
            print(f"⚠️  {test_file} not found")


def demo_features():
    """Demonstrate key features"""
    print("\n🎬 Demonstrating Key Features...")
    
    features = [
        ('video_quality_enhancer.py', 'Video Quality Enhancement'),
        ('enhanced_error_handler.py', 'Enhanced Error Handling'),
        ('config_validator.py', 'Configuration Validation'),
    ]
    
    for script, description in features:
        if Path(script).exists():
            print(f"\n📊 {description}:")
            print(f"   Run: python {script}")
            try:
                result = subprocess.run([sys.executable, script], 
                                     capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"✅ {description} demo completed successfully")
                else:
                    print(f"❌ {description} demo failed")
            except subprocess.TimeoutExpired:
                print(f"⏱️  {description} demo timed out (this is normal)")
            except Exception as e:
                print(f"❌ Failed to run {description}: {e}")
        else:
            print(f"⚠️  {script} not found")


def show_usage_examples():
    """Show practical usage examples"""
    print("\n💡 Usage Examples:")
    
    print("\n1. 🎥 Enhance Video Quality:")
    print("   from video_quality_enhancer import VideoQualityEnhancer")
    print("   enhancer = VideoQualityEnhancer()")
    print("   settings = enhancer.get_optimal_settings(60.0)  # 60 seconds")
    print("   ffmpeg_cmd = enhancer.generate_ffmpeg_command('input.mp4', 'output.mp4', settings)")
    
    print("\n2. 🚨 Setup Error Handling:")
    print("   from enhanced_error_handler import setup_global_error_handler")
    print("   error_handler = setup_global_error_handler()")
    print("   # Now all errors are automatically handled and logged")
    
    print("\n3. ⚙️ Validate Configuration:")
    print("   from config_validator import ConfigValidator")
    print("   validator = ConfigValidator()")
    print("   issues = validator.validate_config(config)")
    print("   optimized = validator.optimize_config(config, 'balanced')")
    
    print("\n4. 🧪 Test Core Functionality:")
    print("   python test_core_functionality.py")
    
    print("\n5. 📊 Run Configuration Validation:")
    print("   python config_validator.py")


def main():
    """Main setup and demonstration"""
    print("🚀 YouTube Video Generator - Setup and Usage Guide")
    print("=" * 60)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Please install required dependencies first:")
        print("pip install pydantic python-dotenv pyyaml aiofiles praw")
        return
    
    # Run tests
    run_tests()
    
    # Demo features
    demo_features()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n🎉 Setup Complete!")
    print("\nNext Steps:")
    print("1. Install optional dependencies for full functionality")
    print("2. Configure your API keys in .env file")
    print("3. Run configuration validation: python config_validator.py")
    print("4. Start using the enhanced video generation features!")
    
    print("\n📚 Documentation:")
    print("• IMPROVEMENTS_SUMMARY.md - Detailed overview of all improvements")
    print("• README.md - Original project documentation")
    print("• config.yaml - Configuration file with all settings")
    
    print("\n🔧 Tools Available:")
    print("• video_quality_enhancer.py - Video quality optimization")
    print("• enhanced_error_handler.py - Advanced error handling")
    print("• config_validator.py - Configuration validation and optimization")
    print("• test_core_functionality.py - Core functionality testing")
    print("• test_imports.py - Import validation testing")


if __name__ == "__main__":
    main()