#!/usr/bin/env python3
"""
Diagnostic script for YouTube automation tool
Checks system requirements, API connectivity, and common issues
"""

import os
import sys
import json
import logging
import subprocess
import pathlib
from typing import Dict, Any, Tuple

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_python_version() -> bool:
    """Check Python version compatibility"""
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("❌ Python 3.8+ required")
        return False
    else:
        logger.info("✅ Python version OK")
        return True

def check_environment_variables() -> Dict[str, bool]:
    """Check required environment variables"""
    required_vars = {
        'REDDIT_CLIENT_ID': 'Reddit Client ID',
        'REDDIT_CLIENT_SECRET': 'Reddit Client Secret', 
        'GEMINI_API_KEY': 'Gemini API Key',
        'GOOGLE_CLIENT_SECRETS_FILE': 'Google Client Secrets File',
        'YOUTUBE_TOKEN_FILE': 'YouTube Token File'
    }
    
    results = {}
    logger.info("\n🔧 Environment Variables:")
    
    for var, description in required_vars.items():
        value = os.environ.get(var)
        if value:
            if var.endswith('_FILE'):
                # Check if file exists
                if os.path.exists(value):
                    logger.info(f"✅ {description}: {value}")
                    results[var] = True
                else:
                    logger.error(f"❌ {description}: File not found - {value}")
                    results[var] = False
            else:
                logger.info(f"✅ {description}: {'*' * 8}...")
                results[var] = True
        else:
            logger.error(f"❌ {description}: Not set")
            results[var] = False
    
    return results

def check_dependencies() -> Dict[str, bool]:
    """Check required Python packages"""
    required_packages = [
        'moviepy', 'yt_dlp', 'praw', 'google-generativeai',
        'googleapiclient', 'google-auth-oauthlib', 'transformers',
        'torch', 'PIL', 'cv2', 'numpy', 'requests'
    ]
    
    results = {}
    logger.info("\n📦 Python Dependencies:")
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            logger.info(f"✅ {package}")
            results[package] = True
        except ImportError:
            logger.error(f"❌ {package}")
            results[package] = False
    
    return results

def check_system_tools() -> Dict[str, bool]:
    """Check system tools like FFmpeg"""
    tools = ['ffmpeg', 'ffprobe']
    results = {}
    
    logger.info("\n🛠️ System Tools:")
    
    for tool in tools:
        try:
            result = subprocess.run([tool, '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                logger.info(f"✅ {tool}: {version_line}")
                results[tool] = True
            else:
                logger.error(f"❌ {tool}: Not working")
                results[tool] = False
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            logger.error(f"❌ {tool}: Not found")
            results[tool] = False
    
    return results

def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability"""
    gpu_info = {}
    
    logger.info("\n🎮 GPU Information:")
    
    # Check PyTorch CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        gpu_info['torch_cuda'] = cuda_available
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            logger.info(f"✅ PyTorch CUDA: {gpu_name} (Device {current_device}/{gpu_count})")
            gpu_info['gpu_name'] = gpu_name
            gpu_info['gpu_count'] = gpu_count
        else:
            logger.warning("⚠️ PyTorch CUDA: Not available")
    except ImportError:
        logger.error("❌ PyTorch: Not installed")
        gpu_info['torch_cuda'] = False
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpus = result.stdout.strip().split('\n')
            logger.info(f"✅ NVIDIA GPUs detected: {', '.join(gpus)}")
            gpu_info['nvidia_gpus'] = gpus
        else:
            logger.warning("⚠️ nvidia-smi: No GPUs found")
            gpu_info['nvidia_gpus'] = []
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("⚠️ nvidia-smi: Not available")
        gpu_info['nvidia_gpus'] = []
    
    return gpu_info

def check_file_permissions() -> bool:
    """Check file permissions in working directory"""
    logger.info("\n📁 File Permissions:")
    
    try:
        # Test write permission
        test_file = pathlib.Path('test_permissions.tmp')
        test_file.write_text('test')
        test_file.unlink()
        logger.info("✅ Write permissions OK")
        
        # Check temp directory
        temp_dir = pathlib.Path('temp_processing')
        temp_dir.mkdir(exist_ok=True)
        logger.info(f"✅ Temp directory: {temp_dir.absolute()}")
        
        return True
    except Exception as e:
        logger.error(f"❌ File permissions issue: {e}")
        return False

def test_api_connections() -> Dict[str, bool]:
    """Test API connections"""
    logger.info("\n🌐 API Connectivity Tests:")
    
    results = {}
    
    # Test Reddit API
    try:
        import praw
        reddit_client_id = os.environ.get('REDDIT_CLIENT_ID')
        reddit_client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
        reddit_user_agent = os.environ.get('REDDIT_USER_AGENT', 'test:script:v1.0')
        
        if reddit_client_id and reddit_client_secret:
            reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=reddit_user_agent
            )
            # Test with a simple request
            subreddit = reddit.subreddit('test')
            next(subreddit.hot(limit=1))
            logger.info("✅ Reddit API: Connected")
            results['reddit'] = True
        else:
            logger.error("❌ Reddit API: Missing credentials")
            results['reddit'] = False
    except Exception as e:
        logger.error(f"❌ Reddit API: {str(e)[:100]}...")
        results['reddit'] = False
    
    # Test Gemini API
    try:
        import google.generativeai as genai
        gemini_key = os.environ.get('GEMINI_API_KEY')
        
        if gemini_key:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-pro')
            # Test with a simple request
            response = model.generate_content("Hello")
            logger.info("✅ Gemini API: Connected")
            results['gemini'] = True
        else:
            logger.error("❌ Gemini API: Missing API key")
            results['gemini'] = False
    except Exception as e:
        logger.error(f"❌ Gemini API: {str(e)[:100]}...")
        results['gemini'] = False
    
    # Test YouTube API
    try:
        youtube_token_file = os.environ.get('YOUTUBE_TOKEN_FILE')
        if youtube_token_file and os.path.exists(youtube_token_file):
            from script import load_youtube_token
            creds = load_youtube_token(youtube_token_file)
            if creds:
                logger.info("✅ YouTube API: Token loaded")
                results['youtube'] = True
            else:
                logger.error("❌ YouTube API: Invalid token")
                results['youtube'] = False
        else:
            logger.error("❌ YouTube API: Token file not found")
            results['youtube'] = False
    except Exception as e:
        logger.error(f"❌ YouTube API: {str(e)[:100]}...")
        results['youtube'] = False
    
    return results

def generate_report() -> Dict[str, Any]:
    """Generate comprehensive diagnostic report"""
    report = {
        'system': {
            'python_ok': check_python_version(),
            'file_permissions_ok': check_file_permissions()
        },
        'environment': check_environment_variables(),
        'dependencies': check_dependencies(),
        'system_tools': check_system_tools(),
        'gpu': check_gpu_availability(),
        'api_connections': test_api_connections()
    }
    
    return report

def print_summary(report: Dict[str, Any]):
    """Print diagnostic summary"""
    logger.info("\n" + "="*50)
    logger.info("📊 DIAGNOSTIC SUMMARY")
    logger.info("="*50)
    
    # Count issues
    total_checks = 0
    passed_checks = 0
    
    for category, results in report.items():
        if isinstance(results, dict):
            for item, status in results.items():
                if isinstance(status, bool):
                    total_checks += 1
                    if status:
                        passed_checks += 1
        elif isinstance(results, bool):
            total_checks += 1
            if results:
                passed_checks += 1
    
    success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    
    logger.info(f"Overall Status: {passed_checks}/{total_checks} checks passed ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        logger.info("🟢 System Status: EXCELLENT")
    elif success_rate >= 75:
        logger.info("🟡 System Status: GOOD (minor issues)")
    elif success_rate >= 50:
        logger.info("🟠 System Status: FAIR (some issues)")
    else:
        logger.info("🔴 System Status: POOR (major issues)")
    
    # Specific recommendations
    logger.info("\n📝 Recommendations:")
    
    if not report['dependencies'].get('torch', False):
        logger.info("• Install PyTorch: pip install torch")
    
    if not report['system_tools'].get('ffmpeg', False):
        logger.info("• Install FFmpeg: https://ffmpeg.org/download.html")
    
    if not report['api_connections'].get('reddit', False):
        logger.info("• Check Reddit API credentials in .env file")
    
    if not report['api_connections'].get('youtube', False):
        logger.info("• Run: python auth_youtube.py to fix YouTube authentication")
    
    if not report['gpu']['torch_cuda']:
        logger.info("• For better performance, ensure CUDA-compatible GPU and PyTorch")

def main():
    """Main diagnostic function"""
    logger.info("🔍 Starting system diagnostics...")
    logger.info("This will check your setup for the YouTube automation tool.")
    
    report = generate_report()
    print_summary(report)
    
    # Save report to file
    try:
        with open('diagnostic_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"\n💾 Detailed report saved to: diagnostic_report.json")
    except Exception as e:
        logger.error(f"Could not save report: {e}")

if __name__ == "__main__":
    main()