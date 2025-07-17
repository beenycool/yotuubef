"""
Autonomous Configuration Handler
Handles configuration for fully autonomous operation with graceful fallbacks
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

from src.config.settings import get_config


class AutonomousConfigHandler:
    """
    Handles configuration for autonomous mode with intelligent defaults
    and graceful fallbacks when API keys are missing
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        self.fallback_mode = False
        self.missing_services = []
        
    def validate_autonomous_config(self) -> Dict[str, Any]:
        """
        Validate configuration for autonomous operation
        Returns configuration status and recommendations
        """
        self.logger.info("🔍 Validating autonomous configuration...")
        
        validation_results = {
            'status': 'ready',
            'missing_services': [],
            'fallback_services': [],
            'recommendations': [],
            'critical_issues': []
        }
        
        # Check API keys and services
        self._check_reddit_config(validation_results)
        self._check_youtube_config(validation_results)
        self._check_ai_config(validation_results)
        self._check_tts_config(validation_results)
        
        # Check file system requirements
        self._check_file_system(validation_results)
        
        # Determine overall status
        if validation_results['critical_issues']:
            validation_results['status'] = 'critical'
        elif validation_results['missing_services']:
            validation_results['status'] = 'degraded'
            validation_results['fallback_services'] = validation_results['missing_services']
        
        # Log validation results
        self._log_validation_results(validation_results)
        
        return validation_results
    
    def _check_reddit_config(self, results: Dict[str, Any]):
        """Check Reddit API configuration"""
        reddit_client_id = os.getenv('REDDIT_CLIENT_ID') or self.config.get('api', {}).get('reddit_client_id')
        reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET') or self.config.get('api', {}).get('reddit_client_secret')
        
        if not reddit_client_id or reddit_client_id == 'your_reddit_client_id':
            results['missing_services'].append('reddit_api')
            results['recommendations'].append(
                "Reddit API: Set REDDIT_CLIENT_ID environment variable or update config.yaml"
            )
        
        if not reddit_client_secret or reddit_client_secret == 'your_reddit_client_secret':
            results['missing_services'].append('reddit_api')
            results['recommendations'].append(
                "Reddit API: Set REDDIT_CLIENT_SECRET environment variable or update config.yaml"
            )
    
    def _check_youtube_config(self, results: Dict[str, Any]):
        """Check YouTube API configuration"""
        youtube_api_key = os.getenv('YOUTUBE_API_KEY') or self.config.get('api', {}).get('youtube_api_key')
        youtube_credentials = Path('youtube_credentials.json')
        
        if not youtube_api_key or youtube_api_key == 'your_youtube_api_key':
            results['missing_services'].append('youtube_api')
            results['recommendations'].append(
                "YouTube API: Set YOUTUBE_API_KEY environment variable or update config.yaml"
            )
        
        if not youtube_credentials.exists():
            results['missing_services'].append('youtube_upload')
            results['recommendations'].append(
                "YouTube Upload: Place youtube_credentials.json in project root"
            )
    
    def _check_ai_config(self, results: Dict[str, Any]):
        """Check AI service configuration"""
        gemini_api_key = os.getenv('GEMINI_API_KEY') or self.config.get('api', {}).get('gemini_api_key')
        
        if not gemini_api_key or gemini_api_key == 'your_gemini_api_key':
            results['missing_services'].append('ai_analysis')
            results['recommendations'].append(
                "AI Analysis: Set GEMINI_API_KEY environment variable or update config.yaml"
            )
    
    def _check_tts_config(self, results: Dict[str, Any]):
        """Check TTS service configuration"""
        # TTS can work with edge-tts without API keys
        # Only warn about premium services
        elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
        azure_key = os.getenv('AZURE_SPEECH_KEY')
        
        if not elevenlabs_key and not azure_key:
            results['recommendations'].append(
                "TTS: Consider adding ELEVENLABS_API_KEY or AZURE_SPEECH_KEY for premium voices"
            )
    
    def _check_file_system(self, results: Dict[str, Any]):
        """Check file system requirements"""
        required_dirs = ['data', 'temp', 'logs', 'downloads', 'processed']
        
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ Created directory: {dir_path}")
                except Exception as e:
                    results['critical_issues'].append(f"Cannot create directory {dir_name}: {e}")
    
    def _log_validation_results(self, results: Dict[str, Any]):
        """Log validation results with appropriate formatting"""
        status = results['status']
        
        if status == 'ready':
            self.logger.info("✅ Autonomous configuration: READY")
        elif status == 'degraded':
            self.logger.warning("⚠️ Autonomous configuration: DEGRADED (fallback mode enabled)")
        else:
            self.logger.error("❌ Autonomous configuration: CRITICAL ISSUES")
        
        # Log missing services
        if results['missing_services']:
            self.logger.warning(f"Missing services: {', '.join(results['missing_services'])}")
        
        # Log recommendations
        for rec in results['recommendations']:
            self.logger.info(f"💡 {rec}")
        
        # Log critical issues
        for issue in results['critical_issues']:
            self.logger.error(f"🚨 {issue}")
    
    def setup_fallback_config(self) -> Dict[str, Any]:
        """
        Setup configuration for fallback mode when services are missing
        """
        self.logger.info("🔄 Setting up fallback configuration...")
        
        fallback_config = {
            'enable_reddit_fallback': True,
            'enable_ai_fallback': True,
            'enable_local_processing': True,
            'use_basic_thumbnails': True,
            'use_edge_tts': True,
            'skip_upload_if_no_auth': True,
            'use_cached_content': True,
            'enable_mock_services': True
        }
        
        return fallback_config
    
    def get_autonomous_ready_config(self) -> Dict[str, Any]:
        """
        Get configuration optimized for autonomous operation
        """
        # Validate current configuration
        validation = self.validate_autonomous_config()
        
        # Base configuration for autonomous mode
        autonomous_config = {
            'autonomous_mode': True,
            'enable_auto_optimization': True,
            'enable_proactive_management': True,
            'enable_intelligent_scheduling': True,
            'enable_error_recovery': True,
            'enable_graceful_degradation': True,
            'max_retries': 3,
            'retry_delay': 300,  # 5 minutes
            'cleanup_interval': 3600,  # 1 hour
            'stats_reporting_interval': 3600,  # 1 hour
        }
        
        # Add fallback configuration if needed
        if validation['status'] == 'degraded':
            autonomous_config.update(self.setup_fallback_config())
        
        return autonomous_config
    
    def create_autonomous_env_template(self):
        """Create template .env file for autonomous operation"""
        env_template = """# Autonomous YouTube Video Generator - Environment Variables
# Copy this file to .env and fill in your actual API keys

# Reddit API (Required for content discovery)
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here

# YouTube API (Required for video upload)
YOUTUBE_API_KEY=your_youtube_api_key_here

# AI Services (Required for content analysis)
GEMINI_API_KEY=your_gemini_api_key_here

# TTS Services (Optional - system will use free edge-tts if not provided)
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
AZURE_SPEECH_KEY=your_azure_speech_key_here
AZURE_SPEECH_REGION=your_azure_region_here

# Development Settings
DEBUG_MODE=false
MOCK_API_CALLS=false
"""
        
        env_file = Path('.env.autonomous.template')
        with open(env_file, 'w') as f:
            f.write(env_template)
        
        self.logger.info(f"✅ Created environment template: {env_file}")
        return env_file


def setup_autonomous_config():
    """Setup configuration for autonomous operation"""
    handler = AutonomousConfigHandler()
    
    # Create environment template if it doesn't exist
    env_template = Path('.env.autonomous.template')
    if not env_template.exists():
        handler.create_autonomous_env_template()
    
    # Get autonomous-ready configuration
    config = handler.get_autonomous_ready_config()
    
    return config, handler.validate_autonomous_config()


if __name__ == "__main__":
    # Allow running this module directly for configuration check
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 Checking autonomous configuration...")
    config, validation = setup_autonomous_config()
    
    print(f"\n📊 Configuration Status: {validation['status'].upper()}")
    
    if validation['missing_services']:
        print(f"⚠️ Missing Services: {', '.join(validation['missing_services'])}")
    
    if validation['recommendations']:
        print("\n💡 Recommendations:")
        for rec in validation['recommendations']:
            print(f"   - {rec}")
    
    if validation['critical_issues']:
        print("\n🚨 Critical Issues:")
        for issue in validation['critical_issues']:
            print(f"   - {issue}")
    
    print("\n✅ Autonomous configuration check complete!")