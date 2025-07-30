"""
Central Application class that coordinates all system components
"""

import logging
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import get_config
from src.scheduling import Scheduler
from src.content import ContentSource
from src.pipeline import PipelineManager

# Import config validator
try:
    from config_validator import ConfigValidator
    CONFIG_VALIDATION_AVAILABLE = True
except ImportError:
    CONFIG_VALIDATION_AVAILABLE = False

# Import existing components with fallbacks
try:
    from src.analysis.advanced_content_analyzer import AdvancedContentAnalyzer
    from src.enhanced_orchestrator import EnhancedVideoOrchestrator
    from src.management.channel_manager import ChannelManager
    from src.integrations.reddit_client import create_reddit_client
    CORE_COMPONENTS_AVAILABLE = True
except ImportError:
    CORE_COMPONENTS_AVAILABLE = False


class Application:
    """
    Central application class that coordinates all system components.
    Single responsibility: Wire together and orchestrate all components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, autonomous_args: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or get_config()
        
        # Store autonomous mode arguments
        self.autonomous_args = autonomous_args or {}
        
        # Initialize core components with dependency injection
        # Pass autonomous args to scheduler for configuration
        scheduler_config = dict(self.config.__dict__ if hasattr(self.config, '__dict__') else self.config)
        if self.autonomous_args:
            # Override configuration with command line arguments
            if 'max_videos_per_day' in self.autonomous_args:
                scheduler_config['max_videos_per_day'] = self.autonomous_args['max_videos_per_day']
            if 'min_videos_per_day' in self.autonomous_args:
                scheduler_config['min_videos_per_day'] = self.autonomous_args['min_videos_per_day']
            if 'video_check_interval' in self.autonomous_args:
                scheduler_config['video_check_interval'] = self.autonomous_args['video_check_interval']
                
        self.scheduler = Scheduler(scheduler_config)
        self.content_source = None
        self.pipeline_manager = None
        
        # Legacy components (to be initialized async)
        self.content_analyzer = None
        self.orchestrator = None
        self.channel_manager = None
        self.reddit_client = None
        
        # Application state
        self.running = False
        self.initialization_complete = False
        
        # Validate configuration on startup
        self._validate_configuration()
        
        self.logger.info("Application initialized with configuration")
        
        # Log autonomous mode configuration
        if self.autonomous_args:
            self.logger.info(f"Autonomous mode configuration:")
            for key, value in self.autonomous_args.items():
                self.logger.info(f"  {key}: {value}")
        
    async def initialize_async_components(self):
        """
        Initialize components that require async setup.
        """
        try:
            self.logger.info("Initializing async components...")
            
            # Initialize content analyzer if available
            if CORE_COMPONENTS_AVAILABLE:
                try:
                    self.content_analyzer = AdvancedContentAnalyzer()
                    self.orchestrator = EnhancedVideoOrchestrator()
                    self.channel_manager = ChannelManager()
                    
                    # Initialize Reddit client
                    self.reddit_client = await create_reddit_client()
                    if self.reddit_client and hasattr(self.reddit_client, 'is_connected') and self.reddit_client.is_connected():
                        self.logger.info("✅ Reddit client connected")
                    else:
                        self.logger.warning("⚠️ Reddit client not available, using simulated content")
                        
                except Exception as e:
                    self.logger.warning(f"Some core components failed to initialize: {e}")
                    
            # Initialize content source with dependencies
            self.content_source = ContentSource(
                reddit_client=self.reddit_client,
                content_analyzer=self.content_analyzer
            )
            
            # Initialize pipeline manager
            # Note: We'll pass None for now instead of global dependencies
            self.pipeline_manager = PipelineManager(
                task_queue=None,  # TODO: Pass actual task queue instance
                parallel_manager=None  # TODO: Pass actual parallel manager instance
            )
            
            self.initialization_complete = True
            self.logger.info("✅ All async components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Async component initialization failed: {e}")
            raise
            
    async def run_autonomous_mode(self):
        """
        Run the application in autonomous mode.
        """
        if not self.initialization_complete:
            await self.initialize_async_components()
            
        self.logger.info("🚀 Starting autonomous video generation mode")
        self.running = True
        
        try:
            while self.running:
                await self._autonomous_cycle()
                
                # Wait before next cycle  
                check_interval = self.autonomous_args.get('video_check_interval', 60)
                await asyncio.sleep(check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"Autonomous mode error: {e}")
        finally:
            await self._cleanup()
            
    async def _autonomous_cycle(self):
        """
        Execute one cycle of autonomous video generation.
        """
        try:
            # Check if we should generate a video
            if not self.scheduler.should_generate_video():
                return
                
            self.logger.info("📹 Starting video generation cycle")
            
            # Find and analyze content
            content_items = await self.content_source.find_and_analyze_content(max_items=3)
            
            if not content_items:
                self.logger.warning("No suitable content found for video generation")
                return
                
            # Select best content item
            best_content = max(content_items, key=lambda x: x.get('analysis', {}).get('overall_score', 0))
            self.logger.info(f"Selected content: {best_content.get('content', {}).get('title', 'Unknown')}")
            
            # Process through pipeline
            result = await self.pipeline_manager.process_content_through_pipeline(best_content)
            
            if result.get('success', False):
                self.scheduler.increment_daily_count()
                self.logger.info("✅ Video generation cycle completed successfully")
                
                # Log performance statistics
                await self._log_performance_stats()
            else:
                self.logger.error(f"❌ Video generation failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.logger.error(f"Autonomous cycle error: {e}")
            
    async def run_single_video_mode(self, topic: str):
        """
        Generate a single video on the specified topic.
        
        Args:
            topic: Topic for video generation
        """
        if not self.initialization_complete:
            await self.initialize_async_components()
            
        self.logger.info(f"🎬 Generating single video on topic: {topic}")
        
        try:
            # Create content item from topic
            content_item = {
                'content': {
                    'title': topic,
                    'selftext': f"Video content about: {topic}",
                    'score': 1000,  # High score for manual topics
                    'source': 'manual'
                },
                'analysis': {
                    'overall_score': 0.8,
                    'keywords': topic.split()[:5]
                },
                'source': 'manual_input',
                'retrieved_at': datetime.now().isoformat()
            }
            
            # Process through pipeline
            result = await self.pipeline_manager.process_content_through_pipeline(content_item)
            
            if result.get('success', False):
                self.logger.info("✅ Single video generation completed successfully")
                return result
            else:
                self.logger.error(f"❌ Single video generation failed: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            self.logger.error(f"Single video generation error: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _log_performance_stats(self):
        """
        Log current performance statistics.
        """
        try:
            scheduler_stats = self.scheduler.get_stats()
            pipeline_stats = self.pipeline_manager.get_pipeline_stats()
            
            self.logger.info("📊 Performance Statistics:")
            self.logger.info(f"   Daily videos: {scheduler_stats['daily_video_count']}/{scheduler_stats['max_videos_per_day']}")
            self.logger.info(f"   Pipeline success rate: {pipeline_stats['success_rate_percent']}%")
            self.logger.info(f"   Total processed: {pipeline_stats['total_processed']}")
            self.logger.info(f"   Next scheduled: {scheduler_stats['next_scheduled_time']}")
            
        except Exception as e:
            self.logger.warning(f"Failed to log performance stats: {e}")
            
    async def _cleanup(self):
        """
        Cleanup resources and shutdown gracefully.
        """
        self.logger.info("🧹 Cleaning up application resources")
        self.running = False
        
        try:
            # Close any open connections
            if self.reddit_client and hasattr(self.reddit_client, 'close'):
                await self.reddit_client.close()
                
            # Additional cleanup can be added here
            
        except Exception as e:
            self.logger.warning(f"Cleanup error (non-critical): {e}")
            
        self.logger.info("✅ Application cleanup completed")
        
    def stop(self):
        """
        Signal the application to stop gracefully.
        """
        self.logger.info("Received stop signal")
        self.running = False
        
    def _validate_configuration(self):
        """
        Validate configuration and fail fast on critical issues.
        """
        if not CONFIG_VALIDATION_AVAILABLE:
            self.logger.warning("Configuration validator not available, skipping validation")
            return
            
        try:
            validator = ConfigValidator()
            
            # Convert ConfigManager to dict for validation
            if hasattr(self.config, '__dict__'):
                config_dict = {}
                for attr_name in ['video', 'audio', 'api', 'ai_features', 'content', 'paths']:
                    if hasattr(self.config, attr_name):
                        attr_obj = getattr(self.config, attr_name)
                        if hasattr(attr_obj, '__dict__'):
                            config_dict[attr_name] = attr_obj.__dict__
                        else:
                            config_dict[attr_name] = attr_obj
            else:
                config_dict = self.config
                
            issues = validator.validate_config(config_dict)
            
            # Check for critical issues
            critical_issues = [issue for issue in issues if issue.severity == "critical"]
            warning_issues = [issue for issue in issues if issue.severity == "warning"]
            
            if critical_issues:
                self.logger.error("❌ Critical configuration issues found:")
                for issue in critical_issues:
                    self.logger.error(f"   • {issue.section}.{issue.key}: {issue.description}")
                    self.logger.error(f"     Fix: {issue.fix_suggestion}")
                    
                raise RuntimeError(f"Critical configuration issues found: {len(critical_issues)} issues must be resolved before starting")
                
            if warning_issues:
                self.logger.warning(f"⚠️ Found {len(warning_issues)} configuration warnings:")
                for issue in warning_issues[:3]:  # Show first 3 warnings
                    self.logger.warning(f"   • {issue.section}.{issue.key}: {issue.description}")
                if len(warning_issues) > 3:
                    self.logger.warning(f"   ... and {len(warning_issues) - 3} more warnings")
                    
            if not critical_issues and not warning_issues:
                self.logger.info("✅ Configuration validation passed")
                
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            # Don't fail startup for validation errors, just warn
            self.logger.warning("Continuing startup despite validation issues")