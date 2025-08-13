"""
Central Application class that coordinates all system components
"""

import logging
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional

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
        
        # Social media integration
        self.social_media_manager = None
        
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
            
            # Initialize social media manager
            try:
                from src.integrations.social_media_manager import create_social_media_manager
                self.social_media_manager = create_social_media_manager(self.config)
                self.logger.info("✅ Social media manager initialized successfully")
            except Exception as e:
                self.logger.warning(f"⚠️ Social media manager not available: {e}")
                self.social_media_manager = None
                    
            # Initialize content source with dependencies
            self.content_source = ContentSource(
                reddit_client=self.reddit_client,
                content_analyzer=self.content_analyzer
            )
            
            # Initialize pipeline manager
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
            
        self.logger.info(f"🎬 Starting single video generation for topic: {topic}")
        
        try:
            # Find content for the topic
            # TODO: Implement topic filtering in ContentSource
            content_items = await self.content_source.find_and_analyze_content(
                max_items=1
            )
            
            if not content_items:
                self.logger.error(f"No content found for topic: {topic}")
                return False
                
            # Process the content
            result = await self.pipeline_manager.process_content_through_pipeline(content_items[0])
            
            if result.get('success', False):
                self.logger.info("✅ Single video generation completed successfully")
                return True
            else:
                self.logger.error(f"❌ Single video generation failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Single video mode error: {e}")
            return False
            
    async def get_system_status(self):
        """
        Get the current system status and health information.
        """
        try:
            self.logger.info("📊 Getting system status...")
            
            # Get scheduler statistics
            scheduler_stats = self.scheduler.get_stats()
            
            # Get component status
            component_status = {
                'scheduler': '✅ Active' if self.scheduler else '❌ Not initialized',
                'content_source': '✅ Active' if self.content_source else '❌ Not initialized',
                'pipeline_manager': '✅ Active' if self.pipeline_manager else '❌ Not initialized',
                'content_analyzer': '✅ Active' if self.content_analyzer else '❌ Not available',
                'orchestrator': '✅ Active' if self.orchestrator else '❌ Not available',
                'channel_manager': '✅ Active' if self.channel_manager else '❌ Not available',
                'reddit_client': '✅ Connected' if (self.reddit_client and hasattr(self.reddit_client, 'is_connected') and self.reddit_client.is_connected()) else '❌ Not connected'
            }
            
            # Print status information
            print("\n" + "=" * 60)
            print("🤖 SYSTEM STATUS")
            print("=" * 60)
            
            print(f"\n📅 Scheduler Status:")
            print(f"   Daily video count: {scheduler_stats.get('daily_video_count', 0)}")
            print(f"   Should generate video: {scheduler_stats.get('should_generate_video', False)}")
            print(f"   Next scheduled time: {scheduler_stats.get('next_scheduled_time', 'Not scheduled')}")
            
            print(f"\n🔧 Component Status:")
            for component, status in component_status.items():
                print(f"   {component.replace('_', ' ').title()}: {status}")
            
            print(f"\n⚙️ Configuration:")
            print(f"   Autonomous mode: {'✅ Enabled' if self.autonomous_args else '❌ Disabled'}")
            if self.autonomous_args:
                for key, value in self.autonomous_args.items():
                    print(f"   {key.replace('_', ' ').title()}: {value}")
            
            print("\n" + "=" * 60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return False
            
    async def _log_performance_stats(self):
        """
        Log performance statistics for monitoring.
        """
        try:
            # Get scheduler statistics
            stats = self.scheduler.get_stats()
            
            # Log key metrics
            self.logger.info(f"📊 Performance Stats - Daily videos: {stats.get('daily_video_count', 0)}")
            
            # Log memory usage if available
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                self.logger.info(f"💾 Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
            except ImportError:
                pass  # psutil not available
                
        except Exception as e:
            self.logger.warning(f"Could not log performance stats: {e}")
            
    async def _cleanup(self):
        """
        Clean up resources before shutdown.
        """
        try:
            self.logger.info("🧹 Cleaning up resources...")
            self.running = False
            
            # Clean up components
            if hasattr(self, 'orchestrator') and self.orchestrator:
                try:
                    await self.orchestrator.cleanup()
                except:
                    pass
                    
            if hasattr(self, 'content_source') and self.content_source:
                try:
                    await self.content_source.cleanup()
                except:
                    pass
                    
            self.logger.info("✅ Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
            
    def _validate_configuration(self):
        """
        Validate the application configuration.
        """
        try:
            # Basic validation
            if not self.config:
                raise ValueError("Configuration is required")
                
            # Validate required paths
            required_paths = ['temp_dir', 'processed_dir', 'logs_dir']
            for path_key in required_paths:
                if hasattr(self.config, path_key):
                    path_value = getattr(self.config, path_key)
                    if path_value:
                        Path(path_value).mkdir(parents=True, exist_ok=True)
                        
            # Validate autonomous mode configuration
            if self.autonomous_args:
                required_autonomous_keys = ['max_videos_per_day', 'min_videos_per_day']
                for key in required_autonomous_keys:
                    if key not in self.autonomous_args:
                        self.logger.warning(f"Missing autonomous configuration: {key}")
                        
            self.logger.info("✅ Configuration validation passed")
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            raise