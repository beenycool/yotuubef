"""
Enhanced Autonomous Video Generator with Advanced Features
Integrates all the new advanced systems for professional-grade automation

⚠️  DEPRECATION NOTICE:
This module contains a large monolithic class that has been refactored.
Use the new Application class from src.application instead:

    from src.application import Application
    app = Application()
    await app.run_autonomous_mode()

The new architecture provides:
- Better separation of concerns
- Easier testing and maintenance
- Clearer dependency management
- Configuration validation at startup

This file is kept for backward compatibility but may be removed in future versions.
"""

import asyncio
import logging
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import random

from src.config.settings import get_config
from src.analysis.advanced_content_analyzer import AdvancedContentAnalyzer
from src.templates.dynamic_video_templates import DynamicVideoTemplateManager
from src.optimization.smart_optimization_engine import SmartOptimizationEngine, TestType
from src.robustness.robust_system import (
    global_task_queue, global_config_manager, 
    RobustRetryHandler, TaskPriority
)
from src.parallel.async_processing import (
    global_parallel_manager, ContentGenerationPipeline,
    WorkerType
)

# Import existing components with fallbacks
try:
    from src.enhanced_orchestrator import EnhancedVideoOrchestrator
    from src.management.channel_manager import ChannelManager
    from src.integrations.reddit_client import create_reddit_client
    CORE_COMPONENTS_AVAILABLE = True
except ImportError:
    CORE_COMPONENTS_AVAILABLE = False


class AdvancedAutonomousGenerator:
    """
    Advanced autonomous video generator with comprehensive improvements:
    - Advanced content analysis with sentiment and trend analysis
    - Dynamic video templates with automated asset sourcing
    - A/B testing and smart optimization
    - Robust error handling with retry mechanisms
    - Parallel processing for improved efficiency
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load enhanced configuration
        self.config = global_config_manager.get_config()
        
        # Initialize advanced components
        self.content_analyzer = AdvancedContentAnalyzer()
        self.template_manager = DynamicVideoTemplateManager()
        self.optimization_engine = SmartOptimizationEngine()
        
        # Initialize core components if available
        if CORE_COMPONENTS_AVAILABLE:
            self.video_orchestrator = EnhancedVideoOrchestrator()
            self.channel_manager = ChannelManager()
        else:
            self.video_orchestrator = None
            self.channel_manager = None
            self.logger.warning("Core components not available - running in simulation mode")
        
        # Initialize Reddit client
        self.reddit_client = None
        
        # Set up parallel processing
        self.content_pipeline = ContentGenerationPipeline(global_parallel_manager)
        
        # Performance tracking
        self.performance_metrics = {
            'videos_analyzed': 0,
            'videos_generated': 0,
            'videos_uploaded': 0,
            'ab_tests_created': 0,
            'optimizations_applied': 0,
            'start_time': datetime.now()
        }
        
        # Operating parameters
        self.max_videos_per_day = int(self.config.get('MAX_VIDEOS_PER_DAY', 8))
        self.min_videos_per_day = int(self.config.get('MIN_VIDEOS_PER_DAY', 3))
        self.video_check_interval = int(self.config.get('VIDEO_CHECK_INTERVAL', 3600))
        
        self.logger.info("Advanced Autonomous Generator initialized")
    
    async def initialize_systems(self):
        """Initialize all subsystems"""
        try:
            self.logger.info("🚀 Initializing advanced autonomous systems...")
            
            # Initialize parallel processing workers
            worker_configs = {
                WorkerType.CONTENT_ANALYZER: 3,
                WorkerType.VIDEO_GENERATOR: 2,
                WorkerType.VIDEO_PROCESSOR: 2,
                WorkerType.UPLOADER: 1,
                WorkerType.ASSET_DOWNLOADER: 2
            }
            
            # Register processors for parallel processing
            self._register_parallel_processors()
            
            # Start parallel processing
            await global_parallel_manager.start_workers(worker_configs)
            
            # Start content pipeline
            await self.content_pipeline.start_pipeline()
            
            # Start task queue processing
            global_task_queue.register_processor('video_generation', self._process_video_generation_task)
            global_task_queue.register_processor('content_analysis', self._process_content_analysis_task)
            global_task_queue.register_processor('ab_test', self._process_ab_test_task)
            
            asyncio.create_task(global_task_queue.start_processing(max_concurrent=5))
            
            # Initialize Reddit client
            if CORE_COMPONENTS_AVAILABLE:
                try:
                    self.reddit_client = await create_reddit_client()
                    if self.reddit_client and self.reddit_client.is_connected():
                        self.logger.info("✅ Reddit client connected")
                    else:
                        self.logger.warning("⚠️ Reddit client not connected")
                except Exception as e:
                    self.logger.warning(f"Reddit client initialization failed: {e}")
            
            self.logger.info("✅ All systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            raise
    
    def _register_parallel_processors(self):
        """Register processors for parallel processing"""
        global_parallel_manager.register_processor(
            WorkerType.CONTENT_ANALYZER, 
            self._parallel_content_analysis
        )
        global_parallel_manager.register_processor(
            WorkerType.VIDEO_GENERATOR, 
            self._parallel_video_generation
        )
        global_parallel_manager.register_processor(
            WorkerType.VIDEO_PROCESSOR, 
            self._parallel_video_processing
        )
        global_parallel_manager.register_processor(
            WorkerType.UPLOADER, 
            self._parallel_video_upload
        )
        global_parallel_manager.register_processor(
            WorkerType.ASSET_DOWNLOADER, 
            self._parallel_asset_download
        )
    
    async def _parallel_content_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parallel processor for content analysis"""
        try:
            content = data.get('content', {})
            title = content.get('title', '')
            description = content.get('description', '')
            metadata = content.get('metadata', {})
            
            analysis_result = await self.content_analyzer.analyze_content(
                title=title,
                description=description,
                metadata=metadata
            )
            
            return {
                'success': True,
                'analysis': analysis_result,
                'content': content
            }
            
        except Exception as e:
            self.logger.error(f"Parallel content analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _parallel_video_generation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parallel processor for video generation"""
        try:
            content = data.get('content', {})
            analysis = data.get('analysis')
            
            # Select optimal template
            template = await self.template_manager.select_optimal_template(
                analysis.__dict__ if analysis else {},
                preferences={}
            )
            
            # Source assets for template
            assets = await self.template_manager.source_assets_for_template(
                template,
                analysis.keywords if analysis else [],
                analysis.topics if analysis else []
            )
            
            # Generate video (simulation if core components not available)
            if self.video_orchestrator:
                # Use real video generation
                video_result = await self.video_orchestrator.process_enhanced_video(
                    content.get('url', ''),
                    options={
                        'template': template,
                        'assets': assets,
                        'analysis': analysis
                    }
                )
            else:
                # Simulate video generation
                await asyncio.sleep(5)  # Simulate processing time
                video_result = {
                    'success': True,
                    'video_path': f"generated_video_{int(datetime.now().timestamp())}.mp4",
                    'template_used': template.name,
                    'assets_used': len(assets.get('background_images', [])),
                    'processing_time': 5.0
                }
            
            return video_result
            
        except Exception as e:
            self.logger.error(f"Parallel video generation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _parallel_video_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parallel processor for video processing"""
        try:
            # Simulate advanced video processing
            await asyncio.sleep(3)
            
            return {
                'success': True,
                'processed_video_path': data.get('video_path', '').replace('.mp4', '_processed.mp4'),
                'enhancements_applied': ['color_correction', 'audio_enhancement', 'stabilization']
            }
            
        except Exception as e:
            self.logger.error(f"Parallel video processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _parallel_video_upload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parallel processor for video upload"""
        try:
            video_data = data.get('video_data', {})
            
            # Generate A/B test variants for upload
            variants = await self._generate_upload_variants(data)
            
            # Select variant for this upload
            selected_variant = random.choice(variants) if variants else {}
            
            # Upload video (simulation if core components not available)
            if self.channel_manager:
                # Use real upload
                upload_result = await self.channel_manager.upload_video_with_optimization(
                    video_path=video_data.get('video_path', ''),
                    metadata=selected_variant
                )
            else:
                # Simulate upload
                await asyncio.sleep(2)
                upload_result = {
                    'success': True,
                    'video_id': f"video_{int(datetime.now().timestamp())}",
                    'video_url': f"https://youtube.com/watch?v=video_{int(datetime.now().timestamp())}",
                    'variant_used': selected_variant
                }
            
            # Track A/B test data if applicable
            if selected_variant.get('test_id'):
                await self._track_ab_test_upload(upload_result, selected_variant)
            
            return upload_result
            
        except Exception as e:
            self.logger.error(f"Parallel video upload failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _parallel_asset_download(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parallel processor for asset download"""
        try:
            assets = data.get('assets', [])
            downloaded_assets = []
            
            for asset in assets[:5]:  # Limit to 5 assets per batch
                # Simulate asset download
                await asyncio.sleep(0.5)
                downloaded_assets.append({
                    'url': asset.get('url', ''),
                    'local_path': f"assets/downloaded_{len(downloaded_assets)}.jpg",
                    'asset_type': asset.get('asset_type', 'image')
                })
            
            return {
                'success': True,
                'downloaded_assets': downloaded_assets
            }
            
        except Exception as e:
            self.logger.error(f"Parallel asset download failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _generate_upload_variants(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate A/B test variants for upload"""
        try:
            original_content = data.get('original_content', {})
            analysis = data.get('analysis')
            
            # Check if we should create A/B tests
            if not await self._should_create_ab_test():
                return [{}]  # Return empty variant (no A/B test)
            
            variants = []
            
            # Generate title variants
            if random.random() < 0.3:  # 30% chance for title A/B test
                title_variants = await self._generate_title_variants(
                    original_content.get('title', ''),
                    analysis.keywords if analysis else []
                )
                
                for i, title in enumerate(title_variants):
                    variants.append({
                        'title': title,
                        'test_type': 'title',
                        'variant_name': f'Title Variant {i+1}',
                        'test_id': f"title_test_{int(datetime.now().timestamp())}"
                    })
            
            # Generate thumbnail variants (placeholder)
            if random.random() < 0.4:  # 40% chance for thumbnail A/B test
                thumbnail_variants = [
                    'thumbnail_variant_1.jpg',
                    'thumbnail_variant_2.jpg',
                    'thumbnail_variant_3.jpg'
                ]
                
                for i, thumbnail in enumerate(thumbnail_variants):
                    variants.append({
                        'thumbnail': thumbnail,
                        'test_type': 'thumbnail',
                        'variant_name': f'Thumbnail Variant {i+1}',
                        'test_id': f"thumbnail_test_{int(datetime.now().timestamp())}"
                    })
            
            return variants[:3]  # Limit to 3 variants
            
        except Exception as e:
            self.logger.error(f"Failed to generate upload variants: {e}")
            return [{}]
    
    async def _should_create_ab_test(self) -> bool:
        """Determine if an A/B test should be created"""
        # Simple logic - create A/B tests for 20% of uploads
        return random.random() < 0.2
    
    async def _generate_title_variants(self, original_title: str, keywords: List[str]) -> List[str]:
        """Generate title variants for A/B testing"""
        try:
            variants = [original_title]  # Include original
            
            # Template-based variants
            if keywords:
                top_keyword = keywords[0] if keywords else "content"
                
                variant_templates = [
                    f"🔥 {original_title}",
                    f"{original_title} - You Won't Believe This!",
                    f"The Truth About {top_keyword}: {original_title}",
                    f"AMAZING: {original_title}",
                    f"{original_title} (MUST WATCH)"
                ]
                
                variants.extend(variant_templates[:2])  # Add 2 variants
            
            return variants
            
        except Exception as e:
            self.logger.error(f"Failed to generate title variants: {e}")
            return [original_title]
    
    async def _track_ab_test_upload(self, upload_result: Dict[str, Any], variant: Dict[str, Any]):
        """Track A/B test data for uploaded video"""
        try:
            if not upload_result.get('success'):
                return
            
            # Schedule A/B test data collection
            await global_task_queue.add_task(
                task_type='ab_test',
                task_name=f"Track A/B test for {upload_result.get('video_id')}",
                task_data={
                    'video_id': upload_result.get('video_id'),
                    'variant': variant,
                    'upload_time': datetime.now().isoformat()
                },
                priority=TaskPriority.NORMAL
            )
            
        except Exception as e:
            self.logger.error(f"Failed to track A/B test upload: {e}")
    
    async def _process_video_generation_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process video generation task"""
        try:
            content_url = task_data.get('content_url')
            
            # Add content to pipeline
            await self.content_pipeline.add_content({
                'url': content_url,
                'title': task_data.get('title', ''),
                'description': task_data.get('description', ''),
                'metadata': task_data.get('metadata', {})
            })
            
            return {'success': True, 'message': 'Content added to pipeline'}
            
        except Exception as e:
            self.logger.error(f"Video generation task failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _process_content_analysis_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process content analysis task"""
        try:
            # Perform advanced content analysis
            analysis_result = await self.content_analyzer.analyze_content(
                title=task_data.get('title', ''),
                description=task_data.get('description', ''),
                metadata=task_data.get('metadata', {})
            )
            
            self.performance_metrics['videos_analyzed'] += 1
            
            return {
                'success': True,
                'analysis': analysis_result.__dict__,
                'should_generate_video': analysis_result.score > 60
            }
            
        except Exception as e:
            self.logger.error(f"Content analysis task failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _process_ab_test_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process A/B test task"""
        try:
            video_id = task_data.get('video_id')
            variant = task_data.get('variant', {})
            
            # Simulate collecting A/B test metrics
            await asyncio.sleep(1)
            
            # Generate simulated metrics
            metrics = {
                'views': random.randint(100, 1000),
                'clicks': random.randint(10, 100),
                'watch_time': random.uniform(30, 180),
                'engagement_rate': random.uniform(0.05, 0.25),
                'conversion_rate': random.uniform(0.01, 0.10),
                'retention_rate': random.uniform(0.40, 0.80)
            }
            
            # Submit metrics to optimization engine
            if variant.get('test_id'):
                await self.optimization_engine.collect_test_data(
                    variant_id=f"{variant['test_id']}_variant",
                    video_id=video_id,
                    metrics=metrics
                )
            
            return {'success': True, 'metrics_collected': True}
            
        except Exception as e:
            self.logger.error(f"A/B test task failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @RobustRetryHandler.with_retry(max_attempts=3, base_delay=5.0)
    async def find_and_analyze_content(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """Find and analyze content with robust retry handling"""
        try:
            self.logger.info(f"🔍 Finding and analyzing content (max: {max_items})")
            
            analyzed_content = []
            
            if self.reddit_client and self.reddit_client.is_connected():
                # Get content from Reddit
                reddit_posts = await self.reddit_client.get_filtered_video_posts(
                    max_posts=max_items
                )
                
                # Analyze content in parallel
                analysis_tasks = []
                for post in reddit_posts:
                    content_data = {
                        'title': post.title,
                        'description': post.description if hasattr(post, 'description') else '',
                        'url': post.url,
                        'metadata': {
                            'subreddit': post.subreddit,
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'created_utc': getattr(post, 'created_utc', None)
                        }
                    }
                    
                    # Submit to parallel content analyzer
                    task_id = await global_parallel_manager.submit_work(
                        WorkerType.CONTENT_ANALYZER,
                        {'content': content_data},
                        priority=1
                    )
                    analysis_tasks.append((task_id, content_data))
                
                # Collect results
                for task_id, content_data in analysis_tasks:
                    result = await global_parallel_manager.get_result(
                        task_id, WorkerType.CONTENT_ANALYZER, timeout=30.0
                    )
                    
                    if result and result.get('success'):
                        analyzed_content.append({
                            'content': content_data,
                            'analysis': result['analysis']
                        })
            else:
                # Fallback: Generate sample content
                self.logger.info("🔄 Reddit not available, generating sample content")
                for i in range(min(max_items, 3)):
                    sample_content = {
                        'title': f"Sample Content {i+1}: Interesting Topic",
                        'description': f"This is sample content number {i+1} for testing purposes.",
                        'url': f"https://example.com/sample_{i+1}",
                        'metadata': {
                            'subreddit': 'sample',
                            'score': random.randint(100, 1000),
                            'num_comments': random.randint(10, 100)
                        }
                    }
                    
                    # Analyze sample content
                    analysis_result = await self.content_analyzer.analyze_content(
                        title=sample_content['title'],
                        description=sample_content['description'],
                        metadata=sample_content['metadata']
                    )
                    
                    analyzed_content.append({
                        'content': sample_content,
                        'analysis': analysis_result
                    })
            
            self.logger.info(f"✅ Analyzed {len(analyzed_content)} content items")
            return analyzed_content
            
        except Exception as e:
            self.logger.error(f"❌ Content finding and analysis failed: {e}")
            raise
    
    async def start_advanced_autonomous_mode(self):
        """Start the advanced autonomous mode with all new features"""
        try:
            self.logger.info("🚀 Starting Advanced Autonomous Mode")
            self.logger.info("🧠 Features: Content Analysis, Dynamic Templates, A/B Testing, Parallel Processing")
            
            # Initialize all systems
            await self.initialize_systems()
            
            daily_video_count = 0
            last_reset_date = datetime.now().date()
            
            while True:
                try:
                    # Reset daily counter if needed
                    current_date = datetime.now().date()
                    if current_date != last_reset_date:
                        daily_video_count = 0
                        last_reset_date = current_date
                        self.logger.info(f"📅 New day started: {current_date}")
                    
                    # Check if we should generate videos
                    if daily_video_count < self.max_videos_per_day:
                        # Find and analyze content
                        analyzed_content = await self.find_and_analyze_content(max_items=5)
                        
                        # Filter high-quality content
                        high_quality_content = [
                            item for item in analyzed_content
                            if item['analysis'].score > 70
                        ]
                        
                        if high_quality_content:
                            self.logger.info(f"📈 Found {len(high_quality_content)} high-quality content items")
                            
                            # Process content through pipeline
                            for content_item in high_quality_content[:2]:  # Process max 2 at a time
                                await self.content_pipeline.add_content(content_item['content'])
                                daily_video_count += 1
                                self.performance_metrics['videos_generated'] += 1
                                
                                if daily_video_count >= self.max_videos_per_day:
                                    break
                        
                        # Create automated A/B tests
                        await self.optimization_engine.schedule_automated_tests()
                        
                        # Log performance statistics
                        await self._log_performance_stats()
                    
                    # Wait before next check
                    await asyncio.sleep(self.video_check_interval)
                    
                except KeyboardInterrupt:
                    self.logger.info("⏹️ Received shutdown signal")
                    break
                except Exception as e:
                    self.logger.error(f"❌ Main loop error: {e}")
                    # Continue operation after error
                    await asyncio.sleep(60)  # Wait 1 minute before retrying
            
        except Exception as e:
            self.logger.error(f"🚨 Advanced autonomous mode failed: {e}")
            raise
        finally:
            await self._cleanup()
    
    async def _log_performance_stats(self):
        """Log comprehensive performance statistics"""
        try:
            # Get system stats
            parallel_stats = global_parallel_manager.get_performance_stats()
            worker_status = global_parallel_manager.get_worker_status()
            queue_status = await global_task_queue.get_queue_status()
            pipeline_status = self.content_pipeline.get_pipeline_status()
            
            # Calculate uptime
            uptime = datetime.now() - self.performance_metrics['start_time']
            
            self.logger.info("📊 ADVANCED SYSTEM PERFORMANCE STATS")
            self.logger.info(f"⏰ Uptime: {uptime.total_seconds() / 3600:.1f} hours")
            self.logger.info(f"🎥 Videos Analyzed: {self.performance_metrics['videos_analyzed']}")
            self.logger.info(f"📹 Videos Generated: {self.performance_metrics['videos_generated']}")
            self.logger.info(f"📤 Videos Uploaded: {self.performance_metrics['videos_uploaded']}")
            self.logger.info(f"🧪 A/B Tests Created: {self.performance_metrics['ab_tests_created']}")
            
            # Parallel processing stats
            self.logger.info(f"⚡ Parallel Processing: {parallel_stats.get('throughput_per_minute', 0):.1f} items/min")
            self.logger.info(f"🔧 Task Queue: {queue_status.get('queue_length', 0)} pending tasks")
            self.logger.info(f"🚀 Content Pipeline: {pipeline_status.get('content_queue_size', 0)} items queued")
            
            # Save stats to file
            stats_data = {
                'performance_metrics': self.performance_metrics,
                'parallel_stats': parallel_stats,
                'worker_status': worker_status,
                'queue_status': queue_status,
                'pipeline_status': pipeline_status,
                'timestamp': datetime.now().isoformat()
            }
            
            stats_file = Path("data/advanced_system_stats.json")
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(stats_file, 'w') as f:
                json.dump(stats_data, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.warning(f"Failed to log performance stats: {e}")
    
    async def _cleanup(self):
        """Cleanup all systems"""
        try:
            self.logger.info("🧹 Cleaning up advanced systems...")
            
            # Stop content pipeline
            self.content_pipeline.stop_pipeline()
            
            # Stop parallel processing
            await global_parallel_manager.shutdown()
            
            # Stop task queue
            global_task_queue.stop_processing()
            
            # Close Reddit client
            if self.reddit_client:
                await self.reddit_client.close()
            
            self.logger.info("✅ Advanced system cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'performance_metrics': self.performance_metrics,
            'parallel_processing': global_parallel_manager.get_performance_stats(),
            'worker_status': global_parallel_manager.get_worker_status(),
            'pipeline_status': self.content_pipeline.get_pipeline_status(),
            'config_status': {
                'max_videos_per_day': self.max_videos_per_day,
                'min_videos_per_day': self.min_videos_per_day,
                'video_check_interval': self.video_check_interval
            },
            'system_health': 'operational',
            'timestamp': datetime.now().isoformat()
        }


async def run_advanced_autonomous_mode():
    """Main entry point for advanced autonomous mode"""
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger = logging.getLogger(__name__)
        logger.info("🚀 Starting Advanced Autonomous YouTube Video Generator")
        
        # Create and start advanced generator
        generator = AdvancedAutonomousGenerator()
        await generator.start_advanced_autonomous_mode()
        
    except KeyboardInterrupt:
        logger.info("⏹️ Advanced autonomous mode stopped by user")
    except Exception as e:
        logger.error(f"🚨 Advanced autonomous mode failed: {e}")
        raise


async def start_autonomous_mode():
    """Simple entry point for autonomous mode (legacy compatibility)"""
    print("🚀 YouTube Video Generator - Autonomous Mode")
    print("=" * 50)
    print("🤖 Starting fully autonomous operation...")
    print("📊 No human input required")
    print("⏹️ Press Ctrl+C to stop")
    print("=" * 50)
    
    # Run the advanced autonomous mode
    await run_advanced_autonomous_mode()


def main():
    """Main entry point for autonomous mode (simple interface)"""
    try:
        # Setup basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Start autonomous mode
        asyncio.run(start_autonomous_mode())
        
    except KeyboardInterrupt:
        print("\n⏹️ Autonomous mode stopped by user")
        print("✅ Shutdown complete")
        sys.exit(0)
    except Exception as e:
        print(f"\n🚨 Critical error: {e}")
        print("❌ Autonomous mode failed to start")
        sys.exit(1)


if __name__ == "__main__":
    # Support both simple and advanced modes
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        main()
    else:
        asyncio.run(run_advanced_autonomous_mode())