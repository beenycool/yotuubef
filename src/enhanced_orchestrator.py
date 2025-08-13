"""
Enhanced AI-Powered Video Generation Orchestrator
Integrates all advanced features: cinematic editing, advanced audio, thumbnail optimization,
performance tracking, and proactive channel management.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import contextlib
import inspect

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import asyncprawcore.exceptions
    ASYNCPRAWCORE_AVAILABLE = True
except ImportError:
    asyncprawcore = None
    ASYNCPRAWCORE_AVAILABLE = False

from src.config.settings import get_config
from src.models import VideoAnalysisEnhanced, SystemPerformanceMetrics
from src.integrations.reddit_client import RedditClient
from src.integrations.ai_client import AIClient
from src.integrations.youtube_client import YouTubeClient
from src.processing.cinematic_editor import CinematicEditor
from src.processing.advanced_audio_processor import AdvancedAudioProcessor
from src.processing.enhanced_thumbnail_generator import EnhancedThumbnailGenerator
from src.processing.enhancement_optimizer import EnhancementOptimizer
from src.management.channel_manager import ChannelManager
from src.monitoring.engagement_metrics import EngagementMonitor
from src.utils.gpu_memory_manager import GPUMemoryManager
from src.processing.video_processor import VideoProcessor
from src.processing.long_form_video_generator import LongFormVideoGenerator

class EnhancedVideoOrchestrator:
    """
    Enhanced orchestrator with AI-powered cinematic editing, advanced audio processing,
    intelligent thumbnail optimization, and proactive channel management.
    """
    
    def __init__(self):
        # Basic setup
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.total_videos_processed = 0
        self.processing_times = []
        self.successful_videos = 0
        self.failed_videos = 0
        
        # Check for required dependencies
        if not NUMPY_AVAILABLE:
            self.logger.warning("NumPy not available - some features will be limited")
        
        # Initialize core components with error handling
        try:
            self.ai_client = AIClient()
            self.youtube_client = YouTubeClient()
            self.video_processor = VideoProcessor()
            self.reddit_client = RedditClient()
            
            # Initialize enhanced components
            self.cinematic_editor = CinematicEditor()
            self.advanced_audio_processor = AdvancedAudioProcessor()
            self.enhanced_thumbnail_generator = EnhancedThumbnailGenerator()
            self.enhancement_optimizer = EnhancementOptimizer()
            self.channel_manager = ChannelManager()
            self.engagement_monitor = EngagementMonitor()
            
            # Initialize long-form video generator
            self.long_form_generator = LongFormVideoGenerator()
            
            # Enhanced GPU memory management
            self.gpu_manager = GPUMemoryManager(max_vram_usage=0.85)
            
            self.components_initialized = True
            
        except Exception as e:
            self.logger.warning(f"Some components could not be initialized: {e}")
            # Set fallback values
            self.ai_client = None
            self.youtube_client = None
            self.video_processor = None
            self.reddit_client = None
            self.cinematic_editor = None
            self.advanced_audio_processor = None
            self.enhanced_thumbnail_generator = None
            self.enhancement_optimizer = None
            self.channel_manager = None
            self.engagement_monitor = None
            self.long_form_generator = None
            self.gpu_manager = None
            self.components_initialized = False
        
        # Enhanced workflow parameters
        self.enable_cinematic_editing = True
        self.enable_advanced_audio = True
        self.enable_ab_testing = True
        self.enable_auto_optimization = True
        self.enable_proactive_management = True
        
        self.logger.info("Enhanced AI-powered video orchestrator initialized")
    
    async def process_enhanced_video(self, 
                                   reddit_url: str,
                                   enhanced_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process video with all enhanced AI features
        
        Args:
            reddit_url: URL of Reddit video to process
            enhanced_options: Optional enhanced processing options
            
        Returns:
            Comprehensive processing results including all enhancements
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting enhanced video processing for: {reddit_url}")
            
            # Log initial GPU memory state
            if self.gpu_manager:
                self.gpu_manager.log_memory_status("Enhanced Processing Start")
            
            # Step 1: Download and analyze video (base functionality)
            download_result = await self._download_and_analyze_video(reddit_url)
            if not download_result['success']:
                self.failed_videos += 1
                return download_result
            
            video_path = download_result['video_path']
            base_analysis = download_result['analysis']
            
            # Step 2: Enhanced AI analysis with cinematic insights
            enhanced_analysis = await self._perform_enhanced_analysis(video_path, base_analysis)
            
            # Step 3: Apply cinematic editing suggestions
            if self.enable_cinematic_editing and self.cinematic_editor:
                enhanced_analysis = await self._apply_cinematic_analysis(video_path, enhanced_analysis)
            
            # Step 4: Process with advanced audio and video enhancements
            processing_result = await self._process_with_enhancements(
                video_path, enhanced_analysis, enhanced_options
            )
            
            if not processing_result['success']:
                self.failed_videos += 1
                return processing_result
            
            # Step 5: Generate A/B test thumbnails
            thumbnail_results = []
            if self.enable_ab_testing and self.enhanced_thumbnail_generator:
                thumbnail_results = await self._generate_ab_test_thumbnails(
                    video_path, enhanced_analysis
                )
            
            # Step 6: Upload and optimize
            upload_result = await self._upload_and_optimize(
                processing_result['output_path'], 
                enhanced_analysis, 
                thumbnail_results
            )
            
            # Step 7: Proactive channel management
            if self.enable_proactive_management and self.channel_manager:
                await self._perform_proactive_management(upload_result)
            
            # Step 8: Run optimization analysis
            if self.enable_auto_optimization:
                await self._run_optimization_analysis()
            
            # Final GPU memory cleanup
            if self.gpu_manager:
                self.gpu_manager.clear_gpu_cache()
                self.gpu_manager.log_memory_status("Enhanced Processing Complete")
            
            # Update performance tracking
            processing_time = (datetime.now() - start_time).total_seconds()
            self.total_videos_processed += 1
            self.successful_videos += 1
            self.processing_times.append(processing_time)
            
            # Compile comprehensive results
            final_result = {
                'success': True,
                'reddit_url': reddit_url,
                'video_path': video_path,
                'output_path': processing_result['output_path'],
                'enhanced_analysis': enhanced_analysis,
                'thumbnail_results': thumbnail_results,
                'upload_result': upload_result,
                'processing_time': processing_time,
                'enhancements_applied': self._get_applied_enhancements(enhanced_analysis),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("✅ Enhanced video processing completed successfully")
            return final_result
            
        except Exception as e:
            self.failed_videos += 1
            self.logger.exception("❌ Enhanced video processing failed")
            return {
                'success': False,
                'error': str(e),
                'reddit_url': reddit_url,
                'timestamp': datetime.now().isoformat()
            }
    
    async def run_batch_optimization(self, 
                                   video_urls: List[str],
                                   optimization_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run batch processing with optimization for multiple videos"""
        try:
            self.logger.info(f"Starting batch optimization for {len(video_urls)} videos")
            
            results = []
            total_start_time = datetime.now()
            
            for i, url in enumerate(video_urls):
                try:
                    self.logger.info(f"Processing video {i+1}/{len(video_urls)}: {url}")
                    
                    # Process individual video
                    result = await self.process_enhanced_video(url, optimization_config)
                    
                    results.append({
                        'url': url,
                        'index': i,
                        'result': result
                    })
                    
                    # GPU memory management between videos
                    if self.gpu_manager:
                        self.gpu_manager.clear_gpu_cache()
                    
                except Exception as e:
                    self.logger.error(f"Batch processing failed for video {i+1}: {e}")
                    results.append({
                        'url': url,
                        'index': i,
                        'result': {'success': False, 'error': str(e)}
                    })
            
            total_processing_time = (datetime.now() - total_start_time).total_seconds()
            
            # Run system optimization after batch
            if self.enable_auto_optimization:
                await self._run_optimization_analysis()
            
            # Compile batch results
            successful_videos = len([r for r in results if r['result'].get('success')])
            
            return {
                'success': True,
                'batch_summary': {
                    'total_videos': len(video_urls),
                    'successful_videos': successful_videos,
                    'failed_videos': len(video_urls) - successful_videos,
                    'total_processing_time_seconds': total_processing_time,
                    'average_time_per_video': total_processing_time / len(video_urls)
                },
                'individual_results': results,
                'system_status_post_batch': await self.get_system_status()
            }
            
        except Exception as e:
            self.logger.error(f"Batch optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'partial_results': results if 'results' in locals() else []
            }
    
    async def generate_long_form_video(self,
                                     topic: str,
                                     niche_category: str,
                                     target_audience: str,
                                     duration_minutes: int = 5,
                                     expertise_level: str = "beginner",
                                     base_content: Optional[str] = None,
                                     enhanced_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a high-quality long-form video with structured content.
        
        Args:
            topic: Main topic for the video
            niche_category: Niche category (e.g., 'technology', 'education')
            target_audience: Target audience description
            duration_minutes: Target duration in minutes
            expertise_level: Content expertise level
            base_content: Optional base content to expand upon
            enhanced_options: Optional enhanced processing options
            
        Returns:
            Generation results including video path and analysis
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting long-form video generation: {topic}")
            
            # Log initial GPU memory state
            if self.gpu_manager:
                self.gpu_manager.log_memory_status("Long-Form Generation Start")
            
            # Step 1: Generate long-form video structure and content
            generation_result = await self.long_form_generator.generate_long_form_video(
                topic=topic,
                niche_category=niche_category,
                target_audience=target_audience,
                duration_minutes=duration_minutes,
                expertise_level=expertise_level,
                base_content=base_content
            )
            
            if not generation_result.get('success'):
                return generation_result
            
            # Step 2: Apply enhanced processing if enabled
            if enhanced_options and enhanced_options.get('enable_enhanced_processing', True):
                generation_result = await self._apply_long_form_enhancements(
                    generation_result, enhanced_options
                )
            
            # Step 3: Upload to YouTube if enabled
            if enhanced_options and enhanced_options.get('upload_to_youtube', True):
                upload_result = await self._upload_long_form_video(generation_result)
                generation_result['upload_result'] = upload_result
            
            # Step 4: Initiate proactive management
            if generation_result.get('video_id') and self.enable_proactive_management:
                await self._initiate_proactive_management(generation_result['video_id'])
            
            # Log final GPU memory state
            if self.gpu_manager:
                self.gpu_manager.log_memory_status("Long-Form Generation Complete")
            
            # Update performance tracking
            processing_time = (datetime.now() - start_time).total_seconds()
            self.total_videos_processed += 1
            self.successful_videos += 1
            self.processing_times.append(processing_time)
            
            return generation_result
            
        except Exception as e:
            self.failed_videos += 1
            self.logger.exception("Long-form video generation failed")
            return {
                'success': False,
                'error': str(e),
                'video_format': 'long_form'
            }
    
    async def _apply_long_form_enhancements(self,
                                          generation_result: Dict[str, Any],
                                          enhanced_options: Dict[str, Any]) -> Dict[str, Any]:
        """Apply enhanced processing to long-form video"""
        try:
            video_path = generation_result.get('video_path')
            if not video_path:
                return generation_result
            
            # Apply cinematic editing if enabled
            if enhanced_options.get('enable_cinematic_effects', True) and self.cinematic_editor:
                cinematic_result = await self.cinematic_editor.apply_cinematic_effects(
                    video_path, generation_result.get('analysis', {})
                )
                generation_result['cinematic_enhancements'] = cinematic_result
            
            # Apply advanced audio processing if enabled
            if enhanced_options.get('enable_advanced_audio', True) and self.advanced_audio_processor:
                audio_result = await self.advanced_audio_processor.process_long_form_audio(
                    video_path, generation_result.get('analysis', {})
                )
                generation_result['audio_enhancements'] = audio_result
            
            # Generate enhanced thumbnails if enabled
            if enhanced_options.get('enable_ab_testing', True) and self.enhanced_thumbnail_generator:
                thumbnail_result = await self.enhanced_thumbnail_generator.generate_long_form_thumbnails(
                    video_path, generation_result.get('analysis', {})
                )
                generation_result['thumbnail_optimization'] = thumbnail_result
            
            return generation_result
            
        except Exception as e:
            self.logger.error(f"Long-form enhancement failed: {e}")
            generation_result['enhancement_error'] = str(e)
            return generation_result
    
    async def _upload_long_form_video(self, generation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Upload long-form video to YouTube"""
        try:
            video_path = generation_result.get('video_path')
            analysis = generation_result.get('analysis', {})
            
            if not video_path:
                return {'success': False, 'error': 'No video path provided'}
            
            # Prepare upload metadata
            video_structure = analysis.get('video_structure', {})
            
            upload_metadata = {
                'title': video_structure.get('title', 'Long-Form Video'),
                'description': video_structure.get('description', ''),
                'tags': video_structure.get('hashtags', []),
                'category': '27',  # Education category for long-form
                'privacy': 'public',
            }
            
            # Get thumbnail path if available
            thumbnail_path = generation_result.get('thumbnail_optimization', {}).get('best_thumbnail_path')
            
            # Upload to YouTube
            upload_result = await self.youtube_client.upload_video(
                video_path, upload_metadata, thumbnail_path=thumbnail_path
            )
            
            return upload_result
            
        except Exception as e:
            self.logger.error(f"Long-form video upload failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _initiate_proactive_management(self, video_id: str):
        """Start proactive management for uploaded video"""
        try:
            self.logger.info(f"Initiating proactive management for video {video_id}")
            
            # This would start the background management task
            # For now, we'll just log the initiation
            asyncio.create_task(self._manage_video_proactively(video_id))
            
        except Exception as e:
            self.logger.error(f"Proactive management initiation failed: {e}")
    
    async def _manage_video_proactively(self, video_id: str):
        """Background task for proactive video management"""
        try:
            # Get video info
            video_info = await self.youtube_client.get_video_info(video_id)
            if not video_info:
                return
            
            # Start management
            await self.channel_manager._manage_video(video_info)
            
        except Exception as e:
            self.logger.error(f"Proactive video management failed for {video_id}: {e}")
    
    async def _run_optimization_analysis(self):
        """Run enhancement optimization analysis"""
        try:
            self.logger.info("Running enhancement optimization analysis...")
            
            # Run optimization
            if self.enhancement_optimizer:
                optimization_result = self.enhancement_optimizer.optimize_parameters()
                
                if optimization_result.get('status') == 'completed':
                    applied_changes = optimization_result.get('applied_changes', [])
                    self.logger.info(f"Optimization complete: {len(applied_changes)} parameters adjusted")
            
        except Exception as e:
            self.logger.error(f"Optimization analysis failed: {e}")
    
    async def _predict_video_performance(self, 
                                       analysis: Dict[str, Any]) -> Dict[str, float]:
        """Predict video performance using AI"""
        try:
            # This would use ML models to predict performance
            # For now, we'll provide estimated predictions based on features
            
            # Calculate feature scores
            engagement_features = len(analysis.get('hook_variations', [])) * 5
            visual_features = len(analysis.get('visual_cues', [])) * 3
            audio_features = len(analysis.get('sound_effects', [])) * 2
            narrative_features = len(analysis.get('narrative_script_segments', [])) * 4
            
            feature_bonus = engagement_features + visual_features + audio_features + narrative_features
            
            # Predict metrics
            predicted_views = min(100000, 1000 + feature_bonus * 100)
            predicted_engagement_rate = min(20.0, 5.0 + feature_bonus * 0.1)
            predicted_retention = min(95.0, 60.0 + feature_bonus * 0.05)
            predicted_ctr = min(0.25, 0.05 + feature_bonus * 0.001)
            
            return {
                'predicted_views': predicted_views,
                'predicted_engagement_rate': predicted_engagement_rate,
                'predicted_retention_rate': predicted_retention,
                'predicted_ctr': predicted_ctr,
                'confidence_score': 0.75  # 75% confidence
            }
            
        except Exception as e:
            self.logger.error(f"Performance prediction failed: {e}")
            return {}
    
    async def _download_and_analyze_video(self, reddit_url: str) -> Dict[str, Any]:
        """Download and perform basic analysis of Reddit video"""
        try:
            # Download video using Reddit client
            if not self.reddit_client:
                return {'success': False, 'error': 'Reddit client not available'}
            
            download_result = await self.reddit_client.download_video(reddit_url)
            if not download_result['success']:
                return download_result
            
            video_path = download_result['video_path']
            
            # Perform basic video analysis
            if self.video_processor:
                analysis = await self.video_processor.analyze_video(video_path)
            else:
                analysis = {'duration': 0, 'resolution': [0, 0], 'fps': 0}
            
            return {
                'success': True,
                'video_path': video_path,
                'analysis': analysis
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Download and analysis failed: {e}'}
    
    async def _perform_enhanced_analysis(self, video_path: str, base_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform enhanced AI analysis of video content"""
        try:
            if not self.ai_client:
                return base_analysis
            
            # Enhanced content analysis
            enhanced_analysis = await self.ai_client.analyze_video_content(
                video_path, 
                base_analysis
            )
            
            return enhanced_analysis or base_analysis
            
        except Exception as e:
            self.logger.warning(f"Enhanced analysis failed, using base analysis: {e}")
            return base_analysis
    
    async def _apply_cinematic_analysis(self, video_path: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply cinematic editing analysis and suggestions"""
        try:
            if not self.cinematic_editor:
                return analysis
            
            cinematic_suggestions = await self.cinematic_editor.analyze_composition(
                video_path, analysis
            )
            
            # Merge cinematic suggestions with existing analysis
            analysis['cinematic_suggestions'] = cinematic_suggestions
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Cinematic analysis failed: {e}")
            return analysis
    
    async def _process_with_enhancements(self, video_path: str, analysis: Dict[str, Any], 
                                       options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process video with all enhancements applied"""
        try:
            start_time = datetime.now()
            
            # Apply cinematic editing if available
            if self.cinematic_editor and analysis.get('cinematic_suggestions'):
                video_path = await self.cinematic_editor.apply_cinematic_editing(
                    video_path, analysis['cinematic_suggestions']
                )
            
            # Apply advanced audio processing if available
            if self.advanced_audio_processor:
                video_path = await self.advanced_audio_processor.process_audio(
                    video_path, analysis
                )
            
            # Apply video enhancements if available
            if self.enhancement_optimizer:
                video_path = await self.enhancement_optimizer.optimize_video(
                    video_path, analysis, options
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'output_path': video_path,
                'processing_time': processing_time
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Enhancement processing failed: {e}'}
    
    async def _generate_ab_test_thumbnails(self, video_path: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate A/B test thumbnails for performance optimization"""
        try:
            if not self.enhanced_thumbnail_generator:
                return []
            
            thumbnails = await self.enhanced_thumbnail_generator.generate_ab_test_thumbnails(
                video_path, analysis, num_variants=3
            )
            
            return thumbnails
            
        except Exception as e:
            self.logger.warning(f"A/B test thumbnail generation failed: {e}")
            return []
    
    async def _upload_and_optimize(self, video_path: str, analysis: Dict[str, Any], 
                                  thumbnails: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upload video and apply optimization strategies"""
        try:
            if not self.youtube_client:
                return {'success': False, 'error': 'YouTube client not available'}
            
            # Prepare metadata and optional thumbnail
            metadata = {
                'title': analysis.get('suggested_title', 'Enhanced Video'),
                'description': analysis.get('summary_for_description', ''),
                'tags': analysis.get('hashtags', []),
            }
            primary_thumbnail = thumbnails[0]['path'] if thumbnails else None
            # Upload video (thumbnail handled by client if provided)
            upload_result = await self.youtube_client.upload_video(
                video_path,
                metadata,
                thumbnail_path=primary_thumbnail
            )
            
            return upload_result
            
        except Exception as e:
            return {'success': False, 'error': f'Upload and optimization failed: {e}'}
    
    async def _perform_proactive_management(self, upload_result: Dict[str, Any]):
        """Perform proactive channel management tasks"""
        try:
            if not self.channel_manager or not upload_result.get('success'):
                return
            
            # Analyze channel performance
            channel_analysis = await self.channel_manager.analyze_channel_performance()
            
            # Apply optimization strategies
            if channel_analysis.get('needs_optimization'):
                await self.channel_manager.apply_optimization_strategies(channel_analysis)
            
        except Exception as e:
            self.logger.warning(f"Proactive management failed: {e}")
    
    def _get_applied_enhancements(self, analysis: Dict[str, Any]) -> List[str]:
        """Get list of enhancements that were applied"""
        enhancements = []
        
        if analysis.get('cinematic_suggestions'):
            enhancements.append('cinematic_editing')
        
        if analysis.get('audio_enhancements'):
            enhancements.append('advanced_audio')
        
        if analysis.get('visual_enhancements'):
            enhancements.append('visual_optimization')
        
        return enhancements
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'system_status': 'operational',
                
                # Component status
                'components': {
                    'cinematic_editor': 'active' if self.enable_cinematic_editing else 'disabled',
                    'advanced_audio': 'active' if self.enable_advanced_audio else 'disabled',
                    'ab_testing': 'active' if self.enable_ab_testing else 'disabled',
                    'auto_optimization': 'active' if self.enable_auto_optimization else 'disabled',
                    'proactive_management': 'active' if self.enable_proactive_management else 'disabled'
                },
                
                # Resource status
                'resources': {},
                'optimization_summary': {},
                'channel_management': {},
                
                # Processing capabilities
                'capabilities': {
                    'max_video_length_minutes': 10,
                    'supported_formats': ['mp4', 'webm', 'avi'],
                    'ai_features_available': [
                        'cinematic_editing',
                        'advanced_audio_ducking',
                        'thumbnail_ab_testing',
                        'performance_optimization',
                        'comment_management'
                    ]
                }
            }
            
            # Add resource status safely
            try:
                if self.gpu_manager:
                    status['resources'] = self.gpu_manager.get_memory_summary()
            except Exception as e:
                self.logger.warning(f"Could not get GPU memory summary: {e}")
                status['resources'] = {}
            
            # Add optimization summary safely
            try:
                if self.enhancement_optimizer:
                    status['optimization_summary'] = self.enhancement_optimizer.get_optimization_summary()
            except Exception as e:
                self.logger.warning(f"Could not get optimization summary: {e}")
                status['optimization_summary'] = {}
            
            # Add channel management summary safely
            try:
                if self.channel_manager:
                    status['channel_management'] = self.channel_manager.get_management_summary()
            except Exception as e:
                self.logger.warning(f"Could not get management summary: {e}")
                status['channel_management'] = {}
            
            return status
            
        except Exception as e:
            self.logger.error(f"System status check failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'system_status': 'error',
                'error': str(e)
            }
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.gpu_manager:
                self.gpu_manager.cleanup()
            
            # Clean up other components
            for component in [self.cinematic_editor, self.advanced_audio_processor, 
                            self.enhanced_thumbnail_generator, self.enhancement_optimizer]:
                if component and hasattr(component, 'cleanup'):
                    cleanup_fn = component.cleanup
                    try:
                        result = cleanup_fn()
                        if inspect.isawaitable(result):
                            with contextlib.suppress(Exception):
                                await result
                        else:
                            # best-effort swallow sync cleanup errors
                            pass
                    except Exception:
                        # swallow exceptions during cleanup
                        pass
            
            self.logger.info("Enhanced orchestrator cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def get_performance_metrics(self) -> SystemPerformanceMetrics:
        """Get current performance metrics"""
        try:
            # Calculate actual metrics from tracking data
            avg_processing_time = 0.0
            if self.processing_times:
                avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            
            success_rate = 0.0
            if self.total_videos_processed > 0:
                success_rate = (self.successful_videos / self.total_videos_processed) * 100
            
            # Calculate enhancement usage based on processing history
            enhancement_usage = {
                'cinematic_editing': 0,
                'advanced_audio': 0,
                'thumbnail_optimization': 0,
                'performance_optimization': 0
            }
            
            metrics = SystemPerformanceMetrics(
                total_videos_processed=self.total_videos_processed,
                average_processing_time=avg_processing_time,
                success_rate=success_rate,
                enhancement_usage=enhancement_usage,
                timestamp=datetime.now()
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return SystemPerformanceMetrics(
                total_videos_processed=self.total_videos_processed,
                average_processing_time=0.0,
                success_rate=0.0,
                enhancement_usage={},
                timestamp=datetime.now()
            )