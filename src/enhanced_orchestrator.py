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
from src.models import VideoAnalysisEnhanced, PerformanceMetrics
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
        
        # Check for required dependencies
        if not NUMPY_AVAILABLE:
            self.logger.warning("NumPy not available - some features will be limited")
        
        # Initialize core components with error handling
        try:
            self.ai_client = AIClient()
            self.youtube_client = YouTubeClient()
            self.video_processor = VideoProcessor()
            
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
        try:
            self.logger.info(f"Starting enhanced video processing for: {reddit_url}")
            
            # Log initial GPU memory state
            if self.gpu_manager:
                self.gpu_manager.log_memory_status("Enhanced Processing Start")
            
            # Step 1: Download and analyze video (base functionality)
            download_result = await self._download_and_analyze_video(reddit_url)
            if not download_result['success']:
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
            
            # Compile comprehensive results
            final_result = {
                'success': True,
                'reddit_url': reddit_url,
                'video_path': video_path,
                'output_path': processing_result['output_path'],
                'enhanced_analysis': enhanced_analysis,
                'thumbnail_results': thumbnail_results,
                'upload_result': upload_result,
                'processing_time': processing_result.get('processing_time', 0),
                'enhancements_applied': self._get_applied_enhancements(enhanced_analysis),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("✅ Enhanced video processing completed successfully")
            return final_result
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced video processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'reddit_url': reddit_url,
                'timestamp': datetime.now().isoformat()
            }
    
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
                base_analysis,
                include_cinematic_insights=True
            )
            
            return enhanced_analysis
            
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
            
            # Upload video
            upload_result = await self.youtube_client.upload_video(
                video_path, 
                title=analysis.get('suggested_title', 'Enhanced Video'),
                description=analysis.get('suggested_description', ''),
                tags=analysis.get('suggested_tags', [])
            )
            
            # Upload thumbnails if available
            if thumbnails and upload_result['success']:
                for thumbnail in thumbnails:
                    await self.youtube_client.upload_thumbnail(
                        upload_result['video_id'], 
                        thumbnail['path']
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
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.gpu_manager:
                self.gpu_manager.cleanup()
            
            # Clean up other components
            for component in [self.cinematic_editor, self.advanced_audio_processor, 
                            self.enhanced_thumbnail_generator, self.enhancement_optimizer]:
                if component and hasattr(component, 'cleanup'):
                    try:
                        await component.cleanup()
                    except:
                        pass
            
            self.logger.info("Enhanced orchestrator cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        try:
            metrics = PerformanceMetrics(
                total_videos_processed=0,  # TODO: Implement tracking
                average_processing_time=0.0,
                success_rate=0.0,
                enhancement_usage={},
                timestamp=datetime.now()
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return PerformanceMetrics(
                total_videos_processed=0,
                average_processing_time=0.0,
                success_rate=0.0,
                enhancement_usage={},
                timestamp=datetime.now()
            )