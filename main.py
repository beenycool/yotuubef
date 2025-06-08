"""
Enhanced AI-Powered YouTube Shorts Generator
Main entry point for the enhanced system with cinematic editing, advanced audio processing,
thumbnail A/B testing, and proactive channel management.
"""

import asyncio
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.enhanced_orchestrator import EnhancedVideoOrchestrator
from src.management.channel_manager import ChannelManager
from src.processing.enhancement_optimizer import EnhancementOptimizer
from src.config.settings import get_config, setup_logging


class EnhancedYouTubeGenerator:
    """
    Enhanced YouTube Shorts generator with full AI-powered automation
    """
    
    def __init__(self):
        self.config = get_config()
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize enhanced orchestrator
        self.orchestrator = EnhancedVideoOrchestrator()
        
        # Initialize management systems
        self.channel_manager = ChannelManager()
        self.enhancement_optimizer = EnhancementOptimizer()
        
        self.logger.info("Enhanced YouTube Generator initialized")
    
    async def process_single_video(self, 
                                 reddit_url: str,
                                 options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a single video with all enhanced features
        
        Args:
            reddit_url: URL of Reddit video to process
            options: Processing options
            
        Returns:
            Processing results
        """
        try:
            self.logger.info(f"Processing single video: {reddit_url}")
            
            # Enhanced processing with all features
            result = await self.orchestrator.process_enhanced_video(reddit_url, options)
            
            # Log results
            if result.get('success'):
                video_id = result.get('video_id')
                self.logger.info(f"Video processed successfully. YouTube ID: {video_id}")
                
                # Print summary
                self._print_processing_summary(result)
            else:
                self.logger.error(f"Video processing failed: {result.get('error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Single video processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def process_batch_videos(self, 
                                 reddit_urls: List[str],
                                 options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process multiple videos in batch with optimization
        
        Args:
            reddit_urls: List of Reddit URLs to process
            options: Processing options
            
        Returns:
            Batch processing results
        """
        try:
            self.logger.info(f"Starting batch processing of {len(reddit_urls)} videos")
            
            # Run batch optimization
            result = await self.orchestrator.run_batch_optimization(reddit_urls, options)
            
            # Print batch summary
            if result.get('success'):
                self._print_batch_summary(result)
            else:
                self.logger.error(f"Batch processing failed: {result.get('error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def start_proactive_management(self):
        """
        Start proactive channel management in background
        """
        try:
            self.logger.info("Starting proactive channel management...")
            
            # Start channel management in background
            management_task = asyncio.create_task(
                self.channel_manager.run_proactive_management()
            )
            
            self.logger.info("Proactive management started. Press Ctrl+C to stop.")
            
            # Keep running until interrupted
            try:
                await management_task
            except KeyboardInterrupt:
                self.logger.info("Stopping proactive management...")
                management_task.cancel()
                try:
                    await management_task
                except asyncio.CancelledError:
                    pass
            
        except Exception as e:
            self.logger.error(f"Proactive management failed: {e}")
    
    async def run_system_optimization(self, force: bool = False) -> Dict[str, Any]:
        """
        Run system-wide optimization analysis
        
        Args:
            force: Force optimization even if not due
            
        Returns:
            Optimization results
        """
        try:
            self.logger.info("Running system optimization...")
            
            result = self.enhancement_optimizer.optimize_parameters(force_optimization=force)
            
            # Print optimization summary
            self._print_optimization_summary(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"System optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = await self.orchestrator.get_system_status()
            self._print_system_status(status)
            return status
            
        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _print_processing_summary(self, result: Dict[str, Any]):
        """Print processing summary"""
        print("\n" + "="*60)
        print("🎬 ENHANCED VIDEO PROCESSING COMPLETE")
        print("="*60)
        
        if result.get('video_url'):
            print(f"📺 YouTube URL: {result['video_url']}")
        
        # Cinematic enhancements
        cinematic = result.get('cinematic_enhancements', {})
        print(f"\n🎭 Cinematic Enhancements:")
        print(f"   📹 Camera movements: {cinematic.get('camera_movements', 0)}")
        print(f"   🎯 Dynamic focus points: {cinematic.get('dynamic_focus_points', 0)}")
        print(f"   ✨ Transitions: {cinematic.get('cinematic_transitions', 0)}")
        
        # Audio enhancements
        audio = result.get('audio_enhancements', {})
        print(f"\n🔊 Audio Enhancements:")
        print(f"   🎵 Advanced ducking: {'Yes' if audio.get('advanced_ducking_enabled') else 'No'}")
        print(f"   🧠 Smart detection: {'Yes' if audio.get('smart_detection_used') else 'No'}")
        print(f"   🎤 Voice enhancement: {'Yes' if audio.get('voice_enhancement_applied') else 'No'}")
        
        # Thumbnail optimization
        thumbnail = result.get('thumbnail_optimization', {})
        print(f"\n🖼️ Thumbnail Optimization:")
        print(f"   🧪 A/B testing: {'Enabled' if thumbnail.get('ab_testing_enabled') else 'Disabled'}")
        print(f"   📊 Variants generated: {thumbnail.get('variants_generated', 0)}")
        
        # Performance prediction
        performance = result.get('performance_prediction', {})
        if performance:
            print(f"\n📈 Performance Prediction:")
            print(f"   👀 Expected views: {performance.get('predicted_views', 'N/A')}")
            print(f"   💪 Engagement rate: {performance.get('predicted_engagement_rate', 0):.1f}%")
            print(f"   ⏱️ Retention rate: {performance.get('predicted_retention_rate', 0):.1f}%")
            print(f"   🎯 Click-through rate: {performance.get('predicted_ctr', 0):.2f}%")
        
        # Processing stats
        processing_time = result.get('processing_time_seconds')
        if processing_time:
            print(f"\n⏱️ Processing Time: {processing_time:.1f} seconds")
        
        analysis_summary = result.get('analysis_summary', {})
        print(f"\n🤖 AI Analysis Summary:")
        print(f"   🔧 Total enhancements: {analysis_summary.get('total_enhancements', 0)}")
        print(f"   🎯 AI confidence: {analysis_summary.get('ai_confidence', 0):.1f}")
        print(f"   📊 Complexity score: {analysis_summary.get('complexity_score', 0)}")
        
        print("="*60 + "\n")
    
    def _print_batch_summary(self, result: Dict[str, Any]):
        """Print batch processing summary"""
        batch_summary = result.get('batch_summary', {})
        
        print("\n" + "="*60)
        print("🎬 BATCH PROCESSING COMPLETE")
        print("="*60)
        
        print(f"📊 Total videos: {batch_summary.get('total_videos', 0)}")
        print(f"✅ Successful: {batch_summary.get('successful_videos', 0)}")
        print(f"❌ Failed: {batch_summary.get('failed_videos', 0)}")
        print(f"⏱️ Total time: {batch_summary.get('total_processing_time_seconds', 0):.1f} seconds")
        print(f"📈 Average per video: {batch_summary.get('average_time_per_video', 0):.1f} seconds")
        
        # List successful videos
        successful_results = [
            r for r in result.get('individual_results', []) 
            if r['result'].get('success')
        ]
        
        if successful_results:
            print(f"\n🎉 Successfully processed videos:")
            for i, res in enumerate(successful_results[:5], 1):  # Show first 5
                video_url = res['result'].get('video_url', 'N/A')
                print(f"   {i}. {video_url}")
            
            if len(successful_results) > 5:
                print(f"   ... and {len(successful_results) - 5} more")
        
        print("="*60 + "\n")
    
    def _print_optimization_summary(self, result: Dict[str, Any]):
        """Print optimization summary"""
        print("\n" + "="*50)
        print("🔧 SYSTEM OPTIMIZATION SUMMARY")
        print("="*50)
        
        if result.get('status') == 'completed':
            recommendations = result.get('recommendations', {})
            applied_changes = result.get('applied_changes', {})
            
            print(f"📊 Analysis Summary:")
            analysis = result.get('analysis_summary', {})
            print(f"   Videos analyzed: {analysis.get('videos_analyzed', 0)}")
            print(f"   Analysis period: {analysis.get('analysis_period_days', 0)} days")
            
            print(f"\n💡 Recommendations:")
            print(f"   Total generated: {recommendations.get('total_generated', 0)}")
            print(f"   High confidence: {recommendations.get('high_confidence', 0)}")
            print(f"   Average confidence: {recommendations.get('average_confidence', 0):.1f}")
            print(f"   Estimated impact: {recommendations.get('estimated_total_impact', 0):.1f}%")
            
            print(f"\n⚙️ Applied Changes:")
            print(f"   Parameters modified: {applied_changes.get('total_applied', 0)}")
            if applied_changes.get('parameters_modified'):
                print(f"   Modified: {', '.join(applied_changes['parameters_modified'])}")
            print(f"   Estimated improvement: {applied_changes.get('estimated_impact', 0):.1f}%")
            
        elif result.get('status') == 'insufficient_data':
            print("⚠️ Insufficient data for optimization")
            print(f"   Minimum required: {result.get('min_required', 0)} videos")
            
        elif result.get('status') == 'skipped':
            print("⏭️ Optimization cycle not due")
            
        else:
            print(f"❌ Optimization failed: {result.get('error', 'Unknown error')}")
        
        print("="*50 + "\n")
    
    def _print_system_status(self, status: Dict[str, Any]):
        """Print system status"""
        print("\n" + "="*50)
        print("🖥️ ENHANCED SYSTEM STATUS")
        print("="*50)
        
        print(f"Status: {status.get('system_status', 'unknown').upper()}")
        print(f"Timestamp: {status.get('timestamp', 'N/A')}")
        
        # Component status
        components = status.get('components', {})
        print(f"\n🔧 Components:")
        for component, state in components.items():
            emoji = "✅" if state == 'active' else "⏸️"
            print(f"   {emoji} {component.replace('_', ' ').title()}: {state}")
        
        # Resource status
        resources = status.get('resources', {})
        if resources.get('vram'):
            vram = resources['vram']
            print(f"\n💾 GPU Resources:")
            print(f"   VRAM: {vram.get('used_gb', 0):.1f}GB / {vram.get('total_gb', 0):.1f}GB")
            print(f"   Usage: {vram.get('percent_used', 0):.1f}%")
        
        if resources.get('system_ram'):
            ram = resources['system_ram']
            print(f"\n🖥️ System RAM:")
            print(f"   Used: {ram.get('used_gb', 0):.1f}GB / {ram.get('total_gb', 0):.1f}GB")
            print(f"   Usage: {ram.get('percent_used', 0):.1f}%")
        
        # Capabilities
        capabilities = status.get('capabilities', {})
        if capabilities.get('ai_features_available'):
            print(f"\n🤖 AI Features Available:")
            for feature in capabilities['ai_features_available']:
                print(f"   ✨ {feature.replace('_', ' ').title()}")
        
        print("="*50 + "\n")


async def main():
    """Main entry point with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Enhanced AI-Powered YouTube Shorts Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video with all enhancements
  python main_enhanced.py single "https://reddit.com/r/videos/comments/abc123"
  
  # Process multiple videos in batch
  python main_enhanced.py batch urls.txt
  
  # Start proactive channel management
  python main_enhanced.py manage
  
  # Run system optimization
  python main_enhanced.py optimize --force
  
  # Check system status
  python main_enhanced.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single video processing
    single_parser = subparsers.add_parser('single', help='Process single video')
    single_parser.add_argument('url', help='Reddit URL to process')
    single_parser.add_argument('--no-cinematic', action='store_true', 
                             help='Disable cinematic editing')
    single_parser.add_argument('--no-audio-ducking', action='store_true',
                             help='Disable advanced audio ducking')
    single_parser.add_argument('--no-ab-testing', action='store_true',
                             help='Disable thumbnail A/B testing')
    
    # Batch processing
    batch_parser = subparsers.add_parser('batch', help='Process multiple videos')
    batch_parser.add_argument('file', help='File containing Reddit URLs (one per line)')
    batch_parser.add_argument('--max-concurrent', type=int, default=3,
                            help='Maximum concurrent video processing')
    
    # Proactive management
    manage_parser = subparsers.add_parser('manage', help='Start proactive channel management')
    
    # System optimization
    optimize_parser = subparsers.add_parser('optimize', help='Run system optimization')
    optimize_parser.add_argument('--force', action='store_true',
                               help='Force optimization even if not due')
    
    # System status
    status_parser = subparsers.add_parser('status', help='Check system status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize enhanced generator
    generator = EnhancedYouTubeGenerator()
    
    try:
        if args.command == 'single':
            # Process single video
            options = {
                'enable_cinematic_effects': not args.no_cinematic,
                'enable_advanced_audio_ducking': not args.no_audio_ducking,
                'enable_ab_testing': not args.no_ab_testing
            }
            
            result = await generator.process_single_video(args.url, options)
            
            # Save result
            if result.get('success'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                result_file = Path(f"results_single_{timestamp}.json")
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"📄 Results saved to: {result_file}")
        
        elif args.command == 'batch':
            # Process batch of videos
            urls_file = Path(args.file)
            if not urls_file.exists():
                print(f"❌ File not found: {urls_file}")
                return
            
            with open(urls_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            if not urls:
                print("❌ No URLs found in file")
                return
            
            options = {
                'max_concurrent_processing': args.max_concurrent
            }
            
            result = await generator.process_batch_videos(urls, options)
            
            # Save result
            if result.get('success'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                result_file = Path(f"results_batch_{timestamp}.json")
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"📄 Results saved to: {result_file}")
        
        elif args.command == 'manage':
            # Start proactive management
            await generator.start_proactive_management()
        
        elif args.command == 'optimize':
            # Run optimization
            result = await generator.run_system_optimization(force=args.force)
            
            # Save result
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = Path(f"optimization_{timestamp}.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"📄 Optimization results saved to: {result_file}")
        
        elif args.command == 'status':
            # Check system status
            await generator.get_system_status()
    
    except KeyboardInterrupt:
        print("\n⏹️ Operation interrupted by user")
    except Exception as e:
        print(f"❌ Operation failed: {e}")
        logging.getLogger(__name__).exception("Main operation failed")


if __name__ == "__main__":
    asyncio.run(main())