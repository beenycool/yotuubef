"""
Enhanced AI-Powered YouTube Shorts Generator
Main entry point for the enhanced system with cinematic editing, advanced audio processing,
thumbnail A/B testing, and proactive channel management.
"""

import asyncio
import logging
import argparse
import json
import sys
import atexit
import signal
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.enhanced_orchestrator import EnhancedVideoOrchestrator
from src.integrations.narrative_analyzer import NarrativeAnalyzer
from src.management.channel_manager import ChannelManager
from src.processing.enhancement_optimizer import EnhancementOptimizer
from src.integrations.reddit_client import RedditClient
from src.analytics.analytics_advisor import AnalyticsAdvisor
from src.config.settings import get_config, setup_logging
from src.utils.cleanup import clear_temp_files, clear_results, clear_logs
from src.integrations.spotify_downloader import SpotifyDownloader
from src.utils.safe_print import safe_print


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
        self.analytics_advisor = AnalyticsAdvisor()
        
        # Initialize music downloader
        self.spotify_downloader = SpotifyDownloader()
        self.downloaded_music_files = []
        
        # Initialize Reddit client for automatic video finding (will be initialized async)
        self.reddit_client = None
        
        # Initialize narrative analyzer for strategic content curation
        self.narrative_analyzer = NarrativeAnalyzer()
        
        # Register cleanup on exit
        atexit.register(self._cleanup_music_files)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Enhanced YouTube Generator initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        if self.reddit_client:
            await self.reddit_client.close()
        self._cleanup_music_files()
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        self.logger.info("Received interrupt signal, cleaning up...")
        self._cleanup_music_files()
        sys.exit(0)
    
    def _cleanup_music_files(self):
        """Delete downloaded music files"""
        try:
            if self.downloaded_music_files:
                self.logger.info(f"Cleaning up {len(self.downloaded_music_files)} downloaded music files...")
                for file_path in self.downloaded_music_files:
                    try:
                        if file_path.exists():
                            file_path.unlink()
                            self.logger.debug(f"Deleted: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete {file_path}: {e}")
                self.downloaded_music_files.clear()
                self.logger.info("Music cleanup complete")
        except Exception as e:
            self.logger.error(f"Error during music cleanup: {e}")
    
    def _safe_print(self, message: str):
        """Safely print Unicode messages, falling back to ASCII if needed"""
        try:
            print(message)
        except UnicodeEncodeError:
            # Fallback to ASCII-safe version
            safe_message = message.encode('ascii', 'replace').decode('ascii')
            print(safe_message)
    
    async def _download_top_music(self):
        """Download top 10 songs at startup"""
        try:
            self.logger.info("Downloading top 10 songs for video creation...")
            safe_print("🎵 Downloading top 10 popular songs...")
            
            # Download top charts
            downloaded_files = self.spotify_downloader.download_top_charts("US", 10)
            
            if downloaded_files:
                self.downloaded_music_files.extend(downloaded_files)
                self.logger.info(f"Downloaded {len(downloaded_files)} songs successfully")
                safe_print(f"✅ Downloaded {len(downloaded_files)} songs to music/ folder")
                
                # Show downloaded songs
                for i, file_path in enumerate(downloaded_files[:5], 1):  # Show first 5
                    safe_print(f"   {i}. {file_path.stem}")
                if len(downloaded_files) > 5:
                    safe_print(f"   ... and {len(downloaded_files) - 5} more")
            else:
                self.logger.warning("No songs were downloaded")
                safe_print(f"❌ No songs were downloaded - check your internet connection or install spotify_dl")
                safe_print(f"💡 You can manually add music files to the music/ folder")
                
                # Check if there are existing music files
                existing_music = list(self.config.paths.music_folder.glob("*.mp3"))
                if existing_music:
                    safe_print(f"✅ Found {len(existing_music)} existing music files in music/ folder")
                    for i, file_path in enumerate(existing_music[:5], 1):
                        safe_print(f"   {i}. {file_path.stem}")
                
        except Exception as e:
            self.logger.error(f"Failed to download music: {e}")
            safe_print(f"❌ Failed to download music: {e}")
            safe_print(f"💡 You can manually add music files to the music/ folder")
    def _print_found_videos_summary(self, result: dict):
        """Print summary of found videos"""
        if not result.get('success') or not result.get('posts'):
            return
            
        print("\nFound Videos Summary:")
        print("=" * 40)
        for i, post in enumerate(result['posts'], 1):
            duration_str = f" ({post.duration:.1f}s)" if post.duration else ""
            # Handle Unicode characters by encoding/decoding properly
            try:
                title = post.title[:60].encode('ascii', 'ignore').decode('ascii')
                if len(post.title) > 60:
                    title += "..."
                print(f"{i}. r/{post.subreddit} - {title}")
                print(f"   Score: {post.score} | Comments: {post.num_comments}{duration_str}")
                print(f"   URL: {post.url}")
            except UnicodeEncodeError:
                # Fallback: replace problematic characters
                safe_title = post.title[:60].encode('ascii', 'replace').decode('ascii')
                if len(post.title) > 60:
                    safe_title += "..."
                print(f"{i}. r/{post.subreddit} - {safe_title}")
                print(f"   Score: {post.score} | Comments: {post.num_comments}{duration_str}")
                print(f"   URL: {post.url}")
            print()
        print("=" * 40)
    
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
            
            # Apply analytics feedback to options
            if options is None:
                options = {}
            
            # Apply analytics feedback to processing options
            enhanced_options = self.analytics_advisor.apply_analytics_feedback_to_options(options)
            if enhanced_options != options:
                self.logger.info("Applied analytics feedback to video processing options")
                options = enhanced_options
            
            # Enhanced processing with all features
            result = await self.orchestrator.process_enhanced_video(reddit_url, options)
            
            # Track performance for future analytics feedback
            if result.get('success') and result.get('video_id'):
                try:
                    # Track basic performance data
                    performance_data = {
                        'processed_at': datetime.now().isoformat(),
                        'reddit_url': reddit_url,
                        'processing_options': options,
                        'success': True
                    }
                    
                    # Add any available metrics
                    if result.get('performance_prediction'):
                        performance_data['predicted_metrics'] = result['performance_prediction']
                    
                    self.analytics_advisor.track_video_performance_feedback(
                        result['video_id'], performance_data
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Failed to track video performance: {e}")
            
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
    
    async def find_and_process_videos(self,
                                     max_videos: int = 5,
                                     subreddit_names: Optional[List[str]] = None,
                                     sort_method: str = 'hot',
                                     time_filter: str = 'day',
                                     dry_run: bool = False,
                                     options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Find videos automatically from Reddit and process them
        
        Args:
            max_videos: Maximum number of videos to find and process
            subreddit_names: Specific subreddits to search (uses config default if None)
            sort_method: Sorting method ('hot', 'top', 'new', 'rising')
            time_filter: Time filter for top posts ('hour', 'day', 'week', 'month')
            dry_run: If True, find videos but don't process them
            options: Processing options
            
        Returns:
            Dict with processing results and statistics
        """
        try:
            self.logger.info(f"Finding videos from Reddit (max: {max_videos}, sort: {sort_method})")
            
            # Apply analytics feedback to options
            if options is None:
                options = {}
            
            # Get analytics-driven content recommendations
            content_recommendations = self.analytics_advisor.get_content_recommendations_for_finder()
            if content_recommendations:
                self.logger.info("Applying analytics insights to video selection...")
                
                # Use preferred subreddits if none specified
                if not subreddit_names and content_recommendations.get('preferred_subreddits'):
                    subreddit_names = content_recommendations['preferred_subreddits'][:5]  # Top 5
                    self.logger.info(f"Using analytics-recommended subreddits: {subreddit_names}")
            
            # Apply analytics feedback to processing options
            enhanced_options = self.analytics_advisor.apply_analytics_feedback_to_options(options)
            if enhanced_options != options:
                self.logger.info("Applied analytics feedback to video processing options")
                options = enhanced_options
            
            # Initialize Reddit client if not already done
            if self.reddit_client is None:
                from src.integrations.reddit_client import create_reddit_client
                self.reddit_client = await create_reddit_client()
            
            # Check if Reddit client is connected
            if not self.reddit_client.is_connected():
                return {
                    'success': False,
                    'error': 'Reddit client not connected. Please check your Reddit API credentials.'
                }
            
            # Find suitable video posts from Reddit
            if sort_method == 'top':
                # For top posts, we need to fetch manually with time filter
                if subreddit_names:
                    all_posts = []
                    for subreddit in subreddit_names:
                        posts = await self.reddit_client.fetch_posts_from_subreddit(
                            subreddit,
                            sort_method='top',
                            time_filter=time_filter,
                            limit=max_videos * 2  # Get more to filter
                        )
                        all_posts.extend(posts)
                else:
                    # Use default subreddits from config
                    all_posts = await self.reddit_client.fetch_posts_from_multiple_subreddits(
                        posts_per_subreddit=max_videos
                    )
                
                # Filter for video posts and apply content filtering
                video_posts = [post for post in all_posts if post.is_video and post.video_url]
                suitable_posts = self.reddit_client.content_filter.filter_posts(video_posts)
                suitable_posts.sort(key=lambda p: p.score, reverse=True)
                reddit_posts = suitable_posts[:max_videos]
            else:
                # Use the built-in filtered method for other sort methods
                reddit_posts = await self.reddit_client.get_filtered_video_posts(
                    subreddit_names=subreddit_names,
                    max_posts=max_videos
                )
            
            if not reddit_posts:
                return {
                    'success': False,
                    'error': 'No suitable video posts found on Reddit',
                    'posts_found': 0
                }
            
            self.logger.info(f"Found {len(reddit_posts)} suitable video posts")
            
            # Print found videos summary
            # Create result dict for summary
            result_summary = {'success': True, 'posts': reddit_posts}
            self._print_found_videos_summary(result_summary)
            
            if dry_run:
                return {
                    'success': True,
                    'dry_run': True,
                    'posts_found': len(reddit_posts),
                    'found_posts': [
                        {
                            'title': post.title,
                            'url': post.url,
                            'subreddit': post.subreddit,
                            'score': post.score,
                            'duration': post.duration
                        } for post in reddit_posts
                    ]
                }
            
            # Convert Reddit posts to proper submission URLs for processing
            reddit_urls = [post.reddit_url for post in reddit_posts]
            
            # Process the found videos using batch processing
            result = await self.process_batch_videos(reddit_urls, options)
            
            # Add Reddit discovery information to results
            if result.get('success'):
                result['reddit_discovery'] = {
                    'posts_found': len(reddit_posts),
                    'sort_method': sort_method,
                    'time_filter': time_filter if sort_method == 'top' else None,
                    'subreddits_searched': subreddit_names or 'default_curated_list',
                    'found_posts_details': [
                        {
                            'title': post.title,
                            'subreddit': post.subreddit,
                            'score': post.score,
                            'url': post.url
                        } for post in reddit_posts
                    ]
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Auto video finding failed: {e}")
            return {'success': False, 'error': str(e)}
    async def find_and_process_narrative_driven_videos(self,
                                                          max_videos: int = 5,
                                                          subreddit_names: Optional[List[str]] = None,
                                                          min_narrative_score: int = 60,
                                                          sort_method: str = 'hot',
                                                          time_filter: str = 'day',
                                                          dry_run: bool = False,
                                                          options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Find and process videos using the strategic narrative gap approach.
        
        This method implements the complete strategic workflow:
        1. Discovers content with narrative gaps (missing context, unexplained reactions)
        2. Analyzes storytelling potential using AI
        3. Processes videos with narrative-optimized settings
        4. Creates content that transforms passive viewing into active engagement
        
        Args:
            max_videos: Maximum number of videos to find and process
            subreddit_names: Specific subreddits to search (uses config default if None)
            min_narrative_score: Minimum narrative potential score (1-100)
            sort_method: Sorting method ('hot', 'top', 'new', 'rising')
            time_filter: Time filter for top posts ('hour', 'day', 'week', 'month')
            dry_run: If True, find videos but don't process them
            options: Processing options
            
        Returns:
            Dict with processing results and narrative insights
        """
        try:
            self.logger.info(f"🎬 Starting narrative-driven content discovery...")
            self.logger.info(f"📊 Target: {max_videos} videos with min narrative score: {min_narrative_score}")
            
            # Apply analytics feedback to options
            if options is None:
                options = {}
            
            # Get analytics-driven recommendations
            content_recommendations = self.analytics_advisor.get_content_recommendations_for_finder()
            if content_recommendations:
                self.logger.info("Applying analytics insights to narrative video selection...")
                
                # Use preferred subreddits if none specified
                if not subreddit_names and content_recommendations.get('preferred_subreddits'):
                    subreddit_names = content_recommendations['preferred_subreddits'][:5]
                    self.logger.info(f"Using analytics-recommended subreddits: {subreddit_names}")
            
            # Enhanced options with analytics feedback
            enhanced_options = self.analytics_advisor.apply_analytics_feedback_to_options(options)
            if enhanced_options != options:
                self.logger.info("Applied analytics feedback to narrative processing options")
                options = enhanced_options
            
            # Initialize Reddit client if needed
            if self.reddit_client is None:
                from src.integrations.reddit_client import create_reddit_client
                self.reddit_client = await create_reddit_client()
            
            # Check connection
            if not self.reddit_client.is_connected():
                return {
                    'success': False,
                    'error': 'Reddit client not connected. Please check your Reddit API credentials.'
                }
            
            # Use narrative-driven discovery
            self.logger.info("🔍 Searching for content with strong narrative potential...")
            narrative_posts = await self.reddit_client.get_narrative_driven_video_posts(
                subreddit_names=subreddit_names,
                max_posts=max_videos,
                min_narrative_score=min_narrative_score
            )
            
            if not narrative_posts:
                return {
                    'success': False,
                    'error': f'No videos found with narrative score >= {min_narrative_score}',
                    'posts_found': 0
                }
            
            self.logger.info(f"✅ Found {len(narrative_posts)} narrative-rich videos")
            
            # Print narrative discovery summary
            self._print_narrative_discovery_summary(narrative_posts)
            
            if dry_run:
                return {
                    'success': True,
                    'dry_run': True,
                    'narrative_driven': True,
                    'posts_found': len(narrative_posts),
                    'narrative_insights': [
                        {
                            'title': post.title,
                            'narrative_score': analysis.narrative_potential_score,
                            'story_arc': analysis.story_arc,
                            'narrator_persona': analysis.narrator_persona,
                            'narrative_gaps': len(analysis.narrative_gaps),
                            'estimated_retention': analysis.estimated_retention
                        } for post, analysis in narrative_posts
                    ]
                }
            
            # Process narrative-driven videos
            self.logger.info("🎭 Starting narrative-driven video processing...")
            results = []
            total_start_time = datetime.now()
            
            for i, (post, narrative_analysis) in enumerate(narrative_posts):
                try:
                    self.logger.info(f"Processing narrative video {i+1}/{len(narrative_posts)}: {post.title[:50]}...")
                    self.logger.info(f"  📈 Narrative score: {narrative_analysis.narrative_potential_score}/100")
                    self.logger.info(f"  🎬 Story arc: {narrative_analysis.story_arc}")
                    self.logger.info(f"  🎙️ Narrator persona: {narrative_analysis.narrator_persona}")
                    
                    # Process with narrative-driven approach
                    result = await self.orchestrator.process_narrative_driven_video(
                        post, narrative_analysis, options
                    )
                    
                    # Track performance for analytics
                    if result.get('success') and result.get('video_id'):
                        try:
                            performance_data = {
                                'processed_at': datetime.now().isoformat(),
                                'narrative_driven': True,
                                'narrative_score': narrative_analysis.narrative_potential_score,
                                'story_arc': narrative_analysis.story_arc,
                                'narrator_persona': narrative_analysis.narrator_persona,
                                'processing_options': options,
                                'success': True
                            }
                            
                            if result.get('performance_prediction'):
                                performance_data['predicted_metrics'] = result['performance_prediction']
                            
                            self.analytics_advisor.track_video_performance_feedback(
                                result['video_id'], performance_data
                            )
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to track narrative video performance: {e}")
                    
                    results.append({
                        'post': post,
                        'narrative_analysis': narrative_analysis,
                        'result': result
                    })
                    
                    if result.get('success'):
                        self.logger.info(f"✅ Narrative video {i+1} processed successfully")
                    else:
                        self.logger.error(f"❌ Narrative video {i+1} failed: {result.get('error')}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to process narrative video {i+1}: {e}")
                    results.append({
                        'post': post,
                        'narrative_analysis': narrative_analysis,
                        'result': {'success': False, 'error': str(e)}
                    })
            
            total_processing_time = (datetime.now() - total_start_time).total_seconds()
            
            # Compile comprehensive results
            successful_videos = len([r for r in results if r['result'].get('success')])
            
            final_result = {
                'success': True,
                'narrative_driven': True,
                'narrative_discovery': {
                    'posts_found': len(narrative_posts),
                    'min_narrative_score': min_narrative_score,
                    'sort_method': sort_method,
                    'time_filter': time_filter if sort_method == 'top' else None,
                    'subreddits_searched': subreddit_names or 'default_curated_list'
                },
                'processing_summary': {
                    'total_videos': len(narrative_posts),
                    'successful_videos': successful_videos,
                    'failed_videos': len(narrative_posts) - successful_videos,
                    'total_processing_time_seconds': total_processing_time,
                    'average_time_per_video': total_processing_time / len(narrative_posts) if narrative_posts else 0
                },
                'narrative_insights': {
                    'average_narrative_score': sum(analysis.narrative_potential_score for _, analysis in narrative_posts) / len(narrative_posts),
                    'story_arcs_used': list(set(analysis.story_arc for _, analysis in narrative_posts)),
                    'narrator_personas_used': list(set(analysis.narrator_persona for _, analysis in narrative_posts)),
                    'total_narrative_gaps_leveraged': sum(len(analysis.narrative_gaps) for _, analysis in narrative_posts)
                },
                'individual_results': results
            }
            
            # Print final summary
            self._print_narrative_processing_summary(final_result)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Narrative-driven video discovery failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _print_narrative_discovery_summary(self, narrative_posts):
        """Print summary of discovered narrative-driven videos"""
        print("\n🎬 Narrative-Driven Content Discovery Results:")
        print("=" * 60)
        
        for i, (post, analysis) in enumerate(narrative_posts, 1):
            try:
                # Handle Unicode safely
                title = post.title[:50].encode('ascii', 'ignore').decode('ascii')
                if len(post.title) > 50:
                    title += "..."
                
                safe_print(f"\n{i}. 📺 r/{post.subreddit} - {title}")
                safe_print(f"   📊 Score: {post.score} | 💬 Comments: {post.num_comments}")
                safe_print(f"   🎭 Narrative Score: {analysis.narrative_potential_score}/100")
                print(f"   📖 Story Arc: {analysis.story_arc}")
                print(f"   🎙️ Narrator: {analysis.narrator_persona}")
                print(f"   🔍 Gaps: {len(analysis.narrative_gaps)} narrative opportunities")
                if analysis.narrative_gaps:
                    safe_print(f"   💡 Primary Gap: {analysis.narrative_gaps[0].description}")
                safe_print(f"   📈 Est. Retention: {analysis.estimated_retention}%")
                
            except UnicodeEncodeError:
                safe_title = post.title[:50].encode('ascii', 'replace').decode('ascii')
                if len(post.title) > 50:
                    safe_title += "..."
                safe_print(f"\n{i}. 📺 r/{post.subreddit} - {safe_title}")
                safe_print(f"   📊 Score: {post.score} | 💬 Comments: {post.num_comments}")
                safe_print(f"   🎭 Narrative Score: {analysis.narrative_potential_score}/100")
        
        print("\n" + "=" * 60)
    
    def _print_narrative_processing_summary(self, result: Dict[str, Any]):
        """Print summary of narrative-driven processing results"""
        if not result.get('success'):
            return
            
        print("\n🎬 Narrative-Driven Processing Complete!")
        print("=" * 50)
        
        summary = result['processing_summary']
        insights = result['narrative_insights']
        
        safe_print(f"📊 Processing Results:")
        safe_print(f"   ✅ Successful: {summary['successful_videos']}/{summary['total_videos']}")
        safe_print(f"   ⏱️ Total Time: {summary['total_processing_time_seconds']:.1f}s")
        safe_print(f"   📈 Avg Time/Video: {summary['average_time_per_video']:.1f}s")

        safe_print(f"\n🎭 Narrative Insights:")
        safe_print(f"   📊 Avg Narrative Score: {insights['average_narrative_score']:.1f}/100")
        safe_print(f"   📖 Story Arcs: {', '.join(insights['story_arcs_used'])}")
        print(f"   🎙️ Personas: {', '.join(insights['narrator_personas_used'])}")
        print(f"   🔍 Total Gaps Leveraged: {insights['total_narrative_gaps_leveraged']}")
        
        # Show successful results
        successful_results = [r for r in result['individual_results'] if r['result'].get('success')]
        if successful_results:
            print(f"\n✅ Successfully Processed Videos:")
            for i, video_result in enumerate(successful_results, 1):
                video_id = video_result['result'].get('video_id', 'N/A')
                analysis = video_result['narrative_analysis']
                print(f"   {i}. Score: {analysis.narrative_potential_score}/100 | Arc: {analysis.story_arc} | ID: {video_id}")
        
        print("=" * 50)
    
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
    
    async def generate_analytics_recommendations(self) -> Dict[str, Any]:
        """
        Generate analytics-based recommendations using Gemini AI
        
        Returns:
            Analytics recommendations and insights
        """
        try:
            self.logger.info("Generating analytics recommendations...")
            
            # Generate comprehensive recommendations
            recommendations = await self.analytics_advisor.generate_startup_recommendations()
            
            # Print recommendations to console
            if recommendations.get('success'):
                self.analytics_advisor.print_recommendations(recommendations)
            else:
                safe_print(f"\n❌ Analytics Error: {recommendations.get('error', 'Unknown error')}")
                safe_print(f"Message: {recommendations.get('message', 'Unable to generate recommendations')}\n")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Analytics recommendation generation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _print_processing_summary(self, result: Dict[str, Any]):
        """Print processing summary"""
        print("\n" + "="*60)
        print("ENHANCED VIDEO PROCESSING COMPLETE")
        print("="*60)
        
        if result.get('video_url'):
            print(f"YouTube URL: {result['video_url']}")
        
        # Cinematic enhancements
        cinematic = result.get('cinematic_enhancements', {})
        print(f"\nCinematic Enhancements:")
        print(f"   Camera movements: {cinematic.get('camera_movements', 0)}")
        print(f"   Dynamic focus points: {cinematic.get('dynamic_focus_points', 0)}")
        print(f"   Transitions: {cinematic.get('cinematic_transitions', 0)}")
        
        # Audio enhancements
        audio = result.get('audio_enhancements', {})
        print(f"\nAudio Enhancements:")
        print(f"   Advanced ducking: {'Yes' if audio.get('advanced_ducking_enabled') else 'No'}")
        print(f"   Smart detection: {'Yes' if audio.get('smart_detection_used') else 'No'}")
        print(f"   Voice enhancement: {'Yes' if audio.get('voice_enhancement_applied') else 'No'}")
        
        # Thumbnail optimization
        thumbnail = result.get('thumbnail_optimization', {})
        print(f"\nThumbnail Optimization:")
        print(f"   A/B testing: {'Enabled' if thumbnail.get('ab_testing_enabled') else 'Disabled'}")
        print(f"   Variants generated: {thumbnail.get('variants_generated', 0)}")
        
        # Performance prediction
        performance = result.get('performance_prediction', {})
        if performance:
            print(f"\nPerformance Prediction:")
            print(f"   Expected views: {performance.get('predicted_views', 'N/A')}")
            print(f"   Engagement rate: {performance.get('predicted_engagement_rate', 0):.1f}%")
            print(f"   Retention rate: {performance.get('predicted_retention_rate', 0):.1f}%")
            print(f"   Click-through rate: {performance.get('predicted_ctr', 0):.2f}%")
        
        # Processing stats
        processing_time = result.get('processing_time_seconds')
        if processing_time:
            print(f"\nProcessing Time: {processing_time:.1f} seconds")
        
        analysis_summary = result.get('analysis_summary', {})
        print(f"\nAI Analysis Summary:")
        print(f"   Total enhancements: {analysis_summary.get('total_enhancements', 0)}")
        print(f"   AI confidence: {analysis_summary.get('ai_confidence', 0):.1f}")
        print(f"   Complexity score: {analysis_summary.get('complexity_score', 0)}")
        
        print("="*60 + "\n")
    
    def _print_batch_summary(self, result: Dict[str, Any]):
        """Print batch processing summary"""
        batch_summary = result.get('batch_summary', {})
        
        print("\n" + "="*60)
        print("BATCH PROCESSING COMPLETE")
        print("="*60)
        
        print(f"Total videos: {batch_summary.get('total_videos', 0)}")
        print(f"Successful: {batch_summary.get('successful_videos', 0)}")
        print(f"FAILED: {batch_summary.get('failed_videos', 0)}")
        print(f"Total time: {batch_summary.get('total_processing_time_seconds', 0):.1f} seconds")
        print(f"Average per video: {batch_summary.get('average_time_per_video', 0):.1f} seconds")
        
        # List successful videos
        successful_results = [
            r for r in result.get('individual_results', []) 
            if r['result'].get('success')
        ]
        
        if successful_results:
            print(f"\nSuccessfully processed videos:")
            for i, res in enumerate(successful_results[:5], 1):  # Show first 5
                video_url = res['result'].get('video_url', 'N/A')
                print(f"   {i}. {video_url}")
            
            if len(successful_results) > 5:
                print(f"   ... and {len(successful_results) - 5} more")
        
        print("="*60 + "\n")
    
    def _print_optimization_summary(self, result: Dict[str, Any]):
        """Print optimization summary"""
        print("\n" + "="*50)
        print("SYSTEM OPTIMIZATION SUMMARY")
        print("="*50)
        
        if result.get('status') == 'completed':
            recommendations = result.get('recommendations', {})
            applied_changes = result.get('applied_changes', {})
            
            print(f"Analysis Summary:")
            analysis = result.get('analysis_summary', {})
            print(f"   Videos analyzed: {analysis.get('videos_analyzed', 0)}")
            print(f"   Analysis period: {analysis.get('analysis_period_days', 0)} days")
            
            print(f"\nRecommendations:")
            print(f"   Total generated: {recommendations.get('total_generated', 0)}")
            print(f"   High confidence: {recommendations.get('high_confidence', 0)}")
            print(f"   Average confidence: {recommendations.get('average_confidence', 0):.1f}")
            print(f"   Estimated impact: {recommendations.get('estimated_total_impact', 0):.1f}%")
            
            print(f"\nApplied Changes:")
            print(f"   Parameters modified: {applied_changes.get('total_applied', 0)}")
            if applied_changes.get('parameters_modified'):
                print(f"   Modified: {', '.join(applied_changes['parameters_modified'])}")
            print(f"   Estimated improvement: {applied_changes.get('estimated_impact', 0):.1f}%")
            
        elif result.get('status') == 'insufficient_data':
            print("Insufficient data for optimization")
            print(f"   Minimum required: {result.get('min_required', 0)} videos")
            
        elif result.get('status') == 'skipped':
            print("Optimization cycle not due")
            
        else:
            print(f"Optimization failed: {result.get('error', 'Unknown error')}")
        
        print("="*50 + "\n")
    
    def _print_system_status(self, status: Dict[str, Any]):
        """Print system status"""
        print("\n" + "="*50)
        print("ENHANCED SYSTEM STATUS")
        print("="*50)
        
        print(f"Status: {status.get('system_status', 'unknown').upper()}")
        print(f"Timestamp: {status.get('timestamp', 'N/A')}")
        
        # Component status
        components = status.get('components', {})
        print(f"\nComponents:")
        for component, state in components.items():
            status_text = "ACTIVE" if state == 'active' else "INACTIVE"
            print(f"   {status_text} {component.replace('_', ' ').title()}: {state}")
        
        # Resource status
        resources = status.get('resources', {})
        if resources.get('vram'):
            vram = resources['vram']
            print(f"\nGPU Resources:")
            print(f"   VRAM: {vram.get('used_gb', 0):.1f}GB / {vram.get('total_gb', 0):.1f}GB")
            print(f"   Usage: {vram.get('percent_used', 0):.1f}%")
        
        if resources.get('system_ram'):
            ram = resources['system_ram']
            print(f"\nSystem RAM:")
            print(f"   Used: {ram.get('used_gb', 0):.1f}GB / {ram.get('total_gb', 0):.1f}GB")
            print(f"   Usage: {ram.get('percent_used', 0):.1f}%")
        
        # Capabilities
        capabilities = status.get('capabilities', {})
        if capabilities.get('ai_features_available'):
            print(f"\nAI Features Available:")
            for feature in capabilities['ai_features_available']:
                print(f"   {feature.replace('_', ' ').title()}")
        
        print("="*50 + "\n")


async def main():
    """Main entry point with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Enhanced AI-Powered YouTube Shorts Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find and process videos automatically (DEFAULT)
  python main.py
  python main.py find --max-videos 3 --sort top --time-filter week
  
  # Find videos from specific subreddits
  python main.py find --subreddits funny videos gifs --max-videos 5
  
  # Dry run to see what videos would be found
  python main.py find --dry-run --max-videos 10
  
  # Process single video with all enhancements
  python main.py single "https://reddit.com/r/videos/comments/abc123"
  
  # Process multiple videos in batch
  python main.py batch urls.txt
  
  # Start proactive channel management
  python main.py manage
  
  # Run system optimization
  python main.py optimize --force
  
  # Check system status
  python main.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Find and process videos automatically (NEW DEFAULT COMMAND)
    find_parser = subparsers.add_parser('find', help='Find and process videos automatically from Reddit')
    find_parser.add_argument('--max-videos', type=int, default=5,
                            help='Maximum number of videos to find and process')
    find_parser.add_argument('--subreddits', nargs='+',
                            help='Specific subreddits to search (space-separated)')
    find_parser.add_argument('--sort', choices=['hot', 'top', 'new', 'rising'], default='hot',
                            help='Sorting method for finding videos')
    find_parser.add_argument('--time-filter', choices=['hour', 'day', 'week', 'month'], default='day',
                            help='Time filter for top posts')
    find_parser.add_argument('--dry-run', action='store_true',
                            help='Find and analyze videos but do not process or upload')
    find_parser.add_argument('--no-cinematic', action='store_true',
                            help='Disable cinematic editing')
    find_parser.add_argument('--no-audio-ducking', action='store_true',
                            help='Disable advanced audio ducking')
    find_parser.add_argument('--no-ab-testing', action='store_true',
                            help='Disable thumbnail A/B testing')
    
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
    
    # Analytics recommendations
    analytics_parser = subparsers.add_parser('analytics', help='Generate AI-powered analytics recommendations')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up temporary files, results, and logs')
    cleanup_parser.add_argument('--logs', action='store_true', help='Also clear log files')
    cleanup_parser.add_argument('--all', action='store_true', help='Clear all data (temp, results, logs)')
    
    args = parser.parse_args()
    
    if not args.command:
        # Default to 'find' command when no command is specified
        print("No command specified, defaulting to 'find' mode...")
        print("Use 'python main.py find --help' for more options\n")
        
        # Create a mock args object with default find parameters
        class MockArgs:
            def __init__(self):
                self.command = 'find'
                self.max_videos = 5
                self.subreddits = None
                self.sort = 'hot'
                self.time_filter = 'day'
                self.dry_run = False
                self.no_cinematic = False
                self.no_audio_ducking = False
                self.no_ab_testing = False
        
        args = MockArgs()
    
    # Initialize enhanced generator
    generator = EnhancedYouTubeGenerator()
    
    # Download top music on startup (unless it's cleanup command)
    if args.command != 'cleanup':
        await generator._download_top_music()
    
    # Generate analytics recommendations on startup (unless it's cleanup command)
    if args.command != 'cleanup':
        safe_print(f"🎯 Generating analytics recommendations...")
        await generator.generate_analytics_recommendations()
    
    try:
        if args.command == 'find':
            # Find and process videos automatically
            options = {
                'enable_cinematic_effects': not args.no_cinematic,
                'enable_advanced_audio_ducking': not args.no_audio_ducking,
                'enable_ab_testing': not args.no_ab_testing
            }
            
            result = await generator.find_and_process_videos(
                max_videos=args.max_videos,
                subreddit_names=args.subreddits,
                sort_method=args.sort,
                time_filter=args.time_filter,
                dry_run=args.dry_run,
                options=options
            )
            
            # Save result
            if result.get('success'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                result_file = Path(f"data/results/results_find_{timestamp}.json")
                result_file.parent.mkdir(parents=True, exist_ok=True)
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Results saved to: {result_file}")
            else:
                print(f"ERROR: Auto video finding failed: {result.get('error')}")
        
        elif args.command == 'single':
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
                result_file = Path(f"data/results/results_single_{timestamp}.json")
                result_file.parent.mkdir(parents=True, exist_ok=True)
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Results saved to: {result_file}")
        
        elif args.command == 'batch':
            # Process batch of videos
            urls_file = Path(args.file)
            if not urls_file.exists():
                safe_print(f"❌ File not found: {urls_file}")
                return
            
            with open(urls_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            if not urls:
                safe_print(f"❌ No URLs found in file")
                return
            
            options = {
                'max_concurrent_processing': args.max_concurrent
            }
            
            result = await generator.process_batch_videos(urls, options)
            
            # Save result
            if result.get('success'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                result_file = Path(f"data/results/results_batch_{timestamp}.json")
                result_file.parent.mkdir(parents=True, exist_ok=True)
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Results saved to: {result_file}")
        
        elif args.command == 'manage':
            # Start proactive management
            await generator.start_proactive_management()
        
        elif args.command == 'optimize':
            # Run optimization
            result = await generator.run_system_optimization(force=args.force)
            
            # Save result
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = Path(f"data/results/optimization_{timestamp}.json")
            result_file.parent.mkdir(parents=True, exist_ok=True)
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Optimization results saved to: {result_file}")
        
        elif args.command == 'status':
            # Check system status
            await generator.get_system_status()
        
        elif args.command == 'analytics':
            # Generate analytics recommendations
            result = await generator.generate_analytics_recommendations()
            
            # Save result
            if result.get('success'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                result_file = Path(f"data/results/analytics_{timestamp}.json")
                result_file.parent.mkdir(parents=True, exist_ok=True)
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"Analytics results saved to: {result_file}")
        
        elif args.command == 'cleanup':
            print("🧹 Starting cleanup process...")
            clear_temp_files()
            clear_results()
            if args.logs or args.all:
                clear_logs()
            if args.all:
                # Add any other all-encompassing cleanup here
                pass
            safe_print(f"✅ Cleanup process finished.")
    
    except KeyboardInterrupt:
        safe_print(f"\n⏹️ Operation interrupted by user")
        print("Cleaning up resources...")
        await generator.cleanup()
    except Exception as e:
        print(f"🚨 ERROR: Operation failed: {e}")
        logging.getLogger(__name__).exception("Main operation failed")
        await generator.cleanup()
    finally:
        # Ensure cleanup always runs
        try:
            await generator.cleanup()
        except:
            pass  # Ignore cleanup errors


def run_main():
    """Safe main entry point with proper async handling"""
    try:
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            safe_print(f"⚠️ Already in async context. Use 'await main()' instead.")
            return 1
        except RuntimeError:
            # No running loop - this is what we want
            pass
        
        # Run the main function
        return asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStartup interrupted by user")
        return 0
    except Exception as e:
        print(f"Critical startup error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(run_main())