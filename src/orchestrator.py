"""
Main orchestrator for the YouTube video generation system.
Coordinates all components and manages the complete workflow.
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import gc

from src.config.settings import get_config, init_config
from src.integrations.reddit_client import RedditClient, RedditPost
from src.integrations.youtube_client import YouTubeClient, VideoMetadata
from src.integrations.ai_client import GeminiClient, VideoAnalysis
from src.processing.video_processor import VideoProcessor
from src.database.db_manager import DatabaseManager
from src.utils import (
    select_and_validate_segments,
    validate_file_paths,
    get_safe_filename,
    calculate_video_metrics,
    validate_analysis_completeness
)


class ProcessingError(Exception):
    """Custom exception for processing errors"""
    pass


class VideoGenerationOrchestrator:
    """Main orchestrator for the video generation pipeline"""
    
    def __init__(self, config_file: Optional[Path] = None):
        """Initialize the orchestrator with all required components"""
        # Initialize configuration
        self.config = init_config(config_file)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.reddit_client = RedditClient()
        self.youtube_client = YouTubeClient()
        self.ai_client = GeminiClient()
        self.video_processor = VideoProcessor()
        self.db_manager = DatabaseManager()
        
        # Processing statistics
        self.stats = {
            'videos_processed': 0,
            'videos_uploaded': 0,
            'processing_errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def run(self, 
            max_videos: int = 5,
            subreddit_names: Optional[List[str]] = None,
            dry_run: bool = False) -> Dict[str, Any]:
        """
        Run the complete video generation pipeline
        
        Args:
            max_videos: Maximum number of videos to process
            subreddit_names: Specific subreddits to target (uses config default if None)
            dry_run: If True, process videos but don't upload to YouTube
        
        Returns:
            Processing statistics and results
        """
        self.stats['start_time'] = datetime.now()
        self.logger.info("Starting video generation pipeline")
        
        try:
            # Validate components
            self._validate_components()
            
            # Fetch suitable Reddit posts
            reddit_posts = self._fetch_reddit_posts(subreddit_names, max_videos * 2)
            
            if not reddit_posts:
                self.logger.warning("No suitable Reddit posts found")
                return self._get_final_stats()
            
            # Process each post
            processed_videos = []
            for post in reddit_posts[:max_videos]:
                try:
                    result = self._process_single_video(post, dry_run)
                    if result:
                        processed_videos.append(result)
                        self.stats['videos_processed'] += 1
                        
                        if not dry_run and result.get('uploaded'):
                            self.stats['videos_uploaded'] += 1
                    
                    # Rate limiting between videos
                    time.sleep(self.config.api.api_delay_seconds)
                    
                except Exception as e:
                    self.logger.error(f"Error processing video from {post.url}: {e}")
                    self.stats['processing_errors'] += 1
                    
                    # Record failed upload
                    self.db_manager.record_upload(
                        reddit_url=post.url,
                        reddit_post_id=post.id,
                        title=post.title,
                        subreddit=post.subreddit,
                        original_score=post.score,
                        status='failed',
                        error_message=str(e)
                    )
                    continue
            
            # Update daily analytics
            self.db_manager.update_daily_analytics()
            
            self.logger.info(f"Pipeline completed. Processed {len(processed_videos)} videos.")
            return self._get_final_stats(processed_videos)
            
        except Exception as e:
            self.logger.error(f"Critical error in video generation pipeline: {e}")
            raise
        finally:
            self.stats['end_time'] = datetime.now()
            self._cleanup_resources()
    
    def _validate_components(self):
        """Validate that all required components are properly initialized"""
        validation_errors = []
        
        if not self.reddit_client.is_connected():
            validation_errors.append("Reddit client not properly connected")
        
        if not self.youtube_client.is_authenticated():
            validation_errors.append("YouTube client not authenticated")
        
        if not self.ai_client.is_available():
            self.logger.warning("AI client not available - will use fallback analysis")
        
        if validation_errors:
            error_msg = "Component validation failed: " + "; ".join(validation_errors)
            self.logger.error(error_msg)
            raise ProcessingError(error_msg)
        
        self.logger.info("All components validated successfully")
    
    def _fetch_reddit_posts(self, 
                            subreddit_names: Optional[List[str]], 
                            max_posts: int) -> List[RedditPost]:
        """Fetch and filter suitable Reddit posts"""
        self.logger.info("Fetching Reddit posts...")
        
        # Get posts from Reddit
        all_posts = self.reddit_client.get_filtered_video_posts(
            subreddit_names=subreddit_names,
            max_posts=max_posts
        )
        
        if not all_posts:
            return []
        
        # Filter out already processed posts
        new_posts = []
        for post in all_posts:
            if not self.db_manager.is_video_processed(post.url):
                new_posts.append(post)
            else:
                self.logger.debug(f"Skipping already processed post: {post.title[:50]}...")
        
        self.logger.info(f"Found {len(new_posts)} new posts out of {len(all_posts)} total")
        return new_posts
    
    def _process_single_video(self, post: RedditPost, dry_run: bool) -> Optional[Dict[str, Any]]:
        """
        Process a single video from Reddit post to YouTube upload
        
        Args:
            post: Reddit post to process
            dry_run: If True, don't actually upload to YouTube
        
        Returns:
            Processing result dictionary or None if failed
        """
        processing_start = datetime.now()
        self.logger.info(f"Processing video: {post.title[:50]}...")
        
        # Record initial upload entry
        upload_id = self.db_manager.record_upload(
            reddit_url=post.url,
            reddit_post_id=post.id,
            title=post.title,
            subreddit=post.subreddit,
            original_score=post.score,
            status='processing'
        )
        
        temp_files = []
        
        try:
            # Step 1: Download video
            step_start = datetime.now()
            download_path = self.config.paths.temp_dir / f"download_{post.id}.mp4"
            temp_files.append(download_path)
            
            if not self.video_processor.downloader.download_video(post.video_url, download_path):
                raise ProcessingError("Failed to download video")
            
            self.db_manager.record_processing_step(
                upload_id, "download", "completed", step_start, datetime.now()
            )
            
            # Step 2: AI Analysis
            step_start = datetime.now()
            analysis = self.ai_client.analyze_video_content(download_path, post)
            
            self.db_manager.record_processing_step(
                upload_id, "ai_analysis", "completed", step_start, datetime.now(),
                metadata={"fallback_used": analysis.fallback}
            )
            
            # Step 2.5: Validate Analysis and Segments
            step_start = datetime.now()
            
            # Check analysis completeness
            is_complete, missing_items = validate_analysis_completeness(analysis)
            if not is_complete:
                self.logger.warning(f"Analysis incomplete, missing: {missing_items}")
            
            # Calculate video metrics for resource planning
            video_metrics = calculate_video_metrics(analysis)
            self.logger.info(f"Video complexity score: {video_metrics['complexity_score']} "
                           f"(priority: {video_metrics['processing_priority']})")
            
            # Validate and select video segments
            valid_segments = select_and_validate_segments(analysis, self.config.model_dump())
            if not valid_segments:
                raise ProcessingError("No valid video segments could be determined")
            
            self.logger.info(f"Selected {len(valid_segments)} valid segments for processing")
            
            self.db_manager.record_processing_step(
                upload_id, "segment_validation", "completed", step_start, datetime.now(),
                metadata={
                    "segments_found": len(valid_segments),
                    "complexity_score": video_metrics['complexity_score'],
                    "analysis_complete": is_complete
                }
            )
            
            # Step 3: Video Processing
            step_start = datetime.now()
            processed_path = self.config.paths.temp_dir / f"processed_{post.id}.mp4"
            temp_files.append(processed_path)
            
            # Select background music based on mood
            background_music_path = self._select_background_music(analysis.mood)
            
            success = self.video_processor.process_video(
                download_path, processed_path, analysis, background_music_path
            )
            
            if not success:
                raise ProcessingError("Video processing failed")
            
            self.db_manager.record_processing_step(
                upload_id, "video_processing", "completed", step_start, datetime.now()
            )
            
            # Step 4: Generate thumbnail
            step_start = datetime.now()
            thumbnail_path = self._generate_thumbnail(processed_path, analysis)
            if thumbnail_path:
                temp_files.append(thumbnail_path)
            
            self.db_manager.record_processing_step(
                upload_id, "thumbnail_generation", "completed", step_start, datetime.now()
            )
            
            # Step 5: Upload to YouTube (unless dry run)
            youtube_result = None
            if not dry_run:
                step_start = datetime.now()
                youtube_result = self._upload_to_youtube(processed_path, analysis, post, thumbnail_path)
                
                self.db_manager.record_processing_step(
                    upload_id, "youtube_upload", 
                    "completed" if youtube_result.success else "failed",
                    step_start, datetime.now(),
                    error_message=youtube_result.error if not youtube_result.success else None
                )
            
            # Calculate final metrics
            processing_duration = (datetime.now() - processing_start).total_seconds()
            video_duration = self._get_video_duration(processed_path)
            file_size_mb = processed_path.stat().st_size / (1024 * 1024) if processed_path.exists() else None
            
            # Update upload record
            final_status = 'completed' if (dry_run or (youtube_result and youtube_result.success)) else 'failed'
            
            self.db_manager.update_upload_status(
                upload_id,
                status=final_status,
                youtube_url=youtube_result.video_url if youtube_result else None,
                youtube_video_id=youtube_result.video_id if youtube_result else None
            )
            
            # Update additional fields
            self.db_manager.record_upload(
                reddit_url=post.url,
                reddit_post_id=post.id,
                title=post.title,
                subreddit=post.subreddit,
                original_score=post.score,
                youtube_url=youtube_result.video_url if youtube_result else None,
                youtube_video_id=youtube_result.video_id if youtube_result else None,
                processing_duration=processing_duration,
                video_duration=video_duration,
                file_size_mb=file_size_mb,
                thumbnail_uploaded=youtube_result.thumbnail_uploaded if youtube_result else False,
                ai_analysis_used=not analysis.fallback,
                status=final_status
            )
            
            result = {
                'reddit_post': post,
                'analysis': analysis,
                'processed_video_path': processed_path,
                'processing_duration': processing_duration,
                'uploaded': youtube_result.success if youtube_result else False,
                'youtube_url': youtube_result.video_url if youtube_result else None,
                'upload_id': upload_id
            }
            
            if dry_run:
                self.logger.info(f"DRY RUN: Would have uploaded: {analysis.suggested_title}")
            else:
                self.logger.info(f"Successfully processed and uploaded: {analysis.suggested_title}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in video processing pipeline: {e}")
            
            # Update upload status to failed
            if upload_id:
                self.db_manager.update_upload_status(
                    upload_id, 'failed', error_message=str(e)
                )
            
            raise
        
        finally:
            # Cleanup temporary files
            self._cleanup_temp_files(temp_files)
    
    def _select_background_music(self, mood: str) -> Optional[Path]:
        """Select appropriate background music based on mood"""
        try:
            # Map moods to music categories
            mood_to_category = {
                'exciting': 'upbeat',
                'funny': 'funny',
                'dramatic': 'suspenseful',
                'educational': 'informative',
                'relaxing': 'relaxing',
                'emotional': 'emotional'
            }
            
            category = mood_to_category.get(mood.lower(), 'upbeat')
            
            # Look for music files in the category
            music_dir = self.config.paths.music_folder
            if music_dir.exists():
                music_files = list(music_dir.glob('*.mp3')) + list(music_dir.glob('*.wav'))
                
                # Try to find music matching the category
                for music_file in music_files:
                    if category.lower() in music_file.name.lower():
                        return music_file
                
                # Fallback to any music file
                if music_files:
                    return music_files[0]
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error selecting background music: {e}")
            return None
    
    def _generate_thumbnail(self, video_path: Path, analysis: VideoAnalysis) -> Optional[Path]:
        """Generate a custom thumbnail for the video"""
        try:
            import cv2
            from PIL import Image, ImageDraw, ImageFont
            
            # Extract frame at specified timestamp
            timestamp = analysis.thumbnail_info.get('timestamp_seconds', 0.0)
            
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
            
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Add text overlay if specified
            headline_text = analysis.thumbnail_info.get('headline_text', '')
            if headline_text:
                draw = ImageDraw.Draw(img)
                
                # Try to load custom font
                try:
                    font_path = self.config.get_font_path('BebasNeue-Regular.ttf')
                    font = ImageFont.truetype(font_path, 72)
                except:
                    font = ImageFont.load_default()
                
                # Calculate text position
                text_bbox = draw.textbbox((0, 0), headline_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                x = (img.width - text_width) // 2
                y = img.height - text_height - 50
                
                # Draw text with outline
                outline_width = 3
                for dx in range(-outline_width, outline_width + 1):
                    for dy in range(-outline_width, outline_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text((x + dx, y + dy), headline_text, font=font, fill='black')
                
                draw.text((x, y), headline_text, font=font, fill='white')
            
            # Save thumbnail
            thumbnail_path = self.config.paths.temp_dir / f"thumbnail_{int(time.time())}.jpg"
            img.save(thumbnail_path, 'JPEG', quality=95)
            
            return thumbnail_path
            
        except Exception as e:
            self.logger.warning(f"Error generating thumbnail: {e}")
            return None
    
    def _upload_to_youtube(self, 
                           video_path: Path, 
                           analysis: VideoAnalysis, 
                           post: RedditPost,
                           thumbnail_path: Optional[Path]) -> Any:
        """Upload video to YouTube with metadata"""
        try:
            # Prepare metadata
            metadata = VideoMetadata(
                title=analysis.suggested_title,
                description=self._create_video_description(analysis, post),
                tags=analysis.hashtags + [post.subreddit, "shorts"],
                category_id=self.config.api.youtube_upload_category_id,
                privacy_status=self.config.api.youtube_upload_privacy_status,
                thumbnail_path=thumbnail_path
            )
            
            # Upload video
            result = self.youtube_client.upload_video(video_path, metadata)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error uploading to YouTube: {e}")
            return type('UploadResult', (), {
                'success': False, 
                'error': str(e), 
                'video_url': None, 
                'video_id': None,
                'thumbnail_uploaded': False
            })()
    
    def _create_video_description(self, analysis: VideoAnalysis, post: RedditPost) -> str:
        """Create YouTube video description"""
        description_parts = [
            analysis.summary_for_description,
            "",
            f"Original post from r/{post.subreddit}",
            f"Credit to u/{post.author}",
            "",
            "🔔 Subscribe for more amazing content!",
            "👍 Like if you enjoyed this video!",
            "",
            f"Tags: {' '.join(['#' + tag.replace('#', '') for tag in analysis.hashtags[:5]])}"
        ]
        
        return "\n".join(description_parts)
    
    def _get_video_duration(self, video_path: Path) -> Optional[float]:
        """Get video duration in seconds"""
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            
            if fps > 0:
                return frame_count / fps
            return None
            
        except Exception as e:
            self.logger.warning(f"Error getting video duration: {e}")
            return None
    
    def _cleanup_temp_files(self, file_paths: List[Path]):
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                self.logger.warning(f"Error cleaning up temp file {file_path}: {e}")
    
    def _cleanup_resources(self):
        """Clean up resources and perform garbage collection"""
        gc.collect()
        self.logger.debug("Resource cleanup completed")
    
    def _get_final_stats(self, processed_videos: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Get final processing statistics"""
        if processed_videos is None:
            processed_videos = []
        
        total_time = None
        if self.stats['start_time'] and self.stats['end_time']:
            total_time = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        return {
            'total_processing_time_seconds': total_time,
            'videos_processed': self.stats['videos_processed'],
            'videos_uploaded': self.stats['videos_uploaded'],
            'processing_errors': self.stats['processing_errors'],
            'success_rate': (self.stats['videos_uploaded'] / self.stats['videos_processed'] * 100) if self.stats['videos_processed'] > 0 else 0,
            'processed_videos': processed_videos,
            'start_time': self.stats['start_time'],
            'end_time': self.stats['end_time']
        }


def main():
    """Main entry point for the video generation system"""
    import argparse
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='YouTube Shorts Video Generation from Reddit')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--max-videos', type=int, default=5, help='Maximum number of videos to process')
    parser.add_argument('--subreddits', nargs='+', help='Specific subreddits to target')
    parser.add_argument('--dry-run', action='store_true', help='Process videos but don\'t upload')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('video_generation.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize orchestrator
        config_file = Path(args.config) if args.config else None
        orchestrator = VideoGenerationOrchestrator(config_file)
        
        # Log configuration summary
        orchestrator.config.log_config_summary()
        
        # Run the pipeline
        results = orchestrator.run(
            max_videos=args.max_videos,
            subreddit_names=args.subreddits,
            dry_run=args.dry_run
        )
        
        # Log final results
        logger.info("=== Final Results ===")
        logger.info(f"Videos processed: {results['videos_processed']}")
        logger.info(f"Videos uploaded: {results['videos_uploaded']}")
        logger.info(f"Success rate: {results['success_rate']:.1f}%")
        logger.info(f"Total time: {results['total_processing_time_seconds']:.1f}s")
        logger.info("====================")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())