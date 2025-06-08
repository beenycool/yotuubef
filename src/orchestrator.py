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
from src.processing.thumbnail_generator import ThumbnailGenerator
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
        self.thumbnail_generator = ThumbnailGenerator()
        
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
                    results = self._process_single_video(post, dry_run)
                    if results:
                        processed_videos.append(results)
                        self.stats['videos_processed'] += 1
                        
                        if not dry_run and results.get('uploaded'):
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
            
            # Get actual video duration to determine appropriate minimum
            actual_duration = self._get_video_duration(download_path)
            if actual_duration is None:
                actual_duration = 60  # Default fallback
            
            # Calculate dynamic minimum duration based on actual video length
            if actual_duration < 15:
                # For very short videos, allow segments as short as 80% of total duration or 3s minimum
                min_duration = max(3.0, actual_duration * 0.8)
            elif actual_duration < 30:
                # For short videos, use 8 seconds minimum
                min_duration = 8.0
            else:
                # For longer videos, use original 15 second minimum
                min_duration = 15.0
            
            self.logger.info(f"Video duration: {actual_duration:.1f}s, using minimum segment duration: {min_duration:.1f}s")
            
            # Validate and select video segments
            valid_segments = select_and_validate_segments(analysis, {
                'video': {
                    'max_short_duration_seconds': self.config.video.target_duration,
                    'min_short_duration_seconds': min_duration
                }
            })
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
            
            # Step 3: Enhanced Video Processing with AI-Powered Features
            step_start = datetime.now()
            processed_path = self.config.paths.temp_dir / f"processed_{post.id}.mp4"
            temp_files.append(processed_path)
            
            # CRITICAL: Select background music based on AI-analyzed mood immediately
            background_music_path = self._select_background_music(analysis.mood)
            self.logger.info(f"Selected background music for mood '{analysis.mood}': {background_music_path.name if background_music_path else 'None'}")
            
            # Enhanced video processing with comprehensive audio processing
            processing_result = self.video_processor.process_video(
                download_path, processed_path, analysis, background_music_path, generate_thumbnail=True
            )
            
            if not processing_result['success']:
                raise ProcessingError("Enhanced video processing failed")
            
            # Extract results
            processed_video_path = processing_result['video_path']
            thumbnail_path = processing_result.get('thumbnail_path')
            processing_stats = processing_result.get('processing_stats', {})
            
            # Generate multiple thumbnail variants for A/B testing
            thumbnail_variants = []
            thumbnail_dir = self.config.paths.temp_dir / f"thumbnails_{post.id}"
            try:
                variants = self.thumbnail_generator.generate_multiple_variants(
                    processed_video_path, analysis, thumbnail_dir, variants=2
                )
                thumbnail_variants.extend(variants)
                temp_files.extend(variants)
                
                self.logger.info(f"Generated {len(thumbnail_variants)} thumbnail variants for A/B testing")
            except Exception as e:
                self.logger.warning(f"Failed to generate thumbnail variants: {e}")
                # Fallback to single thumbnail if available
                if thumbnail_path:
                    thumbnail_variants = [thumbnail_path]
            
            # Use first variant as primary thumbnail
            primary_thumbnail = thumbnail_variants[0] if thumbnail_variants else thumbnail_path
            
            if thumbnail_path:
                temp_files.append(thumbnail_path)
            
            self.db_manager.record_processing_step(
                upload_id, "video_processing", "completed", step_start, datetime.now(),
                metadata={
                    "tts_segments": processing_stats.get('tts_segments', 0),
                    "visual_effects": processing_stats.get('visual_effects', 0),
                    "cta_elements": processing_stats.get('cta_elements', 0),
                    "final_duration": processing_stats.get('duration', 0.0)
                }
            )
            
            # Step 5: Upload to YouTube (unless dry run)
            youtube_result = None
            if not dry_run:
                step_start = datetime.now()
                youtube_result = self._upload_to_youtube(processed_path, analysis, post, primary_thumbnail)
                
                # Start A/B thumbnail test if we have multiple variants
                if youtube_result.success and len(thumbnail_variants) >= 2:
                    self.db_manager.start_thumbnail_ab_test(upload_id)
                    self.logger.info(f"Started A/B thumbnail test for video {youtube_result.video_id}")
                
                self.db_manager.record_processing_step(
                    upload_id, "youtube_upload",
                    "completed" if youtube_result.success else "failed",
                    step_start, datetime.now(),
                    error_message=youtube_result.error if not youtube_result.success else None,
                    metadata={
                        "thumbnail_variants_generated": len(thumbnail_variants),
                        "ab_test_started": youtube_result.success and len(thumbnail_variants) >= 2
                    }
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
        """Select appropriate background music based on AI-analyzed mood with enhanced mapping"""
        try:
            # Enhanced mood to music category mapping based on config
            mood_to_category = {
                # High energy moods
                'exciting': 'upbeat',
                'intense': 'suspenseful',
                'dramatic': 'suspenseful',
                'uplifting': 'upbeat',
                'amazing': 'upbeat',
                
                # Emotional moods
                'funny': 'funny',
                'satisfying': 'relaxing',
                'educational': 'informative',
                'relaxing': 'relaxing',
                'emotional': 'emotional',
                'mysterious': 'suspenseful',
                
                # Fallback mappings
                'happy': 'upbeat',
                'positive': 'upbeat',
                'energetic': 'upbeat',
                'calm': 'relaxing',
                'peaceful': 'relaxing',
                'tense': 'suspenseful',
                'action': 'suspenseful'
            }
            
            # Get primary category
            primary_category = mood_to_category.get(mood.lower(), 'upbeat')
            
            # Look for music files in the category
            music_dir = self.config.paths.music_folder
            if not music_dir.exists():
                self.logger.warning(f"Music directory does not exist: {music_dir}")
                return None
                
            music_files = list(music_dir.glob('*.mp3')) + list(music_dir.glob('*.wav'))
            
            if not music_files:
                self.logger.warning("No music files found in music directory")
                return None
            
            self.logger.info(f"Selecting music for mood '{mood}' -> category '{primary_category}'")
            
            # Strategy 1: Try to find music matching the primary category
            category_matches = []
            for music_file in music_files:
                file_name_lower = music_file.name.lower()
                if primary_category.lower() in file_name_lower:
                    category_matches.append(music_file)
            
            if category_matches:
                selected = category_matches[0]  # Take first match
                self.logger.info(f"Selected music by category match: {selected.name}")
                return selected
            
            # Strategy 2: Try to find music matching related keywords from config
            if primary_category in self.config.audio.music_categories:
                related_keywords = self.config.audio.music_categories[primary_category]
                for keyword in related_keywords:
                    for music_file in music_files:
                        if keyword.lower() in music_file.name.lower():
                            self.logger.info(f"Selected music by keyword '{keyword}': {music_file.name}")
                            return music_file
            
            # Strategy 3: Smart fallback based on mood intensity
            if mood.lower() in ['intense', 'dramatic', 'exciting', 'amazing']:
                # Prefer files with energetic keywords
                energetic_keywords = ['funk', 'beat', 'energy', 'power', 'strong']
                for keyword in energetic_keywords:
                    for music_file in music_files:
                        if keyword.lower() in music_file.name.lower():
                            self.logger.info(f"Selected music by energetic fallback: {music_file.name}")
                            return music_file
            
            # Strategy 4: Final fallback - select first available file
            selected = music_files[0]
            self.logger.info(f"Selected music by final fallback: {selected.name}")
            return selected
            
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
        """Upload video to YouTube with enhanced metadata and CTAs"""
        try:
            # Import enhanced description function
            from src.integrations.youtube_client import create_enhanced_description
            
            # Create compelling description with strong CTAs
            enhanced_description = create_enhanced_description(
                summary=analysis.summary_for_description,
                call_to_action=analysis.call_to_action.text if analysis.call_to_action else "Subscribe for more amazing content!",
                hashtags=analysis.hashtags,
                channel_branding=getattr(self.config, 'channel_branding', None)
            )
            
            # Add source attribution
            source_attribution = f"\n\n📍 SOURCE:\nOriginal post from r/{post.subreddit}\nCredit to u/{post.author}"
            enhanced_description += source_attribution
            
            # Prepare enhanced metadata
            metadata = VideoMetadata(
                title=analysis.suggested_title,
                description=enhanced_description,
                tags=analysis.hashtags + [post.subreddit, "shorts", "viral", "reddit"],
                category_id=self.config.api.youtube_upload_category_id,
                privacy_status=self.config.api.youtube_upload_privacy_status,
                thumbnail_path=thumbnail_path,
                call_to_action=analysis.call_to_action.text if analysis.call_to_action else None
            )
            
            # Upload video with enhanced metadata
            result = self.youtube_client.upload_video(video_path, metadata)
            
            if result.success:
                self.logger.info(f"Successfully uploaded video with enhanced CTAs: {result.video_url}")
            
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
        
        success_rate = (self.stats['videos_uploaded'] / self.stats['videos_processed'] * 100 
                        if self.stats['videos_processed'] > 0 else 0.0)
        
        return {
            'processed_videos': processed_videos,
            'total_videos': len(processed_videos),
            'videos_processed': self.stats['videos_processed'],
            'videos_uploaded': self.stats['videos_uploaded'],
            'processing_errors': self.stats['processing_errors'],
            'success_rate': success_rate,
            'total_processing_time_seconds': total_time,
            'start_time': self.stats['start_time'],
            'end_time': self.stats['end_time']
        }
    
    def run_comment_pinning(self, max_comments: int = 50) -> Dict[str, Any]:
        """
        Analyze and pin engaging comments on recent videos with enhanced AI analysis.
        
        Args:
            max_comments: Maximum comments to analyze per video
            
        Returns:
            Dict with results of the operation including:
            - videos_processed: Number of videos analyzed
            - comments_analyzed: Total comments processed
            - comments_pinned: Comments successfully pinned
            - errors: Number of errors encountered
            - engagement_stats: Engagement metrics
        """
        results = {
            'videos_processed': 0,
            'comments_analyzed': 0,
            'comments_pinned': 0,
            'errors': 0,
            'engagement_stats': {
                'avg_likes': 0,
                'avg_replies': 0,
                'top_comment_score': 0
            }
        }
        
        try:
            # Get recent videos uploaded in the last 3 days
            recent_uploads = self.db_manager.get_recent_uploads(days=3)
            if not recent_uploads:
                self.logger.info("No recent uploads found for comment processing")
                return results
                
            total_likes = 0
            total_replies = 0
            top_scores = []
            
            for upload in recent_uploads:
                if not upload.get('youtube_video_id'):
                    continue
                    
                video_id = upload['youtube_video_id']
                self.logger.info(f"Processing comments for video: {video_id}")
                
                try:
                    # Get comments with engagement metrics
                    comments = self.youtube_client.get_video_comments(
                        video_id,
                        max_results=max_comments,
                        include_engagement=True
                    )
                    if not comments:
                        self.logger.info(f"No comments found for video {video_id}")
                        continue
                        
                    results['videos_processed'] += 1
                    results['comments_analyzed'] += len(comments)
                    
                    # Analyze comments using AI to find best one to pin
                    best_comment = None
                    best_score = 0
                    video_likes = 0
                    video_replies = 0
                    
                    for comment in comments:
                        # Calculate engagement metrics
                        video_likes += comment.get('like_count', 0)
                        video_replies += comment.get('reply_count', 0)
                        
                        # Score comment using AI analysis
                        score = self._score_comment_with_ai(comment, upload)
                        if score > best_score:
                            best_comment = comment
                            best_score = score
                    
                    # Update engagement stats
                    if comments:
                        total_likes += video_likes / len(comments)
                        total_replies += video_replies / len(comments)
                        top_scores.append(best_score)
                    
                    # Pin the best comment if it meets quality threshold
                    if best_comment and best_score >= 7.5:  # Higher quality threshold
                        try:
                            self.youtube_client.pin_comment(video_id, best_comment['id'])
                            
                            # Record pinned comment in database
                            self.db_manager.record_pinned_comment(
                                upload['id'],
                                best_comment['id'],
                                best_comment['text'],
                                best_score,
                                best_comment.get('like_count', 0),
                                best_comment.get('reply_count', 0)
                            )
                            
                            self.logger.info(f"Pinned comment for video {video_id} (score: {best_score:.1f}): {best_comment['text'][:50]}...")
                            results['comments_pinned'] += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to pin comment: {e}")
                            results['errors'] += 1
                            
                except Exception as e:
                    self.logger.error(f"Error processing comments for video {video_id}: {e}")
                    results['errors'] += 1
                    continue
            
            # Calculate aggregate engagement stats
            if results['videos_processed'] > 0:
                results['engagement_stats'] = {
                    'avg_likes': total_likes / results['videos_processed'],
                    'avg_replies': total_replies / results['videos_processed'],
                    'top_comment_score': sum(top_scores) / len(top_scores) if top_scores else 0
                }
            
            self.logger.info(f"Comment pinning completed. Results: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Critical error in comment pinning workflow: {e}", exc_info=True)
            results['errors'] += 1
            return results
    
    def run_ab_thumbnail_tests(self, test_duration_hours: int = 24) -> Dict[str, Any]:
        """
        Run comprehensive A/B thumbnail tests with statistical evaluation.
        
        Args:
            test_duration_hours: Minimum test duration before evaluation
            
        Returns:
            Dictionary with detailed test results including:
            - tests_evaluated: Total tests processed
            - tests_completed: Tests that reached conclusion
            - winning_variants: Count per variant (A/B)
            - avg_ctr_improvement: Average CTR improvement
            - errors: Number of errors encountered
        """
        self.logger.info("Starting A/B thumbnail test evaluation")
        
        results = {
            'tests_evaluated': 0,
            'tests_completed': 0,
            'winning_variants': {'A': 0, 'B': 0},
            'avg_ctr_improvement': 0.0,
            'errors': 0,
            'test_results': []
        }
        
        try:
            # Get videos ready for evaluation (minimum view threshold)
            pending_tests = self.db_manager.get_pending_ab_tests(
                min_duration_hours=test_duration_hours,
                min_views=1000  # Only evaluate tests with sufficient data
            )
            
            if not pending_tests:
                self.logger.info("No A/B tests ready for evaluation")
                return results
                
            ctr_improvements = []
            
            for upload in pending_tests:
                try:
                    upload_id = upload['id']
                    video_id = upload.get('youtube_video_id')
                    
                    if not video_id:
                        self.logger.warning(f"No YouTube video ID for upload {upload_id}")
                        continue
                    
                    results['tests_evaluated'] += 1
                    
                    # Get current performance metrics for both variants
                    ctr_a = upload.get('thumbnail_ctr_a')
                    ctr_b = upload.get('thumbnail_ctr_b')
                    views_a = upload.get('thumbnail_views_a', 0)
                    views_b = upload.get('thumbnail_views_b', 0)
                    
                    # If we don't have both CTRs, fetch current data
                    if ctr_a is None or ctr_b is None:
                        current_metrics = self.youtube_client.get_thumbnail_performance(video_id)
                        if current_metrics:
                            current_variant = upload.get('active_thumbnail', 'A')
                            if current_variant == 'A':
                                ctr_a = current_metrics['ctr']
                                views_a = current_metrics['views']
                            else:
                                ctr_b = current_metrics['ctr']
                                views_b = current_metrics['views']
                            
                            # Update database with latest metrics
                            self.db_manager.update_thumbnail_ctr(
                                upload_id,
                                current_variant,
                                ctr_a if current_variant == 'A' else ctr_b
                            )
                    
                    # Only evaluate if we have data for both variants
                    if ctr_a is not None and ctr_b is not None and views_a + views_b > 0:
                        # Calculate statistical significance
                        is_significant = self._is_ctr_difference_significant(
                            ctr_a, views_a,
                            ctr_b, views_b,
                            confidence_level=0.95
                        )
                        
                        if is_significant:
                            # Determine winner
                            winner = 'A' if ctr_a >= ctr_b else 'B'
                            improvement = (max(ctr_a, ctr_b) - min(ctr_a, ctr_b)) / min(ctr_a, ctr_b)
                            
                            # Update video with winning thumbnail
                            thumbnail_dir = self.config.paths.temp_dir / f"thumbnails_{upload['reddit_post_id']}"
                            winner_thumbnail = next(thumbnail_dir.glob(f"*_{winner}.*"), None)
                            
                            if winner_thumbnail:
                                if self.youtube_client.set_thumbnail(video_id, winner_thumbnail):
                                    self.db_manager.complete_ab_test(upload_id, winner)
                                    results['winning_variants'][winner] += 1
                                    results['tests_completed'] += 1
                                    ctr_improvements.append(improvement)
                                    
                                    test_result = {
                                        'video_id': video_id,
                                        'winner': winner,
                                        'ctr_a': ctr_a,
                                        'ctr_b': ctr_b,
                                        'improvement': improvement,
                                        'views_a': views_a,
                                        'views_b': views_b
                                    }
                                    results['test_results'].append(test_result)
                                    
                                    self.logger.info(
                                        f"Completed A/B test for video {video_id}. Winner: {winner} "
                                        f"(A: {ctr_a:.2%}, B: {ctr_b:.2%}, Improvement: {improvement:.2%})"
                                    )
                        else:
                            self.logger.info(
                                f"Inconclusive A/B test for video {video_id} "
                                f"(A: {ctr_a:.2%}, B: {ctr_b:.2%} - difference not significant)"
                            )
                    else:
                        self.logger.warning(f"Insufficient data for A/B test evaluation (video {video_id})")
                        
                except Exception as e:
                    self.logger.error(f"Error processing A/B test for upload {upload_id}: {e}", exc_info=True)
                    results['errors'] += 1
                    continue
            
            # Calculate average improvement for completed tests
            if ctr_improvements:
                results['avg_ctr_improvement'] = sum(ctr_improvements) / len(ctr_improvements)
            
            self.logger.info(
                f"A/B test evaluation completed. Results: {results['tests_completed']}/{results['tests_evaluated']} "
                f"tests concluded with average CTR improvement of {results['avg_ctr_improvement']:.2%}"
            )
            return results
            
        except Exception as e:
            self.logger.error(f"Critical error in A/B test evaluation: {e}", exc_info=True)
            results['errors'] += 1
            return results
        
        finally:
            self.logger.info(f"A/B test evaluation completed. Processed {results['tests_evaluated']} tests, "
                           f"completed {results['tests_completed']}, made {len([r for r in results['test_results']])} switches")


    def _score_comment_with_ai(self, comment: Dict[str, Any], upload: Dict[str, Any]) -> float:
        """
        Score a comment using AI analysis considering:
        - Engagement metrics (likes, replies)
        - Sentiment and relevance
        - Content quality and originality
        - Match with video topic
        
        Args:
            comment: Comment data including text and engagement metrics
            upload: Upload metadata for context
            
        Returns:
            Quality score from 0-10
        """
        try:
            # Base score from engagement metrics
            engagement_score = (
                comment.get('like_count', 0) * 0.2 +
                comment.get('reply_count', 0) * 0.3
            )
            
            # Get AI analysis of comment content
            analysis = self.ai_client.analyze_comment(
                comment['text'],
                video_title=upload.get('title', ''),
                video_description=upload.get('description', '')
            )
            
            # Calculate content quality score (0-5 scale)
            content_score = (
                analysis.get('sentiment_score', 0) * 2 +  # -1 to 1 -> 0 to 2
                min(analysis.get('relevance_score', 0), 1) * 2 +
                min(analysis.get('originality_score', 0), 1)
            )
            
            # Combine scores with weighting
            total_score = (
                engagement_score * 0.4 +
                content_score * 0.6
            )
            
            # Cap at 10 and round to 1 decimal
            return min(10, round(total_score * 2, 1))
            
        except Exception as e:
            self.logger.warning(f"Error scoring comment with AI: {e}")
            # Fallback to simple engagement scoring
            return min(10, comment.get('like_count', 0) * 0.2)

    def _is_ctr_difference_significant(self, ctr_a: float, views_a: int,
                                     ctr_b: float, views_b: int,
                                     confidence_level: float = 0.95) -> bool:
        """
        Check if CTR difference is statistically significant using chi-squared test
        with additional checks for practical significance.
        
        Args:
            ctr_a: CTR for variant A (0-1)
            views_a: Views for variant A
            ctr_b: CTR for variant B (0-1)
            views_b: Views for variant B
            confidence_level: Desired confidence level (0-1)
            
        Returns:
            True if difference is both statistically and practically significant
        """
        try:
            # Import scipy for statistical testing
            from scipy.stats import chi2_contingency
            
            # Minimum practical difference threshold (5% relative improvement)
            min_practical_diff = 0.05
            
            # Calculate absolute difference
            abs_diff = abs(ctr_a - ctr_b)
            rel_diff = abs_diff / min(ctr_a, ctr_b) if min(ctr_a, ctr_b) > 0 else 0
            
            # Check if difference is practically significant
            if rel_diff < min_practical_diff:
                return False
                
            # Check if we have enough data
            min_views = 1000
            if views_a < min_views or views_b < min_views:
                return False
                
            # Create contingency table
            clicks_a = int(round(ctr_a * views_a))
            clicks_b = int(round(ctr_b * views_b))
            
            table = [
                [clicks_a, views_a - clicks_a],
                [clicks_b, views_b - clicks_b]
            ]
            
            # Perform chi-squared test
            chi2, p, _, _ = chi2_contingency(table)
            
            # Check both statistical and practical significance
            return p < (1 - confidence_level) and rel_diff >= min_practical_diff
            
        except ImportError:
            self.logger.warning("scipy not available for statistical testing, using simple threshold")
            # Fallback: simple threshold-based comparison
            rel_diff = abs(ctr_a - ctr_b) / min(ctr_a, ctr_b) if min(ctr_a, ctr_b) > 0 else 0
            return rel_diff >= 0.1 and views_a >= 500 and views_b >= 500
            
        except Exception as e:
            self.logger.warning(f"Error in statistical significance test: {e}")
            return False


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
        logger.info(f"Processing errors: {results['processing_errors']}")
        logger.info(f"Success rate: {results['success_rate']:.1f}%")
        total_time = results.get('total_processing_time_seconds')
        if total_time is not None:
            logger.info(f"Total time: {total_time:.1f}s")
        else:
            logger.info("Total time: N/A")
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