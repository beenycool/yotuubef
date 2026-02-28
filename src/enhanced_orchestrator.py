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

from moviepy import AudioFileClip, CompositeAudioClip, concatenate_audioclips

import asyncprawcore.exceptions

from src.config.settings import get_config
from src.models import VideoAnalysis, VideoAnalysisEnhanced, PerformanceMetrics
from src.integrations.reddit_client import RedditClient
from src.integrations.search_client import DeepResearchClient, AgenticResearcher
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
from src.processing.background_manager import BackgroundManager
from src.processing.video_processor_fixes import MoviePyCompat
from src.processing.image_search_client import BraveImageClient
from src.processing.caption_generator import CaptionGenerator
from src.processing.sound_effects_manager import SoundEffectsManager


class EnhancedVideoOrchestrator:
    """
    Enhanced orchestrator with AI-powered cinematic editing, advanced audio processing,
    intelligent thumbnail optimization, and proactive channel management.
    """

    def __init__(self):
        # Basic setup
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        # Initialize core components
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

        # Enhanced GPU memory management
        self.gpu_manager = GPUMemoryManager(max_vram_usage=0.85)

        # Enhanced workflow parameters
        self.enable_cinematic_editing = True
        self.enable_advanced_audio = True
        self.enable_ab_testing = True
        self.enable_auto_optimization = True
        self.enable_proactive_management = True

        self.logger.info("Enhanced AI-powered video orchestrator initialized")

    async def process_enhanced_video(
        self, reddit_url: str, enhanced_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run faceless lore pipeline: text post -> research -> script -> TTS -> Minecraft bg."""
        try:
            self.logger.info("Starting faceless lore generation for: %s", reddit_url)

            async with RedditClient() as reddit_client:
                reddit_post = await reddit_client.get_post_by_url(reddit_url)

            if not reddit_post:
                return {"success": False, "error": "Failed to load Reddit post"}
            if reddit_post.is_video:
                return {
                    "success": False,
                    "error": "Provided URL points to a video post. Faceless flow requires text posts.",
                }
            if (
                not getattr(reddit_post, "selftext", "")
                or len(reddit_post.selftext.strip()) < 120
            ):
                return {
                    "success": False,
                    "error": "Text post is too short for lore generation.",
                }

            researcher = DeepResearchClient()
            query = f"{reddit_post.title} {reddit_post.subreddit} history"
            research_facts = await researcher.conduct_deep_research(query)

            reddit_content_dict = {
                "title": reddit_post.title,
                "selftext": reddit_post.selftext,
                "subreddit": reddit_post.subreddit,
                "score": reddit_post.score,
                "num_comments": reddit_post.num_comments,
                "deep_research": research_facts,
            }
            analysis = await self.ai_client.analyze_video_content(
                None, reddit_content_dict
            )
            if not analysis:
                return {"success": False, "error": "Script generation failed"}

            tts_results = (
                self.advanced_audio_processor.tts_service.generate_multiple_segments(
                    analysis.narrative_script_segments
                )
            )
            tts_paths = [
                item.get("audio_path")
                for item in tts_results
                if item.get("success") and item.get("audio_path")
            ]
            if not tts_paths:
                return {"success": False, "error": "TTS generation failed"}

            audio_segments = [AudioFileClip(str(path)) for path in tts_paths]
            audio_clip = concatenate_audioclips(audio_segments)

            bg_manager = BackgroundManager()
            video_clip = bg_manager.get_sliced_background(
                target_duration=audio_clip.duration
            )
            if analysis.text_overlays:
                video_clip = self.video_processor.text_processor.add_text_overlays(
                    video_clip, analysis.text_overlays
                )
            final_video = MoviePyCompat.with_audio(video_clip, audio_clip)

            output_file = self.config.paths.processed_dir / f"lore_{reddit_post.id}.mp4"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            final_video.write_videofile(
                str(output_file),
                fps=30,
                codec="libx264",
                audio_codec="aac",
            )

            upload_metadata = {
                "title": analysis.suggested_title,
                "description": analysis.summary_for_description,
                "tags": [tag.replace("#", "") for tag in analysis.hashtags],
            }
            upload_result = await self.youtube_client.upload_video(
                str(output_file), upload_metadata
            )

            for segment in audio_segments:
                try:
                    segment.close()
                except Exception:
                    pass
            try:
                audio_clip.close()
            except Exception:
                pass
            try:
                final_video.close()
            except Exception:
                pass

            if not upload_result.get("success"):
                return {
                    "success": False,
                    "error": upload_result.get("error", "Upload failed"),
                    "video_path": str(output_file),
                }

            return {
                "success": True,
                "video_id": upload_result.get("video_id"),
                "video_url": upload_result.get("video_url"),
                "video_path": str(output_file),
                "pipeline": "faceless_lore",
            }

        except Exception as e:
            self.logger.error("Lore generation failed: %s", e, exc_info=True)
            self.gpu_manager.clear_gpu_cache()
            return {"success": False, "error": str(e), "stage": "faceless_lore"}

    async def process_ai_production_studio(
        self, reddit_url: str, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        AI Production Studio Pipeline - Uses all 5 Improvements:
        1. Multi-Turn Agentic Research
        2. Perfect Loop Retention
        3. Dynamic B-Roll Injection
        4. Word-Level Dynamic Captions
        5. Sound Design Architecture
        """
        try:
            self.logger.info(
                "Starting AI Production Studio pipeline for: %s", reddit_url
            )
            options = options or {}

            # Step 1: Get Reddit post
            async with RedditClient() as reddit_client:
                reddit_post = await reddit_client.get_post_by_url(reddit_url)

            if not reddit_post:
                return {"success": False, "error": "Failed to load Reddit post"}
            if reddit_post.is_video:
                return {"success": False, "error": "Text posts only for this pipeline"}

            # Improvement 1: Multi-Turn Agentic Research
            self.logger.info("Step 1: Agentic Research (Improvement 1)")
            researcher = AgenticResearcher()
            query = f"{reddit_post.title} {reddit_post.subreddit} history"
            initial_context = f"{reddit_post.title}: {reddit_post.selftext[:500]}"

            research_facts = await researcher.deep_dive(
                topic=query, initial_context=initial_context, max_turns=2
            )

            # Step 2: Generate script with Perfect Loop + B-roll queries
            self.logger.info(
                "Step 2: Script Generation with Perfect Loop (Improvement 2)"
            )
            reddit_content_dict = {
                "title": reddit_post.title,
                "selftext": reddit_post.selftext,
                "subreddit": reddit_post.subreddit,
                "score": reddit_post.score,
                "num_comments": reddit_post.num_comments,
                "deep_research": research_facts,
            }
            analysis = await self.ai_client.analyze_video_content(
                None, reddit_content_dict
            )
            if not analysis:
                return {"success": False, "error": "Script generation failed"}

            # Step 3: Generate TTS audio
            self.logger.info("Step 3: TTS Generation")
            tts_results = (
                self.advanced_audio_processor.tts_service.generate_multiple_segments(
                    analysis.narrative_script_segments
                )
            )
            tts_paths = [
                item.get("audio_path")
                for item in tts_results
                if item.get("success") and item.get("audio_path")
            ]
            if not tts_paths:
                return {"success": False, "error": "TTS generation failed"}

            # Concatenate TTS segments
            audio_segments = [AudioFileClip(str(path)) for path in tts_paths]
            main_audio = concatenate_audioclips(audio_segments)

            # Step 4: Download B-roll images (Improvement 3)
            self.logger.info("Step 4: B-Roll Image Search (Improvement 3)")
            broll_queries = []
            for segment in analysis.narrative_script_segments:
                if (
                    hasattr(segment, "b_roll_search_query")
                    and segment.b_roll_search_query
                ):
                    broll_queries.append(segment.b_roll_search_query)

            async with BraveImageClient() as image_client:
                broll_images = await image_client.get_broll_images(
                    broll_queries, max_per_query=1
                )

            # Map images to moments
            broll_moments = []
            for segment in analysis.narrative_script_segments:
                if (
                    hasattr(segment, "b_roll_search_query")
                    and segment.b_roll_search_query
                ):
                    query = segment.b_roll_search_query
                    if query in broll_images and broll_images[query]:
                        broll_moments.append(
                            {
                                "image_path": str(broll_images[query][0]),
                                "timestamp_seconds": segment.time_seconds,
                                "duration": segment.intended_duration_seconds,
                            }
                        )

            # Step 5: Get background video
            self.logger.info("Step 5: Background Video")
            bg_manager = BackgroundManager()
            video_clip = bg_manager.get_sliced_background(
                target_duration=main_audio.duration
            )

            # Step 6: Apply B-roll images
            if broll_moments:
                self.logger.info("Step 6: Applying B-Roll Overlays (Improvement 3)")
                apply_broll = getattr(self.video_processor, "apply_broll_images", None)
                if callable(apply_broll):
                    video_clip = apply_broll(video_clip, broll_moments)
                else:
                    text_apply_broll = getattr(
                        self.video_processor.text_processor, "apply_broll_images", None
                    )
                    if callable(text_apply_broll):
                        video_clip = text_apply_broll(video_clip, broll_moments)

            # Step 7: Add text overlays
            if analysis.text_overlays:
                video_clip = self.video_processor.text_processor.add_text_overlays(
                    video_clip, analysis.text_overlays
                )

            # Step 8: Add word-level captions (Improvement 4) - optional
            combined_audio_path = None
            if options.get("enable_word_captions", False):
                self.logger.info("Step 8: Word-Level Captions (Improvement 4)")
                caption_gen = CaptionGenerator()
                # Combine all TTS paths for transcription
                combined_audio_path = self.config.paths.temp_dir / "combined_tts.wav"
                combined_audio_path.parent.mkdir(parents=True, exist_ok=True)
                main_audio.write_audiofile(
                    str(combined_audio_path), verbose=False, logger=None
                )

                video_clip = (
                    caption_gen.generate_word_captions(video_clip, combined_audio_path)
                    or video_clip
                )

            # Step 9: Add sound effects (Improvement 5)
            self.logger.info("Step 9: Sound Design (Improvement 5)")
            sfx_manager = SoundEffectsManager()
            audio_layers = [main_audio]

            # Add whoosh for each B-roll moment
            for moment in broll_moments:
                whoosh_path = sfx_manager.get_whoosh_sound()
                if whoosh_path:
                    whoosh_clip = (
                        AudioFileClip(str(whoosh_path))
                        .set_start(moment["timestamp_seconds"])
                        .volumex(0.4)
                    )
                    audio_layers.append(whoosh_clip)

            # Add boom for hook
            boom_path = sfx_manager.get_boom_sound()
            if boom_path and analysis.narrative_script_segments:
                hook_time = analysis.narrative_script_segments[0].time_seconds
                boom_clip = (
                    AudioFileClip(str(boom_path)).set_start(hook_time).volumex(0.5)
                )
                audio_layers.append(boom_clip)

            # Composite audio
            final_audio = None
            final_video = None
            output_file = (
                self.config.paths.processed_dir
                / f"production_studio_{reddit_post.id}.mp4"
            )
            try:
                final_audio = CompositeAudioClip(audio_layers)
                final_video = MoviePyCompat.with_audio(video_clip, final_audio)

                # Step 10: Write output
                output_file.parent.mkdir(parents=True, exist_ok=True)

                self.logger.info("Step 10: Rendering final video")
                final_video.write_videofile(
                    str(output_file),
                    fps=30,
                    codec="libx264",
                    audio_codec="aac",
                )
            finally:
                for clip in audio_segments:
                    try:
                        clip.close()
                    except Exception as e:
                        self.logger.warning("Failed to close audio segment clip: %s", e)
                for clip in audio_layers:
                    if clip is main_audio:
                        continue
                    try:
                        clip.close()
                    except Exception as e:
                        self.logger.warning("Failed to close audio layer clip: %s", e)
                if final_audio is not None:
                    try:
                        final_audio.close()
                    except Exception as e:
                        self.logger.warning(
                            "Failed to close composite audio clip: %s", e
                        )
                try:
                    main_audio.close()
                except Exception as e:
                    self.logger.warning("Failed to close main audio clip: %s", e)
                if final_video is not None:
                    try:
                        final_video.close()
                    except Exception as e:
                        self.logger.warning("Failed to close final video clip: %s", e)
                if combined_audio_path and combined_audio_path.exists():
                    try:
                        combined_audio_path.unlink()
                    except OSError as e:
                        self.logger.warning(
                            "Failed to remove temp caption audio %s: %s",
                            combined_audio_path,
                            e,
                        )

            self.logger.info("AI Production Studio pipeline complete!")

            return {
                "success": True,
                "video_path": str(output_file),
                "pipeline": "ai_production_studio",
                "features_used": [
                    feature
                    for feature in [
                        "agentic_research",
                        "perfect_loop",
                        "broll_injection",
                        "word_captions"
                        if options.get("enable_word_captions")
                        else None,
                        "sound_design",
                    ]
                    if feature is not None
                ],
                "broll_moments": len(broll_moments),
                "research_turns": 2,
            }

        except Exception as e:
            self.logger.error("AI Production Studio failed: %s", e, exc_info=True)
            self.gpu_manager.clear_gpu_cache()
            return {"success": False, "error": str(e), "stage": "ai_production_studio"}

    async def _download_and_analyze_video(self, reddit_url: str) -> Dict[str, Any]:
        """Download video and perform base analysis"""
        try:
            # Get Reddit post
            async with RedditClient() as reddit_client:
                reddit_post = await reddit_client.get_post_by_url(reddit_url)
            if not reddit_post or not reddit_post.video_url:
                return {
                    "success": False,
                    "error": "Failed to get Reddit content or no video URL",
                }

            # Download video using the VideoDownloader component
            output_file = self.config.paths.temp_dir / f"video_{reddit_post.id}"
            success = self.video_processor.downloader.download_video(
                reddit_post.video_url, output_file
            )

            if not success:
                return {"success": False, "error": "Video download failed"}

            # Find the actual downloaded file (yt-dlp adds extension)
            video_path = None
            for ext in [".mp4", ".webm", ".mkv", ".avi"]:
                potential_path = output_file.with_suffix(ext)
                if potential_path.exists():
                    video_path = potential_path
                    break

            if not video_path:
                return {"success": False, "error": "Downloaded video file not found"}

            # Perform base AI analysis
            analysis = await self.ai_client.analyze_video_content(
                video_path, reddit_post
            )

            if not analysis:
                return {"success": False, "error": "Video analysis failed"}

            return {
                "success": True,
                "video_path": video_path,
                "analysis": analysis,
                "reddit_post": reddit_post,
            }

        except asyncprawcore.exceptions.ResponseException as e:
            if e.response.status == 404:
                self.logger.error(
                    f"Error fetching post from URL {reddit_url}: received 404 HTTP response"
                )
                return {
                    "success": False,
                    "error": f"Reddit post not found: {reddit_url}",
                }
            raise  # Re-raise other ResponseExceptions

        except Exception as e:
            self.logger.error(f"Download and analysis failed: {e}")
            return {"success": False, "error": str(e)}

    async def _perform_enhanced_analysis(
        self, video_path: Path, base_analysis
    ) -> VideoAnalysisEnhanced:
        """Perform enhanced AI analysis with additional insights"""
        try:
            # Convert base analysis to enhanced version
            enhanced_analysis = self._convert_to_enhanced_analysis(base_analysis)

            # Add performance predictions
            enhanced_analysis.predicted_performance = (
                await self._predict_video_performance(enhanced_analysis)
            )

            # Generate enhancement recommendations
            enhanced_analysis.enhancement_recommendations = (
                self._generate_enhancement_recommendations(enhanced_analysis)
            )

            # Add advanced audio processing configuration
            enhanced_analysis.audio_ducking_config.duck_during_narration = True
            enhanced_analysis.audio_ducking_config.smart_detection = True
            enhanced_analysis.audio_ducking_config.preserve_music_dynamics = True

            self.logger.info("Enhanced AI analysis completed")
            return enhanced_analysis

        except Exception as e:
            self.logger.error(f"Enhanced analysis failed: {e}")
            # Return base analysis converted to enhanced format
            return self._convert_to_enhanced_analysis(base_analysis)

    def _convert_to_enhanced_analysis(self, base_analysis) -> VideoAnalysisEnhanced:
        """Convert base analysis to enhanced version"""
        try:
            # Extract base fields
            base_data = (
                base_analysis.dict()
                if hasattr(base_analysis, "dict")
                else base_analysis
            )

            # Create enhanced analysis with additional fields
            enhanced_data = {
                **base_data,
                "camera_movements": [],
                "dynamic_focus_points": [],
                "cinematic_transitions": [],
                "audio_ducking_config": {},
                "voice_enhancement_params": {},
                "background_audio_zones": [],
                "thumbnail_variants": [],
                "optimal_thumbnail_elements": {},
                "enhancement_recommendations": [],
                "predicted_performance": {},
                "comment_engagement_targets": [],
                "auto_response_triggers": [],
            }

            return VideoAnalysisEnhanced(**enhanced_data)

        except Exception as e:
            self.logger.error(f"Analysis conversion failed: {e}")
            # Return minimal enhanced analysis
            return VideoAnalysisEnhanced(
                suggested_title="Enhanced Video",
                summary_for_description="AI-enhanced video content",
                mood="exciting",
                has_clear_narrative=True,
                original_audio_is_key=False,
                hook_text="Must Watch!",
                hook_variations=["Amazing!", "Incredible!"],
                visual_hook_moment={"timestamp_seconds": 5.0, "description": "Hook"},
                audio_hook={
                    "type": "dramatic",
                    "sound_name": "whoosh",
                    "timestamp_seconds": 5.0,
                },
                best_segment={
                    "start_seconds": 0,
                    "end_seconds": 30,
                    "reason": "Best part",
                },
                segments=[
                    {"start_seconds": 0, "end_seconds": 30, "reason": "Main segment"}
                ],
                music_genres=["upbeat"],
                hashtags=["#shorts", "#viral"],
                thumbnail_info={
                    "timestamp_seconds": 10.0,
                    "reason": "Good frame",
                    "headline_text": "Must Watch!",
                },
                call_to_action={"text": "Subscribe!", "type": "subscribe"},
            )

    async def _apply_cinematic_analysis(
        self, video_path: Path, analysis: VideoAnalysisEnhanced
    ) -> VideoAnalysisEnhanced:
        """Apply cinematic editing analysis"""
        try:
            self.logger.info("Applying cinematic analysis...")

            # Perform cinematic analysis
            enhanced_analysis = self.cinematic_editor.analyze_video_cinematically(
                video_path, analysis
            )

            self.logger.info(
                f"Cinematic analysis complete: {len(enhanced_analysis.camera_movements)} movements suggested"
            )
            return enhanced_analysis

        except Exception as e:
            self.logger.error(f"Cinematic analysis failed: {e}")
            return analysis

    async def _process_with_enhancements(
        self,
        video_path: Path,
        analysis: VideoAnalysisEnhanced,
        options: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Process video with advanced audio and video enhancements"""
        try:
            self.logger.info("Processing video with advanced enhancements...")

            # Select appropriate background music based on analysis
            background_music_path = self._select_background_music(analysis)

            # This calls the main video processor
            output_file = (
                self.config.paths.processed_dir / f"enhanced_{video_path.stem}.mp4"
            )
            processing_result = self.video_processor.process_video(
                video_path,
                output_file,
                analysis,
                background_music_path=background_music_path,
                generate_thumbnail=False,
            )

            self.logger.info("Advanced enhancement processing complete")
            return processing_result

        except Exception as e:
            self.logger.error(
                f"Advanced enhancement processing failed: {e}", exc_info=True
            )
            return {"success": False, "error": str(e)}

    async def _generate_ab_test_thumbnails(
        self,
        video_path: Path,
        analysis: VideoAnalysisEnhanced,
        processing_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate A/B test thumbnail variants"""
        try:
            if not self.enable_ab_testing:
                return []

            self.logger.info("Generating A/B test thumbnails...")

            # Create thumbnails directory
            thumbnails_dir = (
                self.config.paths.thumbnails_dir
                / f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            # Generate thumbnail variants
            variants = self.enhanced_thumbnail_generator.generate_ab_test_thumbnails(
                video_path, analysis, thumbnails_dir, num_variants=3
            )

            # Convert to result format
            thumbnail_results = []
            for variant in variants:
                thumbnail_results.append(
                    {
                        "variant_id": variant.variant_id,
                        "headline_text": variant.headline_text,
                        "style_config": {
                            "text_style": variant.text_style,
                            "color_scheme": variant.color_scheme,
                            "emotional_tone": variant.emotional_tone,
                        },
                        "file_path": str(thumbnails_dir / f"{variant.variant_id}.jpg"),
                    }
                )

            self.logger.info(f"Generated {len(thumbnail_results)} A/B test thumbnails")
            return thumbnail_results

        except Exception as e:
            self.logger.error(f"A/B thumbnail generation failed: {e}")
            return []

    async def _upload_and_track_video(
        self,
        processing_result: Dict[str, Any],
        analysis: VideoAnalysisEnhanced,
        thumbnail_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Upload video and start performance tracking"""
        try:
            self.logger.info("Uploading video with performance tracking...")

            # Prepare video metadata
            video_metadata = {
                "title": analysis.suggested_title,
                "description": self._create_enhanced_description(analysis),
                "tags": [tag.replace("#", "") for tag in analysis.hashtags],
                "category_id": "22",  # People & Blogs
                "privacy_status": "public",
            }

            # Upload video
            video_path = processing_result.get("video_path")
            if not video_path or not Path(video_path).exists():
                return {"success": False, "error": "Processed video not found"}

            # Use first thumbnail variant as initial thumbnail
            initial_thumbnail = None
            if thumbnail_results:
                initial_thumbnail = thumbnail_results[0]["file_path"]

            upload_result = await self.youtube_client.upload_video(
                video_path, video_metadata, initial_thumbnail
            )

            if not upload_result.get("success"):
                return upload_result

            video_id = upload_result.get("video_id")

            # Record video upload in engagement monitoring
            if video_id:
                enhancements_used = self._extract_enhancements_used(analysis)
                self.engagement_monitor.record_video_upload(
                    video_id,
                    analysis.suggested_title,
                    processing_result.get("duration", 60.0),
                    enhancements_used,
                )

            self.logger.info(f"Video uploaded successfully: {video_id}")
            return {
                "success": True,
                "video_id": video_id,
                "video_url": f"https://youtube.com/watch?v={video_id}",
                "thumbnail_variants": thumbnail_results,
            }

        except Exception as e:
            self.logger.error(f"Upload and tracking failed: {e}")
            return {"success": False, "error": str(e)}

    def _create_enhanced_description(self, analysis: VideoAnalysisEnhanced) -> str:
        """Create enhanced video description with AI optimizations"""
        description_parts = [
            analysis.summary_for_description,
            "",
            "🔥 Enhanced with AI-powered editing for maximum engagement!",
            "",
            "📱 Follow for more amazing content!",
            "",
            "🏷️ Tags: " + " ".join(analysis.hashtags),
        ]

        # Add call to action
        if hasattr(analysis, "call_to_action") and analysis.call_to_action:
            description_parts.insert(-2, f"👍 {analysis.call_to_action.text}")

        return "\n".join(description_parts)

    def _extract_enhancements_used(self, analysis: VideoAnalysisEnhanced) -> List[str]:
        """Extract list of enhancements used in processing"""
        enhancements = []

        if analysis.camera_movements:
            enhancements.append("cinematic_movements")

        if analysis.speed_effects:
            enhancements.append("speed_effects")

        if analysis.visual_cues:
            enhancements.append("visual_effects")

        if analysis.text_overlays:
            enhancements.append("text_overlays")

        if analysis.sound_effects:
            enhancements.append("sound_effects")

        if analysis.narrative_script_segments:
            enhancements.append("ai_narration")

        if (
            analysis.audio_ducking_config
            and analysis.audio_ducking_config.duck_during_narration
        ):
            enhancements.append("advanced_audio_ducking")

        return enhancements

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
            optimization_result = self.enhancement_optimizer.optimize_parameters()

            if optimization_result.get("status") == "completed":
                applied_changes = optimization_result.get("applied_changes", [])
                self.logger.info(
                    f"Optimization complete: {len(applied_changes)} parameters adjusted"
                )

        except Exception as e:
            self.logger.error(f"Optimization analysis failed: {e}")

    async def _predict_video_performance(
        self, analysis: VideoAnalysisEnhanced
    ) -> Dict[str, float]:
        """Predict video performance using AI"""
        try:
            # This would use ML models to predict performance
            # For now, we'll provide estimated predictions based on features

            # Calculate feature scores
            engagement_features = len(analysis.hook_variations) * 5
            visual_features = len(analysis.visual_cues) * 3
            audio_features = len(analysis.sound_effects) * 2
            narrative_features = len(analysis.narrative_script_segments) * 4

            base_score = 50  # Baseline
            feature_bonus = (
                engagement_features
                + visual_features
                + audio_features
                + narrative_features
            )

            # Predict metrics
            predicted_views = min(100000, 1000 + feature_bonus * 100)
            predicted_engagement_rate = min(20.0, 5.0 + feature_bonus * 0.1)
            predicted_retention = min(95.0, 60.0 + feature_bonus * 0.05)
            predicted_ctr = min(0.25, 0.05 + feature_bonus * 0.001)

            return {
                "predicted_views": predicted_views,
                "predicted_engagement_rate": predicted_engagement_rate,
                "predicted_retention_rate": predicted_retention,
                "predicted_ctr": predicted_ctr,
                "confidence_score": 0.75,  # 75% confidence
            }

        except Exception as e:
            self.logger.error(f"Performance prediction failed: {e}")
            return {}

    def _generate_enhancement_recommendations(
        self, analysis: VideoAnalysisEnhanced
    ) -> List[Dict[str, Any]]:
        """Generate enhancement recommendations based on analysis"""
        from src.models import EnhancementOptimization

        recommendations = []

        try:
            # Recommend speed effects for high-energy content
            if (
                analysis.mood in ["exciting", "energetic"]
                and len(analysis.speed_effects) < 2
            ):
                recommendations.append(
                    EnhancementOptimization(
                        effect_type="speed_effects",
                        current_intensity=0.5,
                        performance_score=75.0,
                        usage_frequency=0.3,
                        retention_impact=15.0,
                        engagement_impact=12.0,
                        recommended_adjustment=0.2,
                    )
                )

            # Recommend more visual cues for complex content
            if len(analysis.text_overlays) > 5 and len(analysis.visual_cues) < 3:
                recommendations.append(
                    EnhancementOptimization(
                        effect_type="visual_cues",
                        current_intensity=0.3,
                        performance_score=68.0,
                        usage_frequency=0.6,
                        retention_impact=8.0,
                        engagement_impact=10.0,
                        recommended_adjustment=0.4,
                    )
                )

            # Recommend sound effects for engagement
            if len(analysis.sound_effects) < 3:
                recommendations.append(
                    EnhancementOptimization(
                        effect_type="sound_effects",
                        current_intensity=0.4,
                        performance_score=72.0,
                        usage_frequency=0.5,
                        retention_impact=12.0,
                        engagement_impact=15.0,
                        recommended_adjustment=0.3,
                    )
                )

        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")

        return recommendations

    def _compile_enhanced_results(
        self,
        download_result: Dict[str, Any],
        processing_result: Dict[str, Any],
        thumbnail_results: List[Dict[str, Any]],
        upload_result: Dict[str, Any],
        analysis: VideoAnalysisEnhanced,
    ) -> Dict[str, Any]:
        """Compile comprehensive results from enhanced processing"""
        return {
            "success": True,
            "enhanced_processing": True,
            "timestamp": datetime.now().isoformat(),
            # Core results
            "video_id": upload_result.get("video_id"),
            "video_url": upload_result.get("video_url"),
            "video_path": processing_result.get("video_path"),
            # Enhanced features
            "cinematic_enhancements": {
                "camera_movements": len(analysis.camera_movements),
                "dynamic_focus_points": len(analysis.dynamic_focus_points),
                "cinematic_transitions": len(analysis.cinematic_transitions),
            },
            "audio_enhancements": {
                "advanced_ducking_enabled": analysis.audio_ducking_config.duck_during_narration,
                "smart_detection_used": analysis.audio_ducking_config.smart_detection,
                "voice_enhancement_applied": bool(analysis.voice_enhancement_params),
            },
            "thumbnail_optimization": {
                "ab_testing_enabled": len(thumbnail_results) > 1,
                "variants_generated": len(thumbnail_results),
                "thumbnail_variants": thumbnail_results,
            },
            "performance_prediction": analysis.predicted_performance,
            "enhancement_recommendations": analysis.enhancement_recommendations,
            # Management status
            "proactive_management_enabled": self.enable_proactive_management,
            "auto_optimization_enabled": self.enable_auto_optimization,
            # Processing stats
            "processing_time_seconds": processing_result.get("processing_time"),
            "gpu_usage": self.gpu_manager.get_memory_summary(),
            # Analysis summary
            "analysis_summary": {
                "total_enhancements": len(self._extract_enhancements_used(analysis)),
                "ai_confidence": analysis.predicted_performance.get(
                    "confidence_score", 0.5
                ),
                "complexity_score": len(analysis.visual_cues)
                + len(analysis.text_overlays)
                + len(analysis.sound_effects),
            },
        }

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "system_status": "operational",
                # Component status
                "components": {
                    "cinematic_editor": "active"
                    if self.enable_cinematic_editing
                    else "disabled",
                    "advanced_audio": "active"
                    if self.enable_advanced_audio
                    else "disabled",
                    "ab_testing": "active" if self.enable_ab_testing else "disabled",
                    "auto_optimization": "active"
                    if self.enable_auto_optimization
                    else "disabled",
                    "proactive_management": "active"
                    if self.enable_proactive_management
                    else "disabled",
                },
                # Resource status
                "resources": {},
                "optimization_summary": {},
                "channel_management": {},
                # Processing capabilities
                "capabilities": {
                    "max_video_length_minutes": 10,
                    "supported_formats": ["mp4", "webm", "avi"],
                    "ai_features_available": [
                        "cinematic_editing",
                        "advanced_audio_ducking",
                        "thumbnail_ab_testing",
                        "performance_optimization",
                        "comment_management",
                    ],
                },
            }

            # Add resource status safely
            try:
                status["resources"] = self.gpu_manager.get_memory_summary()
            except Exception as e:
                self.logger.warning(f"Could not get GPU memory summary: {e}")
                status["resources"] = {}

            # Add optimization summary safely
            try:
                status["optimization_summary"] = (
                    self.enhancement_optimizer.get_optimization_summary()
                )
            except Exception as e:
                self.logger.warning(f"Could not get optimization summary: {e}")
                status["optimization_summary"] = {}

            # Add channel management summary safely
            try:
                status["channel_management"] = (
                    self.channel_manager.get_management_summary()
                )
            except Exception as e:
                self.logger.warning(f"Could not get management summary: {e}")
                status["channel_management"] = {}

            return status

        except Exception as e:
            self.logger.error("System status check failed", exc_info=True)
            return {
                "timestamp": datetime.now().isoformat(),
                "system_status": "error",
                "error": str(e),
            }

    async def run_batch_optimization(
        self,
        video_urls: List[str],
        optimization_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run batch processing with optimization for multiple videos"""
        try:
            self.logger.info(
                f"Starting batch optimization for {len(video_urls)} videos"
            )

            results = []
            total_start_time = datetime.now()

            for i, url in enumerate(video_urls):
                try:
                    self.logger.info(
                        f"Processing video {i + 1}/{len(video_urls)}: {url}"
                    )

                    # Process individual video
                    result = await self.process_enhanced_video(url, optimization_config)

                    results.append({"url": url, "index": i, "result": result})

                    # GPU memory management between videos
                    self.gpu_manager.clear_gpu_cache()

                except Exception as e:
                    self.logger.error(f"Batch processing failed for video {i + 1}: {e}")
                    results.append(
                        {
                            "url": url,
                            "index": i,
                            "result": {"success": False, "error": str(e)},
                        }
                    )

            total_processing_time = (datetime.now() - total_start_time).total_seconds()

            # Run system optimization after batch
            if self.enable_auto_optimization:
                await self._run_optimization_analysis()

            # Compile batch results
            successful_videos = len([r for r in results if r["result"].get("success")])

            return {
                "success": True,
                "batch_summary": {
                    "total_videos": len(video_urls),
                    "successful_videos": successful_videos,
                    "failed_videos": len(video_urls) - successful_videos,
                    "total_processing_time_seconds": total_processing_time,
                    "average_time_per_video": total_processing_time / len(video_urls),
                },
                "individual_results": results,
                "system_status_post_batch": await self.get_system_status(),
            }

        except Exception as e:
            self.logger.error(f"Batch optimization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "partial_results": results if "results" in locals() else [],
            }

    def _select_background_music(self, analysis: VideoAnalysis) -> Optional[Path]:
        """
        Select appropriate background music based on video analysis

        Args:
            analysis: Video analysis containing mood and genre information

        Returns:
            Path to selected music file or None if no suitable music found
        """
        try:
            music_folder = self.config.paths.music_folder

            if not music_folder.exists():
                self.logger.warning("Music folder not found")
                return None

            # Determine mood/genre from analysis
            mood = getattr(analysis, "mood", "upbeat").lower()
            music_genres = getattr(analysis, "music_genres", ["upbeat"])

            # Map moods to music categories
            mood_mapping = {
                "exciting": "upbeat",
                "dramatic": "suspenseful",
                "calm": "relaxing",
                "peaceful": "relaxing",
                "funny": "funny",
                "humorous": "funny",
                "emotional": "emotional",
                "sad": "emotional",
                "heartwarming": "emotional",
                "informative": "informative",
                "educational": "informative",
                "mysterious": "suspenseful",
                "tense": "suspenseful",
            }

            # Get music category from mood or genres
            target_category = mood_mapping.get(mood, "upbeat")

            # If analysis has specific genres, use the first one
            if music_genres and music_genres[0] in [
                "upbeat",
                "emotional",
                "suspenseful",
                "relaxing",
                "funny",
                "informative",
            ]:
                target_category = music_genres[0]

            # Look for music in the target category folder first
            category_folder = music_folder / target_category
            if category_folder.exists():
                music_files = list(category_folder.glob("*.mp3"))
                if music_files:
                    selected = music_files[0]  # Take first available
                    self.logger.info(
                        f"Selected background music from {target_category}: {selected.name}"
                    )
                    return selected

            # Fallback: look in main music folder
            main_music_files = list(music_folder.glob("*.mp3"))
            if main_music_files:
                selected = main_music_files[0]  # Take first available
                self.logger.info(f"Selected fallback background music: {selected.name}")
                return selected

            # Last resort: look in any subfolder
            for subfolder in [
                "upbeat",
                "relaxing",
                "informative",
                "funny",
                "emotional",
                "suspenseful",
            ]:
                subfolder_path = music_folder / subfolder
                if subfolder_path.exists():
                    music_files = list(subfolder_path.glob("*.mp3"))
                    if music_files:
                        selected = music_files[0]
                        self.logger.info(
                            f"Selected background music from {subfolder}: {selected.name}"
                        )
                        return selected

            self.logger.warning("No background music files found")
            return None

        except Exception as e:
            self.logger.error(f"Failed to select background music: {e}")
            return None
