"""
Proactive Channel Management System
Handles AI-powered comment analysis, engagement optimization, and thumbnail A/B testing management.
"""

import logging
import asyncio
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict

from src.config.settings import get_config
from src.models import CommentEngagement, ThumbnailVariant, PerformanceMetrics
from src.integrations.youtube_client import YouTubeClient
from src.integrations.ai_client import AIClient
from src.processing.enhanced_thumbnail_generator import EnhancedThumbnailGenerator
from src.monitoring.engagement_metrics import EngagementMonitor
from src.database.db_manager import get_db_manager, DatabaseManager


@dataclass
class CommentAnalysis:
    """Analysis results for a comment"""

    comment_id: str
    comment_text: str
    author_name: str
    engagement_score: float
    sentiment: str
    toxicity_score: float
    reply_urgency: str  # 'high', 'medium', 'low', 'none'
    suggested_response: Optional[str]
    should_pin: bool
    should_heart: bool
    interaction_priority: int  # 1-10, 10 being highest


@dataclass
class ThumbnailTestResult:
    """Results from thumbnail A/B testing"""

    test_id: str
    video_id: str
    variants_tested: List[str]
    winner_variant: str
    confidence_level: float
    performance_improvement: float
    test_duration_hours: int
    total_impressions: int


class ChannelManager:
    """
    Proactive channel management system that automates engagement,
    comment management, and thumbnail optimization.
    """

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        # Initialize services
        self.youtube_client = YouTubeClient()
        self.ai_client = AIClient()
        self.thumbnail_generator = EnhancedThumbnailGenerator()
        self.engagement_monitor = EngagementMonitor()

        # Management parameters
        self.comment_analysis_interval = 300  # 5 minutes
        self.thumbnail_test_duration = 24  # 24 hours
        self.min_impressions_for_test = 1000  # Minimum impressions before switching
        self.min_variant_samples = 1

        # AI prompts for comment analysis
        self.comment_analysis_prompt = """
        Analyze this YouTube comment for engagement potential and response strategy:
        
        Comment: "{comment_text}"
        Video context: {video_context}
        
        Provide analysis in the following format:
        - Engagement Score (0-100): How likely this comment is to drive engagement
        - Sentiment: positive/negative/neutral
        - Toxicity Score (0-100): How toxic or harmful the comment is
        - Reply Urgency: high/medium/low/none
        - Should Pin: true/false (if this comment should be pinned)
        - Should Heart: true/false (if this comment should be hearted)
        - Suggested Response: Brief, engaging response that encourages discussion
        - Interaction Priority: 1-10 priority score
        
        Focus on comments that:
        - Ask engaging questions
        - Share personal experiences
        - Provide valuable insights
        - Could spark healthy discussion
        - Show genuine appreciation
        
        Avoid prioritizing:
        - Spam or promotional content
        - Toxic or negative comments
        - Generic comments (like "first!")
        - Comments that don't add value
        """

        # Active tests tracking
        self.active_thumbnail_tests = {}
        self.comment_interaction_history = {}

    async def run_proactive_management(self):
        """Main loop for proactive channel management"""
        self.logger.info("Starting proactive channel management...")

        try:
            while True:
                # Get recent videos for management
                recent_videos = await self._get_recent_videos()

                # Process each video
                for video_info in recent_videos:
                    try:
                        await self._manage_video(video_info)
                    except Exception as e:
                        self.logger.error(
                            f"Error managing video {video_info.get('id', 'unknown')}: {e}"
                        )

                # Check thumbnail A/B tests
                await self._check_thumbnail_tests()
                await self.run_thumbnail_winner_selection_job()

                # Wait before next cycle
                await asyncio.sleep(self.comment_analysis_interval)

        except Exception as e:
            self.logger.error(f"Proactive management loop failed: {e}")

    async def _get_recent_videos(self) -> List[Dict[str, Any]]:
        """Get recent videos that need management"""
        try:
            # Get videos from last 7 days
            videos = await self.youtube_client.get_recent_videos(days=7)
            return videos

        except Exception as e:
            self.logger.error(f"Failed to get recent videos: {e}")
            return []

    async def _manage_video(self, video_info: Dict[str, Any]):
        """Manage a specific video (comments, thumbnails, etc.)"""
        video_id = video_info.get("id")
        if not video_id:
            return

        try:
            # Analyze and manage comments
            await self._manage_comments(video_id, video_info)

            # Check thumbnail performance
            await self._manage_thumbnail_performance(video_id, video_info)

            # Update performance metrics
            await self._update_video_metrics(video_id, video_info)

        except Exception as e:
            self.logger.error(f"Video management failed for {video_id}: {e}")

    async def _manage_comments(self, video_id: str, video_info: Dict[str, Any]):
        """Analyze and manage comments for a video"""
        try:
            # Get recent comments
            comments = await self.youtube_client.get_video_comments(
                video_id, max_results=50
            )

            if not comments:
                return

            # Analyze comments with AI
            comment_analyses = await self._analyze_comments(comments, video_info)

            # Take actions based on analysis
            await self._execute_comment_actions(comment_analyses, video_id)

            self.logger.info(
                f"Managed {len(comment_analyses)} comments for video {video_id}"
            )

        except Exception as e:
            self.logger.error(f"Comment management failed for {video_id}: {e}")

    async def _analyze_comments(
        self, comments: List[Dict[str, Any]], video_info: Dict[str, Any]
    ) -> List[CommentAnalysis]:
        """Analyze comments using AI for engagement optimization in batches"""
        analyses = []

        try:
            video_context = {
                "title": video_info.get("title", ""),
                "description": video_info.get("description", "")[
                    :200
                ],  # Truncate for context
                "tags": video_info.get("tags", [])[:5],  # First 5 tags
            }

            # Take up to 20 comments to prevent huge context windows
            batch_comments = comments[:20]
            if not batch_comments:
                return []

            formatted_comments = []
            for c in batch_comments:
                formatted_comments.append(
                    {
                        "id": c.get("id", ""),
                        "author": c.get("authorDisplayName", ""),
                        "text": c.get("textDisplay", ""),
                    }
                )

            # Call AI client for batch analysis
            results = []
            if hasattr(self.ai_client, "analyze_comments_batch"):
                results = await self.ai_client.analyze_comments_batch(
                    formatted_comments, video_context
                )
            else:
                self.logger.warning(
                    "AI client does not support batch comment analysis, falling back to individual analysis"
                )
                async def safe_analyze(c):
                    try:
                        return await self._analyze_single_comment(c, video_context)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to analyze comment {c.get('id', 'unknown')}: {e}",
                            exc_info=True,
                            )
                        return None

                tasks = [safe_analyze(c) for c in batch_comments]
                results_gather = await asyncio.gather(*tasks)
                analyses.extend(r for r in results_gather if r is not None)

            # Create a map of comments for easy lookup
            comment_map = {c.get("id"): c for c in batch_comments}

            for res in results:
                comment_id = res.get("comment_id")
                if not comment_id or comment_id not in comment_map:
                    continue

                original_comment = comment_map[comment_id]
                comment_text = original_comment.get("textDisplay", "")
                author_name = original_comment.get("authorDisplayName", "")

                # Derive analysis fields from model output
                engagement_score = float(res.get("engagement_score", 50.0))
                toxicity_score = float(res.get("toxicity_score", 0.0))
                sentiment = res.get("sentiment", "neutral")

                # Determine reply urgency and actions
                if engagement_score > 85 and toxicity_score < 15:
                    reply_urgency = "high"
                elif engagement_score > 60 and toxicity_score < 30:
                    reply_urgency = "medium"
                else:
                    reply_urgency = "low"

                should_pin = (
                    engagement_score > 80
                    and toxicity_score < 20
                    and sentiment == "positive"
                )
                should_heart = (
                    engagement_score > 60
                    and toxicity_score < 10
                    and sentiment != "negative"
                )

                # Scale interaction priority from 0-100 to 1-10
                interaction_priority_100 = res.get("interaction_priority", 50)
                interaction_priority_10 = min(
                    10, max(1, int(interaction_priority_100 / 10))
                )

                analyses.append(
                    CommentAnalysis(
                        comment_id=comment_id,
                        comment_text=comment_text,
                        author_name=author_name,
                        engagement_score=engagement_score,
                        sentiment=sentiment,
                        toxicity_score=toxicity_score,
                        reply_urgency=reply_urgency,
                        suggested_response=res.get("suggested_reply", ""),
                        should_pin=should_pin,
                        should_heart=should_heart,
                        interaction_priority=interaction_priority_10,
                    )
                )

            # Sort by interaction priority
            analyses.sort(key=lambda x: x.interaction_priority, reverse=True)

        except Exception as e:
            self.logger.error("Comment analysis batch failed", exc_info=True)

        return analyses

    async def _analyze_single_comment(
        self, comment: Dict[str, Any], video_context: Dict[str, Any]
    ) -> Optional[CommentAnalysis]:
        """Analyze a single comment for engagement potential"""
        try:
            comment_text = comment.get("textDisplay", "")
            comment_id = comment.get("id", "")
            author_name = comment.get("authorDisplayName", "")

            if not comment_text or len(comment_text.strip()) < 3:
                return None

            # Skip if already analyzed recently
            if self._was_recently_analyzed(comment_id):
                return None

            # Prepare AI analysis prompt
            analysis_prompt = self.comment_analysis_prompt.format(
                comment_text=comment_text,
                video_context=json.dumps(video_context, indent=2),
            )

            # Get AI analysis
            ai_response = await self.ai_client.analyze_comment_engagement(
                comment_text, video_context, analysis_prompt
            )

            if not ai_response:
                return None

            # Parse AI response
            analysis = self._parse_comment_analysis(
                ai_response, comment_id, comment_text, author_name
            )

            # Mark as analyzed
            self._mark_comment_analyzed(comment_id)

            return analysis

        except Exception as e:
            self.logger.warning(f"Single comment analysis failed: {e}")
            return None

    def _parse_comment_analysis(
        self, ai_response: str, comment_id: str, comment_text: str, author_name: str
    ) -> CommentAnalysis:
        """Parse AI response into structured comment analysis"""
        try:
            # This would parse the AI response based on the expected format
            # For now, we'll simulate realistic analysis

            # Calculate engagement score based on comment characteristics
            engagement_score = self._calculate_engagement_score(comment_text)

            # Determine sentiment
            sentiment = self._analyze_sentiment(comment_text)

            # Calculate toxicity (simplified)
            toxicity_score = self._calculate_toxicity_score(comment_text)

            # Determine if comment should be pinned
            should_pin = (
                engagement_score > 80
                and toxicity_score < 20
                and len(comment_text) > 20
                and sentiment == "positive"
            )

            # Determine if comment should be hearted
            should_heart = (
                engagement_score > 60
                and toxicity_score < 10
                and sentiment in ["positive", "neutral"]
            )

            # Generate suggested response
            suggested_response = self._generate_suggested_response(
                comment_text, sentiment
            )

            # Calculate interaction priority
            priority = min(10, int((engagement_score + (100 - toxicity_score)) / 20))

            # Determine reply urgency
            if engagement_score > 85 and toxicity_score < 15:
                reply_urgency = "high"
            elif engagement_score > 60 and toxicity_score < 30:
                reply_urgency = "medium"
            elif engagement_score > 30:
                reply_urgency = "low"
            else:
                reply_urgency = "none"

            return CommentAnalysis(
                comment_id=comment_id,
                comment_text=comment_text,
                author_name=author_name,
                engagement_score=engagement_score,
                sentiment=sentiment,
                toxicity_score=toxicity_score,
                reply_urgency=reply_urgency,
                suggested_response=suggested_response,
                should_pin=should_pin,
                should_heart=should_heart,
                interaction_priority=priority,
            )

        except Exception as e:
            self.logger.warning(f"Comment analysis parsing failed: {e}")
            # Return default analysis
            return CommentAnalysis(
                comment_id=comment_id,
                comment_text=comment_text,
                author_name=author_name,
                engagement_score=50.0,
                sentiment="neutral",
                toxicity_score=0.0,
                reply_urgency="low",
                suggested_response=None,
                should_pin=False,
                should_heart=False,
                interaction_priority=5,
            )

    def _calculate_engagement_score(self, comment_text: str) -> float:
        """Calculate engagement potential score for a comment"""
        score = 0.0
        text_lower = comment_text.lower()

        # Length bonus (but not too long)
        if 20 <= len(comment_text) <= 200:
            score += 20
        elif 10 <= len(comment_text) < 20:
            score += 10

        # Question bonus
        if "?" in comment_text:
            score += 25

        # Personal experience indicators
        personal_indicators = [
            "i think",
            "i believe",
            "my experience",
            "i tried",
            "i found",
        ]
        if any(indicator in text_lower for indicator in personal_indicators):
            score += 20

        # Appreciation/positive feedback
        positive_indicators = [
            "amazing",
            "great",
            "awesome",
            "helpful",
            "thank you",
            "love this",
        ]
        if any(indicator in text_lower for indicator in positive_indicators):
            score += 15

        # Discussion starters
        discussion_starters = [
            "what do you think",
            "anyone else",
            "does anyone",
            "thoughts on",
        ]
        if any(starter in text_lower for starter in discussion_starters):
            score += 30

        # Deduct for spam-like content
        spam_indicators = ["first", "subscribe to me", "check out my", "like if you"]
        if any(indicator in text_lower for indicator in spam_indicators):
            score -= 40

        return max(0, min(100, score))

    def _analyze_sentiment(self, comment_text: str) -> str:
        """Analyze sentiment of comment text"""
        text_lower = comment_text.lower()

        positive_words = [
            "great",
            "amazing",
            "awesome",
            "love",
            "fantastic",
            "brilliant",
            "helpful",
        ]
        negative_words = [
            "hate",
            "terrible",
            "awful",
            "stupid",
            "boring",
            "waste",
            "worst",
        ]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def _calculate_toxicity_score(self, comment_text: str) -> float:
        """Calculate toxicity score (simplified implementation)"""
        text_lower = comment_text.lower()

        toxic_indicators = [
            "stupid",
            "idiot",
            "hate",
            "kill",
            "die",
            "worst",
            "terrible person",
        ]
        toxic_count = sum(
            1 for indicator in toxic_indicators if indicator in text_lower
        )

        # Caps detection (excessive caps can indicate aggression)
        caps_ratio = sum(1 for c in comment_text if c.isupper()) / max(
            len(comment_text), 1
        )
        if caps_ratio > 0.5:
            toxic_count += 1

        return min(100, toxic_count * 25)

    def _generate_suggested_response(
        self, comment_text: str, sentiment: str
    ) -> Optional[str]:
        """Generate suggested response to a comment"""
        text_lower = comment_text.lower()

        if sentiment == "positive":
            if "thank" in text_lower:
                return "Thank you so much! Glad you enjoyed the content! 🙏"
            elif "love" in text_lower:
                return "So happy to hear that! More content like this coming soon! ❤️"
            elif "helpful" in text_lower:
                return "That's exactly what I was hoping for! Thanks for watching! 💪"
            else:
                return "Really appreciate the positive feedback! 😊"

        elif "?" in comment_text:
            return "Great question! Let me think about doing a video on this topic!"

        elif sentiment == "neutral" and len(comment_text) > 20:
            return "Thanks for sharing your thoughts! 👍"

        return None

    def _was_recently_analyzed(self, comment_id: str) -> bool:
        """Check if comment was analyzed recently"""
        last_analyzed = self.comment_interaction_history.get(comment_id, {}).get(
            "last_analyzed"
        )
        if not last_analyzed:
            return False

        last_time = datetime.fromisoformat(last_analyzed)
        return (datetime.now() - last_time).total_seconds() < 24 * 3600

    def _mark_comment_analyzed(self, comment_id: str):
        """Mark comment as analyzed"""
        if comment_id not in self.comment_interaction_history:
            self.comment_interaction_history[comment_id] = {}

        self.comment_interaction_history[comment_id]["last_analyzed"] = (
            datetime.now().isoformat()
        )

    async def _execute_comment_actions(
        self, analyses: List[CommentAnalysis], video_id: str
    ):
        """Execute actions based on comment analysis"""
        try:
            actions_taken = 0

            for analysis in analyses[:10]:  # Limit to top 10 priority comments
                try:
                    # Pin high-priority comments
                    if analysis.should_pin and analysis.interaction_priority >= 8:
                        success = await self.youtube_client.pin_comment(
                            analysis.comment_id
                        )
                        if success:
                            actions_taken += 1
                            self.logger.info(
                                f"Pinned comment from {analysis.author_name}"
                            )

                    # Heart good comments
                    elif analysis.should_heart and analysis.interaction_priority >= 6:
                        success = await self.youtube_client.heart_comment(
                            analysis.comment_id
                        )
                        if success:
                            actions_taken += 1

                    # Reply to high-priority comments
                    if (
                        analysis.reply_urgency == "high"
                        and analysis.suggested_response
                        and analysis.interaction_priority >= 7
                    ):
                        success = await self.youtube_client.reply_to_comment(
                            analysis.comment_id, analysis.suggested_response
                        )
                        if success:
                            actions_taken += 1
                            self.logger.info(
                                f"Replied to comment from {analysis.author_name}"
                            )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to execute action for comment {analysis.comment_id}: {e}"
                    )

            self.logger.info(
                f"Executed {actions_taken} comment actions for video {video_id}"
            )

        except Exception as e:
            self.logger.error(f"Comment action execution failed: {e}")

    async def _manage_thumbnail_performance(
        self, video_id: str, video_info: Dict[str, Any]
    ):
        """Manage thumbnail A/B testing and optimization"""
        try:
            # Check if video is eligible for thumbnail testing
            upload_date = video_info.get("publishedAt")
            if not upload_date:
                return

            # Only test thumbnails for videos less than 48 hours old
            upload_time = datetime.fromisoformat(upload_date.replace("Z", "+00:00"))
            hours_since_upload = (
                datetime.now(timezone.utc) - upload_time
            ).total_seconds() / 3600

            if hours_since_upload > 48:
                return

            # Check if already testing
            if video_id in self.active_thumbnail_tests:
                await self._check_specific_thumbnail_test(video_id)
            else:
                # Start new A/B test if conditions are met
                await self._start_thumbnail_ab_test(video_id, video_info)

        except Exception as e:
            self.logger.error(f"Thumbnail management failed for {video_id}: {e}")

    async def _start_thumbnail_ab_test(self, video_id: str, video_info: Dict[str, Any]):
        """Start A/B testing for thumbnail optimization"""
        try:
            # Get current video performance
            current_stats = await self.youtube_client.get_video_analytics(video_id)

            if not current_stats:
                return

            # Check if video has enough traffic for testing
            impressions = current_stats.get("impressions", 0)
            if impressions < self.min_impressions_for_test:
                return

            # Generate A/B test thumbnails
            video_path = self._get_video_path(video_id)  # Would need to implement
            if not video_path or not video_path.exists():
                return

            # Create analysis object for thumbnail generation
            analysis = self._create_analysis_from_video_info(video_info)

            # Generate thumbnail variants
            output_dir = Path(f"thumbnails/ab_tests/{video_id}")
            variants = self.thumbnail_generator.generate_ab_test_thumbnails(
                video_path, analysis, output_dir, num_variants=3
            )

            if len(variants) < 2:
                return

            # Start the A/B test
            test_info = {
                "video_id": video_id,
                "start_time": datetime.now().isoformat(),
                "variants": [],
                "current_variant_index": 0,
                "baseline_stats": current_stats,
                "switch_times": [],
                "performance_data": [],
            }

            for variant in variants:
                variant_payload = asdict(variant)
                variant_path = output_dir / f"{variant.variant_id}.jpg"
                variant_payload["file_path"] = str(variant_path)
                test_info["variants"].append(variant_payload)

            self.active_thumbnail_tests[video_id] = test_info

            upload_row = self._get_db_manager().get_upload_by_video_id(video_id)
            if upload_row:
                self._get_db_manager().start_thumbnail_ab_test(upload_row["id"])

            self.logger.info(
                f"Started thumbnail A/B test for video {video_id} with {len(variants)} variants"
            )

        except Exception as e:
            self.logger.error(f"Failed to start thumbnail A/B test: {e}")

    async def _check_specific_thumbnail_test(self, video_id: str):
        """Check and potentially switch thumbnail variant for active test"""
        try:
            test_info = self.active_thumbnail_tests[video_id]

            # Get current performance
            current_stats = await self.youtube_client.get_video_analytics(video_id)
            if not current_stats:
                return

            # Record current performance
            test_info["performance_data"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "variant_index": test_info["current_variant_index"],
                    "stats": current_stats,
                }
            )

            # Check if it's time to switch variants or conclude test
            start_time = datetime.fromisoformat(test_info["start_time"])
            hours_elapsed = (datetime.now() - start_time).total_seconds() / 3600

            # Switch variant every 8 hours, or conclude after 24 hours
            if hours_elapsed >= self.thumbnail_test_duration:
                finalized = await self._finalize_thumbnail_test(
                    video_id, test_info, current_stats
                )
                if not finalized:
                    self.logger.info(
                        "Thumbnail test for %s not finalized yet; waiting for thresholds/data",
                        video_id,
                    )
            elif hours_elapsed >= (len(test_info["switch_times"]) + 1) * 8:
                await self._switch_thumbnail_variant(video_id)

        except Exception as e:
            self.logger.error(f"Thumbnail test check failed for {video_id}: {e}")

    async def _switch_thumbnail_variant(self, video_id: str):
        """Switch to next thumbnail variant in A/B test"""
        try:
            test_info = self.active_thumbnail_tests[video_id]
            variants = test_info["variants"]

            # Move to next variant
            current_index = test_info["current_variant_index"]
            next_index = (current_index + 1) % len(variants)

            # Upload new thumbnail
            variant = variants[next_index]
            thumbnail_path = Path(variant["file_path"])  # Would need to track this

            if thumbnail_path.exists():
                success = await self.youtube_client.update_video_thumbnail(
                    video_id, str(thumbnail_path)
                )

                if success:
                    test_info["current_variant_index"] = next_index
                    test_info["switch_times"].append(datetime.now().isoformat())

                    self.logger.info(
                        f"Switched to thumbnail variant {next_index} for video {video_id}"
                    )

        except Exception as e:
            self.logger.error(f"Thumbnail variant switch failed: {e}")

    async def _conclude_thumbnail_test(self, video_id: str):
        """Conclude thumbnail A/B test and select winner"""
        try:
            test_info = self.active_thumbnail_tests.get(video_id)
            if not test_info:
                return

            current_stats = await self.youtube_client.get_video_analytics(video_id)
            await self._finalize_thumbnail_test(video_id, test_info, current_stats)

        except Exception as e:
            self.logger.error(f"Thumbnail test conclusion failed: {e}")

    def _analyze_thumbnail_test_results(
        self, test_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze A/B test results to determine winning thumbnail"""
        try:
            performance_data = test_info["performance_data"]
            variants = test_info["variants"]

            if len(performance_data) < 2:
                return None

            # Group performance by variant
            variant_performance = {}
            for data_point in performance_data:
                variant_index = data_point["variant_index"]
                stats = data_point["stats"]

                if variant_index not in variant_performance:
                    variant_performance[variant_index] = []

                impressions = int(stats.get("impressions", 0) or 0)
                ctr = float(stats.get("ctr", 0) or 0)
                if ctr <= 0 and impressions > 0:
                    clicks = int(stats.get("clicks", 0) or 0)
                    ctr = clicks / impressions if impressions > 0 else 0
                ctr = max(0.0, min(1.0, ctr))

                variant_performance[variant_index].append(ctr)

            # Calculate average CTR for each variant
            variant_ctrs = {}
            for variant_index, ctrs in variant_performance.items():
                if len(ctrs) >= self.min_variant_samples:
                    variant_ctrs[variant_index] = sum(ctrs) / len(ctrs)

            if not variant_ctrs:
                return None

            # Find best performing variant
            winner_index = max(variant_ctrs, key=variant_ctrs.get)
            winner_ctr = variant_ctrs[winner_index]

            # Calculate improvement over baseline
            baseline_ctr = variant_ctrs.get(0, 0)  # Assume first variant is baseline
            improvement = (
                ((winner_ctr - baseline_ctr) / baseline_ctr * 100)
                if baseline_ctr > 0
                else 0
            )

            # Calculate confidence (simplified)
            confidence = min(
                95, len(performance_data) * 10
            )  # More data = higher confidence

            total_impressions = sum(
                d["stats"].get("impressions", 0) for d in performance_data
            )

            return {
                "winner_index": winner_index,
                "confidence": confidence,
                "improvement": improvement,
                "total_impressions": total_impressions,
                "variant_ctrs": variant_ctrs,
            }

        except Exception as e:
            self.logger.error(f"Thumbnail test analysis failed: {e}")
            return None

    def _store_thumbnail_test_results(self, test_result: ThumbnailTestResult):
        """Store thumbnail A/B test results for future optimization"""
        try:
            results_dir = Path("thumbnail_test_results")
            results_dir.mkdir(exist_ok=True)

            result_file = results_dir / f"{test_result.test_id}.json"
            with open(result_file, "w") as f:
                json.dump(asdict(test_result), f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to store test results: {e}")

    async def _check_thumbnail_tests(self):
        """Check all active thumbnail tests"""
        for video_id in list(self.active_thumbnail_tests.keys()):
            try:
                await self._check_specific_thumbnail_test(video_id)
            except Exception as e:
                self.logger.error(f"Error checking thumbnail test for {video_id}: {e}")

    async def run_thumbnail_winner_selection_job(self):
        """Evaluate active thumbnail tests and finalize winners when thresholds are met."""
        for video_id in list(self.active_thumbnail_tests.keys()):
            try:
                test_info = self.active_thumbnail_tests.get(video_id)
                if not test_info:
                    continue

                current_stats = await self.youtube_client.get_video_analytics(video_id)
                await self._finalize_thumbnail_test(video_id, test_info, current_stats)
            except Exception as e:
                self.logger.error("Winner selection job failed for %s: %s", video_id, e)

    async def _update_video_metrics(self, video_id: str, video_info: Dict[str, Any]):
        """Update video performance metrics"""
        try:
            # Get current analytics data
            analytics = await self.youtube_client.get_video_analytics(video_id)

            if analytics:
                # Create performance metrics
                metrics = PerformanceMetrics(
                    video_id=video_id,
                    views=analytics.get("views", 0),
                    likes=analytics.get("likes", 0),
                    comments=analytics.get("comments", 0),
                    shares=analytics.get("shares", 0),
                    watch_time_percentage=analytics.get("average_view_percentage", 0),
                    click_through_rate=analytics.get("ctr", 0),
                )

                # Store metrics
                self.engagement_monitor.update_metrics_from_youtube(video_id, analytics)

        except Exception as e:
            self.logger.error(f"Metrics update failed for {video_id}: {e}")

    async def _finalize_thumbnail_test(
        self,
        video_id: str,
        test_info: Dict[str, Any],
        current_stats: Optional[Dict[str, Any]],
    ) -> bool:
        """Finalize test when safe thresholds are met and winner is clear."""
        ready, reason = self._evaluate_thumbnail_test_readiness(
            test_info, current_stats
        )
        if not ready:
            self.logger.info("Skipping winner selection for %s: %s", video_id, reason)
            return False

        winner_analysis = self._analyze_thumbnail_test_results(test_info)
        if not winner_analysis:
            self.logger.info(
                "Skipping winner selection for %s: insufficient performance data",
                video_id,
            )
            return False

        winner_index = winner_analysis["winner_index"]
        variants = test_info.get("variants", [])
        if winner_index >= len(variants):
            self.logger.info(
                "Skipping winner selection for %s: winner index out of bounds", video_id
            )
            return False

        winner_variant = variants[winner_index]
        raw_thumbnail_path = winner_variant.get("file_path")
        if not raw_thumbnail_path:
            self.logger.info(
                "Skipping winner selection for %s: winner thumbnail file missing at %s",
                video_id,
                raw_thumbnail_path,
            )
            return False

        thumbnail_path = Path(raw_thumbnail_path)
        if not thumbnail_path.exists() or not thumbnail_path.is_file():
            self.logger.info(
                "Skipping winner selection for %s: winner thumbnail file missing at %s",
                video_id,
                thumbnail_path,
            )
            return False

        updated = await self.youtube_client.update_video_thumbnail(
            video_id, str(thumbnail_path)
        )
        if not updated:
            self.logger.info(
                "Skipping winner selection for %s: YouTube thumbnail update failed",
                video_id,
            )
            return False

        winner_variant_id = winner_variant.get(
            "variant_id", f"variant_{winner_index + 1}"
        )
        winner_label = self._variant_label_for_index(winner_index)
        self._archive_loser_variants(variants, winner_index)
        self._record_thumbnail_test_completion(video_id, winner_label, winner_analysis)

        test_result = ThumbnailTestResult(
            test_id=f"{video_id}_{int(datetime.now().timestamp())}",
            video_id=video_id,
            variants_tested=[v.get("variant_id", "") for v in variants],
            winner_variant=winner_variant_id,
            confidence_level=winner_analysis["confidence"],
            performance_improvement=winner_analysis["improvement"],
            test_duration_hours=self.thumbnail_test_duration,
            total_impressions=winner_analysis["total_impressions"],
        )
        self._store_thumbnail_test_results(test_result)

        self.logger.info(
            "Concluded thumbnail test for %s. Winner: %s (label %s)",
            video_id,
            winner_variant_id,
            winner_label,
        )
        self.active_thumbnail_tests.pop(video_id, None)
        return True

    def _evaluate_thumbnail_test_readiness(
        self,
        test_info: Dict[str, Any],
        current_stats: Optional[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """Determine if test has enough time, impressions, and data to pick a winner."""
        if not test_info.get("start_time"):
            return False, "missing start_time"

        if not current_stats:
            return False, "missing current analytics stats"

        start_time = datetime.fromisoformat(test_info["start_time"])
        hours_elapsed = (datetime.now() - start_time).total_seconds() / 3600
        if hours_elapsed < self.thumbnail_test_duration:
            return (
                False,
                f"duration threshold not met ({hours_elapsed:.1f}h/{self.thumbnail_test_duration}h)",
            )

        impressions = int(current_stats.get("impressions", 0) or 0)
        if impressions < self.min_impressions_for_test:
            return False, (
                f"impression threshold not met ({impressions}/{self.min_impressions_for_test})"
            )

        variants = test_info.get("variants", [])
        if len(variants) < 2:
            return False, "less than two variants available"

        performance_data = test_info.get("performance_data", [])
        if len(performance_data) < len(variants):
            return False, "insufficient performance samples across variants"

        return True, "ready"

    def _archive_loser_variants(
        self, variants: List[Dict[str, Any]], winner_index: int
    ):
        """Archive non-winning thumbnail variant files."""
        if not variants:
            return

        winner = variants[winner_index]
        winner_path = Path(winner.get("file_path", ""))
        if not winner_path.parent.exists():
            return

        archive_dir = (
            winner_path.parent / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        archive_dir.mkdir(parents=True, exist_ok=True)

        for idx, variant in enumerate(variants):
            if idx == winner_index:
                continue

            file_path = Path(variant.get("file_path", ""))
            if file_path.exists():
                shutil.move(str(file_path), str(archive_dir / file_path.name))

            variant_id = variant.get("variant_id")
            if variant_id:
                config_path = file_path.parent / f"{variant_id}_config.json"
                if config_path.exists():
                    shutil.move(str(config_path), str(archive_dir / config_path.name))

    def _record_thumbnail_test_completion(
        self,
        video_id: str,
        winner_label: str,
        winner_analysis: Dict[str, Any],
    ):
        """Persist winner and CTR details where upload mapping exists."""
        db_manager = self._get_db_manager()
        upload_row = db_manager.get_upload_by_video_id(video_id)
        if not upload_row:
            self.logger.info(
                "No upload row found for %s; winner persisted only in local test result",
                video_id,
            )
            return

        upload_id = upload_row["id"]
        variant_ctrs = winner_analysis.get("variant_ctrs", {})
        if 0 in variant_ctrs:
            db_manager.update_thumbnail_ctr(upload_id, "A", float(variant_ctrs[0]))
        if 1 in variant_ctrs:
            db_manager.update_thumbnail_ctr(upload_id, "B", float(variant_ctrs[1]))

        db_manager.complete_thumbnail_ab_test(upload_id, winner_label)

    def _variant_label_for_index(self, index: int) -> str:
        """Map 0 -> A, 1 -> B, 2 -> C, ... for DB winner storage."""
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if index < 0:
            return "A"
        if index < len(alphabet):
            return alphabet[index]
        return f"V{index + 1}"

    def _get_db_manager(self) -> DatabaseManager:
        return get_db_manager()

    def _get_video_path(self, video_id: str) -> Optional[Path]:
        """Get local path for video file using database mapping."""
        if not video_id:
            self.logger.warning("Cannot resolve local video path: empty video_id")
            return None

        try:
            db_manager = self._get_db_manager()

            artifacts = db_manager.get_local_artifacts_by_video_id(video_id)
            if not artifacts:
                self.logger.info(f"No local artifacts found for video ID: {video_id}")
                return None

            local_video_path = artifacts.get("local_video_path")
            if not local_video_path:
                self.logger.warning(
                    f"Missing local_video_path in artifact mapping for {video_id}"
                )
                return None

            resolved_path = Path(local_video_path)
            if not resolved_path.exists():
                self.logger.warning(
                    f"Mapped local video path does not exist for {video_id}: {resolved_path}"
                )
                return None

            return resolved_path

        except Exception as e:
            self.logger.error(f"Failed to get local video path for {video_id}: {e}")
            return None

    def _create_analysis_from_video_info(self, video_info: Dict[str, Any]):
        """Create analysis object from video info for thumbnail generation"""
        # This would create a properly structured analysis object
        # For now, return a minimal structure
        from src.models import (
            VideoAnalysisEnhanced,
            ThumbnailInfo,
            HookMoment,
            AudioHook,
            VideoSegment,
            CallToAction,
        )

        return VideoAnalysisEnhanced(
            suggested_title=video_info.get("title", "Video"),
            summary_for_description=video_info.get("description", "")[:100],
            mood="exciting",
            has_clear_narrative=True,
            original_audio_is_key=False,
            hook_text="Must Watch!",
            hook_variations=["Amazing!", "Incredible!", "You Won't Believe!"],
            visual_hook_moment=HookMoment(
                timestamp_seconds=10.0, description="Hook moment"
            ),
            audio_hook=AudioHook(
                type="dramatic", sound_name="whoosh", timestamp_seconds=5.0
            ),
            best_segment=VideoSegment(
                start_seconds=0, end_seconds=30, reason="Best part"
            ),
            segments=[
                VideoSegment(start_seconds=0, end_seconds=30, reason="Main segment")
            ],
            music_genres=["upbeat"],
            hashtags=["#shorts", "#viral"],
            thumbnail_info=ThumbnailInfo(
                timestamp_seconds=10.0, reason="Good frame", headline_text="Must Watch!"
            ),
            call_to_action=CallToAction(text="Subscribe!", type="subscribe"),
        )

    def get_management_summary(self) -> Dict[str, Any]:
        """Get summary of channel management activities"""
        try:
            return {
                "active_thumbnail_tests": len(self.active_thumbnail_tests),
                "comments_analyzed_today": len(
                    [
                        cid
                        for cid, data in self.comment_interaction_history.items()
                        if data.get("last_analyzed")
                        and (
                            datetime.now()
                            - datetime.fromisoformat(data["last_analyzed"])
                        ).days
                        == 0
                    ]
                ),
                "management_status": "active",
                "last_cycle_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return {"status": "error"}
