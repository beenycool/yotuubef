"""
NVIDIA NIM AI Client Integration for Enhanced Video Analysis and Processing
Handles AI-powered content analysis, comment processing, and optimization suggestions using NVIDIA NIM API.
"""

import logging
import json
import asyncio
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.config.settings import get_config
from src.models import VideoAnalysisEnhanced


class RateLimiter:
    """Rate limiter for API requests"""

    def __init__(self, max_requests_per_minute: int = 60):
        self.max_rpm = max_requests_per_minute
        self.requests_this_minute = []
        self.last_reset = time.time()
        self._lock = asyncio.Lock()

    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        while True:
            async with self._lock:
                current_time = time.time()

                # Reset every minute
                if current_time - self.last_reset >= 60:
                    self.requests_this_minute = []
                    self.last_reset = current_time

                # Clean old requests
                minute_ago = current_time - 60
                self.requests_this_minute = [
                    t for t in self.requests_this_minute if t > minute_ago
                ]

                if len(self.requests_this_minute) < self.max_rpm:
                    self.requests_this_minute.append(current_time)
                    return

                wait_time = 60 - (current_time - self.requests_this_minute[0])

            if wait_time > 0:
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(0)


class NvidiaNimAIClient:
    """
    NVIDIA NIM AI client for enhanced video analysis and processing tasks
    Uses OpenAI SDK with NVIDIA NIM backend
    """

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        # Initialize NVIDIA NIM client if available
        if OPENAI_AVAILABLE and hasattr(self.config.api, "nvidia_nim_api_key"):
            api_key = self.config.api.nvidia_nim_api_key
            base_url = getattr(
                self.config.api,
                "nvidia_nim_base_url",
                "https://integrate.api.nvidia.com/v1",
            )

            if api_key:
                self.client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
                self.model = getattr(
                    self.config.api, "nvidia_nim_model", "qwen/qwen3.5-397b-a17b"
                )
                self.alt_model = getattr(
                    self.config.api, "nvidia_nim_alt_model", "moonshotai/kimi-k2-5"
                )
                self.nim_available = True

                # Setup rate limiting
                rpm_limit = getattr(self.config.api, "nvidia_nim_rate_limit_rpm", 60)
                self.rate_limiter = RateLimiter(rpm_limit)
            else:
                self.nim_available = False
                self.client = None
                self.logger.warning(
                    "NVIDIA NIM API key not configured - some AI features will be limited"
                )
        else:
            self.nim_available = False
            self.client = None
            self.logger.warning(
                "NVIDIA NIM not available - some AI features will be limited"
            )

        # AI analysis prompts
        self.video_analysis_prompt = """
        You are a YouTube Shorts scriptwriter. Write a 45-second script based on this Reddit lore and Research context.

        Reddit Post: {title} - {description}
        Research Facts: {deep_research}

        Follow this format for high retention:
        1. THE HOOK (0-3s): Start with a shocking fact.
        2. THE STORY (3-35s): Fast-paced explanation using the research.
        3. THE OUTRO (35-45s): Quick conclusion.

        CRITICAL RETENTION RULE - THE PERFECT LOOP:
        The very last sentence of the video must grammatically and sonically lead 
        directly into the very first sentence. This creates an endless loop that 
        tricks viewers into rewatching the first 3 seconds.
        
        Example:
        First sentence: "the smartest player in Geometry Dash history."
        Last sentence: "And that is exactly why nobody remembers..."
        (Plays as: "And that is exactly why nobody remembers... the smartest player in Geometry Dash history.")

        Output MUST be valid JSON matching this exact structure:
        {{
            "suggested_title": "Optimized Title",
            "summary_for_description": "Short description",
            "mood": "mysterious",
            "hashtags": ["#lore", "#mystery"],
            "narrative_script_segments": [
                {{
                    "text": "Your hook here",
                    "time_seconds": 0.0,
                    "intended_duration_seconds": 3.0,
                    "b_roll_search_query": "specific image search query for visual evidence"
                }},
                {{
                    "text": "The next sentence",
                    "time_seconds": 3.0,
                    "intended_duration_seconds": 5.0,
                    "b_roll_search_query": "image query for this segment"
                }}
            ],
            "text_overlays": [
                {{"text": "CAPTION 1", "timestamp_seconds": 0.0, "duration": 3.0}}
            ],
            "loop_bridge_text": "text that connects end to beginning"
        }}
        """

        self.comment_analysis_prompt = """
        Analyze this YouTube comment for engagement potential and required actions:
        
        Comment: "{comment_text}"
        Video Context: {video_context}
        
        Provide analysis in this format:
        Engagement Score: [0-100]
        Sentiment: [positive/negative/neutral]
        Toxicity Score: [0-100]
        Reply Urgency: [low/medium/high]
        Should Pin: [true/false]
        Should Heart: [true/false]
        Suggested Response: [appropriate response text]
        Interaction Priority: [1-10]
        Comment Type: [question/praise/criticism/spam/other]
        """

    async def analyze_video_content(
        self, video_path: Path, reddit_content: Any
    ) -> Optional[VideoAnalysisEnhanced]:
        """
        Analyze video content for optimization opportunities

        Args:
            video_path: Path to video file
            reddit_content: Original Reddit content data

        Returns:
            Enhanced video analysis or None if failed
        """
        try:
            self.logger.info("Starting NVIDIA NIM video content analysis...")

            # Extract video metadata
            video_metadata = self._extract_video_metadata(video_path)

            # Prepare analysis context
            if (
                hasattr(reddit_content, "__dict__")
                and hasattr(reddit_content, "title")
                and not hasattr(reddit_content, "get")
            ):
                analysis_context = {
                    "title": getattr(reddit_content, "title", ""),
                    "description": getattr(reddit_content, "selftext", ""),
                    "duration": video_metadata.get("duration", 60),
                    "subreddit": getattr(reddit_content, "subreddit", ""),
                    "score": getattr(reddit_content, "score", 0),
                    "num_comments": getattr(reddit_content, "num_comments", 0),
                    "deep_research": getattr(reddit_content, "deep_research", "")
                    if hasattr(reddit_content, "deep_research")
                    else "",
                }
            else:
                analysis_context = {
                    "title": reddit_content.get("title", "")
                    if hasattr(reddit_content, "get")
                    else "",
                    "description": reddit_content.get("selftext", "")
                    if hasattr(reddit_content, "get")
                    else "",
                    "duration": video_metadata.get("duration", 60),
                    "subreddit": reddit_content.get("subreddit", "")
                    if hasattr(reddit_content, "get")
                    else "",
                    "score": reddit_content.get("score", 0)
                    if hasattr(reddit_content, "get")
                    else 0,
                    "num_comments": reddit_content.get("num_comments", 0)
                    if hasattr(reddit_content, "get")
                    else 0,
                    "deep_research": reddit_content.get("deep_research", "")
                    if hasattr(reddit_content, "get")
                    else "",
                }

            # Perform AI analysis
            if self.nim_available:
                analysis_result = await self._analyze_with_nim(analysis_context)
            else:
                analysis_result = self._analyze_with_fallback(analysis_context)

            if not analysis_result:
                return None

            # Convert to VideoAnalysisEnhanced
            enhanced_analysis = self._convert_to_enhanced_analysis(
                analysis_result, reddit_content
            )

            self.logger.info("NVIDIA NIM video content analysis completed")
            return enhanced_analysis

        except Exception as e:
            self.logger.error(f"NVIDIA NIM video analysis failed: {e}")
            return None

    async def analyze_comment_engagement(
        self,
        comment_text: str,
        video_context: Dict[str, Any],
        analysis_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """
        Analyze comment for engagement potential

        Args:
            comment_text: The comment to analyze
            video_context: Context about the video
            analysis_prompt: Analysis prompt template (optional)

        Returns:
            AI analysis response or None if failed
        """
        try:
            if self.nim_available:
                return await self._analyze_comment_with_nim(
                    comment_text, video_context, analysis_prompt
                )
            else:
                return self._analyze_comment_with_fallback(comment_text, video_context)

        except Exception as e:
            self.logger.error(f"Comment analysis failed: {e}")
            return None

    def _extract_video_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Extract basic video metadata"""
        try:
            from moviepy import VideoFileClip

            with VideoFileClip(str(video_path)) as clip:
                return {
                    "duration": clip.duration,
                    "fps": clip.fps,
                    "size": clip.size,
                    "has_audio": clip.audio is not None,
                }

        except Exception as e:
            self.logger.warning(f"Video metadata extraction failed: {e}")
            return {"duration": 60, "fps": 30, "size": (1920, 1080), "has_audio": True}

    async def _analyze_with_nim(
        self, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze content using NVIDIA NIM API"""
        if not self.client:
            self.logger.error("NVIDIA NIM client is not configured or available")
            return None

        try:
            # Wait for rate limit if needed
            await self.rate_limiter.wait_if_needed()

            # Prepare prompt
            prompt = self.video_analysis_prompt.format(**context)

            self.logger.debug(f"Sending prompt to NVIDIA NIM: {prompt[:100]}...")

            # Make API call using OpenAI SDK with NVIDIA NIM backend
            response = await self._chat_completion_with_fallback(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )

            # Parse response
            content = response.choices[0].message.content
            self.logger.debug(f"NVIDIA NIM response: {content[:200]}...")

            # Try to extract JSON from response
            try:
                if content.strip():
                    parsed_response = json.loads(content)
                    if isinstance(parsed_response, dict):
                        return parsed_response
                    else:
                        self.logger.error(
                            f"Expected parsed response to be a dict, got {type(parsed_response)}"
                        )
                        return self._parse_nim_text_response(content, context)
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(
                    f"Failed to parse NVIDIA NIM JSON response directly: {e}",
                    exc_info=True,
                )
                # Fallback: look for JSON block in response
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1

                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    return json.loads(json_str)

            # If JSON parsing fails, create structured response from text
            return self._parse_nim_text_response(content, context)

        except Exception as e:
            self.logger.error(f"NVIDIA NIM analysis failed: {e}")
            return None

    def _parse_nim_text_response(
        self, response_text: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse NVIDIA NIM text response into structured format"""
        try:
            lines = response_text.split("\n")

            analysis = {
                "suggested_title": context.get("title", "Engaging Video"),
                "summary_for_description": (
                    context.get("description", "")
                    or "Reddit lore short with researched context."
                )[:200],
                "mood": "mysterious",
                "hashtags": ["#lore", "#mystery", "#shorts"],
                "narrative_script_segments": [
                    {
                        "text": "This Reddit mystery sounds fake, but the facts get wild fast.",
                        "time_seconds": 0.0,
                        "intended_duration_seconds": 4.0,
                    },
                    {
                        "text": "Here is what happened, and why people are still debating it.",
                        "time_seconds": 4.0,
                        "intended_duration_seconds": 6.0,
                    },
                    {
                        "text": "Drop your theory in the comments if you think this was real.",
                        "time_seconds": 10.0,
                        "intended_duration_seconds": 5.0,
                    },
                ],
                "text_overlays": [
                    {"text": "REDDIT LORE", "timestamp_seconds": 0.0, "duration": 3.0}
                ],
            }

            for line in lines:
                if "title:" in line.lower():
                    title_part = line.split(":", 1)[1].strip()
                    if title_part:
                        analysis["suggested_title"] = title_part[:60]
                        break

            return analysis

        except Exception as e:
            self.logger.warning(f"Response parsing failed: {e}")
            return {}

    def _analyze_with_fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when NVIDIA NIM is not available"""
        try:
            title = context.get("title", "Video")
            duration = context.get("duration", 60)
            score = context.get("score", 0)

            engagement_score = min(100, max(30, score // 10 + 50))

            mood = "exciting"
            title_lower = title.lower()
            if any(word in title_lower for word in ["calm", "peaceful", "relaxing"]):
                mood = "calm"
            elif any(
                word in title_lower for word in ["dramatic", "intense", "serious"]
            ):
                mood = "dramatic"
            elif any(word in title_lower for word in ["funny", "hilarious", "comedy"]):
                mood = "humorous"

            hook_base = "You won't believe this!"
            if "how to" in title_lower:
                hook_base = "Learn this amazing trick!"
            elif any(word in title_lower for word in ["secret", "hidden", "revealed"]):
                hook_base = "The secret is finally revealed!"

            analysis = {
                "suggested_title": self._optimize_title(title),
                "summary_for_description": (
                    context.get("description", "")
                    or "Reddit lore story explained with research."
                )[:200],
                "mood": mood,
                "hashtags": self._generate_hashtags(
                    title, context.get("subreddit", "")
                ),
                "narrative_script_segments": [
                    {
                        "text": hook_base,
                        "time_seconds": 0.0,
                        "intended_duration_seconds": 3.0,
                    },
                    {
                        "text": (context.get("description", "") or title)[:220],
                        "time_seconds": 3.0,
                        "intended_duration_seconds": min(
                            20.0, max(8.0, duration * 0.5)
                        ),
                    },
                    {
                        "text": "Follow for more Reddit mysteries and hidden stories.",
                        "time_seconds": 30.0,
                        "intended_duration_seconds": 6.0,
                    },
                ],
                "text_overlays": [
                    {"text": "UNSOLVED?", "timestamp_seconds": 0.0, "duration": 2.5}
                ],
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Fallback analysis failed: {e}")
            return {}

    def _optimize_title(self, original_title: str) -> str:
        """Optimize title for YouTube engagement"""
        try:
            title = (
                original_title.replace("[OC]", "")
                .replace("[Original Content]", "")
                .strip()
            )

            engagement_words = ["Amazing", "Incredible", "Shocking", "Unbelievable"]
            title_lower = title.lower()

            if not any(word.lower() in title_lower for word in engagement_words):
                if len(title) < 50:
                    title = f"Amazing: {title}"

            if len(title) > 60:
                title = title[:57] + "..."

            return title

        except Exception as e:
            self.logger.warning(f"Title optimization failed: {e}")
            return original_title

    def _generate_key_moments(self, duration: float) -> List[Dict[str, Any]]:
        """Generate key moments based on video duration"""
        moments = []

        moments.append({"timestamp": 2.0, "description": "Opening hook moment"})

        if duration > 20:
            moments.append(
                {"timestamp": duration * 0.4, "description": "Main content peak"}
            )

        if duration > 30:
            moments.append(
                {"timestamp": duration * 0.8, "description": "Closing impact"}
            )

        return moments

    def _generate_hashtags(self, title: str, subreddit: str) -> List[str]:
        """Generate relevant hashtags"""
        hashtags = ["#shorts", "#viral"]

        if subreddit:
            hashtags.append(f"#{subreddit.lower()}")

        title_lower = title.lower()

        if any(word in title_lower for word in ["funny", "comedy", "hilarious"]):
            hashtags.extend(["#funny", "#comedy"])
        elif any(word in title_lower for word in ["amazing", "incredible", "wow"]):
            hashtags.extend(["#amazing", "#mindblowing"])
        elif any(word in title_lower for word in ["how to", "tutorial", "guide"]):
            hashtags.extend(["#howto", "#tutorial"])
        elif any(word in title_lower for word in ["reaction", "responds"]):
            hashtags.extend(["#reaction", "#response"])

        hashtags.extend(["#trending", "#fyp", "#explore"])

        return hashtags[:10]

    async def _analyze_comment_with_nim(
        self,
        comment_text: str,
        video_context: Dict[str, Any],
        analysis_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Analyze comment using NVIDIA NIM"""
        try:
            await self.rate_limiter.wait_if_needed()

            prompt = analysis_prompt or self.comment_analysis_prompt

            full_prompt = prompt.format(
                comment_text=comment_text,
                video_context=json.dumps(video_context, indent=2),
            )

            response = await self._chat_completion_with_fallback(
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.3,
                max_tokens=800,
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"NVIDIA NIM comment analysis failed: {e}")
            return None

    async def analyze_comments_batch(self, comments: list, video_context: dict) -> list:
        """Analyze a batch of comments using NVIDIA NIM"""
        if not self.client:
            self.logger.error("NVIDIA NIM client is not configured or available")
            return []

        try:
            await self.rate_limiter.wait_if_needed()

            prompt = f"""Analyze these YouTube comments for engagement potential.
Video Context: {json.dumps(video_context)}

Comments to analyze:
{json.dumps(comments, indent=2)}

Return a JSON array of the top 3 most engaging comments. Each object in the array should have:
- comment_id: string
- interaction_priority: integer (0-100)
- suggested_reply: string
- sentiment: string (positive/negative/neutral)
- category: string (question/feedback/praise/complaint)
"""
            response = await self._chat_completion_with_fallback(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )

            try:
                content = response.choices[0].message.content
                if content.strip():
                    parsed_response = json.loads(content)
                    if isinstance(parsed_response, list):
                        return parsed_response
                    if isinstance(parsed_response, dict):
                        for key in ("comments", "results", "items", "data"):
                            value = parsed_response.get(key)
                            if isinstance(value, list):
                                return value
                    self.logger.error(
                        f"Expected parsed response to include a list, got {type(parsed_response)}"
                    )
                    return []
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Failed to parse batch comment analysis JSON: {e}")

        except Exception as e:
            self.logger.error("Batch comment analysis failed", exc_info=True)

        return []

    async def _chat_completion_with_fallback(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, str]] = None,
    ):
        """Execute chat completion with primary model and fallback model retry."""
        if not self.client:
            raise RuntimeError("NVIDIA NIM client is not configured")

        models_to_try = [self.model]
        if self.alt_model and self.alt_model != self.model:
            models_to_try.append(self.alt_model)

        last_error = None
        for model_name in models_to_try:
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format:
                    request_kwargs["response_format"] = response_format

                response = await self.client.chat.completions.create(**request_kwargs)
                if model_name != self.model:
                    self.logger.warning(
                        "NVIDIA NIM request succeeded with fallback model: %s",
                        model_name,
                    )
                return response
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "NVIDIA NIM model %s failed, trying next model if available: %s",
                    model_name,
                    exc,
                )

        raise RuntimeError(f"All NVIDIA NIM models failed: {last_error}")

    def _analyze_comment_with_fallback(
        self, comment_text: str, video_context: Dict[str, Any]
    ) -> str:
        """Fallback comment analysis"""
        try:
            text_lower = comment_text.lower()

            engagement_score = 50

            if any(
                word in text_lower for word in ["great", "amazing", "love", "awesome"]
            ):
                engagement_score += 20

            if "?" in comment_text:
                engagement_score += 15

            if any(
                phrase in text_lower
                for phrase in ["i think", "my experience", "i tried"]
            ):
                engagement_score += 25

            positive_words = ["great", "amazing", "love", "awesome", "fantastic"]
            negative_words = ["hate", "terrible", "awful", "boring"]

            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)

            if positive_count > negative_count:
                sentiment = "positive"
            elif negative_count > positive_count:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            response_parts = [
                f"Engagement Score: {engagement_score}",
                f"Sentiment: {sentiment}",
                "Toxicity Score: 5"
                if sentiment != "negative"
                else "Toxicity Score: 25",
                "Reply Urgency: medium"
                if engagement_score > 70
                else "Reply Urgency: low",
                f"Should Pin: {'true' if engagement_score > 80 and sentiment == 'positive' else 'false'}",
                f"Should Heart: {'true' if engagement_score > 60 and sentiment != 'negative' else 'false'}",
                "Suggested Response: Thanks for your comment! 👍",
                f"Interaction Priority: {min(10, max(1, engagement_score // 10))}",
                "Comment Type: other",
            ]

            return "\n".join(response_parts)

        except Exception as e:
            self.logger.error(f"Fallback comment analysis failed: {e}")
            return "Engagement Score: 50\nSentiment: neutral\nInteraction Priority: 5"

    def _convert_to_enhanced_analysis(
        self, analysis_result: Dict[str, Any], reddit_content: Any
    ) -> VideoAnalysisEnhanced:
        """Convert NVIDIA NIM analysis result to VideoAnalysisEnhanced model"""
        try:
            from src.models import (
                HookMoment,
                AudioHook,
                VideoSegment,
                ThumbnailInfo,
                CallToAction,
                AudioDuckingConfig,
                NarrativeSegment,
                TextOverlay,
            )

            raw_narrative_segments = analysis_result.get(
                "narrative_script_segments", []
            )
            narrative_segments: List[NarrativeSegment] = []
            b_roll_moments = []

            for item in raw_narrative_segments:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                if not text:
                    continue

                b_roll_query = item.get("b_roll_search_query")
                narrative_segments.append(
                    NarrativeSegment(
                        text=text,
                        time_seconds=float(item.get("time_seconds", 0.0)),
                        intended_duration_seconds=float(
                            item.get("intended_duration_seconds", 4.0)
                        ),
                        b_roll_search_query=b_roll_query,
                    )
                )

                if b_roll_query:
                    b_roll_moments.append(
                        {
                            "search_query": b_roll_query,
                            "timestamp_seconds": float(item.get("time_seconds", 0.0)),
                            "duration": float(
                                item.get("intended_duration_seconds", 4.0)
                            ),
                        }
                    )

            if not narrative_segments:
                narrative_segments = [
                    NarrativeSegment(
                        text="Here is a wild Reddit mystery that gets even stranger with context.",
                        time_seconds=0.0,
                        intended_duration_seconds=4.0,
                    ),
                    NarrativeSegment(
                        text="These are the key facts people keep bringing up.",
                        time_seconds=4.0,
                        intended_duration_seconds=5.0,
                    ),
                ]

            segments = []
            for i, segment in enumerate(narrative_segments):
                end_seconds = segment.time_seconds + segment.intended_duration_seconds
                if i + 1 < len(narrative_segments):
                    next_start = narrative_segments[i + 1].time_seconds
                    end_seconds = max(end_seconds, next_start)
                segments.append(
                    VideoSegment(
                        start_seconds=segment.time_seconds,
                        end_seconds=end_seconds,
                        reason="Narrative segment",
                    )
                )

            visual_hook_moment = HookMoment(
                timestamp_seconds=narrative_segments[0].time_seconds
                if narrative_segments
                else 0.0,
                description="Opening hook",
            )

            audio_hook = AudioHook(
                type="dramatic",
                sound_name="whoosh",
                timestamp_seconds=visual_hook_moment.timestamp_seconds,
            )

            thumbnail_timestamp = visual_hook_moment.timestamp_seconds
            thumbnail_info = ThumbnailInfo(
                timestamp_seconds=thumbnail_timestamp,
                reason="Most engaging moment",
                headline_text=(
                    narrative_segments[0].text if narrative_segments else "Must Watch!"
                )[:80],
            )

            call_to_action = CallToAction(text="Subscribe for more!", type="subscribe")

            raw_overlays = analysis_result.get("text_overlays", [])
            text_overlays: List[TextOverlay] = []
            for item in raw_overlays:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                text_overlays.append(
                    TextOverlay(
                        text=text,
                        timestamp_seconds=float(item.get("timestamp_seconds", 0.0)),
                        duration=float(item.get("duration", 2.0)),
                    )
                )

            if (
                hasattr(reddit_content, "__dict__")
                and hasattr(reddit_content, "title")
                and not hasattr(reddit_content, "get")
            ):
                title = getattr(reddit_content, "title", "Video")
                description = getattr(reddit_content, "title", "Amazing content!")[:200]
            else:
                title = (
                    reddit_content.get("title", "Video")
                    if hasattr(reddit_content, "get")
                    else "Video"
                )
                description = (
                    reddit_content.get("selftext", "")
                    if hasattr(reddit_content, "get")
                    else ""
                )[:200] or "Amazing content!"

            enhanced_analysis = VideoAnalysisEnhanced(
                suggested_title=analysis_result.get("suggested_title", title),
                summary_for_description=analysis_result.get(
                    "summary_for_description", description
                ),
                mood=analysis_result.get("mood", "exciting"),
                has_clear_narrative=True,
                original_audio_is_key=False,
                hook_text=narrative_segments[0].text
                if narrative_segments
                else "Must watch!",
                hook_variations=["Amazing!", "Incredible!", "Must see!"],
                best_segment=segments[0]
                if segments
                else VideoSegment(
                    start_seconds=0, end_seconds=30, reason="Main content"
                ),
                segments=segments,
                text_overlays=text_overlays,
                narrative_script_segments=narrative_segments,
                visual_hook_moment=visual_hook_moment,
                audio_hook=audio_hook,
                thumbnail_info=thumbnail_info,
                call_to_action=call_to_action,
                music_genres=analysis_result.get(
                    "music_genres", ["upbeat", "electronic"]
                ),
                hashtags=analysis_result.get(
                    "hashtags",
                    analysis_result.get("suggested_hashtags", ["#shorts", "#viral"]),
                ),
                audio_ducking_config=AudioDuckingConfig(
                    duck_during_narration=True, duck_volume=0.3, fade_duration=0.5
                ),
                b_roll_moments=b_roll_moments,
                loop_bridge_text=analysis_result.get("loop_bridge_text"),
            )

            return enhanced_analysis

        except Exception as e:
            self.logger.error(f"Enhanced analysis conversion failed: {e}")
            return self._create_minimal_analysis(reddit_content)

    def _create_minimal_analysis(self, reddit_content: Any) -> VideoAnalysisEnhanced:
        """Create minimal analysis when full analysis fails"""
        try:
            from src.models import (
                HookMoment,
                AudioHook,
                VideoSegment,
                ThumbnailInfo,
                CallToAction,
                AudioDuckingConfig,
            )

            if (
                hasattr(reddit_content, "__dict__")
                and hasattr(reddit_content, "title")
                and not hasattr(reddit_content, "get")
            ):
                title = getattr(reddit_content, "title", "Engaging Video")[:60]
                description = "Check out this content!"
            else:
                title = (
                    reddit_content.get("title", "Engaging Video")
                    if hasattr(reddit_content, "get")
                    else "Engaging Video"
                )[:60]
                description = (
                    reddit_content.get("selftext", "")
                    if hasattr(reddit_content, "get")
                    else ""
                )[:200] or "Check out this content!"

            main_segment = VideoSegment(
                start_seconds=0, end_seconds=30, reason="Main content"
            )

            return VideoAnalysisEnhanced(
                suggested_title=title,
                summary_for_description=description,
                mood="exciting",
                has_clear_narrative=True,
                original_audio_is_key=False,
                hook_text="You won't believe this!",
                hook_variations=["Amazing!", "Incredible!", "Must see!"],
                best_segment=main_segment,
                segments=[main_segment],
                visual_hook_moment=HookMoment(
                    timestamp_seconds=5.0, description="Opening moment"
                ),
                audio_hook=AudioHook(
                    type="dramatic", sound_name="whoosh", timestamp_seconds=5.0
                ),
                thumbnail_info=ThumbnailInfo(
                    timestamp_seconds=5.0,
                    reason="Opening hook",
                    headline_text="Must Watch!",
                ),
                call_to_action=CallToAction(
                    text="Subscribe for more!", type="subscribe"
                ),
                music_genres=["upbeat", "electronic"],
                hashtags=["#shorts", "#viral", "#trending"],
                audio_ducking_config=AudioDuckingConfig(
                    duck_during_narration=True, duck_volume=0.3, fade_duration=0.5
                ),
            )

        except Exception as e:
            self.logger.error(f"Minimal analysis creation failed: {e}")
            raise
