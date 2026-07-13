"""
AI Client Integration for Enhanced Video Analysis and Processing
Handles AI-powered content analysis, comment processing, and optimization suggestions.
Uses NVIDIA NIM.
"""

import logging
import json
import asyncio
import random
import time
from functools import wraps
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.config.settings import get_config
from src.integrations.ai_provider import OpenAIProvider, ProviderRegistry
from src.models import (
    AudioDuckingConfig,
    AudioHook,
    CallToAction,
    HookMoment,
    NarrativeSegment,
    TextOverlay,
    ThumbnailInfo,
    VideoAnalysisEnhanced,
    VideoSegment,
)


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


def with_retry(max_attempts=3, base_delay=1.5, exceptions=(Exception,)):
    """Decorator: retry async function with exponential backoff."""

    def deco(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt == max_attempts - 1:
                        raise
                    delay = base_delay * (attempt + 1)
                    logging.getLogger(__name__).warning(
                        "Retry %d/%d for %s after %.1fs: %s",
                        attempt + 1,
                        max_attempts,
                        fn.__name__,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
            raise last_exc

        return wrapper

    return deco


class AIClient:
    """
    AI client for enhanced video analysis and processing tasks
    Uses NVIDIA NIM
    """

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        self._MOOD_WORDS_CALM = {"calm", "peaceful", "relaxing"}
        self._MOOD_WORDS_DRAMATIC = {"dramatic", "intense", "serious"}
        self._MOOD_WORDS_HUMOROUS = {"funny", "hilarious", "comedy"}
        self._HOOK_WORDS_SECRET = {"secret", "hidden", "revealed"}
        self._ENGAGEMENT_WORDS = {"amazing", "incredible", "shocking", "unbelievable"}
        self._HASHTAG_WORDS_FUNNY = {"funny", "comedy", "hilarious"}
        self._HASHTAG_WORDS_AMAZING = {"amazing", "incredible", "wow"}
        self._HASHTAG_WORDS_HOWTO = {"how to", "tutorial", "guide"}
        self._HASHTAG_WORDS_REACTION = {"reaction", "responds"}

        self._response_cache: Dict[str, Any] = {}
        self._max_cache_size = 128
        self._model_supports_vision: Optional[bool] = None

        # Backward-compatible active_client reference (orchestrator accesses it)
        self.active_client = self

        # Initialize provider registry for multi-provider support
        self.provider_registry = ProviderRegistry()

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
                    timeout=120.0,
                )
                self.model = getattr(
                    self.config.api, "nvidia_nim_model", "moonshotai/kimi-k2.6"
                )
                self.alt_model = getattr(self.config.api, "nvidia_nim_alt_model", None)
                self.nim_available = True
                self.ai_available = True

                # Register NVIDIA NIM as primary provider
                self.active_provider = OpenAIProvider(
                    api_key=api_key,
                    base_url=base_url,
                    model=self.model,
                    alt_model=self.alt_model,
                )
                self.provider_registry.register(self.active_provider, make_default=True)

                # Setup rate limiting
                rpm_limit = getattr(self.config.api, "nvidia_nim_rate_limit_rpm", 60)
                self.rate_limiter = RateLimiter(rpm_limit)
            else:
                self.nim_available = False
                self.ai_available = False
                self.client = None
                self.logger.warning(
                    "NVIDIA_NIM_API_KEY not set. "
                    "Get one from https://build.nvidia.com/ and set it in .env"
                )
        else:
            self.nim_available = False
            self.ai_available = False
            self.client = None
            self.logger.warning(
                "NVIDIA NIM not available - some AI features will be limited"
            )

        # AI analysis prompts
        self.video_analysis_prompt = """
        You are a top-tier YouTube Shorts scriptwriter specializing in gaming history and internet mysteries.
        Your goal is to write a 45-60 second script based on the provided Reddit lore and Research context.

        Reddit Post: {title} - {description}
        Research Facts: {deep_research}

        CRITICAL SCRIPTING FORMULA (Follow exactly):
        1. THE DIRECT HOOK (0-3s): Start with a direct, compelling question or statement about the mystery. NO fluff. Hook segment intended_duration_seconds <= 3.0.
        2. THE MISDIRECTION (3-10s): State what people *usually* think, then debunk it immediately using a specific fact, number, or date.
        3. THE BREADCRUMB TRAIL (10-35s): Walk the viewer through the detective work. Cite specific archival forums, exact dates, deleted tweets, or hidden IDs.
        4. THE REVEAL (35-45s): Reveal the final answer, referencing the visual evidence.
        5. THE PERFECT LOOP: The final sentence MUST grammatically flow back into the hook.
        WORD-COUNT LIMITS (per segment, based on intended_duration_seconds):
        - 3-5s: max 8 words | 6-8s: max 15 words | 9-12s: max 22 words

        FORBIDDEN PHRASES (DO NOT USE THESE):
        - "Did you know?"
        - "In today's video"
        - "Wait for it"
        - "Let's dive in"
        - "Mind-blowing"

        Output MUST be valid JSON matching this exact structure:
        {{
            "suggested_title": "Optimized Title",
            "summary_for_description": "Short description",
            "mood": "dramatic",
            "hashtags": ["#lore", "#mystery"],
            "narrative_script_segments":[
                {{
                    "text": "Your hook here",
                    "time_seconds": 0.0,
                    "intended_duration_seconds": 3.0,
                    "b_roll_search_query": "specific image query for evidence (e.g., 'Touch Arcade forum archive 2013')",
                    "expression_cue": "delivery direction, e.g. 'whispered, intense' or 'sharp, clinical'"
                }}
            ],
            "text_overlays":[
                {{"text": "CAPTION", "timestamp_seconds": 0.0, "duration": 3.0}}
            ],
            "loop_bridge_text": "text that connects the outro to the intro"
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

    def is_available(self) -> bool:
        """
        Check if AI services are available

        Returns:
            True if AI services are available, False otherwise
        """
        return self.ai_available

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status

        Returns:
            Rate limit information
        """
        if not self.ai_available or not self.client:
            return {"available": False}

        return {
            "available": True,
            "requests_this_minute": len(self.rate_limiter.requests_this_minute),
            "max_rpm": self.rate_limiter.max_rpm,
            "api_provider": "NVIDIA NIM",
        }

    # ── Video / content analysis ──────────────────────────────────────

    async def analyze_video_content(
        self, video_path: Optional[Path], reddit_content: Any
    ) -> Optional[VideoAnalysisEnhanced]:
        """
        Analyze video content for optimization opportunities using AI

        Args:
            video_path: Path to video file
            reddit_content: Original Reddit content data

        Returns:
            Enhanced video analysis or None if failed
        """
        try:
            self.logger.info("Starting AI video content analysis...")

            # Extract video metadata when a source file is available.
            if video_path is not None:
                video_metadata = await self._extract_video_metadata(video_path)
            else:
                video_metadata = {
                    "duration": 60,
                    "fps": 30,
                    "size": (1920, 1080),
                    "has_audio": True,
                }

            # Prepare analysis context
            analysis_context = {
                "title": self._get_content_field(reddit_content, "title", ""),
                "description": self._get_content_field(reddit_content, "selftext", ""),
                "duration": video_metadata.get("duration", 60),
                "subreddit": self._get_content_field(reddit_content, "subreddit", ""),
                "score": self._get_content_field(reddit_content, "score", 0),
                "num_comments": self._get_content_field(
                    reddit_content, "num_comments", 0
                ),
                "deep_research": self._get_content_field(
                    reddit_content, "deep_research", ""
                ),
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

            self.logger.info("AI video content analysis completed successfully")
            return enhanced_analysis

        except Exception as e:
            self.logger.error(f"AI video analysis failed: {e}")
            return None

    async def analyze_comment_engagement(
        self,
        comment_text: str,
        video_context: Dict[str, Any],
        analysis_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """
        Analyze comment for engagement potential using AI

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

    async def score_story_potential(self, context: Dict[str, Any]) -> int:
        """Evaluates a post's potential for a viral documentary short."""
        if not self.client:
            return 50

        try:
            await self.rate_limiter.wait_if_needed()
            prompt = f"""Evaluate this Reddit post's potential for a highly viral YouTube Short documentary.
            We need extreme stakes (massive financial loss, cheating scandals, unhinged drama, or eerie lost media).
            
            Title: {context.get("title")}
            Subreddit: {context.get("subreddit")}
            Content: {str(context.get("selftext"))[:1000]}
            
            Return ONLY a valid JSON object with a single 'score' integer between 0 and 100 representing its viral potential.
            Example: {{"score": 85}}
            """

            response = await self._chat_completion_with_fallback(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=50,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)
            score = int(parsed.get("score", 50))
            return max(0, min(100, score))
        except Exception as e:
            self.logger.warning(f"Failed to score story potential: {e}")
            return 50

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

Return a JSON object with a single key "comments" containing an array of the 3 most engaging comments.
Each object in the array should have:
- comment_id: string
- interaction_priority: integer (0-100)
- suggested_reply: string
- sentiment: string (positive/negative/neutral)
- category: string (question/feedback/praise/complaint)

Example:
{{"comments": [{{"comment_id": "abc123", "interaction_priority": 85, ...}}]}}
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

        except Exception:
            self.logger.error("Batch comment analysis failed", exc_info=True)

        return []

    # ── Content generation ────────────────────────────────────────────

    async def generate_content_suggestions(
        self, content_type: str, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate content suggestions using AI

        Args:
            content_type: Type of content to generate (title, description, tags, etc.)
            context: Context for content generation

        Returns:
            Generated content suggestions or None if failed
        """
        try:
            if not self.ai_available or not self.client:
                return None

            # Create a custom prompt based on content type
            prompts = {
                "title": """
                Generate 5 engaging YouTube Shorts titles based on this content:
                {content}
                
                Requirements:
                - Maximum 60 characters each
                - High engagement potential
                - YouTube algorithm friendly
                - Include relevant keywords
                
                Return as JSON array: ["title1", "title2", "title3", "title4", "title5"]
                """,
                "description": """
                Create an optimized YouTube Shorts description for:
                {content}
                
                Include:
                - Engaging opening
                - Key points summary
                - Call to action
                - Relevant hashtags
                - SEO keywords
                
                Maximum 200 characters.
                """,
                "hashtags": """
                Generate trending hashtags for this YouTube Shorts content:
                {content}
                
                Return 10-15 relevant hashtags including:
                - Content-specific tags
                - Trending general tags
                - Niche-specific tags
                - Platform tags (#shorts, #viral, etc.)
                
                Return as JSON array: ["#tag1", "#tag2", ...]
                """,
            }

            if content_type not in prompts:
                self.logger.warning(f"Unknown content type: {content_type}")
                return None

            prompt = prompts[content_type].format(content=json.dumps(context, indent=2))

            content_text = await self._run_generation_prompt(
                prompt,
                temperature=0.8,
                max_tokens=1000,
            )
            if not content_text:
                return None

            # Try to parse JSON response
            try:
                start_idx = content_text.find("[")
                end_idx = content_text.rfind("]") + 1

                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content_text[start_idx:end_idx]
                    return {"suggestions": json.loads(json_str)}
            except json.JSONDecodeError:
                pass

            # Return raw text if JSON parsing fails
            return {"suggestions": content_text}

        except Exception as e:
            self.logger.error(f"Content generation failed: {e}")
            return None

    async def analyze_performance_metrics(
        self, video_metrics: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze video performance metrics and provide optimization suggestions

        Args:
            video_metrics: Video performance data

        Returns:
            Performance analysis and optimization suggestions
        """
        try:
            if not self.ai_available or not self.client:
                return None

            prompt = f"""
            Analyze these YouTube Shorts performance metrics and provide optimization suggestions:

            Metrics: {json.dumps(video_metrics, indent=2)}

            Provide analysis in JSON format:
            {{
                "performance_score": 85,
                "strengths": ["High retention rate", "Good CTR"],
                "weaknesses": ["Low engagement", "Poor thumbnail performance"],
                "optimization_suggestions": [
                    "Improve thumbnail contrast and text readability",
                    "Add more engagement hooks in first 3 seconds"
                ],
                "trend_analysis": "Video shows declining performance after day 2",
                "next_video_recommendations": [
                    "Similar content with improved hooks",
                    "Test different thumbnail styles"
                ]
            }}
            """

            content = await self._run_generation_prompt(
                prompt,
                temperature=0.3,
                max_tokens=1500,
            )
            if not content:
                return None

            # Try to parse JSON response
            try:
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1

                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                pass

            return {"analysis": content}

        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            return None

    async def _run_generation_prompt(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Optional[str]:
        """Run a prompt against the NVIDIA NIM provider."""
        return await self._generate_with_nim(prompt, temperature, max_tokens)

    # ── Internal NIM methods ──────────────────────────────────────────

    def _check_model_supports_vision(self) -> bool:
        if self._model_supports_vision is not None:
            return self._model_supports_vision
        model_lower = (self.model or "").lower()
        self._model_supports_vision = any(
            kw in model_lower
            for kw in ("kimi", "llava", "vision", "gemini", "claude-3", "gpt-4v")
        )
        return self._model_supports_vision

    @staticmethod
    def _validate_script_safety(script: Any, config: Any) -> bool:
        content_config = getattr(config, "content", None) or getattr(
            config, "hard_disallowed", None
        )
        if content_config is None:
            return True
        hard_disallowed = getattr(content_config, "hard_disallowed", [])
        if not hard_disallowed:
            return True
        text = ""
        if isinstance(script, dict):
            text = " ".join(str(v) for v in script.values())
        elif isinstance(script, str):
            text = script
        else:
            text = str(script)
        lower = text.lower()
        for word in hard_disallowed:
            if word.lower() in lower:
                logging.getLogger(__name__).warning(
                    "Script contains disallowed word: %s", word
                )
                return False
        return True

    @staticmethod
    def _get_content_field(content: Any, field: str, default: Any = "") -> Any:
        """Extract a field from reddit content object or dict."""
        if hasattr(content, "get"):
            return content.get(field, default)
        return getattr(content, field, default)

    async def _extract_video_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Extract basic video metadata asynchronously"""
        from src.utils.video import extract_video_metadata

        return await extract_video_metadata(video_path)

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
                parsed_json = self._extract_json_object(content)
                if isinstance(parsed_json, dict):
                    return parsed_json

            # If JSON parsing fails, create structured response from text
            return self._parse_nim_text_response(content, context)

        except Exception as e:
            self.logger.error(f"NVIDIA NIM analysis failed: {e}")
            return None

    def _extract_json_object(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract the first decodable JSON object from mixed text."""
        decoder = json.JSONDecoder()
        start_idx = content.find("{")

        while start_idx >= 0:
            try:
                parsed, _ = decoder.raw_decode(content[start_idx:])
                if isinstance(parsed, dict):
                    return parsed
            except ValueError:
                pass
            start_idx = content.find("{", start_idx + 1)

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

            mood = "exciting"
            title_lower = title.lower()
            if any(word in title_lower for word in self._MOOD_WORDS_CALM):
                mood = "calm"
            elif any(word in title_lower for word in self._MOOD_WORDS_DRAMATIC):
                mood = "dramatic"
            elif any(word in title_lower for word in self._MOOD_WORDS_HUMOROUS):
                mood = "humorous"

            hook_base = "You won't believe this!"
            if "how to" in title_lower:
                hook_base = "Learn this amazing trick!"
            elif any(word in title_lower for word in self._HOOK_WORDS_SECRET):
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

            engagement_words = self._ENGAGEMENT_WORDS
            title_lower = title.lower()

            if not any(word in title_lower for word in engagement_words):
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

        if any(word in title_lower for word in self._HASHTAG_WORDS_FUNNY):
            hashtags.extend(["#funny", "#comedy"])
        elif any(word in title_lower for word in self._HASHTAG_WORDS_AMAZING):
            hashtags.extend(["#amazing", "#mindblowing"])
        elif any(word in title_lower for word in self._HASHTAG_WORDS_HOWTO):
            hashtags.extend(["#howto", "#tutorial"])
        elif any(word in title_lower for word in self._HASHTAG_WORDS_REACTION):
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

    def _make_cache_key(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        model: str,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        import hashlib

        text = json.dumps(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "model": model,
                "response_format": response_format,
            },
            sort_keys=True,
        )
        return hashlib.md5(text.encode()).hexdigest()

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
        if self.alt_model:
            models_to_try.append(self.alt_model)

        # Check cache
        for model_name in models_to_try:
            cache_key = self._make_cache_key(
                messages, temperature, max_tokens, model_name, response_format
            )
            cached = self._response_cache.get(cache_key)
            if cached is not None:
                self.logger.debug("NVIDIA NIM cache hit for model %s", model_name)
                return cached

        max_retries = 6
        base_delay = 1.0
        max_delay = 30.0
        last_error = None
        for model_name in models_to_try:
            for attempt in range(max_retries):
                try:
                    request_kwargs: Dict[str, Any] = {
                        "model": model_name,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if response_format:
                        request_kwargs["response_format"] = response_format

                    response = await self.client.chat.completions.create(
                        **request_kwargs
                    )
                    if model_name != self.model:
                        self.logger.warning(
                            "NVIDIA NIM request succeeded with fallback model: %s",
                            model_name,
                        )
                    # Cache response
                    cache_key = self._make_cache_key(
                        messages, temperature, max_tokens, model_name, response_format
                    )
                    if len(self._response_cache) >= self._max_cache_size:
                        self._response_cache.clear()
                    self._response_cache[cache_key] = response
                    return response
                except Exception as exc:
                    last_error = exc
                    error_str = str(exc).lower()
                    # Graceful response_format degradation
                    if (
                        response_format
                        and attempt == 0
                        and ("response_format" in error_str or "json" in error_str)
                    ):
                        self.logger.warning(
                            "Model %s does not support response_format, retrying without it",
                            model_name,
                        )
                        response_format = None
                        request_kwargs.pop("response_format", None)
                        try:
                            response = await self.client.chat.completions.create(
                                **request_kwargs
                            )
                            return response
                        except Exception:
                            pass

                    if attempt < max_retries - 1 and self._is_retryable_error(exc):
                        delay = min(
                            base_delay * (2**attempt), max_delay
                        ) + random.uniform(0.0, 0.25)
                        self.logger.warning(
                            "NVIDIA NIM transient error on %s (attempt %d/%d): %s. Retrying in %.2fs",
                            model_name,
                            attempt + 1,
                            max_retries,
                            exc,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    self.logger.warning(
                        "NVIDIA NIM model %s failed, trying next model if available: %s",
                        model_name,
                        exc,
                    )
                    break

        raise RuntimeError(f"All NVIDIA NIM models failed: {last_error}")

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        if isinstance(exc, (asyncio.TimeoutError, OSError)):
            return True

        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)

        if isinstance(status_code, int):
            return status_code == 429 or 500 <= status_code < 600

        message = str(exc).lower()
        retry_tokens = (
            "timeout",
            "timed out",
            "rate limit",
            "too many requests",
            "temporar",
        )
        return any(token in message for token in retry_tokens)

    async def _generate_with_nim(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Optional[str]:
        """Generate text using NVIDIA NIM with model fallback support."""
        if not self.client:
            return None

        try:
            await self.rate_limiter.wait_if_needed()

            response = await self._chat_completion_with_fallback(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if not response:
                self.logger.error("NVIDIA NIM returned an empty response object")
                return None

            choices = getattr(response, "choices", None)
            if not choices:
                self.logger.error("NVIDIA NIM returned no choices: %s", response)
                return None

            first_choice = choices[0]
            message = getattr(first_choice, "message", None)
            if not message:
                self.logger.error(
                    "NVIDIA NIM response choice missing message: %s", first_choice
                )
                return None

            content = getattr(message, "content", None)
            if not content:
                self.logger.error(
                    "NVIDIA NIM response message missing content: %s", message
                )
                return None

            return content
        except Exception as e:
            self.logger.error(f"NVIDIA NIM generation failed: {e}")
            return None

    def _analyze_comment_with_fallback(
        self, comment_text: str, video_context: Dict[str, Any]
    ) -> str:
        """Fallback comment analysis"""
        try:
            text_lower = comment_text.lower()

            engagement_score = 50

            if any(
                word in text_lower for word in {"great", "amazing", "love", "awesome"}
            ):
                engagement_score += 20

            if "?" in comment_text:
                engagement_score += 15

            if any(
                phrase in text_lower
                for phrase in {"i think", "my experience", "i tried"}
            ):
                engagement_score += 25

            positive_words = {"great", "amazing", "love", "awesome", "fantastic"}
            negative_words = {"hate", "terrible", "awful", "boring"}

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

    # ── Analysis conversion ───────────────────────────────────────────

    def _convert_to_enhanced_analysis(
        self, analysis_result: Dict[str, Any], reddit_content: Any
    ) -> VideoAnalysisEnhanced:
        """Convert NVIDIA NIM analysis result to VideoAnalysisEnhanced model"""
        try:
            raw_narrative_segments = analysis_result.get(
                "narrative_script_segments", []
            )
            narrative_segments: List[NarrativeSegment] = []
            b_roll_moments = []

            ns_append = narrative_segments.append
            bm_append = b_roll_moments.append

            for item in raw_narrative_segments:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                if not text:
                    continue

                b_roll_query = item.get("b_roll_search_query")
                expression_cue = item.get("expression_cue")
                time_sec = float(item.get("time_seconds", 0.0))
                duration = float(item.get("intended_duration_seconds", 4.0))

                ns_append(
                    NarrativeSegment(
                        text=text,
                        time_seconds=time_sec,
                        intended_duration_seconds=duration,
                        b_roll_search_query=b_roll_query,
                        expression_cue=expression_cue,
                    )
                )

                if b_roll_query:
                    bm_append(
                        {
                            "search_query": b_roll_query,
                            "timestamp_seconds": time_sec,
                            "duration": duration,
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
                description = (
                    getattr(reddit_content, "selftext", "")
                    or getattr(reddit_content, "title", "Amazing content!")
                )[:200]
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
                description = (
                    getattr(reddit_content, "selftext", "") or "Check out this content!"
                )[:200]
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
