"""
AI Client Integration for Enhanced Video Analysis and Processing
Handles AI-powered content analysis, comment processing, and optimization suggestions.
Uses NVIDIA NIM.
"""

import logging
import json
from typing import Dict, Optional, Any
from pathlib import Path

from src.integrations.nvidia_nim_client import NvidiaNimAIClient
from src.config.settings import get_config
from src.models import VideoAnalysisEnhanced


class AIClient:
    """
    AI client for enhanced video analysis and processing tasks
    Uses NVIDIA NIM
    """

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        self.configured_provider = getattr(self.config.api, "ai_provider", "nvidia_nim")
        self.active_provider = "none"

        self.nim_client = NvidiaNimAIClient()

        if self.configured_provider != "nvidia_nim":
            self.logger.warning(
                "AI_PROVIDER '%s' is no longer supported. Using nvidia_nim.",
                self.configured_provider,
            )

        if self.nim_client.nim_available:
            self.active_client = self.nim_client
            self.ai_available = self.nim_client.nim_available
            self.active_provider = "nvidia_nim"
            self.logger.info("Using NVIDIA NIM as primary AI provider")
        else:
            self.active_client = None
            self.ai_available = False
            self.logger.warning("AI not available - some AI features will be limited")

    async def analyze_video_content(
        self, video_path: Path, reddit_content: Dict[str, Any]
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
            provider_name = self.active_provider.upper()
            self.logger.info(
                f"Starting AI video content analysis with {provider_name}..."
            )

            if not self.active_client:
                self.logger.warning("No AI client available, using fallback")
                return None

            # Use active client for analysis
            result = await self.active_client.analyze_video_content(
                video_path, reddit_content
            )

            if result:
                self.logger.info("AI video content analysis completed successfully")
            else:
                self.logger.warning("AI video content analysis failed, using fallback")

            return result

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
            if not self.active_client:
                return None

            return await self.active_client.analyze_comment_engagement(
                comment_text, video_context, analysis_prompt
            )

        except Exception as e:
            self.logger.error(f"Comment analysis failed: {e}")
            return None

    def is_available(self) -> bool:
        """
        Check if AI services are available

        Returns:
            True if AI services are available, False otherwise
        """
        return self.ai_available

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
            if not self.ai_available or not self.active_client:
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
            if not self.ai_available or not self.active_client:
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

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status

        Returns:
            Rate limit information
        """
        if not self.ai_available or not self.active_client:
            return {"available": False}

        if hasattr(self.active_client, "rate_limiter"):
            rate_limiter = self.active_client.rate_limiter
            return {
                "available": True,
                "requests_this_minute": len(rate_limiter.requests_this_minute),
                "max_rpm": rate_limiter.max_rpm,
                "api_provider": "NVIDIA NIM",
                "configured_provider": self.configured_provider,
                "active_provider": self.active_provider,
            }

        return {
            "available": True,
            "configured_provider": self.configured_provider,
            "active_provider": self.active_provider,
        }

    async def _run_generation_prompt(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Optional[str]:
        """Run a prompt against the currently active provider."""
        if not self.active_client:
            return None

        if isinstance(self.active_client, NvidiaNimAIClient):
            return await self._generate_with_nim(prompt, temperature, max_tokens)

        self.logger.error("Unsupported AI client type: %s", type(self.active_client))
        return None

    async def _generate_with_nim(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Optional[str]:
        """Generate text using NVIDIA NIM with model fallback support."""
        nim_client = (
            self.active_client
            if isinstance(self.active_client, NvidiaNimAIClient)
            else None
        )
        if not nim_client or not getattr(nim_client, "client", None):
            return None

        try:
            if hasattr(nim_client, "rate_limiter"):
                await nim_client.rate_limiter.wait_if_needed()

            client = nim_client.client
            if client is None:
                return None

            if hasattr(nim_client, "_chat_completion_with_fallback"):
                response = await nim_client._chat_completion_with_fallback(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                response = await client.chat.completions.create(
                    model=nim_client.model,
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
