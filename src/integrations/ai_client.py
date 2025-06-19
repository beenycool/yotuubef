"""
AI Client Integration for Enhanced Video Analysis and Processing
Handles AI-powered content analysis, comment processing, and optimization suggestions.
Now using Google Gemini API instead of OpenAI.
"""

import logging
import json
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path

# Import the new Gemini client
from src.integrations.gemini_ai_client import GeminiAIClient
from src.config.settings import get_config
from src.models import VideoAnalysisEnhanced

try:
    from google.genai import types
    TYPES_AVAILABLE = True
except ImportError:
    TYPES_AVAILABLE = False


class AIClient:
    """
    AI client for enhanced video analysis and processing tasks
    Now powered by Google Gemini API
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize Gemini client
        self.gemini_client = GeminiAIClient()
        self.ai_available = self.gemini_client.gemini_available
        
        if not self.ai_available:
            self.logger.warning("Gemini AI not available - some AI features will be limited")
    
    async def analyze_video_content(self, 
                                   video_path: Path, 
                                   reddit_content: Dict[str, Any]) -> Optional[VideoAnalysisEnhanced]:
        """
        Analyze video content for optimization opportunities using Gemini AI
        
        Args:
            video_path: Path to video file
            reddit_content: Original Reddit content data
            
        Returns:
            Enhanced video analysis or None if failed
        """
        try:
            self.logger.info("Starting AI video content analysis with Gemini...")
            
            # Use Gemini client for analysis
            result = await self.gemini_client.analyze_video_content(video_path, reddit_content)
            
            if result:
                self.logger.info("AI video content analysis completed successfully")
            else:
                self.logger.warning("AI video content analysis failed, using fallback")
            
            return result
            
        except Exception as e:
            self.logger.error(f"AI video analysis failed: {e}")
            return None
    
    async def analyze_comment_engagement(self, 
                                       comment_text: str,
                                       video_context: Dict[str, Any],
                                       analysis_prompt: str = None) -> Optional[str]:
        """
        Analyze comment for engagement potential using Gemini AI
        
        Args:
            comment_text: The comment to analyze
            video_context: Context about the video
            analysis_prompt: Analysis prompt template (optional)
            
        Returns:
            AI analysis response or None if failed
        """
        try:
            return await self.gemini_client.analyze_comment_engagement(
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
    
    async def generate_content_suggestions(self, 
                                         content_type: str,
                                         context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate content suggestions using Gemini AI
        
        Args:
            content_type: Type of content to generate (title, description, tags, etc.)
            context: Context for content generation
            
        Returns:
            Generated content suggestions or None if failed
        """
        try:
            if not self.ai_available:
                return None
            
            # Create a custom prompt based on content type
            prompts = {
                'title': """
                Generate 5 engaging YouTube Shorts titles based on this content:
                {content}
                
                Requirements:
                - Maximum 60 characters each
                - High engagement potential
                - YouTube algorithm friendly
                - Include relevant keywords
                
                Return as JSON array: ["title1", "title2", "title3", "title4", "title5"]
                """,
                
                'description': """
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
                
                'hashtags': """
                Generate trending hashtags for this YouTube Shorts content:
                {content}
                
                Return 10-15 relevant hashtags including:
                - Content-specific tags
                - Trending general tags
                - Niche-specific tags
                - Platform tags (#shorts, #viral, etc.)
                
                Return as JSON array: ["#tag1", "#tag2", ...]
                """
            }
            
            if content_type not in prompts:
                self.logger.warning(f"Unknown content type: {content_type}")
                return None
            
            prompt = prompts[content_type].format(content=json.dumps(context, indent=2))
            
            # Use Gemini to generate content
            response = await asyncio.to_thread(
                self.gemini_client.client.models.generate_content,
                model=self.gemini_client.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.8,
                    max_output_tokens=1000,
                    candidate_count=1
                ) if TYPES_AVAILABLE else {
                    'temperature': 0.8,
                    'max_output_tokens': 1000,
                    'candidate_count': 1
                }
            )
            
            # Try to parse JSON response
            try:
                content_text = response.text
                start_idx = content_text.find('[')
                end_idx = content_text.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content_text[start_idx:end_idx]
                    return {'suggestions': json.loads(json_str)}
            except json.JSONDecodeError:
                pass
            
            # Return raw text if JSON parsing fails
            return {'suggestions': response.text}
            
        except Exception as e:
            self.logger.error(f"Content generation failed: {e}")
            return None
    
    async def analyze_performance_metrics(self, 
                                        video_metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze video performance metrics and provide optimization suggestions
        
        Args:
            video_metrics: Video performance data
            
        Returns:
            Performance analysis and optimization suggestions
        """
        try:
            if not self.ai_available:
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
            
            response = await asyncio.to_thread(
                self.gemini_client.client.models.generate_content,
                model=self.gemini_client.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1500,
                    candidate_count=1
                ) if TYPES_AVAILABLE else {
                    'temperature': 0.3,
                    'max_output_tokens': 1500,
                    'candidate_count': 1
                }
            )
            
            # Try to parse JSON response
            try:
                content = response.text
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                pass
            
            return {'analysis': response.text}
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            return None
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status
        
        Returns:
            Rate limit information
        """
        if not self.ai_available:
            return {'available': False}
        
        rate_limiter = self.gemini_client.rate_limiter
        return {
            'available': True,
            'requests_today': rate_limiter.requests_today,
            'max_daily': rate_limiter.max_daily,
            'requests_this_minute': len(rate_limiter.requests_this_minute),
            'max_rpm': rate_limiter.max_rpm,
            'remaining_daily': rate_limiter.max_daily - rate_limiter.requests_today,
            'api_provider': 'Google Gemini'
        }