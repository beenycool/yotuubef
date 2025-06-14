"""
Google Gemini AI Client Integration for Enhanced Video Analysis and Processing
Handles AI-powered content analysis, comment processing, and optimization suggestions using Gemini API.
"""

import logging
import json
import asyncio
import time
import os
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from src.config.settings import get_config
from src.models import VideoAnalysisEnhanced


class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, max_requests_per_minute: int = 10, max_requests_per_day: int = 500):
        self.max_rpm = max_requests_per_minute
        self.max_daily = max_requests_per_day
        self.requests_this_minute = []
        self.requests_today = 0
        self.daily_reset_time = time.time() + 86400  # 24 hours
        
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        current_time = time.time()
        
        # Reset daily counter if needed
        if current_time > self.daily_reset_time:
            self.requests_today = 0
            self.daily_reset_time = current_time + 86400
        
        # Check daily limit
        if self.requests_today >= self.max_daily:
            raise Exception(f"Daily rate limit of {self.max_daily} requests exceeded")
        
        # Clean old requests (older than 1 minute)
        minute_ago = current_time - 60
        self.requests_this_minute = [t for t in self.requests_this_minute if t > minute_ago]
        
        # Check if we need to wait
        if len(self.requests_this_minute) >= self.max_rpm:
            wait_time = 60 - (current_time - self.requests_this_minute[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.requests_this_minute.append(current_time)
        self.requests_today += 1


class GeminiAIClient:
    """
    Google Gemini AI client for enhanced video analysis and processing tasks
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Get Gemini API key from environment variables or config
        gemini_api_key = os.getenv('GEMINI_API_KEY') or getattr(self.config.api, 'gemini_api_key', None)
        
        # Initialize Gemini if available
        if GEMINI_AVAILABLE and gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.model = genai.GenerativeModel(
                    model_name=os.getenv('GEMINI_MODEL') or getattr(self.config.api, 'gemini_model', 'gemini-2.0-flash-exp')
                )
                self.gemini_available = True
                
                # Setup rate limiting
                rpm_limit = int(os.getenv('GEMINI_RATE_LIMIT_RPM', '0')) or getattr(self.config.api, 'gemini_rate_limit_rpm', 10)
                daily_limit = int(os.getenv('GEMINI_RATE_LIMIT_DAILY', '0')) or getattr(self.config.api, 'gemini_rate_limit_daily', 500)
                self.rate_limiter = RateLimiter(rpm_limit, daily_limit)
                
                self.logger.info(f"Gemini AI initialized with model: {self.model.model_name}")
                
            except Exception as e:
                self.gemini_available = False
                self.logger.warning(f"Failed to initialize Gemini AI: {e}")
            
        else:
            self.gemini_available = False
            if not GEMINI_AVAILABLE:
                self.logger.warning("Gemini library not installed - install with: pip install google-generativeai")
            else:
                self.logger.warning("Gemini API key not found - set GEMINI_API_KEY environment variable or add to config")
        
        # AI analysis prompts
        self.video_analysis_prompt = """
        Analyze this video content for YouTube Shorts optimization:
        
        Title: {title}
        Description: {description}
        Duration: {duration} seconds
        Subreddit: {subreddit}
        Reddit Score: {score}
        Comments: {num_comments}
        
        Provide analysis in JSON format with the following structure:
        {{
            "suggested_title": "Optimized title for engagement (max 60 chars)",
            "hook_text": "Opening hook text that grabs attention",
            "hook_variations": ["Alternative hook 1", "Alternative hook 2", "Alternative hook 3"],
            "mood": "overall mood (exciting, calm, dramatic, humorous, mysterious)",
            "key_moments": [
                {{"timestamp": 5.0, "description": "Key moment description"}},
                {{"timestamp": 15.0, "description": "Another key moment"}}
            ],
            "suggested_hashtags": ["#tag1", "#tag2", "#tag3"],
            "engagement_score": 85,
            "retention_predictions": {{
                "intro_retention": 90,
                "mid_retention": 75,
                "end_retention": 60
            }},
            "optimal_thumbnail_timestamp": 12.5,
            "call_to_action": "Subscribe for more amazing content!",
            "content_style": "educational/entertainment/dramatic/comedy",
            "target_audience": "general/tech/lifestyle/etc"
        }}
        
        Focus on maximizing engagement and retention for YouTube Shorts format.
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
    
    async def analyze_video_content(self,
                                   video_path: Path,
                                   reddit_content: Any) -> Optional[VideoAnalysisEnhanced]:
        """
        Analyze video content for optimization opportunities
        
        Args:
            video_path: Path to video file
            reddit_content: Original Reddit content data (RedditPost object or dict)
            
        Returns:
            Enhanced video analysis or None if failed
        """
        try:
            self.logger.info("Starting Gemini video content analysis...")
            
            # Extract video metadata
            video_metadata = self._extract_video_metadata(video_path)
            
            # Prepare analysis context - handle both RedditPost object and dict
            if hasattr(reddit_content, '__dict__') and hasattr(reddit_content, 'title') and not hasattr(reddit_content, 'get'):
                # RedditPost object (dataclass/object with attributes)
                analysis_context = {
                    'title': getattr(reddit_content, 'title', ''),
                    'description': '',  # RedditPost doesn't have selftext, could use title as description
                    'duration': video_metadata.get('duration', 60),
                    'subreddit': getattr(reddit_content, 'subreddit', ''),
                    'score': getattr(reddit_content, 'score', 0),
                    'num_comments': getattr(reddit_content, 'num_comments', 0)
                }
            else:
                # Dictionary format (legacy support)
                analysis_context = {
                    'title': reddit_content.get('title', '') if hasattr(reddit_content, 'get') else '',
                    'description': reddit_content.get('selftext', '') if hasattr(reddit_content, 'get') else '',
                    'duration': video_metadata.get('duration', 60),
                    'subreddit': reddit_content.get('subreddit', '') if hasattr(reddit_content, 'get') else '',
                    'score': reddit_content.get('score', 0) if hasattr(reddit_content, 'get') else 0,
                    'num_comments': reddit_content.get('num_comments', 0) if hasattr(reddit_content, 'get') else 0
                }
            
            # Perform AI analysis
            if self.gemini_available:
                analysis_result = await self._analyze_with_gemini(analysis_context)
            else:
                analysis_result = self._analyze_with_fallback(analysis_context)
            
            if not analysis_result:
                return None
            
            # Convert to VideoAnalysisEnhanced
            enhanced_analysis = self._convert_to_enhanced_analysis(analysis_result, reddit_content)
            
            self.logger.info("Gemini video content analysis completed")
            return enhanced_analysis
            
        except Exception as e:
            self.logger.error(f"Gemini video analysis failed: {e}")
            return None
    
    async def analyze_comment_engagement(self, 
                                       comment_text: str,
                                       video_context: Dict[str, Any],
                                       analysis_prompt: str = None) -> Optional[str]:
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
            if self.gemini_available:
                return await self._analyze_comment_with_gemini(comment_text, video_context, analysis_prompt)
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
                    'duration': clip.duration,
                    'fps': clip.fps,
                    'size': clip.size,
                    'has_audio': clip.audio is not None
                }
                
        except Exception as e:
            self.logger.warning(f"Video metadata extraction failed: {e}")
            return {'duration': 60, 'fps': 30, 'size': (1920, 1080), 'has_audio': True}
    
    async def _analyze_with_gemini(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze content using Gemini API"""
        try:
            # Wait for rate limit if needed
            await self.rate_limiter.wait_if_needed()
            
            # Prepare prompt
            prompt = self.video_analysis_prompt.format(**context)
            
            self.logger.debug(f"Sending prompt to Gemini: {prompt[:100]}...")
            
            # Make API call
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2000,
                    candidate_count=1
                )
            )
            
            # Parse response
            content = response.text
            self.logger.debug(f"Gemini response: {content[:200]}...")
            
            # Try to extract JSON from response
            try:
                # Look for JSON block in response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                pass
            
            # If JSON parsing fails, create structured response from text
            return self._parse_gemini_text_response(content, context)
            
        except Exception as e:
            self.logger.error(f"Gemini analysis failed: {e}")
            return None
    
    def _parse_gemini_text_response(self, response_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Gemini text response into structured format"""
        try:
            # Extract key information from text response
            lines = response_text.split('\n')
            
            analysis = {
                'suggested_title': context.get('title', 'Engaging Video'),
                'hook_text': 'Check this out!',
                'hook_variations': ['Amazing!', 'Incredible!', 'Must see!'],
                'mood': 'exciting',
                'key_moments': [
                    {'timestamp': 5.0, 'description': 'Opening hook'},
                    {'timestamp': 15.0, 'description': 'Main content'}
                ],
                'suggested_hashtags': ['#shorts', '#viral', '#trending'],
                'engagement_score': 75,
                'retention_predictions': {
                    'intro_retention': 85,
                    'mid_retention': 70,
                    'end_retention': 55
                },
                'optimal_thumbnail_timestamp': 12.5,
                'call_to_action': 'Subscribe for more!',
                'content_style': 'entertainment',
                'target_audience': 'general'
            }
            
            # Try to extract better title if mentioned
            for line in lines:
                if 'title:' in line.lower():
                    title_part = line.split(':', 1)[1].strip()
                    if title_part:
                        analysis['suggested_title'] = title_part[:60]  # YouTube title limit
                        break
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Response parsing failed: {e}")
            return None
    
    def _analyze_with_fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when Gemini is not available"""
        try:
            title = context.get('title', 'Video')
            duration = context.get('duration', 60)
            score = context.get('score', 0)
            
            # Generate analysis based on content characteristics
            engagement_score = min(100, max(30, score // 10 + 50))
            
            # Determine mood based on title keywords
            mood = 'exciting'
            title_lower = title.lower()
            if any(word in title_lower for word in ['calm', 'peaceful', 'relaxing']):
                mood = 'calm'
            elif any(word in title_lower for word in ['dramatic', 'intense', 'serious']):
                mood = 'dramatic'
            elif any(word in title_lower for word in ['funny', 'hilarious', 'comedy']):
                mood = 'humorous'
            
            # Generate hook variations based on title
            hook_base = "You won't believe this!"
            if 'how to' in title_lower:
                hook_base = "Learn this amazing trick!"
            elif any(word in title_lower for word in ['secret', 'hidden', 'revealed']):
                hook_base = "The secret is finally revealed!"
            
            analysis = {
                'suggested_title': self._optimize_title(title),
                'hook_text': hook_base,
                'hook_variations': [
                    "This is incredible!",
                    "You have to see this!",
                    "Mind-blowing content ahead!",
                    "This changed everything!"
                ],
                'mood': mood,
                'key_moments': self._generate_key_moments(duration),
                'suggested_hashtags': self._generate_hashtags(title, context.get('subreddit', '')),
                'engagement_score': engagement_score,
                'retention_predictions': {
                    'intro_retention': min(95, engagement_score + 10),
                    'mid_retention': max(60, engagement_score - 10),
                    'end_retention': max(45, engagement_score - 20)
                },
                'optimal_thumbnail_timestamp': duration * 0.3,
                'call_to_action': 'Subscribe for more amazing content!',
                'content_style': 'entertainment',
                'target_audience': 'general'
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Fallback analysis failed: {e}")
            return {}
    
    def _optimize_title(self, original_title: str) -> str:
        """Optimize title for YouTube engagement"""
        try:
            # Remove Reddit-specific formatting
            title = original_title.replace('[OC]', '').replace('[Original Content]', '').strip()
            
            # Add engagement triggers if not present
            engagement_words = ['Amazing', 'Incredible', 'Shocking', 'Unbelievable']
            title_lower = title.lower()
            
            if not any(word.lower() in title_lower for word in engagement_words):
                if len(title) < 50:  # Add prefix if title is short enough
                    title = f"Amazing: {title}"
            
            # Ensure proper length (YouTube optimal: 60 characters)
            if len(title) > 60:
                title = title[:57] + "..."
            
            return title
            
        except Exception as e:
            self.logger.warning(f"Title optimization failed: {e}")
            return original_title
    
    def _generate_key_moments(self, duration: float) -> List[Dict[str, Any]]:
        """Generate key moments based on video duration"""
        moments = []
        
        # Opening hook
        moments.append({'timestamp': 2.0, 'description': 'Opening hook moment'})
        
        # Mid-point climax
        if duration > 20:
            moments.append({'timestamp': duration * 0.4, 'description': 'Main content peak'})
        
        # Closing moment
        if duration > 30:
            moments.append({'timestamp': duration * 0.8, 'description': 'Closing impact'})
        
        return moments
    
    def _generate_hashtags(self, title: str, subreddit: str) -> List[str]:
        """Generate relevant hashtags"""
        hashtags = ['#shorts', '#viral']
        
        # Add subreddit-based tags
        if subreddit:
            hashtags.append(f"#{subreddit.lower()}")
        
        # Add content-based tags
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['funny', 'comedy', 'hilarious']):
            hashtags.extend(['#funny', '#comedy'])
        elif any(word in title_lower for word in ['amazing', 'incredible', 'wow']):
            hashtags.extend(['#amazing', '#mindblowing'])
        elif any(word in title_lower for word in ['how to', 'tutorial', 'guide']):
            hashtags.extend(['#howto', '#tutorial'])
        elif any(word in title_lower for word in ['reaction', 'responds']):
            hashtags.extend(['#reaction', '#response'])
        
        # Add trending tags
        hashtags.extend(['#trending', '#fyp', '#explore'])
        
        return hashtags[:10]  # Limit to 10 hashtags
    
    async def _analyze_comment_with_gemini(self, 
                                         comment_text: str,
                                         video_context: Dict[str, Any],
                                         analysis_prompt: str = None) -> Optional[str]:
        """Analyze comment using Gemini"""
        try:
            # Wait for rate limit if needed
            await self.rate_limiter.wait_if_needed()
            
            # Use provided prompt or default
            prompt = analysis_prompt or self.comment_analysis_prompt
            
            # Prepare the full prompt
            full_prompt = prompt.format(
                comment_text=comment_text,
                video_context=json.dumps(video_context, indent=2)
            )
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=800,
                    candidate_count=1
                )
            )
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Gemini comment analysis failed: {e}")
            return None
    
    def _analyze_comment_with_fallback(self, 
                                     comment_text: str,
                                     video_context: Dict[str, Any]) -> str:
        """Fallback comment analysis"""
        try:
            text_lower = comment_text.lower()
            
            # Simple engagement scoring
            engagement_score = 50  # Base score
            
            # Positive indicators
            if any(word in text_lower for word in ['great', 'amazing', 'love', 'awesome']):
                engagement_score += 20
            
            # Question indicates engagement
            if '?' in comment_text:
                engagement_score += 15
            
            # Personal experience sharing
            if any(phrase in text_lower for phrase in ['i think', 'my experience', 'i tried']):
                engagement_score += 25
            
            # Determine sentiment
            positive_words = ['great', 'amazing', 'love', 'awesome', 'fantastic']
            negative_words = ['hate', 'terrible', 'awful', 'boring']
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = 'positive'
            elif negative_count > positive_count:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Generate response
            response_parts = [
                f"Engagement Score: {engagement_score}",
                f"Sentiment: {sentiment}",
                "Toxicity Score: 5" if sentiment != 'negative' else "Toxicity Score: 25",
                "Reply Urgency: medium" if engagement_score > 70 else "Reply Urgency: low",
                f"Should Pin: {'true' if engagement_score > 80 and sentiment == 'positive' else 'false'}",
                f"Should Heart: {'true' if engagement_score > 60 and sentiment != 'negative' else 'false'}",
                "Suggested Response: Thanks for your comment! 👍",
                f"Interaction Priority: {min(10, max(1, engagement_score // 10))}",
                "Comment Type: other"
            ]
            
            return "\n".join(response_parts)
            
        except Exception as e:
            self.logger.error(f"Fallback comment analysis failed: {e}")
            return "Engagement Score: 50\nSentiment: neutral\nInteraction Priority: 5"
    
    def _convert_to_enhanced_analysis(self,
                                    analysis_result: Dict[str, Any],
                                    reddit_content: Any) -> VideoAnalysisEnhanced:
        """Convert Gemini analysis result to VideoAnalysisEnhanced model"""
        try:
            from src.models import (
                HookMoment, AudioHook, VideoSegment, ThumbnailInfo, CallToAction,
                AudioDuckingConfig
            )
            
            # Extract key moments for segments and hooks
            key_moments = analysis_result.get('key_moments', [])
            
            # Create segments from key moments
            segments = []
            if key_moments:
                for i, moment in enumerate(key_moments):
                    start_time = moment['timestamp']
                    end_time = key_moments[i + 1]['timestamp'] if i + 1 < len(key_moments) else start_time + 10
                    
                    segments.append(VideoSegment(
                        start_seconds=start_time,
                        end_seconds=end_time,
                        reason=moment['description']
                    ))
            else:
                # Default segment
                segments.append(VideoSegment(
                    start_seconds=0,
                    end_seconds=30,
                    reason="Main content segment"
                ))
            
            # Create hook moment
            visual_hook_moment = HookMoment(
                timestamp_seconds=key_moments[0]['timestamp'] if key_moments else 5.0,
                description=key_moments[0]['description'] if key_moments else "Opening hook"
            )
            
            # Create audio hook
            audio_hook = AudioHook(
                type="dramatic",
                sound_name="whoosh",
                timestamp_seconds=visual_hook_moment.timestamp_seconds
            )
            
            # Create thumbnail info
            thumbnail_timestamp = analysis_result.get('optimal_thumbnail_timestamp', visual_hook_moment.timestamp_seconds)
            thumbnail_info = ThumbnailInfo(
                timestamp_seconds=thumbnail_timestamp,
                reason="Most engaging moment",
                headline_text=analysis_result.get('hook_text', 'Must Watch!')
            )
            
            # Create call to action
            call_to_action = CallToAction(
                text=analysis_result.get('call_to_action', 'Subscribe for more!'),
                type="subscribe"
            )
            
            # Handle both RedditPost object and dict for content extraction
            if hasattr(reddit_content, '__dict__') and hasattr(reddit_content, 'title') and not hasattr(reddit_content, 'get'):
                # RedditPost object (dataclass/object with attributes)
                title = getattr(reddit_content, 'title', 'Video')
                description = getattr(reddit_content, 'title', 'Amazing content!')[:200]  # Use title as description since RedditPost doesn't have selftext
            else:
                # Dictionary format
                title = reddit_content.get('title', 'Video') if hasattr(reddit_content, 'get') else 'Video'
                description = (reddit_content.get('selftext', '') if hasattr(reddit_content, 'get') else '')[:200] or "Amazing content!"
            
            # Create enhanced analysis
            enhanced_analysis = VideoAnalysisEnhanced(
                suggested_title=analysis_result.get('suggested_title', title),
                summary_for_description=description,
                mood=analysis_result.get('mood', 'exciting'),
                has_clear_narrative=True,
                original_audio_is_key=False,
                hook_text=analysis_result.get('hook_text', 'Must watch!'),
                hook_variations=analysis_result.get('hook_variations', ['Amazing!', 'Incredible!', 'Must see!']),
                best_segment=segments[0] if segments else VideoSegment(start_seconds=0, end_seconds=30, reason="Main content"),
                segments=segments,
                visual_hook_moment=visual_hook_moment,
                audio_hook=audio_hook,
                thumbnail_info=thumbnail_info,
                call_to_action=call_to_action,
                music_genres=analysis_result.get('music_genres', ['upbeat', 'electronic']),
                hashtags=analysis_result.get('suggested_hashtags', ['#shorts', '#viral']),
                audio_ducking_config=AudioDuckingConfig(
                    duck_during_narration=True,
                    duck_volume=0.3,
                    fade_duration=0.5
                )
            )
            
            return enhanced_analysis
            
        except Exception as e:
            self.logger.error(f"Enhanced analysis conversion failed: {e}")
            return self._create_minimal_analysis(reddit_content)
    
    def _create_minimal_analysis(self, reddit_content: Any) -> VideoAnalysisEnhanced:
        """Create minimal analysis when full analysis fails"""
        try:
            from src.models import (
                HookMoment, AudioHook, VideoSegment, ThumbnailInfo, CallToAction,
                AudioDuckingConfig
            )
            
            # Handle both RedditPost object and dict
            if hasattr(reddit_content, '__dict__') and hasattr(reddit_content, 'title') and not hasattr(reddit_content, 'get'):
                # RedditPost object (dataclass/object with attributes)
                title = getattr(reddit_content, 'title', 'Engaging Video')[:60]
                description = "Check out this content!"
            else:
                # Dictionary format
                title = (reddit_content.get('title', 'Engaging Video') if hasattr(reddit_content, 'get') else 'Engaging Video')[:60]
                description = (reddit_content.get('selftext', '') if hasattr(reddit_content, 'get') else '')[:200] or "Check out this content!"
            
            main_segment = VideoSegment(start_seconds=0, end_seconds=30, reason="Main content")
            
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
                visual_hook_moment=HookMoment(timestamp_seconds=5.0, description="Opening moment"),
                audio_hook=AudioHook(type="dramatic", sound_name="whoosh", timestamp_seconds=5.0),
                thumbnail_info=ThumbnailInfo(timestamp_seconds=5.0, reason="Opening hook", headline_text="Must Watch!"),
                call_to_action=CallToAction(text="Subscribe for more!", type="subscribe"),
                music_genres=["upbeat", "electronic"],
                hashtags=["#shorts", "#viral", "#trending"],
                audio_ducking_config=AudioDuckingConfig(
                    duck_during_narration=True,
                    duck_volume=0.3,
                    fade_duration=0.5
                )
            )
            
        except Exception as e:
            self.logger.error(f"Minimal analysis creation failed: {e}")
            raise