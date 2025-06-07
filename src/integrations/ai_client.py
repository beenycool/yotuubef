"""
AI integration for content analysis and generation.
Handles Gemini API for video analysis, content generation, and safety checks.
"""

import json
import logging
import base64
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import cv2
import numpy as np

from src.config.settings import get_config
from src.integrations.reddit_client import RedditPost


@dataclass
class VisualCue:
    """Visual cue for video enhancement"""
    timestamp_seconds: float
    description: str
    effect_type: str  # "zoom", "highlight", "text_overlay", "color_grade"
    intensity: float = 1.0
    duration: float = 1.0


@dataclass
class TextOverlay:
    """Text overlay information"""
    text: str
    timestamp_seconds: float
    duration: float
    position: str = "center"  # "center", "top", "bottom", "left", "right"
    style: str = "default"  # "default", "bold", "highlight", "dramatic"


@dataclass
class NarrativeSegment:
    """TTS narrative segment"""
    text: str
    time_seconds: float
    intended_duration_seconds: float
    emotion: str = "neutral"  # "excited", "calm", "dramatic", "neutral"
    pacing: str = "normal"  # "slow", "normal", "fast"


@dataclass
class VideoAnalysis:
    """Complete video analysis result from AI"""
    # Basic info
    suggested_title: str
    summary_for_description: str
    mood: str
    has_clear_narrative: bool
    original_audio_is_key: bool
    
    # Content elements
    hook_text: str
    hook_variations: List[str]
    visual_hook_moment: Dict[str, Union[float, str]]
    audio_hook: Dict[str, Union[str, float]]
    best_segment: Dict[str, Union[float, str]]
    segments: List[Dict[str, Union[float, str]]]
    
    # Enhancement elements
    key_focus_points: List[Dict[str, float]]
    text_overlays: List[TextOverlay]
    narrative_script_segments: List[NarrativeSegment]
    visual_cues: List[VisualCue]
    speed_effects: List[Dict[str, Union[float, str]]]
    
    # Metadata
    music_genres: List[str]
    sound_effects: List[Dict[str, Union[float, str]]]
    hashtags: List[str]
    original_duration: float
    tts_pacing: str
    emotional_keywords: List[str]
    
    # Thumbnail and engagement
    thumbnail_info: Dict[str, Union[float, str]]
    call_to_action: Dict[str, str]
    retention_tactics: List[str]
    
    # Content safety
    is_explicitly_age_restricted: bool
    fallback: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        
        # Convert dataclass objects to dicts
        result['text_overlays'] = [asdict(overlay) for overlay in self.text_overlays]
        result['narrative_script_segments'] = [asdict(segment) for segment in self.narrative_script_segments]
        result['visual_cues'] = [asdict(cue) for cue in self.visual_cues]
        
        return result


class SafetyChecker:
    """Content safety analysis using AI"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def analyze_content_safety(self, 
                               post: RedditPost, 
                               video_frames: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """
        Analyze content for safety and monetization suitability
        
        Args:
            post: Reddit post to analyze
            video_frames: Optional video frames for visual analysis
        
        Returns:
            Safety analysis result
        """
        prompt_parts = [
            f"Analyze this content for safety and suitability for a general YouTube audience, "
            f"focusing on monetization guidelines. Reddit Title: \"{post.title}\". "
            f"Subreddit: r/{post.subreddit}.",
            f"Is this content problematic? Specifically check for: "
            f"{', '.join(self.config.content.unsuitable_content_types[:10])}.",
            f"Also check for excessive use of words like: "
            f"{', '.join(self.config.content.forbidden_words[:10])}...",
            "Respond ONLY with a JSON object with keys: "
            "'is_problematic' (boolean), 'reason' (string, concise explanation if problematic), "
            "'confidence_percent' (int, 0-100, your confidence in this assessment)."
        ]
        
        # Add visual analysis if frames provided
        if video_frames:
            prompt_parts.append("Consider the provided video frames in your analysis.")
        
        try:
            # Use Gemini for analysis (implementation would depend on specific model)
            # For now, return a basic analysis based on text content
            is_problematic = self._check_text_content(post.title + " " + post.subreddit)
            
            return {
                'is_problematic': is_problematic,
                'reason': 'Text-based content filter detected issues' if is_problematic else 'Content appears suitable',
                'confidence_percent': 85
            }
            
        except Exception as e:
            self.logger.error(f"Error in safety analysis: {e}")
            return {
                'is_problematic': True,
                'reason': 'Error in safety analysis - being conservative',
                'confidence_percent': 50
            }
    
    def _check_text_content(self, text: str) -> bool:
        """Basic text content checking"""
        text_lower = text.lower()
        
        # Check forbidden words
        for word in self.config.content.forbidden_words:
            if word in text_lower:
                return True
        
        # Check unsuitable content types
        for content_type in self.config.content.unsuitable_content_types:
            if content_type in text_lower:
                return True
        
        return False


class GeminiClient:
    """Google Gemini AI client for video analysis"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.safety_checker = SafetyChecker()
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Gemini client"""
        if not self.config.api.gemini_api_key:
            self.logger.warning("Gemini API key not configured - AI analysis will be disabled")
            return
        
        try:
            genai.configure(api_key=self.config.api.gemini_api_key)
            
            # Configure safety settings
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            self.model = genai.GenerativeModel(
                model_name=self.config.api.gemini_model_id,
                safety_settings=safety_settings
            )
            
            self.logger.info(f"Gemini client initialized with model: {self.config.api.gemini_model_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if Gemini client is available"""
        return self.model is not None
    
    def extract_video_frames(self, video_path: Path, max_frames: int = 20) -> List[np.ndarray]:
        """Extract frames from video for analysis"""
        frames = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.error(f"Could not open video: {video_path}")
                return frames
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Extract frames at regular intervals
            if total_frames > 0:
                frame_interval = max(1, total_frames // max_frames)
                
                for i in range(0, total_frames, frame_interval):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    
                    if ret:
                        # Resize frame for efficiency
                        frame = cv2.resize(frame, (512, 288))
                        frames.append(frame)
                        
                        if len(frames) >= max_frames:
                            break
            
            cap.release()
            self.logger.info(f"Extracted {len(frames)} frames from video")
            
        except Exception as e:
            self.logger.error(f"Error extracting video frames: {e}")
        
        return frames
    
    def frames_to_base64(self, frames: List[np.ndarray]) -> List[str]:
        """Convert video frames to base64 for API transmission"""
        base64_frames = []
        
        for frame in frames:
            try:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Encode to JPEG
                _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                # Convert to base64
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                base64_frames.append(frame_base64)
                
            except Exception as e:
                self.logger.warning(f"Error encoding frame to base64: {e}")
        
        return base64_frames
    
    def analyze_video_content(self, 
                              video_path: Path, 
                              post: RedditPost) -> VideoAnalysis:
        """
        Analyze video content and generate enhancement suggestions
        
        Args:
            video_path: Path to the video file
            post: Reddit post associated with the video
        
        Returns:
            VideoAnalysis object with AI-generated insights
        """
        if not self.is_available():
            self.logger.warning("Gemini not available, returning fallback analysis")
            return self._get_fallback_analysis(post)
        
        try:
            # Extract video frames
            frames = self.extract_video_frames(video_path, self.config.api.gemini_max_frames)
            
            if not frames:
                self.logger.warning("No frames extracted, using text-only analysis")
                return self._analyze_text_only(post)
            
            # Prepare prompt for comprehensive video analysis
            prompt = self._create_analysis_prompt(post, len(frames))
            
            # Convert frames to base64 for API
            frame_data = self.frames_to_base64(frames)
            
            # Create content for Gemini
            content_parts = [prompt]
            
            # Add frames to the analysis (Gemini can handle multiple images)
            for i, frame_b64 in enumerate(frame_data[:10]):  # Limit to first 10 frames
                content_parts.append({
                    "mime_type": "image/jpeg",
                    "data": frame_b64
                })
            
            # Generate analysis
            response = self.model.generate_content(
                content_parts,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2048,
                )
            )
            
            # Parse response
            analysis = self._parse_analysis_response(response.text, post)
            
            self.logger.info(f"Generated AI analysis for video: {post.title[:50]}...")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in video analysis: {e}")
            return self._get_fallback_analysis(post)
    
    def _create_analysis_prompt(self, post: RedditPost, num_frames: int) -> str:
        """Create comprehensive analysis prompt for Gemini"""
        return f"""
Analyze this Reddit video content for creating an engaging YouTube Shorts video. 
Title: "{post.title}"
Subreddit: r/{post.subreddit}
Video has {num_frames} frames provided for analysis.

Please provide a comprehensive JSON response with the following structure:
{{
    "suggested_title": "Engaging title under 100 characters",
    "summary_for_description": "Brief engaging description",
    "mood": "overall mood (exciting, funny, dramatic, educational, etc.)",
    "has_clear_narrative": boolean,
    "original_audio_is_key": boolean,
    
    "hook_text": "Attention-grabbing text for the opening",
    "hook_variations": ["Alternative hook 1", "Alternative hook 2", "Alternative hook 3"],
    "visual_hook_moment": {{"timestamp_seconds": 0.0, "description": "Most engaging visual moment"}},
    "audio_hook": {{"type": "sound_effect", "sound_name": "whoosh", "timestamp_seconds": 0.0}},
    
    "best_segment": {{"start_seconds": 0, "end_seconds": 30, "reason": "Why this segment is best"}},
    "segments": [
        {{"start_seconds": 0, "end_seconds": 59, "reason": "Main content segment"}}
    ],
    
    "key_focus_points": [
        {{"x": 0.5, "y": 0.3, "timestamp_seconds": 5.0, "description": "Main subject"}}
    ],
    
    "text_overlays": [
        {{"text": "AMAZING!", "timestamp_seconds": 2.0, "duration": 1.5, "position": "center", "style": "dramatic"}}
    ],
    
    "narrative_script_segments": [
        {{"text": "Watch this incredible moment!", "time_seconds": 1.0, "intended_duration_seconds": 2.0, "emotion": "excited", "pacing": "normal"}}
    ],
    
    "visual_cues": [
        {{"timestamp_seconds": 3.0, "description": "Zoom on main action", "effect_type": "zoom", "intensity": 1.2, "duration": 1.0}}
    ],
    
    "speed_effects": [
        {{"start_seconds": 10, "end_seconds": 15, "speed_factor": 0.5, "effect_type": "slowdown"}}
    ],
    
    "music_genres": ["upbeat", "energetic"],
    "sound_effects": [
        {{"timestamp_seconds": 5.0, "effect_name": "impact", "volume": 0.7}}
    ],
    
    "hashtags": ["#viral", "#shorts", "#amazing"],
    "tts_pacing": "normal",
    "emotional_keywords": ["exciting", "amazing"],
    
    "thumbnail_info": {{"timestamp_seconds": 5.0, "reason": "Most visually striking moment", "headline_text": "INCREDIBLE"}},
    "call_to_action": {{"text": "Subscribe for more!", "type": "subscribe"}},
    "retention_tactics": ["Hook within 3 seconds", "Visual variety", "Text overlays"],
    
    "is_explicitly_age_restricted": false
}}

Focus on creating engaging, family-friendly content suitable for YouTube monetization.
Ensure all suggestions enhance viewer retention and engagement.
"""
    
    def _parse_analysis_response(self, response_text: str, post: RedditPost) -> VideoAnalysis:
        """Parse Gemini response into VideoAnalysis object"""
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_text = response_text[json_start:json_end]
            data = json.loads(json_text)
            
            # Convert to VideoAnalysis object
            text_overlays = [
                TextOverlay(**overlay) for overlay in data.get('text_overlays', [])
            ]
            
            narrative_segments = [
                NarrativeSegment(**segment) for segment in data.get('narrative_script_segments', [])
            ]
            
            visual_cues = [
                VisualCue(**cue) for cue in data.get('visual_cues', [])
            ]
            
            analysis = VideoAnalysis(
                suggested_title=data.get('suggested_title', f"Amazing {post.subreddit} Video"),
                summary_for_description=data.get('summary_for_description', "Check out this incredible video!"),
                mood=data.get('mood', 'exciting'),
                has_clear_narrative=data.get('has_clear_narrative', False),
                original_audio_is_key=data.get('original_audio_is_key', True),
                
                hook_text=data.get('hook_text', 'Watch this!'),
                hook_variations=data.get('hook_variations', ['Amazing!', 'Incredible!', 'Must see!']),
                visual_hook_moment=data.get('visual_hook_moment', {'timestamp_seconds': 0.0, 'description': 'Opening moment'}),
                audio_hook=data.get('audio_hook', {'type': 'sound_effect', 'sound_name': 'whoosh', 'timestamp_seconds': 0.0}),
                
                best_segment=data.get('best_segment', {'start_seconds': 0, 'end_seconds': 30, 'reason': 'Main content'}),
                segments=data.get('segments', [{'start_seconds': 0, 'end_seconds': 59, 'reason': 'Full video'}]),
                
                key_focus_points=data.get('key_focus_points', []),
                text_overlays=text_overlays,
                narrative_script_segments=narrative_segments,
                visual_cues=visual_cues,
                speed_effects=data.get('speed_effects', []),
                
                music_genres=data.get('music_genres', ['upbeat']),
                sound_effects=data.get('sound_effects', []),
                hashtags=data.get('hashtags', ['#shorts', '#viral']),
                original_duration=0.0,  # Will be set by video processor
                tts_pacing=data.get('tts_pacing', 'normal'),
                emotional_keywords=data.get('emotional_keywords', ['interesting']),
                
                thumbnail_info=data.get('thumbnail_info', {'timestamp_seconds': 0.0, 'reason': 'Default', 'headline_text': ''}),
                call_to_action=data.get('call_to_action', {'text': 'Subscribe!', 'type': 'subscribe'}),
                retention_tactics=data.get('retention_tactics', ['Engaging visuals']),
                
                is_explicitly_age_restricted=data.get('is_explicitly_age_restricted', False),
                fallback=False
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error parsing analysis response: {e}")
            return self._get_fallback_analysis(post)
    
    def _analyze_text_only(self, post: RedditPost) -> VideoAnalysis:
        """Analyze based on text content only when video frames unavailable"""
        # Simplified text-based analysis using Gemini
        try:
            prompt = f"""
            Analyze this Reddit post for creating a YouTube Short:
            Title: "{post.title}"
            Subreddit: r/{post.subreddit}
            
            Provide suggestions for title, description, mood, and hashtags in JSON format.
            """
            
            response = self.model.generate_content(prompt)
            return self._parse_analysis_response(response.text, post)
            
        except Exception as e:
            self.logger.error(f"Error in text-only analysis: {e}")
            return self._get_fallback_analysis(post)
    
    def _get_fallback_analysis(self, post: RedditPost) -> VideoAnalysis:
        """Generate fallback analysis when AI is unavailable"""
        return VideoAnalysis(
            suggested_title=f"Amazing {post.subreddit} Video: {post.title[:50]}",
            summary_for_description=f"Check out this incredible video from r/{post.subreddit}!",
            mood='exciting',
            has_clear_narrative=False,
            original_audio_is_key=True,
            
            hook_text='Watch this amazing moment!',
            hook_variations=['Incredible!', 'Must see!', 'Amazing!'],
            visual_hook_moment={'timestamp_seconds': 0.0, 'description': 'Opening scene'},
            audio_hook={'type': 'sound_effect', 'sound_name': 'whoosh', 'timestamp_seconds': 0.0},
            
            best_segment={'start_seconds': 0, 'end_seconds': 30, 'reason': 'Main content'},
            segments=[{'start_seconds': 0, 'end_seconds': 59, 'reason': 'Full video'}],
            
            key_focus_points=[],
            text_overlays=[],
            narrative_script_segments=[
                NarrativeSegment(
                    text="Check out this amazing content!",
                    time_seconds=1.0,
                    intended_duration_seconds=2.0,
                    emotion="excited",
                    pacing="normal"
                )
            ],
            visual_cues=[],
            speed_effects=[],
            
            music_genres=['upbeat'],
            sound_effects=[],
            hashtags=['#shorts', '#viral', f'#{post.subreddit}'],
            original_duration=0.0,
            tts_pacing='normal',
            emotional_keywords=['interesting', 'engaging'],
            
            thumbnail_info={'timestamp_seconds': 0.0, 'reason': 'Default thumbnail', 'headline_text': 'AMAZING'},
            call_to_action={'text': 'Subscribe for more!', 'type': 'subscribe'},
            retention_tactics=['Engaging content', 'Good pacing'],
            
            is_explicitly_age_restricted=False,
            fallback=True
        )


def create_ai_client() -> GeminiClient:
    """Factory function to create an AI client"""
    return GeminiClient()