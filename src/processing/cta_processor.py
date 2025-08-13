"""
Call-to-Action (CTA) processor for adding visual and auditory engagement elements.
Adds subscribe buttons, like reminders, and interactive elements to boost engagement.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from moviepy import (
    VideoFileClip, TextClip, CompositeVideoClip, ColorClip,
    ImageClip, AudioFileClip, concatenate_audioclips
)
from PIL import Image, ImageDraw, ImageFont
import tempfile

from src.config.settings import get_config
from src.models import VideoAnalysis, CallToAction
from src.processing.video_processor_fixes import MoviePyCompat


class CTAProcessor:
    """
    Processes and adds Call-to-Action elements to videos for maximum engagement
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def add_visual_ctas(self, 
                       video_clip: VideoFileClip, 
                       analysis: VideoAnalysis) -> VideoFileClip:
        """
        Add visual CTAs including subscribe buttons, like reminders, and end screens
        
        Args:
            video_clip: Source video clip
            analysis: AI analysis with CTA requirements
            
        Returns:
            Video with visual CTAs added
        """
        try:
            cta_overlays = []
            
            # Add subscribe reminder at strategic moments
            subscribe_overlay = self._create_subscribe_reminder(video_clip, analysis)
            if subscribe_overlay:
                cta_overlays.append(subscribe_overlay)
            
            # Add like reminder
            like_overlay = self._create_like_reminder(video_clip, analysis)
            if like_overlay:
                cta_overlays.append(like_overlay)
            
            # Add comment encouragement
            comment_overlay = self._create_comment_reminder(video_clip, analysis)
            if comment_overlay:
                cta_overlays.append(comment_overlay)
            
            # Add end screen with multiple CTAs
            end_screen = self._create_end_screen_cta(video_clip, analysis)
            if end_screen:
                cta_overlays.append(end_screen)
            
            # Create final composite
            if cta_overlays:
                final_clip = CompositeVideoClip([video_clip] + cta_overlays)
                return final_clip
            else:
                return video_clip
                
        except Exception as e:
            self.logger.error(f"Error adding visual CTAs: {e}")
            return video_clip
    
    def generate_gen_z_cta(self, analysis: Dict[str, Any], audience: str = "gen_z") -> str:
        """
        Generate Gen Z specific call-to-action text with emojis and interactive elements
        
        Args:
            analysis: Video analysis data
            audience: Target audience ("gen_z" or "general")
            
        Returns:
            Generated CTA text
        """
        try:
            if audience == "gen_z":
                # Gen Z specific CTAs with emojis and trending language
                gen_z_ctas = [
                    "Bet you can't do better 😂 Comment below! 👇",
                    "What's your take? Spill the tea ☕ #GenZ",
                    "Duet this if you're brave enough 💪🔥",
                    "Comment your zodiac sign ✨ Let's see who's here!",
                    "Like if you relate, comment if you're different 💯",
                    "Share with your bestie if this made you laugh 😭",
                    "Drop a 🔥 in the comments if this slaps",
                    "Tag someone who needs to see this 👀",
                    "What do you think? Am I right or am I right? 🤔",
                    "Comment your reaction emoji below 😱💀😂",
                    "Like and follow for more chaos 🚀✨",
                    "This is your sign to comment something funny 😂",
                    "Share if you agree, comment if you don't 💁‍♀️",
                    "Drop a 💯 if this is facts",
                    "Comment your thoughts below 👇 I'm listening! 👂"
                ]
                
                # Select random CTA
                import random
                selected_cta = random.choice(gen_z_ctas)
                
                # Add hashtags for Gen Z
                hashtags = ["#GenZ", "#Viral", "#Trending", "#FYP", "#Shorts"]
                selected_hashtags = random.sample(hashtags, 2)
                
                final_cta = f"{selected_cta} {' '.join(selected_hashtags)}"
                
            else:
                # General audience CTAs
                general_ctas = [
                    "Like and subscribe for more content!",
                    "Comment your thoughts below",
                    "Share with friends who might enjoy this",
                    "Follow for daily updates",
                    "Leave a comment with your opinion"
                ]
                
                import random
                final_cta = random.choice(general_ctas)
            
            self.logger.info(f"✅ Generated {audience} CTA: {final_cta}")
            return final_cta
            
        except Exception as e:
            self.logger.error(f"Failed to generate Gen Z CTA: {e}")
            return "Like and subscribe for more! 🔥"
    
    def add_gen_z_interactive_elements(self, video_clip, analysis: Dict[str, Any]) -> VideoFileClip:
        """
        Add Gen Z interactive elements like polls, questions, and challenges
        
        Args:
            video_clip: Video clip to enhance
            analysis: Video analysis data
            
        Returns:
            Video with interactive elements
        """
        try:
            # Check if Gen Z mode is enabled
            config = get_config()
            if not config.ai_features.get('gen_z_mode', False):
                return video_clip
            
            interactive_overlays = []
            
            # Add poll question overlay
            poll_overlay = self._create_poll_overlay(video_clip, analysis)
            if poll_overlay:
                interactive_overlays.append(poll_overlay)
            
            # Add challenge overlay
            challenge_overlay = self._create_challenge_overlay(video_clip, analysis)
            if challenge_overlay:
                interactive_overlays.append(challenge_overlay)
            
            # Add question overlay
            question_overlay = self._create_question_overlay(video_clip, analysis)
            if question_overlay:
                interactive_overlays.append(question_overlay)
            
            # Create final composite
            if interactive_overlays:
                final_clip = CompositeVideoClip([video_clip] + interactive_overlays)
                return final_clip
            else:
                return video_clip
                
        except Exception as e:
            self.logger.error(f"Error adding Gen Z interactive elements: {e}")
            return video_clip
    
    def _create_subscribe_reminder(self, 
                                 video_clip: VideoFileClip, 
                                 analysis: VideoAnalysis) -> Optional[VideoFileClip]:
        """Create animated subscribe button reminder"""
        try:
            # Position at 25% through video for maximum impact
            start_time = video_clip.duration * 0.25
            duration = 3.0
            
            # Create subscribe button image
            button_path = self._create_subscribe_button_image()
            if not button_path or not button_path.exists():
                return None
            
            # Create button clip
            button_clip = ImageClip(str(button_path))
            button_clip = MoviePyCompat.with_position(button_clip, ('right', 'bottom'))
            button_clip = MoviePyCompat.with_start(
                MoviePyCompat.with_duration(button_clip, duration), 
                start_time
            )
            
            # Add pulsing animation
            button_clip = MoviePyCompat.resize(button_clip, lambda t: 1 + 0.1 * np.sin(t * 4))
            
            # Add text reminder
            text_clip = MoviePyCompat.create_text_clip(
                "👆 Subscribe for more!",
                color='white',
                font_size=24
            )
            
            text_clip = MoviePyCompat.with_position(text_clip, ('right', 'bottom'))
            text_clip = MoviePyCompat.with_start(
                MoviePyCompat.with_duration(text_clip, duration),
                start_time
            )
            
            # Combine button and text
            subscribe_composite = CompositeVideoClip([button_clip, text_clip])
            return subscribe_composite
            
        except Exception as e:
            self.logger.warning(f"Error creating subscribe reminder: {e}")
            return None
    
    def _create_like_reminder(self, 
                            video_clip: VideoFileClip, 
                            analysis: VideoAnalysis) -> Optional[VideoFileClip]:
        """Create like reminder with engaging animation"""
        try:
            # Position at peak moment (75% through video)
            start_time = video_clip.duration * 0.75
            duration = 2.5
            
            # Create engaging like text
            like_texts = [
                "❤️ Smash that LIKE!",
                "👍 LIKE if you enjoyed!",
                "💯 Drop a LIKE!",
                "🔥 LIKE for more content!"
            ]
            
            # Choose based on content mood
            if hasattr(analysis, 'mood') and analysis.mood:
                if 'funny' in analysis.mood.lower():
                    like_text = "😂 LIKE if this made you laugh!"
                elif 'amazing' in analysis.mood.lower():
                    like_text = "🤯 LIKE if this amazed you!"
                else:
                    like_text = like_texts[0]
            else:
                like_text = like_texts[0]
            
            # Create text clip with enhanced styling
            text_clip = MoviePyCompat.create_text_clip(
                like_text,
                color='yellow',
                font_size=32,
                stroke_color='black',
                stroke_width=2
            )
            
            # Position at bottom for better visibility without blocking action
            text_clip = MoviePyCompat.with_position(text_clip, ('center', 0.85))
            text_clip = MoviePyCompat.with_start(
                MoviePyCompat.with_duration(text_clip, duration),
                start_time
            )
            
            # Add bounce animation
            def like_animation(t):
                return 1 + 0.3 * max(0, np.sin(t * 6)) if t < 1.0 else 1
            
            text_clip = MoviePyCompat.resize(text_clip, like_animation)
            text_clip = MoviePyCompat.crossfadein(text_clip, 0.4)
            text_clip = MoviePyCompat.crossfadeout(text_clip, 0.4)
            
            return text_clip
            
        except Exception as e:
            self.logger.warning(f"Error creating like reminder: {e}")
            return None
    
    def _create_comment_reminder(self, 
                               video_clip: VideoFileClip, 
                               analysis: VideoAnalysis) -> Optional[VideoFileClip]:
        """Create comment encouragement overlay"""
        try:
            # Position at 60% through video
            start_time = video_clip.duration * 0.6
            duration = 2.0
            
            # Create contextual comment prompt
            comment_prompts = [
                "💬 What's your reaction?",
                "💭 Tell us what you think!",
                "🗨️ Drop a comment below!",
                "💬 Share your thoughts!"
            ]
            
            comment_text = comment_prompts[0]  # Default
            
            # Create text clip
            text_clip = MoviePyCompat.create_text_clip(
                comment_text,
                color='white',
                font_size=26,
                stroke_color='blue',
                stroke_width=1
            )
            
            # Position at bottom center
            text_clip = MoviePyCompat.with_position(text_clip, ('center', 0.8))
            text_clip = MoviePyCompat.with_start(
                MoviePyCompat.with_duration(text_clip, duration),
                start_time
            )
            
            return text_clip
            
        except Exception as e:
            self.logger.warning(f"Error creating comment reminder: {e}")
            return None
    
    def _create_end_screen_cta(self, 
                             video_clip: VideoFileClip, 
                             analysis: VideoAnalysis) -> Optional[VideoFileClip]:
        """Create comprehensive end screen with contextual CTAs"""
        try:
            # End screen in last 4 seconds as suggested
            end_duration = min(4.0, video_clip.duration * 0.12)
            start_time = video_clip.duration - end_duration
            
            end_elements = []
            
            # Dark overlay for better text visibility
            overlay = ColorClip(size=video_clip.size, color=(0, 0, 0))
            overlay = MoviePyCompat.with_opacity(overlay, 0.75)
            overlay = MoviePyCompat.with_start(
                MoviePyCompat.with_duration(overlay, end_duration),
                start_time
            )
            end_elements.append(overlay)
            
            # Main CTA text
            main_cta_text = self._generate_contextual_cta(analysis)
            main_cta = MoviePyCompat.create_text_clip(
                main_cta_text,
                color='white',
                font_size=42,
                stroke_color='red',
                stroke_width=2
            )
            
            main_cta = MoviePyCompat.with_position(main_cta, ('center', 0.25))
            main_cta = MoviePyCompat.with_start(
                MoviePyCompat.with_duration(main_cta, end_duration),
                start_time
            )
            main_cta = MoviePyCompat.crossfadein(main_cta, 0.5)
            
            # Add pulsing animation to main CTA if successful
            if main_cta is not None:
                try:
                    main_cta = MoviePyCompat.resize(main_cta, lambda t: 1 + 0.05 * np.sin(t * 4))
                except Exception as e:
                    self.logger.debug(f"Could not apply pulsing animation: {e}")
                end_elements.append(main_cta)
            
            # Multiple action CTAs
            action_ctas = [
                "👍 LIKE this video!",
                "🔔 SUBSCRIBE & hit the bell!",
                "📤 SHARE with friends!",
                "💬 COMMENT your thoughts!"
            ]
            
            for i, cta_text in enumerate(action_ctas):
                if i < 4:  # Limit to 4 CTAs to avoid overcrowding
                    cta_clip = MoviePyCompat.create_text_clip(
                        cta_text,
                        color='yellow',
                        font_size=28
                    )
                    
                    # Stagger appearance timing
                    cta_start = start_time + (i * 0.3)
                    cta_duration = end_duration - (i * 0.3)
                    
                    if cta_duration > 0.5:  # Only show if there's enough time
                        y_position = 0.55 + (i * 0.06)  # Better spacing
                        cta_clip = MoviePyCompat.with_position(cta_clip, ('center', y_position))
                        cta_clip = MoviePyCompat.with_start(
                            MoviePyCompat.with_duration(cta_clip, cta_duration),
                            cta_start
                        )
                        
                        # Add subtle slide-in animation
                        cta_clip = MoviePyCompat.with_position(
                            cta_clip, 
                            lambda t: ('center', y_position + 20*(1-min(1, t*3)))
                        )
                        
                        end_elements.append(cta_clip)
            
            self.logger.info(f"Created end screen with contextual CTA: {main_cta_text[:50]}...")
            return CompositeVideoClip(end_elements)
            
        except Exception as e:
            self.logger.warning(f"Error creating end screen: {e}")
            return None
    
    def _generate_contextual_cta(self, analysis: VideoAnalysis) -> str:
        """Generate contextual CTA based on content analysis"""
        try:
            # Use AI-provided CTA if available
            if analysis.call_to_action and analysis.call_to_action.text:
                return analysis.call_to_action.text
            
            # Generate based on content mood/type
            if hasattr(analysis, 'mood') and analysis.mood:
                mood = analysis.mood.lower()
                if 'funny' in mood or 'comedy' in mood:
                    return "😂 LAUGHED? Subscribe for daily comedy!"
                elif 'amazing' in mood or 'incredible' in mood:
                    return "🤯 AMAZED? Subscribe for more incredible content!"
                elif 'satisfying' in mood:
                    return "😌 SATISFIED? Subscribe for daily satisfaction!"
                elif 'educational' in mood or 'learning' in mood:
                    return "🧠 LEARNED SOMETHING? Subscribe for daily knowledge!"
                elif 'inspiring' in mood or 'motivational' in mood:
                    return "💪 INSPIRED? Subscribe for daily motivation!"
            
            # Fallback CTAs
            fallback_ctas = [
                "🔥 ENJOYED? Subscribe for more viral content!",
                "💯 LOVED IT? Subscribe & never miss out!",
                "⚡ WANT MORE? Subscribe for daily uploads!",
                "🎯 HOOKED? Subscribe for the best content!"
            ]
            
            return fallback_ctas[0]
            
        except Exception as e:
            self.logger.warning(f"Error generating contextual CTA: {e}")
            return "🔥 SUBSCRIBE for more amazing content!"
    
    def create_engagement_hooks(self, 
                              video_clip: VideoFileClip, 
                              analysis: VideoAnalysis) -> VideoFileClip:
        """Add engagement hooks throughout the video"""
        try:
            hook_overlays = []
            
            # Create curiosity gaps and retention hooks
            hook_points = [
                (video_clip.duration * 0.1, "🔥 Wait for it..."),
                (video_clip.duration * 0.4, "💯 It gets better!"),
                (video_clip.duration * 0.7, "🤯 You won't believe this!")
            ]
            
            for time_point, hook_text in hook_points:
                if time_point < video_clip.duration - 2:  # Ensure enough time
                    hook_clip = MoviePyCompat.create_text_clip(
                        hook_text,
                        color='red',
                        font_size=30,
                        stroke_color='white',
                        stroke_width=1
                    )
                    
                    hook_clip = MoviePyCompat.with_position(hook_clip, ('center', 0.15))
                    hook_clip = MoviePyCompat.with_start(
                        MoviePyCompat.with_duration(hook_clip, 2.0),
                        time_point
                    )
                    
                    # Add subtle animation
                    hook_clip = MoviePyCompat.resize(hook_clip, lambda t: 1 + 0.05 * np.sin(t * 8))
                    
                    hook_overlays.append(hook_clip)
            
            if hook_overlays:
                return CompositeVideoClip([video_clip] + hook_overlays)
            else:
                return video_clip
                
        except Exception as e:
            self.logger.warning(f"Error creating engagement hooks: {e}")
            return video_clip
    
    def add_progress_indicator(self, video_clip: VideoFileClip) -> VideoFileClip:
        """Add a subtle progress bar to encourage watching till the end"""
        try:
            # Create background bar
            bar_width = int(video_clip.w * 0.6)
            bar_height = 4
            
            bg_bar = ColorClip(size=(bar_width, bar_height), color=(128, 128, 128))
            bg_bar = MoviePyCompat.with_opacity(bg_bar, 0.5)
            bg_bar = MoviePyCompat.with_position(bg_bar, ('center', video_clip.h - 30))
            bg_bar = MoviePyCompat.with_duration(bg_bar, video_clip.duration)
            
            # Create progress bar that fills over time
            def progress_width(t):
                progress = min(1.0, t / video_clip.duration)
                return (int(bar_width * progress), bar_height)
            
            progress_bar = ColorClip(size=(bar_width, bar_height), color=(255, 100, 100))
            progress_bar = MoviePyCompat.resize(progress_bar, progress_width)
            progress_bar = MoviePyCompat.with_position(progress_bar, ('center', video_clip.h - 30))
            progress_bar = MoviePyCompat.with_duration(progress_bar, video_clip.duration)
            
            return CompositeVideoClip([video_clip, bg_bar, progress_bar])
            
        except Exception as e:
            self.logger.warning(f"Error adding progress indicator: {e}")
            return video_clip
    
    def _create_subscribe_button_image(self) -> Optional[Path]:
        """Create a subscribe button image programmatically"""
        try:
            # Create a simple subscribe button
            width, height = 120, 40
            image = Image.new('RGBA', (width, height), (255, 0, 0, 255))
            draw = ImageDraw.Draw(image)
            
            # Add text
            try:
                # Try to use a default font, fallback to default if not available
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            text = "SUBSCRIBE"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
            
            # Save to temporary file
            temp_path = Path(tempfile.mktemp(suffix='.png'))
            image.save(temp_path)
            return temp_path
            
        except Exception as e:
            self.logger.warning(f"Error creating subscribe button image: {e}")
            return None
    
    def _create_poll_overlay(self, video_clip, analysis: Dict[str, Any]):
        """Create interactive poll overlay"""
        try:
            # Position poll at 50% through video
            start_time = video_clip.duration * 0.5
            duration = 4.0
            
            # Generate poll question based on content
            poll_questions = [
                "What do you think? 🤔",
                "Vote in the comments! 👇",
                "Which one are you? 💭",
                "Pick your side! ⚖️"
            ]
            
            import random
            poll_text = random.choice(poll_questions)
            
            # Create text clip for poll
            poll_clip = TextClip(
                poll_text,
                fontsize=40,
                color='white',
                stroke_color='black',
                stroke_width=2
            )
            
            # Position and time the poll
            poll_clip = MoviePyCompat.with_position(poll_clip, ('center', 'center'))
            poll_clip = MoviePyCompat.with_start(
                MoviePyCompat.with_duration(poll_clip, duration), 
                start_time
            )
            
            return poll_clip
            
        except Exception as e:
            self.logger.error(f"Failed to create poll overlay: {e}")
            return None
    
    def _create_challenge_overlay(self, video_clip, analysis: Dict[str, Any]):
        """Create challenge overlay for Gen Z engagement"""
        try:
            # Position challenge near the end
            start_time = video_clip.duration * 0.8
            duration = 5.0
            
            # Generate challenge text
            challenges = [
                "Challenge: Duet this! 💪",
                "Challenge: Comment your story 👇",
                "Challenge: Share with 3 friends 🚀",
                "Challenge: Tag someone who needs this 👀"
            ]
            
            import random
            challenge_text = random.choice(challenges)
            
            # Create text clip for challenge
            challenge_clip = TextClip(
                challenge_text,
                fontsize=45,
                color='#FF6B6B',  # Modern red color
                stroke_color='white',
                stroke_width=3
            )
            
            # Position and time the challenge
            challenge_clip = MoviePyCompat.with_position(challenge_clip, ('center', 'bottom'))
            challenge_clip = MoviePyCompat.with_start(
                MoviePyCompat.with_duration(challenge_clip, duration), 
                start_time
            )
            
            return challenge_clip
            
        except Exception as e:
            self.logger.error(f"Failed to create challenge overlay: {e}")
            return None
    
    def _create_question_overlay(self, video_clip, analysis: Dict[str, Any]):
        """Create question overlay to encourage comments"""
        try:
            # Position question at 70% through video
            start_time = video_clip.duration * 0.7
            duration = 4.0
            
            # Generate question based on content
            questions = [
                "What would you do? 🤷‍♀️",
                "Have you experienced this? 💭",
                "What's your opinion? 👂",
                "Can you relate? 🤔"
            ]
            
            import random
            question_text = random.choice(questions)
            
            # Create text clip for question
            question_clip = TextClip(
                question_text,
                fontsize=40,
                color='#4ECDC4',  # Modern teal color
                stroke_color='white',
                stroke_width=2
            )
            
            # Position and time the question
            question_clip = MoviePyCompat.with_position(question_clip, ('center', 'top'))
            question_clip = MoviePyCompat.with_start(
                MoviePyCompat.with_duration(question_clip, duration), 
                start_time
            )
            
            return question_clip
            
        except Exception as e:
            self.logger.error(f"Failed to create question overlay: {e}")
            return None