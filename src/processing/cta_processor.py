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
            end_screen = self._create_end_screen(video_clip, analysis)
            if end_screen:
                cta_overlays.append(end_screen)
            
            # Composite all overlays
            if cta_overlays:
                all_clips = [video_clip] + cta_overlays
                enhanced_video = CompositeVideoClip(all_clips)
                self.logger.info(f"Added {len(cta_overlays)} visual CTA elements")
                return enhanced_video
            else:
                return video_clip
                
        except Exception as e:
            self.logger.warning(f"Error adding visual CTAs: {e}")
            return video_clip
    
    def _create_subscribe_reminder(self, 
                                  video_clip: VideoFileClip, 
                                  analysis: VideoAnalysis) -> Optional[CompositeVideoClip]:
        """Create animated subscribe reminder"""
        try:
            # Position subscribe reminder in early-mid section
            start_time = min(15.0, video_clip.duration * 0.3)
            duration = 3.0
            
            if start_time + duration > video_clip.duration:
                return None
            
            # Create subscribe button graphic
            button_clip = self._create_subscribe_button(video_clip.size)
            if not button_clip:
                return None
            
            # Position in bottom right
            button_clip = button_clip.set_position(('right', 'bottom')).set_margin(20)
            button_clip = button_clip.set_start(start_time).set_duration(duration)
            
            # Add pulsing animation
            button_clip = button_clip.resize(lambda t: 1 + 0.1 * np.sin(t * 4))
            
            # Add accompanying text
            text_clip = TextClip(
                "SUBSCRIBE for more!",
                fontsize=40,
                color='white',
                stroke_color='red',
                stroke_width=2,
                font=str(self.config.get_font_path('Montserrat-Bold.ttf'))
            )
            
            text_clip = text_clip.set_position(('right', 'bottom')).set_margin((20, 80))
            text_clip = text_clip.set_start(start_time).set_duration(duration)
            text_clip = text_clip.crossfadein(0.5).crossfadeout(0.5)
            
            return CompositeVideoClip([button_clip, text_clip])
            
        except Exception as e:
            self.logger.warning(f"Error creating subscribe reminder: {e}")
            return None
    
    def _create_subscribe_button(self, video_size: Tuple[int, int]) -> Optional[ImageClip]:
        """Create an animated subscribe button graphic"""
        try:
            # Create button using PIL
            button_size = (120, 40)
            img = Image.new('RGBA', button_size, (255, 0, 0, 255))  # Red background
            draw = ImageDraw.Draw(img)
            
            # Add border
            draw.rectangle([0, 0, button_size[0]-1, button_size[1]-1], 
                          outline=(180, 0, 0, 255), width=2)
            
            # Add text
            try:
                font = ImageFont.truetype(str(self.config.get_font_path('Montserrat-Bold.ttf')), 16)
            except:
                font = ImageFont.load_default()
            
            text = "SUBSCRIBE"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            text_x = (button_size[0] - text_width) // 2
            text_y = (button_size[1] - text_height) // 2
            
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))
            
            # Save to temporary file
            temp_path = Path(tempfile.mktemp(suffix='.png'))
            img.save(temp_path, 'PNG')
            
            # Create ImageClip
            button_clip = ImageClip(str(temp_path), duration=3.0)
            
            # Clean up temp file
            temp_path.unlink(missing_ok=True)
            
            return button_clip
            
        except Exception as e:
            self.logger.warning(f"Error creating subscribe button: {e}")
            return None
    
    def _create_like_reminder(self,
                             video_clip: VideoFileClip,
                             analysis: VideoAnalysis) -> Optional[TextClip]:
        """Create engaging like reminder overlay at strategic moment"""
        try:
            # Show like reminder at around 10-second mark as suggested
            start_time = min(10.0, video_clip.duration * 0.2)
            duration = 3.0
            
            if start_time + duration > video_clip.duration:
                return None
            
            # Create more engaging like reminder text based on content mood
            like_texts = {
                'intense': "👍 Like if this is INSANE!",
                'dramatic': "👍 Like if this gave you chills!",
                'funny': "👍 Like if this made you laugh!",
                'amazing': "👍 Like if this blew your mind!",
                'satisfying': "👍 Like if this was satisfying!",
                'educational': "👍 Like if you learned something!",
                'uplifting': "👍 Like if this made your day!"
            }
            
            # Get appropriate text based on mood
            mood = analysis.mood.lower() if analysis.mood else 'amazing'
            like_text = like_texts.get(mood, "👍 Like if this is crazy!")
            
            text_clip = TextClip(
                like_text,
                fontsize=38,
                color='#FFD700',  # Gold color for more impact
                stroke_color='#000000',
                stroke_width=3,
                font=str(self.config.get_font_path('Montserrat-Bold.ttf'))
            )
            
            # Position at bottom for better visibility without blocking action
            text_clip = text_clip.set_position(('center', 0.85), relative=True)
            text_clip = text_clip.set_start(start_time).set_duration(duration)
            
            # Enhanced animation: pulse + slight bounce
            def like_animation(t):
                pulse = 1 + 0.2 * np.sin(t * 8)  # Faster pulse
                bounce = 1 + 0.05 * np.sin(t * 3)  # Subtle bounce
                return pulse * bounce
            
            text_clip = text_clip.resize(like_animation)
            text_clip = text_clip.crossfadein(0.4).crossfadeout(0.4)
            
            self.logger.info(f"Created like reminder for mood '{mood}': {like_text}")
            return text_clip
            
        except Exception as e:
            self.logger.warning(f"Error creating like reminder: {e}")
            return None
    
    def _create_comment_reminder(self, 
                                video_clip: VideoFileClip, 
                                analysis: VideoAnalysis) -> Optional[TextClip]:
        """Create comment encouragement overlay"""
        try:
            # Show comment reminder in middle section
            start_time = min(25.0, video_clip.duration * 0.5)
            duration = 3.0
            
            if start_time + duration > video_clip.duration:
                return None
            
            # Use engaging question from analysis if available
            comment_text = "💬 What do you think? Comment below!"
            if analysis.retention_tactics:
                # Look for question-based tactics
                for tactic in analysis.retention_tactics:
                    if '?' in tactic:
                        comment_text = f"💬 {tactic}"
                        break
            
            text_clip = TextClip(
                comment_text,
                fontsize=32,
                color='cyan',
                stroke_color='navy',
                stroke_width=2,
                font=str(self.config.get_font_path('Montserrat-Bold.ttf')),
                method='caption',
                size=(video_clip.w * 0.8, None)  # Wrap text
            )
            
            # Position at bottom center
            text_clip = text_clip.set_position(('center', 0.8), relative=True)
            text_clip = text_clip.set_start(start_time).set_duration(duration)
            text_clip = text_clip.crossfadein(0.4).crossfadeout(0.4)
            
            return text_clip
            
        except Exception as e:
            self.logger.warning(f"Error creating comment reminder: {e}")
            return None
    
    def _create_end_screen(self,
                          video_clip: VideoFileClip,
                          analysis: VideoAnalysis) -> Optional[CompositeVideoClip]:
        """Create comprehensive end screen with contextual CTAs"""
        try:
            # End screen in last 4 seconds as suggested
            end_duration = min(4.0, video_clip.duration * 0.12)
            start_time = video_clip.duration - end_duration
            
            if end_duration < 2.0:  # Not enough time for end screen
                return None
            
            end_elements = []
            
            # Background overlay with gradient effect
            overlay = ColorClip(size=video_clip.size, color=(0, 0, 0))
            overlay = overlay.set_opacity(0.75)  # Slightly more opaque for better text visibility
            overlay = overlay.set_start(start_time).set_duration(end_duration)
            end_elements.append(overlay)
            
            # Create contextual main CTA based on content type and mood
            main_cta_text = self._generate_contextual_cta(analysis)
            
            main_cta = TextClip(
                main_cta_text,
                fontsize=46,
                color='#FF4444',  # YouTube red
                stroke_color='white',
                stroke_width=3,
                font=str(self.config.get_font_path('BebasNeue-Regular.ttf')),
                method='caption',
                size=(video_clip.w * 0.9, None)  # Allow text wrapping
            )
            
            main_cta = main_cta.set_position(('center', 0.25), relative=True)
            main_cta = main_cta.set_start(start_time).set_duration(end_duration)
            main_cta = main_cta.crossfadein(0.5)
            
            # Add pulsing animation to main CTA
            main_cta = main_cta.resize(lambda t: 1 + 0.05 * np.sin(t * 4))
            end_elements.append(main_cta)
            
            # Enhanced secondary CTAs with icons and better spacing
            secondary_ctas = [
                "👍 LIKE for more epic content",
                "🔔 SUBSCRIBE & never miss out",
                "💬 COMMENT what you want to see next",
                "📤 SHARE with your friends"
            ]
            
            for i, cta_text in enumerate(secondary_ctas):
                cta_clip = TextClip(
                    cta_text,
                    fontsize=24,
                    color='#FFFFFF',
                    stroke_color='#333333',
                    stroke_width=1,
                    font=str(self.config.get_font_path('Montserrat-Bold.ttf'))
                )
                
                # Stagger appearance with faster timing
                cta_start = start_time + (i * 0.2)
                cta_duration = end_duration - (i * 0.2)
                
                if cta_duration > 0.5:
                    y_position = 0.55 + (i * 0.06)  # Better spacing
                    cta_clip = cta_clip.set_position(('center', y_position), relative=True)
                    cta_clip = cta_clip.set_start(cta_start).set_duration(cta_duration)
                    cta_clip = cta_clip.crossfadein(0.3)
                    
                    # Add subtle slide-in animation
                    cta_clip = cta_clip.set_position(lambda t: ('center', y_position + 20*(1-min(1, t*3))))
                    
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
            
            # Generate contextual CTA based on content mood and type
            mood = analysis.mood.lower() if analysis.mood else 'amazing'
            
            # Context-specific CTAs
            cta_templates = {
                'sports': "Subscribe for more epic sports moments!",
                'cycling': "Subscribe for more epic race moments!",
                'racing': "Subscribe for more epic race moments!",
                'intense': "Subscribe for more intense content!",
                'dramatic': "Subscribe for more dramatic moments!",
                'funny': "Subscribe for more hilarious content!",
                'amazing': "Subscribe for more amazing content!",
                'satisfying': "Subscribe for more satisfying videos!",
                'educational': "Subscribe for more learning content!",
                'uplifting': "Subscribe for more uplifting moments!"
            }
            
            # Check for sports/racing context in title or subreddit
            title_lower = analysis.suggested_title.lower() if analysis.suggested_title else ""
            
            if any(keyword in title_lower for keyword in ['race', 'racing', 'bike', 'cycling', 'cyclist']):
                return cta_templates.get('cycling', cta_templates['amazing'])
            elif any(keyword in title_lower for keyword in ['sport', 'athletic', 'competition']):
                return cta_templates.get('sports', cta_templates['amazing'])
            else:
                return cta_templates.get(mood, cta_templates['amazing'])
                
        except Exception as e:
            self.logger.warning(f"Error generating contextual CTA: {e}")
            return "Subscribe for more amazing content!"
    
    def add_auditory_ctas(self, 
                         audio_clip: AudioFileClip, 
                         analysis: VideoAnalysis,
                         video_duration: float) -> AudioFileClip:
        """
        Add auditory CTAs using TTS for verbal reminders
        
        Args:
            audio_clip: Source audio clip
            analysis: AI analysis with CTA requirements
            video_duration: Total video duration
            
        Returns:
            Audio with auditory CTAs added
        """
        try:
            # Import TTS service here to avoid circular import
            from src.integrations.tts_service import TTSService
            from src.models import NarrativeSegment, EmotionType, PacingType
            
            tts_service = TTSService()
            
            if not tts_service.is_available():
                self.logger.info("TTS not available for auditory CTAs")
                return audio_clip
            
            # Create auditory CTA segments
            cta_segments = []
            
            # Early like reminder (10% into video)
            like_time = video_duration * 0.1
            like_segment = NarrativeSegment(
                text="Smash that like button if you're enjoying this!",
                time_seconds=like_time,
                intended_duration_seconds=2.5,
                emotion=EmotionType.EXCITED,
                pacing=PacingType.FAST
            )
            cta_segments.append(like_segment)
            
            # Subscribe reminder (70% into video)
            sub_time = video_duration * 0.7
            sub_segment = NarrativeSegment(
                text="Don't forget to subscribe for more amazing content!",
                time_seconds=sub_time,
                intended_duration_seconds=3.0,
                emotion=EmotionType.EXCITED,
                pacing=PacingType.NORMAL
            )
            cta_segments.append(sub_segment)
            
            # Generate TTS for CTA segments
            tts_results = tts_service.generate_multiple_segments(cta_segments)
            
            # Add TTS audio to main audio
            audio_tracks = [audio_clip]
            
            for result in tts_results:
                if result['success'] and result['audio_path']:
                    try:
                        cta_audio = AudioFileClip(str(result['audio_path']))
                        segment = result['segment']
                        
                        # Set timing and volume
                        cta_audio = cta_audio.set_start(segment.time_seconds)
                        try:
                            cta_audio = cta_audio.with_volume_scaled(0.8)  # Slightly quieter than main content
                        except Exception as e:
                            self.logger.warning(f"Error applying volume to CTA audio: {e}")
                        
                        audio_tracks.append(cta_audio)
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing CTA audio: {e}")
            
            if len(audio_tracks) > 1:
                from moviepy.editor import CompositeAudioClip
                enhanced_audio = CompositeAudioClip(audio_tracks)
                self.logger.info(f"Added {len(audio_tracks)-1} auditory CTAs")
                return enhanced_audio
            else:
                return audio_clip
                
        except Exception as e:
            self.logger.warning(f"Error adding auditory CTAs: {e}")
            return audio_clip
    
    def create_engagement_hooks(self, 
                               video_clip: VideoFileClip, 
                               analysis: VideoAnalysis) -> VideoFileClip:
        """
        Add engagement hooks throughout the video to maintain attention
        
        Args:
            video_clip: Source video clip
            analysis: AI analysis with hook requirements
            
        Returns:
            Video with engagement hooks added
        """
        try:
            hook_overlays = []
            
            # Add curiosity gaps at strategic moments
            curiosity_times = [
                video_clip.duration * 0.25,  # 25% mark
                video_clip.duration * 0.60,  # 60% mark
            ]
            
            curiosity_texts = [
                "Wait for it... 🤯",
                "You won't believe what happens next! 👀",
                "The best part is coming up! 🔥",
                "This is getting crazy! 😱"
            ]
            
            for i, time_point in enumerate(curiosity_times):
                if time_point + 2.0 < video_clip.duration:
                    text = curiosity_texts[i % len(curiosity_texts)]
                    
                    hook_clip = TextClip(
                        text,
                        fontsize=38,
                        color='orange',
                        stroke_color='black',
                        stroke_width=2,
                        font=str(self.config.get_font_path('Montserrat-Bold.ttf'))
                    )
                    
                    hook_clip = hook_clip.set_position(('center', 0.15), relative=True)
                    hook_clip = hook_clip.set_start(time_point).set_duration(2.0)
                    hook_clip = hook_clip.crossfadein(0.3).crossfadeout(0.3)
                    
                    # Add subtle animation
                    hook_clip = hook_clip.resize(lambda t: 1 + 0.05 * np.sin(t * 8))
                    
                    hook_overlays.append(hook_clip)
            
            # Add progress indicators for longer videos
            if video_clip.duration > 45:
                progress_clip = self._create_progress_indicator(video_clip)
                if progress_clip:
                    hook_overlays.append(progress_clip)
            
            if hook_overlays:
                all_clips = [video_clip] + hook_overlays
                enhanced_video = CompositeVideoClip(all_clips)
                self.logger.info(f"Added {len(hook_overlays)} engagement hooks")
                return enhanced_video
            else:
                return video_clip
                
        except Exception as e:
            self.logger.warning(f"Error creating engagement hooks: {e}")
            return video_clip
    
    def _create_progress_indicator(self, video_clip: VideoFileClip) -> Optional[CompositeVideoClip]:
        """Create a subtle progress indicator for longer videos"""
        try:
            # Small progress bar at bottom
            bar_width = int(video_clip.w * 0.6)
            bar_height = 4
            
            # Background bar
            bg_bar = ColorClip(size=(bar_width, bar_height), color=(128, 128, 128))
            bg_bar = bg_bar.set_opacity(0.5)
            bg_bar = bg_bar.set_position(('center', video_clip.h - 30))
            bg_bar = bg_bar.set_duration(video_clip.duration)
            
            # Progress bar that grows
            def progress_width(t):
                progress = min(t / video_clip.duration, 1.0)
                return int(bar_width * progress), bar_height
            
            progress_bar = ColorClip(size=(bar_width, bar_height), color=(255, 100, 100))
            progress_bar = progress_bar.resize(progress_width)
            progress_bar = progress_bar.set_position(('center', video_clip.h - 30))
            progress_bar = progress_bar.set_duration(video_clip.duration)
            
            return CompositeVideoClip([bg_bar, progress_bar])
            
        except Exception as e:
            self.logger.warning(f"Error creating progress indicator: {e}")
            return None