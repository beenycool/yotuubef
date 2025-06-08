"""
MoviePy API compatibility fixes for video processing
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from moviepy import (
    VideoFileClip, AudioFileClip, CompositeVideoClip, CompositeAudioClip,
    TextClip, ImageClip, ColorClip, concatenate_videoclips, concatenate_audioclips,
    vfx, afx
)

class MoviePyCompat:
    """Compatibility layer for MoviePy API changes"""
    
    @staticmethod
    def subclip(clip, start_time, end_time=None):
        """Compatible subclip method"""
        if hasattr(clip, 'subclipped'):
            return clip.subclipped(start_time, end_time)
        elif hasattr(clip, 'subclip'):
            return clip.subclip(start_time, end_time)
        else:
            raise AttributeError(f"No subclip method found for {type(clip)}")
    
    @staticmethod
    def resize(clip, new_size_or_func):
        """Compatible resize method"""
        if hasattr(clip, 'resized'):
            return clip.resized(new_size_or_func)
        elif hasattr(clip, 'resize'):
            return clip.resize(new_size_or_func)
        else:
            raise AttributeError(f"No resize method found for {type(clip)}")
    
    @staticmethod
    def with_opacity(clip, opacity):
        """Compatible opacity method"""
        if hasattr(clip, 'with_opacity'):
            return clip.with_opacity(opacity)
        elif hasattr(clip, 'set_opacity'):
            return clip.set_opacity(opacity)
        else:
            raise AttributeError(f"No opacity method found for {type(clip)}")
    
    @staticmethod
    def with_position(clip, position):
        """Compatible position method"""
        if hasattr(clip, 'with_position'):
            return clip.with_position(position)
        elif hasattr(clip, 'set_position'):
            return clip.set_position(position)
        else:
            raise AttributeError(f"No position method found for {type(clip)}")
    
    @staticmethod
    def with_start(clip, start_time):
        """Compatible start time method"""
        if hasattr(clip, 'with_start'):
            return clip.with_start(start_time)
        elif hasattr(clip, 'set_start'):
            return clip.set_start(start_time)
        else:
            raise AttributeError(f"No start method found for {type(clip)}")
    
    @staticmethod
    def with_duration(clip, duration):
        """Compatible duration method"""
        if hasattr(clip, 'with_duration'):
            return clip.with_duration(duration)
        elif hasattr(clip, 'set_duration'):
            return clip.set_duration(duration)
        else:
            raise AttributeError(f"No duration method found for {type(clip)}")
    
    @staticmethod
    def apply_effect(clip, effect_func):
        """Compatible effect application"""
        try:
            # Try new API first
            if hasattr(clip, 'with_effects'):
                return clip.with_effects([effect_func])
            # Fall back to old API
            elif hasattr(clip, 'fl'):
                return clip.fl(effect_func)
            else:
                raise AttributeError(f"No effect method found for {type(clip)}")
        except Exception as e:
            logging.warning(f"Error applying effect: {e}")
            return clip
    
    @staticmethod
    def apply_fx_effect(clip, fx_effect, *args, **kwargs):
        """Compatible fx effect application"""
        try:
            # Try new API first
            if hasattr(clip, 'with_effects'):
                return clip.with_effects([fx_effect(*args, **kwargs)])
            # Fall back to old API
            elif hasattr(clip, 'fx'):
                return clip.fx(fx_effect, *args, **kwargs)
            else:
                raise AttributeError(f"No fx method found for {type(clip)}")
        except Exception as e:
            logging.warning(f"Error applying fx effect: {e}")
            return clip

    @staticmethod
    def create_text_clip(text, **kwargs):
        """Create text clip with compatible parameters"""
        # Handle font parameter conflicts
        if 'font' in kwargs and 'fontsize' in kwargs:
            # Remove duplicate font specification
            font_size = kwargs.pop('fontsize', 50)
            kwargs['font_size'] = font_size
        
        try:
            return TextClip(text, **kwargs)
        except Exception as e:
            logging.warning(f"Error creating text clip: {e}")
            # Try with minimal parameters
            return TextClip(text, color='white', font_size=50)

def ensure_shorts_format(clip: VideoFileClip, target_duration: float = 60.0) -> VideoFileClip:
    """
    Ensure video meets YouTube Shorts requirements:
    - Vertical format (9:16 aspect ratio)
    - Duration ≤ 60 seconds
    - Resolution optimized for mobile
    """
    logger = logging.getLogger(__name__)
    
    # Limit duration to 60 seconds for Shorts
    if clip.duration > target_duration:
        logger.info(f"Trimming video from {clip.duration:.1f}s to {target_duration}s for Shorts")
        clip = MoviePyCompat.subclip(clip, 0, target_duration)
    
    # Get current dimensions
    w, h = clip.size
    
    # Target vertical resolution for Shorts (9:16 aspect ratio)
    target_width = 1080
    target_height = 1920
    
    # Calculate aspect ratios
    current_aspect = w / h
    target_aspect = target_width / target_height  # 9:16 = 0.5625
    
    logger.info(f"Current aspect ratio: {current_aspect:.3f}, target: {target_aspect:.3f}")
    
    if abs(current_aspect - target_aspect) > 0.1:  # Needs aspect ratio correction
        if current_aspect > target_aspect:  # Too wide, crop sides
            new_width = int(h * target_aspect)
            x1 = (w - new_width) // 2
            clip = clip.crop(x1=x1, x2=x1 + new_width)
            logger.info(f"Cropped width from {w} to {new_width}")
        else:  # Too tall, crop top/bottom
            new_height = int(w / target_aspect)
            y1 = (h - new_height) // 2
            clip = clip.crop(y1=y1, y2=y1 + new_height)
            logger.info(f"Cropped height from {h} to {new_height}")
    
    # Resize to target resolution
    clip = MoviePyCompat.resize(clip, (target_width, target_height))
    logger.info(f"Resized to {target_width}x{target_height} for YouTube Shorts")
    
    return clip