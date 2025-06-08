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
            # Ensure effect_func is properly wrapped
            if not callable(effect_func):
                logging.warning(f"Effect function is not callable: {type(effect_func)}")
                return clip
            
            # Try new API first - wrap function properly
            if hasattr(clip, 'with_effects'):
                # For newer MoviePy versions, need to wrap in proper effect format
                def wrapped_effect(clip_inner):
                    return clip_inner.fl(effect_func)
                return clip.with_effects([wrapped_effect])
            # Fall back to old API
            elif hasattr(clip, 'fl'):
                return clip.fl(effect_func)
            elif hasattr(clip, 'fl_image'):
                return clip.fl_image(effect_func)
            else:
                # Final fallback - return original clip if no effect methods work
                logging.warning(f"No compatible effect method found for {type(clip)}, returning original")
                return clip
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
        # Clean up parameter conflicts
        clean_kwargs = {}
        
        # Handle font/fontsize conflicts
        if 'fontsize' in kwargs:
            clean_kwargs['fontsize'] = kwargs['fontsize']
        elif 'font_size' in kwargs:
            clean_kwargs['fontsize'] = kwargs['font_size']
        
        # Handle other parameters, avoiding conflicts
        for key, value in kwargs.items():
            if key not in ['font_size']:  # Skip conflicting parameters
                clean_kwargs[key] = value
        
        try:
            return TextClip(text, **clean_kwargs)
        except Exception as e:
            logging.warning(f"Error creating text clip: {e}")
            # Try with minimal parameters
            try:
                return TextClip(text, color='white', fontsize=50)
            except Exception as e2:
                logging.warning(f"Fallback text clip creation failed: {e2}")
                # Last resort: basic text clip
                return TextClip(text)
    
    @staticmethod
    def crop(clip, x1=None, y1=None, x2=None, y2=None, width=None, height=None):
        """Compatible crop method"""
        try:
            # Try new API first
            if hasattr(clip, 'cropped'):
                return clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2, width=width, height=height)
            # Fall back to old API
            elif hasattr(clip, 'crop'):
                return clip.crop(x1=x1, y1=y1, x2=x2, y2=y2, width=width, height=height)
            else:
                # Manual crop using subclip if direct crop not available
                w, h = clip.size
                x1 = x1 or 0
                y1 = y1 or 0
                x2 = x2 or w
                y2 = y2 or h
                
                # Use resize to achieve crop effect
                new_w = x2 - x1
                new_h = y2 - y1
                
                def crop_frame(get_frame, t):
                    frame = get_frame(t)
                    return frame[y1:y2, x1:x2]
                
                return MoviePyCompat.apply_effect(clip, crop_frame).resize((new_w, new_h))
        except Exception as e:
            logging.warning(f"Error cropping clip: {e}")
            return clip
    
    @staticmethod
    def get_audio_channels(clip):
        """Safely get number of audio channels"""
        try:
            # Try different attribute names for channel count
            if hasattr(clip, 'nchannels'):
                return clip.nchannels
            elif hasattr(clip, 'audio') and hasattr(clip.audio, 'nchannels'):
                return clip.audio.nchannels
            elif hasattr(clip, 'reader') and hasattr(clip.reader, 'nchannels'):
                return clip.reader.nchannels
            else:
                # Default to stereo if can't determine
                return 2
        except Exception as e:
            logging.warning(f"Could not determine audio channels: {e}")
            return 2

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
            clip = MoviePyCompat.crop(clip, x1=x1, x2=x1 + new_width)
            logger.info(f"Cropped width from {w} to {new_width}")
        else:  # Too tall, crop top/bottom
            new_height = int(w / target_aspect)
            y1 = (h - new_height) // 2
            clip = MoviePyCompat.crop(clip, y1=y1, y2=y1 + new_height)
            logger.info(f"Cropped height from {h} to {new_height}")
    
    # Resize to target resolution
    clip = MoviePyCompat.resize(clip, (target_width, target_height))
    logger.info(f"Resized to {target_width}x{target_height} for YouTube Shorts")
    
    return clip