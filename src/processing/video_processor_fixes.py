"""
MoviePy API compatibility fixes for video processing
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from pathlib import Path
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
        if clip is None:
            logging.warning("Cannot resize None object")
            return None
        try:
            if hasattr(clip, 'resized'):
                return clip.resized(new_size_or_func)
            elif hasattr(clip, 'resize'):
                return clip.resize(new_size_or_func)
            else:
                logging.warning(f"No resize method found for {type(clip).__name__}")
                return clip
        except Exception as e:
            logging.warning(f"Error during resize: {e}")
            return clip
    
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
        try:
            if hasattr(clip, 'with_position'):
                return clip.with_position(position)
            elif hasattr(clip, 'set_position'):
                return clip.set_position(position)
            else:
                # For TextClip in newer versions, position might be set differently
                if hasattr(clip, 'at'):
                    return clip.at(position)
                else:
                    logging.warning(f"No position method found for {type(clip).__name__}")
                    return clip
        except Exception as e:
            logging.warning(f"Error setting position: {e}")
            return clip
    
    @staticmethod
    def with_start(clip, start_time):
        """Compatible start time method"""
        try:
            if hasattr(clip, 'with_start'):
                return clip.with_start(start_time)
            elif hasattr(clip, 'set_start'):
                return clip.set_start(start_time)
            elif hasattr(clip, 'subclipped'):
                # If we can't set start directly, use subclip from start_time
                return clip.subclipped(start_time)
            else:
                logging.warning(f"No start method found for {type(clip).__name__}")
                return clip
        except Exception as e:
            logging.warning(f"Error setting start time: {e}")
            return clip
    
    @staticmethod
    def with_duration(clip, duration):
        """Compatible duration method"""
        try:
            if hasattr(clip, 'with_duration'):
                return clip.with_duration(duration)
            elif hasattr(clip, 'set_duration'):
                return clip.set_duration(duration)
            elif hasattr(clip, 'subclipped'):
                # Use subclip to set duration
                start = getattr(clip, 'start', 0)
                return clip.subclipped(start, start + duration)
            else:
                logging.warning(f"No duration method found for {type(clip).__name__}")
                return clip
        except Exception as e:
            logging.warning(f"Error setting duration: {e}")
            return clip
    
    @staticmethod
    def apply_effect(clip, effect_func):
        """Compatible effect application with improved error handling for MoviePy 2.1.2+"""
        try:
            # Ensure effect_func is properly callable
            if not callable(effect_func):
                logging.warning(f"Effect function is not callable: {type(effect_func)}")
                return clip
            
            # For newer MoviePy versions, we need to handle fl() differently
            try:
                # Try the standard fl() method first
                if hasattr(clip, 'fl'):
                    return clip.fl(effect_func)
                elif hasattr(clip, 'fl_image'):
                    return clip.fl_image(effect_func)
                elif hasattr(clip, 'fl_time'):
                    # Try time-based effect for audio clips
                    return clip.fl_time(effect_func)
                else:
                    # For clips without fl methods, try to apply effect manually
                    logging.debug(f"No fl method found for {type(clip).__name__}, trying manual application")
                    
                    # For simple clips like ColorClip or ImageClip, return as-is
                    if hasattr(clip, 'make_frame'):
                        try:
                            # Try to create a lambda effect
                            new_clip = clip.copy()
                            new_clip.make_frame = lambda t: effect_func(clip.make_frame, t)
                            return new_clip
                        except:
                            pass
                    
                    # Return original clip if effect can't be applied
                    logging.debug(f"Returning original {type(clip).__name__} clip - effect not applicable")
                    return clip
                    
            except Exception as e:
                # If fl() fails, try alternative approaches
                logging.debug(f"Standard effect application failed: {e}")
                
                # Try creating a new clip with the effect applied manually
                try:
                    # Create a new clip by applying the effect frame by frame
                    def effect_wrapper(get_frame, t):
                        try:
                            return effect_func(get_frame, t)
                        except Exception as inner_e:
                            logging.debug(f"Effect function failed at t={t}: {inner_e}")
                            # Return original frame on error
                            return get_frame(t)
                    
                    if hasattr(clip, 'fl'):
                        return clip.fl(effect_wrapper)
                    else:
                        # Last resort: return original clip
                        logging.debug("Could not apply effect, returning original clip")
                        return clip
                        
                except Exception as e2:
                    logging.debug(f"Alternative effect application failed: {e2}")
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
        """Create text clip with compatible parameters and robust fallback handling for MoviePy 2.1.2+"""
        # Clean up parameter conflicts and map to new API
        clean_kwargs = {}
        
        # Map old parameter names to new ones for MoviePy 2.1.2+
        param_mapping = {
            'fontsize': 'font_size',
            'fontpath': 'font',
            'align': 'text_align'
        }
        
        # Set the text parameter first
        clean_kwargs['text'] = text
        
        # Handle parameter mapping
        for old_param, new_param in param_mapping.items():
            if old_param in kwargs:
                clean_kwargs[new_param] = kwargs[old_param]
        
        # Handle direct parameters (already using new names)
        for key, value in kwargs.items():
            if key not in param_mapping and key not in clean_kwargs:
                clean_kwargs[key] = value
        
        # Ensure we have sensible defaults
        defaults = {
            'color': 'white',
            'font_size': 50,
            'method': 'label',
            'text_align': 'center',
            'horizontal_align': 'center',
            'vertical_align': 'center'
        }
        
        for key, default_value in defaults.items():
            if key not in clean_kwargs:
                clean_kwargs[key] = default_value
        
        # Check font availability and provide fallback
        if 'font' in clean_kwargs:
            font_path = clean_kwargs['font']
            if not Path(font_path).exists() and font_path != 'Arial':
                logging.warning(f"Font file not found: {font_path}, falling back to Arial")
                clean_kwargs['font'] = 'Arial'
        
        # Try different fallback strategies
        fallback_attempts = [
            # First attempt: Use all cleaned kwargs
            lambda: TextClip(**clean_kwargs),
            
            # Second attempt: Remove problematic font if it fails
            lambda: TextClip(**{k: v for k, v in clean_kwargs.items() if k != 'font'}),
            
            # Third attempt: Remove stroke parameters that might cause issues
            lambda: TextClip(**{k: v for k, v in clean_kwargs.items() 
                              if k not in ['font', 'stroke_color', 'stroke_width']}),
            
            # Fourth attempt: Use absolute minimal parameters with Arial
            lambda: TextClip(
                text=text,
                color=clean_kwargs.get('color', 'white'),
                font_size=clean_kwargs.get('font_size', 50),
                font='Arial'
            ),
            
            # Fifth attempt: Use absolute minimal parameters without font
            lambda: TextClip(
                text=text,
                color=clean_kwargs.get('color', 'white'),
                font_size=clean_kwargs.get('font_size', 50)
            ),
            
            # Last resort: Just text and color
            lambda: TextClip(text=text, color='white')
        ]
        
        for i, attempt in enumerate(fallback_attempts):
            try:
                result = attempt()
                if i > 0:
                    logging.warning(f"Text clip created using fallback method {i+1}")
                return result
            except Exception as e:
                logging.debug(f"Text clip attempt {i+1} failed: {e}")
                continue
        
        # If all attempts fail, log error and return None
        logging.error("All text clip creation attempts failed")
        return None
    
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
    
    @staticmethod
    def crossfadein(clip, duration):
        """Compatible crossfadein method"""
        if clip is None:
            logging.warning("No crossfadein method available for NoneType")
            return None
        try:
            if hasattr(clip, 'crossfadein'):
                result = clip.crossfadein(duration)
                return result if result is not None else clip
            elif hasattr(clip, 'with_effects'):
                # Use fade effect for newer MoviePy versions
                try:
                    if hasattr(clip, 'audio') and clip.audio is not None:
                        # Audio clip
                        from moviepy.audio.fx import audio_fadein
                        result = clip.with_effects([audio_fadein(duration)])
                        return result if result is not None else clip
                    else:
                        # Video clip - use opacity fade
                        from moviepy.video.fx import fadein
                        result = clip.with_effects([fadein(duration)])
                        return result if result is not None else clip
                except (ImportError, AttributeError):
                    # Fallback if fx modules aren't available
                    pass
            elif hasattr(clip, 'fx'):
                # Use fx for older versions
                try:
                    if hasattr(clip, 'audio') and clip.audio is not None:
                        from moviepy.audio.fx import audio_fadein
                        result = clip.fx(audio_fadein, duration)
                        return result if result is not None else clip
                    else:
                        from moviepy.video.fx import fadein
                        result = clip.fx(fadein, duration)
                        return result if result is not None else clip
                except (ImportError, AttributeError):
                    # Fallback if fx modules aren't available
                    pass
            else:
                # Manual fade using opacity
                def fade_func(t):
                    if t < duration:
                        return t / duration
                    return 1.0
                
                if hasattr(clip, 'with_opacity'):
                    return clip.with_opacity(fade_func)
                elif hasattr(clip, 'set_opacity'):
                    return clip.set_opacity(fade_func)
                else:
                    logging.warning(f"No crossfadein method available for {type(clip).__name__}")
                    return clip
        except Exception as e:
            logging.warning(f"Error applying crossfadein: {e}")
            return clip
    
    @staticmethod
    def crossfadeout(clip, duration):
        """Compatible crossfadeout method"""
        if clip is None:
            logging.warning("No crossfadeout method available for NoneType")
            return None
        try:
            if hasattr(clip, 'crossfadeout'):
                result = clip.crossfadeout(duration)
                return result if result is not None else clip
            elif hasattr(clip, 'with_effects'):
                # Use fade effect for newer MoviePy versions
                try:
                    if hasattr(clip, 'audio') and clip.audio is not None:
                        # Audio clip
                        from moviepy.audio.fx import audio_fadeout
                        result = clip.with_effects([audio_fadeout(duration)])
                        return result if result is not None else clip
                    else:
                        # Video clip - use opacity fade
                        from moviepy.video.fx import fadeout
                        result = clip.with_effects([fadeout(duration)])
                        return result if result is not None else clip
                except (ImportError, AttributeError):
                    # Fallback if fx modules aren't available
                    pass
            elif hasattr(clip, 'fx'):
                # Use fx for older versions
                try:
                    if hasattr(clip, 'audio') and clip.audio is not None:
                        from moviepy.audio.fx import audio_fadeout
                        result = clip.fx(audio_fadeout, duration)
                        return result if result is not None else clip
                    else:
                        from moviepy.video.fx import fadeout
                        result = clip.fx(fadeout, duration)
                        return result if result is not None else clip
                except (ImportError, AttributeError):
                    # Fallback if fx modules aren't available
                    pass
            else:
                # Manual fade using opacity
                clip_duration = getattr(clip, 'duration', 1.0)
                def fade_func(t):
                    if t > clip_duration - duration:
                        return (clip_duration - t) / duration
                    return 1.0
                
                if hasattr(clip, 'with_opacity'):
                    return clip.with_opacity(fade_func)
                elif hasattr(clip, 'set_opacity'):
                    return clip.set_opacity(fade_func)
                else:
                    logging.warning(f"No crossfadeout method available for {type(clip).__name__}")
                    return clip
        except Exception as e:
            logging.warning(f"Error applying crossfadeout: {e}")
            return clip

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