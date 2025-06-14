"""
Modular video processing pipeline for YouTube Shorts generation.
Handles video downloading, processing, effects, and optimization.
Optimized for GPU memory usage with FFmpeg GPU acceleration.
"""

import gc
import logging
import math
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import subprocess
import psutil
import os
import time

import cv2
import numpy as np
from moviepy import (
    VideoFileClip, AudioFileClip, CompositeVideoClip, CompositeAudioClip,
    TextClip, ImageClip, ColorClip, concatenate_videoclips, concatenate_audioclips
)

# Import video effects with compatibility handling
try:
    from moviepy import vfx, afx
except ImportError:
    try:
        from moviepy.video import fx as vfx
        from moviepy.audio import fx as afx
    except ImportError:
        vfx = None
        afx = None
import yt_dlp

from src.config.settings import get_config
from src.models import VideoAnalysis, TextOverlay, NarrativeSegment, VisualCue, CallToAction
from src.integrations.tts_service import TTSService
from src.processing.cta_processor import CTAProcessor
from src.processing.thumbnail_generator import ThumbnailGenerator
from src.processing.sound_effects_manager import SoundEffectsManager
from src.processing.video_processor_fixes import MoviePyCompat, ensure_shorts_format
from src.utils.gpu_memory_manager import GPUMemoryManager
from src.processing.speed_optimizer import create_speed_optimizer


class ResourceManager:
    """Manages system resources during video processing with GPU memory optimization"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.active_clips = []
        self.gpu_manager = GPUMemoryManager(max_vram_usage=0.85)  # Use 85% VRAM for maximum performance
        
        # Set FFmpeg to use GPU memory when available
        self._setup_ffmpeg_gpu_memory()
    
    def _setup_ffmpeg_gpu_memory(self):
        """Setup FFmpeg to use GPU memory when available"""
        if self.gpu_manager.device != "cpu":
            # Set environment variables for FFmpeg GPU memory optimization
            os.environ['FFMPEG_NVENC_MEMORY_POOL_SIZE'] = '1024M'  # Increased from 512M to 1024M
            os.environ['FFMPEG_CUDA_DEVICE'] = '0'  # Use first GPU
            os.environ['FFMPEG_HWACCEL_OUTPUT_FORMAT'] = 'cuda'  # Keep frames in GPU memory
            os.environ['FFMPEG_NVENC_PRESET'] = 'p5'  # Balanced preset
            os.environ['FFMPEG_NVENC_RC'] = 'vbr'  # Variable bitrate for better quality
            self.logger.info("FFmpeg configured for GPU memory usage with increased memory pool")
        else:
            self.logger.info("FFmpeg using CPU mode")
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within safe limits for both RAM and VRAM"""
        try:
            # Check system RAM
            memory = psutil.virtual_memory()
            ram_ok = memory.percent < (self.config.video.max_memory_usage * 100)
            
            # Check VRAM if GPU is available
            vram_ok = True
            if self.gpu_manager.device != "cpu":
                vram_info = self.gpu_manager.get_vram_info()
                if vram_info:
                    vram_percent = (vram_info['used'] / vram_info['total']) * 100
                    vram_ok = vram_percent < 90  # Keep 10% VRAM free
                    
                    if not vram_ok:
                        self.logger.warning(f"High VRAM usage: {vram_percent:.1f}%")
            
            return ram_ok and vram_ok
            
        except Exception as e:
            self.logger.warning(f"Could not check memory usage: {e}")
            return True
    
    def register_clip(self, clip):
        """Register a clip for resource tracking"""
        if clip and hasattr(clip, 'close'):
            self.active_clips.append(clip)
    
    def cleanup_clips(self):
        """Clean up all registered clips"""
        for clip in self.active_clips:
            try:
                if hasattr(clip, 'close'):
                    clip.close()
            except Exception as e:
                self.logger.warning(f"Error closing clip: {e}")
        
        self.active_clips.clear()
        gc.collect()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_clips()


class VideoDownloader:
    """Handles video downloading from various sources"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def download_video(self, url: str, output_path: Path) -> bool:
        """
        Download video from URL using yt-dlp
        
        Args:
            url: Video URL to download
            output_path: Path where to save the video
        
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Configure format string based on URL type with improved Reddit handling
            if 'v.redd.it' in url or 'reddit.com' in url:
                # For Reddit videos, use progressive format selection with fallbacks
                format_selector = 'best[ext=mp4][height<=1080]/best[ext=webm][height<=1080]/best[height<=720]/worst[ext=mp4]/worst'
            else:
                # For other platforms, prefer 1080p or lower
                format_selector = 'best[height<=1080]/best'
            
            # Configure yt-dlp options
            ydl_opts = {
                'format': format_selector,
                'outtmpl': str(output_path.with_suffix('.%(ext)s')),
                'quiet': True,
                'no_warnings': True,
                'extractaudio': False,  # Keep as video download
                'audioformat': 'mp3',
                'embed_chapters': False,
                'embed_info': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
                # Add Reddit-specific options
                'ignore_errors': False,
                'no_check_certificate': True,
            }
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Download the video
                    ydl.download([url])
            except Exception as download_error:
                # Enhanced error handling with multiple fallback strategies
                error_msg = str(download_error).lower()
                if any(phrase in error_msg for phrase in ['requested format is not available', 'no video formats found', 'format not found']):
                    self.logger.warning(f"Primary format failed for {url}, trying progressive fallbacks")
                    
                    # Progressive fallback strategy
                    fallback_formats = [
                        'worst[ext=mp4]/worst[ext=webm]/worst',  # Any available format
                        'bestvideo+bestaudio/best',              # Separate video+audio
                        'bestvideo/best',                        # Video only if needed
                        '(mp4,webm,mkv,avi)[height<=1080]/(mp4,webm,mkv,avi)', # Specific containers
                    ]
                    
                    success = False
                    for i, fallback_format in enumerate(fallback_formats):
                        try:
                            self.logger.info(f"Trying fallback format {i+1}/{len(fallback_formats)}: {fallback_format}")
                            fallback_opts = ydl_opts.copy()
                            fallback_opts['format'] = fallback_format
                            fallback_opts['quiet'] = False  # Enable output for debugging
                            
                            with yt_dlp.YoutubeDL(fallback_opts) as ydl_fallback:
                                ydl_fallback.download([url])
                            success = True
                            break
                            
                        except Exception as fallback_error:
                            self.logger.debug(f"Fallback format {i+1} failed: {fallback_error}")
                            continue
                    
                    if not success:
                        # Final attempt with generic extractor for Reddit
                        if 'v.redd.it' in url or 'reddit.com' in url:
                            self.logger.warning(f"All fallbacks failed, trying generic Reddit extraction")
                            return self._try_generic_reddit_extraction(url, output_path)
                        else:
                            raise download_error
                else:
                    raise download_error
            
            # Find the downloaded file (yt-dlp may change the extension)
            downloaded_files = list(output_path.parent.glob(f"{output_path.stem}.*"))
            video_extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov']
            
            for file_path in downloaded_files:
                if file_path.suffix.lower() in video_extensions:
                    # Rename to expected output path
                    final_path = output_path.with_suffix(file_path.suffix)
                    if file_path != final_path:
                        file_path.rename(final_path)
                    
                    self.logger.info(f"Successfully downloaded video to {final_path}")
                    return True
            
            self.logger.error(f"No video file found after download from {url}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error downloading video from {url}: {e}")
            return False
    
    def download_audio(self, url: str, output_path: Path, audio_format: str = 'mp3') -> bool:
        """
        Download audio-only from URL using yt-dlp
        
        Args:
            url: Video URL to extract audio from
            output_path: Path where to save the audio file
            audio_format: Audio format (mp3, wav, m4a, etc.)
        
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Configure yt-dlp options for audio extraction
            ydl_opts = {
                'format': 'bestaudio/best',  # Get best audio quality
                'outtmpl': str(output_path.with_suffix(f'.%(ext)s')),
                'quiet': True,
                'no_warnings': True,
                'extractaudio': True,  # Enable audio extraction
                'audioformat': audio_format,
                'audioquality': '0',  # Best quality
                'embed_chapters': False,
                'embed_info': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
                'ignore_errors': False,
                'no_check_certificate': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': audio_format,
                    'preferredquality': '192' if audio_format == 'mp3' else 'best',
                }],
            }
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Download and extract audio
                    self.logger.info(f"Extracting audio from {url} as {audio_format}")
                    ydl.download([url])
                    
            except Exception as download_error:
                # Fallback strategies for audio extraction
                error_msg = str(download_error).lower()
                if any(phrase in error_msg for phrase in ['requested format is not available', 'no audio formats found']):
                    self.logger.warning(f"Primary audio format failed for {url}, trying fallbacks")
                    
                    # Try different audio extraction strategies
                    fallback_strategies = [
                        {
                            'format': 'worst',  # Any available format
                            'extractaudio': True,
                            'audioformat': audio_format,
                        },
                        {
                            'format': 'bestaudio[ext=m4a]/bestaudio',  # Prefer m4a, then any audio
                            'extractaudio': True,
                            'audioformat': audio_format,
                        },
                        {
                            'format': 'best',  # Download full video then extract
                            'extractaudio': True,
                            'audioformat': audio_format,
                        }
                    ]
                    
                    success = False
                    for i, strategy in enumerate(fallback_strategies):
                        try:
                            self.logger.info(f"Trying audio fallback strategy {i+1}/{len(fallback_strategies)}")
                            fallback_opts = ydl_opts.copy()
                            fallback_opts.update(strategy)
                            fallback_opts['quiet'] = False  # Enable output for debugging
                            
                            with yt_dlp.YoutubeDL(fallback_opts) as ydl_fallback:
                                ydl_fallback.download([url])
                            success = True
                            break
                            
                        except Exception as fallback_error:
                            self.logger.debug(f"Audio fallback strategy {i+1} failed: {fallback_error}")
                            continue
                    
                    if not success:
                        raise download_error
                else:
                    raise download_error
            
            # Find the downloaded audio file
            downloaded_files = list(output_path.parent.glob(f"{output_path.stem}.*"))
            audio_extensions = [f'.{audio_format}', '.m4a', '.webm', '.opus', '.wav']
            
            for file_path in downloaded_files:
                if file_path.suffix.lower() in audio_extensions:
                    # Rename to expected output path with correct extension
                    final_path = output_path.with_suffix(f'.{audio_format}')
                    if file_path != final_path:
                        file_path.rename(final_path)
                    
                    self.logger.info(f"Successfully downloaded audio to {final_path}")
                    return True
            
            self.logger.error(f"No audio file found after download from {url}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error downloading audio from {url}: {e}")
            return False
    
    def download_both(self, url: str, video_output_path: Path, audio_output_path: Path, 
                     audio_format: str = 'mp3') -> Dict[str, bool]:
        """
        Download both video and audio from URL
        
        Args:
            url: Video URL to download from
            video_output_path: Path for video file
            audio_output_path: Path for audio file
            audio_format: Audio format for extraction
        
        Returns:
            Dict with 'video' and 'audio' success status
        """
        results = {
            'video': False,
            'audio': False
        }
        
        # Download video first
        try:
            results['video'] = self.download_video(url, video_output_path)
            self.logger.info(f"Video download {'successful' if results['video'] else 'failed'}")
        except Exception as e:
            self.logger.error(f"Video download failed: {e}")
        
        # Download audio separately
        try:
            results['audio'] = self.download_audio(url, audio_output_path, audio_format)
            self.logger.info(f"Audio download {'successful' if results['audio'] else 'failed'}")
        except Exception as e:
            self.logger.error(f"Audio download failed: {e}")
        
        return results
    
    def _try_generic_reddit_extraction(self, url: str, output_path: Path) -> bool:
        """
        Try generic Reddit video extraction as a last resort
        
        Args:
            url: Reddit video URL
            output_path: Output path for the video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Attempting generic Reddit video extraction")
            
            # Use most permissive options for generic extraction
            generic_opts = {
                'format': None,  # Don't specify format, let yt-dlp decide
                'outtmpl': str(output_path.with_suffix('.%(ext)s')),
                'quiet': False,
                'no_warnings': False,
                'ignore_errors': True,
                'no_check_certificate': True,
                'extract_flat': False,
                'force_generic_extractor': True,  # Force generic extractor
                'writesubtitles': False,
                'writeautomaticsub': False,
                'embed_chapters': False,
                'embed_info': False,
            }
            
            with yt_dlp.YoutubeDL(generic_opts) as ydl_generic:
                ydl_generic.download([url])
            
            # Check if any video file was downloaded
            downloaded_files = list(output_path.parent.glob(f"{output_path.stem}.*"))
            video_extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv']
            
            for file_path in downloaded_files:
                if file_path.suffix.lower() in video_extensions:
                    final_path = output_path.with_suffix(file_path.suffix)
                    if file_path != final_path:
                        file_path.rename(final_path)
                    
                    self.logger.info(f"Successfully extracted Reddit video with generic method: {final_path}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Generic Reddit extraction failed: {e}")
            return False
    
    def get_video_info(self, url: str) -> Optional[Dict[str, Any]]:
        """Get video information without downloading"""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return {
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0),
                    'width': info.get('width', 0),
                    'height': info.get('height', 0),
                    'fps': info.get('fps', 30),
                    'format': info.get('ext', 'unknown'),
                    'formats': info.get('formats', [])  # Include available formats for debugging
                }
                
        except Exception as e:
            self.logger.error(f"Error getting video info from {url}: {e}")
            return None
    
    def list_available_formats(self, url: str) -> List[str]:
        """List available formats for a video URL (useful for debugging)"""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
                'listformats': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                formats = info.get('formats', [])
                
                format_list = []
                for fmt in formats:
                    format_str = f"{fmt.get('format_id', 'unknown')} - {fmt.get('ext', 'unknown')}"
                    if fmt.get('height'):
                        format_str += f" - {fmt.get('height')}p"
                    if fmt.get('vcodec'):
                        format_str += f" - {fmt.get('vcodec')}"
                    format_list.append(format_str)
                
                return format_list
                
        except Exception as e:
            self.logger.error(f"Error listing formats for {url}: {e}")
            return []


class VideoEffects:
    """Video effects and enhancements"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def apply_subtle_zoom(self, clip: VideoFileClip, zoom_factor: float = 1.05) -> VideoFileClip:
        """Apply subtle zoom effect for engagement"""
        if not self.config.effects.subtle_zoom_enabled:
            return clip
        
        try:
            def zoom_effect(get_frame, t):
                frame = get_frame(t)
                progress = t / clip.duration
                current_zoom = 1 + (zoom_factor - 1) * progress
                
                h, w = frame.shape[:2]
                new_h, new_w = int(h / current_zoom), int(w / current_zoom)
                
                # Calculate center crop
                y1 = (h - new_h) // 2
                x1 = (w - new_w) // 2
                y2 = y1 + new_h
                x2 = x1 + new_w
                
                cropped = frame[y1:y2, x1:x2]
                return cv2.resize(cropped, (w, h))
            
            return MoviePyCompat.apply_effect(clip, zoom_effect)
            
        except Exception as e:
            self.logger.warning(f"Error applying zoom effect: {e}")
            return clip
    
    def apply_color_grading(self, clip: VideoFileClip, intensity: float = 0.7) -> VideoFileClip:
        """Apply color grading for visual enhancement"""
        if not self.config.effects.color_grade_enabled:
            return clip
        
        try:
            # Apply brightness and contrast adjustments
            clip = MoviePyCompat.apply_fx_effect(clip, vfx.colorx, factor=1 + intensity * 0.1)  # Slight brightness
            clip = MoviePyCompat.apply_fx_effect(clip, vfx.lum_contrast, lum=0, contrast=intensity * 0.2)  # Contrast
            
            return clip
            
        except Exception as e:
            self.logger.warning(f"Error applying color grading: {e}")
            return clip
    
    def apply_speed_effects(self, clip: VideoFileClip, speed_effects: List[Dict]) -> VideoFileClip:
        """Apply speed effects based on AI analysis with enhanced transitions"""
        if not speed_effects:
            return clip
        
        try:
            segments = []
            current_time = 0
            
            for effect in speed_effects:
                start_time = getattr(effect, 'start_seconds', 0)
                end_time = getattr(effect, 'end_seconds', clip.duration)
                speed_factor = getattr(effect, 'speed_factor', 1.0)
                
                # Validate speed factor
                speed_factor = max(0.5, min(2.0, speed_factor))  # Keep within reasonable bounds
                
                # Add normal segment before effect if needed
                if current_time < start_time:
                    normal_segment = MoviePyCompat.subclip(clip, current_time, start_time)
                    segments.append(normal_segment)
                
                # Add speed-affected segment with transition
                effect_segment = MoviePyCompat.subclip(clip, start_time, end_time)
                if speed_factor != 1.0:
                    try:
                        effect_segment = effect_segment.with_fps(effect_segment.fps * speed_factor)
                    except Exception as e:
                        self.logger.warning(f"Error applying speed effect: {e}")
                    
                    # Add transition if this isn't the first segment
                    if segments and speed_factor > 1.0:  # Only for speed-ups
                        transition_duration = min(0.3, (end_time - start_time) * 0.2)
                        effect_segment = MoviePyCompat.crossfadein(effect_segment, transition_duration)
                
                segments.append(effect_segment)
                current_time = end_time
            
            # Add remaining segment if any
            if current_time < clip.duration:
                segments.append(MoviePyCompat.subclip(clip, current_time, clip.duration))
            
            # Combine segments with transitions
            if len(segments) > 1:
                final_clip = concatenate_videoclips(segments, method="compose")
                return final_clip
            elif segments:
                return segments[0]
            else:
                return clip
                
        except Exception as e:
            self.logger.warning(f"Error applying speed effects: {e}", exc_info=True)
            return clip
    
    def add_visual_cues(self, clip: VideoFileClip, visual_cues: List[VisualCue]) -> VideoFileClip:
        """Add visual cues like zoom, highlights, etc."""
        if not visual_cues:
            return clip
        
        try:
            clips_to_composite = [clip]
            
            for cue in visual_cues:
                if cue.effect_type == "zoom":
                    # Create zoom highlight effect
                    zoom_clip = self._create_zoom_highlight(
                        clip, cue.timestamp_seconds, cue.duration, cue.intensity
                    )
                    if zoom_clip:
                        clips_to_composite.append(zoom_clip)
                
                elif cue.effect_type == "highlight":
                    # Create highlight effect
                    highlight_clip = self._create_highlight_effect(
                        clip, cue.timestamp_seconds, cue.duration
                    )
                    if highlight_clip:
                        clips_to_composite.append(highlight_clip)
            
            if len(clips_to_composite) > 1:
                return CompositeVideoClip(clips_to_composite)
            else:
                return clip
                
        except Exception as e:
            self.logger.warning(f"Error adding visual cues: {e}")
            return clip
    
    def _create_zoom_highlight(self, clip: VideoFileClip, timestamp: float, duration: float, intensity: float) -> Optional[VideoFileClip]:
        """Create a zoom highlight effect"""
        try:
            # Create a subtle overlay to indicate zoom focus
            w, h = clip.size
            
            # Create a translucent overlay
            overlay = ColorClip(size=(w, h), color=(255, 255, 255))
            overlay = MoviePyCompat.with_opacity(overlay, 0.1 * intensity)
            overlay = MoviePyCompat.with_start(MoviePyCompat.with_duration(overlay, duration), timestamp)
            
            return overlay
            
        except Exception as e:
            self.logger.warning(f"Error creating zoom highlight: {e}")
            return None
    
    def apply_dynamic_pan_zoom(self, clip: VideoFileClip, focus_points: List[Dict[str, Any]]) -> VideoFileClip:
        """
        Apply dynamic panning and zooming with enhanced motion smoothing and edge handling.
        Uses AI-identified focus points to create professional cinematic movements.
        
        Args:
            clip: Input video clip
            focus_points: List of focus points with x, y coordinates and timestamps
            
        Returns:
            Video clip with dynamic panning and zooming applied
        """
        if not focus_points or len(focus_points) < 2:
            self.logger.info("Insufficient focus points for dynamic pan/zoom, applying subtle zoom instead")
            return self.apply_subtle_zoom(clip)
        
        try:
            # Sort focus points by timestamp and validate
            sorted_points = sorted(focus_points, key=lambda x: x.get('timestamp_seconds', 0))
            self._validate_focus_points(sorted_points, clip.duration)
            
            # Calculate motion paths and velocities
            motion_path = self._calculate_motion_path(sorted_points, clip.duration)
            
            def make_dynamic_transform(get_frame, t):
                """Create optimized crop transformation with motion smoothing"""
                frame = get_frame(t)
                h, w = frame.shape[:2]
                
                # Get current position and zoom from motion path
                target_x, target_y, zoom_factor = motion_path(t)
                
                # Calculate crop dimensions
                crop_w = int(w / zoom_factor)
                crop_h = int(h / zoom_factor)
                
                # Convert normalized coordinates to pixel coordinates
                focus_x = int(target_x * w)
                focus_y = int(target_y * h)
                
                # Calculate crop window with edge padding
                crop_x1, crop_y1, crop_x2, crop_y2 = self._calculate_crop_window(
                    w, h, focus_x, focus_y, crop_w, crop_h
                )
                
                # Apply crop with fallback to full frame if invalid
                try:
                    cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if cropped.size > 0:
                        return cv2.resize(cropped, (w, h))
                except Exception as crop_error:
                    self.logger.debug(f"Crop error at t={t}: {crop_error}")
                
                return frame
            
            # Apply the transformation with optimized rendering
            transformed_clip = MoviePyCompat.apply_effect(clip, make_dynamic_transform)
            
            # Add motion blur for smoother transitions
            if len(sorted_points) > 3:
                transformed_clip = self._apply_motion_blur(transformed_clip, motion_path)
            
            self.logger.info(f"Applied enhanced dynamic pan/zoom with {len(sorted_points)} focus points")
            return transformed_clip
            
        except Exception as e:
            self.logger.warning(f"Error applying dynamic pan/zoom: {e}")
            return self.apply_subtle_zoom(clip)  # Fallback to subtle zoom
    
    def _validate_focus_points(self, focus_points: List[Dict[str, Any]], duration: float):
        """Validate focus points for consistency and timing"""
        if not focus_points:
            raise ValueError("No focus points provided")
            
        # Check timing sequence
        prev_time = 0
        for point in focus_points:
            point_time = point.get('timestamp_seconds', 0)
            if point_time < prev_time:
                raise ValueError(f"Focus points not in chronological order at {point_time}")
            prev_time = point_time
            
        # Check coverage
        first_time = focus_points[0].get('timestamp_seconds', 0)
        last_time = focus_points[-1].get('timestamp_seconds', duration)
        if first_time > 0 or last_time < duration:
            self.logger.warning("Focus points don't cover full video duration")

    def _calculate_motion_path(self, focus_points: List[Dict[str, Any]], duration: float):
        """Calculate smooth motion path with velocity-based interpolation"""
        # Pre-calculate motion segments between points
        segments = []
        for i in range(len(focus_points) - 1):
            start = focus_points[i]
            end = focus_points[i + 1]
            
            start_time = start.get('timestamp_seconds', 0)
            end_time = end.get('timestamp_seconds', duration)
            
            # Calculate segment duration and velocity
            duration_seg = end_time - start_time
            if duration_seg <= 0:
                continue
                
            x_vel = (end.get('x', 0.5) - start.get('x', 0.5)) / duration_seg
            y_vel = (end.get('y', 0.5) - start.get('y', 0.5)) / duration_seg
            
            # Base zoom with slight variation during movement
            base_zoom = 1.1
            zoom_variation = 0.05
            
            segments.append({
                'start_time': start_time,
                'end_time': end_time,
                'start_x': start.get('x', 0.5),
                'start_y': start.get('y', 0.5),
                'x_vel': x_vel,
                'y_vel': y_vel,
                'base_zoom': base_zoom,
                'zoom_variation': zoom_variation
            })
        
        def motion_path_func(t):
            """Calculate position and zoom at time t"""
            # Find current segment
            current_seg = None
            for seg in segments:
                if seg['start_time'] <= t <= seg['end_time']:
                    current_seg = seg
                    break
            
            if current_seg:
                # Linear interpolation within segment
                seg_progress = (t - current_seg['start_time']) / (
                    current_seg['end_time'] - current_seg['start_time']
                )
                x = current_seg['start_x'] + current_seg['x_vel'] * (t - current_seg['start_time'])
                y = current_seg['start_y'] + current_seg['y_vel'] * (t - current_seg['start_time'])
                
                # Apply easing to progress
                eased_progress = self._ease_in_out_quint(seg_progress)
                
                # Dynamic zoom based on motion
                zoom_factor = current_seg['base_zoom'] + (
                    current_seg['zoom_variation'] * math.sin(eased_progress * math.pi)
                )
                
                return x, y, zoom_factor
            else:
                # Default to last point if beyond segments
                last_point = focus_points[-1]
                return last_point.get('x', 0.5), last_point.get('y', 0.5), 1.1
        
        return motion_path_func

    def _calculate_crop_window(self, w: int, h: int, focus_x: int, focus_y: int, crop_w: int, crop_h: int):
        """Calculate crop window with edge handling and padding"""
        # Calculate initial crop coordinates
        crop_x1 = max(0, focus_x - crop_w // 2)
        crop_y1 = max(0, focus_y - crop_h // 2)
        crop_x2 = min(w, crop_x1 + crop_w)
        crop_y2 = min(h, crop_y1 + crop_h)
        
        # Adjust if crop extends beyond bounds
        if crop_x2 == w:
            crop_x1 = w - crop_w
        if crop_y2 == h:
            crop_y1 = h - crop_h
            
        # Ensure minimum dimensions
        crop_w = max(100, crop_w)
        crop_h = max(100, crop_h)
        
        # Final validation
        crop_x1 = max(0, min(w - crop_w, crop_x1))
        crop_y1 = max(0, min(h - crop_h, crop_y1))
        crop_x2 = crop_x1 + crop_w
        crop_y2 = crop_y1 + crop_h
        
        return crop_x1, crop_y1, crop_x2, crop_y2

    def _apply_motion_blur(self, clip: VideoFileClip, motion_path):
        """Apply subtle motion blur based on movement velocity"""
        # Calculate average velocity
        sample_times = np.linspace(0, clip.duration, 10)
        velocities = []
        
        for t in sample_times:
            x1, y1, _ = motion_path(t)
            x2, y2, _ = motion_path(t + 0.1)
            vel = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            velocities.append(vel)
            
        avg_vel = sum(velocities) / len(velocities)
        
        # Only apply blur if significant movement
        if avg_vel > 0.01:
            blur_amount = min(1.5, avg_vel * 50)
            return MoviePyCompat.apply_fx_effect(clip, vfx.motion_blur, blur_amount, 0.5)
        return clip

    def _ease_in_out_quint(self, t: float) -> float:
        """Quintic easing function for smoother motion"""
        if t < 0.5:
            return 16 * t * t * t * t * t
        else:
            t = 2 * t - 2
            return 0.5 * (t * t * t * t * t + 2)

    def apply_intelligent_focus(self, clip: VideoFileClip, focus_points: List[Dict[str, Any]]) -> VideoFileClip:
        """
        Apply intelligent panning and zooming based on AI-identified focus points.
        This method now uses the new dynamic pan/zoom functionality.
        
        Args:
            clip: Input video clip
            focus_points: List of focus points with coordinates and timestamps
            
        Returns:
            Video clip with intelligent focus applied
        """
        if not focus_points:
            return clip
        
        # Use the new dynamic pan/zoom method for better results
        return self.apply_dynamic_pan_zoom(clip, focus_points)
    
    def _create_focus_segment(self, clip: VideoFileClip, timestamp: float, duration: float,
                             x: float, y: float, description: str) -> Optional[VideoFileClip]:
        """Create a focused segment with intelligent cropping and zooming"""
        try:
            # Extract the segment
            segment = MoviePyCompat.subclip(clip, timestamp, timestamp + duration)
            
            # Calculate zoom and pan parameters
            zoom_factor = 1.2  # Moderate zoom
            w, h = segment.size
            
            # Convert normalized coordinates to pixel coordinates
            focus_x = int(x * w)
            focus_y = int(y * h)
            
            # Calculate crop area centered on focus point
            crop_w = int(w / zoom_factor)
            crop_h = int(h / zoom_factor)
            
            # Ensure crop area is within bounds
            crop_x1 = max(0, focus_x - crop_w // 2)
            crop_y1 = max(0, focus_y - crop_h // 2)
            crop_x2 = min(w, crop_x1 + crop_w)
            crop_y2 = min(h, crop_y1 + crop_h)
            
            # Adjust if crop extends beyond bounds
            if crop_x2 == w:
                crop_x1 = w - crop_w
            if crop_y2 == h:
                crop_y1 = h - crop_h
            
            # Apply crop and resize back to original size
            focused_segment = MoviePyCompat.crop(segment, x1=crop_x1, y1=crop_y1, x2=crop_x2, y2=crop_y2)
            focused_segment = MoviePyCompat.resize(focused_segment, (w, h))
            
            # Add smooth transition at the beginning and end
            if duration > 1.0:
                focused_segment = MoviePyCompat.crossfadein(focused_segment, 0.3)
            focused_segment = MoviePyCompat.crossfadeout(focused_segment, 0.3)
            
            self.logger.debug(f"Created focus segment at ({x:.2f}, {y:.2f}): {description}")
            return focused_segment
            
        except Exception as e:
            self.logger.warning(f"Error creating focus segment: {e}")
            return None
    
    def _create_highlight_effect(self, clip: VideoFileClip, timestamp: float, duration: float) -> Optional[VideoFileClip]:
        """Create a highlight effect"""
        try:
            w, h = clip.size
            
            # Create border highlight
            border_thickness = 5
            border = ColorClip(size=(w, h), color=(255, 255, 0))
            
            # Create mask for border effect
            inner = ColorClip(size=(w - 2*border_thickness, h - 2*border_thickness), color=(0, 0, 0))
            inner = inner.set_position(('center', 'center'))
            
            border = CompositeVideoClip([border, inner])
            border = border.set_opacity(0.8)
            border = MoviePyCompat.with_start(border, timestamp)
            border = MoviePyCompat.with_duration(border, duration)
            
            return border
            
        except Exception as e:
            self.logger.warning(f"Error creating highlight effect: {e}")
            return None


class AdvancedVideoEnhancer:
    """Advanced video enhancement including denoising and sharpening"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def apply_enhancement_filters(self, clip: VideoFileClip, enable_denoising: bool = False, enable_sharpening: bool = False) -> VideoFileClip:
        """Apply advanced enhancement filters via FFmpeg parameters"""
        if not (enable_denoising or enable_sharpening):
            return clip
        
        try:
            # Build FFmpeg filter chain
            filters = []
            
            if enable_denoising:
                # High Quality Denoise 3D filter - use sparingly as it's CPU intensive
                filters.append('hqdn3d=3:3:6:6')
            
            if enable_sharpening:
                # Unsharp mask for subtle sharpening - be careful not to over-sharpen
                filters.append('unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=1.5')
            
            if filters:
                filter_string = ','.join(filters)
                self.logger.info(f"Applying enhancement filters: {filter_string}")
                
                # Apply filters using FFmpeg subprocess for professional quality
                enhanced_clip = self._apply_ffmpeg_filters(clip, filter_string)
                return enhanced_clip if enhanced_clip else clip
            
            return clip
            
        except Exception as e:
            self.logger.warning(f"Error applying enhancement filters: {e}")
            return clip
    
    def _apply_ffmpeg_filters(self, clip: VideoFileClip, filter_string: str) -> Optional[VideoFileClip]:
        """Apply FFmpeg filters using subprocess for professional quality"""
        try:
            import tempfile
            import subprocess
            
            # Create temporary files
            input_path = tempfile.mktemp(suffix='.mp4')
            output_path = tempfile.mktemp(suffix='.mp4')
            
            # Write input clip to temporary file
            clip.write_videofile(input_path, logger=None)
            
            # Build FFmpeg command
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vf', filter_string,
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-y', output_path
            ]
            
            # Run FFmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Load enhanced clip
                enhanced_clip = VideoFileClip(output_path)
                
                # Clean up temporary input file
                Path(input_path).unlink(missing_ok=True)
                
                # Note: output_path will be cleaned up by caller or temp file manager
                self.logger.info("Successfully applied FFmpeg enhancement filters")
                return enhanced_clip
            else:
                self.logger.warning(f"FFmpeg filter failed: {result.stderr}")
                # Clean up temporary files
                Path(input_path).unlink(missing_ok=True)
                Path(output_path).unlink(missing_ok=True)
                return None
                
        except FileNotFoundError:
            self.logger.warning("FFmpeg not found - install FFmpeg for advanced filters")
            return None
        except Exception as e:
            self.logger.warning(f"Error applying FFmpeg filters: {e}")
            return None
    
    def apply_adaptive_color_grading(self, clip: VideoFileClip) -> VideoFileClip:
        """Apply enhanced adaptive color grading to make colors pop and increase contrast"""
        try:
            # Sample more frames for better analysis
            sample_times = [clip.duration * i / 6 for i in range(1, 6)]  # 5 samples across the video
            brightness_samples = []
            contrast_samples = []
            saturation_samples = []
            
            for t in sample_times:
                if t < clip.duration:
                    frame = clip.get_frame(t)
                    
                    # Calculate brightness (luminance)
                    brightness = np.mean(frame)
                    brightness_samples.append(brightness)
                    
                    # Calculate contrast (standard deviation of luminance)
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    contrast = np.std(gray)
                    contrast_samples.append(contrast)
                    
                    # Calculate saturation
                    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    saturation = np.mean(hsv[:, :, 1])
                    saturation_samples.append(saturation)
            
            if brightness_samples:
                avg_brightness = np.mean(brightness_samples) / 255.0
                avg_contrast = np.mean(contrast_samples) / 127.5  # Normalize to 0-2 range
                avg_saturation = np.mean(saturation_samples) / 255.0
                
                self.logger.info(f"Video analysis - Brightness: {avg_brightness:.2f}, Contrast: {avg_contrast:.2f}, Saturation: {avg_saturation:.2f}")
                
                # Enhanced adaptive adjustments for viral video aesthetics
                if avg_brightness < 0.3:  # Dark video - make it pop
                    # Aggressive brightness and contrast boost for dark videos
                    # Apply color and contrast adjustments with fallback
                    if vfx and hasattr(vfx, 'colorx'):
                        clip = MoviePyCompat.apply_fx_effect(clip, vfx.colorx, factor=1.3)
                    if vfx and hasattr(vfx, 'lum_contrast'):
                        clip = MoviePyCompat.apply_fx_effect(clip, vfx.lum_contrast, lum=0.15, contrast=0.4)
                    self.logger.info("Applied dark video enhancement with aggressive pop")
                    
                elif avg_brightness > 0.75:  # Very bright video
                    # Reduce overexposure while maintaining pop
                    # Apply color and contrast adjustments with fallback
                    if vfx and hasattr(vfx, 'colorx'):
                        clip = MoviePyCompat.apply_fx_effect(clip, vfx.colorx, factor=0.85)
                    if vfx and hasattr(vfx, 'lum_contrast'):
                        clip = MoviePyCompat.apply_fx_effect(clip, vfx.lum_contrast, lum=-0.1, contrast=0.35)
                    self.logger.info("Applied bright video enhancement with contrast boost")
                    
                else:  # Normal brightness - standard pop enhancement
                    # Make colors more vibrant and punchy
                    # Apply color and contrast adjustments with fallback
                    if vfx and hasattr(vfx, 'colorx'):
                        clip = MoviePyCompat.apply_fx_effect(clip, vfx.colorx, factor=1.15)
                    if vfx and hasattr(vfx, 'lum_contrast'):
                        clip = MoviePyCompat.apply_fx_effect(clip, vfx.lum_contrast, lum=0.05, contrast=0.3)
                    self.logger.info("Applied standard pop enhancement")
                
                # Additional saturation boost for viral appeal
                if avg_saturation < 0.4:  # Low saturation - boost colors significantly
                    def saturation_boost(get_frame, t):
                        frame = get_frame(t)
                        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
                        hsv[:, :, 1] *= 1.4  # 40% saturation boost
                        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                    
                    clip = MoviePyCompat.apply_effect(clip, saturation_boost)
                    self.logger.info("Applied saturation boost for color pop")
                    
                elif avg_saturation < 0.6:  # Moderate saturation - mild boost
                    def mild_saturation_boost(get_frame, t):
                        frame = get_frame(t)
                        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
                        hsv[:, :, 1] *= 1.2  # 20% saturation boost
                        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                    
                    clip = MoviePyCompat.apply_effect(clip, mild_saturation_boost)
                    self.logger.info("Applied mild saturation boost")
                
                # Shadow/highlight enhancement for better dynamic range
                if avg_contrast < 0.8:  # Low contrast - enhance shadows and highlights
                    def shadow_highlight_enhancement(get_frame, t):
                        frame = get_frame(t)
                        # Convert to float for processing
                        frame_float = frame.astype(np.float32) / 255.0
                        
                        # Enhance shadows (brighten dark areas)
                        shadows = np.where(frame_float < 0.3, frame_float * 1.3, frame_float)
                        
                        # Enhance highlights (slightly darken very bright areas)
                        enhanced = np.where(shadows > 0.85, shadows * 0.95, shadows)
                        
                        return (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
                    
                    clip = MoviePyCompat.apply_effect(clip, shadow_highlight_enhancement)
                    self.logger.info("Applied shadow/highlight enhancement")
            
            return clip
            
        except Exception as e:
            self.logger.warning(f"Error applying enhanced color grading: {e}")
            return clip


class TextOverlayProcessor:
    """Handles text overlays and subtitles"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def add_text_overlays(self, clip: VideoFileClip, overlays: List[TextOverlay]) -> VideoFileClip:
        """Add text overlays to video with enhanced hook support"""
        if not overlays:
            return clip
        
        try:
            text_clips = []
            
            for overlay in overlays:
                text_clip = self._create_text_clip(clip, overlay)
                if text_clip:
                    text_clips.append(text_clip)
            
            if text_clips:
                all_clips = [clip] + text_clips
                return CompositeVideoClip(all_clips)
            else:
                return clip
                
        except Exception as e:
            self.logger.warning(f"Error adding text overlays: {e}")
            return clip
    
    def add_hook_text(self, clip: VideoFileClip, analysis: VideoAnalysis) -> VideoFileClip:
        """Add powerful hook text immediately at the start of the video"""
        try:
            if not analysis.hook_text:
                self.logger.info("No hook text provided, skipping hook overlay")
                return clip
            
            # Create hook text overlay that appears immediately (0-3 seconds)
            hook_overlay = TextOverlay(
                text=analysis.hook_text,
                timestamp_seconds=0.0,
                duration=min(3.0, clip.duration * 0.15),  # Show for 3 seconds or 15% of video
                position="center",
                style="dramatic"
            )
            
            hook_clip = self._create_hook_text_clip(clip, hook_overlay)
            if hook_clip:
                self.logger.info(f"Added hook text: '{analysis.hook_text}'")
                return CompositeVideoClip([clip, hook_clip])
            else:
                return clip
                
        except Exception as e:
            self.logger.warning(f"Error adding hook text: {e}")
            return clip
    
    def _create_hook_text_clip(self, video_clip: VideoFileClip, overlay: TextOverlay) -> Optional[TextClip]:
        """Create special hook text clip with maximum impact styling"""
        try:
            # Hook text gets special treatment - larger, more dramatic
            font_path = self.config.get_font_path('BebasNeue-Regular.ttf')  # Use dramatic font
            video_height = video_clip.h
            
            # Larger font size for hooks
            font_size = int(video_height * 0.08)  # 8% of video height
            
            # Create dramatic hook text with proper error handling
            text_clip = MoviePyCompat.create_text_clip(
                overlay.text.upper(),  # Convert to uppercase for impact
                fontsize=font_size,
                font=font_path,
                color='#FFFF00',  # Bright yellow for attention
                stroke_color='#000000',
                stroke_width=4,  # Thicker stroke for readability
                method='label',
                align='center'
            )
            
            # Position in center for maximum impact
            text_clip = MoviePyCompat.with_position(text_clip, 'center')
            
            # Set timing
            text_clip = MoviePyCompat.with_start(text_clip, overlay.timestamp_seconds)
            text_clip = MoviePyCompat.with_duration(text_clip, overlay.duration)
            
            # Apply dramatic animation - zoom in + pulsing
            def hook_animation(t):
                # Zoom in effect from 0.8 to 1.2 over the duration
                zoom_progress = min(1.0, t / (overlay.duration * 0.3))  # Zoom in first 30%
                zoom_factor = 0.8 + (0.4 * zoom_progress)  # 0.8 to 1.2
                
                # Add pulsing effect
                pulse = 1 + 0.1 * np.sin(t * 6)  # Subtle pulse
                
                return zoom_factor * pulse
            
            # Apply resize animation using compatibility layer
            try:
                text_clip = MoviePyCompat.resize(text_clip, hook_animation)
            except Exception as e:
                self.logger.warning(f"Skipping hook text animation due to resize issue: {e}")

            # Add fade in/out for smooth transitions
            fade_duration = min(0.5, overlay.duration * 0.2)
            text_clip = MoviePyCompat.crossfadein(text_clip, fade_duration)
            text_clip = MoviePyCompat.crossfadeout(text_clip, fade_duration)
            
            return text_clip
            
        except Exception as e:
            self.logger.warning(f"Error creating hook text clip: {e}")
            return None
    
    def _create_text_clip(self, video_clip: VideoFileClip, overlay: TextOverlay) -> Optional[TextClip]:
        """Create a single text clip with dynamic animations and adaptive sizing"""
        try:
            # Get font and size based on style and text length
            font_path = self.config.get_font_path(self.config.text_overlay.graphical_font)
            
            # Calculate adaptive font size based on text length and video resolution
            video_height = video_clip.h
            text_length = len(overlay.text)
            
            # Use font size profiles for better readability
            if text_length <= 10:
                size_ratio = self.config.text_overlay.font_size_ratio_profiles.get('short', 0.06)
            elif text_length <= 30:
                size_ratio = self.config.text_overlay.font_size_ratio_profiles.get('medium', 0.045)
            else:
                size_ratio = self.config.text_overlay.font_size_ratio_profiles.get('long', 0.035)
            
            font_size = int(video_height * size_ratio)
            
            # Adjust font size based on overlay style
            style_multipliers = {
                "dramatic": 1.3,
                "highlight": 1.1,
                "bold": 1.2,
                "default": 1.0
            }
            font_size = int(font_size * style_multipliers.get(overlay.style, 1.0))
            
            # Create text clip with enhanced styling using compatibility layer
            text_clip = MoviePyCompat.create_text_clip(
                overlay.text,
                fontsize=font_size,
                font=font_path,
                color=self.config.text_overlay.graphical_text_color,
                stroke_color=self.config.text_overlay.graphical_stroke_color,
                stroke_width=self.config.text_overlay.graphical_stroke_width,
                method='caption' if len(overlay.text) > 20 else 'label',
                align='center'
            )
            
            # Enhanced positioning with better placement
            position = self._calculate_text_position(overlay.position, video_clip.size, text_clip.size)
            text_clip = MoviePyCompat.with_position(text_clip, position)
            
            # Set timing
            text_clip = MoviePyCompat.with_start(text_clip, overlay.timestamp_seconds)
            text_clip = MoviePyCompat.with_duration(text_clip, overlay.duration)
            
            # Apply dynamic animations based on style
            text_clip = self._apply_text_animation(text_clip, overlay.style, overlay.duration)
            
            return text_clip
            
        except Exception as e:
            self.logger.warning(f"Error creating text clip: {e}")
            return None
    
    def _calculate_text_position(self, position_type: str, video_size: tuple, text_size: tuple) -> tuple:
        """Calculate optimal text position to avoid overlap and ensure readability"""
        video_width, video_height = video_size
        text_width, text_height = text_size
        
        # Safe margins (percentage of video dimensions)
        margin_x = video_width * 0.05
        margin_y = video_height * 0.05
        
        position_map = {
            "center": ('center', 'center'),
            "top": (
                (video_width - text_width) // 2,
                margin_y
            ),
            "bottom": (
                (video_width - text_width) // 2,
                video_height - text_height - margin_y
            ),
            "left": (
                margin_x,
                (video_height - text_height) // 2
            ),
            "right": (
                video_width - text_width - margin_x,
                (video_height - text_height) // 2
            )
        }
        
        return position_map.get(position_type, ('center', 'center'))
    
    def _apply_text_animation(self, text_clip: TextClip, style: str, duration: float) -> TextClip:
        """Apply dynamic animations based on text style"""
        try:
            fade_duration = min(0.3, duration * 0.2)  # 20% of duration, max 0.3s
            
            if style == "dramatic":
                # Dramatic entrance with scale and fade
                text_clip = MoviePyCompat.crossfadein(text_clip, fade_duration)
                text_clip = MoviePyCompat.crossfadeout(text_clip, fade_duration)
                # Add subtle zoom effect
                text_clip = MoviePyCompat.resize(text_clip, lambda t: 1 + 0.1 * (1 - abs(2*t/duration - 1)))
                
            elif style == "highlight":
                # Pop-in effect with bounce
                def bounce_scale(t):
                    if t < fade_duration:
                        # Bounce in
                        progress = t / fade_duration
                        return 0.8 + 0.4 * progress - 0.2 * (progress ** 2)
                    else:
                        return 1.0
                
                text_clip = MoviePyCompat.resize(text_clip, bounce_scale)
                text_clip = MoviePyCompat.crossfadeout(text_clip, fade_duration)
                
            elif style == "bold":
                # Quick fade with slight pulse
                text_clip = MoviePyCompat.crossfadein(text_clip, fade_duration * 0.5)
                text_clip = MoviePyCompat.crossfadeout(text_clip, fade_duration)
                # Subtle pulse effect
                text_clip = MoviePyCompat.resize(text_clip, lambda t: 1 + 0.05 * np.sin(t * 4))
                
            else:  # default
                # Standard fade in/out
                text_clip = MoviePyCompat.crossfadein(text_clip, fade_duration)
                text_clip = MoviePyCompat.crossfadeout(text_clip, fade_duration)
            
            return text_clip
            
        except Exception as e:
            self.logger.warning(f"Error applying text animation: {e}")
            text_clip = MoviePyCompat.crossfadein(text_clip, 0.2)
            return MoviePyCompat.crossfadeout(text_clip, 0.2)  # Fallback animation


class AudioProcessor:
    """Handles audio processing and mixing"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.sound_effects_manager = SoundEffectsManager()
        self.tts_service = TTSService()
    
    def process_audio(self,
                      video_clip: VideoFileClip,
                      narrative_segments: List[NarrativeSegment],
                      background_music_path: Optional[Path] = None,
                      sound_effects: Optional[List[Dict[str, Any]]] = None) -> Optional[CompositeAudioClip]:
        """Process and mix all audio components with enhanced error handling"""
        audio_tracks = []
        
        try:
            # Original video audio (if keeping it)
            original_audio = self._process_original_audio(video_clip)
            if original_audio:
                audio_tracks.append(original_audio)
            
            # Background music with fallback handling
            music_clip = self._add_background_music_safe(
                background_music_path,
                video_clip.duration,
                narrative_segments
            )
            if music_clip:
                audio_tracks.append(music_clip)
            
            # TTS narrative segments with graceful degradation
            tts_audio = self._process_narrative_segments_safe(narrative_segments)
            if tts_audio:
                audio_tracks.extend(tts_audio)
            
            # Combine base audio tracks
            base_composite = None
            if audio_tracks:
                try:
                    base_composite = CompositeAudioClip(audio_tracks)
                except Exception as e:
                    self.logger.error(f"Failed to create base composite audio: {e}")
                    # Fallback to first available track
                    base_composite = audio_tracks[0] if audio_tracks else None
            
            # Add sound effects to the composite
            if sound_effects and base_composite:
                final_audio = self.add_sound_effects(base_composite, sound_effects, video_clip.duration)
                return final_audio
            elif base_composite:
                return base_composite
            else:
                self.logger.warning("No audio tracks available, returning original audio")
                return video_clip.audio
                
        except Exception as e:
            self.logger.error(f"Critical error in audio processing: {e}")
            return video_clip.audio
    
    def _process_original_audio(self, video_clip: VideoFileClip) -> Optional[AudioFileClip]:
        """Process original video audio with error handling"""
        try:
            if video_clip.audio and self.config.audio.original_audio_mix_volume > 0:
                try:
                    original_audio = video_clip.audio.with_volume_scaled(self.config.audio.original_audio_mix_volume)
                except Exception as e:
                    self.logger.warning(f"Error applying volume to original audio: {e}")
                    original_audio = video_clip.audio
                self.logger.debug("Successfully processed original audio")
                return original_audio
            else:
                self.logger.debug("No original audio to process")
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to process original audio: {e}")
            return None
    
    def _add_background_music_safe(self,
                                   music_path: Optional[Path],
                                   video_duration: float,
                                   narrative_segments: List[NarrativeSegment]) -> Optional[AudioFileClip]:
        """Add background music with comprehensive error handling"""
        if not (music_path and music_path.exists() and self.config.audio.background_music_enabled):
            self.logger.debug("Background music not available or disabled")
            return None
        
        try:
            music_clip = self._add_background_music(music_path, video_duration, narrative_segments)
            if music_clip:
                self.logger.info("Successfully added background music")
            return music_clip
            
        except Exception as e:
            self.logger.warning(f"Failed to add background music from {music_path}: {e}")
            return None
    
    def _process_narrative_segments_safe(self, segments: List[NarrativeSegment]) -> List[AudioFileClip]:
        """Process TTS narrative segments with graceful failure handling"""
        if not segments:
            return []
        
        if not self.tts_service.is_available():
            self.logger.warning("No TTS service available, skipping narrative segments")
            return []
        
        try:
            self.logger.info(f"Processing {len(segments)} narrative segments with TTS")
            
            # Generate all TTS audio files
            tts_results = self.tts_service.generate_multiple_segments(segments)
            
            successful_audio_clips = []
            for result in tts_results:
                if not result['success'] or not result['audio_path']:
                    continue
                
                try:
                    segment = result['segment']
                    audio_path = result['audio_path']
                    
                    # Load audio clip
                    audio_clip = AudioFileClip(str(audio_path))
                    
                    # Adjust duration if needed
                    target_duration = segment.intended_duration_seconds
                    if abs(audio_clip.duration - target_duration) > 0.5:
                        # Significant difference, adjust speed
                        adjusted_path = self.tts_service.adjust_audio_speed(
                            audio_path, target_duration
                        )
                        if adjusted_path and adjusted_path != audio_path:
                            audio_clip.close()
                            audio_clip = AudioFileClip(str(adjusted_path))
                    
                    # Set timing and apply effects
                    audio_clip = MoviePyCompat.with_start(audio_clip, segment.time_seconds)
                    
                    # Apply emotional volume adjustments
                    volume_factor = self._get_emotion_volume_factor(segment.emotion)
                    if volume_factor != 1.0:
                        try:
                            audio_clip = audio_clip.with_volume_scaled(volume_factor)
                        except Exception as e:
                            self.logger.warning(f"Error applying volume factor: {e}")
                    
                    successful_audio_clips.append(audio_clip)
                    
                    self.logger.debug(f"Successfully processed TTS segment: '{segment.text[:50]}...'")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process TTS audio for segment: {e}")
                    continue
            
            self.logger.info(f"Successfully processed {len(successful_audio_clips)}/{len(segments)} TTS segments")
            return successful_audio_clips
            
        except Exception as e:
            self.logger.error(f"Critical error in TTS processing: {e}")
            return []
    
    def _add_background_music(self, 
                              music_path: Path, 
                              video_duration: float,
                              narrative_segments: List[NarrativeSegment]) -> Optional[AudioFileClip]:
        """Add background music with ducking during narration"""
        try:
            music_clip = AudioFileClip(str(music_path))
            
            # Loop music if needed
            if music_clip.duration < video_duration:
                loops_needed = math.ceil(video_duration / music_clip.duration)
                music_clips = [music_clip] * loops_needed
                music_clip = concatenate_audioclips(music_clips)
            
            # Trim to video duration
            music_clip = MoviePyCompat.subclip(music_clip, 0, video_duration)
            
            # Apply base volume
            try:
                music_clip = music_clip.with_volume_scaled(self.config.audio.background_music_volume)
            except Exception as e:
                self.logger.warning(f"Error applying base volume to music: {e}")
            
            # Duck music during narration
            if narrative_segments:
                music_clip = self._apply_audio_ducking(music_clip, narrative_segments)
            
            return music_clip
            
        except Exception as e:
            self.logger.warning(f"Error adding background music: {e}")
            return None
    
    def _apply_audio_ducking(self, 
                             music_clip: AudioFileClip, 
                             narrative_segments: List[NarrativeSegment]) -> AudioFileClip:
        """Reduce music volume during narration"""
        try:
            duck_factor = self.config.audio.background_music_narrative_volume_factor
            
            # Create ducking segments
            segments = []
            current_time = 0
            
            for segment in narrative_segments:
                start_time = segment.time_seconds
                end_time = start_time + segment.intended_duration_seconds
                
                # Add normal volume segment before narration
                if current_time < start_time:
                    segments.append(MoviePyCompat.subclip(music_clip, current_time, start_time))
                
                # Add ducked segment during narration
                if start_time < music_clip.duration:
                    duck_end = min(end_time, music_clip.duration)
                    ducked_segment = MoviePyCompat.subclip(music_clip, start_time, duck_end)
                    try:
                        ducked_segment = ducked_segment.with_volume_scaled(duck_factor)
                    except Exception as e:
                        self.logger.warning(f"Error applying ducking volume: {e}")
                    segments.append(ducked_segment)
                
                current_time = end_time
            
            # Add remaining segment
            if current_time < music_clip.duration:
                segments.append(MoviePyCompat.subclip(music_clip, current_time, music_clip.duration))
            
            if segments:
                return concatenate_audioclips(segments)
            else:
                return music_clip
                
        except Exception as e:
            self.logger.warning(f"Error applying audio ducking: {e}")
            return music_clip
    
    def _get_emotion_volume_factor(self, emotion: str) -> float:
        """Get volume adjustment factor based on emotion"""
        emotion_volumes = {
            'excited': 1.1,
            'dramatic': 1.2,
            'calm': 0.9,
            'neutral': 1.0
        }
        return emotion_volumes.get(emotion.lower(), 1.0)
    
    def add_sound_effects(self,
                         base_audio: CompositeAudioClip,
                         sound_effects: List[Dict[str, Any]],
                         video_duration: float) -> CompositeAudioClip:
        """Add sound effects to the composite audio using enhanced SoundEffectsManager"""
        if not sound_effects:
            self.logger.debug("No sound effects to add")
            return base_audio
        
        try:
            audio_tracks = [base_audio] if base_audio else []
            effects_added = 0
            effects_failed = 0
            
            for effect in sound_effects:
                try:
                    effect_name = effect.get('effect_name', '').lower()
                    timestamp = max(0, float(effect.get('timestamp_seconds', 0)))
                    volume = max(0.1, min(1.0, float(effect.get('volume', 0.7))))
                    preferred_category = effect.get('category')  # Optional category hint
                    
                    # Skip if timestamp is beyond video duration
                    if timestamp >= video_duration:
                        self.logger.debug(f"Skipping sound effect '{effect_name}' - timestamp beyond video duration")
                        continue
                    
                    # Use enhanced sound effects manager to find the best match
                    sound_file = self.sound_effects_manager.find_sound_effect(
                        effect_name,
                        preferred_category=preferred_category
                    )
                    
                    if sound_file:
                        # Validate the sound file before using
                        is_valid, validation_msg = self.sound_effects_manager.validate_sound_effect(sound_file)
                        if not is_valid:
                            self.logger.warning(f"Invalid sound effect file '{sound_file}': {validation_msg}")
                            effects_failed += 1
                            continue
                        
                        # Load and process the sound effect
                        sound_clip = AudioFileClip(str(sound_file))
                        
                        # Adjust volume and timing
                        try:
                            sound_clip = sound_clip.with_volume_scaled(volume)
                        except Exception as e:
                            self.logger.warning(f"Error applying volume to sound effect: {e}")
                        sound_clip = MoviePyCompat.with_start(sound_clip, timestamp)
                        
                        # Ensure sound doesn't exceed video duration
                        if timestamp + sound_clip.duration > video_duration:
                            max_duration = video_duration - timestamp
                            if max_duration > 0.1:  # Only add if at least 0.1 seconds remaining
                                sound_clip = MoviePyCompat.subclip(sound_clip, 0, max_duration)
                            else:
                                self.logger.debug(f"Skipping sound effect '{effect_name}' - insufficient time remaining")
                                sound_clip.close()
                                continue
                        
                        # Apply fade in/out for smoother integration (enhanced)
                        fade_duration = min(0.05, sound_clip.duration * 0.1)
                        if sound_clip.duration > fade_duration * 2:
                            try:
                                # Try new MoviePy method names
                                sound_clip = sound_clip.with_fadein(fade_duration).with_fadeout(fade_duration)
                            except AttributeError:
                                try:
                                    # Try older MoviePy method names
                                    sound_clip = sound_clip.audio_fadein(fade_duration).audio_fadeout(fade_duration)
                                except AttributeError:
                                    # Skip fade effects if not available
                                    self.logger.debug("Fade effects not available for sound clips")
                        
                        audio_tracks.append(sound_clip)
                        effects_added += 1
                        self.logger.info(f"Added sound effect '{effect_name}' at {timestamp:.1f}s (volume: {volume:.1f}, file: {sound_file.name})")
                        
                        # Log validation warnings if any
                        if validation_msg:
                            self.logger.info(f"Sound effect note: {validation_msg}")
                    else:
                        self.logger.warning(f"Sound effect file not found for: {effect_name}")
                        effects_failed += 1
                        
                except Exception as e:
                    self.logger.warning(f"Failed to add sound effect '{effect.get('effect_name', 'unknown')}': {e}")
                    effects_failed += 1
                    continue
            
            # Log comprehensive statistics
            total_requested = len(sound_effects)
            success_rate = (effects_added / total_requested * 100) if total_requested > 0 else 0
            self.logger.info(f"Sound effects summary: {effects_added}/{total_requested} added successfully ({success_rate:.1f}% success rate)")
            
            if effects_failed > 0:
                cache_status = self.sound_effects_manager.get_cache_status()
                self.logger.info(f"Available sound effects: {cache_status['total_effects']} across {cache_status['categories']} categories")
            
            if len(audio_tracks) > 1:
                return CompositeAudioClip(audio_tracks)
            elif audio_tracks:
                return audio_tracks[0]
            else:
                return base_audio
                
        except Exception as e:
            self.logger.error(f"Critical error adding sound effects: {e}")
            return base_audio
    
    
    def _process_narrative_segments(self, segments: List[NarrativeSegment]) -> List[AudioFileClip]:
        """Process TTS narrative segments (deprecated - use _process_narrative_segments_safe)"""
        return self._process_narrative_segments_safe(segments)


class TemporaryFileManager:
    """Manages temporary files created during video processing"""
    
    def __init__(self):
        self.temp_files = []
        self.temp_dirs = []
        self.logger = logging.getLogger(__name__)
    
    def register_file(self, file_path: Path) -> Path:
        """Register a temporary file for cleanup"""
        self.temp_files.append(file_path)
        return file_path
    
    def register_dir(self, dir_path: Path) -> Path:
        """Register a temporary directory for cleanup"""
        self.temp_dirs.append(dir_path)
        return dir_path
    
    def create_temp_file(self, suffix: str = "", prefix: str = "video_proc_") -> Path:
        """Create and register a temporary file"""
        temp_file = Path(tempfile.mktemp(suffix=suffix, prefix=prefix))
        return self.register_file(temp_file)
    
    def create_temp_dir(self, prefix: str = "video_proc_") -> Path:
        """Create and register a temporary directory"""
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        return self.register_dir(temp_dir)
    
    def cleanup(self):
        """Clean up all registered temporary files and directories"""
        # Clean up files
        for file_path in self.temp_files:
            try:
                if file_path.exists():
                    file_path.unlink()
                    self.logger.debug(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                self.logger.warning(f"Error cleaning up temp file {file_path}: {e}")
        
        # Clean up directories
        for dir_path in self.temp_dirs:
            try:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    self.logger.debug(f"Cleaned up temp dir: {dir_path}")
            except Exception as e:
                self.logger.warning(f"Error cleaning up temp dir {dir_path}: {e}")
        
        self.temp_files.clear()
        self.temp_dirs.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class VideoProcessor:
    """Main video processing orchestrator with enhanced error handling and modular design"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.downloader = VideoDownloader()
        self.effects = VideoEffects()
        self.text_processor = TextOverlayProcessor()
        self.audio_processor = AudioProcessor()
        self.advanced_enhancer = AdvancedVideoEnhancer()
        self.cta_processor = CTAProcessor()
        self.thumbnail_generator = ThumbnailGenerator()
        self.speed_optimizer = create_speed_optimizer()
    
    def process_video(self,
                      video_path: Path,
                      output_path: Path,
                      analysis: VideoAnalysis,
                      background_music_path: Optional[Path] = None,
                      generate_thumbnail: bool = True) -> Dict[str, Any]:
        """
        Enhanced video processing pipeline with AI-powered narrative, CTAs, and thumbnail generation
        
        Args:
            video_path: Input video path
            output_path: Output video path
            analysis: AI analysis results (validated Pydantic model)
            background_music_path: Optional background music
            generate_thumbnail: Whether to generate thumbnail
        
        Returns:
            Dict with processing results including video_path, thumbnail_path, and success status
        """
        result = {
            'success': False,
            'video_path': None,
            'thumbnail_path': None,
            'processing_stats': {
                'tts_segments': 0,
                'visual_effects': 0,
                'cta_elements': 0,
                'duration': 0.0
            }
        }
        
        with ResourceManager() as resource_manager, TemporaryFileManager() as temp_manager:
            try:
                self.logger.info(f"Starting enhanced video processing: {video_path}")
                
                # Stage 1: Input Validation and Preparation
                video_clip = self._load_source_clip(video_path, output_path, analysis, resource_manager)
                if not video_clip:
                    return result
                
                # Stage 2: Enhanced Audio Synthesis with TTS and Sound Effects
                composite_audio = self._synthesize_audio(video_clip, analysis, background_music_path, resource_manager, temp_manager)
                
                # Stage 3: Advanced Visual Effects and Composition
                enhanced_visuals = self._compose_visuals(video_clip, analysis, resource_manager)
                
                # Stage 4: Add Call-to-Action Elements for Maximum Engagement
                final_visuals = self._add_engagement_elements(enhanced_visuals, analysis, resource_manager)
                
                # Stage 5: Final Audio Enhancement with Auditory CTAs
                if composite_audio:
                    final_audio = self.cta_processor.add_auditory_ctas(
                        composite_audio, analysis, final_visuals.duration
                    )
                    resource_manager.register_clip(final_audio)
                else:
                    final_audio = composite_audio
                
                # Stage 6: Combine and Render Final Video
                final_clip = MoviePyCompat.with_audio(final_visuals, final_audio) if final_audio else final_visuals
                video_success = self._render_video(final_clip, output_path, temp_manager)
                
                if video_success:
                    result['video_path'] = output_path
                    result['processing_stats']['duration'] = final_clip.duration
                    
                    # Stage 7: Generate Enhanced Thumbnail
                    if generate_thumbnail:
                        thumbnail_path = output_path.with_suffix('.jpg')
                        thumbnail_success = self.thumbnail_generator.generate_thumbnail(
                            video_path, analysis, thumbnail_path
                        )
                        if thumbnail_success:
                            result['thumbnail_path'] = thumbnail_path
                    
                    # Update processing stats
                    result['processing_stats'].update({
                        'tts_segments': len(analysis.narrative_script_segments) if analysis.narrative_script_segments else 0,
                        'visual_effects': len(analysis.visual_cues) if analysis.visual_cues else 0,
                        'cta_elements': 1 if analysis.call_to_action else 0
                    })
                    
                    result['success'] = True
                    self.logger.info(f"Enhanced video processing completed: {output_path}")
                else:
                    self.logger.error("Failed to render final video")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Critical error in video processing: {e}", exc_info=True)
                return result
            finally:
                self._cleanup()
    
    def _load_source_clip(self,
                          video_path: Path,
                          output_path: Path,
                          analysis: VideoAnalysis,
                          resource_manager: ResourceManager) -> Optional[VideoFileClip]:
        """
        Stage 1: Validate inputs and load source video clip
        
        Args:
            video_path: Input video path
            output_path: Output video path
            analysis: AI analysis results
            resource_manager: Resource manager for cleanup
        
        Returns:
            Loaded and prepared video clip or None if failed
        """
        try:
            # Validate input parameters
            if not self._validate_inputs(video_path, output_path, analysis):
                return None
            
            # Load and prepare video
            video_clip = self._load_and_prepare_video(video_path, resource_manager)
            if not video_clip:
                self.logger.error("Failed to load source video")
                return None
            
            self.logger.info(f"Successfully loaded source clip: {video_clip.duration:.2f}s")
            return video_clip
            
        except Exception as e:
            self.logger.error(f"Error in source clip loading: {e}")
            return None
    
    def _synthesize_audio(self,
                          source_audio: AudioFileClip,
                          analysis: VideoAnalysis,
                          background_music_path: Optional[Path],
                          resource_manager: ResourceManager,
                          temp_manager: TemporaryFileManager) -> Optional[AudioFileClip]:
        """
        Stage 2: Synthesize complete audio track with TTS, music, and ducking
        
        Args:
            source_audio: Original video audio
            analysis: AI analysis with audio requirements
            background_music_path: Optional background music file
            resource_manager: Resource manager for cleanup
            temp_manager: Temporary file manager
        
        Returns:
            Composite audio clip or None if failed
        """
        try:
            self.logger.info("Starting enhanced audio synthesis with TTS and sound effects")
            
            # Calculate video duration for audio processing
            video_duration = source_audio.duration if source_audio else 60
            
            # Process TTS narration with enhanced integration
            narrative_clips = []
            if analysis.narrative_script_segments:
                try:
                    narrative_clips = self.audio_processor._process_narrative_segments_safe(
                        analysis.narrative_script_segments
                    )
                    self.logger.info(f"Generated {len(narrative_clips)} TTS segments")
                    for clip in narrative_clips:
                        resource_manager.register_clip(clip)
                except Exception as e:
                    self.logger.warning(f"TTS processing failed, continuing without narration: {e}")
            
            # Add background music with intelligent ducking
            music_clip = None
            if background_music_path and background_music_path.exists():
                try:
                    music_clip = self.audio_processor._add_background_music_safe(
                        background_music_path,
                        video_duration,
                        analysis.narrative_script_segments
                    )
                    if music_clip:
                        resource_manager.register_clip(music_clip)
                        self.logger.info("Added background music with intelligent ducking")
                except Exception as e:
                    self.logger.warning(f"Background music processing failed: {e}")
            
            # Composite base audio elements
            audio_elements = []
            
            # Add original audio if it's important
            if analysis.original_audio_is_key and source_audio:
                # Apply volume adjustment for original audio with safe access
                if source_audio:
                    try:
                        # Create a proper mock video clip with audio
                        class MockVideoClip:
                            def __init__(self, audio_clip):
                                self.audio = audio_clip
                        
                        original_audio = self.audio_processor._process_original_audio(
                            MockVideoClip(source_audio)
                        )
                    except Exception as e:
                        self.logger.warning(f"Error processing original audio: {e}")
                        original_audio = source_audio
                else:
                    original_audio = None
                if original_audio:
                    audio_elements.append(original_audio)
                    self.logger.info("Preserved original audio with volume adjustment")
            
            # Add background music
            if music_clip:
                audio_elements.append(music_clip)
            
            # Add TTS narration
            if narrative_clips:
                audio_elements.extend(narrative_clips)
            
            # Create base composite audio
            base_composite = None
            if len(audio_elements) > 1:
                base_composite = CompositeAudioClip(audio_elements)
                resource_manager.register_clip(base_composite)
                self.logger.info(f"Created base composite audio with {len(audio_elements)} elements")
            elif audio_elements:
                base_composite = audio_elements[0]
                self.logger.info("Using single audio element as base")
            
            # Add sound effects to the composite
            if base_composite and analysis.sound_effects:
                try:
                    final_audio = self.audio_processor.add_sound_effects(
                        base_composite,
                        [effect.dict() for effect in analysis.sound_effects],
                        video_duration
                    )
                    resource_manager.register_clip(final_audio)
                    self.logger.info(f"Added {len(analysis.sound_effects)} sound effects")
                    return final_audio
                except Exception as e:
                    self.logger.warning(f"Sound effects processing failed: {e}")
                    return base_composite
            elif base_composite:
                return base_composite
            else:
                self.logger.warning("No audio elements available, creating fallback background music")
                # Try to create fallback background music from available files
                try:
                    video_duration = getattr(source_clip, 'duration', 60.0)
                    fallback_audio = self._create_fallback_audio(video_duration, resource_manager)
                    if fallback_audio:
                        self.logger.info("Created fallback background music")
                        return fallback_audio
                    else:
                        # Last resort: create silent audio
                        self.logger.warning("Creating silent audio as final fallback")
                        from moviepy.audio.AudioClip import AudioClip
                        silent_audio = AudioClip(make_frame=lambda t: [0, 0], duration=video_duration)
                        resource_manager.register_clip(silent_audio)
                        return silent_audio
                except Exception as e:
                    self.logger.error(f"Failed to create fallback audio: {e}")
                    return None
            
        except Exception as e:
            self.logger.error(f"Error in audio synthesis: {e}")
            # Return original audio as fallback
            return source_audio if source_audio else None
    
    def _create_fallback_audio(self,
                               video_duration: float,
                               resource_manager: ResourceManager) -> Optional[AudioFileClip]:
        """
        Create fallback background music when no other audio elements are available
        
        Args:
            video_duration: Duration of the video
            resource_manager: Resource manager for cleanup
            
        Returns:
            AudioFileClip with background music or None if failed
        """
        try:
            # Try to find any available music file
            music_files = []
            
            # Check main music directory
            for music_file in self.config.paths.music_folder.glob("*.mp3"):
                if music_file.exists():
                    music_files.append(music_file)
            
            # Check music subdirectories
            for subdir in ["upbeat", "relaxing", "informative", "funny", "emotional", "suspenseful"]:
                subdir_path = self.config.paths.music_folder / subdir
                if subdir_path.exists():
                    for music_file in subdir_path.glob("*.mp3"):
                        if music_file.exists():
                            music_files.append(music_file)
            
            if not music_files:
                self.logger.warning("No music files found for fallback audio")
                return None
            
            # Select first available music file
            selected_music = music_files[0]
            self.logger.info(f"Using fallback music: {selected_music.name}")
            
            # Create background music with basic processing
            try:
                from moviepy.audio.io.AudioFileClip import AudioFileClip
                from moviepy.editor import concatenate_audioclips
                from src.processing.video_processor_fixes import MoviePyCompat
                import math
                
                music_clip = AudioFileClip(str(selected_music))
                resource_manager.register_clip(music_clip)
                
                # Adjust duration to match video
                if music_clip.duration < video_duration:
                    # Loop the music if it's shorter than video (using correct approach)
                    loops_needed = math.ceil(video_duration / music_clip.duration)
                    music_clips = [music_clip] * loops_needed
                    music_clip = concatenate_audioclips(music_clips)
                    resource_manager.register_clip(music_clip)
                elif music_clip.duration > video_duration:
                    # Trim the music if it's longer than video
                    music_clip = MoviePyCompat.subclip(music_clip, 0, video_duration)
                    resource_manager.register_clip(music_clip)
                
                # Apply volume adjustment (using correct method)
                background_volume = self.config.audio.background_music_volume
                try:
                    music_clip = music_clip.with_volume_scaled(background_volume)
                except Exception as e:
                    self.logger.warning(f"Error applying volume to fallback music: {e}")
                    # Fallback to trying volumex method
                    try:
                        music_clip = music_clip.volumex(background_volume)
                    except Exception as e2:
                        self.logger.warning(f"Error with volumex fallback: {e2}")
                resource_manager.register_clip(music_clip)
                
                self.logger.info(f"Created fallback background music: {video_duration:.2f}s at {background_volume:.2f} volume")
                return music_clip
                
            except Exception as e:
                self.logger.error(f"Failed to process fallback music {selected_music}: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating fallback audio: {e}")
            return None
    
    def _compose_visuals(self,
                        source_clip: VideoFileClip,
                        analysis: VideoAnalysis,
                        resource_manager: ResourceManager) -> VideoFileClip:
        """
        Stage 3: Compose all visual effects, overlays, and enhancements
        
        Args:
            source_clip: Source video clip
            analysis: AI analysis with visual requirements
            resource_manager: Resource manager for cleanup
        
        Returns:
            Final composited video with all visual effects
        """
        try:
            self.logger.info("Starting enhanced visual composition with focus points and effects")
            video_clip = source_clip
            
            # Apply advanced video enhancement filters first
            if self.config.video.video_quality_profile == 'high':
                try:
                    video_clip = self.advanced_enhancer.apply_enhancement_filters(
                        video_clip,
                        enable_denoising=True,
                        enable_sharpening=True
                    )
                    self.logger.info("Applied advanced enhancement filters")
                except Exception as e:
                    self.logger.warning(f"Advanced enhancement failed: {e}")
            
            # CRITICAL: Add hook text IMMEDIATELY as first visual element (3-second rule)
            try:
                video_clip = self.text_processor.add_hook_text(video_clip, analysis)
                self.logger.info("Added hook text for immediate viewer engagement")
            except Exception as e:
                self.logger.warning(f"Hook text processing failed: {e}")
            
            # Apply adaptive color grading for viral aesthetics
            try:
                video_clip = self.advanced_enhancer.apply_adaptive_color_grading(video_clip)
                self.logger.info("Applied enhanced color grading for visual pop")
            except Exception as e:
                self.logger.warning(f"Color grading failed: {e}")
            
            # Apply dynamic pan & zoom to guide viewer's eye
            if analysis.key_focus_points:
                try:
                    focus_data = [point.dict() for point in analysis.key_focus_points]
                    video_clip = self.effects.apply_dynamic_pan_zoom(video_clip, focus_data)
                    self.logger.info(f"Applied dynamic pan/zoom for {len(analysis.key_focus_points)} focus points")
                except Exception as e:
                    self.logger.warning(f"Dynamic pan/zoom failed: {e}")
            else:
                # Fallback: Apply subtle zoom for static videos
                try:
                    video_clip = self.effects.apply_subtle_zoom(video_clip)
                    self.logger.info("Applied subtle zoom as fallback for engagement")
                except Exception as e:
                    self.logger.warning(f"Subtle zoom failed: {e}")
            
            # Apply speed ramping for dramatic effect
            if analysis.speed_effects:
                try:
                    speed_data = [effect.dict() for effect in analysis.speed_effects]
                    video_clip = self.effects.apply_speed_effects(video_clip, speed_data)
                    self.logger.info(f"Applied {len(analysis.speed_effects)} speed effects for drama")
                except Exception as e:
                    self.logger.warning(f"Speed effects failed: {e}")
            
            # Apply visual effects with graceful error handling
            video_clip = self._apply_visual_effects(video_clip, analysis, resource_manager)
            
            # Add enhanced text overlays with dynamic animations
            if analysis.text_overlays:
                try:
                    video_clip = self.text_processor.add_text_overlays(video_clip, analysis.text_overlays)
                    self.logger.info(f"Added {len(analysis.text_overlays)} text overlays with animations")
                except Exception as e:
                    self.logger.warning(f"Text overlay processing failed: {e}")
            
            # Apply visual cues for engagement
            if analysis.visual_cues:
                try:
                    video_clip = self.effects.add_visual_cues(video_clip, analysis.visual_cues)
                    self.logger.info(f"Added {len(analysis.visual_cues)} visual cues")
                except Exception as e:
                    self.logger.warning(f"Visual cues failed: {e}")
            
            # Apply duration constraints
            video_clip = self._apply_duration_constraints(video_clip)
            
            # Register the final clip for cleanup
            resource_manager.register_clip(video_clip)
            
            self.logger.info("Enhanced visual composition completed")
            return video_clip
            
        except Exception as e:
            self.logger.error(f"Error in visual composition: {e}")
            # Return source clip as fallback
            return source_clip
    
    def _add_engagement_elements(self,
                                video_clip: VideoFileClip,
                                analysis: VideoAnalysis,
                                resource_manager: ResourceManager) -> VideoFileClip:
        """
        Stage 4: Add engagement elements including CTAs, hooks, and interactive elements
        
        Args:
            video_clip: Enhanced video with visual effects
            analysis: AI analysis with engagement requirements
            resource_manager: Resource manager for cleanup
        
        Returns:
            Video with all engagement elements added
        """
        try:
            self.logger.info("Adding engagement elements and CTAs")
            enhanced_video = video_clip
            
            # Add visual CTAs (subscribe buttons, like reminders, etc.)
            try:
                enhanced_video = self.cta_processor.add_visual_ctas(enhanced_video, analysis)
                self.logger.info("Added visual call-to-action elements")
            except Exception as e:
                self.logger.warning(f"Visual CTA processing failed: {e}")
            
            # Add engagement hooks throughout the video
            try:
                enhanced_video = self.cta_processor.create_engagement_hooks(enhanced_video, analysis)
                self.logger.info("Added engagement hooks and curiosity gaps")
            except Exception as e:
                self.logger.warning(f"Engagement hooks processing failed: {e}")
            
            # Register the final enhanced video for cleanup
            resource_manager.register_clip(enhanced_video)
            
            self.logger.info("Engagement elements processing completed")
            return enhanced_video
            
        except Exception as e:
            self.logger.error(f"Error adding engagement elements: {e}")
            return video_clip
    
    def _render_video(self,
                      final_clip: VideoFileClip,
                      output_path: Path,
                      temp_manager: TemporaryFileManager) -> bool:
        """
        Stage 4: Render final video to file with retry logic
        
        Args:
            final_clip: Final composited video clip
            output_path: Output file path
            temp_manager: Temporary file manager
        
        Returns:
            True if rendering successful, False otherwise
        """
        try:
            self.logger.info(f"Starting video render to: {output_path}")
            return self._write_video_with_retry(final_clip, output_path, temp_manager)
            
        except Exception as e:
            self.logger.error(f"Error in video rendering: {e}")
            return False
    
    def _process_tts_segments(self, segments: List[NarrativeSegment]) -> List[AudioFileClip]:
        """
        Process TTS narrative segments with enhanced error handling
        
        Args:
            segments: List of narrative segments to process
        
        Returns:
            List of generated audio clips
        """
        return self.audio_processor.process_narrative_segments(segments)
    
    def _cleanup(self):
        """Perform final cleanup operations"""
        try:
            # Force garbage collection
            gc.collect()
            self.logger.debug("Performed final cleanup")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
    
    def _validate_inputs(self, video_path: Path, output_path: Path, analysis: VideoAnalysis) -> bool:
        """Validate input parameters"""
        try:
            if not video_path.exists():
                self.logger.error(f"Input video does not exist: {video_path}")
                return False
            
            if not analysis:
                self.logger.error("Analysis object is None")
                return False
            
            # Validate Pydantic model (this will raise ValidationError if invalid)
            if hasattr(analysis, 'model_validate'):
                analysis.model_validate(analysis.model_dump())
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False
    
    def _load_and_prepare_video(self, video_path: Path, resource_manager: ResourceManager) -> Optional[VideoFileClip]:
        """Load and prepare video with error handling"""
        try:
            self.logger.info(f"Loading video: {video_path}")
            video_clip = VideoFileClip(str(video_path))
            resource_manager.register_clip(video_clip)
            
            # Prepare video for YouTube Shorts processing
            prepared_clip = self._prepare_video_clip(video_clip)
            
            # Ensure YouTube Shorts format (9:16, ≤60s)
            shorts_clip = ensure_shorts_format(prepared_clip, target_duration=60.0)
            if shorts_clip != prepared_clip:
                resource_manager.register_clip(shorts_clip)
                prepared_clip = shorts_clip
            if prepared_clip != video_clip:
                resource_manager.register_clip(prepared_clip)
            
            return prepared_clip
            
        except Exception as e:
            self.logger.error(f"Failed to load video {video_path}: {e}")
            return None
    
    def _apply_visual_effects(self, video_clip: VideoFileClip, analysis: VideoAnalysis, resource_manager: ResourceManager) -> VideoFileClip:
        """Apply visual effects with individual error handling"""
        try:
            # Apply visual cues
            if analysis.visual_cues:
                try:
                    enhanced_clip = self.effects.add_visual_cues(video_clip, analysis.visual_cues)
                    if enhanced_clip != video_clip:
                        resource_manager.register_clip(enhanced_clip)
                        video_clip = enhanced_clip
                except Exception as e:
                    self.logger.warning(f"Failed to apply visual cues: {e}")
            
            # Apply dynamic pan/zoom using focus points if available
            if hasattr(analysis, 'focus_points') and analysis.focus_points:
                try:
                    pan_zoom_clip = self.effects.apply_dynamic_pan_zoom(video_clip, analysis.focus_points)
                    if pan_zoom_clip != video_clip:
                        resource_manager.register_clip(pan_zoom_clip)
                        video_clip = pan_zoom_clip
                except Exception as e:
                    self.logger.warning(f"Failed to apply dynamic pan/zoom: {e}")
            else:
                # Apply subtle zoom as fallback
                try:
                    zoom_clip = self.effects.apply_subtle_zoom(video_clip)
                    if zoom_clip != video_clip:
                        resource_manager.register_clip(zoom_clip)
                        video_clip = zoom_clip
                except Exception as e:
                    self.logger.warning(f"Failed to apply zoom effect: {e}")
            
            # Apply speed effects
            if analysis.speed_effects:
                try:
                    speed_clip = self.effects.apply_speed_effects(video_clip, analysis.speed_effects)
                    if speed_clip != video_clip:
                        resource_manager.register_clip(speed_clip)
                        video_clip = speed_clip
                except Exception as e:
                    self.logger.warning(f"Failed to apply speed effects: {e}")
            
            try:
                # Use adaptive color grading instead of basic color grading for better quality
                graded_clip = self.advanced_enhancer.apply_adaptive_color_grading(video_clip)
                if graded_clip != video_clip:
                    resource_manager.register_clip(graded_clip)
                    video_clip = graded_clip
            except Exception as e:
                self.logger.warning(f"Failed to apply adaptive color grading: {e}")
                # Fallback to basic color grading
                try:
                    graded_clip = self.effects.apply_color_grading(video_clip)
                    if graded_clip != video_clip:
                        resource_manager.register_clip(graded_clip)
                        video_clip = graded_clip
                except Exception as fallback_e:
                    self.logger.warning(f"Failed to apply fallback color grading: {fallback_e}")
            
            # Apply advanced enhancement filters (optional - can be enabled based on quality profile)
            quality_profile = getattr(self.config.video, 'video_quality_profile', 'standard').lower()
            if quality_profile == 'maximum':
                try:
                    enhanced_clip = self.advanced_enhancer.apply_enhancement_filters(
                        video_clip,
                        enable_denoising=True,
                        enable_sharpening=True
                    )
                    if enhanced_clip != video_clip:
                        resource_manager.register_clip(enhanced_clip)
                        video_clip = enhanced_clip
                except Exception as e:
                    self.logger.warning(f"Failed to apply advanced enhancement filters: {e}")
            
            return video_clip
            
        except Exception as e:
            self.logger.error(f"Critical error in visual effects: {e}")
            return video_clip
    
    def _add_text_overlays(self, video_clip: VideoFileClip, analysis: VideoAnalysis) -> VideoFileClip:
        """Add text overlays with error handling"""
        try:
            if analysis.text_overlays:
                overlay_clip = self.text_processor.add_text_overlays(video_clip, analysis.text_overlays)
                return overlay_clip
            return video_clip
            
        except Exception as e:
            self.logger.warning(f"Failed to add text overlays: {e}")
            return video_clip
    
    def _process_audio(self, video_clip: VideoFileClip, analysis: VideoAnalysis,
                      background_music_path: Optional[Path], resource_manager: ResourceManager,
                      temp_manager: TemporaryFileManager) -> VideoFileClip:
        """Process audio with comprehensive error handling"""
        try:
            # Process audio
            final_audio = self.audio_processor.process_audio(
                video_clip,
                analysis.narrative_script_segments,
                background_music_path
            )
            
            if final_audio:
                audio_clip = MoviePyCompat.with_audio(video_clip, final_audio)
                resource_manager.register_clip(final_audio)
                resource_manager.register_clip(audio_clip)
                return audio_clip
            else:
                self.logger.warning("Audio processing returned None, keeping original audio")
                return video_clip
                
        except Exception as e:
            self.logger.error(f"Audio processing failed: {e}")
            # Fallback to original audio
            return video_clip
    
    def _apply_duration_constraints(self, video_clip: VideoFileClip) -> VideoFileClip:
        """Apply duration constraints with error handling"""
        try:
            target_duration = self.config.video.target_duration
            if video_clip.duration > target_duration:
                trimmed_clip = MoviePyCompat.subclip(video_clip, 0, target_duration)
                return trimmed_clip
            return video_clip
            
        except Exception as e:
            self.logger.warning(f"Failed to apply duration constraints: {e}")
            return video_clip
    
    def _write_video_with_retry(self, video_clip: VideoFileClip, output_path: Path,
                               temp_manager: TemporaryFileManager, max_retries: int = 2) -> bool:
        """Write video with retry logic and temporary file management"""
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt} for video writing")
                
                # Create temporary output path for atomic write
                temp_output = temp_manager.create_temp_file(suffix=output_path.suffix)
                
                # Write to temporary file first
                self._write_video(video_clip, temp_output)
                
                # Verify the temporary file was created successfully
                if not temp_output.exists() or temp_output.stat().st_size == 0:
                    raise ValueError("Temporary video file is empty or missing")
                
                # Move to final location atomically
                shutil.move(str(temp_output), str(output_path))
                
                self.logger.info(f"Successfully wrote video to {output_path}")
                return True
                
            except Exception as e:
                self.logger.error(f"Video write attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    self.logger.error("All video write attempts failed")
                    return False
                else:
                    # Clean up failed attempt
                    try:
                        if temp_output and temp_output.exists():
                            temp_output.unlink()
                    except:
                        pass
        
        return False
    
    def _prepare_video_clip(self, clip: VideoFileClip) -> VideoFileClip:
        """Prepare video clip with standard formatting"""
        try:
            # Resize to target resolution
            target_width, target_height = self.config.video.target_resolution
            
            # Calculate crop for 9:16 aspect ratio
            current_width, current_height = clip.size
            current_aspect = current_width / current_height
            target_aspect = target_width / target_height
            
            try:
                if current_aspect > target_aspect:
                    # Video is wider - crop width
                    new_width = int(current_height * target_aspect)
                    x_offset = (current_width - new_width) // 2
                    clip = MoviePyCompat.crop(clip, x1=x_offset, x2=x_offset + new_width)
                elif current_aspect < target_aspect:
                    # Video is taller - crop height
                    new_height = int(current_width / target_aspect)
                    y_offset = (current_height - new_height) // 2
                    clip = MoviePyCompat.crop(clip, y1=y_offset, y2=y_offset + new_height)
                
                # Resize to exact target resolution
                clip = MoviePyCompat.resize(clip, (target_width, target_height))
            except AttributeError as e:
                self.logger.warning(f"Error with crop/resize methods: {e}")
                # Fallback: use subclip if available, or return original clip
                if hasattr(clip, 'subclip'):
                    clip = MoviePyCompat.subclip(clip, 0, min(clip.duration, 60))  # Limit duration
            
            # Set target FPS
            if hasattr(clip, 'fps') and clip.fps:
                if clip.fps != self.config.video.target_fps:
                    clip = clip.set_fps(self.config.video.target_fps)
            
            return clip
            
        except Exception as e:
            self.logger.warning(f"Error preparing video clip: {e}")
            return clip
    
    def _write_video(self, clip: VideoFileClip, output_path: Path):
        """Write video with optimized settings based on quality profile"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Get quality parameters from config
            quality_profile = getattr(self.config.video, 'video_quality_profile', 'standard').lower()
            speed_optimization = getattr(self.config.video, 'enable_speed_optimization', True)
            speed_level = getattr(self.config.video, 'speed_optimization_level', 'aggressive').lower()
            
            # Use speed optimizer for optimal settings if enabled
            if speed_optimization:
                optimal_settings = self.speed_optimizer.get_optimal_settings(quality_profile, speed_level)
                codec = optimal_settings['codec']
                preset = optimal_settings['preset']
                crf = optimal_settings['crf']
                bitrate = optimal_settings['bitrate']
                optimal_threads = optimal_settings['threads']
                ffmpeg_extra_args = optimal_settings['ffmpeg_params']
                additional_params = optimal_settings['additional_params']
                
                self.logger.info(f"Using speed-optimized settings: {codec}/{preset}, CRF={crf}, {optimal_threads} threads")
            else:
                # Fallback to original config-based settings
                codec = self.config.video.video_codec_cpu
                preset = self.config.video.ffmpeg_cpu_preset
                crf = self.config.video.ffmpeg_crf_cpu
                bitrate = self.config.video.video_bitrate_high
                optimal_threads = psutil.cpu_count() or 4
                ffmpeg_extra_args = []
                additional_params = {}
            
            # GPU detection and codec validation
            if 'nvenc' in codec:
                try:
                    subprocess.run(['nvidia-smi'], capture_output=True, check=True)
                    self.logger.info(f"[SUCCESS] Using GPU encoding: {codec}/{preset}")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    self.logger.warning("NVENC requested but GPU not available, falling back to CPU")
                    codec = 'libx264'
                    preset = 'ultrafast'
                    # Update ffmpeg params for CPU fallback
                    ffmpeg_extra_args = ['-crf', crf, '-tune', 'fastdecode,zerolatency']
            else:
                self.logger.info(f"Using CPU encoding: {codec}/{preset}")
            
            # Ensure CRF/CQ parameter is correctly set based on codec
            if 'nvenc' in codec:
                # NVENC uses -cq instead of -crf
                if '-crf' in ffmpeg_extra_args:
                    crf_index = ffmpeg_extra_args.index('-crf')
                    ffmpeg_extra_args[crf_index] = '-cq'
                elif '-cq' not in ffmpeg_extra_args:
                    ffmpeg_extra_args.extend(['-cq', crf])
            elif '264' in codec or '265' in codec:
                # x264/x265 uses -crf
                if '-cq' in ffmpeg_extra_args:
                    cq_index = ffmpeg_extra_args.index('-cq')
                    ffmpeg_extra_args[cq_index] = '-crf'
                elif '-crf' not in ffmpeg_extra_args:
                    ffmpeg_extra_args.extend(['-crf', crf])
            
            # Write video with enhanced settings
            self.logger.info(f"Starting video encoding with profile: {quality_profile}, speed: {speed_level}")
            start_time = time.time()
            
            clip.write_videofile(
                str(output_path),
                codec=codec,
                preset=preset,
                audio_codec=self.config.video.audio_codec,
                threads=optimal_threads,
                logger='bar',  # Shows progress bar
                ffmpeg_params=ffmpeg_extra_args,
                **additional_params
            )
            
            encoding_time = time.time() - start_time
            self.logger.info(f"Video encoding completed in {encoding_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error writing video: {e}")
            raise


def create_video_processor() -> VideoProcessor:
    """Factory function to create a video processor"""
    return VideoProcessor()