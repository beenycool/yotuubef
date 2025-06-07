"""
Modular video processing pipeline for YouTube Shorts generation.
Handles video downloading, processing, effects, and optimization.
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

import cv2
import numpy as np
from moviepy.editor import (
    VideoFileClip, AudioFileClip, CompositeVideoClip, CompositeAudioClip,
    TextClip, ImageClip, ColorClip, concatenate_videoclips, concatenate_audioclips
)
import moviepy.video.fx.all as vfx
from moviepy.video.fx import speedx
import yt_dlp

from src.config.settings import get_config
from src.models import VideoAnalysis, TextOverlay, NarrativeSegment, VisualCue


class ResourceManager:
    """Manages system resources during video processing"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.active_clips = []
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within safe limits"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent < (self.config.video.max_memory_usage * 100)
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
            
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'best[height<=1080]/best',  # Prefer 1080p or lower
                'outtmpl': str(output_path.with_suffix('.%(ext)s')),
                'quiet': True,
                'no_warnings': True,
                'extractaudio': False,
                'audioformat': 'mp3',
                'embed_chapters': False,
                'embed_info': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Download the video
                ydl.download([url])
            
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
                    'format': info.get('ext', 'unknown')
                }
                
        except Exception as e:
            self.logger.error(f"Error getting video info from {url}: {e}")
            return None


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
            
            return clip.fl(zoom_effect)
            
        except Exception as e:
            self.logger.warning(f"Error applying zoom effect: {e}")
            return clip
    
    def apply_color_grading(self, clip: VideoFileClip, intensity: float = 0.7) -> VideoFileClip:
        """Apply color grading for visual enhancement"""
        if not self.config.effects.color_grade_enabled:
            return clip
        
        try:
            # Apply brightness and contrast adjustments
            clip = clip.fx(vfx.colorx, factor=1 + intensity * 0.1)  # Slight brightness
            clip = clip.fx(vfx.lum_contrast, lum=0, contrast=intensity * 0.2)  # Contrast
            
            return clip
            
        except Exception as e:
            self.logger.warning(f"Error applying color grading: {e}")
            return clip
    
    def apply_speed_effects(self, clip: VideoFileClip, speed_effects: List[Dict]) -> VideoFileClip:
        """Apply speed effects based on AI analysis"""
        if not speed_effects:
            return clip
        
        try:
            segments = []
            current_time = 0
            
            for effect in speed_effects:
                start_time = effect.get('start_seconds', 0)
                end_time = effect.get('end_seconds', clip.duration)
                speed_factor = effect.get('speed_factor', 1.0)
                
                # Add normal segment before effect if needed
                if current_time < start_time:
                    segments.append(clip.subclip(current_time, start_time))
                
                # Add speed-affected segment
                effect_segment = clip.subclip(start_time, end_time)
                if speed_factor != 1.0:
                    effect_segment = effect_segment.fx(speedx, speed_factor)
                
                segments.append(effect_segment)
                current_time = end_time
            
            # Add remaining segment if any
            if current_time < clip.duration:
                segments.append(clip.subclip(current_time, clip.duration))
            
            if len(segments) > 1:
                return concatenate_videoclips(segments)
            elif segments:
                return segments[0]
            else:
                return clip
                
        except Exception as e:
            self.logger.warning(f"Error applying speed effects: {e}")
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
            overlay = overlay.set_opacity(0.1 * intensity)
            overlay = overlay.set_start(timestamp).set_duration(duration)
            
            return overlay
            
        except Exception as e:
            self.logger.warning(f"Error creating zoom highlight: {e}")
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
            border = border.set_start(timestamp).set_duration(duration)
            
            return border
            
        except Exception as e:
            self.logger.warning(f"Error creating highlight effect: {e}")
            return None


class TextOverlayProcessor:
    """Handles text overlays and subtitles"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def add_text_overlays(self, clip: VideoFileClip, overlays: List[TextOverlay]) -> VideoFileClip:
        """Add text overlays to video"""
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
    
    def _create_text_clip(self, video_clip: VideoFileClip, overlay: TextOverlay) -> Optional[TextClip]:
        """Create a single text clip"""
        try:
            # Get font and size based on style
            font_path = self.config.get_font_path(self.config.text_overlay.graphical_font)
            
            # Calculate font size based on video resolution
            video_height = video_clip.h
            font_size = int(video_height * self.config.text_overlay.graphical_font_size_ratio)
            
            # Adjust font size based on overlay style
            if overlay.style == "dramatic":
                font_size = int(font_size * 1.3)
            elif overlay.style == "highlight":
                font_size = int(font_size * 1.1)
            
            # Create text clip
            text_clip = TextClip(
                overlay.text,
                fontsize=font_size,
                font=font_path,
                color=self.config.text_overlay.graphical_text_color,
                stroke_color=self.config.text_overlay.graphical_stroke_color,
                stroke_width=self.config.text_overlay.graphical_stroke_width,
                method='caption' if len(overlay.text) > 20 else 'label'
            )
            
            # Set position
            if overlay.position == "center":
                text_clip = text_clip.set_position('center')
            elif overlay.position == "top":
                text_clip = text_clip.set_position(('center', 0.1), relative=True)
            elif overlay.position == "bottom":
                text_clip = text_clip.set_position(('center', 0.9), relative=True)
            else:
                text_clip = text_clip.set_position('center')
            
            # Set timing
            text_clip = text_clip.set_start(overlay.timestamp_seconds)
            text_clip = text_clip.set_duration(overlay.duration)
            
            # Add animation based on style
            if overlay.style == "dramatic":
                # Add fade in/out
                text_clip = text_clip.crossfadein(0.3).crossfadeout(0.3)
            
            return text_clip
            
        except Exception as e:
            self.logger.warning(f"Error creating text clip: {e}")
            return None


class AudioProcessor:
    """Handles audio processing and mixing"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def process_audio(self,
                      video_clip: VideoFileClip,
                      narrative_segments: List[NarrativeSegment],
                      background_music_path: Optional[Path] = None) -> Optional[CompositeAudioClip]:
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
            
            # Combine all audio tracks
            if audio_tracks:
                try:
                    return CompositeAudioClip(audio_tracks)
                except Exception as e:
                    self.logger.error(f"Failed to create composite audio: {e}")
                    # Fallback to first available track
                    return audio_tracks[0] if audio_tracks else None
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
                original_audio = video_clip.audio.multiply_volume(
                    self.config.audio.original_audio_mix_volume
                )
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
        
        try:
            # This would integrate with TTS service to generate audio
            # For now, return empty list but with proper error handling structure
            self.logger.info(f"Processing {len(segments)} narrative segments")
            
            successful_segments = []
            for i, segment in enumerate(segments):
                try:
                    # Placeholder for actual TTS processing
                    # In real implementation, this would:
                    # 1. Call TTS service with segment.text
                    # 2. Apply segment.emotion and segment.pacing
                    # 3. Set timing based on segment.time_seconds
                    # 4. Return AudioFileClip
                    
                    self.logger.debug(f"Processing TTS segment {i+1}: '{segment.text[:50]}...'")
                    # For now, skip actual TTS generation
                    pass
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process TTS segment {i+1}: {e}")
                    continue
            
            self.logger.info(f"Successfully processed {len(successful_segments)} TTS segments")
            return successful_segments
            
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
            music_clip = music_clip.subclip(0, video_duration)
            
            # Apply base volume
            music_clip = music_clip.multiply_volume(self.config.audio.background_music_volume)
            
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
                    segments.append(music_clip.subclip(current_time, start_time))
                
                # Add ducked segment during narration
                if start_time < music_clip.duration:
                    duck_end = min(end_time, music_clip.duration)
                    ducked_segment = music_clip.subclip(start_time, duck_end)
                    ducked_segment = ducked_segment.multiply_volume(duck_factor)
                    segments.append(ducked_segment)
                
                current_time = end_time
            
            # Add remaining segment
            if current_time < music_clip.duration:
                segments.append(music_clip.subclip(current_time, music_clip.duration))
            
            if segments:
                return concatenate_audioclips(segments)
            else:
                return music_clip
                
        except Exception as e:
            self.logger.warning(f"Error applying audio ducking: {e}")
            return music_clip
    
    def _process_narrative_segments(self, segments: List[NarrativeSegment]) -> List[AudioFileClip]:
        """Process TTS narrative segments (placeholder for TTS integration)"""
        # This would integrate with TTS service to generate audio
        # For now, return empty list
        return []


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
    
    def process_video(self,
                      video_path: Path,
                      output_path: Path,
                      analysis: VideoAnalysis,
                      background_music_path: Optional[Path] = None) -> bool:
        """
        Main video processing pipeline with enhanced error handling
        
        Args:
            video_path: Input video path
            output_path: Output video path
            analysis: AI analysis results (validated Pydantic model)
            background_music_path: Optional background music
        
        Returns:
            True if processing successful, False otherwise
        """
        with ResourceManager() as resource_manager, TemporaryFileManager() as temp_manager:
            try:
                self.logger.info(f"Starting video processing: {video_path}")
                
                # Stage 1: Input Validation and Preparation
                video_clip = self._load_source_clip(video_path, output_path, analysis, resource_manager)
                if not video_clip:
                    return False
                
                # Stage 2: Audio Synthesis and Composition
                composite_audio = self._synthesize_audio(video_clip.audio, analysis, background_music_path, resource_manager, temp_manager)
                
                # Stage 3: Visual Effects and Composition
                final_visuals = self._compose_visuals(video_clip, analysis, resource_manager)
                
                # Stage 4: Combine and Render
                final_clip = final_visuals.set_audio(composite_audio) if composite_audio else final_visuals
                success = self._render_video(final_clip, output_path, temp_manager)
                
                if success:
                    self.logger.info(f"Video processing completed: {output_path}")
                else:
                    self.logger.error("Failed to render final video")
                
                return success
                
            except Exception as e:
                self.logger.error(f"Critical error in video processing: {e}", exc_info=True)
                return False
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
            self.logger.info("Starting audio synthesis")
            
            # Process TTS narration if needed
            narrative_clips = []
            if analysis.narrative_script_segments and not analysis.original_audio_is_key:
                try:
                    narrative_clips = self._process_tts_segments(analysis.narrative_script_segments)
                    self.logger.info(f"Generated {len(narrative_clips)} TTS segments")
                except Exception as e:
                    self.logger.warning(f"TTS processing failed, continuing without narration: {e}")
            
            # Add background music if available
            music_clip = None
            if background_music_path and background_music_path.exists():
                try:
                    music_clip = self._add_background_music(
                        background_music_path,
                        source_audio.duration if source_audio else 60,
                        analysis.narrative_script_segments
                    )
                    if music_clip:
                        resource_manager.register_clip(music_clip)
                        self.logger.info("Added background music with ducking")
                except Exception as e:
                    self.logger.warning(f"Background music processing failed: {e}")
            
            # Composite all audio elements
            audio_elements = []
            
            # Add original audio if it's important
            if analysis.original_audio_is_key and source_audio:
                audio_elements.append(source_audio)
                self.logger.info("Preserved original audio")
            
            # Add background music
            if music_clip:
                audio_elements.append(music_clip)
            
            # Add TTS narration
            if narrative_clips:
                audio_elements.extend(narrative_clips)
                for clip in narrative_clips:
                    resource_manager.register_clip(clip)
            
            # Create composite audio
            if len(audio_elements) > 1:
                composite_audio = CompositeAudioClip(audio_elements)
                resource_manager.register_clip(composite_audio)
                self.logger.info(f"Created composite audio with {len(audio_elements)} elements")
                return composite_audio
            elif audio_elements:
                self.logger.info("Using single audio element")
                return audio_elements[0]
            else:
                self.logger.warning("No audio elements available, using silent audio")
                return None
            
        except Exception as e:
            self.logger.error(f"Error in audio synthesis: {e}")
            # Return original audio as fallback
            return source_audio if source_audio else None
    
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
            self.logger.info("Starting visual composition")
            video_clip = source_clip
            
            # Apply visual effects with graceful error handling
            video_clip = self._apply_visual_effects(video_clip, analysis, resource_manager)
            
            # Add text overlays with error handling
            video_clip = self._add_text_overlays(video_clip, analysis)
            
            # Apply duration constraints
            video_clip = self._apply_duration_constraints(video_clip)
            
            self.logger.info("Visual composition completed")
            return video_clip
            
        except Exception as e:
            self.logger.error(f"Error in visual composition: {e}")
            # Return source clip as fallback
            return source_clip
    
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
            
            # Prepare video for processing
            prepared_clip = self._prepare_video_clip(video_clip)
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
            
            # Apply speed effects
            if analysis.speed_effects:
                try:
                    speed_clip = self.effects.apply_speed_effects(video_clip, analysis.speed_effects)
                    if speed_clip != video_clip:
                        resource_manager.register_clip(speed_clip)
                        video_clip = speed_clip
                except Exception as e:
                    self.logger.warning(f"Failed to apply speed effects: {e}")
            
            # Apply visual enhancements
            try:
                zoom_clip = self.effects.apply_subtle_zoom(video_clip)
                if zoom_clip != video_clip:
                    resource_manager.register_clip(zoom_clip)
                    video_clip = zoom_clip
            except Exception as e:
                self.logger.warning(f"Failed to apply zoom effect: {e}")
            
            try:
                graded_clip = self.effects.apply_color_grading(video_clip)
                if graded_clip != video_clip:
                    resource_manager.register_clip(graded_clip)
                    video_clip = graded_clip
            except Exception as e:
                self.logger.warning(f"Failed to apply color grading: {e}")
            
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
                audio_clip = video_clip.set_audio(final_audio)
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
                trimmed_clip = video_clip.subclip(0, target_duration)
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
            
            if current_aspect > target_aspect:
                # Video is wider - crop width
                new_width = int(current_height * target_aspect)
                x_offset = (current_width - new_width) // 2
                clip = clip.crop(x1=x_offset, x2=x_offset + new_width)
            elif current_aspect < target_aspect:
                # Video is taller - crop height
                new_height = int(current_width / target_aspect)
                y_offset = (current_height - new_height) // 2
                clip = clip.crop(y1=y_offset, y2=y_offset + new_height)
            
            # Resize to exact target resolution
            clip = clip.resize((target_width, target_height))
            
            # Set target FPS
            if hasattr(clip, 'fps') and clip.fps:
                if clip.fps != self.config.video.target_fps:
                    clip = clip.set_fps(self.config.video.target_fps)
            
            return clip
            
        except Exception as e:
            self.logger.warning(f"Error preparing video clip: {e}")
            return clip
    
    def _write_video(self, clip: VideoFileClip, output_path: Path):
        """Write video with optimized settings"""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine encoding parameters
            codec = self.config.video.video_codec_cpu
            preset = self.config.video.ffmpeg_cpu_preset
            
            # Check for GPU encoding availability
            try:
                subprocess.run(['nvidia-smi'], capture_output=True, check=True)
                codec = self.config.video.video_codec_gpu
                preset = self.config.video.ffmpeg_gpu_preset
                self.logger.info("Using GPU encoding")
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.logger.info("Using CPU encoding")
            
            # Write video
            clip.write_videofile(
                str(output_path),
                codec=codec,
                preset=preset,
                audio_codec=self.config.video.audio_codec,
                threads=4,
                logger='bar'
            )
            
        except Exception as e:
            self.logger.error(f"Error writing video: {e}")
            raise


def create_video_processor() -> VideoProcessor:
    """Factory function to create a video processor"""
    return VideoProcessor()