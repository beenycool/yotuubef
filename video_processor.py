from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip
import moviepy.video.fx.all as vfx
import math
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
import os
import tempfile
import yt_dlp
from pathlib import Path
import subprocess
import psutil
try:
    from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False

# Text style presets
TEXT_STYLES = {
    "title": {
        "fontsize": 60,
        "color": "white",
        "font": "Arial-Bold",
        "stroke_color": "black",
        "stroke_width": 2,
        "align": "center"
    },
    "subtitle": {
        "fontsize": 40,
        "color": "white",
        "font": "Arial",
        "stroke_color": "black",
        "stroke_width": 1,
        "align": "center"
    },
    "caption": {
        "fontsize": 30,
        "color": "white",
        "font": "Arial",
        "align": "center"
    },
    "narrative": {
        "fontsize": 50,
        "color": "white",
        "font": "Impact",
        "stroke_color": "black",
        "stroke_width": 4,
        "align": "center",
        "bg_color": "transparent"
    }
}

# Narrative text constants
NARRATIVE_FONT = "Impact"
NARRATIVE_FONT_SIZE = 50
NARRATIVE_TEXT_COLOR = "white"
NARRATIVE_STROKE_COLOR = "black"
NARRATIVE_STROKE_WIDTH = 4
NARRATIVE_POSITION = ("center", "center")
NARRATIVE_BG_COLOR = "transparent"

# Audio processing presets
AUDIO_PRESETS = {
    "dynamic_ducking": {
        "threshold": -20,  # dB
        "ratio": 4.0,
        "attack": 0.01,    # seconds
        "release": 0.5     # seconds
    },
    "mood_music": {
        "happy": "music/KOREAN_FUNK_-_SLOWED_128kbps_cut.mp3",
        "serious": "music/The_Art_Of_Ronaldo_128kbps_cut.mp3",
        "funky": "music/Funk_Secreto_Super_Slowed_128kbps_cut.mp3",
        "relaxed": "music/PASSO_BEM_SOLTO_Slowed_128kbps_cut.mp3"
    },
    "tts_pacing": {
        "slow": 0.8,
        "normal": 1.0,
        "fast": 1.2
    }
}

def apply_audio_ducking(clip, background_music, preset="dynamic_ducking"):
    """
    Apply dynamic ducking to background music when speech is detected.
    
    Args:
        clip: Video clip with primary audio
        background_music: Background music clip
        preset: Audio processing preset to use
        
    Returns:
        Composite audio clip with ducking applied
    """
    params = AUDIO_PRESETS.get(preset, AUDIO_PRESETS["dynamic_ducking"])
    
    # Create sidechain effect
    def sidechain(get_frame, t):
        # Get volume of main audio at time t
        main_vol = np.sqrt(np.mean(get_frame(t)**2))
        
        # Apply ducking curve
        if main_vol > 10**(params["threshold"]/20):
            gain = 1/params["ratio"]
        else:
            gain = 1.0
            
        # Smooth transitions
        if t < params["attack"]:
            gain = np.interp(t, [0, params["attack"]], [1.0, gain])
        elif t > clip.duration - params["release"]:
            gain = np.interp(t, [clip.duration-params["release"], clip.duration], [gain, 1.0])
            
        return get_frame(t) * gain
    
    # Apply sidechain to background music
    ducked_music = background_music.fl(sidechain)
    
    # Combine audio tracks
    return CompositeAudioClip([clip.audio, ducked_music])

def select_music_by_mood(mood):
    """
    Select background music based on mood.
    
    Args:
        mood: String representing desired mood
        
    Returns:
        Path to selected music file
    """
    return AUDIO_PRESETS["mood_music"].get(mood, "music/KOREAN_FUNK_-_SLOWED_128kbps_cut.mp3")

def normalize_audio_levels(clip, target_lufs=-14.0):
    """
    Normalize audio to target loudness level.
    
    Args:
        clip: Audio clip to normalize
        target_lufs: Target loudness in LUFS
        
    Returns:
        Normalized audio clip
    """
    # Calculate current loudness
    current_loudness = clip.audio.max_volume()
    
    # Calculate required gain adjustment
    gain = 10**((target_lufs - current_loudness)/20)
    
    # Apply gain with protection against clipping
    return clip.audio.multiply_volume(min(gain, 4.0))  # Max 4x gain

def adjust_tts_pacing(text_clip, speed="normal"):
    """
    Adjust TTS pacing by changing duration.
    
    Args:
        text_clip: TextClip to adjust
        speed: Pacing preset ("slow", "normal", "fast")
        
    Returns:
        Adjusted TextClip
    """
    factor = AUDIO_PRESETS["tts_pacing"].get(speed, 1.0)
    return text_clip.with_duration(text_clip.duration * factor)

# Check for GPU availability and capabilities
GPU_AVAILABLE = False
FFMPEG_GPU_PARAMS = {
    'codec': 'libx264',
    'preset': 'medium',
    'threads': 4
}

# Try NVIDIA first
try:
    subprocess.check_output(['nvidia-smi'])
    GPU_AVAILABLE = True
    FFMPEG_GPU_PARAMS = {
        'codec': 'h264_nvenc',
        'preset': 'fast',
        'threads': 0
    }
except:
    # Try AMD
    try:
        subprocess.check_output(['rocminfo'])
        GPU_AVAILABLE = True
        FFMPEG_GPU_PARAMS = {
            'codec': 'h264_amf',
            'preset': 'fast',
            'threads': 0
        }
    except:
        # Try Intel
        try:
            subprocess.check_output(['vainfo'])
            GPU_AVAILABLE = True
            FFMPEG_GPU_PARAMS = {
                'codec': 'h264_qsv',
                'preset': 'fast',
                'threads': 0
            }
        except:
            GPU_AVAILABLE = False

# Memory management settings
MAX_MEMORY_USAGE = 0.8  # Max 80% of available memory
VIDEO_CHUNK_SIZE = 30  # Process in 30-second chunks for large files

# Focus point can be a single point or a list of points with timestamps
FocusPoint = Dict[str, float]  # {"x": 0.7, "y": 0.3}
TimestampedFocusPoint = Dict[str, Union[float, FocusPoint]]  # {"time": 1.5, "point": {"x": 0.7, "y": 0.3}}

def check_memory_usage():
    """Check if memory usage is within safe limits"""
    mem = psutil.virtual_memory()
    return mem.percent < (MAX_MEMORY_USAGE * 100)

def get_gpu_utilization():
    """Get GPU utilization percentage if available"""
    if not GPU_AVAILABLE or not NVML_AVAILABLE:
        return None
        
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        if device_count == 0:
            return None
            
        handle = nvmlDeviceGetHandleByIndex(0)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        return (mem_info.used / mem_info.total) * 100
    except:
        return None

def create_short_clip(video_path, output_path, start_time, end_time, focus_points, text_overlays=None):
    """
    Creates a short video clip, cropped to 9:16 aspect ratio centered
    around the provided focus_point(s).
    
    Args:
        video_path: Path to the input video file
        output_path: Path to save the output video
        start_time: Start time of the clip in seconds
        end_time: End time of the clip in seconds
        focus_points: Either a single focus point {"x": 0.7, "y": 0.3} or
                     a list of timestamped focus points for dynamic panning
                     [{"time": 1.5, "point": {"x": 0.7, "y": 0.3}}, ...]
    """
    try:
        # Check memory before processing
        if not check_memory_usage():
            raise MemoryError("System memory usage too high for safe video processing")
            
        # Check GPU utilization if available
        gpu_util = get_gpu_utilization()
        if gpu_util and gpu_util > 90:
            print("Warning: High GPU utilization detected - performance may be degraded")
            
        # Process in chunks if video is longer than 2 minutes
        clip_duration = end_time - start_time
        if clip_duration > 120:
            print(f"Processing long video ({clip_duration}s) in chunks...")
            # Chunk processing logic would go here
            # For now we'll proceed with original approach
            # but with memory monitoring
            
        clip = VideoFileClip(video_path).subclip(start_time, end_time)
        original_width, original_height = clip.size
        target_aspect_ratio = 9 / 16
        clip_duration = clip.duration

        # Determine target dimensions for 9:16 crop
        if original_width / original_height > target_aspect_ratio:
            # Original is wider than 9:16 (e.g., 16:9) - crop sides
            new_height = original_height
            new_width = math.ceil(new_height * target_aspect_ratio) # Use ceil to avoid rounding issues
        else:
            # Original is taller than or equal to 9:16 - crop top/bottom
            new_width = original_width
            new_height = math.ceil(new_width / target_aspect_ratio) # Use ceil
        
        # Normalize focus points to a list of timestamped points
        timestamped_focus_points = normalize_focus_points(focus_points, start_time, clip_duration)
        
        if len(timestamped_focus_points) == 1:
            # Static crop with a single focus point
            print("Using static crop with single focus point")
            focus_point = timestamped_focus_points[0]["point"]
            cropped_clip = apply_static_crop(clip, focus_point, new_width, new_height, original_width, original_height)
        else:
            # Dynamic panning between multiple focus points
            print(f"Using dynamic panning with {len(timestamped_focus_points)} focus points")
            cropped_clip = apply_dynamic_panning(clip, timestamped_focus_points, new_width, new_height, original_width, original_height)


        # Resize to standard short resolution (1080x1920)
        target_height = 1920
        target_width = 1080
        final_clip = cropped_clip.resize(height=target_height)

        # Add text overlays with fade animations if provided
        if text_overlays:
            clips = [final_clip]
            for overlay in text_overlays:
                style = TEXT_STYLES.get(overlay.get("style", "caption"), {})
                txt_clip = TextClip(
                    overlay["text"],
                    fontsize=style.get("fontsize", 30),
                    color=style.get("color", "white"),
                    font=style.get("font", "Arial"),
                    stroke_color=style.get("stroke_color", None),
                    stroke_width=style.get("stroke_width", 0),
                    align=style.get("align", "center")
                ).set_position(overlay.get("position", "center"))
                
                # Apply fade in/out if specified
                if overlay.get("fade_in", 0) > 0:
                    txt_clip = txt_clip.crossfadein(overlay["fade_in"])
                if overlay.get("fade_out", 0) > 0:
                    txt_clip = txt_clip.crossfadeout(overlay["fade_out"])
                
                # Set duration and start time
                txt_clip = txt_clip.set_start(overlay.get("start_time", 0))
                txt_clip = txt_clip.set_duration(overlay.get("duration", final_clip.duration))
                
                clips.append(txt_clip)
            
            final_clip = CompositeVideoClip(clips)


        print(f"Writing final clip to: {output_path}")
        print(f"Using {'GPU' if GPU_AVAILABLE else 'CPU'} acceleration")
        final_clip.write_videofile(
            output_path,
            audio_codec="aac",
            logger='bar',
            **FFMPEG_GPU_PARAMS
        )

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        # Optionally re-raise the exception or handle it as needed
        raise
    finally:
        # Ensure clips are closed to release resources
        if 'clip' in locals() and clip:
            clip.close()
        if 'cropped_clip' in locals() and cropped_clip:
            cropped_clip.close()
        if 'final_clip' in locals() and final_clip:
            final_clip.close()

def normalize_focus_points(focus_points, start_time: float, clip_duration: float) -> List[Dict]:
    """
    Normalize focus points to a list of timestamped points.
    
    Args:
        focus_points: Either a single focus point or a list of timestamped focus points
        start_time: Start time of the clip in seconds
        clip_duration: Duration of the clip in seconds
        
    Returns:
        List of timestamped focus points with relative timestamps
    """
    # If focus_points is a single point (not a list), convert to a list with one item
    if not isinstance(focus_points, list):
        # Single focus point - use it for the entire clip
        return [{"time": 0, "point": focus_points}]
    
    # If we have a list of timestamped focus points, normalize them
    normalized_points = []
    for point_data in focus_points:
        # Adjust timestamp to be relative to the clip's start time
        time_in_clip = point_data.get("time", 0) - start_time
        # Clamp time to be within the clip duration
        time_in_clip = max(0, min(time_in_clip, clip_duration))
        
        normalized_points.append({
            "time": time_in_clip,
            "point": point_data.get("point", {"x": 0.5, "y": 0.5})  # Default to center if no point
        })
    
    # Sort by time
    normalized_points.sort(key=lambda x: x["time"])
    
    # If no points at the start, add the first point at time 0
    if not normalized_points or normalized_points[0]["time"] > 0:
        first_point = normalized_points[0]["point"] if normalized_points else {"x": 0.5, "y": 0.5}
        normalized_points.insert(0, {"time": 0, "point": first_point})
    
    # If no points at the end, add the last point at the end
    if normalized_points[-1]["time"] < clip_duration:
        normalized_points.append({"time": clip_duration, "point": normalized_points[-1]["point"]})
    
    return normalized_points

def apply_static_crop(clip, focus_point, new_width, new_height, original_width, original_height):
    """
    Apply a static crop centered around a single focus point.
    """
    # Calculate the ideal center based on the focus point
    ideal_center_x = original_width * focus_point['x']
    ideal_center_y = original_height * focus_point['y']
    
    # Calculate initial crop coordinates (x1, y1) based on ideal center
    x1 = ideal_center_x - new_width / 2
    y1 = ideal_center_y - new_height / 2
    
    # Clamp coordinates to ensure the crop window stays within bounds
    # Adjust x1 and ensure x2 doesn't exceed original_width
    if x1 < 0:
        x1 = 0
    elif x1 + new_width > original_width:
        x1 = original_width - new_width
    
    # Adjust y1 and ensure y2 doesn't exceed original_height
    if y1 < 0:
        y1 = 0
    elif y1 + new_height > original_height:
        y1 = original_height - new_height
    
    # Ensure coordinates are integers for cropping
    x1 = int(x1)
    y1 = int(y1)
    effective_width = int(new_width)
    effective_height = int(new_height)
    
    # Calculate center point after clamping
    crop_center_x = x1 + new_width / 2
    crop_center_y = y1 + new_height / 2
    
    # Apply the crop
    return vfx.crop(clip, width=new_width, height=new_height, 
                   x_center=crop_center_x, y_center=crop_center_y)

def apply_dynamic_panning(clip, focus_points, new_width, new_height, original_width, original_height, initial_zoom=1.0, target_zoom=1.1, easing="ease_in_out"):
    """
    Apply dynamic panning between multiple focus points.
    
    Args:
        clip: The video clip to crop
        focus_points: List of timestamped focus points
        new_width, new_height: Dimensions of the crop window
        original_width, original_height: Original video dimensions
    """
    # Convert to integers for cropping
    effective_width = int(new_width)
    effective_height = int(new_height)
    
    # Calculate zoom progression
    zoom_factors = np.linspace(initial_zoom, target_zoom, len(focus_points))
    max_zoom = max(initial_zoom, target_zoom)
    
    def get_crop_center_at_time(t, zoom_factors=None):
        # Calculate current zoom factor
        current_zoom = np.interp(t,
                               [fp['time'] for fp in focus_points],
                               zoom_factors)
        current_zoom = max(1.0, min(current_zoom, 1.2))  # Clamp zoom between 1.0 and 1.2
        """
        Calculate the crop center coordinates at a specific time using interpolation
        between focus points.
        """
        # Find the two focus points that surround the current time
        next_point_idx = 0
        while next_point_idx < len(focus_points) and focus_points[next_point_idx]["time"] < t:
            next_point_idx += 1
            
        if next_point_idx == 0:
            # Before the first point, use the first point
            point = focus_points[0]["point"]
            center_x = original_width * point["x"]
            center_y = original_height * point["y"]
        elif next_point_idx >= len(focus_points):
            # After the last point, use the last point
            point = focus_points[-1]["point"]
            center_x = original_width * point["x"]
            center_y = original_height * point["y"]
        else:
            # Interpolate between two points
            prev_data = focus_points[next_point_idx - 1]
            next_data = focus_points[next_point_idx]
            
            prev_time = prev_data["time"]
            next_time = next_data["time"]
            
            # Calculate interpolation factor (0 to 1)
            if next_time == prev_time:  # Avoid division by zero
                factor = 0
            else:
                factor = (t - prev_time) / (next_time - prev_time)
                
            # Apply selected easing function
            if easing == "ease_in":
                factor = factor ** 2
            elif easing == "ease_out":
                factor = 1 - (1 - factor) ** 2
            elif easing == "ease_in_out":
                factor = factor ** 2 * (3 - 2 * factor)
            elif easing == "linear":
                pass  # No change
            else:  # Default to ease_in_out
                factor = factor ** 2 * (3 - 2 * factor)
            
            # Interpolate between the two points
            prev_point = prev_data["point"]
            next_point = next_data["point"]
            
            interp_x = prev_point["x"] + factor * (next_point["x"] - prev_point["x"])
            interp_y = prev_point["y"] + factor * (next_point["y"] - prev_point["y"])
            
            center_x = original_width * interp_x
            center_y = original_height * interp_y
            
            # Handle zoom factors if provided
            if zoom_factors:
                current_zoom = np.interp(t,
                                       [fp['time'] for fp in focus_points],
                                       zoom_factors)
                current_zoom = max(1.0, min(current_zoom, 1.2))  # Clamp zoom
                effective_width = new_width / current_zoom
                effective_height = new_height / current_zoom
            else:
                effective_width = new_width
                effective_height = new_height
        
        # Enhanced boundary checks with safe margins
        min_x = max(new_width / 2, 0)
        max_x = original_width - new_width / 2
        min_y = max(new_height / 2, 0)
        max_y = original_height - new_height / 2

        # Apply smooth clamping with edge detection
        center_x = np.clip(center_x, min_x, max_x)
        center_y = np.clip(center_y, min_y, max_y)

        # Add buffer near edges to prevent jarring jumps
        edge_buffer = 50  # pixels
        if center_x <= min_x + edge_buffer:
            center_x = min_x + edge_buffer
        elif center_x >= max_x - edge_buffer:
            center_x = max_x - edge_buffer
        
        if center_y <= min_y + edge_buffer:
            center_y = min_y + edge_buffer
        elif center_y >= max_y - edge_buffer:
            center_y = max_y - edge_buffer
        
        return center_x, center_y
    
    # Create a function that crops each frame based on the interpolated focus point
    def crop_frame(get_frame, t):
        # Get the frame at time t
        frame = get_frame(t)
        
        # Calculate the crop center for this time
        center_x, center_y = get_crop_center_at_time(t)
        
        # Calculate crop coordinates
        x1 = int(center_x - effective_width / 2)
        y1 = int(center_y - effective_height / 2)
        
        # Crop the frame using the original dimensions (zoom is already applied in center calculation)
        cropped_frame = frame[y1:y1+effective_height, x1:x1+effective_width]
        return cropped_frame
    
    # Create a new clip with the dynamic cropping function
    return clip.fl(crop_frame)

# Example usage:
# Single focus point (static crop)
# focus_point_from_gemini = {"x": 0.7, "y": 0.3} 
# create_short_clip(
#     video_path="input.mp4",
#     output_path="output_short.mp4",
#     start_time=10, # seconds
#     end_time=25,   # seconds
#     focus_points=focus_point_from_gemini
# )
#
# Multiple focus points (dynamic panning)
# key_moments = [
#     {"time": 10.5, "point": {"x": 0.3, "y": 0.6}},
#     {"time": 15.2, "point": {"x": 0.7, "y": 0.4}},
#     {"time": 20.0, "point": {"x": 0.5, "y": 0.5}}
# ]
# create_short_clip(
#     video_path="input.mp4",
#     output_path="output_short_dynamic.mp4",
#     start_time=10, # seconds
#     end_time=25,   # seconds
#     focus_points=key_moments
# )

def apply_ken_burns_effect(clip, zoom_direction="in", duration=None, easing="ease_in_out"):
    """
    Apply Ken Burns effect (zoom in/out) to a clip.
    
    Args:
        clip: Video clip to process
        zoom_direction: "in" or "out"
        duration: Effect duration (None for full clip)
        easing: Easing function ("linear", "ease_in", etc.)
        
    Returns:
        Clip with Ken Burns effect applied
    """
    if duration is None:
        duration = clip.duration
        
    def make_frame(get_frame, t):
        frame = get_frame(t)
        
        # Calculate zoom factor based on time
        progress = min(t / duration, 1.0)
        
        # Apply easing
        if easing == "ease_in":
            progress = progress ** 2
        elif easing == "ease_out":
            progress = 1 - (1 - progress) ** 2
        elif easing == "ease_in_out":
            progress = progress ** 2 * (3 - 2 * progress)
            
        # Calculate zoom
        if zoom_direction == "in":
            zoom = 1.0 + progress * 0.3  # Zoom in 30%
        else:
            zoom = 1.3 - progress * 0.3  # Zoom out from 130%
            
        # Apply zoom transform
        return vfx.zoom(frame, zoom)
        
    return clip.fl(make_frame)

def apply_color_grading(clip, preset="cinematic"):
    """
    Apply color grading to a clip.
    
    Args:
        clip: Video clip to process
        preset: Color grading preset ("cinematic", "vibrant", "muted")
        
    Returns:
        Color graded clip
    """
    # Color grading presets (contrast, brightness, saturation)
    presets = {
        "cinematic": (1.1, 0.9, 0.9),
        "vibrant": (1.0, 1.0, 1.3),
        "muted": (0.9, 1.0, 0.7)
    }
    
    contrast, brightness, saturation = presets.get(preset, (1.0, 1.0, 1.0))
    
    # Apply color adjustments
    clip = vfx.colorx(clip, contrast)
    clip = vfx.lum_contrast(clip, lum=0, contrast=contrast, contrast_thr=100)
    clip = vfx.multiply_color(clip, brightness)
    clip = vfx.multiply_speed(clip, saturation)
    
    return clip

def create_complex_animation(clip, animation_type="slide_in", duration=1.0):
    """
    Create complex text/object animations.
    
    Args:
        clip: Clip to animate (usually TextClip)
        animation_type: Animation preset
        duration: Animation duration in seconds
        
    Returns:
        Animated clip
    """
    if animation_type == "slide_in":
        def pos_func(t):
            if t < duration:
                return ('center', 1.5 - t/duration)
            return 'center'
    elif animation_type == "fade_scale":
        def pos_func(t):
            if t < duration:
                scale = 0.5 + 0.5 * (t/duration)
                return ('center', 'center', scale)
            return 'center'
    else:
        return clip
        
    return clip.set_position(pos_func)

def process_video_with_gpu_optimization(input_path, output_path, processing_func, chunk_size=30):
    """
    Process video in chunks with GPU optimization if available.
    Handles sequential text overlays and enhanced audio processing.
    
    Args:
        input_path: Path to input video file
        output_path: Path to save processed video
        processing_func: Function to apply to each chunk
        chunk_size: Duration of each processing chunk in seconds
        
    Returns:
        Path to processed video file
    """
    try:
        # Check system resources
        if not check_memory_usage():
            raise MemoryError("Insufficient system memory for video processing")
            
        gpu_util = get_gpu_utilization()
        if gpu_util and gpu_util > 90:
            print("Warning: High GPU utilization - performance may be degraded")
            
        # Get video duration
        clip = VideoFileClip(input_path)
        duration = clip.duration
        clip.close()
        
        # Process in chunks if video is long
        if duration > chunk_size * 2:
            print(f"Processing {duration:.1f}s video in {chunk_size}s chunks...")
            
            # Create temp directory for chunks
            temp_dir = Path(tempfile.mkdtemp(prefix="video_chunks_"))
            temp_files = []
            
            try:
                # Process each chunk
                for i, start in enumerate(range(0, math.ceil(duration), chunk_size)):
                    end = min(start + chunk_size, duration)
                    chunk_path = temp_dir / f"chunk_{i}.mp4"
                    
                    # Process chunk
                    processing_func(
                        input_path=input_path,
                        output_path=str(chunk_path),
                        start_time=start,
                        end_time=end
                    )
                    temp_files.append(chunk_path)
                    
                # Concatenate chunks
                print("Combining processed chunks...")
                clips = [VideoFileClip(str(f)) for f in temp_files]
                final_clip = concatenate_videoclips(clips)
                final_clip.write_videofile(
                    output_path,
                    audio_codec="aac",
                    **FFMPEG_GPU_PARAMS
                )
                final_clip.close()
                
            finally:
                # Clean up temp files
                for f in temp_files:
                    try:
                        f.unlink()
                    except:
                        pass
                try:
                    temp_dir.rmdir()
                except:
                    pass
                    
        else:
            # Process entire video at once
            processing_func(
                input_path=input_path,
                output_path=output_path,
                start_time=0,
                end_time=duration
            )
            
        return Path(output_path)
        
    except Exception as e:
        print(f"Error processing video: {e}")
        raise
    """
    Process video in chunks using GPU acceleration when available.
    
    Args:
        input_path: Path to input video
        output_path: Path to save processed video
        processing_func: Function that processes a video clip (takes clip, returns processed clip)
        chunk_size: Duration of each processing chunk in seconds
    """
    try:
        # Check system resources
        if not check_memory_usage():
            raise MemoryError("Insufficient system memory for video processing")
            
        gpu_util = get_gpu_utilization()
        if gpu_util and gpu_util > 90:
            print("Warning: High GPU utilization - performance may be affected")

        # Get video duration
        with VideoFileClip(input_path) as clip:
            duration = clip.duration
            
        # Process in chunks if video is long
        if duration > chunk_size * 2:  # Only chunk if at least 2 chunks
            print(f"Processing {duration}s video in {chunk_size}s chunks...")
            
            clips = []
            for start in np.arange(0, duration, chunk_size):
                end = min(start + chunk_size, duration)
                print(f"Processing chunk {start:.1f}-{end:.1f}s")
                
                with VideoFileClip(input_path).subclip(start, end) as chunk:
                    processed = processing_func(chunk)
                    clips.append(processed)
                    
            # Concatenate all processed chunks
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(
                output_path,
                audio_codec="aac",
                logger='bar',
                **FFMPEG_GPU_PARAMS
            )
            final_clip.close()
        else:
            # Process whole video at once
            with VideoFileClip(input_path) as clip:
                processed = processing_func(clip)
                processed.write_videofile(
                    output_path,
                    audio_codec="aac",
                    logger='bar',
                    **FFMPEG_GPU_PARAMS
                )
                processed.close()
                
    except Exception as e:
        print(f"Error in GPU-optimized processing: {e}")
        raise

def _prepare_initial_video(submission, safe_title: str, temp_files: list) -> Tuple[Path, float, int, int]:
    """
    Downloads and prepares the initial video from a Reddit submission.
    
    Args:
        submission: PRAW submission object
        safe_title: Sanitized title for filename
        temp_files: List to track temporary files
        
    Returns:
        Tuple of (video_path, duration, width, height)
    """
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path(tempfile.gettempdir()) / "reddit_videos"
        temp_dir.mkdir(exist_ok=True)
        
        # Download the video using yt-dlp
        video_path = temp_dir / f"{submission.id}_{safe_title}.mp4"
        ydl_opts = {
            'outtmpl': str(video_path),
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'quiet': True,
            'socket_timeout': 30,         # Timeout for socket connections
            'retries': 3,                 # Number of retries for downloads
            'fragment_retries': 3,        # Number of retries for fragments
            'skip_unavailable_fragments': True,  # Skip unavailable fragments
            'extractor_retries': 3,       # Number of retries for extractors
            'noprogress': True,           # Don't show progress bar
            'ignoreerrors': True,         # Continue on errors
        }
        
        # Import necessary modules for timeout handling
        import threading
        import traceback
        
        # Use a thread with timeout to prevent hanging
        def download_with_timeout():
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([submission.url])
                return True
            except Exception as e:
                print(f"  Error in download thread: {e}")
                return False
        
        # Start download in a separate thread
        download_thread = threading.Thread(target=download_with_timeout)
        download_thread.daemon = True
        download_thread.start()
        
        # Wait for download to complete with timeout (60 seconds)
        download_thread.join(timeout=60)
        
        if download_thread.is_alive():
            print("  Download timed out after 60 seconds, skipping this video")
            return None, 0, 0, 0
        
        # Check if file was downloaded successfully
        if not video_path.exists() or video_path.stat().st_size < 10240:  # Less than 10KB
            print("  Download failed or produced invalid file")
            return None, 0, 0, 0
        
        # Track the downloaded file for cleanup
        temp_files.append(video_path)
        
        # Get video properties
        clip = VideoFileClip(str(video_path))
        duration = clip.duration
        width, height = clip.size
        clip.close()  # Close clip to free resources
        
        return video_path, duration, width, height
        
    except Exception as e:
        print(f"Error preparing initial video: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0, 0
