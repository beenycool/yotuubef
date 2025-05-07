from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, CompositeAudioClip
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
        
        if not timestamped_focus_points: # Should be handled by normalize_focus_points, but as a safeguard
             print("Warning: No valid focus points after normalization. Defaulting to static center crop.")
             cropped_clip = apply_static_crop(clip, {"x": 0.5, "y": 0.5}, new_width, new_height, original_width, original_height)
        elif len(timestamped_focus_points) == 1:
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
            clips_to_composite = [final_clip]
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
                
                clips_to_composite.append(txt_clip)
            
            final_clip = CompositeVideoClip(clips_to_composite)


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
        List of timestamped focus points with relative timestamps, or an empty list if input is invalid.
    """
    # Handle null/empty input
    if not focus_points:
        # Default to center point if no focus points provided
        return [{"time": 0, "point": {"x": 0.5, "y": 0.5}}]

    # If single point provided, convert to list
    if not isinstance(focus_points, list):
        # Ensure the single point is valid before returning
        point_dict = focus_points if isinstance(focus_points, dict) else {}
        x = max(0.0, min(1.0, point_dict.get("x", 0.5)))
        y = max(0.0, min(1.0, point_dict.get("y", 0.5)))
        return [{"time": 0, "point": {"x": x, "y": y}}]

    # Validate and normalize each point in the list
    validated_points = []
    for point_data in focus_points:
        # Ensure point_data is a dictionary
        if not isinstance(point_data, dict):
            print(f"Warning: Skipping malformed focus point data (expected dict): {point_data}")
            continue # Skip this point

        point = point_data.get("point", {"x": 0.5, "y": 0.5})
        # Ensure 'point' itself is a dictionary
        if not isinstance(point, dict):
            print(f"Warning: Malformed 'point' in focus_points (expected dict): {point}. Using default.")
            point = {"x": 0.5, "y": 0.5}

        # Validate coordinates
        x = max(0.0, min(1.0, point.get("x", 0.5)))
        y = max(0.0, min(1.0, point.get("y", 0.5)))
        
        time_in_clip = max(0.0, min(
            point_data.get("time", 0) - start_time,
            clip_duration
        ))

        validated_points.append({
            "time": time_in_clip,
            "point": {"x": x, "y": y}
        })
    
    if not validated_points: # If all points were malformed
        print("Warning: No valid focus points found after validation. Defaulting to center point.")
        return [{"time": 0, "point": {"x": 0.5, "y": 0.5}}]

    # Sort by time
    validated_points.sort(key=lambda p: p["time"])
    
    # Ensure a point at the beginning (time 0)
    if validated_points[0]["time"] > 0:
        first_point_coords = validated_points[0]["point"]
        validated_points.insert(0, {"time": 0, "point": first_point_coords})
    
    # Ensure a point at the end (clip_duration)
    if validated_points[-1]["time"] < clip_duration:
        last_point_coords = validated_points[-1]["point"]
        validated_points.append({"time": clip_duration, "point": last_point_coords})
            
    return validated_points

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
    # effective_width = int(new_width) # Not used directly in vfx.crop if x1,y1,x2,y2 are used
    # effective_height = int(new_height)
    
    # Calculate center point after clamping for vfx.crop if it uses center
    crop_center_x = x1 + new_width / 2
    crop_center_y = y1 + new_height / 2
    
    # Apply the crop
    return vfx.crop(clip, width=int(new_width), height=int(new_height),
                   x_center=int(crop_center_x), y_center=int(crop_center_y))

def apply_dynamic_panning(clip, focus_points, new_width, new_height, original_width, original_height, initial_zoom=1.0, target_zoom=1.1, easing="ease_in_out"):
    """
    Apply dynamic panning between multiple focus points.
    
    Args:
        clip: The video clip to crop
        focus_points: List of timestamped focus points
        new_width, new_height: Dimensions of the crop window
        original_width, original_height: Original video dimensions
    """
    # Validate input parameters
    if not focus_points: # This should ideally be caught by normalize_focus_points returning a default
        print("Error: No focus points provided for dynamic panning. Defaulting to static crop.")
        return apply_static_crop(clip, {"x":0.5, "y":0.5}, new_width, new_height, original_width, original_height)
        
    if len(focus_points) < 2:
        # Fall back to static crop if only one point (normalize_focus_points should ensure at least one)
        print("Warning: Dynamic panning requires at least 2 focus points. Using static crop with the first point.")
        return apply_static_crop(
            clip,
            focus_points[0]["point"], # normalize_focus_points ensures this structure
            new_width,
            new_height,
            original_width,
            original_height
        )

    # Convert to integers for cropping
    effective_width = int(new_width)
    effective_height = int(new_height)
    
    # Using vfx.crop with functions for center position only
    def get_x_center(t):
        # Get the focus point at time t
        next_point_idx = 0
        while next_point_idx < len(focus_points) and focus_points[next_point_idx]["time"] < t:
            next_point_idx += 1
            
        if next_point_idx == 0:
            # Before the first point, use the first point
            point = focus_points[0]["point"]
        elif next_point_idx == len(focus_points) or focus_points[next_point_idx]["time"] == t:
            # At or after the last point, or exact match
            idx = next_point_idx - 1 if next_point_idx > 0 and focus_points[next_point_idx]["time"] > t else next_point_idx
            point = focus_points[idx]["point"]
        else:
            # Interpolate between prev_point and next_point
            prev_point_data = focus_points[next_point_idx - 1]
            next_point_data = focus_points[next_point_idx]
            
            time_diff = next_point_data["time"] - prev_point_data["time"]
            interp_factor = (t - prev_point_data["time"]) / time_diff if time_diff else 0
            
            # Apply easing function if specified
            if easing == "ease_in_out":
                interp_factor = (1 - math.cos(interp_factor * math.pi)) / 2
            
            point = {
                "x": prev_point_data["point"]["x"] + interp_factor * (next_point_data["point"]["x"] - prev_point_data["point"]["x"]),
                "y": prev_point_data["point"]["y"] + interp_factor * (next_point_data["point"]["y"] - prev_point_data["point"]["y"])
            }
        
        # Return the x-coordinate based on the point's relative position
        return int(original_width * point["x"])

    def get_y_center(t):
        # Get the focus point at time t
        next_point_idx = 0
        while next_point_idx < len(focus_points) and focus_points[next_point_idx]["time"] < t:
            next_point_idx += 1
            
        if next_point_idx == 0:
            # Before the first point, use the first point
            point = focus_points[0]["point"]
        elif next_point_idx == len(focus_points) or focus_points[next_point_idx]["time"] == t:
            # At or after the last point, or exact match
            idx = next_point_idx - 1 if next_point_idx > 0 and focus_points[next_point_idx]["time"] > t else next_point_idx
            point = focus_points[idx]["point"]
        else:
            # Interpolate between prev_point and next_point
            prev_point_data = focus_points[next_point_idx - 1]
            next_point_data = focus_points[next_point_idx]
            
            time_diff = next_point_data["time"] - prev_point_data["time"]
            interp_factor = (t - prev_point_data["time"]) / time_diff if time_diff else 0
            
            # Apply easing function if specified
            if easing == "ease_in_out":
                interp_factor = (1 - math.cos(interp_factor * math.pi)) / 2
            
            point = {
                "x": prev_point_data["point"]["x"] + interp_factor * (next_point_data["point"]["x"] - prev_point_data["point"]["x"]),
                "y": prev_point_data["point"]["y"] + interp_factor * (next_point_data["point"]["y"] - prev_point_data["point"]["y"])
            }
        
        # Return the y-coordinate based on the point's relative position
        return int(original_height * point["y"])

    # We'll use a different approach with frame-by-frame transformation
    def apply_frame_transform(get_frame, t):
        frame = get_frame(t)
        
        # Get the center position for this frame
        center_x = get_x_center(t)
        center_y = get_y_center(t)
        
        # Calculate crop width and height (fixed values)
        crop_width = int(new_width / initial_zoom)
        crop_height = int(new_height / initial_zoom)
        
        # Calculate the top-left corner of the crop area
        x1 = max(0, center_x - crop_width // 2)
        y1 = max(0, center_y - crop_height // 2)
        
        # Adjust if the crop goes beyond the frame
        if x1 + crop_width > original_width:
            x1 = original_width - crop_width
        if y1 + crop_height > original_height:
            y1 = original_height - crop_height
        
        # Crop the frame
        x2 = x1 + crop_width
        y2 = y1 + crop_height
        cropped_frame = frame[y1:y2, x1:x2]
        
        return cropped_frame
    
    # Apply the frame transformation
    intermediate_cropped_clip = clip.fl(apply_frame_transform)
    
    # Then resize this cropped view to the final target dimensions
    final_panned_zoomed_clip = intermediate_cropped_clip.resize(
        width=new_width, 
        height=new_height
    )
    
    return final_panned_zoomed_clip


def apply_ken_burns_effect(clip, zoom_direction="in", duration=None, easing="ease_in_out"):
    """
    Apply Ken Burns effect (slow zoom and pan).
    
    Args:
        clip: Video clip to apply effect to
        zoom_direction: "in" or "out"
        duration: Duration of the effect (defaults to clip duration)
        easing: Easing function name
    """
    if duration is None:
        duration = clip.duration
        
    original_width, original_height = clip.size

    # Define easing functions (more can be added)
    def ease_linear(t_norm): return t_norm
    def ease_in_quad(t_norm): return t_norm**2
    def ease_out_quad(t_norm): return t_norm * (2 - t_norm)
    def ease_in_out_quad(t_norm): 
        if t_norm < 0.5: return 2 * t_norm**2
        return -1 + (4 - 2 * t_norm) * t_norm
    
    easing_func_map = {
        "linear": ease_linear,
        "ease_in": ease_in_quad, # Using quad as a generic ease_in
        "ease_out": ease_out_quad, # Using quad as a generic ease_out
        "ease_in_out": ease_in_out_quad,
    }
    selected_easing_func = easing_func_map.get(easing, ease_in_out_quad)

    def make_frame(get_frame, t):
        frame = get_frame(t)
        t_norm = t / duration # Normalized time 0 to 1

        eased_t_norm = selected_easing_func(t_norm)

        if zoom_direction == "in":
            zoom = 1 + (0.2 * eased_t_norm)  # Zoom from 1x to 1.2x
        else: # zoom_direction == "out"
            zoom = 1.2 - (0.2 * eased_t_norm) # Zoom from 1.2x to 1x
        
        new_w = original_width / zoom
        new_h = original_height / zoom

        # Simple pan: move slightly from center or a corner
        # For this example, let's pan from top-left towards center when zooming in
        # And from center towards top-left when zooming out
        
        # Max pan displacement (e.g., 5% of the difference between original and zoomed dim)
        max_pan_x = (original_width - new_w) * 0.1 
        max_pan_y = (original_height - new_h) * 0.1

        if zoom_direction == "in":
            pan_x = max_pan_x * eased_t_norm
            pan_y = max_pan_y * eased_t_norm
        else: # zoom_direction == "out"
            pan_x = max_pan_x * (1 - eased_t_norm)
            pan_y = max_pan_y * (1 - eased_t_norm)
            
        x1 = pan_x
        y1 = pan_y
        x2 = x1 + new_w
        y2 = y1 + new_h

        # Ensure crop is within bounds (should be if pan is small)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(original_width, x2), min(original_height, y2)
        
        cropped_region = frame[int(y1):int(y2), int(x1):int(x2)]
        
        # Resize back to original dimensions
        # Need image resizing library here, e.g. Pillow or OpenCV for robust resize
        # For a simple numpy resize (less quality):
        # This is tricky with numpy directly for arbitrary sizes.
        # Using a library is better. For now, assume we have a resize function.
        # Example with Pillow:
        from PIL import Image
        pil_img = Image.fromarray(cropped_region)
        resized_img = pil_img.resize((original_width, original_height), Image.LANCZOS) # Or Image.Resampling.LANCZOS
        return np.array(resized_img)

    return clip.fl(make_frame, apply_to=['mask']) if clip.mask else clip.fl(make_frame)


def apply_color_grading(clip, preset="cinematic"):
    """
    Apply color grading presets.
    (This is a placeholder - actual color grading is complex)
    """
    if preset == "cinematic":
        # Example: slightly increase contrast and saturation, apply a teal/orange look
        # This requires per-pixel manipulation, often done with LUTs or complex functions
        # For a very basic example (not true color grading):
        # clip = clip.fx(vfx.colorx, factor=1.1) # Increase contrast slightly
        # clip = clip.fx(vfx.lum_contrast, lum=0, contrast=0.1, contrast_thr=127)
        print("Cinematic color grading preset applied (simulated).")
    elif preset == "vibrant":
        # clip = clip.fx(vfx.colorx, factor=1.2) # More saturation
        print("Vibrant color grading preset applied (simulated).")
    # More presets can be added
    return clip

def create_complex_animation(clip, animation_type="slide_in", duration=1.0):
    """
    Create complex text or object animations.
    (Placeholder for more advanced animations)
    """
    if animation_type == "slide_in":
        # Example: Text slides in from left
        # This would typically involve a TextClip and animating its position
        def pos_func(t):
            # Slide from off-screen left to center
            # Assuming clip width is W, text width is TW
            # Start_x = -TW, End_x = (W - TW) / 2
            # This needs clip dimensions and text dimensions.
            return ('left', 'center') # Simplified
        # text_clip = TextClip("Animated Text", ...).set_position(pos_func).set_duration(duration)
        # return CompositeVideoClip([clip, text_clip])
        print(f"Slide-in animation applied over {duration}s (simulated).")
    elif animation_type == "fade_and_scale":
        # Example: Text fades in while scaling up
        print(f"Fade and scale animation applied over {duration}s (simulated).")
    return clip

def process_video_with_gpu_optimization(input_path, output_path, processing_func, chunk_size=30):
    """
    Process video in chunks with GPU optimization options.
    (Placeholder for chunked processing and GPU pipeline)
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        processing_func: Function to apply to each chunk (clip -> processed_clip)
        chunk_size: Size of chunks in seconds
    """
    print(f"GPU optimization for {input_path} (simulated).")
    # Basic passthrough for now, actual chunking is complex
    try:
        clip = VideoFileClip(input_path)
        processed_clip = processing_func(clip) # Apply the main processing
        
        # Determine if ffmpeg_params for GPU should be used
        current_ffmpeg_params = FFMPEG_GPU_PARAMS if GPU_AVAILABLE else {'threads': 4, 'preset': 'medium'}

        processed_clip.write_videofile(output_path, codec=current_ffmpeg_params.get('codec', 'libx264'), 
                                       preset=current_ffmpeg_params.get('preset'),
                                       threads=current_ffmpeg_params.get('threads'),
                                       audio_codec="aac", logger='bar')
    except Exception as e:
        print(f"Error in process_video_with_gpu_optimization: {e}")
        raise
    finally:
        if 'clip' in locals(): clip.close()
        if 'processed_clip' in locals() and hasattr(processed_clip, 'close'): processed_clip.close()


def _prepare_initial_video(submission, safe_title: str, temp_files: list) -> Tuple[Optional[Path], float, int, int]:
    """
    Downloads video from submission, gets basic info.
    Returns (path_to_video, duration, width, height) or (None, 0, 0, 0) on failure.
    """
    video_url = submission.url
    temp_dir = Path(tempfile.gettempdir()) / "reddit_videos"
    temp_dir.mkdir(parents=True, exist_ok=True)
    video_path = temp_dir / f"{submission.id}_{safe_title}.mp4"
    temp_files.append(video_path) # Track for cleanup

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': str(video_path),
        'noplaylist': True,
        'quiet': True,
        'merge_output_format': 'mp4',
        # 'postprocessors': [{ # Normalization can be added here if ffmpeg-normalize is available
        #     'key': 'FFmpegNormalize',
        #     'preferredcodec': 'aac', # Example
        # }],
    }

    duration, width, height = 0, 0, 0
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        
        if video_path.exists():
            # Get video info using ffprobe (part of ffmpeg)
            # ffprobe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
            #                '-show_entries', 'stream=width,height,duration', '-of', 'csv=s=x:p=0', str(video_path)]
            # result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
            # w_str, h_str, dur_str = result.stdout.strip().split('x')
            # width, height, duration = int(w_str), int(h_str), float(dur_str)
            
            # Simpler way with MoviePy after download
            with VideoFileClip(str(video_path)) as clip_info:
                duration = clip_info.duration
                width, height = clip_info.size
            print(f"Initial video: {duration:.2f}s, {width}x{height}")
            return video_path, duration, width, height
        else:
            print(f"Failed to download video: {video_url}")
            return None, 0, 0, 0
            
    except Exception as e:
        print(f"Error downloading or getting info for {video_url}: {e}")
        if video_path.exists(): # Clean up partial download if any
             try: os.remove(video_path)
             except OSError: pass
        return None, 0, 0, 0

# Example usage (can be removed or kept for testing)
if __name__ == '__main__':
    # This section will only run if the script is executed directly
    # Create a dummy video file for testing
    from moviepy.editor import ColorClip
    dummy_video_path = "dummy_video.mp4"
    # ColorClip((1280, 720), color=(255,0,0), duration=10).write_videofile(dummy_video_path, fps=25)

    test_output_path = "test_short_clip.mp4"
    test_start_time = 1.0
    test_end_time = 6.0
    
    # Test 1: Single static focus point (center)
    # test_focus_points_static = {"x": 0.5, "y": 0.5} 
    # print(f"\nTesting static crop with: {test_focus_points_static}")
    # create_short_clip(dummy_video_path, "test_static_center.mp4", test_start_time, test_end_time, test_focus_points_static)

    # Test 2: Dynamic focus points
    test_focus_points_dynamic = [
        {"time": 1.0, "point": {"x": 0.2, "y": 0.2}},
        {"time": 3.0, "point": {"x": 0.8, "y": 0.8}},
        {"time": 5.0, "point": {"x": 0.5, "y": 0.5}}
    ]
    print(f"\nTesting dynamic crop with: {test_focus_points_dynamic}")
    # create_short_clip(dummy_video_path, "test_dynamic_pan.mp4", test_start_time, test_end_time, test_focus_points_dynamic)

    # Test 3: Null focus points (should default to center)
    # print("\nTesting null focus points (should default to center static crop)")
    # create_short_clip(dummy_video_path, "test_null_fp.mp4", test_start_time, test_end_time, None)
    
    # Test 4: Malformed focus points in a list
    # test_focus_points_malformed_list = [
    #     {"time": 1.0, "point": {"x": 0.1, "y": 0.1}},
    #     "not_a_dict", # Malformed entry
    #     {"time": 4.0, "point": "not_a_point_dict"}, # Malformed point
    #     {"time": 5.0, "point": {"x": 0.9, "y": 0.9}}
    # ]
    # print(f"\nTesting malformed list: {test_focus_points_malformed_list}")
    # create_short_clip(dummy_video_path, "test_malformed_list.mp4", test_start_time, test_end_time, test_focus_points_malformed_list)

    # Test 5: Single point in list (should behave like static)
    # test_focus_single_in_list = [{"time": 2.0, "point": {"x": 0.7, "y": 0.3}}]
    # print(f"\nTesting single point in list: {test_focus_single_in_list}")
    # create_short_clip(dummy_video_path, "test_single_in_list.mp4", test_start_time, test_end_time, test_focus_single_in_list)

    # Clean up dummy video
    # if os.path.exists(dummy_video_path):
    #     os.remove(dummy_video_path)
    print("\nVideo processing tests complete. Check output files.")
