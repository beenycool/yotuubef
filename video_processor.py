from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, CompositeAudioClip, ColorClip, ImageClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips
import moviepy.video.fx.all as vfx
from moviepy.video.fx import speedx
import math
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
import os
import tempfile
import yt_dlp
from pathlib import Path
import subprocess
import gc
import logging
import psutil
import pathlib
from PIL import Image
import shutil

# Constants
TARGET_RESOLUTION = (1080, 1920)  # Width, Height for vertical video
TEMP_DIR = Path(tempfile.gettempdir()) / "video_processing"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Check for PyTorch CUDA availability
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        torch.cuda.set_device(0)  # Use the first GPU
        print(f"PyTorch CUDA is available: {torch.cuda.get_device_name(0)}")
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    print("PyTorch not available")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil module not available. Memory usage checks will be disabled.")
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

# First check if PyTorch already detected CUDA
if CUDA_AVAILABLE:
    GPU_AVAILABLE = True
    FFMPEG_GPU_PARAMS = {
        'codec': 'h264_nvenc',
        'preset': 'fast',
        'threads': 0
    }
    print("Using NVIDIA GPU detected via PyTorch CUDA")
else:
    # Try NVIDIA first
    try:
        subprocess.check_output(['nvidia-smi'])
        GPU_AVAILABLE = True
        FFMPEG_GPU_PARAMS = {
            'codec': 'h264_nvenc',
            'preset': 'fast',
            'threads': 0
        }
        print("Using NVIDIA GPU detected via nvidia-smi")
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
            print("Using AMD GPU detected via rocminfo")
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
                print("Using Intel GPU detected via vainfo")
            except:
                GPU_AVAILABLE = False
                print("No GPU detected, using CPU encoding")

# Memory management settings
MAX_MEMORY_USAGE = 0.8  # Max 80% of available memory
VIDEO_CHUNK_SIZE = 30  # Process in 30-second chunks for large files

# Focus point can be a single point or a list of points with timestamps
FocusPoint = Dict[str, float]  # {"x": 0.7, "y": 0.3}
TimestampedFocusPoint = Dict[str, Union[float, FocusPoint]]  # {"time": 1.5, "point": {"x": 0.7, "y": 0.3}}

def check_memory_usage():
    """Check if memory usage is within safe limits"""
    if not PSUTIL_AVAILABLE:
        # If psutil is not available, always return True to bypass memory check
        return True
        
    try:
        mem = psutil.virtual_memory()
        return mem.percent < (MAX_MEMORY_USAGE * 100)
    except Exception as e:
        print(f"Error checking memory usage: {e}. Bypassing check.")
        return True

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

def create_short_clip(video_path: str, output_path: str, start_time: float, end_time: float, focus_points: List[Dict]) -> bool:
    """Creates a short clip from the video with proper formatting."""
    try:
        # Force garbage collection before starting
        gc.collect()
        
        # Calculate memory requirements
        file_size = os.path.getsize(video_path)
        system_memory = psutil.virtual_memory()
        required_memory = file_size * 3  # Rough estimate for processing overhead
        
        if required_memory > system_memory.available:
            logging.error(f"Insufficient memory available. Need ~{required_memory/(1024*1024*1024):.1f}GB, have {system_memory.available/(1024*1024*1024):.1f}GB free")
            return False
            
        with VideoFileClip(video_path) as video:
            # Extract the segment
            video = video.subclip(start_time, end_time)
            
            # Get dimensions for vertical format
            target_width = 1080  # Width for vertical video
            target_height = 1920  # Height for vertical video
            
            # Calculate crop dimensions
            source_ar = video.w / video.h
            target_ar = target_width / target_height
            
            if source_ar > target_ar:  # Source is wider
                crop_width = int(video.h * target_ar)
                crop_height = video.h
            else:  # Source is taller
                crop_width = video.w
                crop_height = int(video.w / target_ar)
            
            # Center crop by default
            x_center = video.w / 2
            y_center = video.h / 2
            
            # Apply focus points if available
            if focus_points:
                # Get the focus point for current time
                current_point = focus_points[0]  # Default to first point
                for point in focus_points:
                    if point['time'] <= video.duration:
                        current_point = point
                    else:
                        break
                
                # Extract x,y ratios from point
                x_ratio = current_point['point'].get('x_ratio', 0.5)
                y_ratio = current_point['point'].get('y_ratio', 0.5)
                
                # Calculate center points based on ratios
                x_center = video.w * x_ratio
                y_center = video.h * y_ratio
            
            # Ensure crop region stays within video bounds
            x1 = max(0, min(video.w - crop_width, x_center - crop_width/2))
            y1 = max(0, min(video.h - crop_height, y_center - crop_height/2))
            
            # Crop and resize
            video = video.crop(x1=x1, y1=y1, width=crop_width, height=crop_height)
            video = video.resize((target_width, target_height))
            
            # Write the output file
            video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                audio=True,
                threads=4,  # Limit threads to reduce memory usage
                preset='medium',  # Use a faster preset to reduce memory usage
                ffmpeg_params=['-crf', '23']  # Balanced quality/size
            )
            
            # Force cleanup
            video.close()
            gc.collect()
            
            return True
            
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {str(e)}")
        return False

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

def process_video_with_gpu_optimization(input_path, output_path, processing_func, chunk_size=30, watermark_path=None, loudness_lufs=None):
    """
    Process video in chunks with GPU optimization options.
    Adds watermark if provided, and normalizes audio loudness if possible.
    Args:
        input_path: Path to input video
        output_path: Path to output video
        processing_func: Function to apply to each chunk (clip -> processed_clip)
        chunk_size: Size of chunks in seconds
        watermark_path: Path to watermark image (optional)
        loudness_lufs: Target LUFS for normalization (optional)
    """
    print(f"GPU optimization for {input_path} (simulated).")
    try:
        clip = VideoFileClip(str(input_path))
        processed_clip = processing_func(clip) # Apply the main processing
        clips_to_composite = [processed_clip]
        
        # --- Watermark logic ---
        if watermark_path and os.path.isfile(str(watermark_path)):
            try:
                watermark_clip = (
                    ImageClip(str(watermark_path))
                    .set_duration(processed_clip.duration)
                    .resize(height=int(processed_clip.h * 0.05))
                    .margin(right=10, bottom=10, opacity=0)
                    .set_pos(("right", "bottom"))
                )
                clips_to_composite.append(watermark_clip)
                print("Added watermark.")
            except Exception as e_wm:
                print(f"Could not add watermark: {e_wm}")
                
        # Composite if watermark was added
        if len(clips_to_composite > 1):
            final_video = CompositeVideoClip(clips_to_composite, size=processed_clip.size)
        else:
            final_video = processed_clip
            
        # --- Write video (single-pass, high quality) ---
        current_ffmpeg_params = FFMPEG_GPU_PARAMS if GPU_AVAILABLE else {'threads': 4, 'preset': 'medium'}
        final_video.write_videofile(
            str(output_path),
            codec=current_ffmpeg_params.get('codec', 'libx264'),
            preset=current_ffmpeg_params.get('preset', 'medium'),
            threads=current_ffmpeg_params.get('threads', 4),
            audio_codec="aac",
            logger='bar'
        )
        
        # --- Resource cleanup before normalization ---
        final_video.close()
        if processed_clip is not final_video:
            processed_clip.close()
        if 'clip' in locals() and clip:
            clip.close()
        gc.collect()
        
        # --- Loudness normalization ---
        if loudness_lufs is not None:
            AUDIO_BITRATE = '192k'  # Default audio bitrate
            
            def check_ffmpeg_install(tool_name):
                try:
                    subprocess.run([tool_name, "-version"], capture_output=True, check=True)
                    return True
                except Exception:
                    return False
                    
            if check_ffmpeg_install("ffmpeg-normalize"):
                normalized_path_str = str(pathlib.Path(output_path).with_suffix(".normalized.mp4"))
                cmd_normalize = [
                    "ffmpeg-normalize", str(output_path),
                    "-o", normalized_path_str,
                    "-ar", "48000",
                    "-c:a", "aac",
                    "-b:a", AUDIO_BITRATE,
                    "-l", str(loudness_lufs),
                    "-f"
                ]
                try:
                    print(f"Normalizing audio for {output_path} to {loudness_lufs} LUFS...")
                    subprocess.run(cmd_normalize, check=True, capture_output=True)
                    shutil.move(normalized_path_str, str(output_path))
                    print(f"Audio normalized successfully: {output_path}")
                except subprocess.CalledProcessError as e_norm:
                    print(f"ffmpeg-normalize failed: {e_norm.stderr.decode() if e_norm.stderr else e_norm}")
                except Exception as e_norm_mv:
                    print(f"Error moving normalized file: {e_norm_mv}")
            else:
                print("ffmpeg-normalize not found. Skipping loudness normalization.")
                
    except Exception as e:
        print(f"Error in process_video_with_gpu_optimization: {e}")
        raise
        
    finally:
        # Ensure all clips are closed to release resources
        if 'clip' in locals():
            try: clip.close()
            except: pass
        if 'processed_clip' in locals() and hasattr(processed_clip, 'close'):
            try: processed_clip.close()
            except: pass
        if 'final_video' in locals() and hasattr(final_video, 'close'):
            try: final_video.close()
            except: pass
        gc.collect()


def mute_sections(clip, mute_ranges):
    """
    Mutes specific time ranges of a video clip.
    mute_ranges should be a list of [start, end] times.
    """
    if not mute_ranges:
        return clip

    # Create a new audio array with muted sections
    audio = clip.audio
    # Build a list of (start, end, volume) tuples
    intervals = []
    last_end = 0
    for start, end in sorted(mute_ranges):
        if start > last_end:
            intervals.append((last_end, start, 1.0))  # normal volume
        intervals.append((start, end, 0.0))  # muted
        last_end = end
    if last_end < clip.duration:
        intervals.append((last_end, clip.duration, 1.0))

    # Create subclips for each interval and set volume
    audio_segments = [audio.subclip(start, end).volumex(vol) for start, end, vol in intervals if end > start]
    from moviepy.audio.AudioClip import concatenate_audioclips
    new_audio = concatenate_audioclips(audio_segments)
    return clip.set_audio(new_audio)


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

def create_seamless_loop(clip, crossfade_duration=0.5):
    """
    Create a seamless looping video by adding crossfade transitions between end and beginning.
    
    Args:
        clip: The video clip to make seamlessly looping
        crossfade_duration: Duration of the crossfade effect in seconds
    
    Returns:
        A video clip optimized for seamless looping
    """
    try:
        if clip.duration <= crossfade_duration * 2:
            print(f"Warning: Video too short ({clip.duration}s) for crossfade ({crossfade_duration}s). Returning original.")
            return clip
        
        # Get the end segment that will crossfade with the beginning
        end_segment = clip.subclip(clip.duration - crossfade_duration, clip.duration)
        beginning_segment = clip.subclip(0, crossfade_duration)
        
        # Create crossfade effect
        # Fade out the end segment
        end_fadeout = end_segment.crossfadeout(crossfade_duration)
        # Fade in the beginning segment and composite it over the end
        beginning_fadein = beginning_segment.crossfadein(crossfade_duration)
        
        # Composite the crossfade at the end
        crossfade_composite = CompositeVideoClip([end_fadeout, beginning_fadein.set_position('center')])
        
        # Main body of the video (excluding the crossfade regions)
        main_body = clip.subclip(crossfade_duration, clip.duration - crossfade_duration)
        
        # Reconstruct the video: beginning -> main body -> crossfaded end
        seamless_clip = concatenate_videoclips([
            beginning_segment,  # Clean beginning
            main_body,          # Main content
            crossfade_composite # Crossfaded end that blends to beginning
        ])
        
        # Apply seamless audio looping if audio exists
        if clip.audio:
            seamless_audio = create_seamless_audio_loop(clip.audio, crossfade_duration)
            seamless_clip = seamless_clip.set_audio(seamless_audio)
        
        print(f"Created seamless loop with {crossfade_duration}s crossfade")
        return seamless_clip
        
    except Exception as e:
        print(f"Error creating seamless loop: {e}")
        return clip

def analyze_loop_compatibility(clip, sample_duration=0.5):
    """
    Analyze how well a video's beginning and end match for seamless looping.
    
    Args:
        clip: The video clip to analyze
        sample_duration: Duration to sample from start/end for comparison
    
    Returns:
        A compatibility score between 0 and 1 (1 being perfect for looping)
    """
    try:
        if clip.duration <= sample_duration * 2:
            return 0.5  # Neutral score for very short videos
        
        # Get frames from beginning and end
        start_frame = clip.get_frame(sample_duration / 2)
        end_frame = clip.get_frame(clip.duration - sample_duration / 2)
        
        # Calculate visual similarity (simple approach using frame difference)
        if start_frame.shape == end_frame.shape:
            # Convert to grayscale for simpler comparison
            start_gray = np.mean(start_frame, axis=2) if len(start_frame.shape) == 3 else start_frame
            end_gray = np.mean(end_frame, axis=2) if len(end_frame.shape) == 3 else end_frame
            
            # Calculate normalized difference
            diff = np.mean(np.abs(start_gray - end_gray))
            max_possible_diff = 255  # Max pixel value difference
            similarity = 1 - (diff / max_possible_diff)
            
            return max(0, min(1, similarity))
        else:
            return 0.5  # Neutral if frame shapes don't match
            
    except Exception as e:
        print(f"Error analyzing loop compatibility: {e}")
        return 0.5

def optimize_for_looping(clip, target_duration=None, crossfade_duration=0.5, extend_mode='middle_repeat', trim_from_center=True):
    """
    Optimize a video clip for seamless looping by adjusting timing and adding transitions.
    
    Args:
        clip: The video clip to optimize
        target_duration: Optional target duration for the loop
        crossfade_duration: Duration of crossfade transitions
        extend_mode: Method for extending video ('middle_repeat', 'slow_motion', 'none')
        trim_from_center: Whether to trim from center or start when shortening
    
    Returns:
        An optimized video clip for seamless looping
    """
    try:
        # Analyze current loop compatibility
        compatibility = analyze_loop_compatibility(clip)
        print(f"Loop compatibility score: {compatibility:.2f}")
        
        # If target duration is specified, trim/extend accordingly
        if target_duration and target_duration != clip.duration:
            if target_duration < clip.duration:
                # Trim video to target duration
                if trim_from_center:
                    # Trim from center, preserving important start/end for looping
                    trim_start = (clip.duration - target_duration) / 2
                    clip = clip.subclip(trim_start, trim_start + target_duration)
                    print(f"Trimmed video from center to {target_duration}s for optimal looping")
                else:
                    # Trim from end, preserving start
                    clip = clip.subclip(0, target_duration)
                    print(f"Trimmed video from end to {target_duration}s for optimal looping")
                    
            elif target_duration > clip.duration and extend_mode != 'none':
                extension_needed = target_duration - clip.duration
                
                if extend_mode == 'middle_repeat' and extension_needed <= clip.duration:
                    # Take from the middle section to avoid artifacts
                    middle_start = clip.duration * 0.3
                    middle_duration = min(extension_needed, clip.duration * 0.4)
                    middle_section = clip.subclip(middle_start, middle_start + middle_duration)
                    
                    # Insert the middle section to extend duration
                    clip = concatenate_videoclips([
                        clip.subclip(0, clip.duration * 0.7),
                        middle_section,
                        clip.subclip(clip.duration * 0.7, clip.duration)
                    ])
                    print(f"Extended video to {target_duration}s using middle section repeat")
                    
                elif extend_mode == 'slow_motion':
                    # Apply slow motion to reach target duration
                    speed_factor = clip.duration / target_duration
                    clip = clip.fx(speedx, speed_factor)
                    print(f"Extended video to {target_duration}s using slow motion (factor: {speed_factor:.2f})")
                    
                else:
                    print(f"Cannot extend video: mode '{extend_mode}' not supported or extension too large")
        
        # Apply seamless looping
        looped_clip = create_seamless_loop(clip, crossfade_duration)
        
        return looped_clip
        
    except Exception as e:
        print(f"Error optimizing for looping: {e}")
        return clip

def create_seamless_audio_loop(audio_clip, crossfade_duration=0.5):
    """
    Create seamless audio looping with crossfades to avoid audio pops/clicks.
    
    Args:
        audio_clip: The audio clip to make seamlessly looping
        crossfade_duration: Duration of the audio crossfade in seconds
    
    Returns:
        An audio clip optimized for seamless looping
    """
    try:
        if not audio_clip or audio_clip.duration <= crossfade_duration * 2:
            return audio_clip
        
        # Get end and beginning segments for crossfading
        end_segment = audio_clip.subclip(audio_clip.duration - crossfade_duration, audio_clip.duration)
        beginning_segment = audio_clip.subclip(0, crossfade_duration)
        
        # Create crossfade: fade out end, fade in beginning
        end_fadeout = end_segment.audio_fadeout(crossfade_duration)
        beginning_fadein = beginning_segment.audio_fadein(crossfade_duration)
        
        # Composite the crossfaded audio
        crossfade_audio = CompositeAudioClip([end_fadeout, beginning_fadein])
        
        # Main body without crossfade regions
        main_audio = audio_clip.subclip(crossfade_duration, audio_clip.duration - crossfade_duration)
        
        # Reconstruct: beginning -> main -> crossfaded end
        seamless_audio = concatenate_audioclips([
            beginning_segment,
            main_audio,
            crossfade_audio
        ])
        
        print(f"Created seamless audio loop with {crossfade_duration}s crossfade")
        return seamless_audio
        
    except Exception as e:
        print(f"Error creating seamless audio loop: {e}")
        return audio_clip

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
