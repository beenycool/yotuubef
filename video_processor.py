from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx
import math # Import math for ceiling function
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
import os
import tempfile
import yt_dlp
from pathlib import Path
import subprocess

# Check for GPU availability
try:
    subprocess.check_output(['nvidia-smi'])
    GPU_AVAILABLE = True
    FFMPEG_GPU_PARAMS = {
        'codec': 'h264_nvenc',
        'preset': 'fast',
        'pix_fmt': 'yuv420p',
        'threads': 0
    }
except:
    GPU_AVAILABLE = False
    FFMPEG_GPU_PARAMS = {
        'codec': 'libx264',
        'preset': 'medium',
        'pix_fmt': 'yuv420p',
        'threads': 4
    }

# Focus point can be a single point or a list of points with timestamps
FocusPoint = Dict[str, float]  # {"x": 0.7, "y": 0.3}
TimestampedFocusPoint = Dict[str, Union[float, FocusPoint]]  # {"time": 1.5, "point": {"x": 0.7, "y": 0.3}}

def create_short_clip(video_path, output_path, start_time, end_time, focus_points):
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


        # --- Optional: Resize to a standard short resolution (e.g., 1080x1920) ---
        # Choose the target resolution
        target_height = 1920
        target_width = 1080 # 9:16 aspect ratio

        # Check if resizing is necessary (if cropped dimensions don't match target)
        if cropped_clip.size != [target_width, target_height]:
             # Resize while maintaining aspect ratio (should already be 9:16)
             # Using height=target_height should automatically calculate correct width for 9:16
            final_clip = cropped_clip.resize(height=target_height)
        else:
            final_clip = cropped_clip
        # --- End Optional Resize ---


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
    new_width = int(new_width)
    new_height = int(new_height)
    
    # Calculate zoom progression
    zoom_factors = np.linspace(initial_zoom, target_zoom, len(focus_points))
    max_zoom = max(initial_zoom, target_zoom)
    
    # Calculate center point after clamping
    crop_center_x = x1 + new_width / 2
    crop_center_y = y1 + new_height / 2
    
    # Apply the crop
    return vfx.crop(clip, width=new_width, height=new_height, 
                   x_center=crop_center_x, y_center=crop_center_y)

def apply_dynamic_panning(clip, focus_points, new_width, new_height, original_width, original_height, initial_zoom=1.0, target_zoom=1.1):
    """
    Apply dynamic panning between multiple focus points.
    
    Args:
        clip: The video clip to crop
        focus_points: List of timestamped focus points
        new_width, new_height: Dimensions of the crop window
        original_width, original_height: Original video dimensions
    """
    # Convert to integers for cropping
    new_width = int(new_width)
    new_height = int(new_height)
    
    # Calculate zoom progression
    zoom_factors = np.linspace(initial_zoom, target_zoom, len(focus_points))
    max_zoom = max(initial_zoom, target_zoom)
    
    def get_crop_center_at_time(t):
        # Calculate current zoom factor
        current_zoom = np.interp(t, 
                               [fp['time'] for fp in focus_points],
                               zoom_factors)
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
                
            # Apply smooth easing to the interpolation factor (optional)
            # This makes the panning more natural with acceleration/deceleration
            # Cubic easing for smoother transitions
            factor = factor ** 2 * (3 - 2 * factor)
            
            # Interpolate between the two points
            prev_point = prev_data["point"]
            next_point = next_data["point"]
            
            interp_x = prev_point["x"] + factor * (next_point["x"] - prev_point["x"])
            interp_y = prev_point["y"] + factor * (next_point["y"] - prev_point["y"])
            
            center_x = original_width * interp_x
            center_y = original_height * interp_y
            
            # Adjust for zoom
            effective_width = new_width / current_zoom
            effective_height = new_height / current_zoom
        
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
        
        # Calculate actual crop dimensions
        crop_width = int(new_width / current_zoom)
        crop_height = int(new_height / current_zoom)
        
        # Crop the frame
        cropped_frame = frame[y1:y1+new_height, x1:x1+new_width]
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
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([submission.url])
        
        # Track the downloaded file for cleanup
        temp_files.append(video_path)
        
        # Get video properties
        clip = VideoFileClip(str(video_path))
        duration = clip.duration
        width, height = clip.size
        
        return video_path, duration, width, height
        
    except Exception as e:
        print(f"Error preparing initial video: {e}")
        traceback.print_exc()
        return None, 0, 0, 0
