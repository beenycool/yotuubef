import os
import random
import argparse
import praw
import yt_dlp
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient import errors
from google_auth_oauthlib.flow import InstalledAppFlow
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from elevenlabs import save
import os
from elevenlabs.core import ApiError
import numpy as np
imagemagick_path = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
if os.path.exists(imagemagick_path):
    os.environ["IMAGEMAGICK_BINARY"] = imagemagick_path
    print(f"Set IMAGEMAGICK_BINARY to: {imagemagick_path}")
else:
    # Check if it's already set in env vars
    if not os.environ.get("IMAGEMAGICK_BINARY"):
        print(f"Warning: ImageMagick path not found: {imagemagick_path} and IMAGEMAGICK_BINARY environment variable not set. MoviePy text features might fail.")
    else:
        print(f"Using IMAGEMAGICK_BINARY from environment: {os.environ.get('IMAGEMAGICK_BINARY')}")

from moviepy.editor import (VideoFileClip, AudioFileClip, TextClip, VideoClip, # Added VideoClip
                            CompositeVideoClip, CompositeAudioClip, concatenate_audioclips, ImageClip, concatenate_videoclips,
                            ColorClip)
from moviepy.video.fx.all import crop, colorx, lum_contrast
from joblib import Parallel, delayed
from PIL import Image
import sqlite3
import json
import time
import math
import cv2
import numpy as np
import re
import pathlib

def ensure_frame_shape(frame, target_shape):
    """
    Ensures the frame has the exact target_shape (h, w, c).
    Pads, crops, or resizes as needed, but prefers minimal resizing.
    """
    h, w, c = target_shape
    fh, fw = frame.shape[:2]
    fc = frame.shape[2] if frame.ndim == 3 else 1
    # Add channel if missing
    if frame.ndim == 2:
        frame = np.expand_dims(frame, axis=2)
    if fc != c:
        if fc == 1 and c == 3:
            frame = np.repeat(frame, 3, axis=2)
        else:
            frame = frame[:, :, :c]
    # Pad or crop height
    if fh < h:
        pad = h - fh
        top = pad // 2
        bottom = pad - top
        frame = cv2.copyMakeBorder(frame, top, bottom, 0, 0, cv2.BORDER_REFLECT)
    elif fh > h:
        start = (fh - h) // 2
        frame = frame[start:start + h, :, :]
    # Pad or crop width
    fh, fw = frame.shape[:2]
    if fw < w:
        pad = w - fw
        left = pad // 2
        right = pad - left
        frame = cv2.copyMakeBorder(frame, 0, 0, left, right, cv2.BORDER_REFLECT)
    elif fw > w:
        start = (fw - w) // 2
        frame = frame[:, start:start + w, :]
    # Final check: resize if still not matching
    if frame.shape[:2] != (h, w):
        frame = cv2.resize(frame, (w, h))
    if frame.shape[2] != c:
        frame = frame[:, :, :c]
    return frame
import base64
import prawcore
import traceback
import sys
import shutil
import subprocess

DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

imagemagick_path = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
if os.path.exists(imagemagick_path):
    os.environ["IMAGEMAGICK_BINARY"] = imagemagick_path
    print(f"Set IMAGEMAGICK_BINARY to: {imagemagick_path}")
else:
    # Check if it's already set in env vars
    if not os.environ.get("IMAGEMAGICK_BINARY"):
        print(f"Warning: ImageMagick path not found: {imagemagick_path} and IMAGEMAGICK_BINARY environment variable not set. MoviePy text features might fail.")
    else:
        print(f"Using IMAGEMAGICK_BINARY from environment: {os.environ.get('IMAGEMAGICK_BINARY')}")

# Helper function to ensure a frame is RGBA
def frame_to_rgba(frame):
    if frame.ndim == 2: # Grayscale
        frame = np.expand_dims(frame, axis=2) # Add channel dim
    
    if frame.shape[2] == 1: # Single channel (like grayscale after expand_dims)
        frame = np.repeat(frame, 3, axis=2) # Convert to RGB

    if frame.shape[2] == 3:
        # Add alpha channel if it's RGB
        alpha_channel = np.full(frame.shape[:2], 255, dtype=frame.dtype)
        return np.dstack((frame, alpha_channel))
    elif frame.shape[2] == 4:
        # Already RGBA
        return frame
    else:
        # Fallback for unexpected channel counts, try to force RGB then add Alpha
        print(f"Warning: Unexpected frame shape {frame.shape} in frame_to_rgba. Attempting conversion.")
        rgb_frame = frame[:, :, :3] # Take first 3 channels
        alpha_channel = np.full(rgb_frame.shape[:2], 255, dtype=frame.dtype)
        return np.dstack((rgb_frame, alpha_channel))


# Helper function to ensure a frame is RGBA (moved from later in the script)
def ensure_rgba(frame):
    """Safely convert any frame format to RGBA (4 channels with alpha)"""
    # Handle different input cases
    if frame is None:
        return None
        
    # Convert grayscale to RGB
    if frame.ndim == 2:
        # Grayscale to RGB (3 identical channels)
        frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
    
    # Get frame shape info
    h, w = frame.shape[:2]
    channels = frame.shape[2] if frame.ndim == 3 else 1
    
    # Already RGBA
    if channels == 4:
        return frame
    
    # RGB to RGBA (add alpha channel)
    if channels == 3:
        # Create fully opaque alpha channel (255)
        alpha = np.full((h, w, 1), 255, dtype=frame.dtype)
        # Return RGBA
        return np.dstack((frame, alpha))
        
    # Handle odd number of channels (convert to RGB first, then add alpha)
    if channels != 3 and channels != 4:
        print(f"Warning: Unexpected channel count: {channels}. Converting to RGBA.")
        if channels == 1:
            # Convert single channel to RGB
            rgb = np.repeat(frame, 3, axis=2)
        else:
            # Take first 3 channels for RGB
            rgb = frame[:, :, :3]
        # Add alpha
        alpha = np.full((h, w, 1), 255, dtype=frame.dtype)
        return np.dstack((rgb, alpha))
    
    # Should never reach here
    return frame


def add_visual_emphasis(clip, focus_x=0.5, focus_y=0.5, start_time=0, duration=1.5, zoom_factor=1.1):
    w, h = clip.size

    def zoom_at(t):
        if start_time <= t <= start_time + duration:
            progress = (t - start_time) / duration
            factor = 1 + (zoom_factor - 1) * math.sin(progress * math.pi)
            return factor
        return 1

    zoomed = clip.resize(lambda t: zoom_at(t))

    def make_circle_frame(t):
        frame = np.zeros((h, w, 4), dtype=np.uint8)
        if start_time <= t <= start_time + duration:
            progress = (t - start_time) / duration
            pulse = 0.5 + 0.5 * math.sin(progress * math.pi * 4)
            radius = int(min(w, h) * 0.1)
            thickness = 4
            color = (255, 0, 0, int(255 * pulse * 0.5))
            center = (int(focus_x * w), int(focus_y * h))
            cv2.circle(frame, center, radius, color, thickness, lineType=cv2.LINE_AA)
        return frame

    circle_overlay = (VideoClip(make_circle_frame, duration=clip.duration)
                      .set_position(('center', 'center')))

    def brightness_factor(t):
        if start_time <= t <= start_time + duration:
            progress = (t - start_time) / duration
            return 1 + 0.2 * math.sin(progress * math.pi)
        return 1

    bright_zoomed = zoomed.fl(lambda gf, t: gf(t) * brightness_factor(t))

    # Ensure the base clip is RGBA before compositing with the RGBA overlay
    bright_zoomed_rgba = bright_zoomed.fl_image(frame_to_rgba)

    emphasized = CompositeVideoClip([bright_zoomed_rgba, circle_overlay.set_opacity(0.5)])
    return emphasized.set_duration(clip.duration)

def apply_subtle_zoom(clip, zoom_range=(1.0, 1.05)):
    z_start, z_end = zoom_range
    if random.choice([True, False]):
        z_start, z_end = z_end, z_start
    def zoom(t):
        return z_start + (z_end - z_start) * (t / clip.duration)
    return clip.resize(lambda t: zoom(t))

def parallel_process_frames(clip, frame_func, n_jobs=4):
    """
    Utility to process frames in parallel using joblib.
    Returns a new VideoFileClip with processed frames.
    """
    fps = clip.fps
    frames = list(clip.iter_frames())
    processed = Parallel(n_jobs=n_jobs)(delayed(frame_func)(f, i / fps) for i, f in enumerate(frames))
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    if processed:
        first_shape = processed[0].shape
        # Enforce all frames have the same shape as the first
        processed = [ensure_frame_shape(f, first_shape) if f.shape != first_shape else f for f in processed]
    return ImageSequenceClip(processed, fps=fps).set_duration(clip.duration).set_audio(clip.audio)

def apply_shake(clip, max_translate=5, shake_duration=0.3, shake_times=3, parallel=True, n_jobs=4):
    if clip.duration is None or clip.duration <= shake_duration:
        return clip
    shake_moments = sorted(random.uniform(0, max(0, clip.duration - shake_duration)) for _ in range(shake_times))
    def shake_transform(frame, t):
        offset_x, offset_y = 0, 0
        for moment in shake_moments:
            if moment <= t <= moment + shake_duration:
                dx = random.uniform(-max_translate, max_translate)
                dy = random.uniform(-max_translate, max_translate)
                offset_x += dx
                offset_y += dy
        h, w = frame.shape[:2]
        if abs(offset_x) > 0 or abs(offset_y) > 0:
            M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
            try:
                warped = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                wh, ww = warped.shape[:2]
                # Pad if too small (centered)
                pad_h = max(0, h - wh)
                pad_w = max(0, w - ww)
                if pad_h > 0 or pad_w > 0:
                    top = pad_h // 2
                    bottom = pad_h - top
                    left = pad_w // 2
                    right = pad_w - left
                    warped = cv2.copyMakeBorder(warped, top, bottom, left, right, cv2.BORDER_REFLECT)
                # Crop if too large (centered)
                wh, ww = warped.shape[:2]
                if wh > h or ww > w:
                    start_y = (wh - h) // 2
                    start_x = (ww - w) // 2
                    warped = warped[start_y:start_y + h, start_x:start_x + w]
                # Final check: ensure shape matches
                # Always enforce shape (h, w, c)
                return ensure_frame_shape(warped, (h, w, frame.shape[2]))
            except Exception as e:
                print(f"Warning: cv2.warpAffine failed during shake: {e}")
                return frame
        return frame
    if parallel:
        print(f"Applying shake in parallel using {n_jobs} jobs...")
        return parallel_process_frames(clip, shake_transform, n_jobs=n_jobs)
    else:
        # Fallback to original serial method
        def shake_transform_serial(get_frame, t):
            return shake_transform(get_frame(t), t)
        return clip.fl(shake_transform_serial, apply_to='video')


def apply_color_grade(clip):
    preset = random.choice(['none', 'cool', 'warm', 'vibrant'])
    if preset == 'cool':
        color = random.uniform(0.92, 0.98)
        contrast = random.uniform(0.03, 0.08)
        clip = clip.fx(colorx, color).fx(lum_contrast, contrast=contrast, lum=0)
    elif preset == 'warm':
        color = random.uniform(1.02, 1.08)
        contrast = random.uniform(0.03, 0.08)
        clip = clip.fx(colorx, color).fx(lum_contrast, contrast=contrast, lum=0)
    elif preset == 'vibrant':
        color = random.uniform(1.08, 1.15)
        contrast = random.uniform(0.12, 0.18)
        clip = clip.fx(colorx, color).fx(lum_contrast, contrast=contrast, lum=0)
    return clip


def is_ffmpeg_installed():
    try:
        return shutil.which("ffmpeg") is not None
    except Exception:
        return False

def is_ffmpeg_normalize_installed():
    try:
        return shutil.which("ffmpeg-normalize") is not None
    except Exception:
        return False

class UploadLimitExceededError(Exception):
    pass

REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.environ.get('REDDIT_USER_AGENT', f'python:VideoBot:v1.2 (by /u/YOUR_USERNAME_HERE)')

if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
    raise ValueError("Reddit API credentials (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET) not found in environment variables.")
if REDDIT_USER_AGENT == 'python:VideoBot:v1.2 (by /u/YOUR_USERNAME_HERE)':
    print("Warning: Default REDDIT_USER_AGENT detected. Please set a unique User-Agent in your environment variables.")


SCOPES = ['https://www.googleapis.com/auth/youtube.upload', 'https://www.googleapis.com/auth/youtube.force-ssl']
CLIENT_SECRETS_FILE = os.environ.get('GOOGLE_CLIENT_SECRETS_FILE')
if not CLIENT_SECRETS_FILE:
    raise ValueError("Google Client Secrets file path (GOOGLE_CLIENT_SECRETS_FILE) not found in environment variables.")
if not os.path.exists(CLIENT_SECRETS_FILE):
     raise FileNotFoundError(f"Google Client Secrets file not found at specified path: {CLIENT_SECRETS_FILE}. Check the GOOGLE_CLIENT_SECRETS_FILE environment variable.")

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_ID = 'gemini-2.0-flash'

ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')
if not ELEVENLABS_API_KEY:
     raise ValueError("ELEVENLABS_API_KEY environment variable not set.")

API_DELAY = 6
DB_FILE = 'uploaded_videos.db'
conn = None
c = None

OVERLAY_FONT = 'Impact'
OVERLAY_FONT_SIZE_RATIO = 1 / 10
OVERLAY_TEXT_COLOR = 'white'
OVERLAY_STROKE_COLOR = 'black'
OVERLAY_STROKE_WIDTH = 3
OVERLAY_POSITION = ('center', 'center')
OVERLAY_BG_COLOR = 'transparent'

ADD_VISUAL_EMPHASIS = True

TEMP_DIR = "./temp_processing"
pathlib.Path(TEMP_DIR).mkdir(exist_ok=True)
MUSIC_FOLDER = "./music" # Changed from ./musci
WATERMARK_PATH = "./watermark.png"
FORBIDDEN_WORDS = [ "fuck", "fucking", "fucked", "fuckin", "shit", "shitty", "shitting", "damn", "damned", "bitch", "bitching", "cunt", "asshole", "piss", "pissed" ]

if not is_ffmpeg_installed():
    print("\n" + "="*40)
    print("CRITICAL WARNING: FFmpeg is not installed or not found in PATH.")
    print("Video and audio processing will likely fail.")
    print("Please install FFmpeg and ensure it's in your system's PATH.")
    print("="*40 + "\n")
else:
    print("FFmpeg check successful.")

try:
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    print("ElevenLabs client initialized.")
except Exception as e:
    print(f"Error initializing ElevenLabs client: {e}")
    raise

def setup_database():
    global conn, c
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS uploads
                     (reddit_url TEXT PRIMARY KEY, youtube_url TEXT)''')
        conn.commit()
        print("Database setup complete.")
    except sqlite3.Error as e:
        print(f"Database Error: {e}")
        raise

def close_database():
    global conn
    if conn:
        try:
            conn.close()
            print("Database connection closed.")
        except sqlite3.Error as e:
            print(f"Error closing database: {e}")
        finally:
            conn = None

def get_reddit_instance():
    try:
        reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                           client_secret=REDDIT_CLIENT_SECRET,
                           user_agent=REDDIT_USER_AGENT,
                           check_for_async=False)
        print("Testing Reddit authentication...")
        me = reddit.user.me()
        print(f"Reddit instance created successfully. Authenticated as: {me}")
        return reddit
    except prawcore.exceptions.OAuthException as e:
        print(f"Reddit Authentication Error: {e}. Check your credentials in environment variables.")
        raise
    except prawcore.exceptions.ResponseException as e:
         print(f"Reddit API Response Error: {e.response.status_code} - {e}")
         raise
    except Exception as e:
        print(f"Error creating Reddit instance: {e}")
        traceback.print_exc()
        raise

def get_youtube_service():
    try:
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server(port=8090)
        print("YouTube authentication successful.")
        return build('youtube', 'v3', credentials=credentials)
    except FileNotFoundError:
        print(f"Error: YouTube client secrets file not found at {CLIENT_SECRETS_FILE}")
        raise
    except Exception as e:
        print(f"Error creating YouTube service: {e}")
        traceback.print_exc()
        raise

def cleanup_temp_files(*args):
    cleaned_count = 0
    for file_path in args:
        if file_path and isinstance(file_path, (str, pathlib.Path)):
             path_obj = pathlib.Path(file_path)
             if path_obj.is_file():
                 try:
                     print(f"  - Removing temp file: {path_obj.name}")
                     path_obj.unlink()
                     cleaned_count += 1
                 except OSError as e: print(f"  - Error removing {path_obj.name}: {e}")
    if cleaned_count > 0:
        print(f"Cleanup complete. Removed {cleaned_count} temporary files.")


def combine_video_audio_subs(original_video_path, processed_video_stream_path, tts_audio_path, subtitle_clips, output_path, mix_audio=False,
                             stabilize=False, transforms_file=None, skip_enhancements=False, color_factor=1.05, contrast=0.1, luminance=3,
                             speech_segments=None, gemini_data=None):
    original_clip_for_audio = None
    original_audio = None
    processed_video_clip = None
    tts_clip = None
    tts_audio = None
    bg_music_clip = None
    bg_music_audio = None
    final_video = None
    final_audio = None
    temp_normalized_tts_path = None
    temp_bg_music_path = None # Temp path for potentially processed bg music

    print(f"\n--- Combining Video Elements ---")
    print(f"  Video Stream: {os.path.basename(processed_video_stream_path)}")
    print(f"  Original Audio Ref: {os.path.basename(original_video_path) if original_video_path else 'None'}")
    print(f"  TTS Audio: {os.path.basename(tts_audio_path) if tts_audio_path else 'None'}")
    print(f"  Subtitles: {len(subtitle_clips)}")
    print(f"  Output: {os.path.basename(output_path)}")

    try:
        # 1. Load Processed Video Stream
        print(f"Loading processed video stream...")
        if not os.path.exists(processed_video_stream_path): raise FileNotFoundError(f"Processed video stream not found: {processed_video_stream_path}")
        processed_video_clip = VideoFileClip(processed_video_stream_path)
        if not processed_video_clip: raise ValueError(f"Failed to load processed video: {processed_video_stream_path}")
        video_duration = processed_video_clip.duration
        video_w, video_h = processed_video_clip.size
        print(f"  Video loaded: {video_duration:.2f}s, {video_w}x{video_h}")

        # Apply visual enhancements
        if not skip_enhancements:
            mood = ""
            if gemini_data: mood = gemini_data.get("mood", "").lower()

            if "energetic" in mood or "action" in mood:
                current_color_factor = random.uniform(1.12, 1.18)
                current_contrast = random.uniform(0.13, 0.18)
            elif "sad" in mood or "calm" in mood:
                current_color_factor = random.uniform(0.92, 0.98)
                current_contrast = random.uniform(0.03, 0.08)
            else:
                current_color_factor = random.uniform(color_factor - 0.03, color_factor + 0.03)
                current_contrast = random.uniform(contrast - 0.03, contrast + 0.03)

            print(f"Applying enhancements: colorx({current_color_factor:.2f}), lum_contrast(contrast={current_contrast:.2f}, lum={luminance})")
            processed_video_clip = colorx(processed_video_clip, current_color_factor)
            processed_video_clip = lum_contrast(processed_video_clip, contrast=current_contrast, lum=luminance)

            # Randomize subtle zoom, shake, and color grade parameters
            zoom_start = random.uniform(1.01, 1.04)
            zoom_end = random.uniform(1.04, 1.08)
            processed_video_clip = apply_subtle_zoom(processed_video_clip, zoom_range=(zoom_start, zoom_end))

            max_translate = random.uniform(3, 7)
            shake_duration = random.uniform(0.2, 0.4)
            shake_times = random.randint(2, 5)
            processed_video_clip = apply_shake(processed_video_clip, max_translate=max_translate, shake_duration=shake_duration, shake_times=shake_times)

            processed_video_clip = apply_color_grade(processed_video_clip)
        else:
             print("Skipping visual enhancements.")

        # Add visual emphasis
        if ADD_VISUAL_EMPHASIS and gemini_data:
             try:
                 key_moments_data = gemini_data.get("key_visual_moments", []) # Use the right key
                 if key_moments_data:
                     print(f"Adding emphasis based on {len(key_moments_data)} key moments.")
                     for moment in key_moments_data:
                         t = moment.get('timestamp')
                         fp = moment.get('focus_point')
                         # Ensure timestamp and focus point are valid
                         if t is not None and isinstance(fp, dict) and 'x' in fp and 'y' in fp:
                             try:
                                 t_float = float(t)
                                 fx = float(fp['x'])
                                 fy = float(fp['y'])
                                 # Only apply if within duration
                                 if 0 <= t_float < video_duration:
                                     emphasis_duration = 1.5 # Standard duration
                                     emphasis_start = max(0, t_float - emphasis_duration / 2)
                                     print(f"  - Emphasis at {emphasis_start:.2f}s (moment at {t_float:.2f}s), focus ({fx:.2f}, {fy:.2f})")
                                     processed_video_clip = add_visual_emphasis(
                                         processed_video_clip, focus_x=fx, focus_y=fy,
                                         start_time=emphasis_start, duration=emphasis_duration, zoom_factor=1.1
                                     )
                                 else: print(f"  - Skipping emphasis for moment at {t_float:.2f}s (out of bounds)")
                             except (ValueError, TypeError) as parse_err:
                                 print(f"  - Warning: Skipping emphasis due to invalid data format: {moment}, Error: {parse_err}")
                         else: print(f"  - Warning: Skipping emphasis due to missing data: {moment}")
                 else: print("No key visual moments found in Gemini data for emphasis.")
             except Exception as e:
                 print(f"Warning: Failed to add visual emphasis: {e}")
                 traceback.print_exc()
        else:
             print("Skipping visual emphasis (disabled or no Gemini data).")

        print(f"Processed video duration after effects: {processed_video_clip.duration:.2f}s")

        # 2. Load Original Audio (if requested or needed)
        # This audio should correspond to the processed_video_stream timeframe
        # We assume the `original_video_path` passed IS the correct source (e.g., already trimmed if needed)
        if mix_audio and original_video_path and os.path.exists(original_video_path):
            print(f"Loading original audio from: {os.path.basename(original_video_path)}")
            try:
                # Try loading as VideoFileClip first to get audio
                original_clip_for_audio = VideoFileClip(original_video_path)
                original_audio = original_clip_for_audio.audio
                if original_audio:
                    print(f"  Original audio duration: {original_audio.duration:.2f}s")
                    # Trim/pad original audio to match video duration *exactly* if mixing
                    original_audio = original_audio.set_duration(video_duration)
                    print(f"  Original audio adjusted to video duration: {original_audio.duration:.2f}s")
                else:
                    print("  Warning: Original video has no audio track.")
                # Close the source clip now that we have the audio object
                original_clip_for_audio.close()
                original_clip_for_audio = None
            except Exception as e_orig_aud:
                print(f"  Warning: Could not load original audio from {original_video_path}: {e_orig_aud}")
                original_audio = None # Ensure it's None if loading failed
        elif mix_audio:
             print("Warning: Mix audio requested but original_video_path not valid.")


        # 3. Normalize and Load TTS Audio
        if tts_audio_path and os.path.exists(tts_audio_path):
            print("Preparing TTS audio...")
            tts_base_name = os.path.basename(tts_audio_path)
            temp_normalized_tts_path = os.path.join(TEMP_DIR, os.path.splitext(tts_base_name)[0] + "_norm.aac")

            print(f"Normalizing TTS audio: {tts_base_name} -> {os.path.basename(temp_normalized_tts_path)}")
            norm_success = normalize_loudness(tts_audio_path, temp_normalized_tts_path, target_lufs=-16)

            if norm_success and os.path.exists(temp_normalized_tts_path):
                print(f"Loading normalized TTS audio clip...")
                try:
                    tts_clip = AudioFileClip(temp_normalized_tts_path)
                    if tts_clip:
                        tts_audio = tts_clip
                        print(f"  Loaded normalized TTS audio duration: {tts_audio.duration:.2f}s")
                        # Ensure TTS audio matches video duration (pad with silence if needed)
                        tts_audio = tts_audio.set_duration(video_duration)
                        print(f"  TTS audio adjusted to video duration: {tts_audio.duration:.2f}s")
                    else: print(f"  Warning: Failed to load normalized TTS audio clip.")
                except Exception as e_load_tts:
                    print(f"  Warning: Failed to load normalized TTS audio: {e_load_tts}")
                    if tts_clip: tts_clip.close(); tts_clip = None
                    tts_audio = None # Ensure it's None
            else:
                print(f"  Warning: Normalized TTS file not found or normalization failed. Skipping TTS.")
                cleanup_temp_files(temp_normalized_tts_path)
                temp_normalized_tts_path = None
        else:
            print("No TTS audio path provided or file not found.")


        # 4. Add Background Music
        if os.path.isdir(MUSIC_FOLDER):
            bg_music_files = [f for f in os.listdir(MUSIC_FOLDER) if f.lower().endswith((".mp3", ".wav", ".aac", ".m4a"))]
            if bg_music_files:
                bg_music_path = os.path.join(MUSIC_FOLDER, random.choice(bg_music_files))
                print(f"Attempting to add background music: {os.path.basename(bg_music_path)}")
                try:
                    bg_music_clip = AudioFileClip(bg_music_path)
                    bg_music_audio = bg_music_clip

                    print(f"  Music duration: {bg_music_audio.duration:.2f}s")

                    # Loop or trim background music
                    if bg_music_audio.duration < video_duration:
                        loops = math.ceil(video_duration / bg_music_audio.duration)
                        print(f"  Looping background music {loops} times.")
                        bg_music_audio = concatenate_audioclips([bg_music_audio] * loops)

                    bg_music_audio = bg_music_audio.subclip(0, video_duration)
                    print(f"  Background music adjusted to video duration: {bg_music_audio.duration:.2f}s")

                    # --- Dynamic ducking based on speech segments ---
                    if speech_segments:
                        print(f"  Applying dynamic audio ducking based on {len(speech_segments)} speech segments.")
                        def ducking_volume(t):
                            base_vol = 0.15 # Lower base volume for BG music
                            duck_vol = 0.04 # Volume during speech
                            fade = 0.15     # Fade time

                            for seg in speech_segments:
                                try:
                                    start = float(seg.get('start_time', 0))
                                    end = float(seg.get('end_time', 0))
                                except (ValueError, TypeError): continue # Skip invalid segment

                                if start - fade <= t <= end + fade:
                                    if start - fade <= t < start: # Fade down
                                        progress = max(0, min(1, (t - (start - fade)) / fade))
                                        return base_vol - (base_vol - duck_vol) * progress
                                    elif end < t <= end + fade: # Fade up
                                        progress = max(0, min(1, (t - end) / fade))
                                        return duck_vol + (base_vol - duck_vol) * progress
                                    else: # During speech (start <= t <= end)
                                        return duck_vol
                            # If not in any speech segment or its fade buffer
                            return base_vol

                        if isinstance(bg_music_audio, AudioFileClip): # Changed AudioClip to AudioFileClip
                            bg_music_audio = bg_music_audio.volumex(ducking_volume)
                        else:
                            print("  Warning: bg_music_audio is not an AudioFileClip, skipping ducking.")
                        print("  Applied ducking volume effect to background music.")
                    else:
                        print("  No speech segments provided, applying constant low volume (0.15) to background music.")
                        bg_music_audio = bg_music_audio.volumex(0.15)

                except Exception as e:
                    print(f"  Warning: Failed to load or process background music: {e}")
                    traceback.print_exc()
                    if bg_music_clip: bg_music_clip.close() # Close clip if loaded but failed processing
                    bg_music_clip = None
                    bg_music_audio = None
            else: print(f"No compatible music files found in {MUSIC_FOLDER}.")
        else: print(f"Warning: Music folder '{MUSIC_FOLDER}' not found. Skipping background music.")


        # 5. Choose and Combine Final Audio Tracks
        audio_tracks = []
        is_original_usable = original_audio is not None and original_audio.duration > 0.1
        is_tts_usable = tts_audio is not None and tts_audio.duration > 0.1
        is_bg_music_usable = bg_music_audio is not None and bg_music_audio.duration > 0.1

        if mix_audio and is_original_usable and is_tts_usable:
            print("Mixing original (80% vol) and TTS audio.")
            original_audio = original_audio.volumex(0.8) # Adjust volume before adding
            audio_tracks.append(original_audio)
            audio_tracks.append(tts_audio)
        elif is_tts_usable:
             print("Using TTS audio as primary.")
             audio_tracks.append(tts_audio)
        elif is_original_usable:
            print("Using original audio as primary (TTS unusable or not mixed).")
            audio_tracks.append(original_audio)
        else:
            print("Warning: No primary audio (Original or TTS) is usable.")

        if is_bg_music_usable:
            print("Adding background music to the mix.")
            audio_tracks.append(bg_music_audio)
        else: print("No background music added.")

        if len(audio_tracks) > 1:
            print(f"Compositing {len(audio_tracks)} audio tracks.")
            valid_tracks = [track for track in audio_tracks if hasattr(track, 'duration')]
            if len(valid_tracks) > 1:
                final_audio = CompositeAudioClip(valid_tracks)
            elif len(valid_tracks) == 1:
                final_audio = valid_tracks[0]
            else: final_audio = None
        elif len(audio_tracks) == 1:
            print("Using single audio track.")
            final_audio = audio_tracks[0]
        else:
            print("Warning: No audio tracks available for the final video.")
            final_audio = None

        # Ensure final audio duration matches video duration *exactly*
        if final_audio:
            # Removed explicit set_duration for final_audio to avoid OSError with shorter clips
            # CompositeAudioClip should handle durations correctly when set on the video.
            # if abs(final_audio.duration - video_duration) > 0.05: # Allow tiny discrepancy
            #      print(f"  Adjusting final audio duration ({final_audio.duration:.2f}s) to match video ({video_duration:.2f}s)")
            #      final_audio = final_audio.set_duration(video_duration)
            print(f"Final audio duration determined by composition: {final_audio.duration:.2f}s (Video duration: {video_duration:.2f}s)")
        else: print("Proceeding without audio.")


        # 6. Set final audio onto the processed video clip
        print("Setting final audio onto video clip...")
        final_video_with_audio = processed_video_clip.set_audio(final_audio)


        # 7. Apply overlays (watermark)
        overlays = []
        if os.path.exists(WATERMARK_PATH):
            try:
                # Convert watermark to RGBA immediately when loading
                watermark = (ImageClip(WATERMARK_PATH)
                             .set_duration(final_video_with_audio.duration)
                             .resize(height=max(20, int(final_video_with_audio.h * 0.04))) # Smaller watermark
                             .margin(right=15, bottom=15, opacity=0)
                             .set_pos(("right", "bottom"))
                             .set_opacity(0.5)) # More subtle
                # Ensure watermark is RGBA
                watermark = watermark.fl_image(ensure_rgba)
                overlays.append(watermark)
                print("Added watermark overlay.")
            except Exception as e:
                 print(f"Warning: Failed to create watermark overlay: {e}")
        else:
            print(f"Warning: Watermark file not found at {WATERMARK_PATH}, skipping.")

        # Combine video with overlays
        if overlays:
            # Ensure the base clip is RGBA before adding RGBA overlays (like watermark)
            print("Converting main clip to RGBA before adding overlays...")
            # This is the key fix - ensure consistent RGBA format
            final_video_with_audio_rgba = final_video_with_audio.fl_image(ensure_rgba)
            
            print("Compositing RGBA base clip with overlays...")
            main_with_overlays = CompositeVideoClip([final_video_with_audio_rgba] + overlays, size=final_video_with_audio.size)
            print("Overlays composited.")
        else:
            main_with_overlays = final_video_with_audio # No overlays, keep original format


        # 8. Create final composite with subtitles
        print(f"Adding {len(subtitle_clips)} subtitle clips...")
        if len(subtitle_clips) > 0:
            print(f"Ensuring consistent RGBA format for all clips before compositing...")
            # Convert main clip to RGBA if it isn't already
            if not hasattr(main_with_overlays, 'is_rgba') or not main_with_overlays.is_rgba:
                main_with_overlays_rgba = main_with_overlays.fl_image(ensure_rgba)
            else:
                main_with_overlays_rgba = main_with_overlays
            
            # Ensure all subtitle clips are RGBA
            compatible_subtitles = []
            for sub in subtitle_clips:
                try:
                    sub_rgba = sub.fl_image(ensure_rgba)
                    compatible_subtitles.append(sub_rgba)
                except Exception as e:
                    print(f"Warning: Error converting subtitle to RGBA: {e}. Skipping this subtitle.")
            
            # Composite with consistent RGBA format
            final_video = CompositeVideoClip([main_with_overlays_rgba] + compatible_subtitles)
            print(f"Composited main clip with {len(compatible_subtitles)} subtitle clips.")
        else:
            final_video = main_with_overlays
            print("No subtitles to composite.")


        # 9. Trim final video to max 60 seconds
        if final_video.duration > 60.5: # Allow slightly over 60 before trimming
            print(f"Trimming final video from {final_video.duration:.2f}s to 60s.")
            final_video = final_video.subclip(0, 60)
        elif final_video.duration <= 0:
             raise ValueError(f"Final video has invalid duration: {final_video.duration:.2f}s")


        # 10. Write Final Video
        print(f"Writing final combined video (Duration: {final_video.duration:.2f}s)...")
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path): cleanup_temp_files(output_path)

        # --- GPU Acceleration Check ---
        # Note: has_nvidia_gpu() is defined globally earlier in the script
        gpu_available = has_nvidia_gpu()

        if gpu_available:
            print("GPU detected: Using h264_nvenc for final encoding.")
            write_params = {
                'codec': 'h264_nvenc', # Use NVENC codec
                'audio_codec': 'aac',
                'temp_audiofile': os.path.join(TEMP_DIR, 'final-temp-audio.m4a'),
                'remove_temp': True,
                # 'preset': 'medium', # NVENC uses different presets
                'fps': 30,
                # 'threads': os.cpu_count() or 4, # Not typically needed/used by NVENC
                'logger': 'bar',
                'write_logfile': False
            }
            ffmpeg_extra_params = [
                # '-crf', '23', # CRF is for libx264
                '-cq', '23',      # Use Constant Quality for NVENC (similar target)
                '-preset', 'p5',  # Use NVENC preset (p1=fastest, p7=slowest/best quality, p5=slow)
                '-pix_fmt', 'yuv420p', # Still needed for compatibility
                '-movflags', '+faststart'
            ]
        else:
            print("No GPU detected or nvidia-smi failed: Using libx264 (CPU) for final encoding.")
            write_params = {
                'codec': 'libx264',
                'audio_codec': 'aac',
                'temp_audiofile': os.path.join(TEMP_DIR, 'final-temp-audio.m4a'),
                'remove_temp': True,
                'preset': 'medium', # Keep CPU preset
                'fps': 30,
                'threads': os.cpu_count() or 4, # Keep CPU threads
                'logger': 'bar',
                'write_logfile': False
            }
            ffmpeg_extra_params = [
                '-crf', '23', # Keep CPU CRF
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart'
            ]

        # Add stabilization filter (common logic for both CPU/GPU)
        if stabilize and transforms_file and os.path.exists(transforms_file):
             print(f"Adding stabilization filter using {os.path.basename(transforms_file)}")
             escaped_transforms_path = str(transforms_file).replace('\\', '/')
             # Check if -vf already exists in params (unlikely here, but safer)
             vf_index = -1
             try: vf_index = ffmpeg_extra_params.index('-vf')
             except ValueError: pass

             stabilization_filter = f'vidstabtransform=input="{escaped_transforms_path}":zoom=0:smoothing=10,unsharp=5:5:0.8:3:3:0.4'

             if vf_index != -1:
                 # Append to existing filtergraph (safer if other filters were added)
                 ffmpeg_extra_params[vf_index + 1] += f',{stabilization_filter}'
                 print(f"  Appended stabilization to existing -vf filter.")
             else:
                 # Add new filtergraph
                 ffmpeg_extra_params.extend(['-vf', stabilization_filter])
                 print(f"  Added new -vf filter for stabilization.")
        elif stabilize:
             print("Warning: Stabilization requested but transforms file missing or invalid. Skipping stabilization.")

        write_params['ffmpeg_params'] = ffmpeg_extra_params

        # Before writing, convert final video to RGB format (important fix)
        print("Converting final video to RGB format for encoding...")
        final_video = final_video.fl_image(lambda frame: frame[:,:,:3] if frame.shape[2] == 4 else frame)

        # Ensure final video dimensions are even (required by libx264)
        w, h = final_video.size
        even_w, even_h = w - (w % 2), h - (h % 2)

        if (w != even_w) or (h != even_h):
            print(f"Adjusting final video size from ({w}x{h}) to ({even_w}x{even_h}) for libx264 compatibility.")
            final_video = final_video.resize(newsize=(even_w, even_h))

        final_video.write_videofile(output_path, **write_params)

        print(f"Final video written. Verifying: {output_path}")
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            # Final audio check
            if final_audio and not has_audio_track(output_path):
                 print(f"CRITICAL ERROR: Final combined video {output_path} is missing audio track, but audio was expected!")
                 cleanup_temp_files(output_path) # Clean up failed file
                 return None # Signal failure
            elif not final_audio:
                 print("Final video created without audio (as expected).")
                 return output_path
            else:
                 print(f"Audio track verified for final combined video.")
                 return output_path
        else:
            print(f"CRITICAL ERROR: Final combined video file is missing or empty: {output_path}")
            cleanup_temp_files(output_path)
            return None

    except Exception as e:
        print(f"Error combining video elements: {e}")
        traceback.print_exc()
        cleanup_temp_files(output_path) # Clean up potentially failed output file
        return None
    finally:
        # Ensure all clips are closed
        print("Closing MoviePy clips...")
        clips_to_close = [
            original_clip_for_audio, processed_video_clip, tts_clip,
            bg_music_clip, final_video, final_audio, original_audio
            ]
        # Also close subtitle clips if they are moviepy objects
        if subtitle_clips:
             clips_to_close.extend([sub for sub in subtitle_clips if hasattr(sub, 'close')])

        for i, clip_obj in enumerate(clips_to_close):
            if clip_obj and hasattr(clip_obj, 'close'):
                try:
                    # print(f"Closing clip {i}...") # Verbose
                    clip_obj.close()
                except Exception as e_close:
                    print(f"Warning: Error closing clip object {i}: {e_close}")

        # Clean up temporary normalized TTS file
        cleanup_temp_files(temp_normalized_tts_path)

def normalize_loudness(input_path, output_path, target_lufs=-14):
    """
    Normalize audio loudness using ffmpeg-normalize if available
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file for normalization not found: {input_path}")
        return False

    if not is_ffmpeg_normalize_installed():
        print("Warning: ffmpeg-normalize not found. Copying input to output without loudness normalization.")
        try:
            shutil.copy(input_path, output_path)
            return True # Indicate copy success
        except Exception as e:
            print(f"Error copying file during normalize fallback: {e}")
            return False # Indicate failure

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path): cleanup_temp_files(output_path)

    cmd = [
        "ffmpeg-normalize", str(input_path),
        "-o", str(output_path),
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-f",
        "-nt", "ebu",
        "-t", str(target_lufs),
        "--keep-loudness-range-target" # Try to preserve dynamic range
    ]
    print(f"Running ffmpeg-normalize: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Loudness normalized to {target_lufs} LUFS: {output_path}")
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            return True
        else:
            print("Error: ffmpeg-normalize ran but output file is missing or empty.")
            cleanup_temp_files(output_path)
            return False # Treat as failure if output isn't valid
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg-normalize failed (exit code {e.returncode}): {e.stderr}")
        print("Attempting to copy input to output as fallback.")
        try:
            shutil.copy(input_path, output_path)
            return True # Indicate copy success even if normalization failed
        except Exception as copy_e:
            print(f"Error copying file during normalize fallback: {copy_e}")
            return False
    except Exception as e:
        print(f"Unexpected error during normalization: {e}")
        return False


def has_nvidia_gpu():
    """Check if an NVIDIA GPU is available for encoding"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def analyze_video_with_gemini(video_path, title, subreddit_name):
    """
    Use Gemini to analyze video content and get structured data for a more engaging output.
    Returns a dictionary with analysis data.
    """
    if not os.path.exists(video_path):
        print(f"Video file not found for Gemini analysis: {video_path}")
        return None

    # Extract frames for analysis (at 1fps to keep sample size reasonable)
    temp_frames_dir = os.path.join(TEMP_DIR, "gemini_frames")
    os.makedirs(temp_frames_dir, exist_ok=True)
    
    try:
        print("Extracting frames for Gemini analysis...")
        clip = VideoFileClip(video_path)
        video_duration = clip.duration
        
        # Extract one frame per second (max 20 frames)
        frame_count = min(20, int(video_duration))
        frame_interval = max(1, video_duration / frame_count)
        sample_frames = []
        
        for i in range(frame_count):
            t = i * frame_interval
            if t >= video_duration:
                break
                
            frame_path = os.path.join(temp_frames_dir, f"frame_{i:03d}.jpg")
            try:
                frame = clip.get_frame(t)
                # Convert to RGB if frame has alpha
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                
                # Convert numpy array to PIL Image and save
                Image.fromarray((frame * 255).astype(np.uint8)).save(frame_path, quality=85)
                sample_frames.append({
                    'path': frame_path,
                    'time': t
                })
            except Exception as frame_err:
                print(f"Error extracting frame at {t}s: {frame_err}")
                continue
        
        clip.close()
        
        if not sample_frames:
            print("Failed to extract any frames for analysis")
            return None
            
        print(f"Extracted {len(sample_frames)} frames for analysis")
        
        # Prepare prompt for Gemini
        prompt = f"""
        Analyze this video content from Reddit: r/{subreddit_name}
        Original title: "{title}"
        
        I'm showing you {len(sample_frames)} frames from this video.
        Provide a structured analysis in JSON format with the following information:
        
        1. A brief summary of what's happening in the video (summary_for_description)
        2. A suggested title for YouTube Shorts that's catchy but accurate, under 100 chars (suggested_title)
        3. The overall mood of the video (mood) - choose one: [funny, serious, informative, emotional, action, calm, exciting, sad]
        4. Best segment to feature (if you can determine): provide start_time and end_time in seconds (best_segment)
        5. Key visual moments: list timestamps and focus points (x,y coordinates from 0-1) (key_visual_moments)
        6. Speech segments for subtitles (if applicable): list of segments with start_time, end_time and text (speech_segments)
        7. Relevant hashtags for social media (hashtags) - 3 to 5 hashtags
        
        Return in clean JSON format only, no explanations.
        """
        
        # Convert frames to base64 for Gemini input
        image_parts = []
        for frame in sample_frames[:10]:  # Limit to 10 frames to avoid token limits
            with open(frame['path'], 'rb') as f:
                image_data = f.read()
                image_parts.append({
                    'mime_type': 'image/jpeg',
                    'data': base64.b64encode(image_data).decode('utf-8')
                })
        
        print("Sending frames to Gemini for analysis...")
        try:
            # Set up Gemini model (using placeholder for compatibility)
            gemini_model = genai.GenerativeModel(GEMINI_MODEL_ID)
            
            # Process in batches if there are many frames
            if len(image_parts) > 5:
                print("Processing frames in batches...")
                batch_size = 5
                batch_results = []
                
                for i in range(0, len(image_parts), batch_size):
                    batch = image_parts[i:i+batch_size]
                    batch_prompt = f"{prompt}\n(Batch {i//batch_size + 1} of {(len(image_parts) + batch_size - 1)//batch_size})"
                    
                    response = gemini_model.generate_content(
                        [batch_prompt, *[{'image': img} for img in batch]]
                    )
                    
                    batch_results.append(response.text)
                
                # Combine batch results
                combined_text = "\n".join(batch_results)
                # Extract JSON data from combined text
                json_match = re.search(r'\{.*\}', combined_text, re.DOTALL)
                if json_match:
                    result_text = json_match.group(0)
                else:
                    result_text = combined_text
            else:
                # Process all frames at once if few
                content_parts = [prompt]
                for img in image_parts:
                    content_parts.append({'image': img})
                
                response = gemini_model.generate_content(content_parts)
                result_text = response.text
                
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            else:
                json_text = result_text
                
            # Parse JSON data
            try:
                analysis_data = json.loads(json_text)
                print("Gemini analysis successful!")
                return analysis_data
            except json.JSONDecodeError:
                print("Failed to parse Gemini response as JSON")
                print(f"Response text: {result_text[:500]}...")
                return None
                
        except Exception as gemini_err:
            print(f"Gemini API error: {gemini_err}")
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"Error during video analysis with Gemini: {e}")
        traceback.print_exc()
        return None
    finally:
        # Clean up temp frames
        try:
            if os.path.exists(temp_frames_dir):
                shutil.rmtree(temp_frames_dir)
                print("Cleaned up temporary frame directory")
        except Exception as cleanup_err:
            print(f"Error cleaning up temp frames: {cleanup_err}")


def create_animated_subtitle_clips(speech_segments, video_width, video_height, font_size=None):
    """
    Creates animated subtitle clips for each speech segment
    Returns a list of TextClip objects with timing and animations
    """
    if not speech_segments:
        return []
        
    print(f"Creating {len(speech_segments)} subtitle clips...")
    
    # Calculate font size based on video height if not specified
    if font_size is None:
        font_size = max(24, int(video_height * OVERLAY_FONT_SIZE_RATIO))
        print(f"Using auto-calculated font size: {font_size}")
    
    subtitle_clips = []
    
    for segment in speech_segments:
        try:
            text = segment.get('text', '')
            start_time = float(segment.get('start_time', 0))
            end_time = float(segment.get('end_time', 0))
            
            if not text or start_time >= end_time:
                continue  # Skip invalid segments
                
            # Truncate long text (max 2 lines)
            if len(text) > 80:
                text = text[:77] + "..."
                
            # Create a TextClip
            txt_clip = (TextClip(text, fontsize=font_size, font=OVERLAY_FONT, color=OVERLAY_TEXT_COLOR,
                                 bg_color=OVERLAY_BG_COLOR, stroke_color=OVERLAY_STROKE_COLOR,
                                 stroke_width=OVERLAY_STROKE_WIDTH, align='center',
                                 method='caption', size=(int(video_width * 0.9), None))
                         .set_start(start_time)
                         .set_end(end_time)
                         .set_position(OVERLAY_POSITION))
            
            # Add fade in/out animation
            fade_time = min(0.3, (end_time - start_time) * 0.2)  # 20% of duration or max 0.3s
            txt_clip = txt_clip.fadein(fade_time).fadeout(fade_time)
            
            subtitle_clips.append(txt_clip)
            
        except Exception as e:
            print(f"Error creating subtitle clip for segment: {e}")
            traceback.print_exc()
    
    return subtitle_clips


def generate_tts(text, output_path, voice_id=DEFAULT_VOICE_ID):
    """
    Generate TTS audio from text using ElevenLabs API
    Returns True on success, False on failure
    """
    if not text or not text.strip():
        print("No text provided for TTS generation")
        return False
        
    print(f"Generating TTS for text ({len(text)} chars)...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        cleanup_temp_files(output_path)
    
    try:
        # Send request to ElevenLabs API
        print(f"Using voice ID: {voice_id}")
        
        # Add a slight pause at beginning and end
        processed_text = f"{text}"
        
        tts_response = elevenlabs_client.generate(
            text=processed_text,
            voice=voice_id,
            model_id="eleven_multilingual_v2"
        )
        
        # Save audio file
        save(tts_response, output_path)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            print(f"TTS generated successfully: {output_path}")
            return True
        else:
            print("TTS file is missing or too small")
            return False
    except ApiError as api_err:
        print(f"ElevenLabs API Error: {api_err}")
        return False
    except Exception as e:
        print(f"Error generating TTS: {e}")
        traceback.print_exc()
        return False


def make_video_vertical(input_path, output_path):
    """Convert a horizontal video to vertical format (9:16 aspect ratio)"""
    clip = None
    cropped_clip = None
    try:
        print(f"Attempting to convert {os.path.basename(input_path)} to vertical 9:16 -> {os.path.basename(output_path)}")
        clip = VideoFileClip(input_path)
        original_w, original_h = clip.size
        if original_w == 0 or original_h == 0: raise ValueError("Invalid video dimensions")
        
        target_w, target_h = 1080, 1920  # Standard vertical video resolution
        
        # Calculate new dimensions preserving aspect ratio
        if original_h / original_w >= target_h / target_w:
            # Video is already tall enough or taller than 9:16
            # Crop sides to achieve 9:16
            new_w = int(original_h * target_w / target_h)
            new_h = original_h
            x_center = original_w / 2
            crop_x1 = max(0, int(x_center - new_w / 2))
            crop_x2 = min(original_w, int(x_center + new_w / 2))
            crop_width = crop_x2 - crop_x1
            
            cropped_clip = clip.crop(x1=crop_x1, y1=0, width=crop_width, height=new_h)
            
            # Resize to target size
            final_clip = cropped_clip.resize(height=target_h)
            print(f"Vertical by cropping sides: {crop_width}x{new_h} -> {final_clip.size}")
        else:
            # Video is wider than 9:16 ratio
            # Crop top/bottom to achieve 9:16
            new_h = int(original_w * target_h / target_w)
            new_w = original_w
            y_center = original_h / 2
            crop_y1 = max(0, int(y_center - new_h / 2))
            crop_y2 = min(original_h, int(y_center + new_h / 2))
            crop_height = crop_y2 - crop_y1
            
            cropped_clip = clip.crop(y1=crop_y1, x1=0, width=new_w, height=crop_height)
            
            # Resize to target size
            final_clip = cropped_clip.resize(width=target_w)
            print(f"Vertical by cropping top/bottom: {new_w}x{crop_height} -> {final_clip.size}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write the final clip
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=os.path.join(TEMP_DIR, 'vertical-temp-audio.m4a'),
            remove_temp=True,
            preset='medium',
            threads=4,
            fps=30,
            logger=None
        )
        
        # Verify the output exists and has a reasonable size
        if os.path.exists(output_path) and os.path.getsize(output_path) > 10240:
            print(f"Successfully created vertical video: {os.path.basename(output_path)}")
            return output_path
        else:
            print(f"Error: Vertical video output missing or too small.")
            cleanup_temp_files(output_path)
            return None
    except Exception as e:
        print(f"Error making video vertical: {e}")
        traceback.print_exc()
        cleanup_temp_files(output_path)
        return None
    finally:
        # Close clips to release resources
        if clip:
            try: clip.close()
            except: pass
        if cropped_clip and cropped_clip != clip:
            try: cropped_clip.close() 
            except: pass


def get_video_details(video_path):
    """
    Get basic video details using FFmpeg.
    Returns (duration_in_seconds, width, height)
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return 0, 0, 0

    try:
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'stream=width,height,duration,r_frame_rate',
            '-select_streams', 'v:0',
            '-of', 'json',
            str(video_path)
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFprobe error: {result.stderr}")
            return 0, 0, 0
            
        data = json.loads(result.stdout)
        
        if 'streams' not in data or not data['streams']:
            print("No video streams found in file")
            return 0, 0, 0
            
        stream = data['streams'][0]
        width = int(stream.get('width', 0))
        height = int(stream.get('height', 0))
        
        duration = 0
        if 'duration' in stream:
            duration = float(stream['duration'])
        else:
            # If no duration in stream, try format
            format_cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'json',
                str(video_path)
            ]
            format_result = subprocess.run(format_cmd, capture_output=True, text=True)
            if format_result.returncode == 0:
                format_data = json.loads(format_result.stdout)
                if 'format' in format_data and 'duration' in format_data['format']:
                    duration = float(format_data['format']['duration'])
        
        return duration, width, height
        
    except Exception as e:
        print(f"Error getting video details: {e}")
        traceback.print_exc()
        return 0, 0, 0


def trim_video(input_path, start_time, end_time, output_path):
    """
    Trim video from start_time to end_time
    """
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return None
        
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Trimming video from {start_time:.2f}s to {end_time:.2f}s...")
        
        # Use ffmpeg directly for more consistent trimming
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c:v', 'libx264', 
            '-c:a', 'aac',
            '-preset', 'medium',
            '-crf', '23',
            '-y',
            str(output_path)
        ]
        
        print(f"Running FFmpeg: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return None
            
        if os.path.exists(output_path) and os.path.getsize(output_path) > 10240:
            print(f"Successfully trimmed video: {os.path.basename(output_path)}")
            return output_path
        else:
            print("Trimmed video is missing or too small")
            return None
            
    except Exception as e:
        print(f"Error trimming video: {e}")
        traceback.print_exc()
        return None


def download_media(url, output_path, media_type='best'):
    """
    Download media from URL using yt-dlp
    Returns the path to the downloaded media file
    """
    print(f"Downloading media from: {url}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        ydl_opts = {
            'format': media_type,
            'outtmpl': output_path,
            'quiet': False,
            'no_warnings': False,
            'ignoreerrors': True,
            'noprogress': False,
            'retries': 3,
            'nocheckcertificate': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        if os.path.exists(output_path) and os.path.getsize(output_path) > 10240:
            print(f"Media downloaded successfully: {output_path}")
            return output_path
        else:
            # Check for suffixed files (yt-dlp sometimes adds resolution or format info)
            output_dir = os.path.dirname(output_path)
            base_name = os.path.basename(output_path)
            name, ext = os.path.splitext(base_name)
            
            for file in os.listdir(output_dir):
                if file.startswith(name) and os.path.getsize(os.path.join(output_dir, file)) > 10240:
                    print(f"Found downloaded file with modified name: {file}")
                    return os.path.join(output_dir, file)
                    
            print("Downloaded file not found or too small")
            return None
            
    except Exception as e:
        print(f"Error downloading media: {e}")
        traceback.print_exc()
        return None


def has_audio_track(media_path):
    """Check if a media file has an audio track"""
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', media_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0 and 'audio' in result.stdout.strip()
    except Exception as e:
        print(f"Error checking for audio track: {e}")
        return False

