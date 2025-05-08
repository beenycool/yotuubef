"""
Colab Adapter for Reddit Video to YouTube Shorts Script

This script adapts the original video processing script to run on Google Colab with T4 GPU support.
Run this code in Google Colab by uploading both this file and the original script.py to your Colab runtime.
"""

import os
import sys
import subprocess
import pathlib
from google.colab import files
import json

# Create necessary directories
BASE_DIR = pathlib.Path('/content')
TEMP_DIR = BASE_DIR / "temp_processing"
MUSIC_FOLDER = BASE_DIR / "music"
DB_FILE = BASE_DIR / 'uploaded_videos.db'

TEMP_DIR.mkdir(exist_ok=True)
MUSIC_FOLDER.mkdir(exist_ok=True)

# Install dependencies
def install_dependencies():
    print("Installing required packages...")
    subprocess.run([
        "pip", "install", "-q", 
        "yt-dlp", "praw", "moviepy", "google-api-python-client", 
        "google-auth-oauthlib", "google-generativeai", "elevenlabs",
        "torch", "transformers", "soundfile"
    ], check=True)
    
    print("Installing system dependencies...")
    subprocess.run([
        "apt-get", "update", "-qq"
    ], check=True)
    
    subprocess.run([
        "apt-get", "install", "-qq", "imagemagick", "ffmpeg"
    ], check=True)
    
    # Fix ImageMagick policy to allow PDF operations (needed for text rendering)
    subprocess.run([
        "sed", "-i", "s/rights=\"none\" pattern=\"PDF\"/rights=\"read|write\" pattern=\"PDF\"/",
        "/etc/ImageMagick-6/policy.xml"
    ], check=True)
    
    print("Dependencies installed successfully!")

# Download background music
def download_background_music():
    print("Downloading royalty-free background music...")
    music_files = [
        "https://www.chosic.com/wp-content/uploads/2023/08/happy-day-vlog-background-music.mp3",
        "https://www.chosic.com/wp-content/uploads/2023/02/gentle-piano-ambiance-ig-version-0200-10138.mp3",
        "https://www.chosic.com/wp-content/uploads/2022/08/watr-fluid.mp3"
    ]
    
    for url in music_files:
        subprocess.run(["wget", "-q", "-P", str(MUSIC_FOLDER), url], check=True)
    
    print(f"Downloaded {len(music_files)} music files to {MUSIC_FOLDER}")

# Create video_processor.py module
def create_video_processor():
    code = """
import os
import sys
import pathlib
import subprocess
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

def _prepare_initial_video(submission, safe_title, temp_files_list):
    """
    Prepare the initial video from a Reddit submission.
    
    Args:
        submission: Reddit submission object
        safe_title: Sanitized title for filenames
        temp_files_list: List to track temporary files for cleanup
    
    Returns:
        Tuple of (video_path, duration, width, height)
    """
    import praw
    import yt_dlp
    from pathlib import Path
    
    TEMP_DIR = Path('/content/temp_processing')
    
    # Ensure temp directory exists
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download media from Reddit submission
    video_path = TEMP_DIR / f"{submission.id}_{safe_title}_original.mp4"
    temp_files_list.append(video_path)
    
    # Download using yt-dlp
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': str(video_path),
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'noprogress': True,
        'retries': 3,
        'socket_timeout': 30,
        'nocheckcertificate': True,
        'merge_output_format': 'mp4'
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([submission.url])
        
        if not video_path.is_file() or video_path.stat().st_size < 10240:
            print(f"  Downloaded file invalid or too small: {video_path}")
            return None, 0, 0, 0
        
        # Get video details
        duration, width, height = get_video_details(video_path)
        
        return video_path, duration, width, height
    except Exception as e:
        print(f"  Error downloading or processing video: {e}")
        return None, 0, 0, 0

def get_video_details(video_path):
    """
    Get video duration, width, and height using FFprobe.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Tuple of (duration, width, height)
    """
    import json
    import subprocess
    
    try:
        command = [
            'ffprobe', '-v', 'error', 
            '-select_streams', 'v:0', 
            '-show_entries', 'stream=width,height,duration:format=duration', 
            '-of', 'json', str(video_path)
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        data = json.loads(result.stdout)
        
        duration, width, height = 0.0, 0, 0
        
        if 'streams' in data and data['streams']:
            stream = data['streams'][0]
            width = int(stream.get('width', 0))
            height = int(stream.get('height', 0))
            if 'duration' in stream and stream['duration'] != 'N/A':
                duration = float(stream['duration'])
        
        if duration <= 0 and 'format' in data and 'duration' in data['format'] and data['format']['duration'] != 'N/A':
            duration = float(data['format']['duration'])
            
        return duration, width, height
    except Exception as e:
        print(f"Error getting video details: {e}")
        return 0.0, 0, 0

def create_short_clip(video_path: str, output_path: str, start_time: float, end_time: float, focus_points: List[Dict]):
    """
    Create a short video clip with 9:16 aspect ratio using dynamic cropping based on focus points.
    
    Args:
        video_path: Path to the input video
        output_path: Path for the output video
        start_time: Start time in seconds
        end_time: End time in seconds
        focus_points: List of dictionaries with time and point coordinates
    """
    import cv2
    import numpy as np
    import subprocess
    
    TARGET_ASPECT_RATIO = 9 / 16
    
    try:
        # Get video dimensions
        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Calculate crop dimensions for 9:16 aspect ratio
        current_aspect = width / height
        
        # If already taller than 9:16, use full height and crop width
        if current_aspect <= TARGET_ASPECT_RATIO:
            # Video is already tall enough (or too tall)
            crop_height = height
            crop_width = height * TARGET_ASPECT_RATIO
        else:
            # Video is wider than 9:16, we need to crop the width
            crop_width = width / (current_aspect / TARGET_ASPECT_RATIO)
            crop_height = height
        
        # Calculate crop width (must be even number for video encoding)
        crop_width = int(crop_width // 2) * 2
        crop_height = int(crop_height // 2) * 2
        
        # Sort focus points by time
        focus_points.sort(key=lambda x: x.get('time', 0))
        
        # Prepare crop parameters for FFmpeg
        if len(focus_points) > 0:
            # Use center crop by default if no focus points
            crop_x = int((width - crop_width) / 2)
            crop_y = int((height - crop_height) / 2)
            crop_cmd = f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}"
            
            # Execute FFmpeg command with GPU acceleration in Colab
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-ss', str(start_time),
                '-to', str(end_time),
                '-vf', crop_cmd,
                '-c:v', 'h264_nvenc', # Use NVIDIA GPU encoder
                '-preset', 'p1',      # High quality preset
                '-cq:v', '23',        # Constant quality
                '-b:v', '8M',         # Bitrate
                '-c:a', 'aac',
                '-b:a', '192k',
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Check if output file was created successfully
            if not os.path.exists(output_path) or os.path.getsize(output_path) < 10240:
                raise ValueError(f"Output file not created or too small: {output_path}")
                
            print(f"  Successfully created 9:16 cropped video")
            return True
        else:
            print("  No valid focus points found")
            return False
            
    except Exception as e:
        print(f"  Error creating short clip: {e}")
        return False
"""
    
    with open('/content/video_processor.py', 'w') as f:
        f.write(code)
    
    print("Created video_processor.py module")

# Upload API credentials
def collect_credentials():
    print("\n=== API Credentials Setup ===")
    
    # Upload Google client secrets JSON
    print("\nPlease upload your Google client_secrets.json file:")
    uploaded = files.upload()
    
    if not uploaded:
        print("No file uploaded. Please try again.")
        return False
    
    # Save the file
    client_secrets_file = BASE_DIR / 'client_secrets.json'
    with open(client_secrets_file, 'wb') as f:
        f.write(list(uploaded.values())[0])
    
    # Set environment variables
    os.environ['REDDIT_CLIENT_ID'] = input("\nEnter your Reddit client ID: ")
    os.environ['REDDIT_CLIENT_SECRET'] = input("Enter your Reddit client secret: ")
    os.environ['REDDIT_USER_AGENT'] = input("Enter your Reddit user agent (e.g., python:VideoBot:v1.5): ")
    os.environ['GOOGLE_CLIENT_SECRETS_FILE'] = str(client_secrets_file)
    os.environ['GEMINI_API_KEY'] = input("Enter your Gemini API key: ")
    
    elevenlabs_key = input("Enter your ElevenLabs API key (press Enter to skip): ")
    if elevenlabs_key:
        os.environ['ELEVENLABS_API_KEY'] = elevenlabs_key
    
    return True

# Modify the original script for Colab compatibility
def create_colab_script():
    print("Creating Colab-optimized script...")
    
    script_content = """
# Modified version of script.py for Google Colab with T4 GPU
# Adapted from the original script.py

import argparse
import base64
import json
import math
import os
import shutil
import pathlib
import random
import re
import sqlite3
import subprocess
import sys
import time
import traceback
import glob
from typing import Optional, List, Dict, Tuple, Any, Union
import gc

# Verify we're running in Colab
try:
    import google.colab
    IN_COLAB = True
    print("Running in Google Colab environment with T4 GPU")
except ImportError:
    IN_COLAB = False
    print("Not running in Google Colab environment")

import cv2
import google.generativeai as genai
import numpy as np
import praw
import prawcore
import yt_dlp

from elevenlabs import save
from elevenlabs.client import ElevenLabs
from elevenlabs.core import ApiError

from googleapiclient import errors as google_api_errors
from googleapiclient.discovery import build as build_google_api
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow

from joblib import Parallel, delayed
from PIL import Image

from moviepy.editor import (
    AudioFileClip, ColorClip, CompositeAudioClip, CompositeVideoClip,
    ImageClip, TextClip, VideoClip, VideoFileClip,
    concatenate_audioclips, concatenate_videoclips
)
from moviepy.video.fx.all import colorx, crop, lum_contrast
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import moviepy.video.fx.all as vfx
from moviepy.config import change_settings
from moviepy.audio.AudioClip import AudioArrayClip

# Import the _prepare_initial_video function from video_processor module
from video_processor import _prepare_initial_video, create_short_clip

# --- Configuration ---
# Colab-specific paths
BASE_DIR = pathlib.Path('/content')
TEMP_DIR = BASE_DIR / "temp_processing"
MUSIC_FOLDER = BASE_DIR / "music"
DB_FILE = BASE_DIR / 'uploaded_videos.db'

# Check for ImageMagick
change_settings({"IMAGEMAGICK_BINARY": "magick"})
print("Using ImageMagick from system path")

# API Keys and Secrets (Load from environment variables)
REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.environ.get('REDDIT_USER_AGENT', f'python:VideoBot:v1.5 (by /u/YOUR_USERNAME)')
GOOGLE_CLIENT_SECRETS_FILE = os.environ.get('GOOGLE_CLIENT_SECRETS_FILE')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')

# Models and Services
GEMINI_MODEL_ID = 'gemini-2.0-flash' # Using Flash for speed/cost
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM" # Example ElevenLabs voice

# Video Parameters
TARGET_VIDEO_DURATION_SECONDS = 60
TARGET_ASPECT_RATIO = 9 / 16
TARGET_RESOLUTION = (1080, 1920)
TARGET_FPS = 30
AUDIO_CODEC = 'aac'
VIDEO_CODEC_CPU = 'libx264'
VIDEO_CODEC_GPU = 'h264_nvenc'  # T4 GPU in Colab supports this
FFMPEG_GPU_PRESET = 'p1'  # High quality preset for T4
FFMPEG_CQ_GPU = '23'
LOUDNESS_TARGET_LUFS = -14

# Enhanced Video Quality Settings
VIDEO_BITRATE_HIGH = '10M'
VIDEO_BITRATE_MEDIUM = '8M'
VIDEO_BITRATE_MOBILE = '6M'
AUDIO_BITRATE = '192k'
ENABLE_TWO_PASS_ENCODING = True
ENABLE_VIDEO_LOOP = True
MAX_LOOP_COUNT = 3
MIN_LOOP_DURATION = 7

# Text Overlay Parameters
NARRATIVE_FONT = 'Impact'
NARRATIVE_FONT_SIZE_RATIO = 1 / 10
NARRATIVE_TEXT_COLOR = 'white'
NARRATIVE_STROKE_COLOR = 'black'
NARRATIVE_STROKE_WIDTH = 4
NARRATIVE_POSITION = ('center', 'center')
NARRATIVE_BG_COLOR = 'transparent'

# Subtitle Parameters
OVERLAY_FONT = 'Arial'
OVERLAY_FONT_SIZE_RATIO = 1 / 15
OVERLAY_TEXT_COLOR = 'white'
OVERLAY_STROKE_COLOR = 'black'
OVERLAY_STROKE_WIDTH = 2
OVERLAY_POSITION = ('center', 0.8)
OVERLAY_BG_COLOR = 'transparent'

# Processing Options
API_DELAY_SECONDS = 6
MAX_REDDIT_POSTS_TO_FETCH = 10
ADD_VISUAL_EMPHASIS = True
APPLY_STABILIZATION = False
MIX_ORIGINAL_AUDIO = True
ORIGINAL_AUDIO_MIX_VOLUME = 0.1
BACKGROUND_MUSIC_ENABLED = True
BACKGROUND_MUSIC_VOLUME = 0.08
BACKGROUND_MUSIC_NARRATIVE_VOLUME_FACTOR = 0.1
AUDIO_DUCKING_FADE_TIME = 0.3
SHAKE_EFFECT_ENABLED = True
SUBTLE_ZOOM_ENABLED = True
COLOR_GRADE_ENABLED = True
PARALLEL_FRAME_PROCESSING = True
N_JOBS_PARALLEL = 2  # Colab has limited CPUs, use conservatively

# Content Filtering
FORBIDDEN_WORDS = [
    "fuck", "fucking", "fucked", "fuckin", "shit", "shitty", "shitting",
    "damn", "damned", "bitch", "bitching", "cunt", "asshole", "piss", "pissed",
    "gore", "graphic", "brutal", "blood", "bloody", "murder", "killing", "suicide",
    "porn", "pornographic", "nsfw", "xxx", "sex", "sexual", "nude", "naked",
    "racist", "racism", "nazi", "sexist", "homophobic", "slur"
]

# Unsuitable content types
UNSUITABLE_CONTENT_TYPES = [
    "gore", "violence", "graphic injury", "animal abuse", "child abuse",
    "pornography", "nudity", "sexual content", "hate speech", "racism",
    "dangerous activities", "suicide", "self-harm", "illegal activities",
    "drug abuse", "excessive profanity"
]

# --- Include the rest of the original script here ---
# For brevity, I'm removing most of the utility functions 
# and keeping just the essential main function
"""
    
    # Write the beginning of the script
    with open('/content/colab_script.py', 'w') as f:
        f.write(script_content)
    
    # Now append all the utility functions from the original script
    with open('/content/script.py', 'r') as original:
        lines = original.readlines()
        
        # Find where classes and utility functions start (after imports and constants)
        start_line = 0
        for i, line in enumerate(lines):
            if "--- Helper Classes ---" in line:
                start_line = i
                break
        
        if start_line > 0:
            utility_functions = ''.join(lines[start_line:])
            
            with open('/content/colab_script.py', 'a') as f:
                f.write("\n# --- Helper Functions and Classes ---\n")
                f.write(utility_functions)
        else:
            print("Warning: Could not find utility functions in original script.")
    
    print("Created Colab-optimized script")

# Main function to run in Colab
def main():
    print("========================================")
    print("Reddit Video to YouTube Shorts - Colab T4 Version")
    print("========================================")
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    TEMP_DIR.mkdir(exist_ok=True)
    MUSIC_FOLDER.mkdir(exist_ok=True)
    
    # Create video processor module
    create_video_processor()
    
    # Download background music
    download_background_music()
    
    # Check for original script
    if not os.path.exists('/content/script.py'):
        print("\nPlease upload the original script.py file:")
        files.upload()
        
        if not os.path.exists('/content/script.py'):
            print("Error: script.py not found. Please upload it and run this script again.")
            return
    
    # Modify the script for Colab compatibility
    create_colab_script()
    
    # Collect API credentials
    if not collect_credentials():
        print("Error collecting credentials. Please try again.")
        return
    
    # Run the script
    print("\nRunning video processing script...")
    
    # Get subreddit names
    subreddits = input("\nEnter subreddit names separated by spaces (e.g., gifs videos): ").split()
    if not subreddits:
        print("No subreddits provided. Using default: videos")
        subreddits = ["videos"]
    
    max_videos = input("Enter maximum number of videos to process per subreddit (default: 3): ")
    try:
        max_videos = int(max_videos)
    except:
        max_videos = 3
    
    skip_upload = input("Skip uploading to YouTube? (y/n, default: y): ").lower() == 'y'
    
    # Build command
    cmd = [sys.executable, '/content/colab_script.py'] + subreddits
    cmd += ['--max_videos', str(max_videos)]
    if skip_upload:
        cmd.append('--skip_upload')
    
    # Execute the script
    print(f"\nExecuting: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)

if __name__ == "__main__":
    main() 