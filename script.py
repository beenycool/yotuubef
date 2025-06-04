import argparse
import base64
import json
import math
import os
import signal
import sys
import shutil # Keep one shutil import
import pathlib
import random
import re
import sqlite3
import subprocess
import time
import traceback
import glob
import threading  # Add missing import
import queue  # Add queue import at top level
from typing import Optional, List, Dict, Tuple, Any, Union
import gc
import logging # Import logging
from dotenv import load_dotenv  # Add dotenv support
from google.oauth2.credentials import Credentials
import tempfile
import httplib2
import googleapiclient.discovery
import googleapiclient.errors
import googleapiclient.http
import io
import ssl
from datetime import datetime
import socket
import google.auth.transport.requests  # Add import for Google auth transport

DEFAULT_AUDIO_FPS = 44100

def fix_moviepy_installation():
    """Check and fix the moviepy installation."""
    logging.info("Checking moviepy installation...")
    
    # Get the site-packages directory
    venv_dir = pathlib.Path('.venv')
    site_packages_paths = list(venv_dir.glob('Lib/site-packages'))
    if not site_packages_paths:
        logging.warning("Could not find site-packages directory. Skipping moviepy fix.")
        return True
        
    site_packages = site_packages_paths[0]
    moviepy_dir = site_packages / 'moviepy'
    
    if not moviepy_dir.exists():
        logging.warning(f"moviepy directory not found at {moviepy_dir}")
        return False
    
    # Check if editor.py exists directly in the moviepy directory
    editor_py = moviepy_dir / 'editor.py'
    if not editor_py.exists():
        logging.info(f"MoviePy editor.py not found at {editor_py}")
        logging.info("Reinstalling moviepy to version 1.0.3...")
        
        # Completely reinstall moviepy
        try:
            subprocess.run([
                sys.executable, '-m', 'pip',
                'uninstall', '-y', 'moviepy'
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            subprocess.run([
                sys.executable, '-m', 'pip',
                'install', 'moviepy==1.0.3'
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logging.info("Moviepy reinstalled successfully!")
            
            # Try to import to verify the fix worked
            try:
                import importlib
                if 'moviepy' in sys.modules:
                    importlib.reload(sys.modules['moviepy'])
                if 'moviepy.editor' in sys.modules:
                    importlib.reload(sys.modules['moviepy.editor'])
                logging.info("Moviepy imports refreshed after reinstallation")
            except Exception as e:
                logging.warning(f"Failed to refresh moviepy imports: {e}")
            
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to reinstall moviepy: {e}")
            return False
    
    logging.info("MoviePy installation looks good!")
    return True

# Run the fix before importing moviepy
fix_moviepy_installation()

import cv2
import google.generativeai as genai
import numpy as np
import praw
import prawcore
import yt_dlp

from googleapiclient import errors as google_api_errors
from googleapiclient.discovery import build as build_google_api
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow

from joblib import Parallel, delayed
from PIL import Image, ImageDraw, ImageFont # Ensure ImageDraw, ImageFont are imported if used directly in thumbnail

from moviepy.editor import (
    AudioFileClip, ColorClip, CompositeAudioClip, CompositeVideoClip,
    ImageClip, TextClip, VideoClip, VideoFileClip,
    concatenate_audioclips, concatenate_videoclips
)
from moviepy.video.fx.all import colorx, crop, lum_contrast # crop might not be used directly, but good to have
import moviepy.video.fx.all as vfx
from moviepy.config import change_settings
from moviepy.audio.AudioClip import AudioArrayClip

import torch
from transformers import pipeline
import soundfile as sf

# --- Configure PyTorch to use GPU ---
# Add this code to force PyTorch to use the CUDA GPU if available
CUDA_AVAILABLE = torch.cuda.is_available()
GPU_DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
if CUDA_AVAILABLE:
    torch.cuda.set_device(0)  # Use the first GPU

# --- Logging Configuration ---
# Set default logging level to WARNING, override to INFO/DEBUG if --debug is passed
logging.basicConfig(
    level=logging.WARNING, # Default to WARNING, will be changed in main() if needed
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
        # You can also add: logging.FileHandler("video_automation.log")
    ]
)

# Only set to INFO/DEBUG if --debug is passed (handled in main())
# logging.getLogger().setLevel(logging.INFO)  # <-- Remove this line to avoid always being verbose

# --- External Dependency ---
# Ensure 'video_processor.py' is in the same directory or Python path
# It must contain: _prepare_initial_video, create_short_clip
try:
    from video_processor import _prepare_initial_video, create_short_clip, mute_sections
    logging.info("Successfully imported from video_processor.py")
except ImportError:
    logging.error("CRITICAL: Failed to import from video_processor.py. This script will not run correctly.")
    logging.error("Please ensure video_processor.py exists and contains _prepare_initial_video and create_short_clip functions.")
    # sys.exit(1) # Optionally exit if this dependency is critical for any operation

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

BASE_DIR = pathlib.Path(__file__).parent.resolve()
TEMP_DIR = BASE_DIR / "temp_processing"

MUSIC_FOLDER_ENV = os.environ.get('MUSIC_FILES_DIR')
MUSIC_FOLDER = pathlib.Path(MUSIC_FOLDER_ENV) if MUSIC_FOLDER_ENV else BASE_DIR / "music"

SOUND_EFFECTS_FOLDER_ENV = os.environ.get('SOUND_EFFECTS_DIR')
SOUND_EFFECTS_FOLDER = pathlib.Path(SOUND_EFFECTS_FOLDER_ENV) if SOUND_EFFECTS_FOLDER_ENV else BASE_DIR / "sound_effects"

WATERMARK_PATH_ENV = os.environ.get('WATERMARK_FILE_PATH')
WATERMARK_PATH = pathlib.Path(WATERMARK_PATH_ENV) if WATERMARK_PATH_ENV else BASE_DIR / "watermark.png"

DB_FILE_PATH_ENV = os.environ.get('DB_FILE_PATH')
DB_FILE = pathlib.Path(DB_FILE_PATH_ENV) if DB_FILE_PATH_ENV else BASE_DIR / 'uploaded_videos.db'

REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.environ.get('REDDIT_USER_AGENT', f'python:VideoBot:v1.8 (by /u/YOUR_USERNAME)') # Version bump
GOOGLE_CLIENT_SECRETS_FILE_PATH = os.environ.get('GOOGLE_CLIENT_SECRETS_FILE')
YOUTUBE_TOKEN_FILE_PATH = os.environ.get('YOUTUBE_TOKEN_FILE')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

logging.info(f"REDDIT_CLIENT_ID: {'SET' if REDDIT_CLIENT_ID else 'NOT SET'}")
logging.info(f"REDDIT_CLIENT_SECRET: {'SET' if REDDIT_CLIENT_SECRET else 'NOT SET'}")
logging.info(f"REDDIT_USER_AGENT: {REDDIT_USER_AGENT}")
logging.info(f"GOOGLE_CLIENT_SECRETS_FILE_PATH: {GOOGLE_CLIENT_SECRETS_FILE_PATH}")
logging.info(f"YOUTUBE_TOKEN_FILE_PATH: {YOUTUBE_TOKEN_FILE_PATH}")
logging.info(f"GEMINI_API_KEY: {'SET' if GEMINI_API_KEY else 'NOT SET'}")
logging.info(f"DB_FILE: {DB_FILE}")
logging.info(f"MUSIC_FOLDER: {MUSIC_FOLDER}")
logging.info(f"SOUND_EFFECTS_FOLDER: {SOUND_EFFECTS_FOLDER}")
logging.info(f"WATERMARK_PATH: {WATERMARK_PATH}")

# ImageMagick
IMAGEMAGICK_PATHS = [
    r"C:\Program Files\ImageMagick-*\magick.exe",
    r"C:\Program Files (x86)\ImageMagick-*\magick.exe",
    os.path.expandvars(r"%ProgramFiles%\ImageMagick-*\magick.exe"),
    os.path.expandvars(r"%ProgramFiles(x86)%\ImageMagick-*\magick.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\ImageMagick\magick.exe"),
    "magick"
]
image_magick_found = False
for path_pattern in IMAGEMAGICK_PATHS:
    resolved_paths = glob.glob(path_pattern) if '*' in path_pattern else [path_pattern]
    for path in resolved_paths:
        try:
            change_settings({"IMAGEMAGICK_BINARY": path})
            subprocess.run([path, "-version"], check=True, capture_output=True, timeout=5)
            image_magick_found = True
            logging.info(f"ImageMagick found and working at: {path}")
            break
        except Exception:
            change_settings({"IMAGEMAGICK_BINARY": ""}) # Reset on failure
            continue
    if image_magick_found:
        break
if not image_magick_found:
    logging.warning("ImageMagick not found or not working. Text overlays might be basic or fail.")

# Load environment variables from .env file if present
load_dotenv()

# API Keys - Load from environment variables with improved fallbacks
REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.environ.get('REDDIT_USER_AGENT', f'python:VideoBot:v1.6 (by /u/YOUR_USERNAME)')

# Load YouTube token file from GOOGLE_CLIENT_SECRETS_FILE if it exists
google_secrets_file = os.environ.get('GOOGLE_CLIENT_SECRETS_FILE')
if google_secrets_file and os.path.exists(google_secrets_file):
    YOUTUBE_TOKEN_FILE = google_secrets_file
    logging.info(f"Using YouTube token file from GOOGLE_CLIENT_SECRETS_FILE: {YOUTUBE_TOKEN_FILE}")
else:
    YOUTUBE_TOKEN_FILE = os.environ.get('YOUTUBE_TOKEN_FILE', 'youtube_token.json')
    logging.info(f"Using default YouTube token file: {YOUTUBE_TOKEN_FILE}")

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Print current environment variable values (for debugging)
logging.info(f"REDDIT_CLIENT_ID: {'SET' if REDDIT_CLIENT_ID else 'NOT SET'}")
logging.info(f"REDDIT_CLIENT_SECRET: {'SET' if REDDIT_CLIENT_SECRET else 'NOT SET'}")
logging.info(f"YOUTUBE_TOKEN_FILE: {YOUTUBE_TOKEN_FILE}")
logging.info(f"YOUTUBE_TOKEN_FILE exists: {os.path.exists(YOUTUBE_TOKEN_FILE)}")
logging.info(f"GEMINI_API_KEY: {'SET' if GEMINI_API_KEY else 'NOT SET'}")

# If no environment variables set, check for a credentials file
if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, GEMINI_API_KEY]):
    try:
        with open('credentials.json', 'r') as f:
            creds = json.load(f)
            REDDIT_CLIENT_ID = REDDIT_CLIENT_ID or creds.get('reddit', {}).get('client_id')
            REDDIT_CLIENT_SECRET = REDDIT_CLIENT_SECRET or creds.get('reddit', {}).get('client_secret')
            GEMINI_API_KEY = GEMINI_API_KEY or creds.get('gemini', {}).get('api_key')
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(f"Could not load credentials from credentials.json: {e}")

# YouTube
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
YOUTUBE_SCOPES = ['https://www.googleapis.com/auth/youtube.upload', 'https://www.googleapis.com/auth/youtube.force-ssl', 'https://www.googleapis.com/auth/youtubepartner']
YOUTUBE_UPLOAD_CATEGORY_ID = '24'
YOUTUBE_UPLOAD_PRIVACY_STATUS = 'public'
YOUTUBE_SELF_CERTIFICATION = True

# Models & Services
GEMINI_MODEL_ID = 'gemini-2.0-flash' #KEEP THIS AS GEMINI-2.0-FLASH 
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
        

# Video Parameters
TARGET_VIDEO_DURATION_SECONDS = 59 # Keep slightly under 60 for shorts
TARGET_ASPECT_RATIO = 9 / 16
TARGET_RESOLUTION = (1080, 1920)
TARGET_FPS = 30
AUDIO_CODEC = 'aac'
VIDEO_CODEC_CPU = 'libx264'
VIDEO_CODEC_GPU = 'h264_nvenc' # For NVIDIA GPUs
FFMPEG_CPU_PRESET = 'medium'
FFMPEG_GPU_PRESET = 'p5'
FFMPEG_CRF_CPU = '23' # Constant Rate Factor (lower is better quality, larger file)
FFMPEG_CQ_GPU = '23'  # Constant Quality (for NVENC, similar concept to CRF)
LOUDNESS_TARGET_LUFS = -14.0

# Quality Settings
VIDEO_BITRATE_HIGH = '10M' # Target bitrate if not using CRF/CQ
AUDIO_BITRATE = '192k'
# ENABLE_TWO_PASS_ENCODING = False # Simplified to single pass with CRF/CQ for MoviePy ease

# Text Overlay
# Graphical Text (from Gemini's "text_overlays", e.g., "SO TINY!")
_graphical_text_font_path = BASE_DIR / "fonts" / "Montserrat-Bold.ttf"
GRAPHICAL_TEXT_FONT = str(_graphical_text_font_path) if _graphical_text_font_path.is_file() else 'Arial'
GRAPHICAL_TEXT_FONT_SIZE_RATIO = 1 / 18  # Reduced from 1/12
GRAPHICAL_TEXT_COLOR = 'white'
GRAPHICAL_TEXT_STROKE_COLOR = 'black'
GRAPHICAL_TEXT_STROKE_WIDTH = 2       # Reduced from 3
GRAPHICAL_TEXT_DEFAULT_BG_COLOR = 'rgba(0,0,0,0.5)'

# Subtitles (from TTS narrative_script_segments)
_subtitle_font_filename = "Montserrat-Regular.ttf"
_subtitle_font_path = BASE_DIR / "fonts" / _subtitle_font_filename
SUBTITLE_FONT = str(_subtitle_font_path) if _subtitle_font_path.is_file() else 'Arial'
SUBTITLE_FONT_SIZE_RATIO = 1 / 25     # Smaller for subtitles
SUBTITLE_TEXT_COLOR = 'white'
SUBTITLE_STROKE_COLOR = 'black'
SUBTITLE_STROKE_WIDTH = 1
SUBTITLE_POSITION = ('center', 0.92)   # Relative: 92% from top (bottom 8%)
SUBTITLE_BG_COLOR = 'rgba(0,0,0,0.4)'  # More subtle background


# Processing Options
API_DELAY_SECONDS = 2 # Increased slightly
MAX_REDDIT_POSTS_TO_FETCH = 10
ORIGINAL_AUDIO_MIX_VOLUME = 0.15
BACKGROUND_MUSIC_ENABLED = True
BACKGROUND_MUSIC_VOLUME = 0.06
BACKGROUND_MUSIC_NARRATIVE_VOLUME_FACTOR = 0.2 # How much to reduce music vol when TTS is present
SUBTLE_ZOOM_ENABLED = True
COLOR_GRADE_ENABLED = True

# Seamless looping configuration
ENABLE_SEAMLESS_LOOPING = os.environ.get('ENABLE_SEAMLESS_LOOPING', 'true').lower() == 'true'
LOOP_CROSSFADE_DURATION = float(os.environ.get('LOOP_CROSSFADE_DURATION', '0.3'))
LOOP_COMPATIBILITY_THRESHOLD = float(os.environ.get('LOOP_COMPATIBILITY_THRESHOLD', '0.3'))
LOOP_SAMPLE_DURATION = float(os.environ.get('LOOP_SAMPLE_DURATION', '0.5'))
LOOP_TARGET_DURATION = float(os.environ.get('LOOP_TARGET_DURATION', '0')) if os.environ.get('LOOP_TARGET_DURATION') else None
ENABLE_AUDIO_CROSSFADE = os.environ.get('ENABLE_AUDIO_CROSSFADE', 'true').lower() == 'true'
LOOP_EXTEND_MODE = os.environ.get('LOOP_EXTEND_MODE', 'middle_repeat')  # Options: 'middle_repeat', 'slow_motion', 'none'
LOOP_TRIM_FROM_CENTER = os.environ.get('LOOP_TRIM_FROM_CENTER', 'true').lower() == 'true'

logging.info(f"ENABLE_SEAMLESS_LOOPING: {ENABLE_SEAMLESS_LOOPING}")
logging.info(f"LOOP_CROSSFADE_DURATION: {LOOP_CROSSFADE_DURATION}s")
logging.info(f"LOOP_COMPATIBILITY_THRESHOLD: {LOOP_COMPATIBILITY_THRESHOLD}")
logging.info(f"LOOP_SAMPLE_DURATION: {LOOP_SAMPLE_DURATION}s")
logging.info(f"LOOP_TARGET_DURATION: {LOOP_TARGET_DURATION}s" if LOOP_TARGET_DURATION else "LOOP_TARGET_DURATION: Auto")
logging.info(f"ENABLE_AUDIO_CROSSFADE: {ENABLE_AUDIO_CROSSFADE}")
logging.info(f"LOOP_EXTEND_MODE: {LOOP_EXTEND_MODE}")
logging.info(f"LOOP_TRIM_FROM_CENTER: {LOOP_TRIM_FROM_CENTER}")

# Content Filtering
FORBIDDEN_WORDS = [
    "fuck", "fucking", "fucked", "fuckin", "shit", "shitty", "shitting",
    "wtf", "stfu", "omfg", "porn", "pornographic", "nsfw", "xxx", "sex", "sexual", "nude", "naked",
    "racist", "racism", "nazi", "sexist",
    "dangerous activities glorification", "suicide promotion", "self-harm"
] # Reduced list - removed less serious items

CURATED_SUBREDDITS = [
    "oddlysatisfying", "nextfuckinglevel", "BeAmazed", "woahdude", "MadeMeSmile", "Eyebleach",
    "interestingasfuck", "Damnthatsinteresting", "AnimalsBeingBros", "HumansBeingBros",
    "wholesomememes", "ContagiousLaughter", "foodporn", "CookingVideos", "ArtisanVideos",
    "educationalgifs", "DIY", "gardening", "science", "space", "NatureIsCool", "aww",
    "AnimalsBeingDerps", "rarepuppers", "LifeProTips", "GetMotivated", "toptalent",
    "BetterEveryLoop", "childrenfallingover", "instantregret", "wholesomegifs",
    "Unexpected", "nevertellmetheodds", "whatcouldgoright", "holdmymilk", "maybemaybemaybe",
    "mildlyinteresting"
] # "NatureIsFuckingLit", "EarthPorn" removed due to potential profanity in name affecting strict filters

# Music
MUSIC_CATEGORIES = {
    "upbeat": ["energetic", "positive", "happy", "uplifting"],
    "emotional": ["sad", "heartwarming", "touching", "sentimental"],
    "suspenseful": ["tense", "dramatic", "action", "exciting"],
    "relaxing": ["calm", "peaceful", "ambient", "soothing"],
    "funny": ["quirky", "comedic", "playful", "lighthearted"],
    "informative": ["neutral", "documentary", "educational", "background"]
}
MONETIZATION_TAGS = [
    "family friendly", "educational", "informative", "wholesome", "positive",
    "interesting", "amazing", "satisfying", "relaxing", "shorts", "shortvideo"
]

# Fallback AI Analysis Structure
FALLBACK_ANALYSIS = {
    "suggested_title": "Interesting Reddit Video",
    "summary_for_description": "Check out this cool video from Reddit!",
    "mood": "neutral",
    "has_clear_narrative": False,
    "original_audio_is_key": True,
    "hook_text": "",
    "hook_variations": [  # Added for variation in hooks
        "WAIT FOR IT...", 
        "WATCH THIS!", 
        "WHAT HAPPENS NEXT?"
    ],
    "visual_hook_moment": {  # Added for identifying crucial first moment
        "timestamp_seconds": 0.0,
        "description": "Default visual hook"
    },
    "audio_hook": {  # Added for audio hooks
        "type": "sound_effect",  # Options: "sound_effect", "music_cut", "original_audio"
        "sound_name": "whoosh",  # If type is sound_effect
        "timestamp_seconds": 0.0
    },
    "best_segment": {"start_seconds": 0, "end_seconds": 0, "reason": "N/A"},
    "segments": [{"start_seconds": 0, "end_seconds": 59, "reason": "Default segment"}],
    "key_focus_points": [],
    "text_overlays": [],
    "narrative_script": [],  # Keep for backward compatibility
    "narrative_script_segments": [  # Add default narrative for TTS in every video
        {
            "text": "Get ready for this!",
            "time_seconds": 0.5,
            "intended_duration_seconds": 1.8
        },
        {
            "text": "More cool videos on the way - subscribe!",
            "time_seconds": 10.0,
            "intended_duration_seconds": 2.5
        }
    ],
    "visual_cues": [],
    "visual_cues_suggestions": [],  # Added consistency with the Gemini output
    "speed_effects": [  # Added for speed adjustments
        {
            "start_seconds": 0,
            "end_seconds": 0,
            "speed_factor": 1.0,
            "effect_type": "none"  # Options: "none", "speedup", "slowdown", "freeze_frame"
        }
    ],
    "music_genres": ["neutral"],
    "key_audio_moments": [],
    "key_audio_moments_original": [],  # Added consistency with the Gemini output
    "sound_effects": [  # Added for sound effects
        {
            "timestamp_seconds": 0.5,
            "effect_name": "whoosh",
            "volume": 0.5
        }
    ],
    "retention_tactics": [],
    "hashtags": ["reddit", "video", "shorts"],
    "original_duration": 0.0,
    "tts_pacing": "normal",
    "emotional_keywords": ["neutral", "standard"],  # Added for better emotional guidance
    "thumbnail_info": {
        "timestamp_seconds": 0.0,
        "reason": "Default fallback",
        "headline_text": ""
    },
    "call_to_action": {"text": "", "type": "none"},
    "emotional_arc_description": "N/A",
    "story_structure": {  # Added for clearer story structure 
        "setup": "Video begins",
        "inciting_incident": "Main event occurs",
        "climax": "Peak moment" 
    },
    "is_explicitly_age_restricted": False,
    "fallback": True
}

# Global API Clients (initialized in setup_api_clients)
reddit: Optional[praw.Reddit] = None
youtube_service: Optional[Any] = None # googleapiclient.discovery.Resource
gemini_model: Optional[genai.GenerativeModel] = None
db_conn: Optional[sqlite3.Connection] = None
db_cursor: Optional[sqlite3.Cursor] = None

# --- Helper Functions ---
def check_ffmpeg_install(tool_name: str) -> bool:
    """Checks if an FFmpeg-related tool is installed and executable."""
    try:
        is_windows = os.name == 'nt'
        
        # Special handling for ffmpeg-normalize on Windows
        if tool_name == "ffmpeg-normalize" and is_windows:
            # Check in Python Scripts directory first
            scripts_dir = os.path.join(os.path.dirname(sys.executable), 'Scripts')
            ffmpeg_normalize_path = os.path.join(scripts_dir, 'ffmpeg-normalize.exe')
            if os.path.exists(ffmpeg_normalize_path):
                logging.info(f"Found {tool_name} at: {ffmpeg_normalize_path}")
                # Update PATH to include Scripts directory
                os.environ['PATH'] = scripts_dir + os.pathsep + os.environ['PATH']
                # Try running it
                try:
                    subprocess.run([ffmpeg_normalize_path, "-version"], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL, 
                                check=True)
                    logging.info(f"{tool_name} found and working from Scripts directory.")
                    return True
                except subprocess.CalledProcessError:
                    logging.warning(f"{tool_name} found in Scripts but failed to run.")
                    return False
        
        # Standard check for all other tools
        process = subprocess.run(
            [tool_name, "-version"],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True,
            shell=is_windows # Helps find executables on Windows PATH
        )
        logging.info(f"{tool_name} found and working.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        logging.warning(f"{tool_name} not found or not working properly. Error: {e}")
        return False

def get_video_details(video_path: pathlib.Path) -> Tuple[float, int, int]:
    """Returns (duration_seconds, width_px, height_px) using ffprobe."""
    if not video_path.is_file():
        logging.error(f"Video file not found for details: {video_path}")
        return 0.0, 0, 0
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration,avg_frame_rate',
            '-of', 'json', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        data = json.loads(result.stdout)

        if not data.get('streams'):
            logging.warning(f"No video streams found in {video_path} by ffprobe.")
            return _get_video_details_cv2_fallback(video_path)

        stream_data = data['streams'][0]
        duration_str = stream_data.get('duration')

        if duration_str:
            duration = float(duration_str)
        else: # Fallback for duration if not in stream (e.g., for some containers)
            cmd_format_duration = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
            ]
            result_format_duration = subprocess.run(cmd_format_duration, capture_output=True, text=True, check=True, timeout=10)
            duration = float(result_format_duration.stdout.strip()) if result_format_duration.stdout.strip() else 0.0

        width = int(stream_data.get('width', 0))
        height = int(stream_data.get('height', 0))
        
        # FPS (not returned by this func signature but good to know how)
        # fps_str = stream_data.get('avg_frame_rate', '0/1')
        # num, den = map(int, fps_str.split('/'))
        # fps = num / den if den else 0.0

        if width == 0 or height == 0 or duration == 0.0: # If ffprobe returns bad values
            logging.warning(f"ffprobe returned incomplete data for {video_path}. Trying OpenCV fallback.")
            return _get_video_details_cv2_fallback(video_path)

        return duration, width, height
    except Exception as e:
        logging.error(f"ffprobe error getting details for {video_path}: {e}. Trying OpenCV fallback.")
        return _get_video_details_cv2_fallback(video_path)

def _get_video_details_cv2_fallback(video_path: pathlib.Path) -> Tuple[float, int, int]:
    """OpenCV fallback for video details."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"OpenCV could not open {video_path} for fallback details.")
            return 0.0, 0, 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps and fps > 0 else 0.0
        cap.release()
        if width == 0 or height == 0: # OpenCV sometimes fails to get dimensions for corrupt files
            logging.warning(f"OpenCV also failed to get valid dimensions for {video_path}.")
            return 0.0, 0, 0
        logging.info(f"OpenCV fallback details for {video_path}: dur={duration:.2f}s, {width}x{height}")
        return duration, width, height
    except Exception as e_cv:
        logging.error(f"OpenCV fallback failed for {video_path}: {e_cv}")
        return 0.0, 0, 0

def cleanup_temp_files(file_path: Union[pathlib.Path, str, None]):
    if file_path:
        path_obj = pathlib.Path(file_path)
        if path_obj.is_file():
            try:
                path_obj.unlink()
                # logging.debug(f"Cleaned up temp file: {path_obj}")
            except OSError as e:
                logging.warning(f"Could not delete temp file {path_obj}: {e}")
        elif path_obj.is_dir(): # Added to clean up directories if needed
            try:
                shutil.rmtree(path_obj)
                # logging.debug(f"Cleaned up temp directory: {path_obj}")
            except OSError as e:
                logging.warning(f"Could not delete temp directory {path_obj}: {e}")


def contains_forbidden_words(text: Optional[str]) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    # Build a regex pattern that matches any forbidden word as a whole word
    pattern = r'\\b(' + '|'.join(re.escape(word) for word in FORBIDDEN_WORDS) + r')\\b'
    return re.search(pattern, text_lower) is not None

# --- Setup ---
def validate_environment():
    essential_vars_paths = {
        'REDDIT_CLIENT_ID': REDDIT_CLIENT_ID,
        'REDDIT_CLIENT_SECRET': REDDIT_CLIENT_SECRET,
        'GEMINI_API_KEY': GEMINI_API_KEY,
        'YOUTUBE_TOKEN_FILE': YOUTUBE_TOKEN_FILE_PATH,
    }
    missing_vars = [name for name, value in essential_vars_paths.items() if not value]
    if missing_vars:
        logging.error(f"Missing essential environment variables or their values: {', '.join(missing_vars)}")
        logging.error("Please ensure these are set in your .env file:")
        logging.error("  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, GEMINI_API_KEY, YOUTUBE_TOKEN_FILE")
        logging.error("  (YOUTUBE_TOKEN_FILE should point to the generated token, set via env var)")
        logging.error("  (GOOGLE_CLIENT_SECRETS_FILE should point to your client_secret_....json from Google for initial auth)")
        raise EnvironmentError("Missing critical environment variable configurations. Check .env file.")

    if YOUTUBE_TOKEN_FILE_PATH and not os.path.exists(YOUTUBE_TOKEN_FILE_PATH):
        logging.warning(f"YouTube token file specified by YOUTUBE_TOKEN_FILE env var ('{YOUTUBE_TOKEN_FILE_PATH}') not found.")
        logging.warning("You may need to run an authentication script (like auth_youtube.py) first.")
        logging.warning(f"Make sure GOOGLE_CLIENT_SECRETS_FILE in .env points to your downloaded client_secret_....json file.")
    
    if not GEMINI_API_KEY:
        logging.warning("GEMINI_API_KEY not set. Gemini AI analysis will be disabled.")

    if not check_ffmpeg_install("ffmpeg"):
        raise RuntimeError("FFmpeg is required but not found.")
    logging.info("Environment validation seems okay (values exist, basic checks passed).")

def setup_directories():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    MUSIC_FOLDER.mkdir(parents=True, exist_ok=True)
    for category in MUSIC_CATEGORIES:
        (MUSIC_FOLDER / category).mkdir(parents=True, exist_ok=True)
    
    # Create sound effects directory
    SOUND_EFFECTS_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Create fonts directory and ensure necessary fonts are available
    font_dir = BASE_DIR / "fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    
    # Download Montserrat-Bold.ttf if it doesn't exist (for graphical text)
    montserrat_bold_font_path = font_dir / "Montserrat-Bold.ttf"
    if not montserrat_bold_font_path.is_file():
        try:
            import requests
            logging.info(f"Downloading Montserrat-Bold font to {montserrat_bold_font_path}")
            font_url = "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf"
            response = requests.get(font_url, timeout=10)
            response.raise_for_status()
            with open(montserrat_bold_font_path, 'wb') as f:
                f.write(response.content)
            logging.info("Montserrat-Bold font downloaded successfully.")
        except Exception as e:
            logging.warning(f"Failed to download Montserrat-Bold font: {e}. Will use system fonts.")

    # Download Montserrat-Regular.ttf if it doesn't exist (for subtitles)
    montserrat_regular_font_path = font_dir / "Montserrat-Regular.ttf"
    if not montserrat_regular_font_path.is_file():
        try:
            import requests
            logging.info(f"Downloading Montserrat-Regular font to {montserrat_regular_font_path}")
            font_url = "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Regular.ttf"
            response = requests.get(font_url, timeout=10)
            response.raise_for_status()
            with open(montserrat_regular_font_path, 'wb') as f:
                f.write(response.content)
            logging.info("Montserrat-Regular font downloaded successfully.")
        except Exception as e:
            logging.warning(f"Failed to download Montserrat-Regular font: {e}. Will use system Arial for subtitles if this specific font is not found.")
    
    # Download BebasNeue-Regular.ttf for thumbnails if it doesn't exist
    bebas_font_path = font_dir / "BebasNeue-Regular.ttf" # PIL might prefer .ttf, but .otf often works
    if not bebas_font_path.is_file():
        try:
            import requests
            logging.info(f"Downloading BebasNeue-Regular font to {bebas_font_path}")
            # Common source for Bebas Neue (often OTF, PIL can usually handle it)
            font_url = "https://github.com/dharmatype/Bebas-Neue/raw/master/fonts/otf/BebasNeue-Regular.otf"
            response = requests.get(font_url, timeout=10)
            response.raise_for_status()
            with open(bebas_font_path, 'wb') as f: # Save with .ttf extension for consistency if desired, or keep .otf
                f.write(response.content)
            logging.info("BebasNeue-Regular font downloaded successfully.")
        except Exception as e:
            logging.warning(f"Failed to download BebasNeue-Regular font: {e}. Will use system fonts.")
    
    # Log available fonts
    available_fonts = list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf"))
    logging.info(f"Available fonts in {font_dir}: {[f.name for f in available_fonts]}")
    
    logging.info("Temporary and asset directories ensured.")

def setup_database():
    global db_conn, db_cursor
    try:
        db_conn = sqlite3.connect(DB_FILE, timeout=10)
        db_cursor = db_conn.cursor()
        
        # Check if table exists and has the correct schema
        try:
            # Check if uploads table exists and has all required columns
            db_cursor.execute("PRAGMA table_info(uploads)")
            columns = {row[1] for row in db_cursor.fetchall()}
            
            if not columns:
                # Table doesn't exist, create it with all columns
                db_cursor.execute('''
                    CREATE TABLE uploads (
                        reddit_url TEXT PRIMARY KEY,
                        youtube_url TEXT,
                        upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        title TEXT,
                        subreddit TEXT
                    )
                ''')
            else:
                # Table exists but may need schema update
                missing_columns = []
                if 'title' not in columns:
                    missing_columns.append("ADD COLUMN title TEXT")
                if 'subreddit' not in columns:
                    missing_columns.append("ADD COLUMN subreddit TEXT")
                
                # Add any missing columns
                for alter_cmd in missing_columns:
                    try:
                        db_cursor.execute(f"ALTER TABLE uploads {alter_cmd}")
                        logging.info(f"Database schema updated: {alter_cmd}")
                    except sqlite3.Error as e_col:
                        logging.warning(f"Error updating database schema: {e_col}")
        except sqlite3.Error as e_check:
            logging.warning(f"Error checking database schema: {e_check}")
            # Fallback: Try to create the table anyway
            db_cursor.execute('''
                CREATE TABLE IF NOT EXISTS uploads (
                    reddit_url TEXT PRIMARY KEY,
                    youtube_url TEXT,
                    upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    title TEXT,
                    subreddit TEXT
                )
            ''')
            
        db_conn.commit()
        logging.info(f"Database {DB_FILE} setup complete.")
    except sqlite3.Error as e:
        logging.error(f"Database setup failed: {e}")
        raise RuntimeError(f"Database setup failed: {e}") from e

def close_database():
    global db_conn, db_cursor
    if db_cursor:
        try: db_cursor.close()
        except sqlite3.Error as e: logging.warning(f"Error closing DB cursor: {e}")
    if db_conn:
        try: db_conn.close()
        except sqlite3.Error as e: logging.warning(f"Error closing DB connection: {e}")
    db_conn, db_cursor = None, None
    logging.info("Database connection closed.")

def setup_api_clients():
    """Set up API clients for Reddit, YouTube and Gemini."""
    global reddit, youtube_service, gemini_model

    # Reddit
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET]):
        logging.error("Reddit Client ID or Secret is missing. Check .env file.")
        reddit = None
    else:
        try:
            logging.info(f"Setting up Reddit client with ID: {REDDIT_CLIENT_ID[:5]}...")
            reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT
            )
            username = reddit.user.me()
            logging.info(f"Reddit client initialized. Connected as: {username}")
        except Exception as e:
            logging.error(f"Reddit API setup failed: {e}")
            reddit = None

    # YouTube
    if not YOUTUBE_TOKEN_FILE_PATH:
        logging.error("YOUTUBE_TOKEN_FILE path not set in .env. Cannot initialize YouTube API.")
        youtube_service = None
    elif not os.path.exists(YOUTUBE_TOKEN_FILE_PATH):
        logging.error(f"YouTube token file '{YOUTUBE_TOKEN_FILE_PATH}' not found. Run auth script.")
        youtube_service = None
    else:
        try:
            logging.info(f"Loading YouTube token from: {YOUTUBE_TOKEN_FILE_PATH}")
            credentials = load_youtube_token(YOUTUBE_TOKEN_FILE_PATH)
            if credentials:
                youtube_service = build_google_api(
                    YOUTUBE_API_SERVICE_NAME,
                    YOUTUBE_API_VERSION,
                    credentials=credentials
                )
                channels_response = youtube_service.channels().list(part="snippet", mine=True).execute()
                if 'items' in channels_response and len(channels_response['items']) > 0:
                    channel_name = channels_response['items'][0]['snippet']['title']
                    logging.info(f"YouTube client initialized. Connected to channel: {channel_name}")
                else:
                    logging.info("YouTube client initialized but no channel found for this account.")
            else:
                youtube_service = None
        except Exception as e:
            logging.error(f"YouTube API setup failed: {e}")
            youtube_service = None

    # Gemini
    if not GEMINI_API_KEY:
        logging.warning("GEMINI_API_KEY not set. Gemini features will be disabled.")
        gemini_model = None
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel(GEMINI_MODEL_ID)
            logging.info(f"Gemini client initialized with model {GEMINI_MODEL_ID}.")
        except Exception as e:
            logging.error(f"Gemini API setup failed: {e}")
            gemini_model = None
    
    return reddit is not None, youtube_service is not None, gemini_model is not None

# --- Reddit & Content ---
def get_reddit_submissions(subreddit_name: str, limit: int) -> List[praw.models.Submission]:
    if not reddit:
        logging.warning("Reddit client not available. Cannot fetch submissions.")
        return []
    
    if subreddit_name.lower() not in [s.lower() for s in CURATED_SUBREDDITS]:
        logging.warning(f"Subreddit r/{subreddit_name} is not in the curated whitelist. Fetching anyway but content might be unsuitable.")
    
    submissions = []
    try:
        subreddit_instance = reddit.subreddit(subreddit_name)
        # Only fetch video posts (is_video or common video hosts)
        fetch_limit = min(limit * 5, 100)  # Fetch more to filter
        for submission in subreddit_instance.hot(limit=fetch_limit):
            is_link_video = any(submission.url.endswith(ext) for ext in ['.mp4', '.mov', '.gifv'])
            is_hosted_video = any(host in submission.url for host in ['v.redd.it', 'gfycat.com', 'streamable.com', 'i.imgur.com'])
            if (submission.is_video or is_link_video or is_hosted_video) and \
            not submission.over_18 and not submission.stickied:
                submissions.append(submission)
                if len(submissions) >= limit * 3:  # Get 3x what we need for safety
                    break
            time.sleep(0.2)  # Be respectful to Reddit API
        # logging.info(f"Fetched {len(submissions)} potential video submissions from r/{subreddit_name}.")
        return submissions
    except prawcore.exceptions.PrawcoreException as e:
        logging.error(f"Reddit API error fetching from r/{subreddit_name}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error fetching from r/{subreddit_name}: {e}")
    return []

def is_unsuitable_video(submission: praw.models.Submission, video_path: Optional[pathlib.Path] = None) -> Tuple[bool, str]:
    """Basic checks for unsuitable video content."""
    # Title check
    if contains_forbidden_words(submission.title):
        return True, "Title contains forbidden words"
    if any(ft.lower() in submission.title.lower() for ft in UNSUITABLE_CONTENT_TYPES):
        return True, "Title suggests unsuitable content type"

    # Comments check (with timeout)
    try:
        comment_queue = queue.Queue()
        def fetch_submission_comments():
            try:
                submission.comments.replace_more(limit=0)  # Get top-level comments only
                # Check only a few comments to avoid too many API calls / long processing
                top_comments_text = " ".join([c.body for c in submission.comments.list()[:3] if hasattr(c, 'body')])
                comment_queue.put(top_comments_text)
            except Exception as e_comm:
                comment_queue.put(e_comm)  # Put exception in queue

        comment_thread = threading.Thread(target=fetch_submission_comments, daemon=True)
        comment_thread.start()
        comment_thread.join(timeout=5.0)  # 5 second timeout

        if comment_thread.is_alive():
            logging.warning("Comment retrieval timed out.")
        else:
            comments_result = comment_queue.get_nowait()  # Should not block if thread finished or timed out
            if isinstance(comments_result, str):
                if any(ft.lower() in comments_result.lower() for ft in UNSUITABLE_CONTENT_TYPES):
                    return True, "Comments suggest unsuitable content type"
                if contains_forbidden_words(comments_result):
                    return True, "Comments contain forbidden words"
            elif isinstance(comments_result, Exception):
                logging.warning(f"Error fetching comments for suitability check: {comments_result}")

    except queue.Empty:  # Should not happen if timeout is handled
        logging.warning("Comment queue was empty after thread join (unexpected).")
    except Exception as e:
        logging.warning(f"Error during comment analysis for suitability: {e}")

    if video_path and video_path.is_file():
        try:
            duration, width, height = get_video_details(video_path)
            if duration < 2: return True, f"Video too short ({duration:.1f}s)" 
            if width < 240 or height < 240: return True, f"Video resolution too low ({width}x{height})" # Lowered from 360p to 240p
        except Exception as e: # get_video_details already logs
            logging.warning(f"Could not get video details for suitability check of {video_path}: {e}")
            # Don't fail suitability just because details couldn't be read, Gemini check might still pass
    return False, ""


def gemini_content_safety_check(submission: praw.models.Submission, video_path: Optional[pathlib.Path] = None) -> Tuple[bool, str]:
    """Uses Gemini for advanced content safety check, falling back to basic if needed."""
    global gemini_model
    
    # Basic check first - quick rejection for obvious issues
    unsuitable_basic, reason_basic = is_unsuitable_video(submission, video_path)
    if unsuitable_basic:
        return True, f"Basic Check Failed: {reason_basic}"

    if not gemini_model:
        logging.warning("Gemini model not available for safety check. Relying on basic checks only.")
        return False, "Gemini unavailable, basic check passed" # If basic check passed

    if not video_path or not video_path.is_file():
        logging.warning("Video file not available for Gemini safety check. Skipping Gemini check.")
        return False, "Video file unavailable for Gemini, basic check passed"

    try:
        logging.info(f"Performing Gemini safety check for: {submission.title[:50]}...")
        frame_paths_for_gemini: List[pathlib.Path] = []
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logging.warning(f"Could not open video {video_path} for Gemini frame extraction.")
                return False, "Video open failed for Gemini, basic check passed"
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            # Sample up to 3 frames: start, middle, end
            sample_times = [0.1 * duration, 0.5 * duration, 0.9 * duration]
            # Ensure TEMP_DIR exists for frames
            TEMP_DIR.mkdir(parents=True, exist_ok=True)

            for i, t in enumerate(sample_times):
                if t < duration: # Ensure sample time is within video
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                    ret, frame = cap.read()
                    if ret:
                        frame_path = TEMP_DIR / f"gemini_safety_frame_{submission.id}_{i}.jpg"
                        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        frame_paths_for_gemini.append(frame_path)
            cap.release()
        except Exception as e_frame:
            logging.warning(f"Error extracting frames for Gemini safety check: {e_frame}")
            # Continue without frames if extraction fails, relying on text

        prompt_parts = [
            f"Analyze this content (title, subreddit, and sample video frames if provided) for safety and suitability for a general YouTube audience, focusing on monetization guidelines. Reddit Title: \"{submission.title}\". Subreddit: r/{submission.subreddit.display_name}.",
            "Is this content problematic? Specifically check for: " + ", ".join(UNSUITABLE_CONTENT_TYPES) + ".",
            "Also check for excessive use of words like: " + ", ".join(FORBIDDEN_WORDS[:10]) + "...", # Sample of forbidden
            "Respond ONLY with a JSON object with keys: 'is_problematic' (boolean), 'reason' (string, concise explanation if problematic), 'confidence_percent' (int, 0-100, your confidence in this assessment)."
        ]

        for fp in frame_paths_for_gemini:
            if fp.is_file():
                try:
                    with open(fp, 'rb') as f:
                        img_bytes = f.read()
                    img_base64 = base64.b64encode(img_bytes).decode("utf-8") 
                    img_part = {
                        "mime_type": "image/jpeg",
                        "data": img_base64
                    }
                    # Use dictionary instead of Blob
                    prompt_parts.append(img_part)
                except Exception as e_blob:
                    logging.warning(f"Could not read/attach frame {fp} for Gemini: {e_blob}")
        
        # Stricter safety settings for the API call itself
        safety_settings_api = [
            {"category": c, "threshold": "BLOCK_ONLY_HIGH"} # Block less from API, judge from response
            for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", 
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]
        
        response = gemini_model.generate_content(
            prompt_parts,
            generation_config=genai.types.GenerationConfig(temperature=0.1, max_output_tokens=200),
            safety_settings=safety_settings_api
        )

        # Clean up frames
        for fp in frame_paths_for_gemini: cleanup_temp_files(fp)

        if not response.candidates or not hasattr(response, 'text') or not response.text:
            block_reason = response.prompt_feedback.block_reason if hasattr(response, 'prompt_feedback') else "Unknown"
            logging.warning(f"No valid response or text from Gemini for safety check. Block Reason: {block_reason}. Assuming safe for now.")
            return False, "Gemini response invalid"

        # Extract JSON from response
        json_text = response.text
        match = re.search(r"```json\s*(\{.*?\})\s*```", json_text, re.DOTALL)
        if match:
            json_text = match.group(1)
        else: # Fallback if no markdown code block
            json_start = json_text.find('{')
            json_end = json_text.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_text = json_text[json_start : json_end+1]
            else:
                logging.warning(f"Could not find JSON in Gemini safety response: '{json_text[:200]}...' Assuming safe.")
                return False, "Gemini JSON not found"

        try:
            result = json.loads(json_text)
            is_problematic = result.get("is_problematic", False)
            reason = result.get("reason", "No specific reason from Gemini.")
            confidence = result.get("confidence_percent", 0)

            logging.info(f"Gemini safety check result: Problematic={is_problematic}, Reason='{reason}', Confidence={confidence}%")
            # Increased confidence threshold from 60% to 80% to be more lenient
            if is_problematic and confidence >= 80: 
                return True, f"Gemini: {reason} (Conf: {confidence}%)"
            return False, f"Gemini: Content deemed safe (Problematic={is_problematic}, Conf: {confidence}%)"
        except json.JSONDecodeError:
            logging.warning(f"Could not parse Gemini JSON for safety check: '{json_text[:200]}...' Assuming safe.")
            return False, "Gemini JSON parse error"
            
    except Exception as e:
        logging.error(f"Error in Gemini content safety check: {e}", exc_info=True)
        return False, "Exception during Gemini safety check" # Default to safe on error to avoid false positives blocking too much

# --- AI Analysis (Content Generation) ---
def analyze_video_with_gemini(video_path: pathlib.Path, title: str, subreddit_name: str) -> Dict[str, Any]:
    global gemini_model, FALLBACK_ANALYSIS
    
    if not gemini_model:
        logging.warning("Gemini model not available for video analysis. Using fallback.")
        return {**FALLBACK_ANALYSIS, "original_duration": get_video_details(video_path)[0]}

    if not video_path.is_file():
        logging.error(f"Video path for Gemini analysis is invalid: {video_path}")
        return {**FALLBACK_ANALYSIS}

    try:
        logging.info(f"Analyzing video content with Gemini: {title[:50]}...")
        duration, width, height = get_video_details(video_path)
        if duration == 0.0: # If get_video_details failed
            logging.warning(f"Could not get duration for {video_path}. Using fallback analysis.")
            return {**FALLBACK_ANALYSIS}

        # Prepare video for Gemini (if using vision model that accepts video directly)
        # For models like gemini-1.5-flash, you can upload video files.
        # Ensure the file is not too large (check Gemini documentation for limits)
        uploaded_file = None
        video_blob = None
        try:
            if video_path.stat().st_size < 200 * 1024 * 1024: # Example: 200MB limit
                logging.info(f"Preparing video file {video_path.name} for Gemini analysis...")
                # uploaded_file = genai.upload_file(path=str(video_path), display_name=video_path.name)
                # video_blob = uploaded_file # Use this in prompt_parts
                # For direct data sending (alternative, check model support):
                with open(video_path, 'rb') as f:
                    video_data_bytes = f.read()
                video_data_base64 = base64.b64encode(video_data_bytes).decode("utf-8")
                video_part = {
                    "mime_type": "video/mp4",
                    "data": video_data_base64
                }
                # Use as dictionary instead of Blob
                video_blob = video_part
            else:
                logging.warning(f"Video file {video_path.name} too large for direct Gemini upload. Analysis will be text-based.")
        except Exception as e_upload:
            logging.error(f"Failed to prepare video for Gemini: {e_upload}")
            # Continue with text-based analysis if video upload fails

        prompt = f"""
Analyze this video content from Reddit (r/{subreddit_name}) to create elements for an engaging vertical short video (e.g., YouTube Shorts, TikTok).
Original title: \"{title}\". Video duration: {duration:.2f}s. Dimensions: {width}x{height}.

Your goal is to make the video engaging, shareable, and highly optimized for viewer retention in the first 3 seconds, which is CRITICAL for short-form content success.

IMPORTANT HOOK ANALYSIS (FIRST 3 SECONDS ARE CRITICAL):
- Identify the single most visually arresting or surprising 0.5-2 second moment in the video. This will be our visual hook. Provide its exact timestamp and a brief description.
- Generate 3 ultra-short (2-4 word) hook texts that create curiosity or highlight the unexpected element. One should be a question, one should create anticipation ("WAIT FOR IT..."), and one should be attention-grabbing.
- If possible, identify a specific audio moment (sound effect needed, music drop point, or original audio highlight) to serve as an audio hook.

NARRATIVE & PACING ANALYSIS:
- Distill the video into a clear 1-sentence story with: Setup, Inciting Incident, and Climax/Payoff.
- Identify 4-6 very specific emotional keywords (e.g., "curiosity", "surprise", "relief", "joyful anticipation") to guide music and effects.
- For event-driven videos, is the original audio essential? If so, explicitly set "original_audio_is_key": true and provide details.
- Identify micro-moments (0.5-1.5s segments) that can be rapidly cut together to build anticipation.
- Identify if any part would benefit from speed effects (slow-motion for impact, fast-forward for buildup). Provide exact timestamps and suggested speed factor (e.g., 0.5x, 2x).

VISUAL ENHANCEMENT OPPORTUNITIES:
- Instead of just finding one best segment, identify multiple segments that can be combined (max 59 seconds total).
- For text overlays, specify impactful, brief callouts timed with key visual moments. Each should add value or context.
- Identify 2-3 moments in the final clip that would be enhanced by a specific sound effect (e.g., "punch", "swoosh", "ding", "pop").
- Suggest subtle zoom/pan movements targeting specific elements of interest.

AUDIO & MUSIC RECOMMENDATIONS:
- Based on the emotional keywords, suggest 1-2 very specific music genres (e.g., "upbeat electronic with drops", "playful ukulele").
- If original audio is key, specify exact moments to duck music briefly.

Provide your response ONLY as a valid JSON object with the following structure:
{{
    "suggested_title": "string (concise, catchy, <70 chars, for YouTube)",
    "summary_for_description": "string (2-3 sentences for YouTube description, highlight key aspects)",
    "mood": "string (choose one: funny, heartwarming, informative, suspenseful, action, calm, exciting, awe-inspiring, satisfying, weird, cringe, neutral)",
    "emotional_keywords": ["string", "string", "string", "string (4-6 very specific emotional descriptors)"],
    "has_clear_narrative": "boolean (does the original video tell a story on its own?)",
    "story_structure": {{
        "setup": "string (briefly describe how video begins)",
        "inciting_incident": "string (the main event/action/turn)",
        "climax": "string (the peak moment/payoff)"
    }},
    "original_audio_is_key": "boolean (is the original audio crucial to understanding or enjoying the video?)",
    "hook_text": "string (primary hook text, ALL CAPS, max 5 words, bold statement)",
    "hook_variations": [
        "string (secondary hook option 1, ALL CAPS)",
        "string (question-based hook, ALL CAPS, with ?)",
        "string (anticipation hook, ALL CAPS, e.g., WAIT FOR IT...)"
    ],
    "visual_hook_moment": {{
        "timestamp_seconds": "float (most visually impactful/surprising moment for intro)",
        "description": "string (why this moment is visually arresting)"
    }},
    "audio_hook": {{
        "type": "string (sound_effect, music_cut, original_audio)",
        "sound_name": "string (if sound_effect: e.g., whoosh, ding, boom)",
        "timestamp_seconds": "float (when audio hook should occur in final video)"
    }},
    "segments": [ 
        {{
            "start_seconds": "float (start time in original video)",
            "end_seconds": "float (end time in original video)",
            "reason": "string (why this segment is important)"
        }}
        // Add up to 3 segments, total duration <= 59 seconds
    ],
    "best_segment": {{  // Kept for backward compatibility
        "start_seconds": "float (start time of the most engaging segment - should match first segment)",
        "end_seconds": "float (end time - if multiple segments, this is just the first one)",
        "reason": "string (why this segment is best)"
    }},
    "key_focus_points": [ // For guiding automated cropping if original is not vertical. Times relative to original video. Max 3 points.
        {{"time_seconds": "float", "point": {{"x_ratio": "float (0.0-1.0)", "y_ratio": "float (0.0-1.0)"}}}}
    ],
    "text_overlays": [ // For graphical text, NOT speech subtitles. Max 5 overlays. Times relative to final edited video.
        {{
            "text": "string (short, impactful, ALL CAPS, 1-7 words)",
            "time_seconds": "float (when it should appear, relative to final edited video)",
            "duration_seconds": "float (how long it stays, typically 2-4s)",
            "position": "string (e.g., 'center', 'top_third', 'bottom_third')",
            "background_style": "string (e.g., 'subtle_dark_box', 'none', 'bright_highlight_box')"
        }}
    ],
    "narrative_script_segments": [ // For TTS and corresponding subtitles. Max 3 segments. Times relative to final edited video.
        {{
            "text": "string (conversational, short sentence or two)",
            "time_seconds": "float (when TTS/subtitle should start, relative to final edited video)",
            "intended_duration_seconds": "float (estimated duration for this speech segment)"
        }}
    ],
    "visual_cues_suggestions": [ // For pan/zoom/effects. Max 3 suggestions. Times relative to final edited video.
        {{
            "time_seconds": "float", 
            "suggestion": "string (e.g., 'zoom_in_fast', 'zoom_out_fast', 'pan_left', 'pan_right', 'slow_motion_here', 'quick_cut')",
            "duration_seconds": "float (how long the effect should last)",
            "target_element": "string (what part of frame to focus on, if applicable)"
        }}
    ],
    "speed_effects": [ // For speed adjustments. Times relative to original video.
        {{
            "start_seconds": "float (when speed effect begins)",
            "end_seconds": "float (when speed effect ends)",
            "speed_factor": "float (e.g., 0.5 for slowmo, 2.0 for speedup)",
            "effect_type": "string (e.g., 'slowdown', 'speedup', 'freeze_frame')"
        }}
    ],
    "sound_effects": [ // For adding sound effects. Times relative to final edited video.
        {{
            "timestamp_seconds": "float (when effect should play)",
            "effect_name": "string (e.g., 'whoosh', 'ding', 'pop', 'explosion', 'punch', 'swoosh')", 
            "volume": "float (0.0-1.0 volume level)"
        }}
    ],
    "music_genres": ["string", "string (suggest 1-2 genres like 'upbeat electronic', 'cinematic orchestral', 'chill lofi')"],
    "key_audio_moments_original": [ // If original_audio_is_key is true. Times relative to final edited video.
        {{"time_seconds": "float", "description": "string (e.g., 'character laugh', 'surprising sound effect')",'action': "string (e.g., 'keep_prominent', 'briefly_mute_music')"}}
    ],
    "hashtags": ["string", "string", "string", "string", "string (5 relevant hashtags, mix of specific and broad)"],
    "thumbnail_info": {{
        "timestamp_seconds": "float (best frame for thumbnail, relative to original video start)",
        "reason": "string (why this frame)",
        "headline_text": "string (VERY short text for thumbnail, <4 words, ALL CAPS)"
    }},
    "call_to_action": {{ // Optional
        "text": "string (e.g., 'What do you think?', 'Follow for more!')",
        "type": "string (e.g., 'text_overlay_end', 'description_append', 'none')"
    }},
    "emotional_arc_description": "string (briefly: e.g., 'starts curious, builds to surprise, ends satisfying')",
    "is_explicitly_age_restricted": "boolean (should this content be formally age-restricted on YouTube? Be conservative, default to false.)"
}}
"""
        prompt_parts_list = [prompt]
        if video_blob:
            prompt_parts_list.append(video_blob)
        
        # API safety settings (less restrictive for generation, filtering happens post-response)
        generation_safety_settings = [
            {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", 
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]

        response = gemini_model.generate_content(
            prompt_parts_list,
            generation_config=genai.types.GenerationConfig(temperature=0.7, max_output_tokens=4096), # Increased tokens
            safety_settings=generation_safety_settings # Apply safety settings here
        )

        # if uploaded_file: # Clean up uploaded file if used
        #     try: genai.delete_file(uploaded_file.name); logging.info(f"Deleted Gemini uploaded file: {uploaded_file.name}")
        #     except Exception as e_del: logging.warning(f"Could not delete Gemini file {uploaded_file.name}: {e_del}")


        if not response.candidates or not hasattr(response, 'text') or not response.text:
            block_reason = response.prompt_feedback.block_reason if hasattr(response, 'prompt_feedback') else "Unknown"
            logging.warning(f"No valid response or text from Gemini for analysis. Block Reason: {block_reason}. Using fallback.")
            return {**FALLBACK_ANALYSIS, "original_duration": duration, "fallback": True}

        result_text = response.text
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            json_start = result_text.find('{')
            json_end = result_text.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_text = result_text[json_start : json_end + 1]
            else:
                logging.warning(f"Could not find JSON in Gemini analysis response. Using fallback. Response: {result_text[:300]}")
                return {**FALLBACK_ANALYSIS, "original_duration": duration, "fallback": True}
        
        try:
            parsed_data = json.loads(json_text)
            # Basic validation and default setting
            parsed_data.setdefault('suggested_title', FALLBACK_ANALYSIS['suggested_title'])
            parsed_data.setdefault('summary_for_description', FALLBACK_ANALYSIS['summary_for_description'])
            parsed_data.setdefault('mood', FALLBACK_ANALYSIS['mood'])
            parsed_data.setdefault('text_overlays', FALLBACK_ANALYSIS['text_overlays'])
            if not isinstance(parsed_data['text_overlays'], list): parsed_data['text_overlays'] = []
            
            # Ensure narrative_script_segments exists and has valid content for TTS
            parsed_data.setdefault('narrative_script_segments', FALLBACK_ANALYSIS['narrative_script_segments'])
            if not isinstance(parsed_data['narrative_script_segments'], list) or not parsed_data['narrative_script_segments']:
                parsed_data['narrative_script_segments'] = FALLBACK_ANALYSIS['narrative_script_segments']
            
            # ... add more robust validation for all expected keys and their types
            
            parsed_data['original_duration'] = duration # Ensure this is set from actual video
            parsed_data['fallback'] = False
            logging.info(f"Gemini analysis successful for: {title[:50]}")
            # logging.debug(f"Gemini Analysis Data: {json.dumps(parsed_data, indent=2)}")
            return parsed_data
        except json.JSONDecodeError as json_err:
            logging.error(f"Failed to parse Gemini JSON analysis response: {json_err}. Response: {json_text[:300]}")
            return {**FALLBACK_ANALYSIS, "original_duration": duration, "fallback": True}

    except Exception as e:
        logging.error(f"Error during Gemini analysis pipeline: {e}", exc_info=True)
        return {**FALLBACK_ANALYSIS, "original_duration": get_video_details(video_path)[0], "fallback": True}


# --- TTS ---
def generate_tts_dia(text: str, output_path: pathlib.Path) -> bool:
    """Generate TTS using suno/bark from HuggingFace (GPU-accelerated if available)."""
    try:
        device_id = 0 if CUDA_AVAILABLE else -1
        logging.info(f"TTS using device: {'CUDA GPU' if CUDA_AVAILABLE else 'CPU'}")
        
        # Set reasonable batch size for GPU memory
        batch_size = 1  # Default conservative batch size
        
        # Configure pipeline with GPU optimization settings
        pipe_kwargs = {
            'model': 'suno/bark',
            'device': device_id
        }
        
        if CUDA_AVAILABLE:
            # Try to use more efficient settings for GPU
            try:
                # Get GPU memory info if possible
                free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                free_mem_gb = free_mem / (1024 ** 3)  # Convert to GB
                
                # Adjust batch size based on available memory
                if free_mem_gb > 8:
                    batch_size = 4  # Larger batch for GPUs with more memory
                elif free_mem_gb > 4:
                    batch_size = 2  # Medium batch for medium memory
                
                logging.info(f"GPU has {free_mem_gb:.2f}GB free, using batch_size={batch_size}")
                
                # Add torch compile for speedup if torch version supports it
                if hasattr(torch, 'compile') and float(torch.__version__[:3]) >= 2.0:
                    pipe_kwargs['torch_dtype'] = torch.float16  # Use half precision
            except Exception as e:
                logging.warning(f"Could not optimize GPU settings: {e}. Using defaults.")
        
        tts = pipeline('text-to-speech', **pipe_kwargs)
        
        # Process with configured batch size
        output = tts(text, batch_size=batch_size)
        
        audio = output["audio"] if isinstance(output, dict) else output[0]["audio"]
        sampling_rate = output["sampling_rate"] if isinstance(output, dict) else output[0]["sampling_rate"]
        
        # FIX: Use ffmpeg directly to create WAV instead of using soundfile
        # Create a temporary numpy file and then convert to audio with ffmpeg
        temp_npy_file = output_path.with_suffix('.npy')
        np.save(str(temp_npy_file), audio)
        
        # Create WAV using ffmpeg
        wav_output_path = output_path.with_suffix('.wav')
        try:
            # Load the numpy array back
            audio_data = np.load(str(temp_npy_file))
            
            # Scale to 16-bit PCM range if needed
            if audio_data.dtype != np.int16:
                # Scale floating point [-1.0, 1.0] to int16 range
                if audio_data.dtype in [np.float32, np.float64]:
                    audio_data = np.int16(audio_data * 32767)
                # Or convert other types to int16
                else:
                    audio_data = audio_data.astype(np.int16)
            
            # Save as WAV using ffmpeg subprocess
            cmd = [
                'ffmpeg', '-y',
                '-f', 's16le',  # 16-bit signed little endian PCM
                '-ar', str(sampling_rate),  # Sample rate
                '-ac', '1',  # Mono audio
                '-i', 'pipe:',  # Read from stdin
                str(wav_output_path)
            ]
            
            process = subprocess.Popen(
                cmd, 
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Write audio data to ffmpeg's stdin
            process.stdin.write(audio_data.tobytes())
            process.stdin.close()
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode != 0:
                stderr = process.stderr.read().decode('utf-8')
                logging.error(f"ffmpeg error: {stderr}")
                raise RuntimeError(f"ffmpeg process failed with code {process.returncode}")
                
            # Clean up the temp numpy file
            temp_npy_file.unlink(missing_ok=True)
            
        except Exception as e:
            logging.error(f"Error creating WAV with ffmpeg: {e}", exc_info=True)
            return False
        
        # Convert WAV to MP3 using ffmpeg if needed
        if output_path.suffix.lower() == '.mp3':
            try:
                cmd = ['ffmpeg', '-y', '-i', str(wav_output_path), '-codec:a', 'libmp3lame', '-qscale:a', '2', str(output_path)]
                subprocess.run(cmd, check=True, capture_output=True)
                # Remove the temporary WAV file
                wav_output_path.unlink(missing_ok=True)
                return output_path.is_file() and output_path.stat().st_size > 1000
            except Exception as e:
                logging.error(f"Error converting WAV to MP3: {e}", exc_info=True)
                # Keep the WAV file if MP3 conversion fails
                return wav_output_path.is_file() and wav_output_path.stat().st_size > 1000
        
        return wav_output_path.is_file() and wav_output_path.stat().st_size > 1000
    except Exception as e:
        logging.error(f"Error generating TTS with suno/bark: {e}", exc_info=True)
        
        # If we got a CUDA out of memory error, try again with CPU
        if CUDA_AVAILABLE and "CUDA out of memory" in str(e):
            logging.warning("CUDA out of memory. Falling back to CPU for TTS...")
            try:
                tts = pipeline('text-to-speech', model='suno/bark', device=-1)
                output = tts(text)
                audio = output["audio"] if isinstance(output, dict) else output[0]["audio"]
                sampling_rate = output["sampling_rate"] if isinstance(output, dict) else output[0]["sampling_rate"]
                
                # FIX: Use ffmpeg directly to create WAV instead of using soundfile
                # Create a temporary numpy file and then convert to audio with ffmpeg
                temp_npy_file = output_path.with_suffix('.npy')
                np.save(str(temp_npy_file), audio)
                
                # Create WAV using ffmpeg
                wav_output_path = output_path.with_suffix('.wav')
                try:
                    # Load the numpy array back
                    audio_data = np.load(str(temp_npy_file))
                    
                    # Scale to 16-bit PCM range if needed
                    if audio_data.dtype != np.int16:
                        # Scale floating point [-1.0, 1.0] to int16 range
                        if audio_data.dtype in [np.float32, np.float64]:
                            audio_data = np.int16(audio_data * 32767)
                        # Or convert other types to int16
                        else:
                            audio_data = audio_data.astype(np.int16)
                    
                    # Save as WAV using ffmpeg subprocess
                    cmd = [
                        'ffmpeg', '-y',
                        '-f', 's16le',  # 16-bit signed little endian PCM
                        '-ar', str(sampling_rate),  # Sample rate
                        '-ac', '1',  # Mono audio
                        '-i', 'pipe:',  # Read from stdin
                        str(wav_output_path)
                    ]
                    
                    process = subprocess.Popen(
                        cmd, 
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    # Write audio data to ffmpeg's stdin
                    process.stdin.write(audio_data.tobytes())
                    process.stdin.close()
                    
                    # Wait for process to complete
                    process.wait()
                    
                    if process.returncode != 0:
                        stderr = process.stderr.read().decode('utf-8')
                        logging.error(f"ffmpeg error: {stderr}")
                        raise RuntimeError(f"ffmpeg process failed with code {process.returncode}")
                        
                    # Clean up the temp numpy file
                    temp_npy_file.unlink(missing_ok=True)
                    
                except Exception as e:
                    logging.error(f"Error creating WAV with ffmpeg: {e}", exc_info=True)
                    return False
                
                # Convert WAV to MP3 using ffmpeg if needed
                if output_path.suffix.lower() == '.mp3':
                    try:
                        cmd = ['ffmpeg', '-y', '-i', str(wav_output_path), '-codec:a', 'libmp3lame', '-qscale:a', '2', str(output_path)]
                        subprocess.run(cmd, check=True, capture_output=True)
                        # Remove the temporary WAV file
                        wav_output_path.unlink(missing_ok=True)
                        return output_path.is_file() and output_path.stat().st_size > 1000
                    except Exception as e:
                        logging.error(f"Error converting WAV to MP3: {e}", exc_info=True)
                        # Keep the WAV file if MP3 conversion fails
                        return wav_output_path.is_file() and wav_output_path.stat().st_size > 1000
                
                return wav_output_path.is_file() and wav_output_path.stat().st_size > 1000
            except Exception as e2:
                logging.error(f"CPU fallback also failed: {e2}")
        return False

def generate_tts(text: str, output_path: pathlib.Path, voice_settings: Optional[Dict] = None, voice_id: str = DEFAULT_VOICE_ID) -> bool:
    """Main TTS generation function using Dia-1.6B."""
    processed_text = enhance_text_for_tts(text)
    if not processed_text:
        return False
    if generate_tts_dia(processed_text, output_path):
        logging.info(f"Generated TTS with Dia-1.6B: {output_path.name}")
        return True
    logging.warning("Dia-1.6B TTS generation failed.")
    return False

def enhance_text_for_tts(text: str) -> str:
    if not text: return ""
    text = re.sub(r'\.(\s+)', '. ', text) # Normalize sentence spacing
    text = re.sub(r'([!?])(\s+)', r'\1 ', text)
    # Avoid excessive newlines if not using SSML
    # Consider adding pauses for SSML-capable TTS if you switch
    # text = text.replace(". ", ".<break time='500ms'/> ") # Example SSML for some engines
    return text.strip()

def select_voice_for_content(analysis: Dict) -> str: # Kept for potential future use
    mood = analysis.get('mood', 'neutral').lower()
    # Simplified: use default voice or one based on mood if specific mapping exists.
    # Add more complex voice selection logic here if needed.
    # voice_mapping = { 'funny': "some_funny_voice_id", ... }
    # if mood in voice_mapping: return voice_mapping[mood]
    return DEFAULT_VOICE_ID

def get_voice_settings_for_narrative(analysis: Dict) -> Dict: # Kept for potential future use
    mood = analysis.get('mood', 'neutral').lower()
    # Default settings can be adjusted based on mood
    settings = {"stability": 0.7, "similarity_boost": 0.75, "style": 0.2, "use_speaker_boost": True}
    if mood == 'energetic' or mood == 'exciting': settings['style'] = 0.4
    elif mood == 'calm' or mood == 'heartwarming': settings['stability'] = 0.75; settings['style'] = 0.1
    return settings

def generate_narrative_audio_segments(narrative_script_segments: List[Dict], analysis: Dict, temp_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """Generates individual audio files for narrative segments and returns info including text."""
    if not narrative_script_segments: return []
    
    audio_clips_info = [] # Will store {"file_path": ..., "start_time": ..., "duration": ..., "text": ...}
    voice_id = select_voice_for_content(analysis) 
    base_voice_settings = get_voice_settings_for_narrative(analysis)

    for i, segment_data in enumerate(narrative_script_segments):
        text = segment_data.get('text', '').strip()
        if not text: continue

        segment_start_time = float(segment_data.get('time_seconds', 0.0))
        
        segment_voice_settings = base_voice_settings.copy()
        
        segment_audio_file = temp_dir / f"narrative_seg_{i}.mp3" # Target MP3
        
        if generate_tts(text, segment_audio_file, segment_voice_settings, voice_id):
            # TTS function might save as .wav if .mp3 conversion fails, or if output_path was .wav
            # We need to check which one exists. generate_tts should ideally return the actual path.
            # For now, assume it creates output_path.mp3 or output_path.wav
            actual_segment_audio_file = segment_audio_file
            if not actual_segment_audio_file.exists():
                actual_segment_audio_file = segment_audio_file.with_suffix(".wav")
            
            if not actual_segment_audio_file.exists():
                logging.warning(f"TTS file not found for segment: '{text[:30]}...' (Checked .mp3 and .wav)")
                continue

            actual_audio_duration = get_audio_duration(actual_segment_audio_file)
            if actual_audio_duration is None or actual_audio_duration == 0.0:
                logging.warning(f"Could not get valid duration for TTS segment: {actual_segment_audio_file}. Skipping.")
                cleanup_temp_files(actual_segment_audio_file) # Clean up invalid TTS file
                continue

            audio_clips_info.append({
                "file_path": actual_segment_audio_file, # Path to the generated audio (mp3 or wav)
                "start_time": segment_start_time, 
                "duration": actual_audio_duration,
                "text": text # Store the original text for subtitle generation
            })
        else:
            logging.warning(f"Failed to generate TTS for segment: '{text[:30]}...'")
            
    return audio_clips_info


def get_audio_duration(audio_path: pathlib.Path) -> Optional[float]:
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        logging.warning(f"Audio file for duration check is invalid or empty: {audio_path}")
        return None
    try:
        # Try to get duration using ffprobe, which works for both MP3 and WAV
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        duration = float(result.stdout.strip())
        
        # Validate the duration is reasonable
        if duration <= 0:
            logging.warning(f"Got invalid duration ({duration}) for {audio_path}, using fallback method")
            raise ValueError("Invalid duration")
            
        return duration
    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired) as e:
        logging.error(f"Error getting audio duration for {audio_path}: {e}")
        
        # Fallback: try using soundfile for WAV files
        if audio_path.suffix.lower() == '.wav':
            try:
                import soundfile as sf
                info = sf.info(str(audio_path))
                return info.duration
            except Exception as sf_e:
                logging.error(f"Soundfile fallback also failed: {sf_e}")
        
        # Last resort: try to read file and calculate duration based on sample rate
        try:
            import wave
            if audio_path.suffix.lower() == '.wav':
                with wave.open(str(audio_path), 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = frames / float(rate)
                    return duration
        except Exception as wave_e:
            logging.error(f"Wave fallback also failed: {wave_e}")
            
    return None

# --- Video Processing ---
def map_gemini_text_position(gemini_pos: str, clip_w: int, clip_h: int, text_w: int, text_h: int) -> Tuple[Union[str, int], Union[str, int]]:
    """Maps Gemini text position string to MoviePy coordinates."""
    # Simple mapping, can be expanded
    gemini_pos = gemini_pos.lower()
    if gemini_pos == 'center': return ('center', 'center')
    if gemini_pos == 'top_third': return ('center', clip_h * 0.15) # Centered horizontally, in top third
    if gemini_pos == 'bottom_third': return ('center', clip_h * 0.75 - text_h / 2) # Adjusted for text height
    if gemini_pos == 'bottom_left': return (clip_w * 0.05, clip_h * 0.9 - text_h)
    if gemini_pos == 'top_right': return (clip_w * 0.95 - text_w, clip_h * 0.05)
    return ('center', 'center') # Default

def map_gemini_background_style(style_str: str) -> str:
    """Maps Gemini background style string to a color/rgba value."""
    style_str = style_str.lower()
    if style_str == 'subtle_dark_box': return 'rgba(0,0,0,0.6)'
    if style_str == 'bright_highlight_box': return 'rgba(255,255,0,0.7)' # Example: yellow highlight
    if style_str == 'none': return 'transparent'
    return GRAPHICAL_TEXT_DEFAULT_BG_COLOR # Default

def download_royalty_free_music():
    """
    Downloads or initializes royalty-free music for use in videos.
    This is a placeholder function that just logs a message.
    The actual implementation would download music files from a source.
    """
    logging.info("Skipping music download - using existing files instead")
    # Just make sure the directories exist
    MUSIC_FOLDER.mkdir(parents=True, exist_ok=True)
    for category in MUSIC_CATEGORIES:
        (MUSIC_FOLDER / category).mkdir(parents=True, exist_ok=True)
    return

def process_video_with_effects(
    source_video_path: pathlib.Path,
    analysis: Dict,
    output_path: pathlib.Path,
    temp_files_list: List[pathlib.Path],
    force_default_narrative: bool = True
) -> Tuple[bool, Dict]:
    """Processes video with AI-driven effects, TTS, music, etc."""
    if not source_video_path.is_file():
        logging.error(f"Source video for processing not found: {source_video_path}")
        return False, {}

    from moviepy.editor import ColorClip, ImageClip, CompositeVideoClip, TextClip, AudioFileClip, VideoFileClip, concatenate_videoclips, CompositeAudioClip
    import numpy as np
    import gc
    
    video_clip = None
    title_tts_audio_clip = None
    desc_tts_audio_clip = None
    intro_background = None
    main_video_clip_for_effects = None
    final_composed_video = None
    enhanced_video_clip = None
    
    try:
        logging.info(f"Starting video processing for: {source_video_path.name}")
        video_clip = VideoFileClip(str(source_video_path))
        TARGET_RESOLUTION = (1080, 1920)
        
        # Get video fps or use default for audio
        video_fps = getattr(video_clip, 'fps', 30)
        audio_fps = DEFAULT_AUDIO_FPS
        if hasattr(video_clip, 'audio') and video_clip.audio is not None:
            if hasattr(video_clip.audio, 'fps') and video_clip.audio.fps is not None:
                audio_fps = video_clip.audio.fps
        
        # --- 1. Prepare and Narrate AI Title ---
        ai_title_text = analysis.get('suggested_title', "Amazing Video Moment")
        title_tts_filename = f"{source_video_path.stem}_title_tts.mp3"
        title_tts_path = TEMP_DIR / title_tts_filename
        temp_files_list.append(title_tts_path)
        title_tts_audio_clip = None
        title_tts_duration = 3.0
        if generate_tts(ai_title_text, title_tts_path):
            try:
                title_tts_audio_clip = AudioFileClip(str(title_tts_path))
                # Set fps for audio clip to match video fps or a default value
                title_tts_audio_clip.fps = audio_fps
                title_tts_duration = title_tts_audio_clip.duration
            except Exception as e:
                logging.warning(f"Could not load title TTS audio: {e}")
                title_tts_audio_clip = None
        intro_visual_duration = title_tts_duration + 0.5
        intro_background = ColorClip(size=TARGET_RESOLUTION, color=(0,0,0), duration=intro_visual_duration)
        intro_text_overlay = TextClip(
            ai_title_text,
            fontsize=int(TARGET_RESOLUTION[1] * 0.08),
            font=GRAPHICAL_TEXT_FONT,
            color=GRAPHICAL_TEXT_COLOR,
            stroke_color=GRAPHICAL_TEXT_STROKE_COLOR,
            stroke_width=GRAPHICAL_TEXT_STROKE_WIDTH,
            method='caption',
            align='center',
            size=(int(TARGET_RESOLUTION[0] * 0.9), None)
        ).set_position('center').set_duration(intro_visual_duration).set_start(0)
        intro_segment = CompositeVideoClip([intro_background, intro_text_overlay], size=TARGET_RESOLUTION)
        if title_tts_audio_clip:
            # Set fps on title audio clip and then set it to the intro segment
            title_tts_audio_clip.fps = audio_fps
            intro_segment = intro_segment.set_audio(title_tts_audio_clip.set_start(0.25))
        
        # --- 2. Prepare Main Video Content ---
        main_video_clip_for_effects = VideoFileClip(str(source_video_path))
        original_main_video_audio = main_video_clip_for_effects.audio
        
        # --- 3. Prepare and Narrate AI Description ---
        ai_description_text = analysis.get('summary_for_description', "Watch this incredible moment unfold!")
        max_description_length = 250
        if len(ai_description_text) > max_description_length:
            ai_description_text = ai_description_text[:max_description_length].rsplit(' ',1)[0] + "..."
        desc_tts_filename = f"{source_video_path.stem}_desc_tts.mp3"
        desc_tts_path = TEMP_DIR / desc_tts_filename
        temp_files_list.append(desc_tts_path)
        desc_tts_audio_clip = None
        desc_tts_duration = 5.0
        if generate_tts(ai_description_text, desc_tts_path):
            try:
                desc_tts_audio_clip = AudioFileClip(str(desc_tts_path))
                # Set fps for audio clip to match video fps or a default value
                desc_tts_audio_clip.fps = audio_fps
                desc_tts_duration = desc_tts_audio_clip.duration
            except Exception as e:
                logging.warning(f"Could not load description TTS audio: {e}")
                desc_tts_audio_clip = None
        
        # --- 4. Combine Description TTS with Main Video's Original Audio ---
        description_start_offset_in_main_video = 0.5
        audio_tracks_for_main_video = []
        
        # Set a default audio fps
        audio_fps = 44100  # Standard audio sample rate
        
        if original_main_video_audio:
            # Set fps for original audio if needed
            if not hasattr(original_main_video_audio, 'fps') or original_main_video_audio.fps is None:
                original_main_video_audio.fps = audio_fps
            else:
                # If original audio has a valid fps, use that as our standard
                audio_fps = original_main_video_audio.fps
            audio_tracks_for_main_video.append(original_main_video_audio.volumex(0.25))
        
        if desc_tts_audio_clip:
            # Ensure desc_tts_audio_clip has fps set
            if not hasattr(desc_tts_audio_clip, 'fps') or desc_tts_audio_clip.fps is None:
                desc_tts_audio_clip.fps = audio_fps
            audio_tracks_for_main_video.append(desc_tts_audio_clip.set_start(description_start_offset_in_main_video))
        
        composed_main_audio = None
        if audio_tracks_for_main_video:
            try:
                # Ensure all audio clips have the same fps before compositing
                for clip in audio_tracks_for_main_video:
                    if not hasattr(clip, 'fps') or clip.fps is None:
                        clip.fps = audio_fps
                
                composed_main_audio = CompositeAudioClip(audio_tracks_for_main_video)
                # Ensure the composite audio has fps set
                composed_main_audio.fps = audio_fps
            except Exception as e:
                logging.error(f"Error creating composite audio: {e}")
                # Fall back to just using the original audio if available
                if original_main_video_audio:
                    composed_main_audio = original_main_video_audio
        
        main_video_clip_for_effects = main_video_clip_for_effects.set_audio(composed_main_audio)
        
        # --- 5. Adjust Main Video Visual Duration to accommodate description narration ---
        required_visual_duration_for_main = description_start_offset_in_main_video + (desc_tts_duration if desc_tts_audio_clip else 0)
        actual_visual_duration_of_main = main_video_clip_for_effects.duration
        if actual_visual_duration_of_main < required_visual_duration_for_main:
            logging.info(f"Extending main video visuals from {actual_visual_duration_of_main:.2f}s to {required_visual_duration_for_main:.2f}s for description.")
            freeze_time = max(0, actual_visual_duration_of_main - (1/main_video_clip_for_effects.fps if main_video_clip_for_effects.fps else 0.1))
            last_frame_image = main_video_clip_for_effects.get_frame(freeze_time)
            freeze_clip = ImageClip(last_frame_image).set_duration(required_visual_duration_for_main - actual_visual_duration_of_main).set_position('center')
            if composed_main_audio:
                pass
            main_video_clip_for_effects = concatenate_videoclips([
                main_video_clip_for_effects.subclip(0, actual_visual_duration_of_main),
                freeze_clip
            ])
            if composed_main_audio:
                main_video_clip_for_effects = main_video_clip_for_effects.set_audio(composed_main_audio)
        elif actual_visual_duration_of_main > required_visual_duration_for_main and required_visual_duration_for_main > 0:
            logging.info(f"Trimming main video visuals from {actual_visual_duration_of_main:.2f}s to {required_visual_duration_for_main:.2f}s.")
            main_video_clip_for_effects = main_video_clip_for_effects.subclip(0, required_visual_duration_for_main)
            
        # --- 6. Concatenate all parts (Intro + Main Video with Description) ---
        final_clips_to_concatenate = [intro_segment, main_video_clip_for_effects]
        final_composed_video = concatenate_videoclips(final_clips_to_concatenate, method="compose")
        
        # --- 6.5. Apply Seamless Looping ---
        if ENABLE_SEAMLESS_LOOPING:
            # Import the seamless looping functions
            from video_processor import optimize_for_looping, analyze_loop_compatibility
            
            logging.info("Analyzing video loop compatibility...")
            compatibility_score = analyze_loop_compatibility(
                final_composed_video, 
                sample_duration=LOOP_SAMPLE_DURATION
            )
            logging.info(f"Loop compatibility score: {compatibility_score:.3f}")
            
            if compatibility_score >= LOOP_COMPATIBILITY_THRESHOLD:
                logging.info("Applying seamless looping optimization...")
                final_composed_video = optimize_for_looping(
                    final_composed_video, 
                    target_duration=LOOP_TARGET_DURATION,
                    crossfade_duration=LOOP_CROSSFADE_DURATION
                )
                logging.info("Seamless looping applied successfully")
            else:
                logging.info(f"Skipping seamless looping: compatibility score {compatibility_score:.3f} below threshold {LOOP_COMPATIBILITY_THRESHOLD}")
        else:
            logging.info("Seamless looping disabled in configuration")
        
        # --- 7. Graphical Text Overlays (adjusted for intro duration) ---
        graphical_text_clips_final = []
        if analysis.get("text_overlays") and isinstance(analysis["text_overlays"], list):
            for overlay_data in analysis["text_overlays"]:
                text = overlay_data.get("text", "").strip()
                if not text: continue
                original_start_time = float(overlay_data.get("time_seconds", 0.0))
                adjusted_start_time = intro_segment.duration + original_start_time
                duration = float(overlay_data.get("duration_seconds", 3.0))
                if adjusted_start_time + duration > final_composed_video.duration:
                    duration = max(0.1, final_composed_video.duration - adjusted_start_time)
                if duration <=0: continue
                font_size = int(final_composed_video.h * GRAPHICAL_TEXT_FONT_SIZE_RATIO)
                gemini_pos_str = overlay_data.get("position", "center")
                bg_style_str = overlay_data.get("background_style", "subtle_dark_box")
                position_map = {
                    "center": ('center', 'center'),
                    "top_third": ('center', 0.15),
                    "bottom_third": ('center', 0.80)
                }
                position = position_map.get(gemini_pos_str.lower(), ('center', 'center'))
                bg_color = map_gemini_background_style(bg_style_str)
                try:
                    txt_clip = TextClip(
                        text,
                        fontsize=font_size,
                        font=GRAPHICAL_TEXT_FONT,
                        color=GRAPHICAL_TEXT_COLOR,
                        stroke_color=GRAPHICAL_TEXT_STROKE_COLOR,
                        stroke_width=GRAPHICAL_TEXT_STROKE_WIDTH,
                        method='caption',
                        align='center',
                        size=(final_composed_video.w * 0.9, None),
                        bg_color=bg_color
                    )
                    txt_clip = txt_clip.set_position(position, relative=True if isinstance(position[1], float) else False)
                    txt_clip = txt_clip.set_start(adjusted_start_time).set_duration(duration)
                    txt_clip = txt_clip.crossfadein(0.3).crossfadeout(0.3)
                    if text.strip().upper() in {"BOOM!", "BOOM", "BANG!", "BANG", "EXPLOSION", "BLAST!", "BLAST", "WOW!", "WOW"}:
                        # Use a safer approach for special text animation
                        try:
                            import numpy as np
                            
                            # Store the original clip for reference
                            base_txt_clip = txt_clip
                            
                            # Define a safer scale_and_shake function
                            def scale_and_shake(get_frame, t):
                                # Get the original frame
                                frame = get_frame(t)
                                if frame is None or frame.size == 0:
                                    # Return black frame of expected size as fallback
                                    h, w = base_txt_clip.size[::-1] if hasattr(base_txt_clip, 'size') else (100, 400)
                                    return np.zeros((h, w, 3), dtype=np.uint8)
                                
                                # Calculate scale factor (larger at start, normalizes quickly)
                                scale = 1.0 + 0.3 * np.exp(-10 * t)
                                
                                # Get original dimensions
                                h_orig, w_orig = frame.shape[:2]
                                
                                # Handle alpha channel conversion carefully
                                has_alpha = frame.shape[2] == 4 if len(frame.shape) > 2 else False
                                
                                if not has_alpha and len(frame.shape) > 2:
                                    # Convert BGR to BGRA
                                    frame_with_alpha = np.zeros((h_orig, w_orig, 4), dtype=np.uint8)
                                    frame_with_alpha[:,:,0:3] = frame
                                    frame_with_alpha[:,:,3] = 255  # Fully opaque
                                    frame = frame_with_alpha
                                
                                # Calculate new dimensions
                                new_w, new_h = int(w_orig * scale), int(h_orig * scale)
                                if new_w <= 0 or new_h <= 0:
                                    return frame  # Return original if scaling fails
                                
                                # Resize the frame
                                try:
                                    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                                except Exception:
                                    return frame  # Return original if resize fails
                                
                                # Calculate centering offsets
                                x1 = (new_w - w_orig) // 2
                                y1 = (new_h - h_orig) // 2
                                
                                # Create output frame with original dimensions
                                output_frame = np.zeros_like(frame)
                                
                                # Calculate valid source and destination regions
                                src_x_start = max(0, -x1)
                                src_y_start = max(0, -y1)
                                src_x_end = min(new_w, w_orig - x1)
                                src_y_end = min(new_h, h_orig - y1)
                                dst_x_start = max(0, x1)
                                dst_y_start = max(0, y1)
                                
                                # Safety checks for array copying
                                if src_x_end > src_x_start and src_y_end > src_y_start:
                                    # Calculate actual dimensions for copying
                                    actual_w = min(src_x_end - src_x_start, w_orig - dst_x_start)
                                    actual_h = min(src_y_end - src_y_start, h_orig - dst_y_start)
                                    
                                    if actual_w > 0 and actual_h > 0:
                                        try:
                                            # Copy portion of resized frame to output
                                            output_frame[dst_y_start:dst_y_start+actual_h, 
                                                        dst_x_start:dst_x_start+actual_w] = \
                                                resized_frame[src_y_start:src_y_start+actual_h, 
                                                            src_x_start:src_x_start+actual_w]
                                        except Exception:
                                            return frame  # Return original if copy fails
                                
                                # Add shake effect for the first 0.4 seconds
                                if t < 0.4:
                                    try:
                                        # Calculate displacement
                                        dx = int(8 * np.sin(50 * t))
                                        dy = int(8 * np.cos(45 * t))
                                        
                                        # Create transformation matrix
                                        M = np.float32([[1, 0, dx], [0, 1, dy]])
                                        
                                        # Apply the transformation
                                        h, w = output_frame.shape[:2]
                                        borderValue = (0,0,0,0) if output_frame.shape[2]==4 else (0,0,0)
                                        output_frame = cv2.warpAffine(
                                            output_frame, M, (w, h), 
                                            borderMode=cv2.BORDER_CONSTANT, 
                                            borderValue=borderValue
                                        )
                                    except Exception:
                                        pass  # Ignore shake effect if it fails
                                
                                return output_frame
                            
                            # Apply the transformation to the clip
                            txt_clip = txt_clip.fl(scale_and_shake, apply_to=['mask', 'video'])
                            
                        except Exception as e_anim:
                            logging.warning(f"Could not apply animation effect to '{text}': {e_anim}")
                    
                    graphical_text_clips_final.append(txt_clip)
                except Exception as e_txt:
                    logging.error(f"Could not create graphical text overlay for '{text}': {e_txt}", exc_info=True)
        
        # --- 8. Compose Final Video ---
        composite_layers = [final_composed_video] + graphical_text_clips_final
        enhanced_video_clip = CompositeVideoClip(composite_layers, size=TARGET_RESOLUTION)

        # --- 8.5. Mute Sections as requested by Gemini analysis ---
        # Check for mute ranges in analysis['key_audio_moments_original']
        mute_ranges = []
        for moment in analysis.get('key_audio_moments_original', []):
            action = moment.get('action', '').lower()
            if 'mute' in action:
                t = float(moment.get('time_seconds', 0))
                # Default mute duration: 2s if not specified, or try to parse from description
                duration = 2.0
                desc = moment.get('description', '').lower()
                # Try to extract duration from description if present (e.g., 'mute for 1.5s')
                import re
                match = re.search(r'(\d+(?:\.\d+)?)\s*s', desc)
                if match:
                    duration = float(match.group(1))
                mute_ranges.append([t, t + duration])
        if mute_ranges:
            try:
                from video_processor import mute_sections
                enhanced_video_clip = mute_sections(enhanced_video_clip, mute_ranges)
                logging.info(f"Applied muting to {len(mute_ranges)} sections as per Gemini analysis.")
            except Exception as e:
                logging.warning(f"Failed to apply mute_sections: {e}")

        # Ensure the final audio has an fps value
        if enhanced_video_clip.audio is not None:
            if not hasattr(enhanced_video_clip.audio, 'fps') or enhanced_video_clip.audio.fps is None:
                enhanced_video_clip.audio.fps = audio_fps
        
        # --- 9. Use ffmpeg directly instead of MoviePy's write_videofile to avoid bug ---
        try:
            # First export without audio
            temp_video_no_audio = TEMP_DIR / f"{output_path.stem}_no_audio.mp4"
            temp_audio_file = TEMP_DIR / f"{output_path.stem}_audio.aac"
            temp_files_list.extend([temp_video_no_audio, temp_audio_file])
            
            # Write video without audio
            enhanced_video_clip.without_audio().write_videofile(
                str(temp_video_no_audio), 
                codec=VIDEO_CODEC_CPU, 
                preset=FFMPEG_CPU_PRESET, 
                ffmpeg_params=['-crf', FFMPEG_CRF_CPU], 
                threads=os.cpu_count(),
                logger='bar'
            )
            
            # Extract audio separately - avoiding the buggy audio pipeline
            if enhanced_video_clip.audio:
                # Ensure fps is set before writing
                if not hasattr(enhanced_video_clip.audio, 'fps') or enhanced_video_clip.audio.fps is None:
                    enhanced_video_clip.audio.fps = audio_fps
                
                enhanced_video_clip.audio.write_audiofile(
                    str(temp_audio_file), 
                    codec='aac', 
                    bitrate=AUDIO_BITRATE,
                    fps=enhanced_video_clip.audio.fps,  # Explicitly pass fps
                    logger='bar'
                )
                
                # Combine video and audio with direct ffmpeg command
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(temp_video_no_audio),
                    '-i', str(temp_audio_file),
                    '-c:v', 'copy',  # Copy video stream without re-encoding
                    '-c:a', 'copy',  # Copy audio stream without re-encoding
                    '-shortest',     # End when shortest input stream ends
                    str(output_path)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            else:
                # Just copy the video file if no audio
                shutil.copy(str(temp_video_no_audio), str(output_path))
                
            logging.info(f"Successfully created enhanced video at {output_path}")
            return True, {}
        except Exception as e_ffmpeg:
            logging.error(f"Error during ffmpeg muxing: {e_ffmpeg}")
            # Fall back to direct copy if ffmpeg muxing fails
            if temp_video_no_audio.exists() and temp_video_no_audio.stat().st_size > 1000:
                shutil.copy(str(temp_video_no_audio), str(output_path))
                logging.warning(f"Copied video without audio as fallback to {output_path}")
                return True, {}
            return False, {}
            
    except Exception as e:
        logging.error(f"Error during video processing pipeline: {e}", exc_info=True)
        try:
            if video_clip: video_clip.close()
        except Exception: pass
        gc.collect()
        return False, {}
    finally:
        # Cleanup resources
        try:
            if video_clip: video_clip.close()
            if title_tts_audio_clip: title_tts_audio_clip.close()
            if desc_tts_audio_clip: desc_tts_audio_clip.close()
            if intro_background: intro_background.close()
            if main_video_clip_for_effects: main_video_clip_for_effects.close()
            if final_composed_video: final_composed_video.close()
            if enhanced_video_clip: enhanced_video_clip.close()
        except Exception as e:
            logging.warning(f"Error during resource cleanup: {e}")
        gc.collect()

def generate_custom_thumbnail(video_path: pathlib.Path, analysis: Dict, thumbnail_path: pathlib.Path) -> pathlib.Path:
    """Generates a custom thumbnail for the video with text overlay."""
    if not video_path.is_file():
        logging.error(f"Video file not found for thumbnail generation: {video_path}")
        return thumbnail_path

    try:
        import cv2
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Make sure the parent directory exists
        os.makedirs(thumbnail_path.parent, exist_ok=True)
        
        # Get thumbnail timestamp from analysis or use a default
        thumbnail_info = analysis.get('thumbnail_info', {})
        timestamp = float(thumbnail_info.get('timestamp_seconds', 0.5))
        headline_text = thumbnail_info.get('headline_text', '').strip()
        
        if not headline_text and 'suggested_title' in analysis:
            # Create headline from title if none provided
            headline_parts = analysis['suggested_title'].split(':')
            if len(headline_parts) > 1:
                headline_text = headline_parts[0].strip()
            else:
                # Just use the first few words
                words = analysis['suggested_title'].split()
                headline_text = ' '.join(words[:min(5, len(words))])
        
        # Ensure we have some text
        if not headline_text:
            headline_text = "Amazing Moment!"
            
        # Extract the frame using OpenCV
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"Could not open video for thumbnail extraction: {video_path}")
            # Fall back to simple alternative
            return _generate_fallback_thumbnail(video_path, thumbnail_path)
            
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = video_frame_count / video_fps if video_fps > 0 else 0
        
        # Validate timestamp is within video bounds
        if timestamp >= video_duration:
            timestamp = video_duration * 0.5  # Use middle of video if timestamp is invalid
            
        # Set position to the timestamp
        frame_position = int(timestamp * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        
        # Read the frame
        success, frame = cap.read()
        cap.release()
        
        if not success or frame is None:
            logging.error(f"Could not extract frame at {timestamp}s from video.")
            return _generate_fallback_thumbnail(video_path, thumbnail_path)
        
        # Simple basic thumbnail using OpenCV only (fallback if PIL fails)
        cv2.imwrite(str(thumbnail_path) + ".backup.jpg", frame)
            
        try:
            # Convert from BGR to RGB (OpenCV uses BGR, PIL uses RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create PIL Image from NumPy array
            pil_img = Image.fromarray(frame_rgb)
            
            # Resize to 1280x720 (standard YouTube thumbnail size)
            thumbnail_size = (1280, 720)
            pil_img = pil_img.resize(thumbnail_size, Image.LANCZOS)
            
            # Add semi-transparent overlay for better text visibility
            overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Add gradient overlay at the bottom (more visible text area)
            gradient_height = int(pil_img.height * 0.4)
            for y in range(gradient_height):
                # Calculate alpha (opacity) - more transparent at top, more opaque at bottom
                alpha = int(180 * (y / gradient_height))
                draw.rectangle(
                    [(0, pil_img.height - gradient_height + y), (pil_img.width, pil_img.height - gradient_height + y + 1)],
                    fill=(0, 0, 0, alpha)
                )
                
            # Try to load a nice font, fall back to default if not available
            try:
                # Try to load a custom font if available
                font_path = pathlib.Path(__file__).parent / "fonts" / "Impact.ttf"
                if not font_path.exists():
                    # Try default system font locations
                    system_fonts = [
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                        "C:\\Windows\\Fonts\\Arial.ttf",  # Windows
                        "/Library/Fonts/Arial.ttf"  # macOS
                    ]
                    for sys_font in system_fonts:
                        if os.path.exists(sys_font):
                            font_path = sys_font
                            break
                    
                # If we found a font, use it
                if font_path and os.path.exists(font_path):
                    font_large = ImageFont.truetype(str(font_path), size=80)
                    font_small = ImageFont.truetype(str(font_path), size=40)
                else:
                    # Fall back to default
                    font_large = ImageFont.load_default()
                    font_small = ImageFont.load_default()
            except Exception as e:
                logging.warning(f"Error loading font: {e}. Using default.")
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            # Wrap text to fit in the thumbnail
            def wrap_text(text, font, max_width):
                lines = []
                words = text.split()
                current_line = []
                
                for word in words:
                    test_line = ' '.join(current_line + [word])
                    # Use textbbox for newer PIL versions, fallback to textlength
                    if hasattr(draw, 'textbbox'):
                        bbox = draw.textbbox((0, 0), test_line, font=font)
                        text_width = bbox[2] - bbox[0]
                    elif hasattr(draw, 'textlength'):
                        text_width = draw.textlength(test_line, font=font)
                    else:
                        # Very basic estimation if nothing else works
                        text_width = len(test_line) * font.size * 0.6
                    
                    if text_width <= max_width:
                        current_line.append(word)
                    else:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        
                if current_line:
                    lines.append(' '.join(current_line))
                    
                return lines
                
            # Get wrapped text lines
            max_text_width = pil_img.width * 0.85
            headline_lines = wrap_text(headline_text.upper(), font_large, max_text_width)
            
            # Draw text on overlay
            text_y = pil_img.height - 30 - (len(headline_lines) * 90)
            
            # Draw each line of headline text
            for line in headline_lines:
                # Calculate text width to center it
                if hasattr(draw, 'textbbox'):
                    bbox = draw.textbbox((0, 0), line, font=font_large)
                    text_width = bbox[2] - bbox[0]
                elif hasattr(draw, 'textlength'):
                    text_width = draw.textlength(line, font=font_large)
                else:
                    text_width = len(line) * font_large.size * 0.6
                    
                text_x = (pil_img.width - text_width) // 2
                
                # Draw text shadow/outline for better visibility
                for offset_x, offset_y in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                    draw.text((text_x + offset_x, text_y + offset_y), line, font=font_large, fill=(0, 0, 0, 255))
                    
                # Draw main text
                draw.text((text_x, text_y), line, font=font_large, fill=(255, 255, 255, 255))
                text_y += 90
            
            # Add attribution text at the bottom
            attribution = "Created by Amazing AI Video Maker"
            if hasattr(draw, 'textbbox'):
                bbox = draw.textbbox((0, 0), attribution, font=font_small)
                text_width = bbox[2] - bbox[0]
            elif hasattr(draw, 'textlength'):
                text_width = draw.textlength(attribution, font=font_small)
            else:
                text_width = len(attribution) * font_small.size * 0.6
                
            draw.text(
                (pil_img.width - text_width - 20, pil_img.height - 60),
                attribution,
                font=font_small,
                fill=(200, 200, 200, 220)
            )
            
            # Composite the image with the overlay
            pil_img = pil_img.convert("RGBA")
            thumbnail_img = Image.alpha_composite(pil_img, overlay)
            
            # Convert back to RGB for saving as JPEG
            thumbnail_img = thumbnail_img.convert("RGB")
            
            # Save the thumbnail
            thumbnail_img.save(str(thumbnail_path), format="JPEG", quality=95)
            
            # Verify the thumbnail was created successfully
            if thumbnail_path.exists() and thumbnail_path.stat().st_size > 0:
                logging.info(f"Successfully created custom thumbnail at: {thumbnail_path}")
                return thumbnail_path
            else:
                logging.error(f"Failed to save thumbnail to: {thumbnail_path}")
                # Use the backup we created
                if os.path.exists(str(thumbnail_path) + ".backup.jpg"):
                    shutil.copy(str(thumbnail_path) + ".backup.jpg", str(thumbnail_path))
                    logging.warning(f"Using basic backup thumbnail instead")
                    return thumbnail_path
        except Exception as pil_error:
            logging.error(f"Error during PIL thumbnail processing: {pil_error}")
            # Use the backup we created with OpenCV
            if os.path.exists(str(thumbnail_path) + ".backup.jpg"):
                shutil.copy(str(thumbnail_path) + ".backup.jpg", str(thumbnail_path))
                logging.warning(f"Using basic backup thumbnail instead")
                return thumbnail_path
            
        return thumbnail_path
        
    except Exception as e:
        logging.error(f"Error creating custom thumbnail: {e}", exc_info=True)
        return _generate_fallback_thumbnail(video_path, thumbnail_path)

def _generate_fallback_thumbnail(video_path: pathlib.Path, thumbnail_path: pathlib.Path) -> pathlib.Path:
    """Generate a basic thumbnail using multiple fallback methods."""
    # Ensure output directory exists
    os.makedirs(thumbnail_path.parent, exist_ok=True)
    
    # Try multiple methods to generate a thumbnail
    
    # 1. Direct OpenCV method
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        # Try to get a frame from 25% into the video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 4)
        success, frame = cap.read()
        cap.release()
        
        if success and frame is not None:
            # Save the raw frame as a thumbnail
            cv2.imwrite(str(thumbnail_path), frame)
            if thumbnail_path.exists():
                logging.warning(f"Created basic thumbnail as fallback at: {thumbnail_path}")
                return thumbnail_path
    except Exception as e:
        logging.warning(f"OpenCV fallback thumbnail failed: {e}")
    
    # 2. Try using ffmpeg directly
    try:
        import subprocess
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-ss', '00:00:01.000',
            '-vframes', '1',
            str(thumbnail_path)
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=30)
        
        if thumbnail_path.exists():
            logging.warning(f"Created thumbnail using ffmpeg as fallback")
            return thumbnail_path
    except Exception as e:
        logging.warning(f"ffmpeg fallback thumbnail failed: {e}")
    
    # 3. Try using MoviePy as last resort
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(str(video_path))
        clip.save_frame(str(thumbnail_path), t=1.0)
        clip.close()
        
        if thumbnail_path.exists():
            logging.warning(f"Created thumbnail using MoviePy as fallback")
            return thumbnail_path
    except Exception as e:
        logging.warning(f"MoviePy fallback thumbnail failed: {e}")
    
    # 4. Ultimate fallback - create a blank image with text
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (1280, 720), color=(0, 20, 40))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        video_name = video_path.stem
        draw.text((640, 360), f"Thumbnail for: {video_name}", fill=(255, 255, 255), anchor="mm", font=font)
        img.save(str(thumbnail_path))
        
        logging.warning(f"Created blank thumbnail as ultimate fallback")
        return thumbnail_path
    except Exception as e:
        logging.error(f"All thumbnail generation attempts failed: {e}")
        
    return thumbnail_path

def execute_upload_with_retries(insert_request, video_file_path, thumbnail_file_path, youtube, max_retries=5):
    """
    Execute a YouTube upload request with proper progress tracking and retry logic.
    
    Args:
        insert_request: The YouTube API insert request
        video_file_path: Path to the video file
        thumbnail_file_path: Path to the thumbnail file (optional)
        youtube: The YouTube API service
        max_retries: Maximum number of retry attempts
        
    Returns:
        str: YouTube video URL if successful, None otherwise
    """
    retry_delay_base = 10  # seconds
    
    # Execute the upload with retries
    response = None
    last_exception = None
    
    for retry in range(max_retries):
        try:
            logging.info(f"Starting video upload (attempt {retry+1}/{max_retries})...")
            
            # Reset response for each retry
            response = None
            progress_status = None
            
            # Use resumable upload with progress reporting
            while response is None:
                try:
                    status, response = insert_request.next_chunk()
                    if status:
                        progress = int(status.progress() * 100)
                        # Only log if progress has changed significantly
                        if progress_status is None or progress >= progress_status + 10:
                            logging.info(f"Upload progress: {progress}%")
                            progress_status = progress
                except ConnectionError as conn_err:
                    # Handle connection-related errors with backoff retry
                    retry_delay = retry_delay_base * (2 ** retry)  # Exponential backoff
                    logging.warning(f"Connection error during upload: {conn_err}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue  # Retry current chunk
                except google_api_errors.HttpError as http_err:
                    if http_err.resp.status in (500, 502, 503, 504, 408, 429):  # Server errors or rate limiting
                        retry_delay = retry_delay_base * (2 ** retry)
                        logging.warning(f"Server error during upload (HTTP {http_err.resp.status}). Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue  # Retry current chunk
                    else:
                        # Re-raise for other HTTP errors
                        raise
                except ssl.SSLError as ssl_err:
                    retry_delay = retry_delay_base * (2 ** retry)
                    logging.warning(f"SSL error during upload: {ssl_err}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue  # Retry current chunk
                except Exception as chunk_err:
                    # For other errors, retry the chunk with backoff
                    retry_delay = retry_delay_base * (1 + retry)
                    logging.warning(f"Error during upload chunk: {chunk_err}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue  # Retry current chunk
            
            # If we got a response, break out of the retry loop
            if response:
                break
            
        except Exception as e:
            last_exception = e
            retry_delay = retry_delay_base * (2 ** retry)  # Exponential backoff
            logging.warning(f"Upload attempt {retry+1} failed: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
    
    # If all retries failed, log the error and return None
    if not response:
        if last_exception:
            logging.error(f"All upload attempts failed. Last error: {last_exception}")
        else:
            logging.error("All upload attempts failed with no specific error.")
        return None
    
    # Upload successful
    video_id = response.get("id")
    
    if not video_id:
        logging.error("Upload response missing video ID.")
        return None
    
    logging.info(f"Video uploaded successfully! Video ID: {video_id}")
    
    # Upload thumbnail if provided
    if thumbnail_file_path and os.path.exists(thumbnail_file_path):
        try:
            max_thumbnail_retries = 3
            retry_delays = [5, 15, 30]  # Seconds between retries
            
            for thumbnail_attempt in range(max_thumbnail_retries):
                try:
                    # Check thumbnail file size and validity
                    if os.path.getsize(thumbnail_file_path) == 0:
                        logging.warning("Thumbnail file is empty (0 bytes). Skipping thumbnail upload.")
                        break
                        
                    with open(thumbnail_file_path, "rb") as thumbnail_file:
                        # Determine mime type based on extension
                        if str(thumbnail_file_path).lower().endswith(('.jpg', '.jpeg')):
                            mime_type = "image/jpeg"
                        elif str(thumbnail_file_path).lower().endswith('.png'):
                            mime_type = "image/png"
                        else:
                            mime_type = "image/jpeg"  # Default to JPEG
                            
                        youtube.thumbnails().set(
                            videoId=video_id,
                            media_body=MediaFileUpload(
                                str(thumbnail_file_path),
                                mimetype=mime_type,
                                resumable=True
                            )
                        ).execute()
                        
                    logging.info(f"Thumbnail uploaded successfully for video {video_id}")
                    break
                except Exception as thumb_err:
                    if thumbnail_attempt < max_thumbnail_retries - 1:
                        delay = retry_delays[thumbnail_attempt]
                        logging.warning(f"Thumbnail upload error (attempt {thumbnail_attempt+1}/{max_thumbnail_retries}): {thumb_err}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logging.warning(f"Failed to upload thumbnail after {max_thumbnail_retries} attempts: {thumb_err}")
        except Exception as e:
            logging.warning(f"Error during thumbnail upload process: {e}")
            # Continue as thumbnail is optional
    
    # Success! Return the video URL
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    logging.info(f"Video published successfully: {video_url}")
    return video_url

def upload_to_youtube(
    video_file_path: pathlib.Path,
    title: str,
    description: str,
    thumbnail_file_path: Optional[pathlib.Path] = None,
    tags: Optional[List[str]] = None,
    publish_time: Optional[datetime] = None,
    privacy_status: str = "public"
) -> Optional[str]:
    """
    Upload a video to YouTube using the YouTube Data API.
    
    Args:
        video_file_path: Path to the video file
        title: Video title
        description: Video description
        thumbnail_file_path: Path to thumbnail image (optional)
        tags: List of tags (optional)
        publish_time: Scheduled publish time (optional)
        privacy_status: Privacy status (default: "public")
        
    Returns:
        str: YouTube video URL if successful, None otherwise
    """
    if not os.path.exists(video_file_path):
        logging.error(f"Video file does not exist: {video_file_path}")
        return None
        
    try:
        # Check for token file
        if not YOUTUBE_TOKEN_FILE:
            logging.error("YouTube token file path not specified.")
            return None
            
        if not os.path.exists(YOUTUBE_TOKEN_FILE):
            logging.error(f"YouTube token file not found: {YOUTUBE_TOKEN_FILE}")
            return None
            
        # Load credentials
        credentials = load_youtube_token(YOUTUBE_TOKEN_FILE)
        if not credentials:
            logging.error("Failed to load YouTube credentials.")
            return None
            
        # Create an authorized HTTP object
        http = httplib2.Http(timeout=120)  # 2-minute timeout
        
        # Fix: Modern oauth2 credentials don't use authorize method
        # Instead, create authorized HTTP directly:
        authorized_request = google.auth.transport.requests.Request()
        credentials.refresh(authorized_request)
        
        # Build the YouTube service
        youtube = build_google_api(
            YOUTUBE_API_SERVICE_NAME,
            YOUTUBE_API_VERSION,
            credentials=credentials
        )
            
        # Create body for the video insert request
        body = {
            "snippet": {
                "title": title[:100],  # YouTube title limit is 100 chars
                "description": description[:5000],  # YouTube description limit
                "tags": tags[:500] if tags else [],  # YouTube has a limit on total tag characters
                "categoryId": YOUTUBE_UPLOAD_CATEGORY_ID,
                "defaultLanguage": "en",
                "defaultAudioLanguage": "en"
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": False,
                "embeddable": True,
                "license": "youtube"
            }
        }
        
        # Add scheduled publishing if specified
        if publish_time:
            body["status"]["publishAt"] = publish_time.isoformat() + "Z"
            
        # Create the upload request
        try:
            # Initialize MediaFileUpload object with chunksize
            media = MediaFileUpload(
                str(video_file_path),
                mimetype="video/mp4",
                resumable=True,
                chunksize=1024*1024*5  # 5MB chunks
            )
            
            # Create insert request
            insert_request = youtube.videos().insert(
                part=",".join(body.keys()),
                body=body,
                media_body=media,
                notifySubscribers=True
            )
            
            # Execute upload with progress tracking and retry logic
            return execute_upload_with_retries(insert_request, video_file_path, thumbnail_file_path, youtube)
            
        except google_api_errors.HttpError as e:
            logging.error(f"HTTP error occurred during YouTube upload: {e}")
            error_content = json.loads(e.content.decode('utf-8'))
            error_reason = error_content.get('error', {}).get('errors', [{'reason': 'unknown'}])[0].get('reason')
            
            if error_reason == 'quotaExceeded':
                logging.critical("YouTube API quota exceeded. Try again tomorrow.")
            elif error_reason == 'authError':
                logging.critical("YouTube authentication error. Token may be expired or invalid.")
            
            return None
            
        except (socket.timeout, socket.error, ssl.SSLError, ConnectionError) as e:
            logging.error(f"Network error during YouTube upload: {e}")
            return None
            
    except Exception as e:
        logging.error(f"Error uploading to YouTube: {e}", exc_info=True)
        return None

def add_upload_record(reddit_url: str, youtube_url: str, title: str, subreddit: str):
    """Adds an upload record to the database."""
    if not db_conn:
        logging.error("Database connection not available. Cannot add upload record.")
        return

    try:
        db_cursor.execute("INSERT INTO uploads (reddit_url, youtube_url, title, subreddit) VALUES (?, ?, ?, ?) ON CONFLICT(reddit_url) DO UPDATE SET youtube_url=excluded.youtube_url, title=excluded.title, subreddit=excluded.subreddit", (reddit_url, youtube_url, title, subreddit))
        db_conn.commit()
        logging.info(f"Upload record added for {reddit_url}")
    except Exception as e:
        logging.error(f"Error adding upload record: {e}")
        db_conn.rollback()

def set_video_self_certification(video_id: str, is_age_restricted: bool):
    """
    Set video self certification status (age restriction and content rating).
    
    Args:
        video_id: YouTube video ID
        is_age_restricted: Whether video should be age restricted
    """
    if not video_id:
        logging.error("Cannot set self certification: No video ID provided")
        return
    
    try:
        # Load YouTube credentials
        credentials = load_youtube_token(YOUTUBE_TOKEN_FILE)
        if not credentials:
            logging.error("Failed to load YouTube credentials for self certification.")
            return
        
        # Create an authorized HTTP object using the new method
        authorized_request = google.auth.transport.requests.Request()
        credentials.refresh(authorized_request)
        
        # Build the YouTube service
        youtube = build_google_api(
            YOUTUBE_API_SERVICE_NAME,
            YOUTUBE_API_VERSION,
            credentials=credentials
        )
        
        # Define the self certification payload
        self_certification_body = {
            "contentAttributes": {
                "youtubeStandardGuidelines": {
                    "harmful": False,
                    "hatefulAbusive": False,
                    "dangerous": False,
                    "sexualContent": False,
                    "violent": False,
                    "childProtection": False,
                    "privacyGuidelines": False
                },
                "ageRestrictions": {
                    "restricted": is_age_restricted
                },
                "contentRatings": {
                    "mpaaNone": True,
                    "tvpgNone": True,
                    "ytRatingNone": True
                }
            }
        }
        
        # Execute the API call
        response = youtube.videos().setContentAttributes(
            videoId=video_id,
            body=self_certification_body
        ).execute()
        
        logging.info(f"Successfully set self certification for video {video_id}. Age restricted: {is_age_restricted}")
        return response
        
    except google_api_errors.HttpError as e:
        status_code = e.resp.status if hasattr(e, 'resp') else None
        
        if status_code == 403:
            logging.warning(f"Permission denied (HTTP 403) when setting self certification. This may be normal if your account doesn't have this permission.")
        else:
            logging.error(f"HTTP error setting video self certification: {e}")
    except Exception as e:
        logging.error(f"Error setting video self certification: {e}")

def get_music_attribution(music_meta: Dict) -> str:
    """Generates a music attribution string based on the music metadata."""
    if not music_meta:
        return ""
    music_genre = music_meta.get('music_genres', [''])[0]
    if not music_genre:
        return ""
    return f"\n\nMusic: {music_genre} - {music_meta.get('summary_for_description', '')}"

def ensure_default_sound_effects():
    """Downloads or creates a basic set of sound effects if they don't exist."""
    if not SOUND_EFFECTS_FOLDER.exists():
        SOUND_EFFECTS_FOLDER.mkdir(parents=True, exist_ok=True)
        
    # List of common sound effects URLs (from freesound.org or other royalty-free sources)
    default_effects = {
        "whoosh.mp3": "https://freesound.org/data/previews/553/553384_9677479-lq.mp3",
        "ding.mp3": "https://freesound.org/data/previews/337/337049_3301583-lq.mp3",
        "pop.mp3": "https://freesound.org/data/previews/533/533034_11836123-lq.mp3",
        "punch.mp3": "https://freesound.org/data/previews/376/376954_6935166-lq.mp3",
        "boing.mp3": "https://freesound.org/data/previews/316/316921_5385832-lq.mp3",
        "boom.mp3": "https://freesound.org/data/previews/33/33637_92045-lq.mp3",
        "beep.mp3": "https://freesound.org/data/previews/198/198841_285977-lq.mp3"
    }
    
    effects_count = 0
    for filename, url in default_effects.items():
        effect_path = SOUND_EFFECTS_FOLDER / filename
        if not effect_path.exists():
            try:
                import requests
                logging.info(f"Downloading sound effect: {filename}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                with open(effect_path, 'wb') as f:
                    f.write(response.content)
                effects_count += 1
            except Exception as e:
                logging.warning(f"Failed to download sound effect {filename}: {e}")
                
    if effects_count > 0:
        logging.info(f"Downloaded {effects_count} default sound effects")
    else:
        logging.info("No new sound effects were downloaded")

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Reddit to YouTube Shorts Pipeline")
    parser.add_argument('--test', action='store_true', help='Run in test mode (setup and system tests only)')
    parser.add_argument('--subreddit', type=str, default=None, help='Specify a single subreddit to process (default: random from curated list)')
    parser.add_argument('--skip-download', action='store_true', help='Skip video download step (for debugging)')
    parser.add_argument('--skip-safety-check', action='store_true', help='Skip Gemini content safety check (for debugging)')
    parser.add_argument('--sleep', type=int, default=60, help='Sleep time (seconds) between cycles (default: 60)')
    return parser.parse_args()

def run_system_tests():
    logging.info("System tests are not implemented. Skipping.")
    return True

def is_already_uploaded(reddit_permalink):
    """
    Check if a Reddit post has already been processed and uploaded to YouTube.
    
    Args:
        reddit_permalink: The Reddit post permalink to check
        
    Returns:
        bool: True if already uploaded, False otherwise
    """
    if not db_conn or not db_cursor:
        logging.warning("Database connection not available. Cannot check upload status.")
        return False
    
    try:
        # Check if the permalink exists in the uploads table
        db_cursor.execute("SELECT youtube_url FROM uploads WHERE reddit_url = ?", (reddit_permalink,))
        result = db_cursor.fetchone()
        
        if result:
            youtube_url = result[0]
            logging.info(f"Already uploaded: {reddit_permalink} → {youtube_url}")
            return True
        return False
    except sqlite3.Error as e:
        logging.error(f"Database error checking upload status: {e}")
        return False  # Better to assume not uploaded than to skip if DB is having issues
    except Exception as e:
        logging.error(f"Unexpected error checking upload status: {e}")
        return False

def clean_temp_files(path=None):
    """
    Clean temporary files either at a specific path or in the default temp directory.
    
    Args:
        path: Specific file/directory to clean, or None to clean default temp dirs
    """
    if path:
        # Clean specific file or directory
        path_obj = pathlib.Path(path)
        try:
            if path_obj.is_file():
                # Try to delete file, with multiple retries for Windows "file in use" issues
                for attempt in range(3):
                    try:
                        path_obj.unlink()
                        logging.debug(f"Cleaned up temp file: {path_obj}")
                        break
                    except PermissionError:
                        # On Windows, sometimes files are still in use - wait and retry
                        if attempt < 2:
                            time.sleep(0.5)
                        else:
                            logging.warning(f"Could not delete temp file after retries: {path_obj}")
                    except OSError as e:
                        logging.warning(f"Could not delete temp file {path_obj}: {e}")
                        break
            elif path_obj.is_dir():
                # Try to delete directory tree with shutil
                try:
                    shutil.rmtree(path_obj, ignore_errors=True)
                    logging.debug(f"Cleaned up temp directory: {path_obj}")
                except OSError as e:
                    logging.warning(f"Could not delete temp directory {path_obj}: {e}")
        except Exception as e:
            logging.warning(f"Error during temp file cleanup of {path}: {e}")
        return
    
    # If no specific path provided, clean default temp directories
    temp_paths_to_clean = [
        TEMP_DIR,  # Main temp processing directory
        pathlib.Path(tempfile.gettempdir()) / "reddit_videos"  # Directory for downloaded videos
    ]
    
    for temp_dir in temp_paths_to_clean:
        if not temp_dir.exists():
            continue
            
        try:
            # Get all files in the temp directory
            for temp_file in temp_dir.glob("*"):
                # Skip important permanent files
                if any(pattern in str(temp_file) for pattern in ['.gitkeep', 'README', '.git']):
                    continue
                    
                # Delete files older than 24 hours
                try:
                    file_age = time.time() - temp_file.stat().st_mtime
                    if file_age > (24 * 3600):  # 24 hours in seconds
                        if temp_file.is_file():
                            try:
                                temp_file.unlink()
                                logging.debug(f"Cleaned old temp file: {temp_file}")
                            except OSError as e:
                                if 'being used by another process' in str(e):
                                    logging.debug(f"File in use, skipping: {temp_file}")
                                else:
                                    logging.warning(f"Could not delete old temp file {temp_file}: {e}")
                        elif temp_file.is_dir():
                            try:
                                shutil.rmtree(temp_file, ignore_errors=True)
                                logging.debug(f"Cleaned old temp directory: {temp_file}")
                            except OSError as e:
                                logging.warning(f"Could not delete old temp directory {temp_file}: {e}")
                except OSError:
                    # Ignore errors accessing file stats
                    pass
            
            # Log summary
            remaining_files = list(temp_dir.glob("*"))
            if remaining_files:
                logging.debug(f"Temp directory {temp_dir} has {len(remaining_files)} files remaining after cleanup")
            else:
                logging.debug(f"Temp directory {temp_dir} is now empty")
                
        except Exception as e:
            logging.warning(f"Error during temp directory cleanup of {temp_dir}: {e}")
    
    # Run garbage collection to free memory
    gc.collect()

def create_youtube_token_from_env():
    """Create YouTube token file from environment variables (for Colab)"""
    try:
        # Check if we have token data in YOUTUBE_TOKEN_JSON
        token_json = os.environ.get('YOUTUBE_TOKEN_JSON')
        if token_json:
            logging.info("Found YOUTUBE_TOKEN_JSON in environment, creating token file...")
            token_data = json.loads(token_json)
            
            # Ensure directory exists
            token_file = os.environ.get('YOUTUBE_TOKEN_FILE', 'youtube_token.json')
            token_path = pathlib.Path(token_file)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write token file
            with open(token_path, 'w') as f:
                json.dump(token_data, f, indent=2)
            
            logging.info(f"YouTube token file created: {token_path}")
            return str(token_path)
            
    except Exception as e:
        logging.error(f"Error creating token file from environment: {e}")
    
    return None

def load_youtube_token(token_file_path):
    """Load YouTube OAuth token from a JSON file and return google.oauth2.credentials.Credentials object."""
    import json
    import os
    
    # First try to create token from environment if file doesn't exist
    if not token_file_path or not os.path.exists(token_file_path):
        logging.warning(f"YouTube token file not found: {token_file_path}")
        
        # Try to create from environment
        created_token_file = create_youtube_token_from_env()
        if created_token_file:
            token_file_path = created_token_file
        else:
            logging.error(f"YouTube token file '{token_file_path}' not found. Run auth script.")
            return None
    
    try:
        with open(token_file_path, 'r') as f:
            token_data = json.load(f)
        
        # Check if this is a client secrets file or a token file
        if 'installed' in token_data:
            # This is a client secrets file, not a token file
            logging.error(f"The file {token_file_path} is a client secrets file, not a token file.")
            logging.error("Please authorize your YouTube account first and save the token.")
            return None
        
        # Check for required fields
        required_fields = ['token', 'refresh_token', 'token_uri', 'client_id', 'client_secret', 'scopes']
        missing_fields = [field for field in required_fields if field not in token_data]
        
        if missing_fields:
            logging.error(f"Token file missing required fields: {', '.join(missing_fields)}")
            # If missing fields are minimal, try to set defaults
            if 'token_uri' in missing_fields:
                token_data['token_uri'] = 'https://oauth2.googleapis.com/token'
                logging.warning("Added default token_uri to credentials")
            
            if 'scopes' in missing_fields:
                token_data['scopes'] = YOUTUBE_SCOPES
                logging.warning(f"Added default scopes to credentials: {YOUTUBE_SCOPES}")
            
            # Check again if we've fixed the required fields
            missing_fields = [field for field in required_fields if field not in token_data]
            if missing_fields:
                logging.error(f"Still missing required fields: {', '.join(missing_fields)}")
                return None
        
        creds = Credentials(
            token=token_data.get('token'),
            refresh_token=token_data.get('refresh_token'),
            token_uri=token_data.get('token_uri'),
            client_id=token_data.get('client_id'),
            client_secret=token_data.get('client_secret'),
            scopes=token_data.get('scopes')
        )
        
        # Refresh token if expired
        if creds.expired and creds.refresh_token:
            logging.info("Token expired, attempting to refresh...")
            try:
                creds.refresh(google.auth.transport.requests.Request())
                logging.info("Token refreshed successfully")
                
                # Save refreshed token back to file
                refreshed_data = {
                    'token': creds.token,
                    'refresh_token': creds.refresh_token,
                    'token_uri': creds.token_uri,
                    'client_id': creds.client_id,
                    'client_secret': creds.client_secret,
                    'scopes': creds.scopes
                }
                if creds.expiry:
                    refreshed_data['expiry'] = creds.expiry.isoformat()
                
                with open(token_file_path, 'w') as f:
                    json.dump(refreshed_data, f, indent=2)
                logging.info("Refreshed token saved")
                
            except Exception as refresh_error:
                logging.error(f"Failed to refresh token: {refresh_error}")
                return None
        
        return creds
    except Exception as e:
        logging.error(f"Failed to load YouTube token: {e}")
        return None

def enhanced_reddit_download(submission_url: str, output_path: pathlib.Path, user_agents: List[str] = None) -> bool:
    """Enhanced Reddit video downloader with 403 error mitigation"""
    if user_agents is None:
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebLib/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
    
    for attempt, user_agent in enumerate(user_agents):
        try:
            logging.info(f"Attempting Reddit download (attempt {attempt + 1}) with user agent: {user_agent[:50]}...")
            
            # Configure yt-dlp with rotating user agent and other options
            ydl_opts = {
                'format': 'best[height<=1080]/best',
                'outtmpl': str(output_path),
                'no_warnings': False,
                'ignoreerrors': False,
                'extract_flat': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
                'http_headers': {
                    'User-Agent': user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                },
                'retries': 3,
                'fragment_retries': 3,
                'skip_unavailable_fragments': True,
                'keepvideo': False,
                'noplaylist': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Add delay between attempts
                if attempt > 0:
                    delay = random.uniform(2, 5)
                    logging.info(f"Waiting {delay:.1f} seconds before retry...")
                    time.sleep(delay)
                
                ydl.download([submission_url])
                
                # Check if file was created successfully
                if output_path.exists() and output_path.stat().st_size > 1000:  # At least 1KB
                    logging.info(f"Successfully downloaded: {output_path}")
                    return True
                else:
                    logging.warning(f"Download completed but file is invalid or too small")
                    if output_path.exists():
                        output_path.unlink()  # Remove invalid file
                        
        except Exception as e:
            logging.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
            if output_path.exists():
                try:
                    output_path.unlink()  # Clean up partial download
                except:
                    pass
            
            # If this is a 403 error, try next user agent
            if "403" in str(e) or "Blocked" in str(e):
                logging.warning("403/Blocked error detected, trying next user agent...")
                continue
            
            # For other errors, wait a bit longer
            if attempt < len(user_agents) - 1:
                delay = random.uniform(3, 8)
                logging.info(f"Waiting {delay:.1f} seconds before next attempt...")
                time.sleep(delay)
    
    logging.error(f"All download attempts failed for: {submission_url}")
    return False

def main():
    """Main function to execute the script with all the logic from the first if __name__ block."""
    # Handle graceful exit for Ctrl+C
    def signal_handler(sig, frame):
        logging.info('Ctrl+C received. Shutting down gracefully...')
        close_database() # Ensure DB is closed
        # Add any other cleanup needed
        clean_temp_files(TEMP_DIR) # General cleanup of the temp dir
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    main_start_time = time.time()
    processed_count = 0
    args_main = parse_args() # Parse args once for the main loop
    temp_files_to_clean_master = []  # Initialize here to ensure proper scope

    try:
        validate_environment()
        setup_directories() # This now also handles font downloads
        setup_database()
        
        # Run system tests if not in test-only mode
        if not args_main.test and not run_system_tests():
            logging.error("Critical system tests failed. Exiting.")
            sys.exit(1)

        if args_main.test: # If --test was passed, exit after setup and tests
            logging.info("Test mode enabled. Setup and tests complete. Exiting.")
            sys.exit(0)

        reddit_ok, youtube_ok, gemini_ok = setup_api_clients()
        if not reddit_ok:
            logging.error("Reddit client failed to initialize. Cannot fetch content. Exiting.")
            sys.exit(1)
        if not gemini_ok:
            logging.warning("Gemini client failed to initialize. Analysis will be basic.")
        if not youtube_ok:
            logging.warning("YouTube client failed to initialize. Uploads will be skipped.")

        subreddits_to_process = [args_main.subreddit] if args_main.subreddit else CURATED_SUBREDDITS
        
        # Infinite loop for continuous operation
        while True:
            logging.warning(f"\n--- Starting video processing cycle ---")
            iteration_start_time = time.time()
            
            current_subreddit = random.choice(subreddits_to_process)
            # logging.info(f"Selected subreddit: r/{current_subreddit}")

            submissions = get_reddit_submissions(current_subreddit, limit=5) # Fetch a few to pick from
            if not submissions:
                logging.warning(f"No suitable submissions found in r/{current_subreddit}. Skipping this cycle.")
                time.sleep(30) # Short sleep if nothing found
                continue

            selected_submission = None
            downloaded_video_path: Optional[pathlib.Path] = None
            
            # Sanitize title for filenames
            safe_title_base = re.sub(r'[^\w\s-]', '', submissions[0].title).strip().replace(' ', '_')[:50]

            for submission_candidate in submissions:
                if is_already_uploaded(submission_candidate.permalink):
                    logging.info(f"Skipping already uploaded: {submission_candidate.title[:30]}...")
                    continue
                
                # Download step (if not skipped)
                if not args_main.skip_download:
                    path_info = _prepare_initial_video(submission_candidate, safe_title_base, temp_files_to_clean_master)
                    downloaded_video_path = path_info[0] if path_info else None
                    if not downloaded_video_path or not downloaded_video_path.is_file():
                        logging.warning(f"Failed to download video for: {submission_candidate.title[:30]}...")
                        continue # Try next submission
                else: # If skipping download, we can't proceed with this submission unless a file is manually provided
                    logging.info("Skipping download as per --skip-download. Manual video file needed for this flow to work.")
                    # This part would need modification to accept a local file path if downloads are skipped
                    continue

                # Safety Check (if not skipped)
                if not args_main.skip_safety_check:
                    is_unsafe, reason = gemini_content_safety_check(submission_candidate, downloaded_video_path)
                    if is_unsafe:
                        logging.warning(f"Content deemed unsuitable: {submission_candidate.title[:30]}... Reason: {reason}")
                        cleanup_temp_files(downloaded_video_path) # Clean up downloaded unsafe video
                        downloaded_video_path = None
                        continue # Try next submission
                    else:
                        logging.info(f"Content safety check passed for: {submission_candidate.title[:30]}... Reason: {reason}")
                
                selected_submission = submission_candidate
                break # Found a suitable submission

            if not selected_submission or not downloaded_video_path:
                logging.warning("No suitable new submissions found after filtering and checks.")
                if args_main.sleep > 0:
                    logging.info(f"Sleeping for {args_main.sleep // 4} seconds before trying next subreddit/cycle...")
                    time.sleep(args_main.sleep // 4)
                continue


            # AI Analysis
            analysis = analyze_video_with_gemini(downloaded_video_path, selected_submission.title, selected_submission.subreddit.display_name)
            if analysis.get("fallback", True):
                logging.warning(f"Gemini analysis used fallback for: {selected_submission.title[:30]}...")
            
            # Check for multiple segments first, otherwise fall back to single best segment
            segments_info = analysis.get('segments', [])
            original_video_duration = analysis.get('original_duration', get_video_details(downloaded_video_path)[0])
            
            # If no segments or empty array, fall back to best_segment
            if not segments_info:
                best_segment_info = analysis.get('best_segment', FALLBACK_ANALYSIS['best_segment'])
                segments_info = [best_segment_info]
                
            # Ensure segments have start_seconds and end_seconds
            valid_segments = []
            total_duration = 0
            for segment in segments_info:
                start_crop_time = float(segment.get('start_seconds', 0))
                end_crop_time = float(segment.get('end_seconds', start_crop_time + 15))  # Default 15s if not specified
                
                # Validate segment times
                if start_crop_time >= original_video_duration:
                    logging.warning(f"Segment start {start_crop_time}s is beyond video duration {original_video_duration}s. Skipping segment.")
                    continue
                    
                if end_crop_time > original_video_duration:
                    logging.warning(f"Segment end {end_crop_time}s exceeds video duration {original_video_duration}s. Trimming to {original_video_duration}s.")
                    end_crop_time = original_video_duration
                
                segment_duration = end_crop_time - start_crop_time
                if segment_duration <= 0:
                    logging.warning(f"Invalid segment duration: {segment_duration}s. Skipping segment.")
                    continue
                    
                if total_duration + segment_duration > TARGET_VIDEO_DURATION_SECONDS:
                    # Trim the last segment to fit within target duration
                    allowed_duration = TARGET_VIDEO_DURATION_SECONDS - total_duration
                    if allowed_duration >= 3:  # Only add segment if at least 3 seconds can be included
                        logging.warning(f"Trimming segment from {segment_duration}s to {allowed_duration}s to fit target duration.")
                        end_crop_time = start_crop_time + allowed_duration
                        segment_duration = allowed_duration
                        valid_segments.append({
                            "start_seconds": start_crop_time,
                            "end_seconds": end_crop_time,
                            "duration": segment_duration
                        })
                        total_duration += segment_duration
                    break  # Stop adding segments once we hit the limit
                else:
                    valid_segments.append({
                        "start_seconds": start_crop_time,
                        "end_seconds": end_crop_time,
                        "duration": segment_duration
                    })
                    total_duration += segment_duration
            
            # If no valid segments, create a default one
            if not valid_segments:
                logging.warning("No valid segments found. Using default segment (first 59 seconds).")
                valid_segments = [{
                    "start_seconds": 0,
                    "end_seconds": min(original_video_duration, TARGET_VIDEO_DURATION_SECONDS),
                    "duration": min(original_video_duration, TARGET_VIDEO_DURATION_SECONDS)
                }]
            
            # Prepare output paths
            safe_title_processed = re.sub(r'[^\w\s-]', '', analysis.get('suggested_title', 'processed_video')).strip().replace(' ', '_')[:50]
            final_video_output_path = TEMP_DIR / f"{selected_submission.id}_{safe_title_processed}.mp4"
            thumbnail_output_path = TEMP_DIR / f"{selected_submission.id}_{safe_title_processed}_thumb.jpg"
            temp_files_to_clean_master.extend([final_video_output_path, thumbnail_output_path])

            # Process each segment and then concatenate
            segment_clips = []
            for i, segment in enumerate(valid_segments):
                # Create individual segment clip
                segment_output_path = TEMP_DIR / f"{selected_submission.id}_segment_{i}.mp4"
                temp_files_to_clean_master.append(segment_output_path)
                
                # Focus points times in analysis are relative to original video.
                # Filter focus points to those in this segment's time range
                key_focus_points_original_times = analysis.get('key_focus_points', [])
                focus_points_for_segment = []
                
                for fp_data in key_focus_points_original_times:
                    original_time = float(fp_data.get('time_seconds', 0))
                    if segment["start_seconds"] <= original_time < segment["end_seconds"]:
                        focus_points_for_segment.append({
                            "time": original_time - segment["start_seconds"],  # Make time relative to segment start
                            "point": fp_data.get("point", {"x_ratio": 0.5, "y_ratio": 0.5})  # Ensure point data exists
                        })
                
                if not focus_points_for_segment:
                    # Add default center focus point if none exist for this segment
                    focus_points_for_segment = [{"time": 0, "point": {"x_ratio": 0.5, "y_ratio": 0.5}}]
                
                logging.info(f"Creating segment {i+1}/{len(valid_segments)}: {segment['start_seconds']}s to {segment['end_seconds']}s")
                
                # Use the imported create_short_clip for each segment
                create_short_clip(
                    video_path=str(downloaded_video_path),
                    output_path=str(segment_output_path),
                    start_time=segment["start_seconds"],
                    end_time=segment["end_seconds"],
                    focus_points=focus_points_for_segment
                )
                
                if not segment_output_path.is_file() or segment_output_path.stat().st_size == 0:
                    logging.error(f"Failed to create segment {i+1}. Skipping this segment.")
                    continue
                
                segment_clips.append(segment_output_path)
            
            if not segment_clips:
                logging.error(f"No valid segments were created for {selected_submission.title[:30]}. Skipping.")
                cleanup_temp_files(downloaded_video_path)
                continue
            
            # If we only have one segment, use it directly
            if len(segment_clips) == 1:
                cropped_segment_path = segment_clips[0]
            else:
                # Concatenate multiple segments
                concatenated_path = TEMP_DIR / f"{selected_submission.id}_concatenated.mp4"
                temp_files_to_clean_master.append(concatenated_path)
                
                # Use ffmpeg to concatenate the clips
                concat_success = False
                try:
                    # Create a file list for ffmpeg
                    file_list_path = TEMP_DIR / f"{selected_submission.id}_segments.txt"
                    with open(file_list_path, 'w') as f:
                        for clip_path in segment_clips:
                            f.write(f"file '{clip_path.resolve()}'\n")
                    # Run ffmpeg concatenate
                    cmd = [
                        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                        '-i', str(file_list_path),
                        '-c', 'copy',
                        str(concatenated_path)
                    ]
                    subprocess.run(cmd, check=True, capture_output=True, timeout=60)  # Increased timeout
                    # Clean up file list
                    cleanup_temp_files(file_list_path)
                    if concatenated_path.is_file() and concatenated_path.stat().st_size > 1000:  # Check for valid file with some content
                        logging.info(f"Successfully concatenated {len(segment_clips)} segments.")
                        cropped_segment_path = concatenated_path
                        concat_success = True
                    else:
                        logging.warning(f"Concatenated file invalid or empty. Trying alternative approach.")
                except Exception as e:
                    logging.warning(f"Error with primary concatenation method: {e}. Trying alternative approach.")
                    
                    # Alternative concatenation approach if the first method failed
                    if not concat_success:
                        try:
                            # Try alternative method using filter_complex for more compatibility
                            alternative_cmd = [
                                'ffmpeg', '-y'
                            ]
                            
                            # Add all input files
                            for clip_path in segment_clips:
                                alternative_cmd.extend(['-i', str(clip_path)])
                            
                            # Create filter_complex string for concatenation
                            filter_str = f"concat=n={len(segment_clips)}:v=1:a=1[outv][outa]"
                            
                            # Complete the command
                            alternative_cmd.extend([
                                '-filter_complex', filter_str,
                                '-map', '[outv]', '-map', '[outa]',
                                '-c:v', 'libx264', '-c:a', 'aac',
                                str(concatenated_path)
                            ])
                            
                            subprocess.run(alternative_cmd, check=True, capture_output=True, timeout=120)
                            
                            if concatenated_path.is_file() and concatenated_path.stat().st_size > 1000:
                                logging.info(f"Successfully concatenated segments using alternative method.")
                                cropped_segment_path = concatenated_path
                                concat_success = True
                            else:
                                logging.error(f"Both concatenation methods failed. Falling back to first segment.")
                                cropped_segment_path = segment_clips[0]
                        except Exception as alt_e:
                            logging.error(f"Alternative concatenation also failed: {alt_e}. Falling back to first segment.")
                            cropped_segment_path = segment_clips[0]
                    
                    if not concat_success:
                        logging.warning("Using first segment only as concatenation methods failed.")
                        cropped_segment_path = segment_clips[0]
            
            # Process with effects, TTS, music
            # Pass `force_default_narrative=True` to ensure TTS/subtitles even if Gemini doesn't provide them
            process_success, music_meta = process_video_with_effects(cropped_segment_path, analysis, final_video_output_path, temp_files_to_clean_master, force_default_narrative=True)
            
            if not process_success or not final_video_output_path.is_file() or final_video_output_path.stat().st_size == 0:
                logging.error(f"Video processing failed for {selected_submission.title[:30]}. Skipping.")
                cleanup_temp_files(downloaded_video_path)
                cleanup_temp_files(cropped_segment_path)
                continue

            # Generate Thumbnail
            generated_thumbnail_path = generate_custom_thumbnail(final_video_output_path, analysis, thumbnail_output_path)

            # Upload to YouTube
            if youtube_ok and YOUTUBE_TOKEN_FILE and os.path.exists(YOUTUBE_TOKEN_FILE):
                youtube_title = analysis.get('suggested_title', f"Cool Reddit Video: {selected_submission.title}")[:100]
                youtube_description = analysis.get('summary_for_description', selected_submission.title)
                youtube_description += f"\n\nOriginal post: https://reddit.com{selected_submission.permalink}"
                youtube_description += get_music_attribution(music_meta)
                youtube_description = youtube_description[:5000]
                
                youtube_tags = analysis.get('hashtags', MONETIZATION_TAGS) + [selected_submission.subreddit.display_name, "shorts"]
                youtube_tags = list(set(tag.replace("#","") for tag in youtube_tags if tag))[:15] # Max ~15 tags, ensure unique

                youtube_url = upload_to_youtube(
                    final_video_output_path,
                    youtube_title,
                    youtube_description,
                    generated_thumbnail_path,
                    tags=youtube_tags
                )
                if youtube_url:
                    logging.info(f"Successfully uploaded to YouTube: {youtube_url}")
                    add_upload_record(selected_submission.permalink, youtube_url, youtube_title, selected_submission.subreddit.display_name)
                    video_id = youtube_url.split('/')[-1]
                    set_video_self_certification(video_id, analysis.get('is_explicitly_age_restricted', False))
                    processed_count += 1
                else:
                    logging.error(f"YouTube upload failed for {selected_submission.title[:30]}.")
            else:
                logging.warning("YouTube client not available or not authenticated. Skipping upload.")
                logging.info(f"Video processing complete. Final video at: {final_video_output_path}")

            # Cleanup specific temp files for this iteration (TTS audio, downloaded video, cropped segment)
            # Master list handles final video and thumbnail if not uploaded or if errors occur before full cycle cleanup
            cleanup_temp_files(downloaded_video_path)
            cleanup_temp_files(cropped_segment_path)
            # TTS files are added to temp_files_to_clean_master by process_video_with_effects
            
            iteration_time = time.time() - iteration_start_time
            logging.info(f"--- Cycle completed in {iteration_time:.2f} seconds ---")

            # Sleep between cycles
            time.sleep(args_main.sleep)
        
    except Exception as e:
        logging.critical(f"An unhandled error occurred in the main loop: {e}", exc_info=True)
    finally:
        close_database()
        logging.info(f"Cleaning up all temporary files from the session...")
        for f_path in temp_files_to_clean_master: # Clean any remaining tracked files
            cleanup_temp_files(f_path)
        clean_temp_files(TEMP_DIR) # General cleanup of the temp dir

        total_execution_time = time.time() - main_start_time
        logging.info(f"\n=== Script finished in {total_execution_time:.2f} seconds. Processed {processed_count} videos. ===")

if __name__ == "__main__":
    try:
        main()
    except NameError as e:
        print(f"Error: {e}")
        print("Falling back to running the main code directly...")
        # As a fallback in case main() isn't properly defined, run the main code inline
        # Handle graceful exit for Ctrl+C
        def signal_handler(sig, frame):
            logging.info('Ctrl+C received. Shutting down gracefully...')
            close_database() # Ensure DB is closed
            # Add any other cleanup needed
            clean_temp_files(TEMP_DIR) # General cleanup of the temp dir
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        
        main_start_time = time.time()
        processed_count = 0
        args_main = parse_args() # Parse args once for the main loop
        temp_files_to_clean_master = []  # Initialize here to ensure proper scope

        try:
            validate_environment()
            setup_directories() # This now also handles font downloads
            setup_database()
            
            # Run system tests if not in test-only mode
            if not args_main.test and not run_system_tests():
                logging.error("Critical system tests failed. Exiting.")
                sys.exit(1)

            if args_main.test: # If --test was passed, exit after setup and tests
                logging.info("Test mode enabled. Setup and tests complete. Exiting.")
                sys.exit(0)

            # Rest of the main loop...
            logging.info("Main loop would run here in fallback mode.")
            
        except Exception as e:
            logging.critical(f"An unhandled error occurred in the main loop: {e}", exc_info=True)
        finally:
            close_database()
            logging.info(f"Cleaning up all temporary files from the session...")
            try:
                for f_path in temp_files_to_clean_master: # Clean any remaining tracked files
                    cleanup_temp_files(f_path)
                clean_temp_files(TEMP_DIR) # General cleanup of the temp dir
            except Exception as cleanup_error:
                logging.error(f"Error during cleanup: {cleanup_error}")

            total_execution_time = time.time() - main_start_time
            logging.info(f"\n=== Script finished in {total_execution_time:.2f} seconds. Processed {processed_count} videos. ===")