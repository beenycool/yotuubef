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

# Ensure that we don't truncate the terminal output
logging.getLogger().setLevel(logging.INFO)

# --- External Dependency ---
# Ensure 'video_processor.py' is in the same directory or Python path
# It must contain: _prepare_initial_video, create_short_clip
try:
    from video_processor import _prepare_initial_video, create_short_clip
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

# Content Filtering
FORBIDDEN_WORDS = [
    "fuck", "fucking", "fucked", "fuckin", "shit", "shitty", "shitting",
    "damn", "damned", "bitch", "bitching", "cunt", "asshole", "piss", "pissed",
    "wtf", "stfu", "bs", "omfg", "af", "hell", "ass", "dumbass", "jackass",
    "gore", "graphic", "brutal", "blood", "bloody", "murder", "killing", "suicide",
    "kill", "killed", "stabbed", "shot", "death", "died", "fatal", "lethal",
    "porn", "pornographic", "nsfw", "xxx", "sex", "sexual", "nude", "naked",
    "sexy", "hot girl", "hot boy", "only fans", "seductive", "erotic", "orgasm", "cum",
    "racist", "racism", "nazi", "sexist", "homophobic", "slur", "bigot",
    "discriminatory", "hateful", "supremacist", "ethnic slur", "kkk",
    "drug", "drugs", "weed", "smoking", "cocaine", "heroin", "marijuana", "meth", "overdose",
    "covid hoax", "vaccine hoax", "conspiracy", "5g conspiracy",
    "not for kids", "adults only", "sensitive content", "graphic content", "child abuse"
]
UNSUITABLE_CONTENT_TYPES = [
    "gore", "extreme violence", "graphic injury", "animal abuse", "child exploitation",
    "pornography", "nudity (non-artistic)", "sexual solicitation", "hate speech",
    "dangerous activities glorification", "suicide promotion", "self-harm", "illegal acts",
    "drug abuse promotion", "excessive profanity", "graphic controversial political",
    "medical misinformation", "harmful conspiracy theories", "bullying", "harassment",
    "gambling promotion", "unregulated financial products"
]
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
    "best_segment": {"start_seconds": 0, "end_seconds": 0, "reason": "N/A"},
    "segments": [{"start_seconds": 0, "end_seconds": 59, "reason": "Default segment"}],  # Added segments array
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
    "music_genres": ["neutral"],
    "key_audio_moments": [],
    "retention_tactics": [],
    "hashtags": ["reddit", "video", "shorts"],
    "original_duration": 0.0,
    "tts_pacing": "normal",
    "thumbnail_info": { # Added for consistency
        "timestamp_seconds": 0.0,
        "reason": "Default fallback",
        "headline_text": ""
    },
    "call_to_action": {"text": "", "type": "none"}, # Added
    "emotional_arc_description": "N/A", # Added
    "is_explicitly_age_restricted": False, # Added
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
    if not text: return False
    text_lower = text.lower()
    return any(word in text_lower for word in FORBIDDEN_WORDS)

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
        logging.info(f"Fetching hot posts from r/{subreddit_name}...")
        
        # Fetch more posts initially to have enough after filtering
        fetch_limit = min(limit * 5, 100)  # Fetch 5x what we need, max 100
        for submission in subreddit_instance.hot(limit=fetch_limit):
            # Basic filters: video, not NSFW, not stickied
            is_link_video = any(submission.url.endswith(ext) for ext in ['.mp4', '.mov', '.gifv'])
            is_hosted_video = any(host in submission.url for host in ['v.redd.it', 'gfycat.com', 'streamable.com', 'i.imgur.com'])
            
            if (submission.is_video or is_link_video or is_hosted_video) and \
               not submission.over_18 and not submission.stickied:
                submissions.append(submission)
                if len(submissions) >= limit * 3:  # Get 3x what we need for safety
                    break
            time.sleep(0.2)  # Be respectful to Reddit API
        
        logging.info(f"Fetched {len(submissions)} potential video submissions from r/{subreddit_name}.")
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
            if duration < 3: return True, f"Video too short ({duration:.1f}s)"
            if width < 360 or height < 360: return True, f"Video resolution too low ({width}x{height})" # Min 360p
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
            if is_problematic and confidence >= 60: # Adjust confidence threshold as needed
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

Your goal is to make the video engaging, shareable, and suitable for a general audience (monetization-friendly).

IMPORTANT: For event-driven videos (e.g., explosions, animal sounds, human reactions), answer the following:
- Is there any specific, impactful original sound (e.g., an explosion, an animal's unique call, a surprising human reaction) that is essential to the video's impact? If so, explicitly set \"original_audio_is_key\": true and describe the key audio moment(s) in \"key_audio_moments_original\" (with time_seconds and action: e.g., 'keep_prominent', 'briefly_mute_music'). Otherwise, set to false.
- Based on the video content, suggest 1-2 appropriate music genres. For example, if it's an explosion, suggest \"dramatic orchestral\", \"epic trailer music\", or \"heavy rock intro\". If it's a cute animal, suggest \"playful ukulele\" or \"lighthearted acoustic\".
- Instead of just finding one best segment, identify multiple segments that can be combined. You can identify up to 3 segments for a total max duration of 59 seconds. For example, you might want to include a setup from the beginning, a key moment in the middle, and a reaction at the end.
- For text overlays (graphical callouts, NOT subtitles for speech), specify \"time_seconds\" relative to the total edited clip. Hooks should appear early. Impact words (e.g., \"WOW!\") should be timed with key visual moments.
- If generating a narrative script for Text-to-Speech (TTS), keep segments very short and punchy, suitable for a fast-paced clip. The first segment should act as a hook. This script will be used for both TTS audio and its corresponding on-screen subtitles.

Provide your response ONLY as a valid JSON object with the following structure:
{{
    "suggested_title": "string (concise, catchy, <70 chars, for YouTube)",
    "summary_for_description": "string (2-3 sentences for YouTube description, highlight key aspects)",
    "mood": "string (choose one: funny, heartwarming, informative, suspenseful, action, calm, exciting, awe-inspiring, satisfying, weird, cringe, neutral)",
    "has_clear_narrative": "boolean (does the original video tell a story on its own?)",
    "original_audio_is_key": "boolean (is the original audio (dialogue, specific sounds) crucial to understanding or enjoying the video?)",
    "hook_text": "string (short, punchy text for the first 2-3 seconds of the video, ALL CAPS, max 5 words - this is a graphical overlay, not TTS)",
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
    "text_overlays": [ // For graphical/callout text, NOT speech subtitles. Max 5 overlays. Times relative to final edited video.
        {{
            "text": "string (short, impactful, ALL CAPS, 1-7 words)",
            "time_seconds": "float (when it should appear, relative to final edited video)",
            "duration_seconds": "float (how long it stays, typically 2-4s)",
            "position": "string (e.g., 'center', 'top_third', 'bottom_third')",
            "background_style": "string (e.g., 'subtle_dark_box', 'none', 'bright_highlight_box')"
        }}
    ],
    "narrative_script_segments": [ // For TTS and its corresponding subtitles. Max 3 segments. Times relative to final edited video.
        {{
            "text": "string (natural, conversational, short sentence or two)",
            "time_seconds": "float (when TTS/subtitle should start, relative to final edited video)",
            "intended_duration_seconds": "float (estimated duration for this speech segment)"
        }}
    ],
    "visual_cues_suggestions": [ // Max 3 suggestions. Times relative to final edited video.
        {{"time_seconds": "float", "suggestion": "string (e.g., 'subtle_zoom_in', 'slow_motion_here', 'quick_cut_transition', 'color_pop_effect')"}}
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
                return output_path.is_file() and output_path.stat().st_size > 100
            except Exception as e:
                logging.error(f"Error converting WAV to MP3: {e}", exc_info=True)
                # Keep the WAV file if MP3 conversion fails
                return wav_output_path.is_file() and wav_output_path.stat().st_size > 100
        
        return wav_output_path.is_file() and wav_output_path.stat().st_size > 100
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
                        return output_path.is_file() and output_path.stat().st_size > 100
                    except Exception as e:
                        logging.error(f"Error converting WAV to MP3: {e}", exc_info=True)
                        # Keep the WAV file if MP3 conversion fails
                        return wav_output_path.is_file() and wav_output_path.stat().st_size > 100
                
                return wav_output_path.is_file() and wav_output_path.stat().st_size > 100
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
    temp_files_list: List[pathlib.Path], # To track temp files for cleanup
    force_default_narrative: bool = True # Add parameter to force default narrative
) -> Tuple[bool, Dict]:
    """Processes video with AI-driven effects, TTS, music, etc."""
    if not source_video_path.is_file():
        logging.error(f"Source video for processing not found: {source_video_path}")
        return False, {}

    video_clip = None # Initialize to ensure it's in scope for finally
    try:
        logging.info(f"Starting video processing for: {source_video_path.name}")
        video_clip = VideoFileClip(str(source_video_path))
        
        # Graphical Text Overlays (from Gemini "text_overlays")
        graphical_text_clips = []
        if analysis.get("text_overlays") and isinstance(analysis["text_overlays"], list):
            for overlay_data in analysis["text_overlays"]:
                text = overlay_data.get("text", "").strip()
                if not text: continue

                start_time = float(overlay_data.get("time_seconds", 0.0))
                duration = float(overlay_data.get("duration_seconds", 3.0))
                
                font_size = int(video_clip.h * GRAPHICAL_TEXT_FONT_SIZE_RATIO)
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
                        font=GRAPHICAL_TEXT_FONT, # Use new constant
                        color=GRAPHICAL_TEXT_COLOR,
                        stroke_color=GRAPHICAL_TEXT_STROKE_COLOR,
                        stroke_width=GRAPHICAL_TEXT_STROKE_WIDTH, # Use new constant
                        method='caption', 
                        align='center',
                        size=(video_clip.w * 0.9, None), 
                        bg_color=bg_color
                    )
                    txt_clip = txt_clip.set_position(position, relative=True if isinstance(position[1], float) else False)
                    txt_clip = txt_clip.set_start(start_time).set_duration(duration)
                    txt_clip = txt_clip.crossfadein(0.3).crossfadeout(0.3)
                    
                    if text.strip().upper() in {"BOOM!", "BOOM", "BANG!", "BANG", "EXPLOSION", "BLAST!", "BLAST", "WOW!", "WOW"}: # Added WOW
                        import numpy as np # Ensure numpy is imported here or globally
                        base_txt_clip = txt_clip
                        def scale_and_shake(get_frame, t):
                            frame = get_frame(t)
                            scale = 1.0 + 0.7 * np.exp(-10 * t) 
                            h_orig, w_orig = frame.shape[:2]
                            
                            # Ensure frame has alpha if bg_color is transparent
                            if base_txt_clip.bg_color == 'transparent' and frame.shape[2] == 3:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                                frame[:, :, 3] = np.where(np.all(frame[:,:,:3] == [0,0,0], axis=-1), 0, 255)


                            new_w, new_h = int(w_orig * scale), int(h_orig * scale)
                            if new_w <= 0 or new_h <= 0: return frame # safety
                            
                            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                            
                            x1 = (new_w - w_orig) // 2
                            y1 = (new_h - h_orig) // 2
                            
                            # Create output frame matching original size and type
                            output_frame = np.zeros_like(frame) # Handles 3 or 4 channels

                            # Calculate valid source and destination slices for ROI
                            src_x_start, src_y_start = max(0, -x1), max(0, -y1)
                            src_x_end, src_y_end = min(new_w, w_orig - x1), min(new_h, h_orig - y1)
                            
                            dst_x_start, dst_y_start = max(0, x1), max(0, y1)
                            dst_x_end, dst_y_end = min(w_orig, x1 + new_w), min(h_orig, y1 + new_h)

                            # Ensure calculated slice dimensions match
                            actual_w = min(src_x_end - src_x_start, dst_x_end - dst_x_start)
                            actual_h = min(src_y_end - src_y_start, dst_y_end - dst_y_start)

                            if actual_w > 0 and actual_h > 0:
                                output_frame[dst_y_start : dst_y_start+actual_h, dst_x_start : dst_x_start+actual_w] = \
                                    resized_frame[src_y_start : src_y_start+actual_h, src_x_start : src_x_start+actual_w]
                            
                            current_frame = output_frame

                            if t < 0.4:
                                dx = int(8 * np.sin(50 * t))
                                dy = int(8 * np.cos(45 * t))
                                M = np.float32([[1, 0, dx], [0, 1, dy]])
                                current_frame = cv2.warpAffine(current_frame, M, (w_orig, h_orig), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0) if current_frame.shape[2]==4 else (0,0,0))
                            return current_frame
                        txt_clip = txt_clip.fl(scale_and_shake, apply_to=['mask', 'video']) # Mask for transparent bg
                    graphical_text_clips.append(txt_clip)
                except Exception as e_txt:
                    logging.error(f"Could not create graphical text overlay for '{text}': {e_txt}", exc_info=True)

        # Narrative TTS Audio & Subtitles
        tts_audio_clips = []
        subtitle_text_clips = [] # New list for subtitles
        narrative_segments_from_analysis = analysis.get("narrative_script_segments", [])
        
        # Force default narrative if specified and no segments from analysis
        actual_narrative_segments = narrative_segments_from_analysis
        if force_default_narrative and (not actual_narrative_segments or len(actual_narrative_segments) == 0):
            logging.info("No narrative segments from analysis. Adding default narrative for TTS/Subtitles.")
            actual_narrative_segments = FALLBACK_ANALYSIS["narrative_script_segments"]
            
        if actual_narrative_segments:
            processed_tts_segments = generate_narrative_audio_segments(actual_narrative_segments, analysis, TEMP_DIR)
            for seg_info in processed_tts_segments:
                # TTS Audio Clip
                tts_audioclip = AudioFileClip(str(seg_info["file_path"]))
                tts_audioclip = tts_audioclip.set_start(seg_info["start_time"])
                tts_audio_clips.append(tts_audioclip)
                temp_files_list.append(seg_info["file_path"]) 

                # Corresponding Subtitle Clip
                text_for_subtitle = seg_info["text"]
                tts_start_time = seg_info["start_time"]
                tts_duration = seg_info["duration"]
                
                # Ensure subtitle font exists or fallback
                actual_subtitle_font = SUBTITLE_FONT
                if not pathlib.Path(actual_subtitle_font).is_file() and actual_subtitle_font.lower() != 'arial':
                    if pathlib.Path(SUBTITLE_FONT.replace("-Regular", "-Bold")).is_file(): # Try bold as fallback
                         actual_subtitle_font = SUBTITLE_FONT.replace("-Regular", "-Bold")
                         logging.warning(f"Subtitle font {SUBTITLE_FONT} not found, using {actual_subtitle_font}.")
                    else:
                        logging.warning(f"Subtitle font {SUBTITLE_FONT} not found, using Arial.")
                        actual_subtitle_font = 'Arial'
                
                try:
                    subtitle_clip = TextClip(
                        text_for_subtitle,
                        fontsize=int(video_clip.h * SUBTITLE_FONT_SIZE_RATIO),
                        font=actual_subtitle_font,
                        color=SUBTITLE_TEXT_COLOR,
                        stroke_color=SUBTITLE_STROKE_COLOR,
                        stroke_width=SUBTITLE_STROKE_WIDTH,
                        method='caption',
                        align='center',
                        size=(video_clip.w * 0.85, None), 
                        bg_color=SUBTITLE_BG_COLOR
                    )
                    subtitle_clip = subtitle_clip.set_position(SUBTITLE_POSITION, relative=True)
                    subtitle_clip = subtitle_clip.set_start(tts_start_time).set_duration(tts_duration)
                    subtitle_text_clips.append(subtitle_clip)
                except Exception as e_sub:
                    logging.error(f"Could not create subtitle for '{text_for_subtitle[:30]}...': {e_sub}")
            
            if not tts_audio_clips and force_default_narrative:
                logging.warning("TTS generation failed for all default segments. Continuing without TTS/Subtitles.")
                
        # Background Music (logic remains similar)
        music_clip_final = None
        used_music_metadata = {}
        dynamic_original_audio_mix_volume = ORIGINAL_AUDIO_MIX_VOLUME
        dynamic_background_music_volume = BACKGROUND_MUSIC_VOLUME
        if analysis.get('original_audio_is_key', False):
            dynamic_original_audio_mix_volume = 0.7 
            dynamic_background_music_volume = 0.03  
        if BACKGROUND_MUSIC_ENABLED:
            music_files = list(MUSIC_FOLDER.glob("**/*.mp3")) + list(MUSIC_FOLDER.glob("*.mp3")) # Check subfolders then main
            music_files = list(set(music_files)) # Remove duplicates
            if not music_files:
                logging.info("No music files found. Creating basic folders for future use.")
                download_royalty_free_music()
                music_files = list(MUSIC_FOLDER.glob("**/*.mp3")) + list(MUSIC_FOLDER.glob("*.mp3"))
                music_files = list(set(music_files))

            if music_files:
                music_path, used_music_metadata = select_music_for_content(analysis)
                if music_path and music_path.is_file():
                    try:
                        music_audio_clip = AudioFileClip(str(music_path))
                        if music_audio_clip.duration < video_clip.duration:
                            num_loops = math.ceil(video_clip.duration / music_audio_clip.duration)
                            music_audio_clip = concatenate_audioclips([music_audio_clip] * int(num_loops))
                        music_clip_final = music_audio_clip.subclip(0, video_clip.duration)
                        
                        current_music_volume = dynamic_background_music_volume
                        if tts_audio_clips: 
                            current_music_volume *= BACKGROUND_MUSIC_NARRATIVE_VOLUME_FACTOR
                            
                        if analysis.get('original_audio_is_key', False) and analysis.get('key_audio_moments_original'):
                            key_moments = analysis['key_audio_moments_original']
                            keyframes = [(0, current_music_volume)]
                            for moment in key_moments:
                                moment_time = float(moment.get('time_seconds', 0))
                                if moment.get('action') == 'briefly_mute_music':
                                    keyframes.append((max(0, moment_time - 0.2), current_music_volume))
                                    keyframes.append((moment_time, 0.005)) # Duck very low
                                    keyframes.append((moment_time + float(moment.get('duration_seconds', 0.5)), 0.005))
                                    keyframes.append((moment_time + float(moment.get('duration_seconds', 0.5)) + 0.2, current_music_volume))
                            keyframes.sort(key=lambda x: x[0])
                            if not keyframes or keyframes[-1][0] < video_clip.duration: # Ensure last keyframe covers duration
                                keyframes.append((video_clip.duration, keyframes[-1][1] if keyframes else current_music_volume))
                            
                            # Apply volume envelope using a lambda function for MoviePy
                            def volume_envelope(t):
                                for i in range(len(keyframes) - 1):
                                    t1, v1 = keyframes[i]
                                    t2, v2 = keyframes[i+1]
                                    if t1 <= t < t2:
                                        return v1 + (v2 - v1) * (t - t1) / (t2 - t1) if (t2-t1) !=0 else v1
                                return keyframes[-1][1] # Return last volume if t >= last keyframe time
                            music_clip_final = music_clip_final.fl_time(lambda t: volume_envelope(t), apply_to='audio')
                            # The above .fl_time applies the function to the audio samples over time.
                            # A simpler way if moviepy's .volumex supports a function (it usually doesn't directly for audio volume like this):
                            # music_clip_final = music_clip_final.fx(vfx.volumex, volume_envelope) -> this is for video effects
                            # So, the fl_time or a custom Audioসম্পাদকAudioClip.fl method is needed.
                            # For simplicity, if complex ducking is hard, just lower overall volume when original audio is key
                            # music_clip_final = music_clip_final.volumex(current_music_volume * 0.5 if key_moments else current_music_volume)

                        else:
                            music_clip_final = music_clip_final.volumex(current_music_volume)
                            
                        music_clip_final = music_clip_final.audio_fadein(1.0).audio_fadeout(1.0)
                        logging.info(f"Added background music: {music_path.name}")
                    except Exception as e_music:
                        logging.error(f"Could not load or process background music {music_path}: {e_music}", exc_info=True)
                        music_clip_final = None
            else:
                logging.warning("No music files available. Continuing without background music.")

        # Visual Effects (logic remains similar)
        enhanced_video_clip = video_clip 
        if COLOR_GRADE_ENABLED:
            try:
                enhanced_video_clip = enhanced_video_clip.fx(vfx.colorx, 1.05) 
                enhanced_video_clip = enhanced_video_clip.fx(vfx.lum_contrast, lum=5, contrast=0.05, contrast_thr=127) 
            except Exception as e_color: logging.warning(f"Color grading failed: {e_color}")
        
        if SUBTLE_ZOOM_ENABLED and enhanced_video_clip.duration > 0: 
            try:
                key_moments_for_zoom = analysis.get('key_audio_moments_original', []) if analysis.get('original_audio_is_key', False) else []
                # Filter visual cues for zoom/shake
                visual_cues_for_zoom = [vc for vc in analysis.get('visual_cues_suggestions', []) if 'zoom' in vc.get('suggestion','').lower() or 'shake' in vc.get('suggestion','').lower()]

                def zoom_in_out_effect(get_frame, t):
                    frame = get_frame(t)
                    h, w, _ = frame.shape
                    progress = t / enhanced_video_clip.duration if enhanced_video_clip.duration > 0 else 0
                    
                    zoom_factor = 1.0 + 0.03 * progress # Default slow zoom
                    apply_shake = False
                    shake_intensity = 0

                    # Check key audio moments for punch-in
                    for moment in key_moments_for_zoom:
                        event_time = float(moment.get('time_seconds', -100))
                        if 0 <= t - event_time < 0.4: # Punch-in for 0.4s around audio event
                            zoom_factor = 1.18 - 0.45 * (t - event_time) 
                            apply_shake = True
                            shake_intensity = 10
                            break 
                    
                    # Check visual cues for zoom/shake
                    if not apply_shake: # Only if not already triggered by audio
                        for cue in visual_cues_for_zoom:
                            cue_time = float(cue.get('time_seconds', -100))
                            cue_duration = float(cue.get('duration_seconds', 0.4)) # Gemini might provide this
                            if 0 <= t - cue_time < cue_duration:
                                if 'zoom_in_fast' in cue.get('suggestion',''): zoom_factor = 1.15 
                                elif 'zoom_out_fast' in cue.get('suggestion',''): zoom_factor = 0.90
                                if 'shake' in cue.get('suggestion',''):
                                    apply_shake = True
                                    shake_intensity = int(cue.get('intensity', 8)) # Gemini might provide intensity
                                break

                    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
                    if new_w <=0 or new_h <=0 : return frame 
                    
                    crop_x_center = w // 2
                    crop_y_center = h // 2 # Default center, Gemini focus points already applied in create_short_clip

                    x1 = crop_x_center - new_w // 2
                    y1 = crop_y_center - new_h // 2
                    
                    # Ensure crop boundaries are within the original frame
                    x1_clamped = max(0, x1)
                    y1_clamped = max(0, y1)
                    new_w_clamped = min(new_w, w - x1_clamped)
                    new_h_clamped = min(new_h, h - y1_clamped)
                    
                    if new_w_clamped <=0 or new_h_clamped <=0 : return frame # safety

                    cropped_frame_roi = frame[y1_clamped : y1_clamped + new_h_clamped, x1_clamped : x1_clamped + new_w_clamped]
                    
                    # Resize back to original dimensions
                    resized_cropped_frame = cv2.resize(cropped_frame_roi, (w, h), interpolation=cv2.INTER_LINEAR)

                    if apply_shake and shake_intensity > 0:
                        dx = int(shake_intensity * np.sin(60 * t)) # Adjust frequency/amplitude
                        dy = int(shake_intensity * np.cos(55 * t))
                        M = np.float32([[1, 0, dx], [0, 1, dy]])
                        resized_cropped_frame = cv2.warpAffine(resized_cropped_frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE) # Replicate border to avoid black edges
                    return resized_cropped_frame

                enhanced_video_clip = enhanced_video_clip.fl(zoom_in_out_effect)
            except Exception as e_zoom: logging.warning(f"Subtle zoom effect failed: {e_zoom}", exc_info=True)
        
        # Composite video: Video + Graphical Text + Subtitles + Watermark (order matters for layering)
        final_clips_to_composite = [enhanced_video_clip] + graphical_text_clips + subtitle_text_clips
        
        watermark_img_clip = None # Initialize
        if WATERMARK_PATH.is_file():
            try:
                watermark_img_clip = (ImageClip(str(WATERMARK_PATH))
                                  .set_duration(enhanced_video_clip.duration)
                                  .resize(height=int(enhanced_video_clip.h * 0.04)) 
                                  .margin(right=15, bottom=15, opacity=0) 
                                  .set_pos(("right","bottom")))
                final_clips_to_composite.append(watermark_img_clip)
            except Exception as e_wm: logging.warning(f"Could not add watermark: {e_wm}")

        final_video_render = CompositeVideoClip(final_clips_to_composite, size=TARGET_RESOLUTION)

        # Audio Mixing
        all_audio_tracks = []
        if analysis.get('original_audio_is_key', False) and video_clip.audio:
            original_audio = video_clip.audio.volumex(dynamic_original_audio_mix_volume)
            all_audio_tracks.append(original_audio)
        
        all_audio_tracks.extend(tts_audio_clips)
        if music_clip_final:
            all_audio_tracks.append(music_clip_final)

        if all_audio_tracks:
            final_audio_mix = CompositeAudioClip(all_audio_tracks)
            final_video_render = final_video_render.set_audio(final_audio_mix)
        elif video_clip.audio: 
             if not analysis.get('original_audio_is_key', False):
                 final_video_render = final_video_render.without_audio()

        # Render final video (GPU/CPU logic remains similar)
        use_gpu = False 
        ffmpeg_extra_params = []
        video_codec_to_use = VIDEO_CODEC_CPU
        ffmpeg_preset = FFMPEG_CPU_PRESET
        quality_param = ["-crf", FFMPEG_CRF_CPU]

        if CUDA_AVAILABLE:
            use_gpu = True
            video_codec_to_use = VIDEO_CODEC_GPU
            ffmpeg_preset = FFMPEG_GPU_PRESET
            quality_param = ["-cq", FFMPEG_CQ_GPU]
            logging.info("Using NVIDIA GPU (detected via PyTorch) with NVENC.")
        else:
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, check=True, timeout=5)
                if "NVIDIA-SMI" in result.stdout.decode():
                    use_gpu = True
                    video_codec_to_use = VIDEO_CODEC_GPU
                    ffmpeg_preset = FFMPEG_GPU_PRESET
                    quality_param = ["-cq", FFMPEG_CQ_GPU] 
                    logging.info("NVIDIA GPU detected via nvidia-smi. Will attempt to use NVENC.")
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                logging.info("NVIDIA GPU not detected or nvidia-smi failed. Using CPU encoding.")
        
        final_ffmpeg_params = [
            *quality_param,
            "-preset", ffmpeg_preset,
            "-b:a", AUDIO_BITRATE,
            "-ar", "48000",
            "-movflags", "+faststart", 
            "-pix_fmt", "yuv420p",
            "-profile:v", "high", 
            "-level", "4.2"       
        ]
        final_ffmpeg_params.extend(ffmpeg_extra_params)

        logging.info(f"Rendering final video to {output_path.name} using {'GPU ('+video_codec_to_use+')' if use_gpu else 'CPU ('+video_codec_to_use+')'}")
        final_video_render.write_videofile(
            str(output_path),
            codec=video_codec_to_use,
            audio_codec=AUDIO_CODEC,
            fps=TARGET_FPS,
            threads=max(1, os.cpu_count() // 2 if not use_gpu else 4), 
            ffmpeg_params=final_ffmpeg_params,
            logger='bar' 
        )
        
        # Loudness Normalization (after video is written)
        if check_ffmpeg_install("ffmpeg-normalize"):
            normalized_output_path = output_path.with_name(output_path.stem + "_normalized" + output_path.suffix)
            cmd_normalize = [
                "ffmpeg-normalize", str(output_path),
                "-o", str(normalized_output_path),
                "-ar", "48000",
                "-c:a", AUDIO_CODEC, 
                "-l", str(LOUDNESS_TARGET_LUFS),
                "-f" 
            ]
            try:
                logging.info(f"Normalizing audio of {output_path.name} to {LOUDNESS_TARGET_LUFS} LUFS...")
                norm_proc = subprocess.run(cmd_normalize, check=True, capture_output=True, text=True, timeout=120)
                logging.debug(f"ffmpeg-normalize stdout: {norm_proc.stdout}")
                logging.debug(f"ffmpeg-normalize stderr: {norm_proc.stderr}")
                
                cleanup_temp_files(output_path) 
                shutil.move(str(normalized_output_path), str(output_path))
                logging.info(f"Audio normalized successfully. Final file: {output_path.name}")
            except subprocess.CalledProcessError as e_norm:
                logging.error(f"ffmpeg-normalize failed for {output_path.name}: {e_norm.stderr}")
                cleanup_temp_files(normalized_output_path) 
            except subprocess.TimeoutExpired:
                logging.error(f"ffmpeg-normalize timed out for {output_path.name}.")
                cleanup_temp_files(normalized_output_path)
            except Exception as e_norm_mv:
                logging.error(f"Error moving normalized file for {output_path.name}: {e_norm_mv}")
                cleanup_temp_files(normalized_output_path)
        else:
            if ensure_ffmpeg_normalize_installed():
                try:
                    normalized_output_path = output_path.with_name(output_path.stem + "_normalized" + output_path.suffix)
                    cmd_normalize = [
                        "ffmpeg-normalize", str(output_path),
                        "-o", str(normalized_output_path),
                        "-ar", "48000",
                        "-c:a", AUDIO_CODEC,
                        "-l", str(LOUDNESS_TARGET_LUFS),
                        "-f"
                    ]
                    logging.info(f"Normalizing audio after installing ffmpeg-normalize...")
                    norm_proc = subprocess.run(cmd_normalize, check=True, capture_output=True, text=True, timeout=120)
                    
                    cleanup_temp_files(output_path)
                    shutil.move(str(normalized_output_path), str(output_path))
                    logging.info(f"Audio normalized successfully after install. Final file: {output_path.name}")
                except Exception as e_retry:
                    logging.error(f"Retry after installation also failed: {e_retry}")
                    cleanup_temp_files(normalized_output_path)
            else:
                logging.warning("ffmpeg-normalize not found and installation failed. Skipping loudness normalization.")

        return True, used_music_metadata

    except Exception as e:
        logging.error(f"Error in process_video_with_effects for {source_video_path.name}: {e}", exc_info=True)
        return False, {}
    finally:
        # Ensure clips are closed to free resources
        if video_clip: video_clip.close() # Changed from 'video_clip' in locals()
        if 'enhanced_video_clip' in locals() and enhanced_video_clip and enhanced_video_clip != video_clip : enhanced_video_clip.close()
        if 'final_video_render' in locals() and final_video_render: final_video_render.close()
        if 'music_clip_final' in locals() and music_clip_final: music_clip_final.close()
        if 'tts_audio_clips' in locals():
            for ac in tts_audio_clips: ac.close()
        if 'graphical_text_clips' in locals(): # Renamed from text_overlay_clips for clarity
            for tc in graphical_text_clips: tc.close()
        if 'subtitle_text_clips' in locals():
            for stc in subtitle_text_clips: stc.close()
        if 'watermark_img_clip' in locals() and watermark_img_clip: watermark_img_clip.close()

        gc.collect()


# --- YouTube Upload ---
def upload_to_youtube(video_path: pathlib.Path, title: str, description: str,
                     thumbnail_path: Optional[pathlib.Path] = None, # Changed to Path
                     category_id: str = YOUTUBE_UPLOAD_CATEGORY_ID,
                     privacy_status: str = YOUTUBE_UPLOAD_PRIVACY_STATUS,
                     tags: Optional[List[str]] = None) -> Optional[str]:
    global youtube_service
    if not youtube_service:
        logging.error("YouTube service not available. Cannot upload.")
        return None
    if not video_path.is_file():
        logging.error(f"Video file for upload not found: {video_path}")
        return None

    try:
        body = {
            'snippet': {
                'title': title[:100], # YouTube title limit
                'description': description[:5000], # YouTube description limit
                'tags': (tags if tags else MONETIZATION_TAGS)[:500], # Tag limit (char count also applies)
                'categoryId': category_id
            },
            'status': {
                'privacyStatus': privacy_status,
                'selfDeclaredMadeForKids': False,
                # 'publishAt': 'YYYY-MM-DDTHH:MM:SS.sssZ' # For scheduled uploads
            }
        }
        
        logging.info(f"Starting YouTube upload for: {title[:50]}...")
        media = MediaFileUpload(str(video_path), mimetype='video/mp4', resumable=True, chunksize=4*1024*1024) # 4MB chunks
        
        request = youtube_service.videos().insert(
            part=','.join(body.keys()), # snippet,status
            body=body,
            media_body=media,
            notifySubscribers=True if privacy_status == 'public' else False
        )
        
        response = None
        upload_progress = 0
        while response is None:
            try:
                status, response = request.next_chunk()
                if status:
                    new_progress = int(status.progress() * 100)
                    if new_progress > upload_progress + 5 or new_progress == 100 : # Log every 5% or at 100%
                        logging.info(f"YouTube Upload progress: {new_progress}%")
                        upload_progress = new_progress
            except google_api_errors.HttpError as e:
                if e.resp.status in [500, 502, 503, 504]: # Retriable errors
                    logging.warning(f"Retriable YouTube API error (status {e.resp.status}): {e}. Retrying in 5s...")
                    time.sleep(5)
                else:
                    logging.error(f"Non-retriable YouTube API error during upload: {e}")
                    return None # Non-retriable error
            except Exception as e_chunk: # Catch other potential chunk errors
                logging.error(f"Error during YouTube upload chunk: {e_chunk}")
                return None


        video_id = response.get('id')
        if not video_id:
            logging.error(f"YouTube upload succeeded but no video ID returned. Response: {response}")
            return None
        
        youtube_url = f"https://youtu.be/{video_id}"
        logging.info(f"Successfully uploaded video: {youtube_url}")

        if thumbnail_path and thumbnail_path.is_file():
            try:
                logging.info(f"Uploading custom thumbnail: {thumbnail_path.name}")
                youtube_service.thumbnails().set(
                    videoId=video_id,
                    media_body=MediaFileUpload(str(thumbnail_path))
                ).execute()
                logging.info("Custom thumbnail uploaded successfully.")
            except Exception as thumb_error:
                logging.error(f"Error uploading thumbnail for {video_id}: {thumb_error}")
        
        return youtube_url

    except google_api_errors.HttpError as e:
        logging.error(f"YouTube API HTTP error: {e.content.decode() if e.content else e}")
    except Exception as e:
        logging.error(f"Error uploading to YouTube: {e}", exc_info=True)
    return None

def set_video_self_certification(video_id: str, is_age_restricted_content: bool = False) -> bool:
    global youtube_service
    if not youtube_service or not YOUTUBE_SELF_CERTIFICATION:
        return False
    
    logging.info(f"Attempting self-certification for video {video_id}. Age restricted: {is_age_restricted_content}")
    
    # Note: The 'selfDeclaration' part of the YouTube API (videos.update)
    # is not widely available or fully documented for all users/API keys.
    # This might not work as expected. Primary way is through YouTube Studio.
    # The 'contentDetails.contentRating' is more standard.

    if is_age_restricted_content:
        try:
            youtube_service.videos().update(
                part="contentDetails",
                body={ "id": video_id, "contentDetails": { "contentRating": { "ytRating": "ytAgeRestricted"}}}
            ).execute()
            logging.info(f"Set ytAgeRestricted for video {video_id}")
        except Exception as e_rating:
            logging.warning(f"Could not set content rating for {video_id}: {e_rating}")
            # Don't fail the whole function for this if selfDeclaration might work

    # Attempt self-declaration (might not be effective for all API users)
    try:
        # This part of the API has limited availability/effect for standard users.
        # body_self_declare = {
        #     "id": video_id,
        #     "selfDeclarations": { # Note: API might expect 'selfDeclaration' singular
        #         "contentHasAdRestrictions": { # This structure is speculative based on various docs
        #             "noAdRestrictions": True # Assuming content is clean
        #         }
        #     }
        # }
        # youtube_service.videos().update(part="selfDeclarations", body=body_self_declare).execute()
        logging.info(f"Self-certification 'attempt' made for video {video_id}. Actual effect depends on API access level.")
        # Since direct full self-cert is tricky via API, we mainly rely on `selfDeclaredMadeForKids=False`
        # and careful content selection.
        return True # Indicate attempt was made
    except Exception as e:
        logging.warning(f"Error during self-certification API call for {video_id}: {e}")
    return False


# --- Database ---
def is_already_uploaded(reddit_url: str) -> bool:
    if not db_cursor:
        logging.warning("DB cursor not available for is_already_uploaded check.")
        return True # Fail safe, assume uploaded
    try:
        db_cursor.execute("SELECT 1 FROM uploads WHERE reddit_url = ?", (reddit_url,))
        return db_cursor.fetchone() is not None
    except sqlite3.Error as e:
        logging.error(f"DB error checking URL {reddit_url}: {e}")
        return True # Fail safe

def add_upload_record(reddit_url: str, youtube_url: str, title: str, subreddit: str):
    if not db_conn or not db_cursor:
        logging.warning("DB connection/cursor not available for add_upload_record.")
        return
    try:
        db_cursor.execute(
            "INSERT INTO uploads (reddit_url, youtube_url, title, subreddit) VALUES (?, ?, ?, ?)",
            (reddit_url, youtube_url, title, subreddit)
        )
        db_conn.commit()
        logging.info(f"Added record to DB: {reddit_url} -> {youtube_url}")
    except sqlite3.IntegrityError:
        logging.warning(f"Record for {reddit_url} already exists in DB (IntegrityError).")
    except sqlite3.Error as e:
        logging.error(f"DB error adding record for {reddit_url}: {e}")


# --- Thumbnail Generation ---
def generate_custom_thumbnail(video_path: pathlib.Path, analysis: Dict, output_path: pathlib.Path) -> Optional[pathlib.Path]:
    if not video_path.is_file():
        logging.error(f"Video for thumbnail generation not found: {video_path}")
        return None
        
    try:
        thumbnail_info = analysis.get('thumbnail_info', FALLBACK_ANALYSIS['thumbnail_info'])
        thumbnail_moment_sec = float(thumbnail_info.get('timestamp_seconds', 0.0))
        headline_text = thumbnail_info.get('headline_text', "").strip().upper()

        video_duration, vid_w, vid_h = get_video_details(video_path)
        if video_duration == 0: # Could not get details
            logging.warning(f"Could not get video duration for thumbnail of {video_path.name}. Using first frame.")
            thumbnail_moment_sec = 0.0
        elif thumbnail_moment_sec >= video_duration: # Ensure timestamp is within bounds
            logging.warning(f"Thumbnail moment {thumbnail_moment_sec}s is out of bounds for video {video_path.name} (duration {video_duration}s). Using middle frame.")
            thumbnail_moment_sec = video_duration / 2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"Cannot open video for thumbnail: {video_path}")
            return None
            
        cap.set(cv2.CAP_PROP_POS_MSEC, thumbnail_moment_sec * 1000)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            logging.error(f"Failed to extract frame at {thumbnail_moment_sec}s for thumbnail from {video_path.name}")
            return None
        
        # First save the original frame as JPG using OpenCV
        temp_frame_path = output_path.with_suffix('.raw.jpg')
        cv2.imwrite(str(temp_frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        try:
            # Image processing for thumbnail - use OpenCV for this
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.2, 0, 255)  # Saturation
            hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.1, 0, 255)  # Value/Brightness
            enhanced_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) # Sharpening
            sharpened_frame = cv2.filter2D(enhanced_frame, -1, kernel)
            
            # Use PIL only for text overlay
            img_pil = Image.fromarray(cv2.cvtColor(sharpened_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # Font for thumbnail headline
            font = None
            try:
                # First try to use Bebas Neue font
                bebas_font_path = BASE_DIR / "fonts" / "BebasNeue-Regular.ttf" # or .otf
                if bebas_font_path.is_file():
                    font = ImageFont.truetype(str(bebas_font_path), size=int(img_pil.height / 7))
                    logging.info(f"Using BebasNeue-Regular font for thumbnail text")
                else:
                    # Try Montserrat as fallback
                    montserrat_font_path = BASE_DIR / "fonts" / "Montserrat-Bold.ttf"
                    if montserrat_font_path.is_file():
                        font = ImageFont.truetype(str(montserrat_font_path), size=int(img_pil.height / 7))
                        logging.info(f"Using Montserrat-Bold.ttf for thumbnail text")
                    else:
                        # Final fallback - use system font
                        logging.warning("No custom fonts found for thumbnail. Using system default font.")
                        font = ImageFont.load_default() # This will be small
            except Exception as e_font:
                logging.warning(f"Thumbnail font error: {e_font}. Using default font.")
                font = ImageFont.load_default()

            if headline_text and font is not None:
                # Calculate text size and position
                # For PIL versions >= 9.2.0, textbbox is preferred
                try:
                    text_box = draw.textbbox((0,0), headline_text, font=font)
                    text_w = text_box[2] - text_box[0]
                    text_h = text_box[3] - text_box[1]
                except (AttributeError, TypeError) as e: # Fallback for older PIL
                    try:
                        text_w, text_h = draw.textsize(headline_text, font=font) # type: ignore
                    except Exception as textsize_err:
                        # For very old PIL or other issues, use estimated size
                        logging.warning(f"Text size calculation error: {textsize_err}. Using estimated size.")
                        font_size_val = font.size if hasattr(font, 'size') else int(img_pil.height / 7)
                        text_w = len(headline_text) * font_size_val * 0.6
                        text_h = font_size_val * 1.2

                # Position at bottom center with some padding
                text_x = (img_pil.width - text_w) / 2
                text_y = img_pil.height - text_h - (img_pil.height * 0.05) # 5% padding from bottom

                # Draw stroke (outline) then text
                stroke_width_pil = 4
                for dx in range(-stroke_width_pil, stroke_width_pil + 1):
                    for dy in range(-stroke_width_pil, stroke_width_pil + 1):
                        if dx*dx + dy*dy >= stroke_width_pil*stroke_width_pil: continue # circular stroke
                        draw.text((text_x + dx, text_y + dy), headline_text, font=font, fill="black")
                draw.text((text_x, text_y), headline_text, font=font, fill="white")

            # Save as JPEG to output path
            try:
                img_pil.save(output_path, "JPEG", quality=90)
                logging.info(f"Generated custom thumbnail with text: {output_path.name}")
                # Clean up temp file
                if temp_frame_path.is_file():
                    temp_frame_path.unlink()
                return output_path
            except Exception as save_err:
                logging.error(f"Error saving thumbnail with PIL: {save_err}. Using basic OpenCV thumbnail.")
                # Fallback to just using the enhanced frame without text
                cv2.imwrite(str(output_path), sharpened_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                logging.info(f"Generated basic thumbnail without text: {output_path.name}")
                # Clean up temp file
                if temp_frame_path.is_file():
                    temp_frame_path.unlink()
                return output_path

        except Exception as pil_err:
            logging.error(f"Error during PIL processing: {pil_err}. Using basic OpenCV thumbnail.")
            # If PIL fails, try with just OpenCV
            try:
                cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                logging.info(f"Generated fallback thumbnail: {output_path.name}")
                # Clean up temp file
                if temp_frame_path.is_file():
                    temp_frame_path.unlink()
                return output_path
            except Exception as cv_err:
                logging.error(f"Error during OpenCV thumbnail generation: {cv_err}")
                # If OpenCV wrote the temp file successfully, use it
                if temp_frame_path.is_file():
                    try:
                        shutil.move(str(temp_frame_path), str(output_path))
                        logging.info(f"Used raw frame as thumbnail: {output_path.name}")
                        return output_path
                    except Exception as move_err:
                        logging.error(f"Error moving temp thumbnail: {move_err}")
                return None
        
    except Exception as e:
        logging.error(f"Error generating custom thumbnail for {video_path.name}: {e}", exc_info=True)
        return None

def create_short_clip_fallback(input_path, output_path, start_time, end_time, focus_points=None):
    """Fallback function to create a short clip without relying on video_processor.py"""
    try:
        from moviepy.editor import VideoFileClip
        
        logging.info(f"Using fallback create_short_clip implementation")
        # Simple trimming
        clip = VideoFileClip(input_path).subclip(start_time, end_time)
        
        # Resize to vertical format if needed
        width, height = clip.size
        target_aspect_ratio = 9 / 16
        
        if width / height > target_aspect_ratio:
            # Original is wider than target - crop sides
            new_width = int(height * target_aspect_ratio)
            # Center crop by default
            x1 = (width - new_width) // 2
            cropped_clip = clip.crop(x1=x1, width=new_width)
        else:
            # Original is already tall enough - no need to crop width
            cropped_clip = clip
            
        # Resize to standard short resolution (1080x1920)
        target_height = 1920
        target_width = 1080
        final_clip = cropped_clip.resize(height=target_height)
        
        # Write output
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            bitrate='5000k',
            threads=4
        )
        
        # Clean up
        clip.close()
        cropped_clip.close()
        final_clip.close()
        
        # Explicitly check if file exists and has size
        path_obj = pathlib.Path(output_path)
        success = path_obj.exists() and path_obj.stat().st_size > 0
        if success:
            logging.info(f"Fallback create_short_clip created file successfully: {output_path}")
            logging.info(f"File size: {path_obj.stat().st_size / (1024*1024):.2f} MB")
        else:
            logging.error(f"Fallback create_short_clip failed to create valid file: {output_path}")
        
        return success
    except Exception as e:
        logging.error(f"Error in fallback create_short_clip: {e}", exc_info=True)
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YouTube Video Creator")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--skip-download", action="store_true", help="Skip the Reddit download step")
    parser.add_argument("--skip-safety-check", action="store_true", help="Skip content safety checks")
    parser.add_argument("--limit", type=int, default=1, help="Number of videos to process (default: 1)")
    parser.add_argument("--sleep", type=int, default=600, help="Sleep duration between uploads in seconds (default: 600)")
    parser.add_argument("--subreddit", type=str, default=None, help="Process specific subreddit (default: random from list)")
    parser.add_argument("--resolution", type=str, default="720p", help="Video resolution: 480p, 720p, 1080p (default: 720p)")
    parser.add_argument("--test", action="store_true", help="Run tests only and exit")
    return parser.parse_args()

def main():
    logging.info("Starting YouTube Monetization-Ready Video Processor")
    start_time_total = time.time()

    # Print PyTorch CUDA information
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
        logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logging.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("CUDA is not available. Using CPU for PyTorch operations.")

    # Parse command-line arguments
    args = parse_args()
    
    if args.debug:
        # Set up more detailed logging
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug mode enabled")
    
    # Make sure MoviePy is properly installed
    if not fix_moviepy_installation():
        logging.error("Failed to fix MoviePy installation. The script may encounter errors.")
    
    # Check and configure ImageMagick for MoviePy
    check_and_configure_imagemagick()
    
    # Ensure ffmpeg-normalize is installed and in PATH
    ensure_ffmpeg_normalize_installed()
    
    # Run system tests
    if not run_system_tests():
        logging.warning("Some system tests failed. The script may encounter errors during execution.")
    
    # If test mode, exit after tests
    if args.test:
        logging.info("Test mode: Exiting after running system tests.")
        return
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Setup database
    setup_database()
    
    # Clear temporary files from previous runs
    clean_temp_files()

def get_music_attribution(music_metadata: Dict) -> str:
    """Returns attribution string for used music."""
    if not music_metadata:
        return ""
    attribution = "\nMusic Attribution:\n"
    if title := music_metadata.get('title'): attribution += f"Title: {title}\n"
    if artist := music_metadata.get('artist'): attribution += f"Artist: {artist}\n"
    if license := music_metadata.get('license'): attribution += f"License: {license}\n"
    if source := music_metadata.get('source'): attribution += f"Source: {source}\n"
    return attribution.strip()

def select_music_for_content(analysis: Dict) -> Tuple[Optional[pathlib.Path], Dict]:
    """Selects appropriate background music from the music folder."""
    if not MUSIC_FOLDER.exists():
        logging.warning("Music folder not found. Cannot select music.")
        return None, {}
    
    # First try to select music by mood from analysis
    preferred_genres = analysis.get("music_genres", [])
    
    # Find music files matching preferred genres/categories
    matching_files = []
    for genre in preferred_genres:
        genre_lower = genre.lower()
        # Look in all category folders
        for category in MUSIC_CATEGORIES:
            if any(keyword in genre_lower for keyword in MUSIC_CATEGORIES.get(category, [])):
                category_folder = MUSIC_FOLDER / category
                if category_folder.exists():
                    matching_files.extend(list(category_folder.glob("*.mp3")))
    
    # If no matches, get all music files from all categories
    if not matching_files:
        for category in MUSIC_CATEGORIES:
            category_folder = MUSIC_FOLDER / category
            if category_folder.exists():
                matching_files.extend(list(category_folder.glob("*.mp3")))
    
    # If still no matches, look for MP3 files directly in the music folder
    if not matching_files:
        matching_files = list(MUSIC_FOLDER.glob("*.mp3"))
    
    # If no music files found, return empty
    if not matching_files:
        logging.warning("No suitable music files found in any folder.")
        return None, {}
    
    # Randomly select a music file
    selected_music = random.choice(matching_files)
    
    # Basic metadata extraction
    metadata = {
        'title': selected_music.stem,
        'source': f"Music Collection ({selected_music.parent.name})"
    }
    
    logging.info(f"Selected music: {selected_music.name}")
    return selected_music, metadata

def verify_music_files():
    """Verifies the existing music folder and available music files."""
    if not MUSIC_FOLDER.exists():
        logging.warning(f"Music folder {MUSIC_FOLDER} does not exist.")
        return
        
    music_files = list(MUSIC_FOLDER.glob("*.mp3"))
    num_files = len(music_files)
    
    if num_files == 0:
        logging.warning("No music files found in music folder. Background music will be disabled.")
    else:
        logging.info(f"Found {num_files} music file(s) in music folder.")
        for music_file in music_files:
            logging.info(f"Available music: {music_file.name}")

# Add a new function for better YouTube token handling
def load_youtube_token(token_file_path):
    """Load and validate YouTube token file, returning credentials or None."""
    if not os.path.exists(token_file_path):
        logging.error(f"YouTube token file not found: {token_file_path}")
        return None
        
    try:
        # Read and parse the token file
        with open(token_file_path, 'r') as f:
            token_data = json.load(f)
            
        # Basic validation
        required_fields = ['client_id', 'client_secret', 'refresh_token']
        missing_fields = [field for field in required_fields if field not in token_data]
        if missing_fields:
            logging.error(f"YouTube token missing required fields: {missing_fields}")
            return None
            
        # Create credentials
        from google.oauth2.credentials import Credentials
        credentials = Credentials.from_authorized_user_info(token_data, YOUTUBE_SCOPES)
        
        # Refresh if needed
        if credentials.expired and credentials.refresh_token:
            from google.auth.transport.requests import Request
            credentials.refresh(Request())
            # Save refreshed token
            with open(token_file_path, 'w') as f:
                token_json = credentials.to_json()
                # Make sure we have a json object, not a string
                token_obj = json.loads(token_json) if isinstance(token_json, str) else token_json
                json.dump(token_obj, f, indent=4)
            logging.info("Refreshed expired YouTube token")
            
        return credentials
    except json.JSONDecodeError as e:
        logging.error(f"YouTube token file contains invalid JSON: {e}")
    except Exception as e:
        logging.error(f"Error loading YouTube token: {e}")
    return None

def ensure_ffmpeg_normalize_installed():
    """Checks if ffmpeg-normalize is installed, and installs it if missing."""
    try:
        # Check if ffmpeg-normalize is installed and working
        result = subprocess.run(['ffmpeg-normalize', '-version'], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                               check=False)
        if result.returncode == 0:
            logging.info("ffmpeg-normalize is already installed and working.")
            return True
    except (FileNotFoundError, subprocess.SubprocessError):
        logging.warning("ffmpeg-normalize not found. Will attempt to install it.")
    
    # Try to install ffmpeg-normalize using pip
    try:
        logging.info("Installing ffmpeg-normalize via pip...")
        
        # Get Python Scripts directory
        is_windows = os.name == 'nt'
        python_exe_dir = os.path.dirname(sys.executable)
        
        # For virtual environments on Windows, sys.executable is already in the Scripts directory
        # So scripts_dir should be the same as python_exe_dir, not python_exe_dir/Scripts/Scripts
        scripts_dir = python_exe_dir
        
        # Log paths for debugging
        logging.info(f"Python executable path: {sys.executable}")
        logging.info(f"Scripts directory: {scripts_dir}")
        
        # Install ffmpeg-normalize
        pip_cmd = [sys.executable, '-m', 'pip', 'install', 'ffmpeg-normalize']
        
        # Suppress output to avoid encoding issues on Windows
        process = subprocess.run(
            pip_cmd,
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        
        # Verify installation - for Windows, check in Scripts directory
        try:
            if is_windows:
                ffmpeg_normalize_path = os.path.join(scripts_dir, 'ffmpeg-normalize.exe')
                
                # Check if the file exists and log the result
                exists = os.path.exists(ffmpeg_normalize_path)
                logging.info(f"ffmpeg-normalize.exe path: {ffmpeg_normalize_path}")
                logging.info(f"ffmpeg-normalize.exe exists: {exists}")
                
                # List all files in the Scripts directory
                script_files = os.listdir(scripts_dir)
                logging.info(f"Files in Scripts directory: {', '.join(script_files)}")
                
                # Check for any file matching ffmpeg-normalize.*
                normalize_files = [f for f in script_files if f.startswith('ffmpeg-normalize')]
                if normalize_files:
                    logging.info(f"Found ffmpeg-normalize files: {normalize_files}")
                    ffmpeg_normalize_path = os.path.join(scripts_dir, normalize_files[0])
                    
                if exists or normalize_files:
                    logging.info(f"Found ffmpeg-normalize at: {ffmpeg_normalize_path}")
                    # Update PATH to include Scripts directory
                    os.environ['PATH'] = scripts_dir + os.pathsep + os.environ['PATH']
                    return True
                else:
                    logging.error(f"ffmpeg-normalize not found in {scripts_dir} after installation")
            else:
                # For non-Windows, try regular verification
                verify_result = subprocess.run(
                    ['ffmpeg-normalize', '-version'], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL, 
                    check=True
                )
                logging.info("Successfully installed ffmpeg-normalize")
                return True
                
        except (FileNotFoundError, subprocess.SubprocessError) as e:
            logging.error(f"Failed to verify ffmpeg-normalize installation: {e}")
            return False
    except Exception as e:
        logging.error(f"Failed to install ffmpeg-normalize: {e}")
        return False

def check_and_configure_imagemagick():
    """
    Checks if ImageMagick is installed and properly configured for MoviePy.
    Attempts to find the correct path and update MoviePy's settings.
    """
    from moviepy.config import change_settings

    image_magick_found = False
    
    # Define potential ImageMagick paths based on OS
    if os.name == 'nt':  # Windows
        IMAGEMAGICK_PATHS = [
            r"C:\Program Files\ImageMagick-*\magick.exe",
            r"C:\Program Files (x86)\ImageMagick-*\magick.exe",
            os.path.expandvars(r"%ProgramFiles%\ImageMagick-*\magick.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\ImageMagick-*\magick.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\ImageMagick\magick.exe"),
            "magick"  # If in PATH
        ]
    else:  # Linux/Mac
        IMAGEMAGICK_PATHS = [
            "/usr/bin/convert",
            "/usr/local/bin/convert",
            "convert"  # If in PATH
        ]
    
    for path_pattern in IMAGEMAGICK_PATHS:
        try:
            # Handle glob patterns (with *)
            if '*' in path_pattern:
                resolved_paths = glob.glob(path_pattern)
            else:
                resolved_paths = [path_pattern]
                
            for path in resolved_paths:
                try:
                    change_settings({"IMAGEMAGICK_BINARY": path})
                    # Test if it works by running a simple command
                    cmd = [path, "-version"] if os.name == 'nt' else ["convert", "-version"]
                    subprocess.run(cmd, check=True, capture_output=True, timeout=5)
                    logging.info(f"ImageMagick found and working at: {path}")
                    image_magick_found = True
                    break
                except Exception:
                    change_settings({"IMAGEMAGICK_BINARY": ""})  # Reset on failure
                    continue
            if image_magick_found:
                break
        except Exception as e:
            logging.warning(f"Error checking ImageMagick path {path_pattern}: {e}")
            continue
    
    if not image_magick_found:
        logging.warning("ImageMagick not found or not working. Text overlays might be basic or fail.")
        
        # Attempt to install on compatible systems
        if os.name != 'nt':  # Linux or Mac
            try:
                logging.info("Attempting to install ImageMagick...")
                if os.path.exists("/usr/bin/apt"):  # Debian/Ubuntu
                    subprocess.run(["sudo", "apt", "update"], check=True)
                    subprocess.run(["sudo", "apt", "install", "-y", "imagemagick"], check=True)
                elif os.path.exists("/usr/bin/yum"):  # CentOS/RHEL
                    subprocess.run(["sudo", "yum", "install", "-y", "ImageMagick"], check=True)
                elif os.path.exists("/usr/bin/brew"):  # Mac with Homebrew
                    subprocess.run(["brew", "install", "imagemagick"], check=True)
                logging.info("ImageMagick installation attempt completed. Please check if it works.")
            except Exception as install_err:
                logging.error(f"Failed to install ImageMagick: {install_err}")
        else:
            logging.info("For Windows, please manually install ImageMagick from: https://imagemagick.org/script/download.php")
    
    return image_magick_found

def run_system_tests():
    """
    Run basic tests to verify the system is properly set up
    and common components are working.
    """
    logging.info("Running system tests...")
    tests_passed = True
    
    # Test 1: Check essential imports
    import_tests = [
        ("cv2", "OpenCV"),
        ("PIL", "PIL/Pillow"),
        ("torch", "PyTorch"),
        ("google", "Google API"),
        ("praw", "PRAW (Reddit)"),
        # ("mutagen", "Mutagen (Audio)"), # Mutagen not directly used, sf is
        ("soundfile", "SoundFile"),
        ("transformers", "Transformers"),
        # ("elevenlabs", "ElevenLabs"), # ElevenLabs not used, suno/bark is
        ("moviepy.editor", "MoviePy")
    ]
    
    for module_name, friendly_name in import_tests:
        try:
            if module_name == "moviepy.editor":
                from moviepy.editor import VideoFileClip
            else:
                __import__(module_name)
            logging.info(f"✓ {friendly_name} import test passed")
        except ImportError:
            tests_passed = False
            logging.error(f"✗ {friendly_name} import test failed")
    
    # Test 2: Check temp directory can be created/written
    temp_dir_test_path = TEMP_DIR / f"test_{int(time.time())}" # Use main TEMP_DIR
    try:
        temp_dir_test_path.mkdir(parents=True, exist_ok=True)
        test_file = temp_dir_test_path / "test.txt"
        test_file.write_text("Test content")
        test_file.unlink()  # Delete the test file
        temp_dir_test_path.rmdir()  # Remove test directory
        logging.info("✓ Filesystem write test passed")
    except Exception as e:
        tests_passed = False
        logging.error(f"✗ Filesystem write test failed: {e}")
    
    # Test 3: Check GPU
    try:
        if torch.cuda.is_available():
            # Try to create and manipulate a small tensor on GPU
            test_tensor = torch.ones(10, 10).cuda()
            test_result = (test_tensor + 1).sum().item()
            if test_result == 200.0:  # 10x10 of 2s = 200
                logging.info("✓ GPU tensor test passed")
            else:
                tests_passed = False
                logging.warning(f"✗ GPU tensor test result unexpected: {test_result}")
        else:
            logging.warning("GPU not available - skipping GPU tests")
    except Exception as e:
        tests_passed = False
        logging.error(f"✗ GPU test failed: {e}")
    
    # Final result
    if tests_passed:
        logging.info("All system tests passed successfully!")
    else:
        logging.warning("Some system tests failed. Check logs above for details.")
    
    return tests_passed

def clean_temp_files(directory: pathlib.Path = TEMP_DIR):
    """Cleans up files in the specified temporary directory."""
    if directory.exists():
        logging.info(f"Cleaning up temporary files in {directory}...")
        cleaned_count = 0
        for item in directory.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                    cleaned_count +=1
                elif item.is_dir():
                    shutil.rmtree(item)
                    cleaned_count +=1
            except Exception as e:
                logging.warning(f"Could not remove temp item {item}: {e}")
        logging.info(f"Cleaned {cleaned_count} items from temp directory.")


if __name__ == "__main__":
    # Handle graceful exit for Ctrl+C
    def signal_handler(sig, frame):
        logging.info('Ctrl+C received. Shutting down gracefully...')
        close_database() # Ensure DB is closed
        # Add any other cleanup needed
        clean_temp_files() # Clean temp files on exit
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    main_start_time = time.time()
    processed_count = 0
    args_main = parse_args() # Parse args once for the main loop

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
        
        temp_files_to_clean_master = [] # Keep track of all temp files across iterations

        for i in range(args_main.limit):
            logging.info(f"\n--- Starting video processing cycle {i+1} of {args_main.limit} ---")
            iteration_start_time = time.time()
            
            current_subreddit = random.choice(subreddits_to_process)
            logging.info(f"Selected subreddit: r/{current_subreddit}")

            submissions = get_reddit_submissions(current_subreddit, limit=5) # Fetch a few to pick from
            if not submissions:
                logging.warning(f"No suitable submissions found in r/{current_subreddit}. Skipping this cycle.")
                if args_main.limit > 1: time.sleep(30) # Short sleep if in loop
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
                if args_main.limit > 1 and i < args_main.limit -1 :
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
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                    # Clean up file list
                    cleanup_temp_files(file_list_path)
                    
                    if not concatenated_path.is_file() or concatenated_path.stat().st_size == 0:
                        logging.error(f"Failed to concatenate segments. Falling back to first segment.")
                        cropped_segment_path = segment_clips[0]
                    else:
                        logging.info(f"Successfully concatenated {len(segment_clips)} segments.")
                        cropped_segment_path = concatenated_path
                        
                except Exception as e:
                    logging.error(f"Error concatenating segments: {e}. Falling back to first segment.")
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
            if youtube_ok and YOUTUBE_TOKEN_FILE_PATH and os.path.exists(YOUTUBE_TOKEN_FILE_PATH) :
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
                    processed_count +=1
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
            logging.info(f"--- Cycle {i+1} completed in {iteration_time:.2f} seconds ---")

            if args_main.limit > 1 and i < args_main.limit - 1:
                logging.info(f"Sleeping for {args_main.sleep} seconds before next cycle...")
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
    main()