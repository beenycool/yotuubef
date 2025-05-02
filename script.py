import argparse
import base64
import json
import math
import os
import signal
import sys
import shutil
import pathlib
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
import glob
from typing import Optional, List, Dict, Tuple, Any, Union
import gc # Added for explicit garbage collection

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
from moviepy.audio.AudioClip import AudioArrayClip # Added import

# Import the _prepare_initial_video function from video_processor module
from video_processor import _prepare_initial_video, create_short_clip

# --- Configuration ---
# ImageMagick configuration with comprehensive path search
IMAGEMAGICK_PATHS = [
    r"C:\Program Files\ImageMagick-*\magick.exe",  # Wildcard for any version
    r"C:\Program Files (x86)\ImageMagick-*\magick.exe",
    os.path.expandvars(r"%ProgramFiles%\ImageMagick-*\magick.exe"),
    os.path.expandvars(r"%ProgramFiles(x86)%\ImageMagick-*\magick.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\ImageMagick\magick.exe"),
    "magick"  # Try system path as last resort
]

image_magick_found = False
found_path = None

# Try all paths including wildcards
for path_pattern in IMAGEMAGICK_PATHS:
    if '*' in path_pattern:
        # Handle wildcard paths
        try:
            matches = glob.glob(path_pattern)
            for path in matches:
                try:
                    change_settings({"IMAGEMAGICK_BINARY": path})
                    subprocess.run([path, "-version"], check=True, capture_output=True)
                    image_magick_found = True
                    found_path = path
                    print(f"ImageMagick found at: {path}")
                    break
                except Exception:
                    continue
            if image_magick_found:
                break
        except Exception:
            continue
    else:
        # Handle exact paths
        try:
            change_settings({"IMAGEMAGICK_BINARY": path_pattern})
            subprocess.run([path_pattern, "-version"], check=True, capture_output=True)
            image_magick_found = True
            found_path = path_pattern
            print(f"ImageMagick found at: {path_pattern}")
            break
        except Exception:
            continue

if not image_magick_found:
    print("""
    WARNING: ImageMagick not found or not working properly.
    Text overlays and other features will not work.
    
    Possible solutions:
    1. Install ImageMagick from https://imagemagick.org/
    2. If already installed, ensure it's added to your system PATH
    3. Or manually specify the path to magick.exe in script.py
    
    Current search paths tried:
    """)
    for path in IMAGEMAGICK_PATHS:
        print(f"    - {path}")

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

# Directories and Files
BASE_DIR = pathlib.Path(__file__).parent.resolve()
TEMP_DIR = BASE_DIR / "temp_processing"
MUSIC_FOLDER = BASE_DIR / "music"
WATERMARK_PATH = BASE_DIR / "watermark.png"
DB_FILE = BASE_DIR / 'uploaded_videos.db'

# Video Parameters
TARGET_VIDEO_DURATION_SECONDS = 60
TARGET_ASPECT_RATIO = 9 / 16
TARGET_RESOLUTION = (1080, 1920)
TARGET_FPS = 30
AUDIO_CODEC = 'aac'
VIDEO_CODEC_CPU = 'libx264'
VIDEO_CODEC_GPU = 'h264_nvenc'
FFMPEG_CPU_PRESET = 'medium' # Faster preset for CPU
FFMPEG_GPU_PRESET = 'p5' # Balanced preset for GPU
FFMPEG_CRF_CPU = '23'
FFMPEG_CQ_GPU = '23'
LOUDNESS_TARGET_LUFS = -14 # Adjusted target for better compatibility

# Text Overlay Parameters (Narrative Style)
NARRATIVE_FONT = 'Impact' # Ensure this font is installed or provide path
NARRATIVE_FONT_SIZE_RATIO = 1 / 10
NARRATIVE_TEXT_COLOR = 'white'
NARRATIVE_STROKE_COLOR = 'black'
NARRATIVE_STROKE_WIDTH = 4
NARRATIVE_POSITION = ('center', 'center')
NARRATIVE_BG_COLOR = 'transparent'

# Subtitle Parameters (Optional, if keeping dialogue subtitles)
OVERLAY_FONT = 'Arial' # Changed to more common font
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
APPLY_STABILIZATION = False # Keep disabled unless vidstab is reliable
MIX_ORIGINAL_AUDIO = False # Narrative style usually replaces original audio
ORIGINAL_AUDIO_MIX_VOLUME = 0.1 # Very low if mixed
BACKGROUND_MUSIC_ENABLED = True
BACKGROUND_MUSIC_VOLUME = 0.08 # Lower default volume
BACKGROUND_MUSIC_NARRATIVE_VOLUME_FACTOR = 0.1 # Multiplier when narrative TTS is present
AUDIO_DUCKING_FADE_TIME = 0.3 # Less relevant if not mixing original audio heavily
SHAKE_EFFECT_ENABLED = True
SUBTLE_ZOOM_ENABLED = True
COLOR_GRADE_ENABLED = True
PARALLEL_FRAME_PROCESSING = True # Use if beneficial
N_JOBS_PARALLEL = max(1, os.cpu_count() // 2) # Adjust based on system

# Content Filtering
FORBIDDEN_WORDS = [
    "fuck", "fucking", "fucked", "fuckin", "shit", "shitty", "shitting",
    "damn", "damned", "bitch", "bitching", "cunt", "asshole", "piss", "pissed"
]

# --- Content Filtering ---
# Enhanced list of forbidden words for filtering
FORBIDDEN_WORDS = [
    # Profanity
    "fuck", "fucking", "fucked", "fuckin", "shit", "shitty", "shitting",
    "damn", "damned", "bitch", "bitching", "cunt", "asshole", "piss", "pissed",
    # Violence
    "gore", "graphic", "brutal", "blood", "bloody", "murder", "killing", "suicide",
    # Sexual
    "porn", "pornographic", "nsfw", "xxx", "sex", "sexual", "nude", "naked",
    # Hate speech
    "racist", "racism", "nazi", "sexist", "homophobic", "slur"
]

# Add a list of unsuitable content types
UNSUITABLE_CONTENT_TYPES = [
    "gore", "violence", "graphic injury", "animal abuse", "child abuse",
    "pornography", "nudity", "sexual content", "hate speech", "racism",
    "dangerous activities", "suicide", "self-harm", "illegal activities",
    "drug abuse", "excessive profanity"
]

def is_unsuitable_video(submission, video_path: Optional[pathlib.Path] = None) -> Tuple[bool, str]:
    """
    Checks if a video is unsuitable for processing.
    
    Args:
        submission: Reddit submission
        video_path: Path to downloaded video if available
        
    Returns:
        Tuple of (is_unsuitable, reason)
    """
    # Check NSFW flag
    if submission.over_18:
        return True, "NSFW content flagged"
        
    # Check title for forbidden words
    if contains_forbidden_words(submission.title):
        return True, "Title contains forbidden words"
        
    # Check subreddit name for unsuitable indicators
    unsuitable_subreddit_indicators = ['nsfw', 'porn', 'gore', 'death', 'wtf']
    if any(indicator in submission.subreddit.display_name.lower() for indicator in unsuitable_subreddit_indicators):
        return True, f"Unsuitable subreddit: r/{submission.subreddit.display_name}"
    
    # Check submission flair for unsuitable indicators
    if submission.link_flair_text and any(word in submission.link_flair_text.lower() for word in FORBIDDEN_WORDS):
        return True, f"Unsuitable flair: {submission.link_flair_text}"
    
    # If comments are available, check top comments for unsuitable content indicators
    # MODIFIED: Added timeout and better error handling to prevent hanging
    try:
        # Add a timeout mechanism to prevent hanging
        import threading
        import queue

        def fetch_comments():
            try:
                # Limit the number of comments to replace to avoid long operations
                submission.comments.replace_more(limit=0)
                # Only fetch the first few comments to avoid excessive API calls
                comment_list = list(submission.comments.list()[:5])
                comment_queue.put(comment_list)
            except Exception as e:
                comment_queue.put(e)

        comment_queue = queue.Queue()
        comment_thread = threading.Thread(target=fetch_comments)
        comment_thread.daemon = True
        comment_thread.start()
        
        # Wait for 5 seconds maximum
        comment_thread.join(timeout=5)
        
        if comment_thread.is_alive():
            # Thread is still running after timeout
            print("  Comment retrieval timed out, skipping comment analysis")
        else:
            # Get result from queue
            result = comment_queue.get(block=False)
            
            if isinstance(result, Exception):
                print(f"  Error retrieving comments: {result}")
            else:
                # Process the comments
                top_comments = [comment.body for comment in result]
                top_comments_text = " ".join(top_comments)
                
                # Check if any comments mention unsuitable content
                if any(content_type.lower() in top_comments_text.lower() for content_type in UNSUITABLE_CONTENT_TYPES):
                    return True, "Comments suggest unsuitable content"
                
    except Exception as e:
        print(f"  Error analyzing comments: {e}")
        # Continue processing even if comment analysis fails
    
    # If video is available, perform basic video analysis
    if video_path and video_path.is_file():
        try:
            # Check duration
            duration, width, height = get_video_details(video_path)
            
            # Videos that are too short or too long might be unsuitable
            if duration < 3:
                return True, f"Video too short ({duration:.1f}s)"
            
            # Check resolution - very low res might be unsuitable
            if width > 0 and height > 0 and (width < 240 or height < 240):
                return True, f"Video resolution too low ({width}x{height})"
        except Exception as e:
            print(f"  Error analyzing video file: {e}")
            
    return False, ""

# YouTube Upload Parameters
YOUTUBE_SCOPES = ['https://www.googleapis.com/auth/youtube.upload', 'https://www.googleapis.com/auth/youtube.force-ssl']
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
YOUTUBE_UPLOAD_CATEGORY_ID = '24' # Entertainment Category
YOUTUBE_UPLOAD_PRIVACY_STATUS = 'public' # Change to 'public' or 'unlisted' for release

# --- Global Variables ---
db_conn: Optional[sqlite3.Connection] = None
db_cursor: Optional[sqlite3.Cursor] = None
elevenlabs_client: Optional[ElevenLabs] = None
gemini_model: Optional[genai.GenerativeModel] = None
reddit: Optional[praw.Reddit] = None
youtube_service: Optional[Any] = None

# --- Helper Classes ---
class UploadLimitExceededError(Exception):
    pass

FALLBACK_ANALYSIS = {
    'fallback': True,
    'suggested_title': 'Interesting Reddit Clip',
    'summary_for_description': 'Check out this video from Reddit!',
    'mood': 'neutral',
    'best_segment': None,
    'key_visual_moments': [],
    'speech_segments': [],
    'narrative_script': [], # Add narrative script field to fallback
    'hashtags': ['#reddit', '#shorts', '#video'],
    'original_duration': 0.0
}

# --- Utility Functions ---
def check_ffmpeg_install(command: str) -> bool:
    try:
        return shutil.which(command) is not None
    except Exception:
        return False

def has_nvidia_gpu() -> bool:
    if sys.platform == "win32": command = ['nvidia-smi']
    else: command = ['nvidia-smi']
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
        return result.returncode == 0 and "NVIDIA-SMI" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False

# Enhanced GPU detection - check for specific encoders
def check_gpu_encoder_availability() -> str:
    """Check which GPU encoder is available on the system (NVIDIA, AMD, or Intel)"""
    try:
        # Check for NVIDIA encoder
        if has_nvidia_gpu():
            # Verify h264_nvenc encoder is available
            cmd = ["ffmpeg", "-encoders"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
            if "h264_nvenc" in result.stdout:
                print("NVIDIA h264_nvenc encoder detected")
                return "h264_nvenc"
            elif "hevc_nvenc" in result.stdout:
                print("NVIDIA hevc_nvenc encoder detected")
                return "hevc_nvenc"
        
        # Check for AMD encoder
        cmd = ["ffmpeg", "-encoders"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
        if "h264_amf" in result.stdout:
            print("AMD h264_amf encoder detected")
            return "h264_amf"
        
        # Check for Intel encoder
        if "h264_qsv" in result.stdout:
            print("Intel h264_qsv encoder detected")
            return "h264_qsv"
        
        # Fallback to CPU
        print("No hardware encoder detected, using CPU encoding")
        return VIDEO_CODEC_CPU
    except Exception as e:
        print(f"Error detecting GPU encoder: {e}")
        return VIDEO_CODEC_CPU

def cleanup_temp_files(*file_paths: Optional[pathlib.Path]):
    for file_path in file_paths:
        if file_path and file_path.is_file():
            try: file_path.unlink()
            except OSError as e: print(f"  - Warning: Error removing {file_path.name}: {e}")

def contains_forbidden_words(text: str) -> bool:
    if not text: return False
    text_lower = text.lower()
    return any(word in text_lower for word in FORBIDDEN_WORDS)

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.replace(" ", "_")
    return name[:100]

def get_video_details(video_path: pathlib.Path) -> Tuple[float, int, int]:
    if not video_path.is_file() or not check_ffmpeg_install("ffprobe"): return 0.0, 0, 0
    try:
        command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,duration,r_frame_rate:format=duration', '-of', 'json', str(video_path)]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        data = json.loads(result.stdout)
        duration, width, height = 0.0, 0, 0
        if 'streams' in data and data['streams']:
            stream = data['streams'][0]
            width = int(stream.get('width', 0)); height = int(stream.get('height', 0))
            if 'duration' in stream and stream['duration'] != 'N/A': duration = float(stream['duration'])
        if duration <= 0 and 'format' in data and 'duration' in data['format'] and data['format']['duration'] != 'N/A': duration = float(data['format']['duration'])
        return duration, width, height
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        print(f"Error getting video details for {video_path.name}: {e}")
        return 0.0, 0, 0

def has_audio_track(media_path: pathlib.Path) -> bool:
    if not media_path.is_file() or not check_ffmpeg_install("ffprobe"): return False
    try:
        command = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', str(media_path)]
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
        return result.returncode == 0 and 'audio' in result.stdout.strip()
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"Error checking audio track in {media_path.name}: {e}")
        return False

# --- Setup Functions ---
def validate_environment():
    essential_vars = {'REDDIT_CLIENT_ID': REDDIT_CLIENT_ID, 'REDDIT_CLIENT_SECRET': REDDIT_CLIENT_SECRET, 'GOOGLE_CLIENT_SECRETS_FILE': GOOGLE_CLIENT_SECRETS_FILE, 'GEMINI_API_KEY': GEMINI_API_KEY}
    missing_vars = [name for name, value in essential_vars.items() if not value]
    if missing_vars: raise ValueError(f"Missing essential environment variables: {', '.join(missing_vars)}")
    if not GOOGLE_CLIENT_SECRETS_FILE or not os.path.exists(GOOGLE_CLIENT_SECRETS_FILE): raise FileNotFoundError(f"Google Client Secrets file not found: {GOOGLE_CLIENT_SECRETS_FILE}")
    if not check_ffmpeg_install("ffmpeg"): raise RuntimeError("FFmpeg is required but not found.")
    if not check_ffmpeg_install("ffprobe"): print("Warning: ffprobe not found.")
    if not check_ffmpeg_install("ffmpeg-normalize"): print("Warning: ffmpeg-normalize not found. Audio normalization will be skipped.")
    if not ELEVENLABS_API_KEY: print("Warning: ELEVENLABS_API_KEY not set. ElevenLabs TTS will not be available.")

def setup_directories():
    for dir_path in [TEMP_DIR, MUSIC_FOLDER]: dir_path.mkdir(parents=True, exist_ok=True)

def setup_database():
    global db_conn, db_cursor
    try:
        db_conn = sqlite3.connect(DB_FILE); db_cursor = db_conn.cursor()
        db_cursor.execute('CREATE TABLE IF NOT EXISTS uploads (reddit_url TEXT PRIMARY KEY, youtube_url TEXT, upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
        db_conn.commit()
    except sqlite3.Error as e: raise RuntimeError(f"Database setup failed: {e}") from e

def close_database():
    global db_conn
    if db_conn:
        try: db_conn.close()
        except sqlite3.Error as e: print(f"Warning: Error closing database: {e}")
        finally: db_conn = None

def setup_api_clients():
    global reddit, youtube_service, gemini_model, elevenlabs_client
    try:
        reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT, check_for_async=False)
        reddit.user.me()
    except Exception as e: raise ConnectionError(f"Failed to initialize Reddit client: {e}") from e
    try:
        flow = InstalledAppFlow.from_client_secrets_file(GOOGLE_CLIENT_SECRETS_FILE, YOUTUBE_SCOPES)
        credentials = flow.run_local_server(port=0)
        youtube_service = build_google_api(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, credentials=credentials)
    except Exception as e: raise ConnectionError(f"Failed to initialize YouTube client: {e}") from e
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_ID)
    except Exception as e: raise ConnectionError(f"Failed to initialize Gemini client: {e}") from e
    if ELEVENLABS_API_KEY:
        try: elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        except Exception as e: print(f"Warning: Failed to initialize ElevenLabs client: {e}")

# --- YouTube Upload Function ---
def upload_to_youtube(video_path: pathlib.Path, title: str, description: str,
                     category_id: str = YOUTUBE_UPLOAD_CATEGORY_ID,
                     privacy_status: str = YOUTUBE_UPLOAD_PRIVACY_STATUS) -> Optional[str]:
    """
    Uploads a video to YouTube using the YouTube Data API.
    
    Args:
        video_path: Path to the video file to upload
        title: Video title
        description: Video description
        category_id: YouTube category ID (default: Entertainment)
        privacy_status: Privacy status (public/unlisted/private)
        
    Returns:
        YouTube video URL if successful, None otherwise
    """
    if not youtube_service or not video_path.is_file():
        return None
        
    try:
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'categoryId': category_id
            },
            'status': {
                'privacyStatus': privacy_status
            }
        }
        
        media = MediaFileUpload(
            str(video_path),
            mimetype='video/mp4',
            resumable=True
        )
        
        request = youtube_service.videos().insert(
            part='snippet,status',
            body=body,
            media_body=media
        )
        
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"Upload progress: {int(status.progress() * 100)}%")
        
        print(f"Successfully uploaded video: {response['id']}")
        return f"https://youtu.be/{response['id']}"
        
    except google_api_errors.HttpError as e:
        print(f"An HTTP error occurred: {e}")
    except Exception as e:
        print(f"Error uploading to YouTube: {e}")
        
    return None

# --- Database Functions ---
def is_already_uploaded(reddit_url: str) -> bool:
    if not db_cursor: return True
    try:
        db_cursor.execute("SELECT 1 FROM uploads WHERE reddit_url = ?", (reddit_url,)); return db_cursor.fetchone() is not None
    except sqlite3.Error as e: print(f"Error checking DB for URL {reddit_url}: {e}"); return True

def add_upload_record(reddit_url: str, youtube_url: str):
    if not db_conn or not db_cursor: return
    try:
        db_cursor.execute("INSERT INTO uploads (reddit_url, youtube_url) VALUES (?, ?)", (reddit_url, youtube_url)); db_conn.commit()
    except sqlite3.IntegrityError: print(f"Warning: Record for {reddit_url} already exists.")
    except sqlite3.Error as e: print(f"Error adding DB record for {reddit_url}: {e}")

# --- Reddit & Download ---
def get_reddit_submissions(subreddit_name: str, limit: int) -> List[praw.models.Submission]:
    if not reddit: return []
    submissions = []
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for submission in subreddit.hot(limit=limit * 2):
            if len(submissions) >= limit: break
            if (submission.is_video or any(submission.url.endswith(ext) for ext in ['.mp4', '.mov', '.gifv']) or any(host in submission.url for host in ['v.redd.it', 'gfycat.com', 'streamable.com'])) and not submission.over_18 and not submission.stickied:
                submissions.append(submission)
            time.sleep(0.1)
        return submissions
    except Exception as e: print(f"Error fetching from r/{subreddit_name}: {e}"); return []

def download_media(url: str, output_path: pathlib.Path) -> Optional[pathlib.Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True); cleanup_temp_files(output_path)
    ydl_opts = {'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', 'outtmpl': str(output_path), 'quiet': True, 'no_warnings': True, 'ignoreerrors': True, 'noprogress': True, 'retries': 3, 'socket_timeout': 30, 'nocheckcertificate': True, 'merge_output_format': 'mp4'}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if output_path.is_file() and output_path.stat().st_size > 10240:
            return output_path
        else:
            base_name = output_path.stem
            for file in output_path.parent.iterdir():
                if file.stem.startswith(base_name) and file.is_file() and file.stat().st_size > 10240:
                    if file != output_path:
                        try:
                            file.rename(output_path)
                        except OSError:
                            pass
                    return output_path
            cleanup_temp_files(output_path)
            return None
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        cleanup_temp_files(output_path)
        return None

# --- AI Analysis ---
def analyze_video_with_gemini(video_path: pathlib.Path, title: str, subreddit_name: str, style_preferences: Optional[Dict] = None) -> Dict[str, Any]:
    global gemini_model
    analysis_result = FALLBACK_ANALYSIS.copy()
    if not gemini_model or not video_path.is_file():
        return {**analysis_result, "music_suggestion": "neutral", "tts_pacing": "normal"}

    try:
        duration, _, _ = get_video_details(video_path)
        if not duration or duration <= 0: return analysis_result
        analysis_result['original_duration'] = duration

        style_prompt = ""
        if style_preferences:
            if style_preferences.get("target_mood"):
                style_prompt += f" The desired mood is {style_preferences['target_mood']}."
            if style_preferences.get("pacing"):
                style_prompt += f" The desired pacing is {style_preferences['pacing']}."
            if style_preferences.get("visual_style"):
                style_prompt += f" The preferred visual style is {style_preferences['visual_style']}."

        prompt = f'''Analyze this video from Reddit (r/{subreddit_name}) and create a short, engaging vertical video (TikTok/Shorts style).
Original title: "{title}". Video duration: {duration:.2f}s.

Requirements:
1. Use ALL CAPS text overlays (1-4 words each)
2. Create concise narrative script segments
3. Suggest visual effects for key moments
4. Recommend appropriate music genres
5. Identify important audio moments

Return JSON with these fields:
{{
    "suggested_title": "string (<70 chars)",
    "summary_for_description": "string (2-3 sentences)",
    "mood": "string (from: funny, heartwarming, informative, suspenseful, action, calm, exciting, sad, shocking, weird, cringe)",
    "has_clear_narrative": "boolean",
    "original_audio_is_key": "boolean",
    "best_segment": {{
        "start": float,
        "end": float,
        "reason": "string"
    }},
    "key_focus_points": [
        {{
            "time": float,
            "point": {{"x": float, "y": float}}
        }}
    ],
    "text_overlays": [
        {{
            "text": "string (ALL CAPS)",
            "time": float,
            "duration": float
        }}
    ],
    "narrative_script": [
        {{
            "text": "string",
            "time": float,
            "duration": float
        }}
    ],
    "visual_cues": [
        {{
            "time": float,
            "suggestion": "string"
        }}
    ],
    "music_genres": ["string"],
    "key_audio_moments": [
        {{
            "time": float,
            "action": "string"
        }}
    ],
    "hashtags": ["string"],
    "original_duration": float
}}

Return ONLY valid JSON.'''

        with open(video_path, 'rb') as f: video_data = f.read()
        content_parts = [prompt, {"mime_type": "video/mp4", "data": base64.b64encode(video_data).decode('utf-8')}]
        safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]

        try:
            response = gemini_model.generate_content(content_parts, generation_config=genai.types.GenerationConfig(temperature=0.7), safety_settings=safety_settings) # Slightly higher temp
            result_text = ""
            if response.candidates:
                try: result_text = response.text
                except ValueError:
                    print(f"  Warning: Gemini response blocked or invalid. Finish Reason: {response.candidates[0].finish_reason if response.candidates else 'N/A'}")
                    if hasattr(response, 'prompt_feedback'): print(f"  Safety Feedback: {response.prompt_feedback}")
                    return analysis_result
            else: # Handle cases where response.candidates is empty or None
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                     print(f"  Content blocked: {response.prompt_feedback.block_reason}")
                 else:
                     print("  Warning: Gemini returned no candidates in response.")
                 return analysis_result


            json_match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if json_match: json_text = json_match.group(1)
            else:
                 json_start = result_text.find('{'); json_end = result_text.rfind('}')
                 if json_start == -1 or json_end == -1 or json_end < json_start:
                     print("  Warning: Could not find JSON in Gemini response.")
                     return analysis_result
                 json_text = result_text[json_start : json_end + 1]

            try:
                parsed_data = json.loads(json_text)
                if "text_overlays" not in parsed_data or not isinstance(parsed_data["text_overlays"], list):
                    parsed_data["text_overlays"] = [] # Ensure list exists, even if empty

                # Generate narrative_script from text_overlays if not provided
                if "narrative_script" not in parsed_data or not isinstance(parsed_data["narrative_script"], list):
                    parsed_data["narrative_script"] = []
                    for overlay in parsed_data["text_overlays"]:
                        parsed_data["narrative_script"].append({
                            "timestamp": overlay["timestamp"],
                            "duration": overlay["duration"],
                            "narrative_text": overlay["text"].lower().capitalize()  # Convert to natural speech
                        })
                # Add default values for new fields if not provided
                parsed_data.setdefault('music_suggestion', 'neutral')
                parsed_data.setdefault('tts_pacing', 'normal')
                parsed_data['original_duration'] = duration
                parsed_data['fallback'] = False
                return parsed_data
            except json.JSONDecodeError as json_err:
                print(f"  Error: Failed to parse Gemini JSON response: {json_err}")
                return analysis_result

        except Exception as gemini_err:
            print(f"  Error: Gemini API call failed: {gemini_err}")
            return analysis_result

    except Exception as e:
        print(f"  Error during Gemini analysis pipeline: {e}")
        return analysis_result


# --- TTS Generation ---
def generate_tts_elevenlabs(text: str, output_path: pathlib.Path, voice_id: str = DEFAULT_VOICE_ID) -> bool:
    """Generate TTS using ElevenLabs API (Primary if key exists)"""
    global elevenlabs_client
    if not elevenlabs_client or not text or not text.strip(): return False
    output_path.parent.mkdir(parents=True, exist_ok=True); cleanup_temp_files(output_path)
    try:
        audio = elevenlabs_client.generate(text=text, voice=voice_id)
        save(audio, str(output_path))
        return output_path.is_file() and output_path.stat().st_size > 500
    except ApiError as e: print(f"  ElevenLabs API Error: {e}"); return False
    except Exception as e: print(f"  Error generating TTS with ElevenLabs: {e}"); return False

def hugging_face_tts(text: str, output_path: pathlib.Path) -> bool:
    """Generate TTS using local Dia-1.6B model (Fallback)"""
    if not text or not text.strip(): return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path = output_path.with_suffix('.wav')
    cleanup_temp_files(output_path, wav_path)
    
    try:
        import torch
        import soundfile as sf
        from transformers import AutoProcessor, AutoModelForTextToSpeech
        import numpy as np
        
        print("  Generating TTS with Dia-1.6B (local)")
        model_name = "nari-labs/Dia-1.6B"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForTextToSpeech.from_pretrained(model_name)
        
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Using {device.upper()} for TTS")
        
        try:
            model = model.to(device)
        except RuntimeError:
            print("  Warning: Couldn't move model to GPU, using CPU")
            device = "cpu"
            model = model.to(device)

        # Create speaker embedding
        speaker_embeddings = torch.zeros((1, model.config.speaker_embedding_dim), device=device)

        # Process in smaller chunks if text is long
        max_length = 600
        text_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        audio_chunks = []

        for chunk in text_chunks:
            inputs = processor(text=chunk, return_tensors="pt").to(device)
            with torch.no_grad():
                speech = model.generate_speech(inputs["input_ids"], speaker_embeddings)
                audio_chunks.append(speech.cpu().numpy())
                # Clean up to prevent memory leaks
                del speech
                if device == "cuda":
                    torch.cuda.empty_cache()

        # Combine and save audio
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            sf.write(str(wav_path), full_audio, model.config.sampling_rate)

            # Convert to MP3 using ffmpeg
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', str(wav_path),
                    '-acodec', 'libmp3lame', '-q:a', '2', str(output_path)
                ], check=True, capture_output=True, timeout=30)
            except subprocess.CalledProcessError as e:
                print(f"  FFmpeg conversion failed: {e.stderr.decode()}")
                return False

            return output_path.is_file() and output_path.stat().st_size > 1024
        
        return False
        
    except ImportError:
        print("  Error: Required TTS libraries (torch, soundfile, transformers) not installed")
        return False
    except Exception as e:
        print(f"  TTS generation failed: {str(e)}")
        return False
    finally:
        cleanup_temp_files(wav_path)
        if 'torch' in locals() and 'device' in locals() and device == "cuda":
            torch.cuda.empty_cache()

def generate_tts(text: str, output_path: pathlib.Path, voice_id: str = DEFAULT_VOICE_ID) -> bool:
    """Attempts local Hugging Face TTS first, falls back to ElevenLabs if needed."""
    if hugging_face_tts(text, output_path):
        print("  Using local Hugging Face TTS (nari-labs/Dia-1.6B).")
        return True
    elif elevenlabs_client and generate_tts_elevenlabs(text, output_path, voice_id):
        print("  Falling back to ElevenLabs TTS.")
        return True
    return False

# --- Video Processing Functions ---
def trim_video(input_path: pathlib.Path, start_time: float, end_time: float, output_path: pathlib.Path) -> Optional[pathlib.Path]:
    """Trim video to the specified time range using FFmpeg directly for speed"""
    if not input_path.is_file(): return None
    output_path.parent.mkdir(parents=True, exist_ok=True); cleanup_temp_files(output_path)
    
    try:
        # Check if we can use GPU acceleration
        hw_accel = ['-hwaccel', 'cuda'] if has_nvidia_gpu() else []
        
        # Use FFmpeg directly for faster trimming
        command = ['ffmpeg', '-y'] + hw_accel + [
            '-i', str(input_path),
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c:v', VIDEO_CODEC_GPU if has_nvidia_gpu() else VIDEO_CODEC_CPU,
            '-c:a', 'copy',
            '-threads', str(N_JOBS_PARALLEL),
            str(output_path)
        ]
        
        subprocess.run(command, check=True, capture_output=True, timeout=300)
        
        if output_path.is_file() and output_path.stat().st_size > 1024:
            return output_path
        else:
            print("  Warning: Trimmed video file seems invalid")
            return None
    except Exception as e:
        print(f"  Error trimming video: {e}")
        cleanup_temp_files(output_path)
        return None

def apply_shake(clip: VideoClip) -> VideoClip:
    """Apply subtle camera shake effect to the video if available"""
    if not SHAKE_EFFECT_ENABLED:
        return clip
        
    try:
        # Skip for very short clips
        if clip.duration < 3:
            return clip
            
        # Check if shake effect is available
        if not hasattr(vfx, 'shake'):
            print("  Shake effect not available in this moviepy version - skipping")
            return clip
             
        # Random shake intensity
        shake_intensity = random.uniform(2, 8)
        
        # Apply shake effect using moviepy's vfx module
        return vfx.shake(clip, displacement_range=shake_intensity, shake_duration=0.1)
    except Exception as e:
        print(f"  Error applying shake effect: {e}")
        return clip

def process_video_with_gpu_optimization(processing_path: pathlib.Path,
                                      text_overlays: List[Dict],
                                      narrative_script: List[Dict], 
                                      original_audio_is_key: bool, 
                                      final_path: pathlib.Path,
                                      temp_files_list: List[pathlib.Path]) -> bool:
    """
    Process video with effects using GPU acceleration where possible.
    Now handles dynamic narrative overlays and TTS audio mixing based on AI analysis.
    """
    try:
        # First check if video has valid fps before loading with MoviePy
        fps_check_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                          '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', 
                          str(processing_path)]
        try:
            fps_result = subprocess.run(fps_check_cmd, capture_output=True, text=True, check=True)
            fps_str = fps_result.stdout.strip()
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                video_fps = num / den if den != 0 else 0
            else:
                video_fps = float(fps_str)
                
            if video_fps <= 0:
                print("  Warning: Video has invalid FPS. Using default value.")
                video_fps = TARGET_FPS
        except Exception as e:
            print(f"  Warning: Couldn't determine video FPS: {e}. Using default.")
            video_fps = TARGET_FPS
            
        # Get video dimensions for filters
        dimension_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                         '-show_entries', 'stream=width,height', '-of', 'csv=p=0:s=x',
                         str(processing_path)]
        try:
            dimension_result = subprocess.run(dimension_cmd, capture_output=True, text=True, check=True)
            dimensions = dimension_result.stdout.strip().split('x')
            width, height = int(dimensions[0]), int(dimensions[1])
        except Exception as e:
            print(f"  Warning: Couldn't determine video dimensions: {e}. Using target resolution.")
            width, height = TARGET_RESOLUTION
            
        # Try simpler GPU acceleration approach first
        has_gpu = has_nvidia_gpu()
        
        # First, apply base effects (color grading, etc) with FFmpeg
        enhanced_path = pathlib.Path(str(final_path).replace('_final.mp4', '_enhanced.mp4'))
        
        # Select a random effect preset
        preset = random.choice(['none', 'cool', 'warm', 'vibrant', 'subtle_contrast'])
        
        # Initialize audio clips list
        audio_clips = []
        
        # Try simplified GPU approach with safer settings
        try:
            # Use safer GPU settings with minimal filters
            if has_gpu:
                print("  Attempting GPU acceleration with simplified settings...")
                # Use simpler command with more compatible settings
                ffmpeg_enhance_cmd = [
                    'ffmpeg', '-y',
                    '-i', str(processing_path),
                    '-c:v', 'h264_nvenc',
                    '-preset', 'medium',  # Use more compatible preset
                    '-b:v', '5M',        # Use bitrate instead of CRF/CQ
                    '-c:a', 'copy',
                    str(enhanced_path)
                ]
                subprocess.run(ffmpeg_enhance_cmd, check=True, capture_output=True, timeout=300)
                base_video_path = enhanced_path
                print("  GPU acceleration successful!")
            else:
                # For CPU just use the original path
                base_video_path = processing_path
                print("  Using CPU processing (no GPU detected)")
        except Exception as e:
            print(f"  GPU acceleration failed: {e}")
            print("  Falling back to CPU processing...")
            # On failure, fall back to CPU with a simple copy
            try:
                ffmpeg_enhance_cmd = [
                    'ffmpeg', '-y',
                    '-i', str(processing_path),
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '23',
                    '-c:a', 'copy',
                    str(enhanced_path)
                ]
                subprocess.run(ffmpeg_enhance_cmd, check=True, capture_output=True, timeout=300)
                base_video_path = enhanced_path
            except Exception as e2:
                print(f"  CPU processing also failed: {e2}")
                # Last resort: just use the original file
                base_video_path = processing_path
            
        # Now use MoviePy for the more complex effects like text overlays and TTS
        with VideoFileClip(str(base_video_path), fps_source="fps") as video_clip:
            # Force the fps if not properly detected
            if not hasattr(video_clip, 'fps') or video_clip.fps is None or video_clip.fps <= 0:
                video_clip.fps = video_fps
                
            print(f"  Loaded video: {video_clip.duration:.2f}s, {video_clip.size}, fps={video_clip.fps}")
            
            # Apply shake effect if enabled and not already applied with FFmpeg
            processed_clip = video_clip
            if SHAKE_EFFECT_ENABLED:
                processed_clip = apply_shake(processed_clip)
            
            # Process text overlays with enhanced styling
            text_clips_list = []
            base_font_size = int(video_clip.h * NARRATIVE_FONT_SIZE_RATIO)
            
            for overlay in text_overlays:
                # Adjust font size based on text length
                text_length = len(overlay['text'])
                font_size = base_font_size
                if text_length > 10:
                    font_size = int(base_font_size * 0.8)
                
                # Create text clip with enhanced styling
                txt_clip = TextClip(
                    overlay['text'],
                    fontsize=font_size,
                    color=NARRATIVE_TEXT_COLOR,
                    font=NARRATIVE_FONT,
                    stroke_color=NARRATIVE_STROKE_COLOR,
                    stroke_width=NARRATIVE_STROKE_WIDTH,
                    method='caption',
                    size=(video_clip.w * 0.9, None),  # Max width 90% of video
                    align='center'
                )
                txt_clip = txt_clip.set_position(NARRATIVE_POSITION)
                txt_clip = txt_clip.set_duration(overlay['duration'])
                txt_clip = txt_clip.set_start(overlay['timestamp'])
                text_clips_list.append(txt_clip)
            
            # --- Conditional Audio Processing ---
            audio_components = []

            # A. Prioritize Original Audio if Key
            if original_audio_is_key:
                print("  Prioritizing original audio.")
                if video_clip.audio:
                    # Use original audio at near full volume
                    audio_components.append(video_clip.audio.volumex(0.95)) # Keep slightly below 1.0 to avoid clipping
                
                # Add subtle background music if desired (very low volume)
                if BACKGROUND_MUSIC_ENABLED:
                    # Choose music carefully - maybe only add if original audio is sparse?
                    music_file = random.choice(list(MUSIC_FOLDER.glob('*.mp3')))
                    if music_file and music_file.is_file():
                        try:
                            # Use a much lower volume when original audio is key
                            bg_music = AudioFileClip(str(music_file)).volumex(BACKGROUND_MUSIC_VOLUME * 0.3) # e.g., 0.024
                            audio_components.append(bg_music)
                        except Exception as e:
                            print(f"  Warning: Could not load background music: {e}")
                # NO TTS in this mode
            
            # B. Use Narrative TTS if Original Audio Isn't Key (and script exists)
            else:
                print("  Using narrative TTS and background music.")
                tts_audio_clips = []
                if narrative_script: # Check if AI provided a script
                    for i, item in enumerate(narrative_script):
                        tts_text = item.get("narrative_text")
                        tts_timestamp = item.get("timestamp", 0)
                        if tts_text:
                            tts_file = TEMP_DIR / f"tts_{i}_{processing_path.stem}.mp3"
                            temp_files_list.append(tts_file)
                            if generate_tts(tts_text, tts_file):
                                try:
                                    tts_clip = AudioFileClip(str(tts_file))
                                    # Ensure TTS clip doesn't extend beyond video duration
                                    max_tts_duration = video_clip.duration - tts_timestamp
                                    if tts_clip.duration > max_tts_duration:
                                        tts_clip = tts_clip.subclip(0, max_tts_duration)
                                    tts_clip = tts_clip.set_start(tts_timestamp)
                                    tts_audio_clips.append(tts_clip)
                                except Exception as e:
                                    print(f"  Warning: Could not load TTS file {tts_file}: {e}")

                    if tts_audio_clips:
                        narrative_audio = CompositeAudioClip(tts_audio_clips).volumex(0.95) # Slightly lower volume for TTS too
                        audio_components.append(narrative_audio)

                # Add background music (ducked if TTS is present)
                if BACKGROUND_MUSIC_ENABLED:
                    music_file = random.choice(list(MUSIC_FOLDER.glob('*.mp3')))
                    if music_file and music_file.is_file():
                        try:
                            bg_music_vol = BACKGROUND_MUSIC_VOLUME # e.g., 0.08
                            if tts_audio_clips:
                                # Duck background music when TTS is playing
                                bg_music_vol *= BACKGROUND_MUSIC_NARRATIVE_VOLUME_FACTOR # e.g., 0.08 * 0.1 = 0.008 (very low)
                            bg_music = AudioFileClip(str(music_file)).volumex(bg_music_vol)
                            audio_components.append(bg_music)
                        except Exception as e:
                            print(f"  Warning: Could not load background music: {e}")

                # Always include original audio but balance with TTS
                if video_clip.audio:
                    # Set volume to 40% when TTS present, full otherwise
                    original_vol = 0.4 if tts_audio_clips else 1.0
                    audio_components.append(video_clip.audio.volumex(original_vol))
                
                # Always include TTS at moderate volume
                if tts_audio_clips:
                    for tts_clip in tts_audio_clips:
                        tts_clip = tts_clip.volumex(0.6)  # Set TTS to 60% volume
                    audio_components.extend(tts_audio_clips)
            
            # Create final audio track
            if audio_components:
                # Ensure final audio duration matches video clip duration
                final_audio = CompositeAudioClip(audio_components).set_duration(processed_clip.duration)
                processed_clip = processed_clip.set_audio(final_audio)
            else:
                processed_clip = processed_clip.without_audio()
            
            # Composite text overlays
            if text_clips_list:
                processed_clip = CompositeVideoClip([processed_clip] + text_clips_list, size=processed_clip.size) # Ensure size is set
            
            # Write out the processed video with better quality
            print("  Writing final video...")
            final_render_bitrate = '8000k' # e.g., 8 Mbps for 1080p30
            processed_clip.write_videofile(
                str(final_path),
                codec='libx264',  # Always use CPU encoding for final output
                preset='medium',
                audio_codec=AUDIO_CODEC,
                threads=N_JOBS_PARALLEL,
                fps=video_fps,  # Use the detected fps or our default
                bitrate=final_render_bitrate, # Specify bitrate
                logger='bar' # Show progress
            )
            
            # Clean up open clips and temporary files
            if audio_clips:
                for clip in audio_clips:
                    try:
                        clip.close()
                    except Exception as e:
                        print(f"  Warning: Error closing audio clip: {e}")
                        pass
                    
            if enhanced_path.exists() and enhanced_path != processing_path:
                cleanup_temp_files(enhanced_path)
                
            return True
    except Exception as e:
        print(f"  Error in video processing: {e}")
        traceback.print_exc()
        return False
    finally:
        # Explicitly close clips to prevent memory leaks
        locals_dict = locals()
        for var_name in ['video_clip', 'processed_clip', 'bg_music', 'tts_clip', 'narrative_audio', 'final_audio']:
            if var_name in locals_dict and locals_dict[var_name] is not None:
                try:
                    locals_dict[var_name].close()
                except:
                    pass
        # Force garbage collection
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Process and upload Reddit video.")
    parser.add_argument('subreddits', nargs='+', help="Subreddits to fetch videos from")
    parser.add_argument('--max_videos', type=int, default=5, help="Maximum number of videos to process per subreddit")
    parser.add_argument('--skip_upload', action='store_true', help="Skip uploading to YouTube")
    args = parser.parse_args()
    
    try:
        validate_environment()
        setup_directories()
        setup_database()
        setup_api_clients()
    except Exception as e:
        print(f"Setup failed: {e}")
        sys.exit(1)
        
    print(f"Processing subreddits: {', '.join(args.subreddits)}")
    
    # Process each subreddit
    videos_processed = 0
    for subreddit_name in args.subreddits:
        print(f"\nProcessing r/{subreddit_name}...")
        
        # Get submissions from subreddit
        submissions = get_reddit_submissions(subreddit_name, args.max_videos)
        if not submissions:
            print(f"  No suitable videos found in r/{subreddit_name}")
            continue
            
        print(f"  Found {len(submissions)} potential videos")
        
        # Process each submission
        for submission in submissions:
            if is_already_uploaded(submission.url):
                print(f"  Skipping {submission.id}: already processed")
                continue
                
            print(f"\n  Processing submission: {submission.id} - {submission.title}")
            
            # Sanitize title for filenames
            safe_title = sanitize_filename(submission.title)
            if not safe_title:
                safe_title = submission.id
                
            # Temporary files for processing
            temp_files_list = []
            
            # Signal handler for clean exit
            def signal_handler(sig, frame):
                print("\n\nKeyboard interrupt received. Cleaning up...")
                try:
                    # Clean temp files
                    if os.path.exists("temp_processing"):
                        shutil.rmtree("temp_processing")
                    # Close database connections
                    if 'processed_urls' in locals():
                        processed_urls.close()
                    # Exit with status 1 to indicate interrupted
                    sys.exit(1)
                except Exception as e:
                    print(f"Cleanup error: {str(e)}")
                    sys.exit(1)

            signal.signal(signal.SIGINT, signal_handler)

            try:
                # Prepare initial video
                print("  Preparing initial video...")
                initial_video, duration, width, height = _prepare_initial_video(submission, safe_title, temp_files_list)

                if not initial_video:
                    print("  Initial video preparation failed, skipping")
                    continue
                
                print(f"  Initial video: {duration:.2f}s, {width}x{height}")
                
                # Check for unsuitable content
                unsuitable, reason = is_unsuitable_video(submission, initial_video)
                if unsuitable:
                    print(f"  Skipping video due to unsuitable content: {reason}")
                    continue
                
                # AI Analysis
                print("  Analyzing video with AI...")
                analysis = analyze_video_with_gemini(initial_video, submission.title, subreddit_name)
                
                # Extract analysis data
                best_segment = analysis.get('best_segment')
                key_focus_points = analysis.get('key_focus_points', [])
                narrative_overlays = analysis.get('narrative_text_overlays', [])
                narrative_script = analysis.get('narrative_script', [])
                original_audio_is_key = analysis.get('original_audio_is_key', False)
                
                # Determine start/end times
                start_time = 0.0
                end_time = duration
                
                if best_segment and isinstance(best_segment, dict):
                    start_time = float(best_segment.get('start_time', 0))
                    end_time = float(best_segment.get('end_time', duration))
                    # Ensure end_time doesn't exceed available duration
                    end_time = min(end_time, duration)
                    # Clip duration based on segment, max TARGET_VIDEO_DURATION_SECONDS
                    clip_duration = min(end_time - start_time, TARGET_VIDEO_DURATION_SECONDS)
                    end_time = start_time + clip_duration
                else:
                    # If no segment, take the first TARGET_VIDEO_DURATION_SECONDS
                    end_time = min(duration, TARGET_VIDEO_DURATION_SECONDS)
                
                # Filter focus points to be within the selected time range
                relevant_focus_points = [
                    fp for fp in key_focus_points
                    if start_time <= fp.get("time", -1) <= end_time
                ]
                if not relevant_focus_points: # Ensure at least one point
                    relevant_focus_points = [{"time": start_time, "point": {"x": 0.5, "y": 0.5}}]
                
                # Apply dynamic 9:16 crop
                cropped_video_path = TEMP_DIR / f"{submission.id}_{safe_title}_cropped_9x16.mp4"
                temp_files_list.append(cropped_video_path)
                print(f"  Applying dynamic 9:16 crop (from {start_time:.2f}s to {end_time:.2f}s)...")
                try:
                    create_short_clip(
                        video_path=str(initial_video),
                        output_path=str(cropped_video_path),
                        start_time=start_time,
                        end_time=end_time,
                        focus_points=relevant_focus_points
                    )
                    if not cropped_video_path.is_file() or cropped_video_path.stat().st_size < 1024:
                        raise ValueError("Cropped video file is invalid or empty.")
                    processing_path = cropped_video_path
                    print("  Dynamic crop successful.")
                except Exception as crop_error:
                    print(f"  ERROR during dynamic cropping: {crop_error}")
                    print("  Skipping video due to cropping failure.")
                    cleanup_temp_files(cropped_video_path)
                    continue
                
                # Adjust overlay timestamps relative to the cropped clip's start time
                adjusted_overlays = []
                for overlay in narrative_overlays:
                    ts = overlay.get('timestamp', 0) - start_time
                    if 0 <= ts < (end_time - start_time):
                        overlay['timestamp'] = max(0, ts)
                        adjusted_overlays.append(overlay)
                
                adjusted_script = []
                for item in narrative_script:
                    ts = item.get('timestamp', 0) - start_time
                    if 0 <= ts < (end_time - start_time):
                        item['timestamp'] = max(0, ts)
                        adjusted_script.append(item)
                
                # Use the cropped video path as our processing path
                processing_path = cropped_video_path
                
                # Process video with effects (using the cropped video)
                print("  Applying overlays and final effects...")
                final_path = TEMP_DIR / f"{submission.id}_{safe_title}_final.mp4"
                temp_files_list.append(final_path)

                success = process_video_with_gpu_optimization(
                    processing_path,  # This is the 9x16 cropped video
                    adjusted_overlays,  # Use adjusted timestamps
                    adjusted_script,    # Use adjusted timestamps
                    original_audio_is_key,  # Pass the audio importance flag
                    final_path,
                    temp_files_list
                )
                
                if not success or not final_path.is_file():
                    print("  Final video not created successfully, skipping")
                    continue
                
                # Skip upload if requested
                if args.skip_upload:
                    print("  Skipping upload as requested")
                    print(f"  Final video saved at: {final_path}")
                    continue
                
                # Upload to YouTube
                try:
                    print("  Uploading to YouTube...")
                    
                    # Prepare upload metadata
                    title = analysis.get('suggested_title', f"Reddit: {submission.title}")
                    description = f"{analysis.get('summary_for_description', 'Check out this Reddit video!')}\n\nOriginal post: https://reddit.com{submission.permalink}"
                    tags = analysis.get('hashtags', [])
                    
                    # Add subreddit as tag
                    if f"r/{subreddit_name}" not in tags:
                        tags.append(f"r/{subreddit_name}")
                    
                    # Add general tags if needed
                    if len(tags) < 3:
                        tags.extend(['reddit', 'viral', 'shorts'])
                    
                    # Filter out any forbidden words
                    title = ' '.join(word for word in title.split() if not contains_forbidden_words(word))
                    
                    # Upload
                    youtube_url = upload_to_youtube(final_path, title, description)
                    
                    if youtube_url:
                        print(f"  Uploaded to YouTube: {youtube_url}")
                        add_upload_record(submission.url, youtube_url)
                        videos_processed += 1
                    else:
                        print("  Upload failed")
                except UploadLimitExceededError:
                    print("  YouTube upload limit reached, stopping")
                    break
                except Exception as e:
                    print(f"  Error during upload: {e}")
            finally:
                # Clean up temporary files
                for file_path in temp_files_list:
                    cleanup_temp_files(file_path)
    
    print(f"\nProcessing complete. {videos_processed} videos processed.")
    close_database()


if __name__ == "__main__":
    main()
