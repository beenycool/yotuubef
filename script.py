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
MUSIC_FOLDER = BASE_DIR / "royalty_free_music"  # Updated folder name
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

# --- Enhanced Video Quality Settings ---
# Higher bitrate for better quality videos
VIDEO_BITRATE_HIGH = '10M'  # 10 Mbps for high quality
VIDEO_BITRATE_MEDIUM = '8M'  # 8 Mbps for medium quality
VIDEO_BITRATE_MOBILE = '6M'  # 6 Mbps for mobile-optimized
AUDIO_BITRATE = '192k'       # Higher audio bitrate
ENABLE_TWO_PASS_ENCODING = True  # Better quality with two-pass encoding
ENABLE_VIDEO_LOOP = True     # Enable video looping for shorts
MAX_LOOP_COUNT = 3           # Maximum number of times to loop short clips
MIN_LOOP_DURATION = 7        # Minimum duration in seconds for loopable clips

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
MIX_ORIGINAL_AUDIO = True # Narrative style usually replaces original audio
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

# Curated subreddit whitelist - high-quality, family-friendly content
CURATED_SUBREDDITS = [
    "oddlysatisfying", "nextfuckinglevel", "BeAmazed", "woahdude", 
    "NatureIsFuckingLit", "EarthPorn", "aww", "MadeMeSmile", "Eyebleach",
    "interestingasfuck", "Damnthatsinteresting", "AnimalsBeingBros", 
    "HumansBeingBros", "wholesomememes", "ContagiousLaughter",
    "foodporn", "CookingVideos", "ArtisanVideos", "educationalgifs",
    "DIY", "gardening", "science", "space", "NatureIsCool",
    "AnimalsBeingDerps", "rarepuppers", "LifeProTips"
]

# Music Configuration
MUSIC_CATEGORIES = {
    "upbeat": ["energetic", "positive", "happy", "uplifting"],
    "emotional": ["sad", "heartwarming", "touching", "sentimental"],
    "suspenseful": ["tense", "dramatic", "action", "exciting"],
    "relaxing": ["calm", "peaceful", "ambient", "soothing"],
    "funny": ["quirky", "comedic", "playful", "lighthearted"],
    "informative": ["neutral", "documentary", "educational", "background"]
}

# Create subdirectories for organized music
for category in MUSIC_CATEGORIES:
    (MUSIC_FOLDER / category).mkdir(parents=True, exist_ok=True)

# Function to download royalty-free music (if needed)
def download_royalty_free_music():
    """
    Downloads a selection of royalty-free music from trusted sources.
    Creates categorized folders for different moods/styles.
    """
    # Only run if music folders are empty
    total_music_files = sum(len(list((MUSIC_FOLDER / category).glob("*.mp3"))) for category in MUSIC_CATEGORIES)
    
    if total_music_files > 0:
        print(f"Found {total_music_files} existing music files. Skipping download.")
        return
        
    print("Downloading royalty-free music...")
    
    # URLs for trusted royalty-free music sources
    # These are public domain or CC0 licensed music collections
    sources = {
        "upbeat": [
            "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Chad_Crouch/Arps/Chad_Crouch_-_Shipping_Lanes.mp3",
            "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Tours/Enthusiast/Tours_-_01_-_Enthusiast.mp3"
        ],
        "emotional": [
            "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Chad_Crouch/Drifter/Chad_Crouch_-_Drifter.mp3",
            "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/WFMU/Broke_For_Free/Directionless_EP/Broke_For_Free_-_01_-_Night_Owl.mp3"
        ],
        "suspenseful": [
            "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Kai_Engel/Satin/Kai_Engel_-_03_-_Contention.mp3",
            "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/Zach_Heffelfinger/Pneumatic_Tubes/Emotional_Technology/Pneumatic_Tubes_-_05_-_The_Secret_Engine.mp3"
        ],
        "relaxing": [
            "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/Music_for_Video/Blue_Dot_Sessions/Bitters/Blue_Dot_Sessions_-_Sage_the_Hunter.mp3",
            "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Chad_Crouch/Field_Report_Vol_I_Oaks_Bottom/Chad_Crouch_-_Egret.mp3"
        ],
        "funny": [
            "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/WFMU/Deerhoof/Deerhoof_session_WFMU_on_the_Web_2008_01_22/Deerhoof_-_01_-_Fresh_Born_Live_on_WFMU.mp3",
            "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/WFMU/Syna_So_Pro/Live_on_WFMUs_Busy_Doing_Nothing_with_Charlie_Oct_14_2015/Syna_So_Pro_-_01_-_Unidentifiable.mp3"
        ],
        "informative": [
            "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Jahzzar/Tumbling_Dishes_Like_Old-Mans_Memories/Jahzzar_-_05_-_Siesta.mp3",
            "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Kai_Engel/Sustains/Kai_Engel_-_08_-_Augmentations.mp3"
        ]
    }
    
    for category, urls in sources.items():
        category_dir = MUSIC_FOLDER / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        for i, url in enumerate(urls):
            try:
                # Extract filename from URL or create a numbered filename
                filename = url.split("/")[-1]
                if not filename.endswith(".mp3"):
                    filename = f"{category}_{i+1}.mp3"
                    
                output_path = category_dir / filename
                
                # Download file
                print(f"  Downloading {filename} to {category} category...")
                
                # Use requests if available, otherwise use urllib
                try:
                    import requests
                    response = requests.get(url, timeout=30)
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                except ImportError:
                    import urllib.request
                    urllib.request.urlretrieve(url, output_path)
                    
                print(f"  Downloaded {filename}")
                
            except Exception as e:
                print(f"  Error downloading {url}: {e}")
    
    # Count downloaded files
    total_music_files = sum(len(list((MUSIC_FOLDER / category).glob("*.mp3"))) for category in MUSIC_CATEGORIES)
    print(f"Downloaded {total_music_files} royalty-free music files.")

def select_music_for_content(analysis: Dict) -> Optional[pathlib.Path]:
    """
    Selects appropriate royalty-free music based on content analysis
    
    Args:
        analysis: Video analysis dictionary
        
    Returns:
        Path to selected music file or None
    """
    # Extract mood and music genres from analysis
    mood = analysis.get('mood', 'neutral').lower()
    music_genres = analysis.get('music_genres', [])
    
    # Determine best category based on mood and genres
    target_category = "informative"  # Default category
    
    # Map mood to music category
    mood_category_map = {
        "funny": "funny",
        "heartwarming": "emotional",
        "informative": "informative",
        "suspenseful": "suspenseful",
        "action": "suspenseful",
        "calm": "relaxing",
        "exciting": "upbeat",
        "sad": "emotional",
        "shocking": "suspenseful",
        "weird": "funny",
        "cringe": "funny"
    }
    
    if mood in mood_category_map:
        target_category = mood_category_map[mood]
    
    # Check if any specified genres match our categories or tags
    for genre in music_genres:
        genre_lower = genre.lower()
        
        # Direct category match
        if genre_lower in MUSIC_CATEGORIES:
            target_category = genre_lower
            break
            
        # Tag match
        for category, tags in MUSIC_CATEGORIES.items():
            if any(tag.lower() == genre_lower for tag in tags):
                target_category = category
                break
    
    # Get list of music files in the target category
    category_dir = MUSIC_FOLDER / target_category
    if not category_dir.exists():
        # Fall back to main music folder
        music_files = list(MUSIC_FOLDER.glob("*.mp3"))
    else:
        music_files = list(category_dir.glob("*.mp3"))
    
    # If no music files found in target category, try to find any music file
    if not music_files:
        # Search all categories
        for category in MUSIC_CATEGORIES:
            cat_dir = MUSIC_FOLDER / category
            if cat_dir.exists():
                music_files = list(cat_dir.glob("*.mp3"))
                if music_files:
                    break
    
    # If still no music files, return None
    if not music_files:
        print(f"  No music files found for category: {target_category}")
        return None
    
    # Randomly select a music file from the appropriate category
    selected_music = random.choice(music_files)
    print(f"  Selected music: {selected_music.name} from {target_category} category")
    
    return selected_music

def generate_custom_thumbnail(video_path: pathlib.Path, analysis: Dict, output_path: pathlib.Path) -> Optional[pathlib.Path]:
    """
    Generates a custom thumbnail for the video
    
    Args:
        video_path: Path to the video file
        analysis: Video analysis dictionary
        output_path: Path to save the thumbnail
        
    Returns:
        Path to the thumbnail file or None if generation failed
    """
    if not video_path.is_file():
        return None
        
    try:
        # Get thumbnail moment from analysis, or use middle of video
        thumbnail_moment = analysis.get('thumbnail_moment', None)
        if thumbnail_moment is None:
            # Get video duration
            duration, _, _ = get_video_details(video_path)
            if duration <= 0:
                thumbnail_moment = 0
            else:
                # Use a moment about 1/3 into the video for thumbnail
                thumbnail_moment = duration / 3
        
        # Extract frame at specified moment
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
            
        # Set position to thumbnail moment
        cap.set(cv2.CAP_PROP_POS_MSEC, thumbnail_moment * 1000)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Enhance thumbnail with image processing
        # 1. Increase contrast and saturation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = hsv[:,:,1] * 1.3  # Increase saturation
        hsv[:,:,2] = hsv[:,:,2] * 1.1  # Increase brightness slightly
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 2. Apply sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Save the base thumbnail
        cv2.imwrite(str(output_path), sharpened)
        
        # Optional: Add text overlay using title from analysis
        title = analysis.get('suggested_title', '')
        if title:
            # Create a version with text overlay using PIL
            from PIL import Image, ImageDraw, ImageFont
            
            img = Image.open(output_path)
            draw = ImageDraw.Draw(img)
            
            # Try to load a nice font, fall back to default
            try:
                # Try a few common fonts that might be available
                font_paths = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                    "C:\\Windows\\Fonts\\Arial.ttf",  # Windows
                    "/Library/Fonts/Arial.ttf"  # Mac
                ]
                
                font = None
                for font_path in font_paths:
                    try:
                        font = ImageFont.truetype(font_path, size=img.height // 12)
                        break
                    except:
                        continue
                        
                if font is None:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # Prepare title text (limit length)
            if len(title) > 40:
                title = title[:37] + "..."
            
            # Add semi-transparent background for text
            text_position = (img.width // 20, img.height * 3 // 4)
            text_size = draw.textbbox(text_position, title, font=font)
            
            # Create semi-transparent black rectangle behind text
            overlay = Image.new('RGBA', img.size, (0,0,0,0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(
                [text_size[0] - 10, text_size[1] - 10, text_size[2] + 10, text_size[3] + 10],
                fill=(0, 0, 0, 160)
            )
            
            # Composite the overlay onto the main image
            img = Image.alpha_composite(img.convert('RGBA'), overlay)
            
            # Draw white text with black outline for visibility
            draw = ImageDraw.Draw(img)
            
            # Draw text outline by offsetting black text slightly
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                draw.text((text_position[0]+dx, text_position[1]+dy), title, font=font, fill=(0,0,0))
            
            # Draw main text in white
            draw.text(text_position, title, font=font, fill=(255,255,255))
            
            # Save the thumbnail with text overlay
            img.convert('RGB').save(output_path)
        
        return output_path
        
    except Exception as e:
        print(f"  Error generating thumbnail: {e}")
        return None

def gemini_content_safety_check(submission, video_path: Optional[pathlib.Path] = None) -> Tuple[bool, str]:
    """
    Uses Gemini to perform advanced content safety check
    
    Args:
        submission: Reddit submission
        video_path: Path to downloaded video if available
        
    Returns:
        Tuple of (is_problematic, reason)
    """
    global gemini_model
    
    if not gemini_model:
        print("  Warning: Gemini not available, using basic safety check")
        return is_unsuitable_video(submission, video_path)
    
    # First run basic check - quick rejection
    unsuitable, reason = is_unsuitable_video(submission, video_path)
    if unsuitable:
        return True, reason
        
    # If video is not available, we can't perform detailed check
    if video_path is None or not video_path.is_file():
        print("  Warning: Video file not available for Gemini safety check")
        return False, ""
    
    try:
        # Extract a few frames for image-based safety check
        frame_paths = []
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print("  Warning: Could not open video for safety check")
                return False, ""
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Sample frames at strategic points (start, middle, end)
            sample_points = [0.1, 0.5, 0.9]  # Sample at 10%, 50%, and 90% of duration
            
            for i, point in enumerate(sample_points):
                frame_time = point * duration
                frame_num = int(frame_time * fps)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    frame_path = TEMP_DIR / f"safety_check_frame_{i}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(frame_path)
            
            cap.release()
            
        except Exception as e:
            print(f"  Warning: Error extracting frames for safety check: {e}")
            
        # Prepare prompt for Gemini
        prompt = f"""Analyze this content from Reddit for safety and suitability.
Title: "{submission.title}"
Subreddit: r/{submission.subreddit.display_name}

Your task is to determine if this content is suitable for a general audience.
Specifically check for:
1. Violence, gore, or disturbing imagery
2. Sexual content or nudity
3. Hate speech, discrimination, or offensive language
4. Dangerous activities that could lead to harm or injury
5. Drug use or other illegal activities
6. Excessive profanity
7. Content that objectifies or demeans individuals
8. Harmful misinformation

Respond with the following format:
{{
  "is_suitable": true/false,
  "reason_if_unsuitable": "explanation", 
  "concerns": ["list", "of", "concerns"],
  "confidence": 0-100
}}
"""

        # Prepare content for Gemini model
        content_parts = [prompt]
        
        # Add frames if available
        for frame_path in frame_paths:
            if frame_path.is_file():
                with open(frame_path, 'rb') as f:
                    image_data = f.read()
                    content_parts.append({"mime_type": "image/jpeg", "data": base64.b64encode(image_data).decode('utf-8')})
        
        # Use strict safety settings
        safety_settings = [
            {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} 
            for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", 
                      "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]
        
        # Get Gemini response
        response = gemini_model.generate_content(
            content_parts, 
            generation_config=genai.types.GenerationConfig(temperature=0.2),
            safety_settings=safety_settings
        )
        
        # Clean up frame files
        for frame_path in frame_paths:
            cleanup_temp_files(frame_path)
        
        if not response.candidates or not response.text:
            print("  Warning: No valid response from Gemini for safety check")
            return False, ""
            
        # Parse JSON response
        result_text = response.text
        
        # Extract JSON from response (handle various formats)
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            json_start = result_text.find('{')
            json_end = result_text.rfind('}')
            if json_start == -1 or json_end == -1 or json_end < json_start:
                print("  Warning: Could not find JSON in Gemini response")
                return False, ""
            json_text = result_text[json_start:json_end + 1]
            
        try:
            result = json.loads(json_text)
            is_suitable = result.get("is_suitable", True)
            reason = result.get("reason_if_unsuitable", "")
            confidence = result.get("confidence", 0)
            
            # Only reject if we're reasonably confident
            if not is_suitable and confidence >= 70:
                return True, reason
                
            return False, ""
        except json.JSONDecodeError:
            print("  Warning: Could not parse Gemini JSON response for safety check")
            return False, ""
            
    except Exception as e:
        print(f"  Error in Gemini content safety check: {e}")
        return False, ""

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
        
    # Check if subreddit is in our curated whitelist
    if submission.subreddit.display_name.lower() not in [s.lower() for s in CURATED_SUBREDDITS]:
        return True, f"Subreddit not in curated whitelist: r/{submission.subreddit.display_name}"
        
    # Check submission flair for unsuitable indicators
    if submission.link_flair_text and any(word in submission.link_flair_text.lower() for word in FORBIDDEN_WORDS):
        return True, f"Unsuitable flair: {submission.link_flair_text}"
    
    # Check comments for unsuitable content indicators (improved with timeout)
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
                
                # Check comments for forbidden words
                if contains_forbidden_words(top_comments_text):
                    return True, "Comments contain forbidden words"
                
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

# --- Modified get_reddit_submissions function ---
def get_reddit_submissions(subreddit_name: str, limit: int) -> List[praw.models.Submission]:
    if not reddit: return []
    
    # Verify subreddit is in our curated list
    if subreddit_name.lower() not in [s.lower() for s in CURATED_SUBREDDITS]:
        print(f"  Warning: r/{subreddit_name} is not in the curated subreddit whitelist")
        return []
        
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
                     thumbnail_path: Optional[str] = None,
                     category_id: str = YOUTUBE_UPLOAD_CATEGORY_ID,
                     privacy_status: str = YOUTUBE_UPLOAD_PRIVACY_STATUS) -> Optional[str]:
    """
    Uploads a video to YouTube using the YouTube Data API.
    
    Args:
        video_path: Path to the video file to upload
        title: Video title
        description: Video description
        thumbnail_path: Path to thumbnail image (optional)
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
        
        # Upload the video
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
        
        video_id = response['id']
        print(f"Successfully uploaded video: {video_id}")
        
        # If thumbnail provided, upload it
        if thumbnail_path and os.path.exists(thumbnail_path):
            try:
                youtube_service.thumbnails().set(
                    videoId=video_id,
                    media_body=MediaFileUpload(thumbnail_path)
                ).execute()
                print("Custom thumbnail uploaded successfully")
            except Exception as thumb_error:
                print(f"Error uploading thumbnail: {thumb_error}")
        
        return f"https://youtu.be/{video_id}"
        
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
1. Create an authentic and engaging opening in the first 3 seconds
2. Use concise text overlays (1-4 words) strategically at key moments 
3. Create a narrative that builds genuine interest without being over-the-top
4. Identify natural moments of surprise, satisfaction, or interest
5. Suggest subtle visual effects at appropriate moments
6. Recommend music that matches the authentic mood of the content
7. Focus on honest storytelling that respects the viewer's intelligence
8. Ensure the video has a satisfying conclusion when possible

Return JSON with these fields:
{{
    "suggested_title": "string (authentic, <70 chars, descriptive but intriguing)",
    "summary_for_description": "string (2-3 sentences, factual with subtle interest)",
    "mood": "string (from: funny, heartwarming, informative, suspenseful, action, calm, exciting, sad, shocking, weird, cringe)",
    "has_clear_narrative": "boolean",
    "original_audio_is_key": "boolean",
    "hook_text": "string (authentic opening hook)",
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
            "text": "string (concise, clear, not overhyped)",
            "time": float,
            "duration": float
        }}
    ],
    "narrative_script": [
        {{
            "text": "string (natural, conversational)",
            "time": float,
            "duration": float
        }}
    ],
    "visual_cues": [
        {{
            "time": float,
            "suggestion": "string (e.g., 'subtle zoom', 'gentle transition', 'focus effect')"
        }}
    ],
    "music_genres": ["string (appropriate mood-matching options)"],
    "key_audio_moments": [
        {{
            "time": float,
            "action": "string (e.g., 'highlight reaction', 'emphasize moment')"
        }}
    ],
    "retention_tactics": [
        {{
            "tactic": "string (e.g., 'natural pause', 'question prompt', 'interest point')",
            "time": float
        }}
    ],
    "hashtags": ["string (relevant, not excessive)"],
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

                # Process hook text - add as first overlay if present
                if "hook_text" in parsed_data and parsed_data["hook_text"].strip():
                    hook_text = parsed_data["hook_text"].strip().upper()
                    # Insert hook at the beginning of text_overlays
                    parsed_data["text_overlays"].insert(0, {
                        "text": hook_text,
                        "timestamp": 0.0,  # Start immediately
                        "duration": 3.0    # Show for 3 seconds
                    })

                # Generate narrative_script from text_overlays if not provided
                if "narrative_script" not in parsed_data or not isinstance(parsed_data["narrative_script"], list):
                    parsed_data["narrative_script"] = []
                    for overlay in parsed_data["text_overlays"]:
                        parsed_data["narrative_script"].append({
                            "timestamp": overlay["timestamp"],
                            "duration": overlay["duration"],
                            "narrative_text": overlay["text"].lower().capitalize()  # Convert to natural speech
                        })

                # Process retention tactics
                if "retention_tactics" in parsed_data and isinstance(parsed_data["retention_tactics"], list):
                    # Sort tactics by time
                    retention_tactics = sorted(parsed_data["retention_tactics"], key=lambda x: x.get("time", 0))
                    
                    # Add tactics that don't already have corresponding text overlays
                    existing_times = [overlay.get("timestamp", 0) for overlay in parsed_data["text_overlays"]]
                    
                    for tactic in retention_tactics:
                        tactic_time = tactic.get("time", 0)
                        tactic_text = tactic.get("tactic", "").upper()
                        
                        # Skip if there's already an overlay at this time (within 1 second)
                        if not any(abs(tactic_time - t) < 1.0 for t in existing_times) and tactic_text:
                            # Add as text overlay
                            parsed_data["text_overlays"].append({
                                "text": tactic_text,
                                "timestamp": tactic_time,
                                "duration": 2.5  # Default duration
                            })
                            
                            # Also add to narrative script
                            parsed_data["narrative_script"].append({
                                "timestamp": tactic_time,
                                "duration": 2.5,
                                "narrative_text": tactic_text.lower().capitalize()
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
def generate_tts_elevenlabs(text: str, output_path: pathlib.Path, voice_id: str = DEFAULT_VOICE_ID, voice_settings: Optional[Dict] = None) -> bool:
    """Generate TTS using ElevenLabs API with enhanced voice settings"""
    global elevenlabs_client
    if not elevenlabs_client or not text or not text.strip(): return False
    output_path.parent.mkdir(parents=True, exist_ok=True); cleanup_temp_files(output_path)
    try:
        # Default voice settings for more natural speech
        default_settings = {
            "stability": 0.71,       # Slightly higher stability for consistent quality
            "similarity_boost": 0.75, # Good balance between clarity and voice character
            "style": 0.15,           # Small amount of style variation for interest
            "use_speaker_boost": True # Clearer audio
        }
        
        # Use provided settings or defaults
        settings = voice_settings if voice_settings else default_settings
        
        # Apply voice settings
        voice_settings_obj = {
            "stability": settings.get("stability", 0.71),
            "similarity_boost": settings.get("similarity_boost", 0.75),
            "style": settings.get("style", 0.15),
            "use_speaker_boost": settings.get("use_speaker_boost", True)
        }
        
        # Generate audio with the configured settings
        audio = elevenlabs_client.generate(
            text=text,
            voice=voice_id,
            model="eleven_turbo_v2",  # Use the latest model for best quality
            voice_settings=voice_settings_obj
        )
        
        save(audio, str(output_path))
        return output_path.is_file() and output_path.stat().st_size > 500
    except ApiError as e: print(f"  ElevenLabs API Error: {e}"); return False
    except Exception as e: print(f"  Error generating TTS with ElevenLabs: {e}"); return False

def generate_tts(text: str, output_path: pathlib.Path, voice_settings: Optional[Dict] = None, voice_id: str = DEFAULT_VOICE_ID) -> bool:
    """
    Enhanced TTS generation with support for voice characteristics and emotion
    
    Args:
        text: Text to convert to speech
        output_path: Path to save the audio file
        voice_settings: Dictionary of voice settings to adjust characteristics
        voice_id: ElevenLabs voice ID
        
    Returns:
        True if successful, False otherwise
    """
    # Process text to add SSML for better speech patterns
    processed_text = enhance_text_for_tts(text)
    
    # Try using Dia model first (with enhanced text)
    if dia_tts(processed_text, output_path):
        print("  Successfully generated TTS with Dia-1.6B model")
        return True
    # Fall back to ElevenLabs with voice settings
    elif elevenlabs_client and generate_tts_elevenlabs(processed_text, output_path, voice_id, voice_settings):
        print("  Generated TTS with ElevenLabs using custom voice settings")
        return True
    return False

def enhance_text_for_tts(text: str) -> str:
    """
    Enhances text with natural pauses and emphasis for more engaging TTS
    
    Args:
        text: Original text
        
    Returns:
        Enhanced text with natural speech patterns
    """
    if not text: 
        return ""
        
    # Add natural pauses after sentences and before important conjunctions
    text = re.sub(r'\.(\s+)', '.\n\n', text)  # Add paragraph breaks after periods
    text = re.sub(r'([!?])(\s+)', r'\1\n', text)  # Add line breaks after ! and ?
    
    # Add emphasis to important words (simple heuristic)
    emphasis_words = ["amazing", "incredible", "stunning", "important", "critical", 
                     "fascinating", "remarkable", "extraordinary", "crucial"]
    
    for word in emphasis_words:
        text = re.sub(r'\b' + word + r'\b', f" {word} ", text, flags=re.IGNORECASE)
    
    # Clean up any double spaces
    text = re.sub(r'\s{2,}', ' ', text)
    
    return text.strip()

def select_voice_for_content(analysis: Dict) -> str:
    """
    Selects an appropriate voice ID based on the content analysis
    
    Args:
        analysis: The video analysis dictionary
        
    Returns:
        The voice ID to use for TTS
    """
    # Get voice characteristics and content mood
    voice_characteristics = analysis.get('tts_voice_characteristics', '').lower()
    mood = analysis.get('mood', 'neutral').lower()
    
    # Default voice
    default_voice = DEFAULT_VOICE_ID  # "21m00Tcm4TlvDq8ikWAM" - Default ElevenLabs voice
    
    # Define voice mappings based on characteristics and mood
    # These IDs should be replaced with actual ElevenLabs voice IDs from your account
    voice_mapping = {
        # Characteristics-based mapping
        'warm': "pNInz6obpgDQGcFmaJgB",         # Adam - warm, natural male voice
        'conversational': "pNInz6obpgDQGcFmaJgB", # Adam 
        'friendly': "pNInz6obpgDQGcFmaJgB",      # Adam
        'authoritative': "ErXwobaYiN019PkySvjV", # Antoni - deeper male voice
        'serious': "ErXwobaYiN019PkySvjV",       # Antoni
        'professional': "ErXwobaYiN019PkySvjV",  # Antoni
        'energetic': "yoZ06aMxZJJ28mfd3POQ",     # Josh - energetic young male
        'upbeat': "yoZ06aMxZJJ28mfd3POQ",        # Josh
        'enthusiastic': "yoZ06aMxZJJ28mfd3POQ",  # Josh
        'calm': "EXAVITQu4vr4xnSDxMaL",          # Elli - calm female voice
        'soft': "EXAVITQu4vr4xnSDxMaL",          # Elli
        'soothing': "EXAVITQu4vr4xnSDxMaL",      # Elli
        
        # Mood-based mapping
        'funny': "yoZ06aMxZJJ28mfd3POQ",        # Josh - good for humor
        'heartwarming': "EXAVITQu4vr4xnSDxMaL",  # Elli - warm female voice
        'informative': "ErXwobaYiN019PkySvjV",   # Antoni - good for educational
        'suspenseful': "VR6AewLTigWG4xSOukaG",   # Sam - dramatic male voice
        'action': "VR6AewLTigWG4xSOukaG",        # Sam
        'exciting': "yoZ06aMxZJJ28mfd3POQ",      # Josh
        'sad': "EXAVITQu4vr4xnSDxMaL",           # Elli
        'shocking': "VR6AewLTigWG4xSOukaG",      # Sam
        'weird': "t0jbNlBVZ17f02VDIeMI",         # Matilda - quirky female voice
        'cringe': "t0jbNlBVZ17f02VDIeMI"         # Matilda
    }
    
    # Try to match voice characteristics first, then mood
    for term in voice_characteristics.split():
        if term in voice_mapping:
            return voice_mapping[term]
    
    if mood in voice_mapping:
        return voice_mapping[mood]
    
    # Default to conversational voice if no matches
    return default_voice

def get_voice_settings_for_narrative(analysis: Dict) -> Dict:
    """
    Creates voice settings based on the narrative analysis
    
    Args:
        analysis: The video analysis dictionary
        
    Returns:
        Dictionary with voice settings
    """
    # Get narrative style and mood
    narrative_style = analysis.get('narrative_style', '').lower()
    mood = analysis.get('mood', 'neutral').lower()
    
    # Default settings for a balanced voice
    settings = {
        "stability": 0.71,       # Balanced stability
        "similarity_boost": 0.75, # Default similarity
        "style": 0.15,           # Slight style variation
        "use_speaker_boost": True
    }
    
    # Adjust settings based on narrative style
    if 'dramatic' in narrative_style or 'suspenseful' in narrative_style:
        settings["stability"] = 0.65       # Less stability for more variation
        settings["similarity_boost"] = 0.70 # Slightly lower similarity
        settings["style"] = 0.30           # More style variation for drama
    
    elif 'educational' in narrative_style or 'informative' in narrative_style:
        settings["stability"] = 0.75       # Higher stability for clarity
        settings["similarity_boost"] = 0.80 # Higher similarity for consistency
        settings["style"] = 0.10           # Less style variation for clear information
    
    elif 'humorous' in narrative_style or 'funny' in mood:
        settings["stability"] = 0.60       # Lower stability for expressiveness
        settings["similarity_boost"] = 0.65 # Lower similarity for character
        settings["style"] = 0.40           # More style for humor
    
    elif 'calm' in narrative_style or 'heartwarming' in mood:
        settings["stability"] = 0.80       # Higher stability for smooth delivery
        settings["similarity_boost"] = 0.75 # Balanced similarity
        settings["style"] = 0.20           # Moderate style for warmth
    
    # Return the customized settings
    return settings

def generate_narrative_audio(narrative_script: List[Dict], analysis: Dict, temp_dir: pathlib.Path, temp_files_list: List[pathlib.Path]) -> List[Dict]:
    """
    Generates audio files for the narrative script with appropriate voice and settings
    
    Args:
        narrative_script: List of narrative segments
        analysis: Video analysis dictionary
        temp_dir: Directory for temporary files
        temp_files_list: List to track temporary files
        
    Returns:
        List of dictionaries with audio clip information
    """
    if not narrative_script:
        return []
        
    # Select appropriate voice based on content
    voice_id = select_voice_for_content(analysis)
    
    # Get voice settings based on narrative style
    voice_settings = get_voice_settings_for_narrative(analysis)
    
    print(f"  Generating narrative audio using voice settings: {voice_settings}")
    
    audio_clips_info = []
    
    for i, segment in enumerate(narrative_script):
        segment_text = segment.get('narrative_text', '')
        if not segment_text:
            continue
            
        # Adjust voice settings based on segment tone if specified
        segment_settings = voice_settings.copy()
        if 'tone' in segment:
            tone = segment['tone'].lower()
            if 'excited' in tone or 'energetic' in tone:
                segment_settings["stability"] = 0.60
                segment_settings["style"] = 0.40
            elif 'serious' in tone or 'dramatic' in tone:
                segment_settings["stability"] = 0.75
                segment_settings["style"] = 0.25
            elif 'thoughtful' in tone or 'reflective' in tone:
                segment_settings["stability"] = 0.85
                segment_settings["style"] = 0.15
        
        # Generate TTS for this segment
        segment_file = temp_dir / f"narrative_{i}.mp3"
        temp_files_list.append(segment_file)
        
        if generate_tts(segment_text, segment_file, segment_settings, voice_id):
            # Get the actual duration of the generated audio
            audio_duration = get_audio_duration(segment_file)
            
            audio_clips_info.append({
                "file_path": segment_file,
                "timestamp": segment.get('timestamp', 0),
                "duration": audio_duration or segment.get('duration', 5.0),
                "original_text": segment_text
            })
    
    return audio_clips_info

def get_audio_duration(audio_path: pathlib.Path) -> Optional[float]:
    """Get the duration of an audio file in seconds"""
    if not audio_path.is_file():
        return None
        
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
               '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"  Error getting audio duration: {e}")
        return None

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
                                      temp_files_list: List[pathlib.Path],
                                      visual_cues: Optional[List[Dict]] = None,
                                      analysis: Optional[Dict] = None) -> bool:
    """
    Process video with effects using GPU acceleration where possible.
    Now handles dynamic narrative overlays, TTS audio mixing, and targeted visual effects based on AI analysis.
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
                print("  Attempting GPU acceleration with enhanced quality settings...")
                # Use simpler command with more compatible settings and higher quality
                ffmpeg_enhance_cmd = [
                    'ffmpeg', '-y',
                    '-i', str(processing_path),
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p5',  # Quality-focused preset (p1-p7, lower is better quality)
                    '-rc', 'vbr',     # Variable bitrate mode
                    '-qmin', '17',    # Minimum quantizer
                    '-qmax', '21',    # Maximum quantizer (lower = better quality)
                    '-b:v', VIDEO_BITRATE_HIGH,  # Higher bitrate for better quality
                    '-c:a', AUDIO_CODEC,         # Use specified audio codec
                    '-b:a', AUDIO_BITRATE,       # Higher audio bitrate
                    str(enhanced_path)
                ]
                subprocess.run(ffmpeg_enhance_cmd, check=True, capture_output=True, timeout=300)
                base_video_path = enhanced_path
                print("  GPU acceleration successful with enhanced quality!")
            else:
                # For CPU use high-quality encoding with two-pass if enabled
                print("  Using CPU processing with enhanced quality settings...")
                if ENABLE_TWO_PASS_ENCODING:
                    # First pass
                    ffmpeg_pass1_cmd = [
                        'ffmpeg', '-y',
                        '-i', str(processing_path),
                        '-c:v', VIDEO_CODEC_CPU,
                        '-preset', 'slow',  # Slower preset for better quality
                        '-crf', '18',       # Lower CRF for higher quality (15-18 is high quality)
                        '-b:v', VIDEO_BITRATE_HIGH,
                        '-pass', '1',
                        '-an',  # No audio in first pass
                        '-f', 'mp4',
                        '-threads', str(N_JOBS_PARALLEL),
                        '-y', '/dev/null' if os.name != 'nt' else 'NUL'
                    ]
                    subprocess.run(ffmpeg_pass1_cmd, check=True, capture_output=True, timeout=300)
                    
                    # Second pass
                    ffmpeg_pass2_cmd = [
                        'ffmpeg', '-y',
                        '-i', str(processing_path),
                        '-c:v', VIDEO_CODEC_CPU,
                        '-preset', 'slow',
                        '-crf', '18',
                        '-b:v', VIDEO_BITRATE_HIGH,
                        '-pass', '2',
                        '-c:a', AUDIO_CODEC,
                        '-b:a', AUDIO_BITRATE,
                        '-threads', str(N_JOBS_PARALLEL),
                        str(enhanced_path)
                    ]
                    subprocess.run(ffmpeg_pass2_cmd, check=True, capture_output=True, timeout=300)
                else:
                    # Single pass with high quality
                    ffmpeg_enhance_cmd = [
                        'ffmpeg', '-y',
                        '-i', str(processing_path),
                        '-c:v', VIDEO_CODEC_CPU,
                        '-preset', 'slow',
                        '-crf', '18',
                        '-b:v', VIDEO_BITRATE_HIGH,
                        '-c:a', AUDIO_CODEC,
                        '-b:a', AUDIO_BITRATE,
                        '-threads', str(N_JOBS_PARALLEL),
                        str(enhanced_path)
                    ]
                    subprocess.run(ffmpeg_enhance_cmd, check=True, capture_output=True, timeout=300)
                
                base_video_path = enhanced_path
                print("  CPU processing successful with enhanced quality!")
        except Exception as e:
            print(f"  Enhanced encoding failed: {e}")
            print("  Falling back to standard processing...")
            # On failure, fall back to standard settings
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
                print(f"  Standard processing also failed: {e2}")
                # Last resort: just use the original file
                base_video_path = processing_path
            
        # Now use MoviePy for the more complex effects like text overlays and TTS
        with VideoFileClip(str(base_video_path), fps_source="fps") as video_clip:
            # Force the fps if not properly detected
            if not hasattr(video_clip, 'fps') or video_clip.fps is None or video_clip.fps <= 0:
                video_clip.fps = video_fps
                
            print(f"  Loaded video: {video_clip.duration:.2f}s, {video_clip.size}, fps={video_clip.fps}")
            
            # Apply base effects to the entire clip
            processed_clip = video_clip
            if SHAKE_EFFECT_ENABLED:
                processed_clip = apply_shake(processed_clip)
            
            # Apply dynamic visual effects at specific timestamps based on AI analysis
            if visual_cues and isinstance(visual_cues, list) and len(visual_cues) > 0:
                print(f"  Applying {len(visual_cues)} targeted visual effects...")
                
                # Create a list of subclips with their own effects
                subclips = []
                last_end_time = 0
                
                # Sort visual cues by time
                visual_cues.sort(key=lambda x: x.get('time', 0))
                
                for cue in visual_cues:
                    cue_time = cue.get('time', 0)
                    suggestion = cue.get('suggestion', '').lower()
                    
                    # Default effect duration (can be adjusted based on effect type)
                    effect_duration = 1.2  # Reduced from 1.5 seconds
                    
                    # Skip if cue time is beyond video duration
                    if cue_time >= video_clip.duration:
                        continue
                        
                    # Add normal segment before the effect if needed
                    if cue_time > last_end_time:
                        normal_subclip = video_clip.subclip(last_end_time, cue_time)
                        subclips.append(normal_subclip)
                    
                    # Determine effect end time
                    effect_end_time = min(cue_time + effect_duration, video_clip.duration)
                    
                    # Get the segment for the effect
                    effect_subclip = video_clip.subclip(cue_time, effect_end_time)
                    
                    # Apply appropriate effect based on suggestion
                    if 'zoom' in suggestion:
                        # Use a more subtle zoom factor
                        zoom_factor = 1.15  # Reduced from 1.3
                        effect_subclip = effect_subclip.fx(vfx.zoom, zoom_factor)
                    elif 'shake' in suggestion or 'explosion' in suggestion:
                        # More gentle shake for emphasis
                        effect_subclip = vfx.shake(effect_subclip, displacement_range=5, shake_duration=0.15)
                    elif 'slow' in suggestion or 'motion' in suggestion:
                        # Less dramatic slow motion
                        slowdown_factor = 0.85  # 85% speed instead of 70%
                        effect_subclip = effect_subclip.fx(vfx.speedx, slowdown_factor)
                    elif 'bright' in suggestion or 'flash' in suggestion:
                        # More subtle brightness adjustment
                        effect_subclip = effect_subclip.fx(vfx.colorx, 1.2)  # Reduced from 1.5
                    elif 'contrast' in suggestion:
                        # More subtle contrast
                        effect_subclip = effect_subclip.fx(vfx.lum_contrast, contrast=1.2)  # Reduced from 1.5
                    elif 'focus' in suggestion or 'highlight' in suggestion:
                        # Subtle focus effect (slight zoom + mild brightness)
                        effect_subclip = effect_subclip.fx(vfx.zoom, 1.1)
                        effect_subclip = effect_subclip.fx(vfx.colorx, 1.1)
                    
                    # Add the effect subclip
                    subclips.append(effect_subclip)
                    last_end_time = effect_end_time
                
                # Add remaining video after the last effect
                if last_end_time < video_clip.duration:
                    final_subclip = video_clip.subclip(last_end_time, video_clip.duration)
                    subclips.append(final_subclip)
                
                # Concatenate all subclips
                if subclips:
                    processed_clip = concatenate_videoclips(subclips)
            
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
            
            # --- Video Looping for Shorts ---
            # Check if video should be looped based on duration
            final_clip = processed_clip
            if ENABLE_VIDEO_LOOP and processed_clip.duration < MIN_LOOP_DURATION:
                # Create smooth loop transition
                print(f"  Video duration ({processed_clip.duration:.2f}s) is short, creating seamless loop...")
                
                # Calculate how many times to loop to reach target duration
                # While still respecting MAX_LOOP_COUNT
                num_loops = min(
                    MAX_LOOP_COUNT,
                    max(1, math.ceil(TARGET_VIDEO_DURATION_SECONDS / processed_clip.duration))
                )
                
                if num_loops > 1:
                    # Use cross-fade transition between loops for smoother effect
                    # Calculate transition duration (10% of clip duration or 0.5s max)
                    transition_duration = min(0.5, processed_clip.duration * 0.1)
                    
                    # For cross-fade, we need to extend the clips slightly to overlap
                    extended_clips = []
                    for i in range(num_loops):
                        if i == 0:
                            # First clip is just the original
                            extended_clips.append(processed_clip)
                        else:
                            # Subsequent clips need to be positioned with crossfade
                            next_clip = processed_clip.copy()
                            next_clip = next_clip.set_start(
                                processed_clip.duration * i - transition_duration
                            )
                            extended_clips.append(next_clip)
                    
                    # Create composite with crossfaded clips
                    final_clip = CompositeVideoClip(
                        extended_clips,
                        size=processed_clip.size
                    ).subclip(0, processed_clip.duration * num_loops)
                    
                    print(f"  Created {num_loops}x loop with seamless transitions")
            
            # Write out the processed video with better quality
            print("  Writing final video...")
            final_render_bitrate = VIDEO_BITRATE_HIGH  # Use high bitrate for final output
            
            # Determine best CPU threads based on system
            system_threads = max(1, os.cpu_count() or 4)
            render_threads = min(system_threads - 1, 16)  # Use at most 16 threads, leave 1 for OS
            
            final_clip.write_videofile(
                str(final_path),
                codec='libx264',  # Always use CPU encoding for final output
                preset='slow',    # Use slow preset for better quality
                audio_codec=AUDIO_CODEC,
                threads=render_threads,
                fps=video_fps,    # Use the detected fps or our default
                bitrate=final_render_bitrate,   # Specify bitrate
                ffmpeg_params=[
                    "-profile:v", "high", 
                    "-level", "4.2",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-b:a", AUDIO_BITRATE
                ],
                logger='bar'      # Show progress
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
        for var_name in ['video_clip', 'processed_clip', 'final_clip', 'bg_music', 'tts_clip', 'narrative_audio', 'final_audio']:
            if var_name in locals_dict and locals_dict[var_name] is not None:
                try:
                    locals_dict[var_name].close()
                except:
                    pass
        # Force garbage collection
        gc.collect()

# Function for checking if a video could be looped well
def analyze_for_looping(video_path: pathlib.Path) -> bool:
    """
    Analyzes a video to determine if it's a good candidate for seamless looping.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Boolean indicating if the video is a good candidate for looping
    """
    try:
        # Check if video exists and has minimum duration
        duration, _, _ = get_video_details(video_path)
        if duration < 3.0:
            return False  # Too short to be worth looping
            
        # If video is already long enough, no need to loop
        if duration >= MIN_LOOP_DURATION:
            return False
            
        with VideoFileClip(str(video_path)) as clip:
            # For short videos, check start and end frames for similarity
            # Extract first and last frames
            first_frame = clip.get_frame(0)
            last_frame = clip.get_frame(clip.duration - 0.1)  # 0.1s before end
            
            # Calculate frame similarity using mean squared error
            mse = np.mean((first_frame - last_frame) ** 2)
            similarity = 1 / (1 + mse)
            
            # Check audio if present
            audio_similarity = 0
            if clip.audio:
                try:
                    # Get audio samples for first and last 0.2 seconds
                    first_audio = clip.audio.subclip(0, 0.2).to_soundarray()
                    last_audio = clip.audio.subclip(clip.duration - 0.2, clip.duration).to_soundarray()
                    
                    # Calculate audio similarity
                    audio_mse = np.mean((first_audio - last_audio) ** 2)
                    audio_similarity = 1 / (1 + audio_mse)
                except:
                    audio_similarity = 0
            
            # Combined score with more weight on visual similarity
            combined_score = (similarity * 0.7) + (audio_similarity * 0.3)
            
            # Return True if similarity is high enough for good looping
            return combined_score > 0.7
    except Exception as e:
        print(f"  Error analyzing video for looping: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Process and upload Reddit video.")
    parser.add_argument('subreddits', nargs='+', help="Subreddits to fetch videos from")
    parser.add_argument('--max_videos', type=int, default=5, help="Maximum number of videos to process per subreddit")
    parser.add_argument('--skip_upload', action='store_true', help="Skip uploading to YouTube")
    parser.add_argument('--list_curated', action='store_true', help="List curated subreddits and exit")
    parser.add_argument('--download_music', action='store_true', help="Download royalty-free music and exit")
    args = parser.parse_args()
    
    # Add option to list curated subreddits
    if args.list_curated:
        print("Curated Subreddit Whitelist:")
        for subreddit in sorted(CURATED_SUBREDDITS):
            print(f"  - r/{subreddit}")
        sys.exit(0)
        
    # Add option to download music only
    if args.download_music:
        print("Downloading royalty-free music...")
        download_royalty_free_music()
        sys.exit(0)
    
    try:
        validate_environment()
        setup_directories()
        setup_database()
        setup_api_clients()
        
        # Check for royalty-free music and download if missing
        if not any(MUSIC_FOLDER.glob("**/*.mp3")):
            print("No royalty-free music found. Downloading now...")
            download_royalty_free_music()
            
    except Exception as e:
        print(f"Setup failed: {e}")
        sys.exit(1)
        
    print(f"Processing subreddits: {', '.join(args.subreddits)}")
    
    # Validate subreddits against whitelist
    invalid_subreddits = [s for s in args.subreddits if s.lower() not in [cs.lower() for cs in CURATED_SUBREDDITS]]
    if invalid_subreddits:
        print(f"\nWARNING: The following subreddits are not in the curated whitelist and will be skipped:")
        for sub in invalid_subreddits:
            print(f"  - r/{sub}")
        
        # Filter to only whitelist subreddits
        valid_subreddits = [s for s in args.subreddits if s.lower() in [cs.lower() for cs in CURATED_SUBREDDITS]]
        if not valid_subreddits:
            print("No valid subreddits to process. Use --list_curated to see available options.")
            sys.exit(1)
        
        args.subreddits = valid_subreddits
        print(f"\nProceeding with valid subreddits: {', '.join(args.subreddits)}")
        
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
                    for file_path in temp_files_list:
                        cleanup_temp_files(file_path)
                    # Close database connections
                    if db_conn:
                        close_database()
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
                
                # Enhanced content safety check using Gemini
                print("  Performing Gemini content safety check...")
                unsuitable, reason = gemini_content_safety_check(submission, initial_video)
                if unsuitable:
                    print(f"  Skipping video due to content safety: {reason}")
                    continue
                
                # AI Analysis with enhanced narrative capabilities
                print("  Analyzing video with Gemini for transformative narrative...")
                analysis = analyze_video_with_gemini(initial_video, submission.title, subreddit_name)
                
                # Extract analysis data
                best_segment = analysis.get('best_segment')
                key_focus_points = analysis.get('key_focus_points', [])
                text_overlays = analysis.get('text_overlays', [])
                narrative_script = analysis.get('narrative_script', [])
                visual_cues = analysis.get('visual_cues', [])
                original_audio_is_key = analysis.get('original_audio_is_key', False)
                
                # Determine start/end times
                start_time = 0.0
                end_time = duration
                
                if best_segment and isinstance(best_segment, dict):
                    segment_start = best_segment.get('start', 0)
                    segment_end = best_segment.get('end', duration)
                    
                    # Use float values if available
                    start_time = float(segment_start) if isinstance(segment_start, (int, float)) else 0
                    end_time = float(segment_end) if isinstance(segment_end, (int, float)) else duration
                    
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
                    if start_time <= float(fp.get("time", -1)) <= end_time
                ]
                if not relevant_focus_points: # Ensure at least one point
                    relevant_focus_points = [{"time": start_time, "point": {"x": 0.5, "y": 0.5}}]
                
                # Also adjust visual cues to be relative to the trimmed segment
                adjusted_visual_cues = []
                for cue in visual_cues:
                    cue_time = float(cue.get('time', 0)) - start_time
                    if 0 <= cue_time < (end_time - start_time):
                        adjusted_cue = cue.copy()
                        adjusted_cue['time'] = cue_time
                        adjusted_visual_cues.append(adjusted_cue)

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
                for overlay in text_overlays:
                    # Handle different key names in the overlay dict
                    ts_key = 'timestamp' if 'timestamp' in overlay else 'time'
                    ts = float(overlay.get(ts_key, 0)) - start_time
                    
                    if 0 <= ts < (end_time - start_time):
                        overlay_copy = overlay.copy()
                        overlay_copy['timestamp'] = max(0, ts)
                        adjusted_overlays.append(overlay_copy)
                
                adjusted_script = []
                for item in narrative_script:
                    ts = float(item.get('timestamp', 0)) - start_time
                    if 0 <= ts < (end_time - start_time):
                        script_item = item.copy()
                        script_item['timestamp'] = max(0, ts)
                        adjusted_script.append(script_item)
                
                # Use the cropped video path as our processing path
                processing_path = cropped_video_path
                
                # Generate thumbnail
                thumbnail_path = TEMP_DIR / f"{submission.id}_{safe_title}_thumbnail.jpg"
                temp_files_list.append(thumbnail_path)
                print("  Generating custom thumbnail...")
                generate_custom_thumbnail(processing_path, analysis, thumbnail_path)
                
                # Process video with effects (using the cropped video)
                print("  Applying narrative, overlays and final effects...")
                final_path = TEMP_DIR / f"{submission.id}_{safe_title}_final.mp4"
                temp_files_list.append(final_path)

                success = process_video_with_gpu_optimization(
                    processing_path,         # The 9:16 cropped video
                    adjusted_overlays,       # Adjusted text overlays
                    adjusted_script,         # Adjusted narrative script
                    original_audio_is_key,   # Audio flag
                    final_path,              # Output path
                    temp_files_list,         # Temp files list
                    adjusted_visual_cues,    # Adjusted visual cues
                    analysis                 # Full analysis for advanced features
                )
                
                if not success or not final_path.is_file():
                    print("  Final video not created successfully, skipping")
                    continue
                
                # Skip upload if requested
                if args.skip_upload:
                    print("  Skipping upload as requested")
                    print(f"  Final video saved at: {final_path}")
                    
                    # If skipping upload, keep files instead of deleting
                    output_dir = BASE_DIR / "output_videos"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy final video to output directory
                    final_output_path = output_dir / f"{submission.id}_{safe_title}_final.mp4"
                    shutil.copy2(final_path, final_output_path)
                    
                    # Copy thumbnail if exists
                    if thumbnail_path.is_file():
                        thumbnail_output_path = output_dir / f"{submission.id}_{safe_title}_thumbnail.jpg"
                        shutil.copy2(thumbnail_path, thumbnail_output_path)
                        
                    print(f"  Files copied to output directory: {output_dir}")
                    continue
                
                # Upload to YouTube
                try:
                    print("  Uploading to YouTube...")
                    
                    # Prepare upload metadata with enhanced narrative approach
                    suggested_title = analysis.get('suggested_title', '')
                    narrative_style = analysis.get('narrative_style', '').capitalize()
                    
                    if not suggested_title or contains_forbidden_words(suggested_title):
                        # Create fallback title
                        title = f"My Take: Reddit Highlights"
                    else:
                        # Format title to highlight transformative nature
                        if narrative_style:
                            title = f"{narrative_style} Take: {suggested_title}"
                        else:
                            title = f"My Analysis: {suggested_title}"
                    
                    # Limit title length
                    title = title[:70]
                    
                    # Enhanced description with narrative summary
                    summary = analysis.get('summary_for_description', 'A unique perspective on this Reddit content.')
                    narrative_angle = analysis.get('narrative_unique_angle', '')
                    
                    # Create a description that emphasizes original commentary
                    description = f"{summary}\n\n"
                    
                    if narrative_angle:
                        description += f"My unique take: {narrative_angle}\n\n"
                        
                    description += "If you enjoyed this perspective, consider subscribing for more original commentary.\n\n"
                    
                    # Add hashtags strategically
                    tags = analysis.get('hashtags', [])
                    
                    # Add subreddit as tag
                    if f"r/{subreddit_name}" not in tags:
                        tags.append(f"r/{subreddit_name}")
                    
                    # Add tags for narrative style
                    if narrative_style and narrative_style.lower() not in [t.lower() for t in tags]:
                        tags.append(narrative_style.lower())
                    
                    # Add hashtags to description (limited number)
                    hashtag_str = " ".join(["#" + tag.strip("#") for tag in tags[:6]])
                    description += f"\n{hashtag_str}\n\n"
                    
                    # Add source attribution
                    description += f"Original content source: https://reddit.com{submission.permalink}\n"
                    description += "This video features substantial transformative commentary and analysis."
                    
                    # Filter out any forbidden words
                    title = ' '.join(word for word in title.split() if not contains_forbidden_words(word))
                    
                    # Upload with custom thumbnail if available
                    thumbnail_file = str(thumbnail_path) if thumbnail_path.is_file() else None
                    
                    # Upload
                    youtube_url = upload_to_youtube(final_path, title, description, thumbnail_file)
                    
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
