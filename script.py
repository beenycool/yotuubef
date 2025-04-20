import argparse
import base64
import json
import math
import os
import pathlib
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from typing import Optional, List, Dict, Tuple, Any, Union

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
from video_processor import _prepare_initial_video
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

change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})

imagemagick_path_str = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
imagemagick_env_var = "IMAGEMAGICK_BINARY"

if os.path.exists(imagemagick_path_str):
    os.environ[imagemagick_env_var] = imagemagick_path_str
else:
    existing_env = os.environ.get(imagemagick_env_var)
    if not existing_env:
        print(f"Warning: ImageMagick path not found: {imagemagick_path_str} and {imagemagick_env_var} environment variable not set. MoviePy text features might fail.")

REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.environ.get('REDDIT_USER_AGENT', f'python:VideoBot:v1.4 (by /u/YOUR_USERNAME_HERE)')
GOOGLE_CLIENT_SECRETS_FILE = os.environ.get('GOOGLE_CLIENT_SECRETS_FILE')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')

GEMINI_MODEL_ID = 'gemini-2.0-flash' # KEEP 2.0 FLASH
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

BASE_DIR = pathlib.Path(__file__).parent.resolve()
TEMP_DIR = BASE_DIR / "temp_processing"
MUSIC_FOLDER = BASE_DIR / "music"
WATERMARK_PATH = BASE_DIR / "watermark.png"
DB_FILE = BASE_DIR / 'uploaded_videos.db'

TARGET_VIDEO_DURATION_SECONDS = 60
TARGET_ASPECT_RATIO = 9 / 16
TARGET_RESOLUTION = (1080, 1920)
TARGET_FPS = 30
AUDIO_CODEC = 'aac'
VIDEO_CODEC_CPU = 'libx264'
VIDEO_CODEC_GPU = 'h264_nvenc'
FFMPEG_CPU_PRESET = 'medium'
FFMPEG_GPU_PRESET = 'p5'
FFMPEG_CRF_CPU = '23'
FFMPEG_CQ_GPU = '23'
LOUDNESS_TARGET_LUFS = -16

OVERLAY_FONT = 'Impact'
OVERLAY_FONT_SIZE_RATIO = 1 / 12
OVERLAY_TEXT_COLOR = 'white'
OVERLAY_STROKE_COLOR = 'black'
OVERLAY_STROKE_WIDTH = 3
OVERLAY_POSITION = ('center', 0.8)
OVERLAY_BG_COLOR = 'transparent'

API_DELAY_SECONDS = 6
MAX_REDDIT_POSTS_TO_FETCH = 10
ADD_VISUAL_EMPHASIS = True
APPLY_STABILIZATION = False
MIX_ORIGINAL_AUDIO = False
ORIGINAL_AUDIO_MIX_VOLUME = 0.7
BACKGROUND_MUSIC_VOLUME = 0.15
BACKGROUND_MUSIC_DUCKED_VOLUME = 0.04
AUDIO_DUCKING_FADE_TIME = 0.3
SHAKE_EFFECT_ENABLED = True
SUBTLE_ZOOM_ENABLED = True
COLOR_GRADE_ENABLED = True
PARALLEL_FRAME_PROCESSING = True
N_JOBS_PARALLEL = max(1, os.cpu_count() // 2)

FORBIDDEN_WORDS = [
    "fuck", "fucking", "fucked", "fuckin", "shit", "shitty", "shitting",
    "damn", "damned", "bitch", "bitching", "cunt", "asshole", "piss", "pissed"
]

YOUTUBE_SCOPES = ['https://www.googleapis.com/auth/youtube.upload', 'https://www.googleapis.com/auth/youtube.force-ssl']
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
YOUTUBE_UPLOAD_CATEGORY_ID = '22'
YOUTUBE_UPLOAD_PRIVACY_STATUS = 'private'

db_conn: Optional[sqlite3.Connection] = None
db_cursor: Optional[sqlite3.Cursor] = None
elevenlabs_client: Optional[ElevenLabs] = None
gemini_model: Optional[genai.GenerativeModel] = None
reddit: Optional[praw.Reddit] = None
youtube_service: Optional[Any] = None

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
    'hashtags': ['#reddit', '#shorts', '#video'],
    'original_duration': 0.0
}

def check_ffmpeg_install(command: str) -> bool:
    try:
        return shutil.which(command) is not None
    except Exception:
        return False

def has_nvidia_gpu() -> bool:
    if sys.platform == "win32":
        command = ['nvidia-smi']
    else:
        command = ['nvidia-smi']
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
        return result.returncode == 0 and "NVIDIA-SMI" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False

def cleanup_temp_files(*file_paths: Optional[pathlib.Path]):
    for file_path in file_paths:
        if file_path and file_path.is_file():
            try:
                file_path.unlink()
            except OSError as e:
                print(f"  - Warning: Error removing {file_path.name}: {e}")

def contains_forbidden_words(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(word in text_lower for word in FORBIDDEN_WORDS)

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.replace(" ", "_")
    return name[:100]

def get_video_details(video_path: pathlib.Path) -> Tuple[float, int, int]:
    if not video_path.is_file(): return 0.0, 0, 0
    if not check_ffmpeg_install("ffprobe"): return 0.0, 0, 0

    try:
        command = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration,r_frame_rate:format=duration',
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
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        print(f"Error getting video details for {video_path.name}: {e}")
        return 0.0, 0, 0

def has_audio_track(media_path: pathlib.Path) -> bool:
    if not media_path.is_file(): return False
    if not check_ffmpeg_install("ffprobe"): return False

    try:
        command = [
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', str(media_path)
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
        return result.returncode == 0 and 'audio' in result.stdout.strip()
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"Error checking audio track in {media_path.name}: {e}")
        return False

def validate_environment():
    essential_vars = {
        'REDDIT_CLIENT_ID': REDDIT_CLIENT_ID,
        'REDDIT_CLIENT_SECRET': REDDIT_CLIENT_SECRET,
        'GOOGLE_CLIENT_SECRETS_FILE': GOOGLE_CLIENT_SECRETS_FILE,
        'GEMINI_API_KEY': GEMINI_API_KEY,
        'ELEVENLABS_API_KEY': ELEVENLABS_API_KEY,
    }
    missing_vars = [name for name, value in essential_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing essential environment variables: {', '.join(missing_vars)}")
    if not GOOGLE_CLIENT_SECRETS_FILE or not os.path.exists(GOOGLE_CLIENT_SECRETS_FILE):
        raise FileNotFoundError(f"Google Client Secrets file not found: {GOOGLE_CLIENT_SECRETS_FILE}")
    if not check_ffmpeg_install("ffmpeg"):
        raise RuntimeError("FFmpeg is required but not found.")
    if not check_ffmpeg_install("ffprobe"):
        print("Warning: ffprobe not found.")
    if not check_ffmpeg_install("ffmpeg-normalize"):
        print("Warning: ffmpeg-normalize not found. Audio normalization will be skipped.")

def setup_directories():
    for dir_path in [TEMP_DIR, MUSIC_FOLDER]:
        dir_path.mkdir(parents=True, exist_ok=True)

def setup_database():
    global db_conn, db_cursor
    try:
        db_conn = sqlite3.connect(DB_FILE)
        db_cursor = db_conn.cursor()
        db_cursor.execute('''
            CREATE TABLE IF NOT EXISTS uploads (
                reddit_url TEXT PRIMARY KEY,
                youtube_url TEXT,
                upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        db_conn.commit()
    except sqlite3.Error as e:
        raise RuntimeError(f"Database setup failed: {e}") from e

def close_database():
    global db_conn
    if db_conn:
        try: db_conn.close()
        except sqlite3.Error as e: print(f"Warning: Error closing database: {e}")
        finally: db_conn = None

def setup_api_clients():
    global reddit, youtube_service, gemini_model, elevenlabs_client
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT, check_for_async=False
        )
        reddit.user.me()
    except (prawcore.exceptions.OAuthException, prawcore.exceptions.ResponseException, Exception) as e:
        raise ConnectionError(f"Failed to initialize Reddit client: {e}") from e

    try:
        flow = InstalledAppFlow.from_client_secrets_file(GOOGLE_CLIENT_SECRETS_FILE, YOUTUBE_SCOPES)
        credentials = flow.run_local_server(port=0)
        youtube_service = build_google_api(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, credentials=credentials)
    except Exception as e:
        raise ConnectionError(f"Failed to initialize YouTube client: {e}") from e

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_ID)
    except Exception as e:
        raise ConnectionError(f"Failed to initialize Gemini client: {e}") from e

    try:
        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    except Exception as e:
        raise ConnectionError(f"Failed to initialize ElevenLabs client: {e}") from e

def is_already_uploaded(reddit_url: str) -> bool:
    if not db_cursor: return True
    try:
        db_cursor.execute("SELECT 1 FROM uploads WHERE reddit_url = ?", (reddit_url,))
        return db_cursor.fetchone() is not None
    except sqlite3.Error as e:
        print(f"Error checking DB for URL {reddit_url}: {e}")
        return True

def add_upload_record(reddit_url: str, youtube_url: str):
    if not db_conn or not db_cursor: return
    try:
        db_cursor.execute("INSERT INTO uploads (reddit_url, youtube_url) VALUES (?, ?)", (reddit_url, youtube_url))
        db_conn.commit()
    except sqlite3.IntegrityError:
        print(f"Warning: Record for {reddit_url} already exists.")
    except sqlite3.Error as e:
        print(f"Error adding DB record for {reddit_url}: {e}")

def get_reddit_submissions(subreddit_name: str, limit: int) -> List[praw.models.Submission]:
    if not reddit: return []
    submissions = []
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for submission in subreddit.hot(limit=limit * 2):
            if len(submissions) >= limit: break
            if (submission.is_video or any(submission.url.endswith(ext) for ext in ['.mp4', '.mov', '.gifv']) or
                any(host in submission.url for host in ['v.redd.it', 'gfycat.com', 'streamable.com'])) and \
               not submission.over_18 and not submission.stickied:
                submissions.append(submission)
            time.sleep(0.1)
        return submissions
    except (prawcore.exceptions.NotFound, prawcore.exceptions.PrawcoreException, Exception) as e:
        print(f"Error fetching from r/{subreddit_name}: {e}")
        return []

def download_media(url: str, output_path: pathlib.Path) -> Optional[pathlib.Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_temp_files(output_path)

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': str(output_path),
        'quiet': True, 'no_warnings': True, 'ignoreerrors': True, 'noprogress': True,
        'retries': 3, 'socket_timeout': 30, 'nocheckcertificate': True,
        'merge_output_format': 'mp4',
        # 'max_filesize': '100M',
        # 'download_ranges': yt_dlp.utils.download_range_func(None, [(0, TARGET_VIDEO_DURATION_SECONDS + 10)])
    }

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
                        try: file.rename(output_path)
                        except OSError: pass
                    return output_path
            cleanup_temp_files(output_path)
            return None
    except (yt_dlp.utils.DownloadError, Exception) as e:
        print(f"Error downloading {url}: {e}")
        cleanup_temp_files(output_path)
        return None

def analyze_video_with_gemini(video_path: pathlib.Path, title: str, subreddit_name: str) -> Dict[str, Any]:
    global gemini_model
    if not gemini_model or not video_path.is_file(): return FALLBACK_ANALYSIS.copy()

    analysis_result = FALLBACK_ANALYSIS.copy()

    try:
        # Get video duration and details first
        duration, width, height = get_video_details(video_path)
        if not duration or duration <= 0: return analysis_result

        analysis_result['original_duration'] = duration # Store original duration

        # Prepare the prompt for Gemini
        prompt = f"""Analyze this video from Reddit (r/{subreddit_name}). Original title: "{title}". Video duration: {duration:.2f}s. 
Output ONLY a JSON object with keys: 
- "summary_for_description" (string, 2-3 sentences)
- "suggested_title" (string, < 70 chars)
- "mood" (string, ONE from [funny, heartwarming, informative, suspenseful, action, calm, exciting, sad, shocking, weird])
- "best_segment" (object {{"start_time": float, "end_time": float}} or null)
- "key_visual_moments" (list of objects {{"timestamp": float, "description": string, "focus_point": {{"x": float, "y": float}}}} or empty list)
- "speech_segments" (list of objects {{"start_time": float, "end_time": float, "text": string}} or empty list)
- "hashtags" (list of 3-5 strings)

Focus point 0.5,0.5 is center. Ensure times are within duration. Return ONLY JSON."""

        # Read the video file as binary data
        with open(video_path, 'rb') as f:
            video_data = f.read()

        # Create the content parts for Gemini
        content_parts = [
            prompt,
            {"mime_type": "video/mp4", "data": base64.b64encode(video_data).decode('utf-8')}
        ]

        # Configure safety settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        # Generate response from Gemini
        try:
            response = gemini_model.generate_content(
                content_parts,
                generation_config=genai.types.GenerationConfig(temperature=0.6),
                safety_settings=safety_settings
            )

            result_text = ""
            if response.candidates:
                try:
                    result_text = response.text
                except ValueError as e: # Handles cases where .text access fails (e.g., blocked)
                    print(f"  Warning: Gemini response blocked or invalid. Finish Reason: {response.candidates[0].finish_reason if response.candidates else 'N/A'}")
                    if hasattr(response, 'prompt_feedback'): print(f"  Safety Feedback: {response.prompt_feedback}")
                    return analysis_result

            json_start = result_text.find('{')
            json_end = result_text.rfind('}')
            if json_start == -1 or json_end == -1 or json_end < json_start:
                 if response.prompt_feedback and response.prompt_feedback.block_reason:
                      print(f"  Content blocked: {response.prompt_feedback.block_reason}")
                 return analysis_result

            json_text = result_text[json_start : json_end + 1]
            try:
                parsed_data = json.loads(json_text)
                parsed_data['original_duration'] = duration # Ensure duration is in the final dict
                parsed_data['fallback'] = False # Indicate success
                return parsed_data
            except json.JSONDecodeError:
                return analysis_result

        except Exception as gemini_err:
            print(f"  Error: Gemini API call failed: {gemini_err}")
            return analysis_result

    except Exception as e:
        print(f"  Error during Gemini analysis pipeline: {e}")
        return analysis_result

def generate_tts(text: str, output_path: pathlib.Path, voice_id: str = DEFAULT_VOICE_ID) -> bool:
    """Generate TTS using ElevenLabs API (fallback only)"""
    if not text or not text.strip():
        print("  Error: Empty text for TTS")
        return False
    
    print("  ElevenLabs TTS disabled - using local TTS only")
    return False

def hugging_face_tts(text: str, output_path: pathlib.Path) -> bool:
    """Generate TTS using local SpeechT5 model with enhanced reliability"""
    if not text or not text.strip():
        print("  Error: Empty text for TTS")
        return False
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path = output_path.with_suffix('.wav')
    cleanup_temp_files(output_path, wav_path)
    
    try:
        import torch
        import soundfile as sf
        from transformers import SpeechT5ForTextToSpeech, AutoProcessor
        import numpy as np
        
        print("  Generating TTS with SpeechT5 (local)")
        
        # Use smaller model for better reliability
        model_name = "microsoft/speecht5_tts"
        processor = AutoProcessor.from_pretrained(model_name)
        model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
        
        # Configure logging
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        # Device selection with fallback
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Using {device.upper()} for TTS generation")
        
        try:
            model = model.to(device)
        except RuntimeError as e:
            print(f"  Warning: Couldn't move model to {device}, using CPU")
            device = "cpu"
            model = model.to(device)
        
        # Simple deterministic speaker embedding
        speaker_embeddings = torch.zeros((1, 512)).to(device)
        speaker_embeddings[0, :256] = 0.5  # Neutral voice
        
        # Process text in chunks if too long
        max_length = 500
        if len(text) > max_length:
            print(f"  Splitting long text into chunks of {max_length} characters")
            text_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            audio_chunks = []
            
            for chunk in text_chunks:
                inputs = processor(text=chunk, return_tensors="pt").to(device)
                with torch.no_grad():
                    output = model.generate_speech(
                        inputs["input_ids"],
                        speaker_embeddings=speaker_embeddings,
                        vocoder=None
                    )
                    audio_chunks.append(output.cpu().numpy())
                    del output
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            audio_data = np.concatenate(audio_chunks)
        else:
            inputs = processor(text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate_speech(
                    inputs["input_ids"],
                    speaker_embeddings=speaker_embeddings,
                    vocoder=None
                )
                audio_data = output.cpu().numpy()
                del output
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Clean up
        del inputs, model, speaker_embeddings
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Ensure proper audio format
        if len(audio_data.shape) > 1:
            if audio_data.shape[0] > 2:  # Multi-channel -> mono
                audio_data = np.mean(audio_data, axis=0)
            elif audio_data.shape[0] == 2:  # Stereo -> mono
                audio_data = np.mean(audio_data, axis=0)
        
        # Save as WAV first
        sf.write(str(wav_path), audio_data, 16000)
        
        # Convert to desired format
        if output_path.suffix.lower() != '.wav':
            try:
                if check_ffmpeg_install("ffmpeg"):
                    cmd = [
                        "ffmpeg", "-y", "-i", str(wav_path),
                        "-ac", "1", "-ar", "44100",
                        "-hide_banner", "-loglevel", "error",
                        str(output_path)
                    ]
                    subprocess.run(cmd, check=True, timeout=30)
                    if not output_path.is_file() or output_path.stat().st_size < 500:
                        raise ValueError("Invalid output file")
                else:
                    shutil.copy(str(wav_path), str(output_path))
            except Exception as e:
                print(f"  Warning: Format conversion failed: {e}")
                shutil.copy(str(wav_path), str(output_path))
        
        cleanup_temp_files(wav_path)
        return output_path.is_file() and output_path.stat().st_size > 500
            
    except ImportError as e:
        print(f"  Error: Required TTS libraries not installed: {e}")
        print("  Please run: pip install torch transformers soundfile")
        return False
    except Exception as e:
        print(f"  Error generating TTS with Kokoro: {e}")
        traceback.print_exc()
        return False

def upload_to_youtube(video_path: pathlib.Path, title: str, description: str, tags: List[str]) -> Optional[str]:
    global youtube_service
    if not youtube_service or not video_path.is_file(): return None

    try:
        body = {
            'snippet': {'title': title, 'description': description, 'tags': tags, 'categoryId': YOUTUBE_UPLOAD_CATEGORY_ID},
            'status': {'privacyStatus': YOUTUBE_UPLOAD_PRIVACY_STATUS, 'selfDeclaredMadeForKids': False}
        }
        media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
        request = youtube_service.videos().insert(part=",".join(body.keys()), body=body, media_body=media)

        response = None
        while response is None:
            try:
                status, response = request.next_chunk()
                # if status: print(f"  Upload progress: {int(status.progress() * 100)}%") # Optional progress
            except google_api_errors.HttpError as e:
                if e.resp.status in [403, 500, 503]:
                    if 'quotaExceeded' in str(e) or 'uploadLimitExceeded' in str(e):
                        raise UploadLimitExceededError("YouTube daily upload limit likely reached.") from e
                    time.sleep(API_DELAY_SECONDS)
                else: raise ConnectionError(f"YouTube API HTTP error: {e}") from e
            except Exception as e: raise ConnectionError(f"YouTube upload failed: {e}") from e

        return f"https://www.youtube.com/watch?v={response['id']}"
    except UploadLimitExceededError as quota_err:
        raise quota_err
    except Exception as e:
        print(f"  Error uploading video: {e}")
        return None

def normalize_loudness(input_path: pathlib.Path, output_path: pathlib.Path, target_lufs: float = LOUDNESS_TARGET_LUFS) -> bool:
    if not input_path.is_file(): return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_temp_files(output_path)

    if not check_ffmpeg_install("ffmpeg-normalize"):
        try:
            shutil.copy(str(input_path), str(output_path))
            return output_path.is_file() and output_path.stat().st_size > 100
        except Exception: return False

    command = [
        "ffmpeg-normalize", str(input_path), "-o", str(output_path),
        "-c:a", AUDIO_CODEC, "-b:a", "192k", "-ar", "44100", "-f",
        "-nt", "ebu", "-t", str(target_lufs), "--keep-loudness-range-target"
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        return output_path.is_file() and output_path.stat().st_size > 100
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception) as e:
        print(f"ffmpeg-normalize failed: {e}. Falling back to copy.")
        cleanup_temp_files(output_path)
        try:
            shutil.copy(str(input_path), str(output_path))
            return output_path.is_file() and output_path.stat().st_size > 100
        except Exception: return False

def make_video_vertical(input_path: pathlib.Path, output_path: pathlib.Path) -> Optional[pathlib.Path]:
    clip, final_clip = None, None
    try:
        clip = VideoFileClip(str(input_path))
        original_w, original_h = clip.size
        if original_w <= 0 or original_h <= 0: return None

        target_w, target_h = TARGET_RESOLUTION
        target_aspect = target_w / target_h
        original_aspect = original_w / original_h

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleanup_temp_files(output_path)

        if abs(original_aspect - target_aspect) < 0.01:
            final_clip = clip.resize(TARGET_RESOLUTION)
        else:
            if original_aspect > target_aspect:
                new_w = int(target_aspect * original_h)
                final_clip = crop(clip, width=new_w, height=original_h, x_center=original_w / 2, y_center=original_h / 2)
                final_clip = final_clip.resize(height=target_h)
            else:
                new_h = int(original_w / target_aspect)
                final_clip = crop(clip, width=original_w, height=new_h, x_center=original_w / 2, y_center=original_h / 2)
                final_clip = final_clip.resize(width=target_w)

        if final_clip.size != TARGET_RESOLUTION:
             final_clip = final_clip.resize(TARGET_RESOLUTION)

        # Use GPU encoding if available
        use_gpu = has_nvidia_gpu()
        codec = VIDEO_CODEC_GPU if use_gpu else VIDEO_CODEC_CPU
        preset = FFMPEG_GPU_PRESET if use_gpu else FFMPEG_CPU_PRESET
        quality_param = ['-cq', FFMPEG_CQ_GPU] if use_gpu else ['-crf', FFMPEG_CRF_CPU]
        ffmpeg_extra_params = quality_param + ['-pix_fmt', 'yuv420p']
        
        temp_audio_path = TEMP_DIR / f"{output_path.stem}_temp_audio.m4a"
        final_clip.write_videofile(
            str(output_path), codec=codec, audio_codec=AUDIO_CODEC,
            temp_audiofile=str(temp_audio_path), remove_temp=True,
            preset=preset, fps=TARGET_FPS, threads=N_JOBS_PARALLEL, logger=None,
            ffmpeg_params=ffmpeg_extra_params
        )
        cleanup_temp_files(temp_audio_path)

        return output_path if output_path.is_file() and output_path.stat().st_size > 10240 else None
    except Exception as e:
        print(f"  Error making video vertical: {e}")
        cleanup_temp_files(output_path)
        return None
    finally:
        if clip:
            try: clip.close()
            except Exception: pass
        if final_clip and final_clip != clip:
            try: final_clip.close()
            except Exception: pass

def trim_video(input_path: pathlib.Path, start_time: float, end_time: float, output_path: pathlib.Path) -> Optional[pathlib.Path]:
    if not input_path.is_file() or start_time < 0 or end_time <= start_time: return None
    if not check_ffmpeg_install("ffmpeg"): return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_temp_files(output_path)

    try:
        print(f"  Attempting to trim video: {input_path.name}")
        print(f"  Time range: {start_time:.2f}s to {end_time:.2f}s")
        
        # First try with stream copy (fast but not always accurate for exact frames)
        command_copy = [
            'ffmpeg', '-i', str(input_path), '-ss', str(start_time), '-to', str(end_time),
            '-c:v', 'copy', '-c:a', 'copy', '-avoid_negative_ts', 'make_zero',
            '-loglevel', 'warning', '-y', str(output_path)
        ]
        print("  Running ffmpeg copy command:", ' '.join(command_copy))
        result = subprocess.run(command_copy, capture_output=True, text=False, check=False, timeout=60)
        
        stderr = result.stderr.decode('utf-8', errors='replace') if result.stderr else ''
        stdout = result.stdout.decode('utf-8', errors='replace') if result.stdout else ''
        
        print("\n  FFmpeg copy result:")
        print(f"  - Return code: {result.returncode}")
        print(f"  - Error output: {stderr if stderr else 'None'}")
        print(f"  - Standard output: {stdout if stdout else 'None'}")
        
        # Verify input file still exists
        if not input_path.is_file():
            print(f"  Error: Input file no longer exists: {input_path}")
            return None
            
        # Check output file
        size = output_path.stat().st_size if output_path.is_file() else 0
        if result.returncode != 0:
            print(f"  FFmpeg copy failed with code {result.returncode}")
            cleanup_temp_files(output_path)
        elif not output_path.is_file():
            print("  FFmpeg copy completed but output file not found")
            cleanup_temp_files(output_path)
        elif size < 1024:
            print(f"  FFmpeg copy output too small: {size} bytes")
            cleanup_temp_files(output_path)
        else:
            print(f"  FFmpeg copy successful: {size} bytes")
            if size >= 10240:
                return output_path
            
            # If copy method failed, use re-encoding with GPU if available
            print("  Fast copy failed, attempting re-encode method...")
            use_gpu = has_nvidia_gpu()
            codec = VIDEO_CODEC_GPU if use_gpu else VIDEO_CODEC_CPU
            preset = FFMPEG_GPU_PRESET if use_gpu else FFMPEG_CPU_PRESET
            quality = FFMPEG_CQ_GPU if use_gpu else FFMPEG_CRF_CPU
            
            print(f"  Using {'GPU' if use_gpu else 'CPU'} encoding:")
            print(f"  - Codec: {codec}")
            print(f"  - Preset: {preset}")
            print(f"  - Quality: {quality}")
            
            command_recode = [
                'ffmpeg', '-i', str(input_path), '-ss', str(start_time), '-to', str(end_time),
                '-c:v', codec, '-preset', preset,
                '-loglevel', 'warning',
                '-max_muxing_queue_size', '1024'
            ]
            
            # Add quality parameter with appropriate flag
            if use_gpu:
                command_recode.extend(['-cq:v', quality])
            else:
                command_recode.extend(['-crf', quality])
                
            command_recode.extend([
                '-c:a', AUDIO_CODEC, '-b:a', '192k',
                '-avoid_negative_ts', 'make_zero',
                '-y', str(output_path)
            ])
            
            print("  Executing ffmpeg re-encode command...")
            
            print("  Running ffmpeg command:", ' '.join(command_recode))
            result = subprocess.run(command_recode, capture_output=True, text=False, check=False, timeout=120)
            
            stderr = result.stderr.decode('utf-8', errors='replace') if result.stderr else ''
            stdout = result.stdout.decode('utf-8', errors='replace') if result.stdout else ''
            
            print("  FFmpeg recode result:")
            print(f"  - Return code: {result.returncode}")
            print(f"  - Error output: {stderr if stderr else 'None'}")
            print(f"  - Standard output: {stdout if stdout else 'None'}")
            
            if result.returncode != 0:
                print(f"  FFmpeg recode failed with code {result.returncode}")
                cleanup_temp_files(output_path)
                return None

            if not output_path.is_file():
                print("  FFmpeg recode completed but output file not found")
                cleanup_temp_files(output_path)
                return None
                
            size = output_path.stat().st_size if output_path.is_file() else 0
            if size < 10240:
                print(f"  FFmpeg recode output too small: {size} bytes")
                cleanup_temp_files(output_path)
                return None
                
            print(f"  Successfully created trimmed video: {size} bytes")
            
        return output_path
        
    except subprocess.TimeoutExpired as e:
        print(f"  Error: FFmpeg process timed out: {e}")
        cleanup_temp_files(output_path)
        return None
    except Exception as e:
        print(f"  Error trimming video: {str(e)}")
        cleanup_temp_files(output_path)
        return None

def ensure_frame_shape(frame: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    h, w, c = target_shape
    fh, fw = frame.shape[:2]
    fc = frame.shape[2] if frame.ndim == 3 else 1

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR if c == 3 else cv2.COLOR_GRAY2BGRA)
        fc = frame.shape[2]
    if fc != c:
        if fc == 3 and c == 4: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        elif fc == 4 and c == 3: frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif fc < c:
             if fc == 1 and c >= 3: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR if c == 3 else cv2.COLOR_GRAY2BGRA)
             else: # Basic padding
                 padding = np.zeros((fh, fw, c - fc), dtype=frame.dtype)
                 frame = np.concatenate((frame, padding), axis=2)
        else: frame = frame[:, :, :c] # Slice channels

    fh, fw = frame.shape[:2]
    if fh != h:
        if fh < h:
            pad_top = (h - fh) // 2; pad_bottom = h - fh - pad_top
            frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, 0, 0, cv2.BORDER_REFLECT_101)
        else: crop_top = (fh - h) // 2; frame = frame[crop_top:crop_top + h, :]
    fw = frame.shape[1]
    if fw != w:
        if fw < w:
            pad_left = (w - fw) // 2; pad_right = w - fw - pad_left
            frame = cv2.copyMakeBorder(frame, 0, 0, pad_left, pad_right, cv2.BORDER_REFLECT_101)
        else: crop_left = (fw - w) // 2; frame = frame[:, crop_left:crop_left + w]

    if frame.shape[0] != h or frame.shape[1] != w:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    if frame.shape[2] != c: # Final channel check
         frame = frame[:, :, :c] if frame.shape[2] > c else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR if c == 3 else cv2.COLOR_GRAY2BGRA) # Simplistic correction

    return frame

def ensure_rgba(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if frame is None: return None
    if frame.ndim == 2: return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)
    elif frame.shape[2] == 3: return cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    elif frame.shape[2] == 4: return frame
    elif frame.shape[2] == 1: return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)
    else:
        try: return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGBA)
        except Exception: return frame

def apply_subtle_zoom(clip: VideoClip, zoom_range: Tuple[float, float] = (1.01, 1.06)) -> VideoClip:
    z_start, z_end = zoom_range
    if random.choice([True, False]): z_start, z_end = z_end, z_start
    def zoom_factor(t):
        return z_start + (z_end - z_start) * (t / clip.duration) if clip.duration and clip.duration > 0 else 1.0
    return clip.resize(zoom_factor)

def apply_shake(clip: VideoClip, max_translate: int = 5, shake_duration: float = 0.3, shake_times: int = 3, parallel: bool = PARALLEL_FRAME_PROCESSING, n_jobs: int = N_JOBS_PARALLEL) -> VideoClip:
    if clip.duration is None or clip.duration <= shake_duration or shake_times <= 0: return clip

    shake_moments = sorted(random.uniform(0, max(0.01, clip.duration - shake_duration)) for _ in range(shake_times))
    target_shape = (*clip.size[::-1], 3 if not clip.ismask else 4)

    def shake_transform_frame(frame: np.ndarray, t: float) -> np.ndarray:
        offset_x, offset_y = 0.0, 0.0
        active_shake = False
        for moment in shake_moments:
            if moment <= t <= moment + shake_duration:
                progress = (t - moment) / shake_duration
                intensity = math.sin(progress * math.pi)
                offset_x += random.uniform(-max_translate, max_translate) * intensity
                offset_y += random.uniform(-max_translate, max_translate) * intensity
                active_shake = True
        if active_shake:
            h, w = frame.shape[:2]
            M = np.float32([[1, 0, int(round(offset_x))], [0, 1, int(round(offset_y))]])
            try:
                warped_frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
                return ensure_frame_shape(warped_frame, target_shape)
            except Exception: return ensure_frame_shape(frame, target_shape)
        else: return ensure_frame_shape(frame, target_shape)

    if parallel and n_jobs > 1:
        frames = list(clip.iter_frames(with_times=True, dtype='uint8', logger=None))
        if not frames: return clip
        processed_frames = Parallel(n_jobs=n_jobs, prefer="threads")( # Use threads for less overhead with numpy/cv2
            delayed(shake_transform_frame)(f, t) for t, f in frames
        )
        if not processed_frames or len(processed_frames) != len(frames): return clip
        try:
            return ImageSequenceClip(processed_frames, fps=clip.fps).set_duration(clip.duration).set_audio(clip.audio)
        except Exception as e:
            print(f"Error creating ImageSequenceClip for shake: {e}")
            return clip # Fallback
    else:
        return clip.fl(lambda gf, t: shake_transform_frame(gf(t), t), apply_to='video')

def apply_color_grade(clip: VideoClip) -> VideoClip:
    preset = random.choice(['none', 'cool', 'warm', 'vibrant', 'subtle_contrast'])
    if preset == 'cool':
        clip = clip.fx(colorx, factor=random.uniform(0.95, 0.99))
        clip = clip.fx(lum_contrast, contrast=random.uniform(0.03, 0.07), lum=random.uniform(-2, 1))
    elif preset == 'warm':
        clip = clip.fx(colorx, factor=random.uniform(1.01, 1.05))
        clip = clip.fx(lum_contrast, contrast=random.uniform(0.03, 0.07), lum=random.uniform(-1, 2))
    elif preset == 'vibrant':
        clip = clip.fx(colorx, factor=random.uniform(1.08, 1.15))
        clip = clip.fx(lum_contrast, contrast=random.uniform(0.10, 0.18), lum=random.uniform(-3, 3))
    elif preset == 'subtle_contrast':
        clip = clip.fx(lum_contrast, contrast=random.uniform(0.05, 0.12), lum=random.uniform(-4, 4))
    return clip

def add_visual_emphasis(clip: VideoClip, focus_x: float = 0.5, focus_y: float = 0.5,
                        start_time: float = 0, duration: float = 1.5, zoom_factor: float = 1.1) -> VideoClip:
    w, h = clip.size
    if duration <= 0 or zoom_factor <= 1.0: return clip

    def calculate_zoom(t):
        if start_time <= t <= start_time + duration:
            progress = (t - start_time) / duration
            return 1 + (zoom_factor - 1) * math.sin(progress * math.pi)
        return 1.0

    # Create a function that resizes and crops to keep focus point centered
    def zoom_and_crop_frame(get_frame, t):
        frame = get_frame(t)
        current_zoom = calculate_zoom(t)
        if current_zoom == 1.0: return frame # No effect outside duration

        # Resize
        zoomed_h, zoomed_w = int(h * current_zoom), int(w * current_zoom)
        # Need resizing that might create non-standard shapes, OpenCV is better here
        frame_resized = cv2.resize(frame, (zoomed_w, zoomed_h), interpolation=cv2.INTER_LINEAR)

        # Calculate crop coordinates to keep focus point centered
        crop_x = int(focus_x * zoomed_w - w / 2)
        crop_y = int(focus_y * zoomed_h - h / 2)

        # Clamp crop coordinates to be within the resized frame bounds
        crop_x = max(0, min(crop_x, zoomed_w - w))
        crop_y = max(0, min(crop_y, zoomed_h - h))

        # Crop
        cropped_frame = frame_resized[crop_y:crop_y + h, crop_x:crop_x + w]

        # Ensure output shape matches original (handling potential rounding errors)
        if cropped_frame.shape[0] != h or cropped_frame.shape[1] != w:
             cropped_frame = cv2.resize(cropped_frame, (w, h), interpolation=cv2.INTER_AREA)

        return cropped_frame

    # Use fl for frame-by-frame transformation
    return clip.fl(zoom_and_crop_frame)


def create_animated_subtitle_clips(speech_segments: List[Dict], video_width: int, video_height: int) -> List[TextClip]:
    if not speech_segments: return []

    subtitle_clips = []
    font_size = max(24, int(video_height * OVERLAY_FONT_SIZE_RATIO))

    for i, segment in enumerate(speech_segments):
        try:
            text = segment.get('text', '').strip()
            start_time = float(segment.get('start_time', -1))
            end_time = float(segment.get('end_time', -1))
            if not text or start_time < 0 or end_time <= start_time: continue

            max_chars_per_line = 35; lines = []; current_line = ""
            for word in text.split():
                if not current_line: current_line = word
                elif len(current_line) + len(word) + 1 <= max_chars_per_line: current_line += " " + word
                else: lines.append(current_line); current_line = word
            lines.append(current_line)
            processed_text = "\n".join(lines[:3])
            if len(lines) > 3: processed_text += "..."

            txt_clip = TextClip(
                txt=processed_text, fontsize=font_size, font=OVERLAY_FONT,
                color=OVERLAY_TEXT_COLOR, bg_color=OVERLAY_BG_COLOR,
                stroke_color=OVERLAY_STROKE_COLOR, stroke_width=OVERLAY_STROKE_WIDTH,
                method='caption', align='center', size=(int(video_width * 0.9), None)
            )
            clip_duration = end_time - start_time
            txt_clip = txt_clip.set_start(start_time).set_duration(clip_duration).set_position(OVERLAY_POSITION)
            fade_duration = min(0.2, clip_duration * 0.15)
            if fade_duration > 0.01: txt_clip = txt_clip.fadein(fade_duration).fadeout(fade_duration)

            subtitle_clips.append(txt_clip)
        except Exception as e:
            print(f"  Error creating subtitle clip {i}: {e}")
            traceback.print_exc()

    return subtitle_clips

def _prepare_audio_tracks(
    original_video_for_audio_path: Optional[pathlib.Path],
    tts_audio_path: Optional[pathlib.Path],
    video_duration: float,
    gemini_data: Optional[Dict[str, Any]],
    clips_to_close: List[Any],
    temp_files_to_clean: List[pathlib.Path]
) -> Tuple[List[AudioFileClip], Optional[AudioFileClip]]:
    """Loads, processes, and prepares original, TTS, and music audio tracks."""
    print(f"  Preparing audio tracks for {video_duration:.2f}s video")
    audio_tracks = []
    original_audio, tts_audio, bg_music_audio = None, None, None
    
    def safely_load_audio(path: pathlib.Path, description: str) -> Optional[AudioFileClip]:
        try:
            print(f"  Loading {description} from: {path.name}")
            clip = AudioFileClip(str(path))
            if clip.duration <= 0:
                print(f"  Warning: {description} has zero duration")
                clip.close()
                return None
            return clip
        except Exception as e:
            print(f"  Error loading {description}: {e}")
            return None

    # Original Audio
    if MIX_ORIGINAL_AUDIO and original_video_for_audio_path and original_video_for_audio_path.is_file():
        orig_audio_clip = safely_load_audio(original_video_for_audio_path, "original audio")
        if orig_audio_clip:
            clips_to_close.append(orig_audio_clip)
            try:
                # Ensure audio duration matches video duration
                if orig_audio_clip.duration > video_duration:
                    original_audio = orig_audio_clip.subclip(0, video_duration)
                else:
                    original_audio = orig_audio_clip
                    
                original_audio = original_audio.set_duration(video_duration)
                original_audio = original_audio.volumex(ORIGINAL_AUDIO_MIX_VOLUME)
                print(f"  Prepared original audio: {original_audio.duration:.2f}s")
                audio_tracks.append(original_audio)
            except Exception as e:
                print(f"  Error processing original audio: {e}")
                if orig_audio_clip in clips_to_close:
                    clips_to_close.remove(orig_audio_clip)
                    orig_audio_clip.close()

    # TTS Audio
    if tts_audio_path and tts_audio_path.is_file() and tts_audio_path.stat().st_size > 0:
        print(f"  Processing TTS audio from: {tts_audio_path.name}")
        norm_tts_path = TEMP_DIR / f"{tts_audio_path.stem}_norm.aac"
        temp_files_to_clean.append(norm_tts_path)
        
        # First convert to proper format using ffmpeg
        converted_tts_path = TEMP_DIR / f"{tts_audio_path.stem}_converted.wav"
        temp_files_to_clean.append(converted_tts_path)
        
        try:
            # Convert to standard format: mono, 44100Hz, 16-bit WAV
            command = [
                "ffmpeg", "-y", "-i", str(tts_audio_path),
                "-ac", "1", "-ar", "44100", "-acodec", "pcm_s16le",
                "-hide_banner", "-loglevel", "error",
                str(converted_tts_path)
            ]
            result = subprocess.run(command, check=True, timeout=30, capture_output=True)
            
            if not converted_tts_path.is_file() or converted_tts_path.stat().st_size == 0:
                raise ValueError("FFmpeg conversion failed - empty output file")
                
            # Normalize loudness with more robust checks
            if normalize_loudness(converted_tts_path, norm_tts_path, target_lufs=LOUDNESS_TARGET_LUFS):
                tts_audio_clip = safely_load_audio(norm_tts_path, "normalized TTS audio")
                if not tts_audio_clip or tts_audio_clip.duration <= 0:
                    raise ValueError("Invalid TTS audio clip after normalization")
        except Exception as e:
            print(f"  Error processing TTS audio: {e}")
            # Fallback to direct processing if conversion fails
            try:
                if normalize_loudness(tts_audio_path, norm_tts_path, target_lufs=LOUDNESS_TARGET_LUFS):
                    tts_audio_clip = safely_load_audio(norm_tts_path, "normalized TTS audio")
                    if not tts_audio_clip or tts_audio_clip.duration <= 0:
                        raise ValueError("Invalid TTS audio in fallback processing")
            except Exception as fallback_error:
                print(f"  Fallback processing failed: {fallback_error}")
                return audio_tracks, None
                
        if tts_audio_clip:
            clips_to_close.append(tts_audio_clip)
            try:
                # Handle TTS duration
                if tts_audio_clip.duration > video_duration:
                    tts_audio = tts_audio_clip.subclip(0, video_duration)
                else:
                    tts_audio = tts_audio_clip
                
                tts_audio = tts_audio.set_duration(video_duration)
                print(f"  Prepared TTS audio: {tts_audio.duration:.2f}s")
                audio_tracks.append(tts_audio)
            except Exception as e:
                print(f"  Error finalizing TTS audio: {e}")
                if tts_audio_clip in clips_to_close:
                    clips_to_close.remove(tts_audio_clip)
                    tts_audio_clip.close()
                    if tts_audio_clip in clips_to_close:
                        clips_to_close.remove(tts_audio_clip)
                        tts_audio_clip.close()
        else:
            print("  Warning: TTS audio normalization failed")

    # Background Music
    if MUSIC_FOLDER.is_dir():
        print("  Processing background music")
        music_files = []
        for ext in ['.mp3', '.wav', '.aac', '.m4a']:
            music_files.extend(MUSIC_FOLDER.glob(f"*{ext}"))
            
        if music_files:
            bg_music_path = random.choice(music_files)
            print(f"  Selected music: {bg_music_path.name}")
            
            try:
                bg_music_clip_orig = safely_load_audio(bg_music_path, "background music")
                if bg_music_clip_orig:
                    clips_to_close.append(bg_music_clip_orig)
                    
                    # Calculate how many loops we need
                    loops_needed = math.ceil(video_duration / bg_music_clip_orig.duration)
                    print(f"  Looping music {loops_needed} times to cover {video_duration:.2f}s")
                    
                    # Create looped version if needed
                    if loops_needed > 1:
                        bg_music_list = [bg_music_clip_orig] * loops_needed
                        bg_music_concat = concatenate_audioclips(bg_music_list)
                        clips_to_close.append(bg_music_concat)
                    else:
                        bg_music_concat = bg_music_clip_orig
                    
                    # Trim to exact duration and apply volume
                    bg_music_processed = bg_music_concat.subclip(0, video_duration).set_duration(video_duration)
                    
                    # Apply ducking if needed
                    # Move ducking_volume_func outside of _prepare_audio_tracks
                    def ducking_volume_func(t: float, segments: List[Dict]) -> float:
                        """Calculate volume multiplier for audio ducking at a given time."""
                        for seg in segments:
                            try:
                                start = float(seg.get('start_time', -1))
                                end = float(seg.get('end_time', -1))
                                if start >= 0 and end > start:
                                    fade_start = start - AUDIO_DUCKING_FADE_TIME
                                    fade_end = end + AUDIO_DUCKING_FADE_TIME
                                    if fade_start <= t <= fade_end:
                                        if fade_start <= t < start:  # Fade down
                                            progress = (t - fade_start) / AUDIO_DUCKING_FADE_TIME
                                            return BACKGROUND_MUSIC_VOLUME - (BACKGROUND_MUSIC_VOLUME - BACKGROUND_MUSIC_DUCKED_VOLUME) * progress
                                        elif end < t <= fade_end:  # Fade up
                                            progress = (t - end) / AUDIO_DUCKING_FADE_TIME
                                            return BACKGROUND_MUSIC_DUCKED_VOLUME + (BACKGROUND_MUSIC_VOLUME - BACKGROUND_MUSIC_DUCKED_VOLUME) * progress
                                        else:  # During speech
                                            return BACKGROUND_MUSIC_DUCKED_VOLUME
                            except (ValueError, TypeError, KeyError):
                                continue
                        return BACKGROUND_MUSIC_VOLUME
                
                    speech_segments = gemini_data.get("speech_segments", []) if gemini_data else []
                    if isinstance(speech_segments, list) and speech_segments:
                        print("  Applying audio ducking for speech segments")
                        bg_music_audio = bg_music_processed.fl(
                            lambda gf, t: ducking_volume_func(t, speech_segments) * gf(t)
                        )
                    else:
                        print("  Applying constant volume to background music")
                        bg_music_audio = bg_music_processed.volumex(BACKGROUND_MUSIC_VOLUME)
                    
                    clips_to_close.append(bg_music_audio)
                    audio_tracks.append(bg_music_audio)
                    print(f"  Successfully prepared background music: {bg_music_audio.duration:.2f}s")
                    
            except Exception as e:
                print(f"  Warning: Failed to process background music {bg_music_path.name}: {e}")
                # Clean up any opened clips
                for clip in [bg_music_clip_orig, bg_music_concat, bg_music_audio]:
                    if clip in clips_to_close:
                        clips_to_close.remove(clip)
                        clip.close()

                bg_music_processed = bg_music_concat.subclip(0, video_duration).set_duration(video_duration)

                # Apply ducking
                speech_segments = gemini_data.get("speech_segments", []) if gemini_data else []
                if isinstance(speech_segments, list) and speech_segments:
                    # Create a fixed volume function that returns a float based on time
                    def ducking_volume_func(t):
                        for seg in speech_segments:
                            try:
                                start = float(seg.get('start_time', -1))
                                end = float(seg.get('end_time', -1))
                                if start >= 0 and end > start:
                                    fade_start = start - AUDIO_DUCKING_FADE_TIME
                                    fade_end = end + AUDIO_DUCKING_FADE_TIME
                                    if fade_start <= t <= fade_end:
                                        if fade_start <= t < start: # Fade down
                                            progress = max(0, min(1, (t - fade_start) / AUDIO_DUCKING_FADE_TIME))
                                            return BACKGROUND_MUSIC_VOLUME - (BACKGROUND_MUSIC_VOLUME - BACKGROUND_MUSIC_DUCKED_VOLUME) * progress
                                        elif end < t <= fade_end: # Fade up
                                            progress = max(0, min(1, (t - end) / AUDIO_DUCKING_FADE_TIME))
                                            return BACKGROUND_MUSIC_DUCKED_VOLUME + (BACKGROUND_MUSIC_VOLUME - BACKGROUND_MUSIC_DUCKED_VOLUME) * progress
                                        else: # During speech
                                            return BACKGROUND_MUSIC_DUCKED_VOLUME
                            except (ValueError, TypeError): continue
                        return BACKGROUND_MUSIC_VOLUME
                    
                    # Apply the volumex effect with our volume function
                    bg_music_audio = bg_music_processed.fl(lambda gf, t: ducking_volume_func(t) * gf(t))
                else:
                    bg_music_audio = bg_music_processed.volumex(BACKGROUND_MUSIC_VOLUME)

                clips_to_close.append(bg_music_audio) # Track the final processed music clip
                audio_tracks.append(bg_music_audio)

            except Exception as e:
                print(f"  Warning: Failed to process background music {bg_music_path.name}: {e}")
                # Clean up potentially opened clips related to this music file
                if 'bg_music_clip_orig' in locals() and bg_music_clip_orig in clips_to_close: clips_to_close.remove(bg_music_clip_orig); bg_music_clip_orig.close()
                if 'bg_music_concat' in locals() and bg_music_concat in clips_to_close: clips_to_close.remove(bg_music_concat); bg_music_concat.close()
                if 'bg_music_audio' in locals() and bg_music_audio in clips_to_close: clips_to_close.remove(bg_music_audio); bg_music_audio.close()
                bg_music_audio = None # Ensure it's not used

    # Combine Audio Tracks
    print(f"\n  Compositing {len(audio_tracks)} audio tracks:")
    final_audio_composite = None
    
    try:
        if len(audio_tracks) > 1:
            # Filter and validate tracks
            valid_tracks = []
            for i, track in enumerate(audio_tracks):
                if hasattr(track, 'duration') and track.duration is not None:
                    if track.duration > 0.01:
                        try:
                            print(f"    Processing track {i+1}: {track.duration:.2f}s")
                            # Get audio data and reshape properly
                            fps = track.fps if hasattr(track, 'fps') else 44100
                            print(f"      Track FPS: {fps}")
                            
                            try:
                                print(f"      Processing audio clip...")
                                
                                # Create a copy of the track
                                new_track = track.copy()
                                
                                # Set exact duration
                                if new_track.duration > video_duration:
                                    new_track = new_track.subclip(0, video_duration)
                                new_track = new_track.set_duration(video_duration)
                                
                                # Force stereo if needed
                                try:
                                    if hasattr(new_track, 'nchannels') and new_track.nchannels == 1:
                                        print("      Converting mono to stereo")
                                        temp_file = TEMP_DIR / f"temp_audio_{i}_{int(time.time())}.wav"
                                        new_track.write_audiofile(
                                            str(temp_file),
                                            fps=44100,
                                            nbytes=2,
                                            codec='pcm_s16le',
                                            ffmpeg_params=['-ac', '2']
                                        )
                                        new_track.close()
                                        new_track = AudioFileClip(str(temp_file))
                                        new_track = new_track.set_duration(video_duration)
                                        temp_file.unlink(missing_ok=True)
                                except Exception as e:
                                    print(f"      Warning during stereo conversion: {e}")
                                
                                print(f"      Successfully processed audio track")
                                valid_tracks.append(new_track)
                                clips_to_close.append(new_track)
                                
                            except Exception as e:
                                print(f"      Error processing audio track: {e}")
                                if 'new_track' in locals() and new_track != track:
                                    try: new_track.close()
                                    except: pass
                            
                            # Create new clip from processed array
                            from moviepy.audio.AudioClip import AudioArrayClip
                            new_track = AudioArrayClip(audio_array, fps=fps)
                            new_track = new_track.set_fps(fps)  # Ensure fps is set
                            new_track = new_track.set_duration(video_duration)
                            
                            print(f"    Track {i+1} processed successfully")
                            valid_tracks.append(new_track)
                            clips_to_close.append(new_track)
                        except Exception as e:
                            print(f"    Error processing track {i+1}: {e}")
                            continue
                    else:
                        print(f"    Skipping track {i+1}: Duration too short ({track.duration:.2f}s)")
                else:
                    print(f"    Skipping track {i+1}: Invalid duration")
            
            if len(valid_tracks) >= 1:
                try:
                    print(f"  Creating composite from {len(valid_tracks)} valid tracks")
                    final_audio_composite = CompositeAudioClip(valid_tracks)
                    final_audio_composite = final_audio_composite.set_duration(video_duration)
                    clips_to_close.append(final_audio_composite)
                except Exception as e:
                    print(f"  Error creating audio composite: {e}")
                    for track in valid_tracks:
                        if track in clips_to_close:
                            clips_to_close.remove(track)
                            try: track.close()
                            except: pass
                    final_audio_composite = None
            else:
                print("  Warning: No valid audio tracks to composite")
                
        elif len(audio_tracks) == 1:
            print("  Using single audio track")
            final_audio_composite = audio_tracks[0].set_duration(video_duration)
        else:
            print("  Warning: No audio tracks available")
            
        if final_audio_composite:
            print(f"  Final audio duration: {final_audio_composite.duration:.2f}s")
            
    except Exception as e:
        print(f"  Error during audio composition: {e}")
        if final_audio_composite in clips_to_close:
            clips_to_close.remove(final_audio_composite)
            final_audio_composite.close()
        final_audio_composite = None

    return audio_tracks, final_audio_composite


def _apply_video_effects(
    video_clip: VideoClip,
    gemini_data: Optional[Dict[str, Any]]
) -> VideoClip:
    """Applies sequential video effects like zoom, shake, color grade, emphasis."""
    mood = gemini_data.get("mood", "unknown").lower() if gemini_data else "unknown"

    if SUBTLE_ZOOM_ENABLED:
        video_clip = apply_subtle_zoom(video_clip)
    if SHAKE_EFFECT_ENABLED:
        video_clip = apply_shake(
            video_clip, max_translate=int(random.uniform(3, 6)),
            shake_duration=random.uniform(0.2, 0.4), shake_times=random.randint(2, 4)
        )
    if COLOR_GRADE_ENABLED:
        video_clip = apply_color_grade(video_clip)
    if ADD_VISUAL_EMPHASIS and gemini_data and "key_visual_moments" in gemini_data:
        key_moments = gemini_data.get("key_visual_moments", [])
        if isinstance(key_moments, list):
            for moment in key_moments:
                try:
                    t = float(moment.get('timestamp', -1))
                    fp = moment.get('focus_point', {})
                    fx = float(fp.get('x', 0.5)); fy = float(fp.get('y', 0.5))
                    if 0 <= t < video_clip.duration:
                        emphasis_duration = random.uniform(1.2, 1.8)
                        emphasis_start = max(0, t - emphasis_duration / 2)
                        emphasis_end = min(video_clip.duration, emphasis_start + emphasis_duration)
                        emphasis_duration = emphasis_end - emphasis_start
                        if emphasis_duration > 0.1:
                             video_clip = add_visual_emphasis(
                                video_clip, focus_x=fx, focus_y=fy, start_time=emphasis_start,
                                duration=emphasis_duration, zoom_factor=random.uniform(1.08, 1.15)
                            )
                except (ValueError, TypeError, KeyError) as parse_err:
                    print(f"  - Warning: Skipping emphasis due to invalid moment data: {moment}, Error: {parse_err}")
    return video_clip

def _prepare_overlays(
    video_w: int,
    video_h: int,
    video_duration: float,
    subtitle_clips: List[TextClip],
    clips_to_close: List[Any]
) -> List[Union[ImageClip, TextClip]]:
    """Creates watermark and prepares subtitle clips."""
    overlays = []
    if WATERMARK_PATH.is_file():
        try:
            # Fixed watermark handling - use ImageClip without ismask parameter
            watermark_clip = (ImageClip(str(WATERMARK_PATH))
                             .set_duration(video_duration)
                             .resize(height=max(20, int(video_h * 0.05)))
                             .margin(right=15, bottom=15, opacity=0)
                             .set_position(("right", "bottom"))
                             .set_opacity(0.6))
            clips_to_close.append(watermark_clip)
            overlays.append(watermark_clip)
        except Exception as e_wm: print(f"  Warning: Failed to create watermark overlay: {e_wm}")

    if subtitle_clips:
        overlays.extend(subtitle_clips)
        clips_to_close.extend(subtitle_clips)
    return overlays

def combine_video_audio_subs(
    processed_video_path: pathlib.Path,
    original_video_for_audio_path: Optional[pathlib.Path],
    tts_audio_path: Optional[pathlib.Path],
    subtitle_clips: List[TextClip],
    output_path: pathlib.Path,
    gemini_data: Optional[Dict[str, Any]] = None,
    stabilization_transforms_file: Optional[pathlib.Path] = None
) -> Optional[pathlib.Path]:

    clips_to_close: List[Any] = []
    temp_files_to_clean: List[pathlib.Path] = []
    final_composite_clip = None # Define here for finally block

    try:
        if not processed_video_path.is_file(): raise FileNotFoundError(f"Input video not found: {processed_video_path}")

        video_clip = VideoFileClip(str(processed_video_path))
        clips_to_close.append(video_clip)
        video_duration = video_clip.duration
        video_w, video_h = video_clip.size
        if not video_duration or video_duration <= 0 or video_w <= 0 or video_h <= 0:
            raise ValueError("Loaded video clip has invalid properties.")

        video_clip = _apply_video_effects(video_clip, gemini_data)

        # Process audio tracks individually using ffmpeg directly
        print("  Processing audio tracks...")
        temp_final_audio = TEMP_DIR / f"{output_path.stem}_final_audio.m4a"
        temp_files_to_clean.append(temp_final_audio)
        final_audio = None
        
        try:
            # Process TTS if it exists
            if tts_audio_path and tts_audio_path.is_file():
                print(f"  Processing TTS audio from: {tts_audio_path}")
                temp_tts = TEMP_DIR / f"{output_path.stem}_tts.wav"
                temp_files_to_clean.append(temp_tts)
                
                try:
                    # First check if input file is valid
                    probe_result = subprocess.run(
                        ['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type', '-of', 'default=nw=1:nk=1', str(tts_audio_path)],
                        capture_output=True, text=True, check=False
                    )
                    
                    if 'audio' in probe_result.stdout:
                        print("  Converting TTS to WAV format...")
                        # First convert to intermediate format
                        temp_tts_raw = TEMP_DIR / f"{output_path.stem}_tts_raw.wav"
                        temp_files_to_clean.append(temp_tts_raw)
                        
                        def run_ffmpeg(cmd_list):
                            """Run ffmpeg command with proper encoding handling"""
                            try:
                                result = subprocess.run(
                                    cmd_list,
                                    capture_output=True,
                                    encoding='utf-8',
                                    errors='replace',
                                    check=False
                                )
                                return result.returncode == 0, result.stderr
                            except Exception as e:
                                return False, str(e)
                        
                        print("  Converting TTS to WAV...")
                        success, error = run_ffmpeg([
                            'ffmpeg', '-y',
                            '-i', str(tts_audio_path),
                            '-vn',  # No video
                            '-acodec', 'pcm_s16le',
                            '-hide_banner',
                            str(temp_tts_raw)
                        ])
                        
                        if success:
                            print("  Normalizing audio format...")
                            success, error = run_ffmpeg([
                                'ffmpeg', '-y',
                                '-i', str(temp_tts_raw),
                                '-ar', '44100',
                                '-ac', '2',
                                '-acodec', 'pcm_s16le',
                                '-hide_banner',
                                str(temp_tts)
                            ])
                            if not success:
                                print(f"  Error normalizing audio: {error}")
                        else:
                            print(f"  Error converting to WAV: {error}")
                            temp_tts_raw.unlink(missing_ok=True)
                    else:
                        print("  Error: TTS file contains no valid audio stream")
                        raise ValueError("No audio stream found in TTS file")
                        
                except Exception as e:
                    print(f"  Failed to convert TTS: {e}")
                    if temp_tts.exists():
                        temp_tts.unlink()
                    raise
                
                if temp_tts.is_file():
                    # Process background music if it exists
                    if original_video_for_audio_path and original_video_for_audio_path.is_file():
                        print(f"  Processing background music from: {original_video_for_audio_path}")
                        temp_music = TEMP_DIR / f"{output_path.stem}_music.wav"
                        temp_files_to_clean.append(temp_music)
                        
                        try:
                            # Check if input file has audio
                            probe_result = subprocess.run(
                                ['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type', '-of', 'default=nw=1:nk=1', str(original_video_for_audio_path)],
                                capture_output=True, text=True, check=False
                            )
                            
                            if 'audio' in probe_result.stdout:
                                print("  Converting background music to WAV format...")
                                # First convert to intermediate format
                                temp_music_raw = TEMP_DIR / f"{output_path.stem}_music_raw.wav"
                                temp_files_to_clean.append(temp_music_raw)
                                
                                print("  Converting music to WAV...")
                                success, error = run_ffmpeg([
                                    'ffmpeg', '-y',
                                    '-i', str(original_video_for_audio_path),
                                    '-vn',  # No video
                                    '-acodec', 'pcm_s16le',
                                    '-ar', '44100',
                                    '-ac', '2',
                                    '-hide_banner',
                                    str(temp_music_raw)
                                ])
                                
                                if success:
                                    print("  Adjusting music volume...")
                                    success, error = run_ffmpeg([
                                        'ffmpeg', '-y',
                                        '-i', str(temp_music_raw),
                                        '-filter:a', f'volume={BACKGROUND_MUSIC_VOLUME}',
                                        '-acodec', 'pcm_s16le',
                                        '-hide_banner',
                                        str(temp_music)
                                    ])
                                    if not success:
                                        print(f"  Error adjusting volume: {error}")
                                else:
                                    print(f"  Error converting music to WAV: {error}")
                                    temp_music_raw.unlink(missing_ok=True)
                                
                                if result.returncode != 0:
                                    print(f"  FFmpeg error: {result.stderr}")
                                    raise subprocess.CalledProcessError(result.returncode, result.args)
                                    
                                print("  Background music processed successfully")
                            else:
                                print("  Error: Background music file contains no valid audio stream")
                                raise ValueError("No audio stream found in background music file")
                                
                        except Exception as e:
                            print(f"  Failed to convert background music: {e}")
                            if temp_music.exists():
                                temp_music.unlink()
                        
                        if temp_music.is_file():
                            # Mix audio files if both exist
                            if temp_tts.is_file() and temp_music.is_file():
                                print("  Mixing TTS and background music...")
                                success, error = run_ffmpeg([
                                    'ffmpeg', '-y',
                                    '-i', str(temp_tts),
                                    '-i', str(temp_music),
                                    '-filter_complex',
                                    f'[1:a]volume={BACKGROUND_MUSIC_VOLUME}[bg];[bg]apad[bg1];[0:a]apad[tts1];[tts1][bg1]amerge=inputs=2,pan=stereo|c0<c0+c2|c1<c1+c3[a]',
                                    '-map', '[a]',
                                    '-c:a', 'aac',
                                    '-b:a', '192k',
                                    '-shortest',
                                    '-hide_banner',
                                    '-loglevel', 'error',
                                    str(temp_final_audio)
                                ])
                                
                                if not success:
                                    print(f"  FFmpeg mixing error: {error}")
                                    raise RuntimeError(f"Failed to mix audio: {error}")
                                print("  Audio mixing completed successfully")
                        else:
                            # Just use TTS if music conversion failed
                            shutil.copy(str(temp_tts), str(temp_final_audio))
                    else:
                        # Just use TTS if no music
                        shutil.copy(str(temp_tts), str(temp_final_audio))
                        
            if temp_final_audio.is_file() and temp_final_audio.stat().st_size > 1024:
                print("  Setting final audio...")
                final_audio = AudioFileClip(str(temp_final_audio))
                final_audio = final_audio.set_duration(video_duration)
                video_clip = video_clip.set_audio(final_audio)
                clips_to_close.append(final_audio)
            else:
                print("  Warning: No valid audio produced")
                video_clip = video_clip.set_audio(None)
                
        except Exception as e:
            print(f"  Error processing audio: {e}")
            video_clip = video_clip.set_audio(None)
        
        # Add overlays
        overlays = _prepare_overlays(video_w, video_h, video_duration, subtitle_clips, clips_to_close)
        final_composite_clip = CompositeVideoClip(
            [video_clip] + overlays,
            size=(video_w, video_h)
        ).set_duration(video_duration)
        clips_to_close.append(final_composite_clip)

        # Final duration clamp
        target_duration = TARGET_VIDEO_DURATION_SECONDS
        if final_composite_clip.duration > target_duration + 0.1:
            final_composite_clip = final_composite_clip.subclip(0, target_duration)
        elif final_composite_clip.duration <= 0:
            raise ValueError(f"Final composite clip has invalid duration: {final_composite_clip.duration:.2f}s")

        # Ensure dimensions are even
        final_w, final_h = final_composite_clip.size
        if final_w % 2 != 0 or final_h % 2 != 0:
            target_w_even = final_w - (final_w % 2); target_h_even = final_h - (final_h % 2)
            final_composite_clip = final_composite_clip.crop(width=target_w_even, height=target_h_even, x_center=final_w/2, y_center=final_h/2)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleanup_temp_files(output_path)

        use_gpu = has_nvidia_gpu()
        codec = VIDEO_CODEC_GPU if use_gpu else VIDEO_CODEC_CPU
        preset = FFMPEG_GPU_PRESET if use_gpu else FFMPEG_CPU_PRESET
        quality_param = ['-cq', FFMPEG_CQ_GPU] if use_gpu else ['-crf', FFMPEG_CRF_CPU]
        ffmpeg_extra_params = quality_param + ['-pix_fmt', 'yuv420p', '-movflags', '+faststart']

        if APPLY_STABILIZATION and stabilization_transforms_file and stabilization_transforms_file.is_file():
            escaped_transforms_path = str(stabilization_transforms_file).replace('\\', '/')
            stabilization_filter = f'vidstabtransform=input="{escaped_transforms_path}":zoom=0:smoothing=15,unsharp=5:5:0.8:3:3:0.4'
            try:
                vf_index = ffmpeg_extra_params.index('-vf'); ffmpeg_extra_params[vf_index + 1] += f',{stabilization_filter}'
            except ValueError: ffmpeg_extra_params.extend(['-vf', stabilization_filter])
        elif APPLY_STABILIZATION: print("  Warning: Stabilization requested but transforms file invalid.")

        temp_final_audio_path = TEMP_DIR / f"{output_path.stem}_final_audio.m4a"
        final_composite_clip.write_videofile(
            str(output_path), codec=codec, audio_codec=AUDIO_CODEC,
            temp_audiofile=str(temp_final_audio_path), remove_temp=True,
            preset=preset, fps=TARGET_FPS, threads=N_JOBS_PARALLEL,
            logger=None, ffmpeg_params=ffmpeg_extra_params
        )
        cleanup_temp_files(temp_final_audio_path)


        if not output_path.is_file() or output_path.stat().st_size < 10240:
             raise IOError(f"Final video file missing or empty: {output_path}")
        if final_audio and not has_audio_track(output_path):
             raise IOError(f"Final video missing audio track: {output_path.name}")

        return output_path

    except Exception as e:
        print(f"\n--- CRITICAL ERROR during video combination: {e} ---")
        traceback.print_exc()
        cleanup_temp_files(output_path) # Clean up potentially failed output
        return None
    finally:
        # Ensure all tracked clips are closed, even on error
        for clip_obj in clips_to_close:
            if clip_obj and hasattr(clip_obj, 'close'):
                try: clip_obj.close()
                except Exception: pass
        cleanup_temp_files(*temp_files_to_clean)


def _prepare_initial_video(submission: praw.models.Submission, safe_title: str, temp_files_list: List[pathlib.Path]) -> Tuple[Optional[pathlib.Path], float, int, int]:
    """Downloads, gets details, and converts to vertical."""
    download_path = TEMP_DIR / f"{submission.id}_{safe_title}_download.mp4"
    vertical_path = TEMP_DIR / f"{submission.id}_{safe_title}_vertical.mp4"
    temp_files_list.extend([download_path, vertical_path])

    actual_download_path = download_media(submission.url, download_path)
    if not actual_download_path: return None, 0.0, 0, 0

    duration, width, height = get_video_details(actual_download_path)
    if duration <= 0 or width <= 0 or height <= 0: return None, 0.0, 0, 0

    aspect_ratio = width / height
    if abs(aspect_ratio - TARGET_ASPECT_RATIO) > 0.05:
        processed_path = make_video_vertical(actual_download_path, vertical_path)
        if not processed_path: return None, 0.0, 0, 0
        # Get details of the vertical video
        duration_v, width_v, height_v = get_video_details(processed_path)
        return processed_path, duration_v, width_v, height_v
    else:
        return actual_download_path, duration, width, height # Return original if already vertical


def _get_processing_params(gemini_analysis: Dict[str, Any], original_duration: float) -> Tuple[float, float, str, bool]:
    """Determines start/end times and TTS details from Gemini analysis."""
    start_time = 0.0
    end_time = min(original_duration, TARGET_VIDEO_DURATION_SECONDS)
    trim_duration = end_time - start_time
    best_segment = gemini_analysis.get("best_segment")

    if isinstance(best_segment, dict) and 'start_time' in best_segment and 'end_time' in best_segment:
        try:
            seg_start = float(best_segment['start_time'])
            seg_end = float(best_segment['end_time'])
            seg_dur = seg_end - seg_start
            if 5 < seg_dur <= TARGET_VIDEO_DURATION_SECONDS and 0 <= seg_start < seg_end <= original_duration:
                start_time = seg_start; end_time = seg_end
            elif seg_dur > TARGET_VIDEO_DURATION_SECONDS and 0 <= seg_start < original_duration:
                start_time = seg_start; end_time = min(original_duration, seg_start + TARGET_VIDEO_DURATION_SECONDS)
            trim_duration = end_time - start_time # Recalculate trimmed duration
        except (ValueError, TypeError): pass

    tts_text = gemini_analysis.get("summary_for_description", "")
    generate_tts_flag = bool(tts_text and not MIX_ORIGINAL_AUDIO)

    return start_time, end_time, tts_text, generate_tts_flag


def _generate_assets(
    gemini_analysis: Dict[str, Any],
    tts_text: str,
    generate_tts_flag: bool,
    start_time: float,
    end_time: float,
    trimmed_video_path: pathlib.Path,
    submission_id: str,
    safe_title: str,
    temp_files_list: List[pathlib.Path]
) -> Tuple[Optional[pathlib.Path], List[TextClip]]:
    """Generates TTS audio and subtitle clips."""
    actual_tts_path = None
    tts_path = TEMP_DIR / f"{submission_id}_{safe_title}_tts.mp3"
    temp_files_list.append(tts_path)
    if generate_tts_flag:
        # Generate TTS using Hugging Face
        if hugging_face_tts(tts_text, tts_path):
            actual_tts_path = tts_path
            print("  Hugging Face TTS generation successful")
        else:
            print("Warning: Hugging Face TTS generation failed")

    subtitle_clips = []
    speech_segments = gemini_analysis.get("speech_segments", [])
    if isinstance(speech_segments, list) and speech_segments:
        adjusted_segments = []
        trim_duration = end_time - start_time
        for seg in speech_segments:
            try:
                seg_start = float(seg['start_time']); seg_end = float(seg['end_time'])
                if max(start_time, seg_start) < min(end_time, seg_end):
                    new_start = max(0, seg_start - start_time)
                    new_end = min(trim_duration, seg_end - start_time)
                    if new_end > new_start + 0.1:
                         adjusted_segments.append({"start_time": new_start, "end_time": new_end, "text": seg.get("text", "")})
            except (ValueError, TypeError, KeyError): continue

        if adjusted_segments:
            _, trim_w, trim_h = get_video_details(trimmed_video_path)
            if trim_w > 0 and trim_h > 0:
                subtitle_clips = create_animated_subtitle_clips(adjusted_segments, trim_w, trim_h)
            else: print("Warning: Could not get dimensions of trimmed video for subtitles.")

    return actual_tts_path, subtitle_clips


def process_submission(submission: praw.models.Submission, config: Dict[str, Any]) -> bool:
    if is_already_uploaded(submission.permalink):
        print(f"Skipping {submission.id}: Already uploaded.")
        return True
    if contains_forbidden_words(submission.title) or contains_forbidden_words(submission.selftext):
        print(f"Skipping {submission.id}: Forbidden words.")
        return True

    print(f"\n{'='*20} Processing: {submission.id} | {submission.title[:50]}... {'='*20}")
    safe_title = sanitize_filename(submission.title)
    submission_temp_files = []

    # Paths used across multiple steps
    final_output_path = TEMP_DIR / f"{submission.id}_{safe_title}_final.mp4"
    trimmed_path = TEMP_DIR / f"{submission.id}_{safe_title}_trimmed.mp4"
    stab_analysis_path = TEMP_DIR / f"{submission.id}_{safe_title}_stab.trf"
    submission_temp_files.extend([final_output_path, trimmed_path, stab_analysis_path])

    try:
        # 1. Prepare Initial Video (Download, Details, Vertical)
        initial_video_path, duration, width, height = _prepare_initial_video(submission, safe_title, submission_temp_files)
        if not initial_video_path:
            print("Failed initial video preparation.")
            return False
        print(f"Initial video prepared: {initial_video_path.name}, {duration:.2f}s, {width}x{height}")

        # 2. Analyze Content (Gemini)
        gemini_analysis = analyze_video_with_gemini(initial_video_path, submission.title, submission.subreddit.display_name)
        if gemini_analysis.get('fallback', True):
            print("Gemini analysis failed or returned fallback. Aborting.")
            return False # Stop if core analysis fails
        print("Gemini analysis successful.")

        # 3. Determine Processing Parameters
        start_time, end_time, tts_text, generate_tts_flag = _get_processing_params(gemini_analysis, duration)
        print(f"Processing segment: {start_time:.2f}s - {end_time:.2f}s. TTS needed: {generate_tts_flag}")

        # 4. Trim Video
        trimmed_video_path = trim_video(initial_video_path, start_time, end_time, trimmed_path)
        if not trimmed_video_path:
            print("Failed to trim video.")
            return False
        current_video_path = trimmed_video_path
        print(f"Video trimmed: {trimmed_video_path.name}")

        # 5. Generate Assets (TTS, Subtitles)
        actual_tts_path, subtitle_clips = _generate_assets(
            gemini_analysis, tts_text, generate_tts_flag, start_time, end_time,
            trimmed_video_path, submission.id, safe_title, submission_temp_files
        )
        print(f"Assets generated. TTS: {'Yes' if actual_tts_path else 'No'}, Subtitles: {len(subtitle_clips)}")

        # 6. Stabilization Analysis (Optional)
        stabilization_file = None
        if config.get('apply_stabilization', False) and check_ffmpeg_install("ffmpeg"):
             print("--- Applying Stabilization (Experimental) ---")
             stab_analyze_cmd = ['ffmpeg', '-y', '-i', str(current_video_path), '-vf', f'vidstabdetect=result="{str(stab_analysis_path)}":shakiness=5:accuracy=15', '-f', 'null', '-']
             try:
                 subprocess.run(stab_analyze_cmd, check=True, capture_output=True, text=True, timeout=120)
                 if stab_analysis_path.is_file(): stabilization_file = stab_analysis_path
                 else: print("  Warning: Stabilization analysis file not created.")
             except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception) as e: print(f"  Error during stabilization analysis: {e}")

        # 7. Combine Final Video
        original_audio_ref = current_video_path if config.get('mix_original_audio', False) else None
        final_video_path = combine_video_audio_subs(
            processed_video_path=current_video_path,
            original_video_for_audio_path=original_audio_ref,
            tts_audio_path=actual_tts_path,
            subtitle_clips=subtitle_clips,
            output_path=final_output_path,
            gemini_data=gemini_analysis,
            stabilization_transforms_file=stabilization_file
        )
        if not final_video_path:
            print("Failed to combine final video.")
            return False
        print(f"Final video combined: {final_video_path.name}")

        # 8. Upload (if not skipped)
        if not config.get('skip_upload', False):
            youtube_title = gemini_analysis.get("suggested_title", submission.title[:70])[:100]
            description_summary = gemini_analysis.get("summary_for_description", submission.title)
            youtube_description = f"{description_summary}\n\nOriginal: https://www.reddit.com{submission.permalink}\nr/{submission.subreddit.display_name}"
            youtube_tags = list(set(tag.strip().replace("#", "") for tag in gemini_analysis.get("hashtags", []) if tag.strip()))
            youtube_tags.extend([submission.subreddit.display_name, "reddit", "shorts"])
            youtube_tags = list(dict.fromkeys(youtube_tags))[:15]

            youtube_url = upload_to_youtube(final_video_path, youtube_title, youtube_description, youtube_tags)
            if not youtube_url:
                print("Failed to upload video to YouTube.")
                # Don't return False here if the error was UploadLimitExceededError, as it's handled by the caller
                # For other upload errors, we might consider it a failure for this submission.
                # Check if exception was raised and handled by caller instead. For now, assume non-quota error is failure.
                # The exception handling in the main loop should catch UploadLimitExceededError.
                # If upload_to_youtube returns None without raising that specific error, it's likely another issue.
                return False # Treat other upload failures as processing failure for this item

            add_upload_record(submission.permalink, youtube_url)
            print(f"Successfully uploaded: {youtube_url}")
        else:
            print("--- Skipping YouTube Upload as requested ---")


        print(f"--- Successfully processed submission {submission.id} ---")
        return True

    except UploadLimitExceededError:
        print("YouTube upload limit reached during processing.")
        raise # Re-raise for the main loop to catch
    except Exception as e:
        print(f"\n--- ERROR processing submission {submission.id}: {e} ---")
        traceback.print_exc()
        return False
    finally:
        cleanup_temp_files(*submission_temp_files)
        # Ensure subtitle clips are closed if they exist and weren't handled by combine
        if 'subtitle_clips' in locals() and subtitle_clips:
             for sub_clip in subtitle_clips:
                 if hasattr(sub_clip, 'close'):
                     try: sub_clip.close()
                     except Exception: pass


def main():
    parser = argparse.ArgumentParser(description="Reddit Video Bot")
    parser.add_argument("subreddits", nargs='+', help="Subreddit names (e.g., 'videos funny cats').")
    parser.add_argument("-n", "--num_posts", type=int, default=5, help="Max posts to process.")
    parser.add_argument("--skip_upload", action="store_true", help="Process but do not upload.")
    args = parser.parse_args()

    start_time = time.time()
    processed_count = 0
    success_count = 0
    quota_error_encountered = False
    interrupted = False

    # Simplified config dict for passing options
    config = {
        'skip_upload': args.skip_upload,
        'mix_original_audio': MIX_ORIGINAL_AUDIO,
        'apply_stabilization': APPLY_STABILIZATION
        # Add other relevant global constants if needed by process_submission indirectly
    }

    try:
        validate_environment()
        setup_directories()
        setup_database()
        setup_api_clients()

        for subreddit_name in args.subreddits:
            if interrupted or quota_error_encountered: break
            print(f"\n===== Processing Subreddit: r/{subreddit_name} =====")
            submissions = get_reddit_submissions(subreddit_name, args.num_posts)
            if not submissions: continue

            for submission in submissions:
                if interrupted or quota_error_encountered: break
                if processed_count >= args.num_posts: break

                processed_count += 1
                try:
                    success = process_submission(submission, config)
                    if success: success_count += 1
                    time.sleep(2) # Small delay between posts
                except KeyboardInterrupt:
                    print("\n*** KeyboardInterrupt detected. Stopping... ***")
                    interrupted = True; break
                except UploadLimitExceededError:
                    print("\nStopping processing due to YouTube Upload Quota Error.")
                    quota_error_encountered = True; break
                except Exception as e: # Catch unexpected errors from process_submission itself
                    print(f"\n--- Unhandled Error processing submission {submission.id} from main loop: {e} ---")
                    traceback.print_exc()

    except (ValueError, FileNotFoundError, ConnectionError, sqlite3.Error, RuntimeError) as setup_err:
        print(f"\nFATAL SETUP ERROR: {setup_err}")
    except KeyboardInterrupt: # Catch interrupt during setup or subreddit iteration
         print("\n*** KeyboardInterrupt detected during main execution. Stopping... ***")
         interrupted = True
    except Exception as e:
        print(f"\n--- An unexpected error occurred in the main execution: {e} ---")
        traceback.print_exc()
    finally:
        close_database()
        end_time = time.time()
        print("\n--- Processing Summary ---")
        if interrupted: print("*** Processing Interrupted ***")
        if quota_error_encountered: print("*** Stopped due to YouTube Quota ***")
        print(f"Attempted: {processed_count} submissions")
        print(f"Succeeded: {success_count} submissions")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print("--------------------------")
        # Optional: Clean up TEMP_DIR more aggressively if needed
        # if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()

