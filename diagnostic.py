import os
import sys
import praw
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import subprocess
import traceback

def check_env_variables():
    print("=== Environment Variables Check ===")
    required_vars = [
        'REDDIT_CLIENT_ID', 
        'REDDIT_CLIENT_SECRET', 
        'GOOGLE_CLIENT_SECRETS_FILE', 
        'GEMINI_API_KEY', 
        'ELEVENLABS_API_KEY'
    ]
    
    all_set = True
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            # Show first 5 chars to validate it exists but don't expose full credentials
            display_val = value[:5] + "..." if len(value) > 5 else value
            print(f"✅ {var}: SET ({display_val})")
        else:
            print(f"❌ {var}: NOT SET")
            all_set = False
    
    return all_set

def check_ffmpeg():
    print("\n=== FFmpeg Check ===")
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is installed")
            # Show first line of version
            version = result.stdout.split('\n')[0]
            print(f"   {version}")
            return True
        else:
            print("❌ FFmpeg check failed with return code", result.returncode)
            return False
    except Exception as e:
        print(f"❌ FFmpeg check error: {e}")
        return False

def check_reddit():
    print("\n=== Reddit API Check ===")
    try:
        client_id = os.environ.get('REDDIT_CLIENT_ID')
        client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
        user_agent = os.environ.get('REDDIT_USER_AGENT', 'python:DiagnosticScript:v1.0')
        
        if not client_id or not client_secret:
            print("❌ Missing Reddit credentials")
            return False
        
        print("Trying to initialize Reddit...")
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            check_for_async=False
        )
        
        # Try to access Reddit API
        print("Testing Reddit connection...")
        me = reddit.user.me()  # This will be None for read-only/script type apps
        print(f"✅ Reddit connection successful (authenticated as: {me})")
        
        # Test subreddit access
        print("Testing subreddit access...")
        subreddit = reddit.subreddit("CrazyFuckingVideos")
        for post in subreddit.hot(limit=1):
            print(f"✅ Successfully retrieved post: {post.title[:40]}...")
            break
        
        return True
    except Exception as e:
        print(f"❌ Reddit API check failed: {e}")
        traceback.print_exc()
        return False

def check_gemini():
    print("\n=== Gemini API Check ===")
    try:
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            print("❌ Missing Gemini API key")
            return False
        
        print("Configuring Gemini...")
        genai.configure(api_key=api_key)
        
        print("Testing Gemini API with simple prompt...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say 'Gemini API is working properly' if you can read this.")
        
        print(f"✅ Gemini API response: {response.text}")
        return True
    except Exception as e:
        print(f"❌ Gemini API check failed: {e}")
        traceback.print_exc()
        return False

def check_elevenlabs():
    print("\n=== ElevenLabs API Check ===")
    try:
        api_key = os.environ.get('ELEVENLABS_API_KEY')
        if not api_key:
            print("❌ Missing ElevenLabs API key")
            return False
        
        print("Initializing ElevenLabs client...")
        client = ElevenLabs(api_key=api_key)
        
        print("Fetching available voices...")
        voices = client.voices.get_all()
        print(f"✅ ElevenLabs API working. Found {len(voices)} voices")
        return True
    except Exception as e:
        print(f"❌ ElevenLabs API check failed: {e}")
        traceback.print_exc()
        return False

def check_youtube():
    print("\n=== YouTube API Check ===")
    try:
        client_secrets_file = os.environ.get('GOOGLE_CLIENT_SECRETS_FILE')
        if not client_secrets_file or not os.path.exists(client_secrets_file):
            print(f"❌ Client secrets file not found: {client_secrets_file}")
            return False
        
        print("Starting YouTube authentication flow...")
        print("NOTE: This will open a browser window to authenticate with Google")
        scopes = ['https://www.googleapis.com/auth/youtube.upload', 'https://www.googleapis.com/auth/youtube.force-ssl']
        
        flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
        credentials = flow.run_local_server(port=8090)
        
        print("Building YouTube service...")
        youtube = build('youtube', 'v3', credentials=credentials)
        
        # Test the API with a simple call
        request = youtube.channels().list(part="snippet", mine=True)
        response = request.execute()
        
        if 'items' in response and len(response['items']) > 0:
            channel_name = response['items'][0]['snippet']['title']
            print(f"✅ YouTube API working. Connected to channel: {channel_name}")
            return True
        else:
            print("❌ YouTube API connection successful but no channel found")
            return False
    except Exception as e:
        print(f"❌ YouTube API check failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("=== YouTube Video Bot Diagnostic Tool ===")
    
    # Check environment variables first
    env_ok = check_env_variables()
    if not env_ok:
        print("\n❌ Some required environment variables are missing. Please set them before continuing.")
        return
        
    # Check FFmpeg installation
    ffmpeg_ok = check_ffmpeg()
    if not ffmpeg_ok:
        print("\n⚠️ FFmpeg issues detected. Video processing may fail.")
    
    # Ask which component to check
    print("\nWhich component would you like to check?")
    print("1. Reddit API")
    print("2. Gemini API")
    print("3. ElevenLabs API")
    print("4. YouTube API")
    print("5. Check All")
    print("6. Exit")
    
    choice = input("Enter your choice (1-6): ")
    
    if choice == '1':
        check_reddit()
    elif choice == '2':
        check_gemini()
    elif choice == '3':
        check_elevenlabs()
    elif choice == '4':
        check_youtube()
    elif choice == '5':
        reddit_ok = check_reddit()
        gemini_ok = check_gemini()
        elevenlabs_ok = check_elevenlabs()
        youtube_ok = check_youtube()
        
        print("\n=== Summary ===")
        print(f"Environment Variables: {'✅' if env_ok else '❌'}")
        print(f"FFmpeg: {'✅' if ffmpeg_ok else '❌'}")
        print(f"Reddit API: {'✅' if reddit_ok else '❌'}")
        print(f"Gemini API: {'✅' if gemini_ok else '❌'}")
        print(f"ElevenLabs API: {'✅' if elevenlabs_ok else '❌'}")
        print(f"YouTube API: {'✅' if youtube_ok else '❌'}")
        
        if env_ok and ffmpeg_ok and reddit_ok and gemini_ok and elevenlabs_ok and youtube_ok:
            print("\n✅ All components are working correctly!")
        else:
            print("\n⚠️ Some components have issues. Please fix them before running the main script.")
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()