# API Credentials Setup for Colab Notebook
import os
import json
from pathlib import Path

# Create temp directory for credentials
TEMP_DIR = Path('/content/temp_processing')
TEMP_DIR.mkdir(exist_ok=True)

# Get Reddit credentials from https://www.reddit.com/prefs/apps
REDDIT_CLIENT_ID = "your_reddit_client_id_here"  # Replace with your Reddit client ID
REDDIT_CLIENT_SECRET = "your_reddit_client_secret_here"  # Replace with your Reddit client secret
REDDIT_USER_AGENT = "python:RedditVideoProcessor:v1.0" 

# Get Gemini API key from https://aistudio.google.com/app/apikey
GEMINI_API_KEY = "your_gemini_api_key_here"  # Replace with your Gemini API key

# Get ElevenLabs API key from https://elevenlabs.io/app/api-key
ELEVENLABS_API_KEY = "your_elevenlabs_api_key_here"  # Replace with your ElevenLabs API key

# YouTube credentials (already set)
YOUTUBE_CLIENT_SECRET_JSON = {
    "installed": {
        "client_id": "7181003830-d8avb8jo0p235bl37a2v5tkiv2d3j172.apps.googleusercontent.com",
        "project_id": "gen-lang-client-0383841226",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": "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",
        "redirect_uris": ["http://localhost"]
    }
}

# Save YouTube credentials to file and set environment variables
youtube_creds_path = TEMP_DIR / "client_secret.json"
with open(youtube_creds_path, 'w') as f:
    json.dump(YOUTUBE_CLIENT_SECRET_JSON, f)

os.environ['REDDIT_CLIENT_ID'] = REDDIT_CLIENT_ID
os.environ['REDDIT_CLIENT_SECRET'] = REDDIT_CLIENT_SECRET
os.environ['REDDIT_USER_AGENT'] = REDDIT_USER_AGENT
os.environ['GOOGLE_CLIENT_SECRETS_FILE'] = str(youtube_creds_path)
os.environ['GOOGLE_AI_API_KEY'] = GEMINI_API_KEY
os.environ['ELEVENLABS_API_KEY'] = ELEVENLABS_API_KEY

print("✅ API credentials set up. Remember to replace the placeholder values with your actual API keys.") 