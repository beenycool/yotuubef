#!/usr/bin/env python3
"""
YouTube OAuth2 Authentication Script for Google Colab
Creates youtube_token.json from environment variables or interactive flow
"""

import json
import os
import sys
import logging
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YouTube API scopes
SCOPES = [
    'https://www.googleapis.com/auth/youtube.upload',
    'https://www.googleapis.com/auth/youtube.force-ssl',
    'https://www.googleapis.com/auth/youtubepartner'
]

def create_credentials_from_env():
    """Create YouTube credentials from environment variables"""
    try:
        # Check if we have token data in YOUTUBE_TOKEN_JSON
        token_json = os.environ.get('YOUTUBE_TOKEN_JSON')
        if token_json:
            logger.info("Found YOUTUBE_TOKEN_JSON in environment")
            token_data = json.loads(token_json)
            
            # Create credentials object
            creds = Credentials(
                token=token_data.get('token'),
                refresh_token=token_data.get('refresh_token'),
                token_uri=token_data.get('token_uri', 'https://oauth2.googleapis.com/token'),
                client_id=token_data.get('client_id'),
                client_secret=token_data.get('client_secret'),
                scopes=token_data.get('scopes', SCOPES)
            )
            
            # Refresh if expired
            if creds.expired and creds.refresh_token:
                logger.info("Token expired, refreshing...")
                creds.refresh(Request())
                logger.info("Token refreshed successfully")
            
            return creds
            
    except Exception as e:
        logger.error(f"Error creating credentials from environment: {e}")
    
    return None

def create_credentials_interactive():
    """Create credentials using interactive OAuth flow"""
    try:
        # Look for client secrets file
        client_secrets_file = os.environ.get('GOOGLE_CLIENT_SECRETS_FILE')
        if not client_secrets_file or not os.path.exists(client_secrets_file):
            logger.error("Google client secrets file not found")
            return None
        
        logger.info(f"Using client secrets file: {client_secrets_file}")
        
        # Run OAuth flow
        flow = InstalledAppFlow.from_client_secrets_file(
            client_secrets_file, SCOPES
        )
        
        # For Colab, use run_local_server with port 0 for auto-assignment
        creds = flow.run_local_server(port=0, open_browser=False)
        
        return creds
        
    except Exception as e:
        logger.error(f"Error in interactive OAuth flow: {e}")
        return None

def save_token_file(creds, token_file_path='youtube_token.json'):
    """Save credentials to token file"""
    try:
        token_data = {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'token_uri': creds.token_uri,
            'client_id': creds.client_id,
            'client_secret': creds.client_secret,
            'scopes': creds.scopes
        }
        
        if creds.expiry:
            token_data['expiry'] = creds.expiry.isoformat()
        
        # Ensure directory exists
        token_path = Path(token_file_path)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(token_path, 'w') as f:
            json.dump(token_data, f, indent=2)
        
        logger.info(f"Token saved to: {token_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving token file: {e}")
        return False

def main():
    """Main authentication function"""
    logger.info("Starting YouTube authentication...")
    
    # Try to get credentials from environment first
    creds = create_credentials_from_env()
    
    if not creds:
        logger.info("Environment credentials not available, trying interactive flow...")
        creds = create_credentials_interactive()
    
    if not creds:
        logger.error("Failed to create YouTube credentials")
        return False
    
    # Determine token file path
    token_file = os.environ.get('YOUTUBE_TOKEN_FILE', 'youtube_token.json')
    
    # Save token file
    if save_token_file(creds, token_file):
        logger.info("YouTube authentication completed successfully!")
        
        # Set environment variable for the current session
        os.environ['YOUTUBE_TOKEN_FILE'] = token_file
        
        return True
    else:
        logger.error("Failed to save token file")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)