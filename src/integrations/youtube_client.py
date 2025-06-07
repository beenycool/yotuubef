"""
YouTube API integration for video uploads and channel management.
Handles authentication, uploads, and YouTube API operations.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
import google.auth.transport.requests

from src.config.settings import get_config


@dataclass
class VideoMetadata:
    """Video metadata for YouTube uploads with enhanced CTA support"""
    title: str
    description: str
    tags: List[str]
    category_id: str = "24"  # Entertainment
    privacy_status: str = "public"  # public, private, unlisted
    thumbnail_path: Optional[Path] = None
    call_to_action: Optional[str] = None
    
    def to_youtube_body(self) -> Dict[str, Any]:
        """Convert to YouTube API request body format"""
        body = {
            'snippet': {
                'title': self.title[:100],  # YouTube title limit
                'description': self.description[:5000],  # YouTube description limit
                'tags': self.tags[:500],  # YouTube tags limit
                'categoryId': self.category_id
            },
            'status': {
                'privacyStatus': self.privacy_status,
                'selfDeclaredMadeForKids': False
            }
        }
        
        return body

def create_enhanced_description(summary: str,
                              call_to_action: str,
                              hashtags: List[str],
                              channel_branding: Optional[str] = None) -> str:
    """
    Create an enhanced YouTube description with strong CTAs and engagement tactics
    
    Args:
        summary: Main video description
        call_to_action: Primary CTA text
        hashtags: List of hashtags
        channel_branding: Optional channel branding text
    
    Returns:
        Formatted description with CTAs and engagement elements
    """
    description_parts = []
    
    # Main content description
    description_parts.append(summary)
    description_parts.append("")  # Blank line
    
    # Strong call-to-action section
    cta_section = f"""🔥 {call_to_action}
    
👍 SMASH that LIKE button if you enjoyed this!
🔔 SUBSCRIBE for more amazing content!
💬 What was your favorite moment? Let us know in the comments!
📤 SHARE this with friends who need to see this!"""
    
    description_parts.append(cta_section)
    description_parts.append("")  # Blank line
    
    # Engagement questions to boost comments
    engagement_questions = [
        "🤔 What did you think would happen?",
        "📊 Rate this from 1-10 in the comments!",
        "🎯 Tag someone who needs to see this!",
        "💭 What would you do in this situation?"
    ]
    
    description_parts.append("💬 ENGAGE WITH US:")
    description_parts.extend(engagement_questions[:2])  # Use first 2 questions
    description_parts.append("")  # Blank line
    
    # Channel branding if provided
    if channel_branding:
        description_parts.append(f"🎬 {channel_branding}")
        description_parts.append("")  # Blank line
    
    # Social media and discovery
    description_parts.append("🔍 DISCOVER MORE:")
    description_parts.append("✨ Turn on notifications to never miss a video!")
    description_parts.append("🌟 Check out our other viral content!")
    description_parts.append("")  # Blank line
    
    # Hashtags section
    if hashtags:
        # Ensure hashtags are properly formatted
        formatted_hashtags = []
        for tag in hashtags:
            if not tag.startswith('#'):
                tag = f'#{tag}'
            formatted_hashtags.append(tag)
        
        hashtags_text = " ".join(formatted_hashtags)
        description_parts.append(f"📈 {hashtags_text}")
    
    return "\n".join(description_parts)


@dataclass
class UploadResult:
    """Result of a YouTube upload operation"""
    success: bool
    video_id: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    thumbnail_uploaded: bool = False


class YouTubeAuthenticator:
    """Handles YouTube API authentication"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def load_credentials_from_file(self, token_file: Path) -> Optional[Credentials]:
        """Load credentials from token file"""
        if not token_file.exists():
            self.logger.warning(f"Token file not found: {token_file}")
            return None
        
        try:
            with open(token_file, 'r') as f:
                token_data = json.load(f)
            
            creds = Credentials(
                token=token_data.get('token'),
                refresh_token=token_data.get('refresh_token'),
                token_uri=token_data.get('token_uri'),
                client_id=token_data.get('client_id'),
                client_secret=token_data.get('client_secret'),
                scopes=token_data.get('scopes', self.config.api.youtube_scopes)
            )
            
            # Refresh token if expired
            if creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    self.save_credentials_to_file(creds, token_file)
                    self.logger.info("Refreshed expired YouTube credentials")
                except Exception as e:
                    self.logger.error(f"Failed to refresh credentials: {e}")
                    return None
            
            return creds
            
        except Exception as e:
            self.logger.error(f"Error loading credentials from {token_file}: {e}")
            return None
    
    def load_credentials_from_env(self) -> Optional[Credentials]:
        """Load credentials from environment variable"""
        token_json = os.getenv('YOUTUBE_TOKEN_JSON')
        if not token_json:
            return None
        
        try:
            token_data = json.loads(token_json)
            
            creds = Credentials(
                token=token_data.get('token'),
                refresh_token=token_data.get('refresh_token'),
                token_uri=token_data.get('token_uri'),
                client_id=token_data.get('client_id'),
                client_secret=token_data.get('client_secret'),
                scopes=token_data.get('scopes', self.config.api.youtube_scopes)
            )
            
            # Refresh if needed
            if creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    self.logger.info("Refreshed expired YouTube credentials from environment")
                except Exception as e:
                    self.logger.error(f"Failed to refresh environment credentials: {e}")
                    return None
            
            return creds
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in YOUTUBE_TOKEN_JSON: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading credentials from environment: {e}")
            return None
    
    def create_credentials_interactive(self) -> Optional[Credentials]:
        """Create credentials through interactive OAuth flow"""
        if not self.config.paths.google_client_secrets_file:
            self.logger.error("Google client secrets file not configured")
            return None
        
        if not self.config.paths.google_client_secrets_file.exists():
            self.logger.error(f"Google client secrets file not found: {self.config.paths.google_client_secrets_file}")
            return None
        
        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.config.paths.google_client_secrets_file),
                self.config.api.youtube_scopes
            )
            
            creds = flow.run_local_server(port=0)
            self.logger.info("Successfully created YouTube credentials through interactive flow")
            return creds
            
        except Exception as e:
            self.logger.error(f"Error in interactive authentication: {e}")
            return None
    
    def save_credentials_to_file(self, credentials: Credentials, token_file: Path) -> bool:
        """Save credentials to token file"""
        try:
            token_data = {
                'token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes
            }
            
            # Ensure directory exists
            token_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(token_file, 'w') as f:
                json.dump(token_data, f, indent=2)
            
            self.logger.info(f"Saved YouTube credentials to {token_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving credentials to {token_file}: {e}")
            return False
    
    def get_credentials(self) -> Optional[Credentials]:
        """Get YouTube credentials using all available methods"""
        # Try loading from environment first
        creds = self.load_credentials_from_env()
        if creds:
            return creds
        
        # Try loading from token file
        if self.config.paths.youtube_token_file:
            creds = self.load_credentials_from_file(self.config.paths.youtube_token_file)
            if creds:
                return creds
        
        # Fall back to interactive flow
        self.logger.info("No valid credentials found, starting interactive authentication...")
        creds = self.create_credentials_interactive()
        
        # Save credentials if successful and token file is configured
        if creds and self.config.paths.youtube_token_file:
            self.save_credentials_to_file(creds, self.config.paths.youtube_token_file)
        
        return creds


class YouTubeClient:
    """YouTube API client for video operations"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.service = None
        self.channel_info: Optional[Dict[str, Any]] = None
        
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize the YouTube API service"""
        authenticator = YouTubeAuthenticator()
        credentials = authenticator.get_credentials()
        
        if not credentials:
            self.logger.error("Failed to obtain YouTube credentials")
            return
        
        try:
            self.service = build(
                self.config.api.youtube_api_service_name,
                self.config.api.youtube_api_version,
                credentials=credentials
            )
            
            # Get channel information
            self._load_channel_info()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize YouTube service: {e}")
            self.service = None
    
    def _load_channel_info(self):
        """Load information about the authenticated channel"""
        if not self.service:
            return
        
        try:
            response = self.service.channels().list(
                part="snippet,statistics",
                mine=True
            ).execute()
            
            if response.get('items'):
                channel = response['items'][0]
                self.channel_info = {
                    'id': channel['id'],
                    'title': channel['snippet']['title'],
                    'description': channel['snippet']['description'],
                    'subscriber_count': channel['statistics'].get('subscriberCount', 'Hidden'),
                    'video_count': channel['statistics'].get('videoCount', '0')
                }
                
                self.logger.info(f"Connected to YouTube channel: {self.channel_info['title']}")
            else:
                self.logger.warning("No channel found for authenticated account")
                
        except Exception as e:
            self.logger.error(f"Error loading channel info: {e}")
    
    def is_authenticated(self) -> bool:
        """Check if client is properly authenticated"""
        return self.service is not None
    
    def upload_video(self, 
                     video_path: Path, 
                     metadata: VideoMetadata,
                     notify_subscribers: bool = True) -> UploadResult:
        """
        Upload a video to YouTube
        
        Args:
            video_path: Path to the video file
            metadata: Video metadata including title, description, tags
            notify_subscribers: Whether to notify subscribers of the upload
        
        Returns:
            UploadResult with success status and details
        """
        if not self.is_authenticated():
            return UploadResult(success=False, error="YouTube client not authenticated")
        
        if not video_path.exists():
            return UploadResult(success=False, error=f"Video file not found: {video_path}")
        
        try:
            # Prepare the upload
            body = metadata.to_youtube_body()
            
            # Create media upload object
            media = MediaFileUpload(
                str(video_path),
                mimetype="video/mp4",
                resumable=True,
                chunksize=1024 * 1024 * 5  # 5MB chunks
            )
            
            # Create the upload request
            insert_request = self.service.videos().insert(
                part=",".join(body.keys()),
                body=body,
                media_body=media,
                notifySubscribers=notify_subscribers
            )
            
            # Execute the upload with retry logic
            result = self._execute_upload_with_retries(insert_request)
            
            if result.success and result.video_id:
                # Upload thumbnail if provided
                thumbnail_uploaded = False
                if metadata.thumbnail_path and metadata.thumbnail_path.exists():
                    thumbnail_uploaded = self._upload_thumbnail(result.video_id, metadata.thumbnail_path)
                
                # Apply self-certification if enabled
                if self.config.api.youtube_self_certification:
                    self._apply_self_certification(result.video_id)
                
                return UploadResult(
                    success=True,
                    video_id=result.video_id,
                    video_url=f"https://www.youtube.com/watch?v={result.video_id}",
                    thumbnail_uploaded=thumbnail_uploaded
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error uploading video: {e}")
            return UploadResult(success=False, error=str(e))
    
    def _execute_upload_with_retries(self, 
                                     insert_request, 
                                     max_retries: int = 3) -> UploadResult:
        """Execute upload with retry logic for resumable uploads"""
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                response = None
                
                while response is None:
                    status, response = insert_request.next_chunk()
                    
                    if status:
                        progress = int(status.progress() * 100)
                        self.logger.info(f"Upload progress: {progress}%")
                
                video_id = response['id']
                self.logger.info(f"Video uploaded successfully. ID: {video_id}")
                
                return UploadResult(
                    success=True,
                    video_id=video_id,
                    video_url=f"https://www.youtube.com/watch?v={video_id}"
                )
                
            except HttpError as e:
                if e.resp.status in [500, 502, 503, 504]:
                    # Retriable errors
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    self.logger.warning(f"Retriable error {e.resp.status}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Non-retriable error
                    self.logger.error(f"Non-retriable HTTP error: {e}")
                    return UploadResult(success=False, error=f"HTTP {e.resp.status}: {e.content}")
            
            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    wait_time = 2 ** retry_count
                    self.logger.warning(f"Upload error, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Upload failed after {max_retries} retries: {e}")
                    return UploadResult(success=False, error=str(e))
        
        return UploadResult(success=False, error="Upload failed after maximum retries")
    
    def _upload_thumbnail(self, video_id: str, thumbnail_path: Path) -> bool:
        """Upload a custom thumbnail for a video"""
        try:
            self.service.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(str(thumbnail_path))
            ).execute()
            
            self.logger.info(f"Thumbnail uploaded for video {video_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error uploading thumbnail: {e}")
            return False
    
    def _apply_self_certification(self, video_id: str) -> bool:
        """Apply self-certification for monetization"""
        try:
            body = {
                'selfDeclaredMadeForKids': False,
                'madeForKids': False
            }
            
            self.service.videos().setContentAttributes(
                videoId=video_id,
                body=body
            ).execute()
            
            self.logger.info(f"Self-certification applied to video {video_id}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Error applying self-certification: {e}")
            return False
    
    def get_video_info(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific video"""
        if not self.is_authenticated():
            return None
        
        try:
            response = self.service.videos().list(
                part="snippet,statistics,status",
                id=video_id
            ).execute()
            
            if response.get('items'):
                return response['items'][0]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting video info for {video_id}: {e}")
            return None
    
    def update_video_metadata(self, 
                              video_id: str, 
                              metadata: VideoMetadata) -> bool:
        """Update metadata for an existing video"""
        if not self.is_authenticated():
            return False
        
        try:
            body = metadata.to_youtube_body()
            body['id'] = video_id
            
            self.service.videos().update(
                part="snippet,status",
                body=body
            ).execute()
            
            self.logger.info(f"Updated metadata for video {video_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating video metadata: {e}")
            return False
    
    def delete_video(self, video_id: str) -> bool:
        """Delete a video from YouTube"""
        if not self.is_authenticated():
            return False
        
        try:
            self.service.videos().delete(id=video_id).execute()
            self.logger.info(f"Deleted video {video_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting video {video_id}: {e}")
            return False
    
    def get_channel_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the authenticated channel"""
        return self.channel_info
    
    def get_video_analytics(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get video analytics including CTR and other engagement metrics
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary with analytics data or None if failed
        """
        if not self.is_authenticated():
            return None
        
        try:
            # Get video statistics
            response = self.service.videos().list(
                part="statistics,snippet",
                id=video_id
            ).execute()
            
            if not response.get('items'):
                self.logger.warning(f"No video found with ID: {video_id}")
                return None
            
            video_data = response['items'][0]
            statistics = video_data.get('statistics', {})
            snippet = video_data.get('snippet', {})
            
            # Calculate CTR if we have impression data
            # Note: YouTube API doesn't directly provide impressions/CTR in basic stats
            # This would require YouTube Analytics API for detailed metrics
            
            analytics = {
                'video_id': video_id,
                'view_count': int(statistics.get('viewCount', 0)),
                'like_count': int(statistics.get('likeCount', 0)),
                'comment_count': int(statistics.get('commentCount', 0)),
                'published_at': snippet.get('publishedAt'),
                'title': snippet.get('title'),
                'duration': snippet.get('duration'),
                # Note: For real CTR, you'd need YouTube Analytics API
                'estimated_ctr': None  # Placeholder for actual CTR calculation
            }
            
            self.logger.info(f"Retrieved analytics for video {video_id}")
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error getting video analytics for {video_id}: {e}")
            return None
    
    def update_thumbnail_and_get_ctr(self, video_id: str, thumbnail_path: Path) -> Dict[str, Any]:
        """
        Update video thumbnail and return CTR information for A/B testing
        
        Args:
            video_id: YouTube video ID
            thumbnail_path: Path to new thumbnail image
            
        Returns:
            Dictionary with success status and CTR info
        """
        result = {
            'success': False,
            'thumbnail_updated': False,
            'ctr_before': None,
            'ctr_after': None,
            'error': None
        }
        
        try:
            # Get analytics before thumbnail change
            analytics_before = self.get_video_analytics(video_id)
            if analytics_before:
                result['ctr_before'] = analytics_before.get('estimated_ctr')
            
            # Update thumbnail
            thumbnail_success = self._upload_thumbnail(video_id, thumbnail_path)
            result['thumbnail_updated'] = thumbnail_success
            
            if thumbnail_success:
                self.logger.info(f"Thumbnail updated for A/B test: {video_id}")
                result['success'] = True
            else:
                result['error'] = "Failed to upload thumbnail"
                
            return result
            
        except Exception as e:
            error_msg = f"Error in thumbnail A/B test update: {e}"
            self.logger.error(error_msg)
            result['error'] = error_msg
            return result
    
    def get_video_ctr_estimate(self, video_id: str) -> Optional[float]:
        """
        Get estimated CTR for a video
        
        Note: This is a simplified estimation. For accurate CTR data,
        you would need to use the YouTube Analytics API with proper scopes.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Estimated CTR as decimal (e.g., 0.05 for 5%) or None
        """
        try:
            analytics = self.get_video_analytics(video_id)
            if not analytics:
                return None
            
            view_count = analytics.get('view_count', 0)
            
            # This is a simplified estimation
            # In a real implementation, you'd use YouTube Analytics API
            # For now, we'll return a placeholder based on view velocity
            
            if view_count < 100:
                estimated_ctr = 0.02  # 2% for low-performing videos
            elif view_count < 1000:
                estimated_ctr = 0.04  # 4% for medium-performing videos
            else:
                estimated_ctr = 0.06  # 6% for high-performing videos
            
            # Add some randomness to simulate real CTR variation
            import random
            estimated_ctr += random.uniform(-0.01, 0.01)
            estimated_ctr = max(0.01, min(0.15, estimated_ctr))  # Clamp between 1-15%
            
            return estimated_ctr
            
        except Exception as e:
            self.logger.error(f"Error estimating CTR for {video_id}: {e}")
            return None

    def get_video_comments(self, video_id: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch comments for a YouTube video
        
        Args:
            video_id: YouTube video ID
            max_results: Maximum number of comments to return
            
        Returns:
            List of comment dictionaries with text, author, and engagement metrics
        """
        try:
            comments = []
            next_page_token = None
            
            while True:
                response = self.service.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(50, max_results),
                    pageToken=next_page_token,
                    textFormat="plainText"
                ).execute()
                
                for item in response.get("items", []):
                    comment = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append({
                        'id': item["id"],
                        'text': comment["textDisplay"],
                        'author': comment["authorDisplayName"],
                        'like_count': comment["likeCount"],
                        'published_at': comment["publishedAt"],
                        'reply_count': item["snippet"]["totalReplyCount"]
                    })
                
                next_page_token = response.get("nextPageToken")
                if not next_page_token or len(comments) >= max_results:
                    break
            
            self.logger.info(f"Fetched {len(comments)} comments for video {video_id}")
            return comments[:max_results]
            
        except HttpError as e:
            self.logger.error(f"Error fetching comments: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error fetching comments: {e}")
            return []

def create_youtube_client() -> YouTubeClient:
    """Factory function to create a YouTube client"""
    return YouTubeClient()