"""
Instagram API Client for video uploads and content management.
Provides integration with Instagram's API for automated video posting.
"""

import asyncio
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

import aiohttp
import requests
from pydantic import BaseModel, Field

# Try to import Instagram API libraries with fallbacks
try:
    from instagrapi import Client as InstagrapiClient
    INSTAGRAPI_AVAILABLE = True
except ImportError:
    INSTAGRAPI_AVAILABLE = False

try:
    from instagram_private_api import Client as InstagramPrivateClient
    INSTAGRAM_PRIVATE_API_AVAILABLE = True
except ImportError:
    INSTAGRAM_PRIVATE_API_AVAILABLE = False


@dataclass
class InstagramUploadResult:
    """Result of an Instagram video upload operation."""
    success: bool
    media_id: Optional[str] = None
    share_url: Optional[str] = None
    error_message: Optional[str] = None
    upload_time: Optional[datetime] = None
    processing_status: Optional[str] = None
    media_type: Optional[str] = None  # 'reel', 'post', 'story'


class InstagramVideoMetadata(BaseModel):
    """Metadata for Instagram video uploads."""
    title: str = Field(..., description="Video title/caption")
    description: Optional[str] = Field(None, description="Video description")
    tags: List[str] = Field(default_factory=list, description="Hashtags for the video")
    location: Optional[str] = Field(None, description="Location tag")
    media_type: str = Field("reel", description="Media type: reel, post, story")
    visibility: str = Field("public", description="Post visibility")
    allow_comments: bool = Field(True, description="Allow comments")
    allow_likes: bool = Field(True, description="Allow likes")
    music_info: Optional[Dict[str, Any]] = Field(None, description="Music information")
    cover_image: Optional[str] = Field(None, description="Custom cover image path")


class InstagramClient:
    """
    Instagram API client for video uploads and content management.
    
    Supports multiple authentication methods and provides fallback mechanisms
    for different API libraries.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # API configuration
        self.username = config.get('instagram_username')
        self.password = config.get('instagram_password')
        self.session_id = config.get('instagram_session_id')
        self.access_token = config.get('instagram_access_token')
        self.client_id = config.get('instagram_client_id')
        self.client_secret = config.get('instagram_client_secret')
        
        # Rate limiting
        self.rate_limit_requests = config.get('instagram_rate_limit_requests', 50)
        self.rate_limit_window = config.get('instagram_rate_limit_window', 3600)  # 1 hour
        self.request_timestamps = []
        
        # API endpoints
        self.base_url = "https://graph.instagram.com/v18.0"
        self.upload_url = f"{self.base_url}/me/media"
        self.status_url = f"{self.base_url}/me/media"
        
        # Initialize API clients
        self._initialize_api_clients()
        
        self.logger.info("Instagram client initialized")
    
    def _initialize_api_clients(self):
        """Initialize available Instagram API clients."""
        self.api_clients = []
        
        if INSTAGRAPI_AVAILABLE:
            try:
                client = InstagrapiClient()
                if self.username and self.password:
                    client.login(self.username, self.password)
                elif self.session_id:
                    client.login_by_sessionid(self.session_id)
                
                self.api_clients.append(('instagrapi', client))
                self.logger.info("Instagrapi client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Instagrapi client: {e}")
        
        if INSTAGRAM_PRIVATE_API_AVAILABLE:
            try:
                client = InstagramPrivateClient(
                    self.username,
                    self.password,
                    auto_patch=True,
                    drop_incompat_keys=False
                )
                self.api_clients.append(('instagram_private_api', client))
                self.logger.info("Instagram Private API client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Instagram Private API client: {e}")
        
        if not self.api_clients:
            self.logger.warning("No Instagram API clients available - using Graph API fallback")
    
    async def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()
        window_start = now - self.rate_limit_window
        
        # Remove old timestamps
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > window_start]
        
        if len(self.request_timestamps) >= self.rate_limit_requests:
            return False
        
        self.request_timestamps.append(now)
        return True
    
    async def _make_api_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                               headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Make an API request with proper error handling."""
        if not await self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        default_headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        if headers:
            default_headers.update(headers)
        
        async with aiohttp.ClientSession() as session:
            try:
                if method.upper() == 'GET':
                    async with session.get(endpoint, headers=default_headers) as response:
                        return await response.json()
                elif method.upper() == 'POST':
                    async with session.post(endpoint, headers=default_headers, json=data) as response:
                        return await response.json()
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
            except Exception as e:
                self.logger.error(f"API request failed: {e}")
                raise
    
    async def upload_video(self, video_path: Union[str, Path], 
                          metadata: InstagramVideoMetadata) -> InstagramUploadResult:
        """
        Upload a video to Instagram.
        
        Args:
            video_path: Path to the video file
            metadata: Video metadata including title, description, tags, etc.
        
        Returns:
            InstagramUploadResult with upload status and details
        """
        try:
            self.logger.info(f"Starting Instagram video upload: {video_path}")
            
            # Validate video file
            video_path = Path(video_path)
            if not video_path.exists():
                return InstagramUploadResult(
                    success=False,
                    error_message=f"Video file not found: {video_path}"
                )
            
            # Check file size (Instagram has limits)
            file_size = video_path.stat().st_size
            max_size = 100 * 1024 * 1024  # 100MB limit for most content
            if file_size > max_size:
                return InstagramUploadResult(
                    success=False,
                    error_message=f"Video file too large: {file_size / (1024*1024):.1f}MB > 100MB"
                )
            
            # Try using available API clients first
            for client_name, client in self.api_clients:
                try:
                    result = await self._upload_with_client(client, client_name, video_path, metadata)
                    if result.success:
                        return result
                except Exception as e:
                    self.logger.warning(f"Upload failed with {client_name}: {e}")
                    continue
            
            # Fallback to Graph API
            return await self._upload_with_graph_api(video_path, metadata)
            
        except Exception as e:
            self.logger.error(f"Instagram upload failed: {e}")
            return InstagramUploadResult(
                success=False,
                error_message=str(e)
            )
    
    async def _upload_with_client(self, client: Any, client_name: str, 
                                 video_path: Path, metadata: InstagramVideoMetadata) -> InstagramUploadResult:
        """Upload using a specific API client."""
        try:
            if client_name == 'instagrapi':
                # Instagrapi library (synchronous) -> run in thread
                if metadata.media_type == 'reel':
                    result = await asyncio.to_thread(
                        client.clip_upload,
                        str(video_path),
                        metadata.title,
                        extra_data={
                            'custom_accessibility_caption': metadata.description or "",
                            'like_and_view_counts_disabled': not metadata.allow_likes,
                            'commenting_enabled': metadata.allow_comments
                        }
                    )
                else:
                    result = await asyncio.to_thread(
                        client.video_upload,
                        str(video_path),
                        metadata.title,
                        extra_data={
                            'custom_accessibility_caption': metadata.description or "",
                            'like_and_view_counts_disabled': not metadata.allow_likes,
                            'commenting_enabled': metadata.allow_comments
                        }
                    )
                
                return InstagramUploadResult(
                    success=True,
                    media_id=str(result.id),
                    share_url=f"https://www.instagram.com/p/{result.code}/",
                    upload_time=datetime.now(),
                    processing_status='uploaded',
                    media_type=metadata.media_type
                )
                
            elif client_name == 'instagram_private_api':
                # Instagram Private API library (synchronous) -> run in thread
                if metadata.media_type == 'reel':
                    result = await asyncio.to_thread(
                        client.clip_upload,
                        str(video_path),
                        metadata.title,
                        extra_data={
                            'custom_accessibility_caption': metadata.description or "",
                            'like_and_view_counts_disabled': not metadata.allow_likes,
                            'commenting_enabled': metadata.allow_comments
                        }
                    )
                else:
                    result = await asyncio.to_thread(
                        client.post_video,
                        str(video_path),
                        metadata.title,
                        extra_data={
                            'custom_accessibility_caption': metadata.description or "",
                            'like_and_view_counts_disabled': not metadata.allow_likes,
                            'commenting_enabled': metadata.allow_comments
                        }
                    )
                
                return InstagramUploadResult(
                    success=True,
                    media_id=str(result.get('media', {}).get('id')),
                    share_url=f"https://www.instagram.com/p/{result.get('media', {}).get('code')}/",
                    upload_time=datetime.now(),
                    processing_status='uploaded',
                    media_type=metadata.media_type
                )
            
        except Exception as e:
            self.logger.error(f"Client upload failed: {e}")
            raise
    
    async def _upload_with_graph_api(self, video_path: Path, 
                                    metadata: InstagramVideoMetadata) -> InstagramUploadResult:
        """Upload using Instagram Graph API as fallback."""
        try:
            # First, create a media container
            container_data = {
                'media_type': 'REELS' if metadata.media_type == 'reel' else 'VIDEO',
                'video_url': str(video_path),  # This would need to be a public URL
                'caption': metadata.title,
                'access_token': self.access_token
            }
            
            # Create media container
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.upload_url,
                    data=container_data
                ) as response:
                    result = await response.json()
                    
                    if response.status == 200 and 'id' in result:
                        media_id = result['id']
                        
                        # Publish the media
                        publish_data = {
                            'creation_id': media_id,
                            'access_token': self.access_token
                        }
                        
                        async with session.post(
                            f"{self.base_url}/me/media_publish",
                            data=publish_data
                        ) as publish_response:
                            publish_result = await publish_response.json()
                            
                            if publish_response.status == 200 and 'id' in publish_result:
                                return InstagramUploadResult(
                                    success=True,
                                    media_id=publish_result['id'],
                                    share_url=f"https://www.instagram.com/p/{publish_result.get('permalink', '')}/",
                                    upload_time=datetime.now(),
                                    processing_status='uploaded',
                                    media_type=metadata.media_type
                                )
                            else:
                                return InstagramUploadResult(
                                    success=False,
                                    error_message=publish_result.get('error', {}).get('message', 'Publish failed')
                                )
                    else:
                        return InstagramUploadResult(
                            success=False,
                            error_message=result.get('error', {}).get('message', 'Container creation failed')
                        )
                            
        except Exception as e:
            self.logger.error(f"Graph API upload failed: {e}")
            raise
    
    async def get_upload_status(self, media_id: str) -> Dict[str, Any]:
        """Get the status of a media upload."""
        try:
            endpoint = f"{self.status_url}/{media_id}"
            result = await self._make_api_request('GET', endpoint)
            return result
        except Exception as e:
            self.logger.error(f"Failed to get upload status: {e}")
            return {'error': str(e)}
    
    async def delete_media(self, media_id: str) -> bool:
        """Delete media from Instagram."""
        try:
            endpoint = f"{self.base_url}/{media_id}"
            result = await self._make_api_request('DELETE', endpoint)
            return result.get('success', False)
        except Exception as e:
            self.logger.error(f"Failed to delete media: {e}")
            return False
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get Instagram account information."""
        try:
            endpoint = f"{self.base_url}/me"
            result = await self._make_api_request('GET', endpoint)
            return result
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            return {'error': str(e)}
    
    async def get_media_list(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Get list of recent media."""
        try:
            endpoint = f"{self.status_url}?limit={limit}"
            result = await self._make_api_request('GET', endpoint)
            return result.get('data', [])
        except Exception as e:
            self.logger.error(f"Failed to get media list: {e}")
            return []
    
    async def refresh_access_token(self) -> bool:
        """Refresh the access token."""
        try:
            if not self.access_token:
                self.logger.warning("No access token available")
                return False
            
            endpoint = f"{self.base_url}/refresh_access_token"
            data = {
                'grant_type': 'ig_refresh_token',
                'access_token': self.access_token
            }
            
            result = await self._make_api_request('POST', endpoint, data)
            
            if 'access_token' in result:
                self.access_token = result['access_token']
                self.logger.info("Access token refreshed successfully")
                return True
            else:
                self.logger.error("Failed to refresh access token")
                return False
                
        except Exception as e:
            self.logger.error(f"Token refresh failed: {e}")
            return False
    
    def get_upload_limits(self) -> Dict[str, Any]:
        """Get Instagram upload limits and restrictions."""
        return {
            'max_file_size_mb': 100,
            'max_duration_seconds': 90,  # Reels limit
            'supported_formats': ['mp4', 'mov'],
            'max_caption_length': 2200,
            'max_tags': 30,
            'reel_aspect_ratio': '9:16',
            'post_aspect_ratio': '1.91:1 to 4:5'
        }


def create_instagram_client(config: Dict[str, Any]) -> InstagramClient:
    """Factory function to create an Instagram client."""
    return InstagramClient(config)