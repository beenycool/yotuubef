"""
TikTok API Client for video uploads and content management.
Provides integration with TikTok's API for automated video posting.
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

# Try to import TikTok API libraries with fallbacks
try:
    from tiktok_api import TikTokApi
    TIKTOK_API_AVAILABLE = True
except ImportError:
    TIKTOK_API_AVAILABLE = False

try:
    from python_tiktok_api import TikTokApi as PythonTikTokApi
    PYTHON_TIKTOK_API_AVAILABLE = True
except ImportError:
    PYTHON_TIKTOK_API_AVAILABLE = False


@dataclass
class TikTokUploadResult:
    """Result of a TikTok video upload operation."""
    success: bool
    video_id: Optional[str] = None
    share_url: Optional[str] = None
    error_message: Optional[str] = None
    upload_time: Optional[datetime] = None
    processing_status: Optional[str] = None


class TikTokVideoMetadata(BaseModel):
    """Metadata for TikTok video uploads."""
    title: str = Field(..., description="Video title/caption")
    description: Optional[str] = Field(None, description="Video description")
    tags: List[str] = Field(default_factory=list, description="Hashtags for the video")
    category: Optional[str] = Field(None, description="Video category")
    visibility: str = Field("public", description="Video visibility (public, private, friends)")
    allow_duet: bool = Field(True, description="Allow duet reactions")
    allow_comment: bool = Field(True, description="Allow comments")
    allow_stitch: bool = Field(True, description="Allow stitch reactions")
    music_info: Optional[Dict[str, Any]] = Field(None, description="Music information")


class TikTokClient:
    """
    TikTok API client for video uploads and content management.
    
    Supports multiple authentication methods and provides fallback mechanisms
    for different API libraries.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # API configuration
        self.api_key = config.get('tiktok_api_key')
        self.api_secret = config.get('tiktok_api_secret')
        self.access_token = config.get('tiktok_access_token')
        self.refresh_token = config.get('tiktok_refresh_token')
        self.client_key = config.get('tiktok_client_key')
        
        # Rate limiting
        self.rate_limit_requests = config.get('tiktok_rate_limit_requests', 100)
        self.rate_limit_window = config.get('tiktok_rate_limit_window', 3600)  # 1 hour
        self.request_timestamps = []
        
        # API endpoints
        self.base_url = "https://open.tiktokapis.com/v2"
        self.upload_url = f"{self.base_url}/video/upload/"
        self.status_url = f"{self.base_url}/video/query/"
        
        # Initialize API clients
        self._initialize_api_clients()
        
        self.logger.info("TikTok client initialized")
    
    def _initialize_api_clients(self):
        """Initialize available TikTok API clients."""
        self.api_clients = []
        
        if TIKTOK_API_AVAILABLE:
            try:
                api_client = TikTokApi(
                    api_key=self.api_key,
                    api_secret=self.api_secret
                )
                self.api_clients.append(('tiktok_api', api_client))
                self.logger.info("TikTok API client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize TikTok API client: {e}")
        
        if PYTHON_TIKTOK_API_AVAILABLE:
            try:
                api_client = PythonTikTokApi(
                    client_key=self.client_key,
                    access_token=self.access_token
                )
                self.api_clients.append(('python_tiktok_api', api_client))
                self.logger.info("Python TikTok API client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Python TikTok API client: {e}")
        
        if not self.api_clients:
            self.logger.warning("No TikTok API clients available - using REST API fallback")
    
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
                          metadata: TikTokVideoMetadata) -> TikTokUploadResult:
        """
        Upload a video to TikTok.
        
        Args:
            video_path: Path to the video file
            metadata: Video metadata including title, description, tags, etc.
        
        Returns:
            TikTokUploadResult with upload status and details
        """
        try:
            self.logger.info(f"Starting TikTok video upload: {video_path}")
            
            # Validate video file
            video_path = Path(video_path)
            if not video_path.exists():
                return TikTokUploadResult(
                    success=False,
                    error_message=f"Video file not found: {video_path}"
                )
            
            # Check file size (TikTok has limits)
            file_size = video_path.stat().st_size
            max_size = 287 * 1024 * 1024  # 287MB limit
            if file_size > max_size:
                return TikTokUploadResult(
                    success=False,
                    error_message=f"Video file too large: {file_size / (1024*1024):.1f}MB > 287MB"
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
            
            # Fallback to REST API
            return await self._upload_with_rest_api(video_path, metadata)
            
        except Exception as e:
            self.logger.error(f"TikTok upload failed: {e}")
            return TikTokUploadResult(
                success=False,
                error_message=str(e)
            )
    
    async def _upload_with_client(self, client: Any, client_name: str, 
                                 video_path: Path, metadata: TikTokVideoMetadata) -> TikTokUploadResult:
        """Upload using a specific API client."""
        try:
            if client_name == 'tiktok_api':
                # TikTok API library
                result = await client.upload_video(
                    video_file=str(video_path),
                    title=metadata.title,
                    description=metadata.description or "",
                    tags=metadata.tags,
                    category=metadata.category,
                    visibility=metadata.visibility
                )
                
                return TikTokUploadResult(
                    success=True,
                    video_id=result.get('video_id'),
                    share_url=result.get('share_url'),
                    upload_time=datetime.now(),
                    processing_status='uploaded'
                )
                
            elif client_name == 'python_tiktok_api':
                # Python TikTok API library
                result = await asyncio.to_thread(
                    client.upload_video,
                    video_path=str(video_path),
                    caption=metadata.title,
                    description=metadata.description or "",
                    hashtags=metadata.tags
                )
                
                return TikTokUploadResult(
                    success=True,
                    video_id=result.get('id'),
                    share_url=result.get('share_url'),
                    upload_time=datetime.now(),
                    processing_status='uploaded'
                )
            
        except Exception as e:
            self.logger.error(f"Client upload failed: {e}")
            raise
    
    async def _upload_with_rest_api(self, video_path: Path, 
                                   metadata: TikTokVideoMetadata) -> TikTokUploadResult:
        """Upload using REST API as fallback."""
        try:
            # Prepare upload data
            upload_data = {
                'title': metadata.title,
                'description': metadata.description or "",
                'tags': metadata.tags,
                'category': metadata.category,
                'visibility': metadata.visibility,
                'allow_duet': metadata.allow_duet,
                'allow_comment': metadata.allow_comment,
                'allow_stitch': metadata.allow_stitch
            }
            
            # Upload video file
            with open(video_path, 'rb') as video_file:
                files = {'video': video_file}
                data = {'metadata': json.dumps(upload_data)}
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.upload_url,
                        headers={'Authorization': f'Bearer {self.access_token}'},
                        data=data,
                        files=files
                    ) as response:
                        result = await response.json()
                        
                        if response.status == 200 and result.get('success'):
                            return TikTokUploadResult(
                                success=True,
                                video_id=result.get('data', {}).get('video_id'),
                                share_url=result.get('data', {}).get('share_url'),
                                upload_time=datetime.now(),
                                processing_status='uploaded'
                            )
                        else:
                            return TikTokUploadResult(
                                success=False,
                                error_message=result.get('message', 'Upload failed')
                            )
                            
        except Exception as e:
            self.logger.error(f"REST API upload failed: {e}")
            raise
    
    async def get_upload_status(self, video_id: str) -> Dict[str, Any]:
        """Get the status of a video upload."""
        try:
            endpoint = f"{self.status_url}?video_id={video_id}"
            result = await self._make_api_request('GET', endpoint)
            return result
        except Exception as e:
            self.logger.error(f"Failed to get upload status: {e}")
            return {'error': str(e)}
    
    async def delete_video(self, video_id: str) -> bool:
        """Delete a video from TikTok."""
        try:
            endpoint = f"{self.base_url}/video/delete/"
            data = {'video_id': video_id}
            result = await self._make_api_request('POST', endpoint, data)
            return result.get('success', False)
        except Exception as e:
            self.logger.error(f"Failed to delete video: {e}")
            return False
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get TikTok account information."""
        try:
            endpoint = f"{self.base_url}/user/info/"
            result = await self._make_api_request('GET', endpoint)
            return result
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            return {'error': str(e)}
    
    async def refresh_access_token(self) -> bool:
        """Refresh the access token."""
        try:
            if not self.refresh_token:
                self.logger.warning("No refresh token available")
                return False
            
            endpoint = f"{self.base_url}/oauth/refresh_token/"
            data = {
                'refresh_token': self.refresh_token,
                'client_key': self.client_key
            }
            
            result = await self._make_api_request('POST', endpoint, data)
            
            if result.get('success'):
                self.access_token = result['data']['access_token']
                self.refresh_token = result['data']['refresh_token']
                self.logger.info("Access token refreshed successfully")
                return True
            else:
                self.logger.error("Failed to refresh access token")
                return False
                
        except Exception as e:
            self.logger.error(f"Token refresh failed: {e}")
            return False
    
    def get_upload_limits(self) -> Dict[str, Any]:
        """Get TikTok upload limits and restrictions."""
        return {
            'max_file_size_mb': 287,
            'max_duration_seconds': 600,  # 10 minutes
            'supported_formats': ['mp4', 'mov', 'avi'],
            'max_title_length': 150,
            'max_description_length': 2200,
            'max_tags': 20
        }


def create_tiktok_client(config: Dict[str, Any]) -> TikTokClient:
    """Factory function to create a TikTok client."""
    return TikTokClient(config)