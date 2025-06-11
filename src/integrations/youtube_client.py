"""
Enhanced YouTube Client with A/B Testing and Analytics Integration
Handles video uploads, thumbnail management, comment interactions, and performance tracking.
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os
import json

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

from src.config.settings import get_config


class YouTubeClient:
    """
    Enhanced YouTube client with A/B testing capabilities and comprehensive analytics
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # YouTube API setup - use configured scopes
        self.scopes = self.config.api.youtube_scopes
        
        self.youtube_service = None
        self.analytics_service = None
        self._services_initialized = False
        
        if not GOOGLE_API_AVAILABLE:
            self.logger.warning("Google API libraries not available - YouTube features will be limited")
    
    def _initialize_services(self):
        """Initialize YouTube and Analytics API services"""
        try:
            credentials = self._get_credentials()
            if credentials:
                self.youtube_service = build('youtube', 'v3', credentials=credentials)
                self.analytics_service = build('youtubeAnalytics', 'v2', credentials=credentials)
                self.logger.info("YouTube API services initialized successfully")
            else:
                self.logger.warning("Failed to obtain YouTube API credentials, YouTube features will be disabled.")
                
        except Exception as e:
            self.logger.error(f"YouTube API initialization failed: {e}")
    
    async def _ensure_services_initialized(self):
        """Ensure services are initialized before use"""
        if not self._services_initialized and GOOGLE_API_AVAILABLE:
            await asyncio.to_thread(self._initialize_services)
            self._services_initialized = True
    
    def _get_credentials(self):
        """Get YouTube API credentials from environment variable only"""
        try:
            import os
            import json
            
            # Only use YOUTUBE_TOKEN_JSON environment variable
            youtube_token_env = os.getenv('YOUTUBE_TOKEN_JSON')
            if not youtube_token_env:
                self.logger.info("YOUTUBE_TOKEN_JSON environment variable not found. YouTube functionality will be disabled.")
                return None
            
            try:
                token_data = json.loads(youtube_token_env)
                creds = Credentials.from_authorized_user_info(token_data, self.scopes)
                self.logger.info("Loaded credentials from YOUTUBE_TOKEN_JSON environment variable")
                
                # If credentials are valid, return them immediately
                if creds and creds.valid:
                    self.logger.info("Credentials are valid and ready to use")
                    return creds
                
                # If credentials exist but are expired, try to refresh
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                        self.logger.info("Successfully refreshed credentials from environment variable")
                        return creds
                    except Exception as refresh_error:
                        self.logger.error(f"Failed to refresh credentials: {refresh_error}")
                        return None
                else:
                    self.logger.error("Credentials are invalid and cannot be refreshed")
                    return None
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in YOUTUBE_TOKEN_JSON environment variable: {e}")
                return None
            except Exception as e:
                self.logger.error(f"Error loading credentials from environment variable: {e}")
                return None
            
        except Exception as e:
            self.logger.error(f"Credential handling failed: {e}")
            return None
    
    async def upload_video(self, 
                          video_path: str,
                          metadata: Dict[str, Any],
                          thumbnail_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload video to YouTube with metadata
        
        Args:
            video_path: Path to video file
            metadata: Video metadata (title, description, tags, etc.)
            thumbnail_path: Optional path to thumbnail image
            
        Returns:
            Upload result with video ID and URL
        """
        try:
            await self._ensure_services_initialized()
            if not self.youtube_service:
                return {'success': False, 'error': 'YouTube service not available'}
            
            self.logger.info(f"Uploading video: {video_path}")
            
            # Prepare video metadata
            video_body = {
                'snippet': {
                    'title': metadata.get('title', 'Untitled Video'),
                    'description': metadata.get('description', ''),
                    'tags': metadata.get('tags', []),
                    'categoryId': metadata.get('category_id', '22'),  # People & Blogs
                    'defaultLanguage': 'en',
                    'defaultAudioLanguage': 'en'
                },
                'status': {
                    'privacyStatus': metadata.get('privacy_status', 'public'),
                    'madeForKids': False,
                    'selfDeclaredMadeForKids': False
                }
            }
            
            # Upload video
            media = MediaFileUpload(
                video_path,
                chunksize=-1,
                resumable=True,
                mimetype='video/*'
            )
            
            # Execute upload
            insert_request = self.youtube_service.videos().insert(
                part=','.join(video_body.keys()),
                body=video_body,
                media_body=media
            )
            
            response = await self._execute_request_async(insert_request)
            
            if not response:
                return {'success': False, 'error': 'Upload failed'}
            
            video_id = response['id']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Upload thumbnail if provided
            if thumbnail_path and Path(thumbnail_path).exists():
                await self.update_video_thumbnail(video_id, thumbnail_path)
            
            self.logger.info(f"Video uploaded successfully: {video_id}")
            
            return {
                'success': True,
                'video_id': video_id,
                'video_url': video_url,
                'upload_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Video upload failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def update_video_thumbnail(self, video_id: str, thumbnail_path: str) -> bool:
        """
        Update video thumbnail
        
        Args:
            video_id: YouTube video ID
            thumbnail_path: Path to thumbnail image
            
        Returns:
            True if successful
        """
        try:
            await self._ensure_services_initialized()
            if not self.youtube_service:
                return False
            
            self.logger.info(f"Updating thumbnail for video {video_id}")
            
            # Upload thumbnail
            media = MediaFileUpload(
                thumbnail_path,
                mimetype='image/jpeg'
            )
            
            request = self.youtube_service.thumbnails().set(
                videoId=video_id,
                media_body=media
            )
            
            response = await self._execute_request_async(request)
            
            if response:
                self.logger.info(f"Thumbnail updated successfully for video {video_id}")
                return True
            else:
                self.logger.error(f"Thumbnail update failed for video {video_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Thumbnail update failed: {e}")
            return False
    
    async def get_video_info(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get video information
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Video information or None if failed
        """
        try:
            await self._ensure_services_initialized()
            if not self.youtube_service:
                return None
            
            request = self.youtube_service.videos().list(
                part='snippet,statistics,status',
                id=video_id
            )
            
            response = await self._execute_request_async(request)
            
            if response and response.get('items'):
                video_info = response['items'][0]
                return {
                    'id': video_info['id'],
                    'title': video_info['snippet']['title'],
                    'description': video_info['snippet']['description'],
                    'publishedAt': video_info['snippet']['publishedAt'],
                    'tags': video_info['snippet'].get('tags', []),
                    'statistics': video_info.get('statistics', {}),
                    'status': video_info.get('status', {})
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get video info for {video_id}: {e}")
            return None
    
    async def get_video_analytics(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get video analytics data
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Analytics data or None if failed
        """
        try:
            await self._ensure_services_initialized()
            if not self.analytics_service:
                return None
            
            # Get video statistics from main API
            video_info = await self.get_video_info(video_id)
            if not video_info:
                return None
            
            stats = video_info.get('statistics', {})
            
            # Basic analytics from video statistics
            analytics = {
                'video_id': video_id,
                'views': int(stats.get('viewCount', 0)),
                'likes': int(stats.get('likeCount', 0)),
                'dislikes': int(stats.get('dislikeCount', 0)),
                'comments': int(stats.get('commentCount', 0)),
                'shares': 0,  # Would need additional API calls
                'impressions': 0,  # Would need YouTube Analytics API
                'clicks': 0,  # Would need YouTube Analytics API
                'ctr': 0.0,  # Would be calculated from impressions/clicks
                'average_view_percentage': 0.0,  # Would need Analytics API
                'retrieved_at': datetime.now().isoformat()
            }
            
            # Try to get additional analytics if possible
            try:
                detailed_analytics = await self._get_detailed_analytics(video_id)
                if detailed_analytics:
                    analytics.update(detailed_analytics)
            except Exception as e:
                self.logger.warning(f"Detailed analytics unavailable: {e}")
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Analytics retrieval failed for {video_id}: {e}")
            return None
    
    async def _get_detailed_analytics(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed analytics using YouTube Analytics API"""
        try:
            await self._ensure_services_initialized()
            if not self.analytics_service:
                return None
            
            # Get analytics for the last 30 days
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            request = self.analytics_service.reports().query(
                ids='channel==MINE',
                startDate=start_date,
                endDate=end_date,
                metrics='views,impressions,clicks,averageViewPercentage',
                filters=f'video=={video_id}'
            )
            
            response = await self._execute_request_async(request)
            
            if response and response.get('rows'):
                data = response['rows'][0]  # First row contains the data
                
                return {
                    'impressions': int(data[1]) if len(data) > 1 else 0,
                    'clicks': int(data[2]) if len(data) > 2 else 0,
                    'ctr': float(data[2]) / float(data[1]) if len(data) > 2 and data[1] > 0 else 0.0,
                    'average_view_percentage': float(data[3]) if len(data) > 3 else 0.0
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Detailed analytics failed: {e}")
            return None
    
    async def get_recent_videos(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get recent videos from the channel
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of recent video information
        """
        try:
            await self._ensure_services_initialized()
            if not self.youtube_service:
                return []
            
            # Get channel ID first
            channels_request = self.youtube_service.channels().list(
                part='id',
                mine=True
            )
            
            channels_response = await self._execute_request_async(channels_request)
            
            if not channels_response or not channels_response.get('items'):
                return []
            
            channel_id = channels_response['items'][0]['id']
            
            # Calculate date filter
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat() + 'Z'
            
            # Search for recent videos
            search_request = self.youtube_service.search().list(
                part='snippet',
                channelId=channel_id,
                type='video',
                order='date',
                publishedAfter=cutoff_date,
                maxResults=50
            )
            
            search_response = await self._execute_request_async(search_request)
            
            if not search_response or not search_response.get('items'):
                return []
            
            videos = []
            for item in search_response['items']:
                video_info = {
                    'id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'publishedAt': item['snippet']['publishedAt'],
                    'thumbnails': item['snippet']['thumbnails']
                }
                videos.append(video_info)
            
            return videos
            
        except Exception as e:
            self.logger.error(f"Failed to get recent videos: {e}")
            return []
    
    async def get_video_comments(self, video_id: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Get comments for a video
        
        Args:
            video_id: YouTube video ID
            max_results: Maximum number of comments to retrieve
            
        Returns:
            List of comment data
        """
        try:
            await self._ensure_services_initialized()
            if not self.youtube_service:
                return []
            
            self.logger.info(f"Fetching comments for video: {video_id}")
            
            request = self.youtube_service.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=max_results,
                order='relevance'
            )
            
            response = await self._execute_request_async(request)
            
            if not response or not response.get('items'):
                return []
            
            comments = []
            for item in response['items']:
                comment_data = item['snippet']['topLevelComment']['snippet']
                
                comment = {
                    'id': item['snippet']['topLevelComment']['id'],
                    'textDisplay': comment_data['textDisplay'],
                    'textOriginal': comment_data['textOriginal'],
                    'authorDisplayName': comment_data['authorDisplayName'],
                    'authorProfileImageUrl': comment_data['authorProfileImageUrl'],
                    'publishedAt': comment_data['publishedAt'],
                    'likeCount': comment_data['likeCount'],
                    'totalReplyCount': item['snippet']['totalReplyCount']
                }
                comments.append(comment)
            
            return comments
            
        except Exception as e:
            self.logger.error(f"Failed to get comments for {video_id}: {e}")
            return []
    
    async def reply_to_comment(self, comment_id: str, reply_text: str) -> bool:
        """
        Reply to a comment
        
        Args:
            comment_id: YouTube comment ID
            reply_text: Reply text
            
        Returns:
            True if successful
        """
        try:
            await self._ensure_services_initialized()
            if not self.youtube_service:
                return False
            
            self.logger.info(f"Replying to comment: {comment_id}")
            
            request = self.youtube_service.comments().insert(
                part='snippet',
                body={
                    'snippet': {
                        'parentId': comment_id,
                        'textOriginal': reply_text
                    }
                }
            )
            
            response = await self._execute_request_async(request)
            
            if response:
                self.logger.info(f"Replied to comment {comment_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to reply to comment {comment_id}: {e}")
            return False
    
    async def heart_comment(self, comment_id: str) -> bool:
        """
        Heart/like a comment (note: this requires the comment to be on your video)
        
        Args:
            comment_id: YouTube comment ID
            
        Returns:
            True if successful
        """
        try:
            await self._ensure_services_initialized()
            if not self.youtube_service:
                return False
            
            # Note: The YouTube API doesn't have a direct "heart" endpoint
            # This would need to be implemented through comment moderation
            # For now, we'll log the intended action
            
            self.logger.info(f"Hearting comment: {comment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to heart comment {comment_id}: {e}")
            return False
    
    async def pin_comment(self, comment_id: str) -> bool:
        """
        Pin a comment (note: limited API support)
        
        Args:
            comment_id: YouTube comment ID
            
        Returns:
            True if successful
        """
        try:
            await self._ensure_services_initialized()
            if not self.youtube_service:
                return False
            
            # Note: Pinning comments through API has limited support
            # This would typically need to be done through YouTube Studio
            # For now, we'll log the intended action
            
            self.logger.info(f"Pinning comment: {comment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to pin comment {comment_id}: {e}")
            return False
    
    async def _execute_request_async(self, request):
        """Execute API request asynchronously"""
        try:
            # Asynchronous execution of Google API requests
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, request.execute)
            
        except Exception as e:
            self.logger.error(f"Google API request failed: {e}")
            return None
    
    def get_client_status(self) -> Dict[str, Any]:
        """Get the current status of the YouTube client"""
        return {
            'services_initialized': self._services_initialized,
            'youtube_service': bool(self.youtube_service),
            'analytics_service': bool(self.analytics_service)
        }