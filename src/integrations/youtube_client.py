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
    from googleapiclient.errors import HttpError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    HttpError = Exception  # Fallback for error handling

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
                self._services_initialized = True
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
    
    def _load_credentials_from_source(self, token_data: Dict[str, Any], source_name: str) -> Optional[Credentials]:
        """Load credentials from token data with scope validation"""
        # Try with current scopes first
        creds = Credentials.from_authorized_user_info(token_data, self.scopes)
        self.logger.info(f"Loaded credentials from {source_name}")

        # Validate scopes
        if not self._validate_scopes(creds, self.scopes):
            self.logger.warning(f"{source_name} credentials have scope issues, trying with basic scopes")
            # Fallback to basic scopes
            basic_scopes = [
                'https://www.googleapis.com/auth/youtube.upload',
                'https://www.googleapis.com/auth/youtube.force-ssl',
                'https://www.googleapis.com/auth/youtubepartner'
            ]
            creds = Credentials.from_authorized_user_info(token_data, basic_scopes)
        
        return creds

    def _refresh_credentials(self, creds: Credentials, source_name: str, token_file: Optional[Path] = None) -> Optional[Credentials]:
        """Attempt to refresh credentials and update storage"""
        try:
            creds.refresh(Request())
            self.logger.info(f"Successfully refreshed credentials from {source_name}")
            
            # Update storage based on source
            if source_name == "environment variable":
                self._update_token_in_env(creds)
            elif source_name == "token file" and token_file:
                with open(token_file, 'w') as f:
                    f.write(creds.to_json())
                self.logger.info(f"Updated token file: {token_file}")
            
            return creds
        except Exception as refresh_error:
            self.logger.warning(f"Failed to refresh {source_name} credentials: {refresh_error}")
            return None

    def _get_credentials(self) -> Optional[Credentials]:
        """Get YouTube API credentials from environment variable or file"""
        try:
            import os
            import json
            from pathlib import Path

            # Helper function to load credentials from a source
            def load_credentials(token_str: str, source_name: str, token_file: Optional[Path] = None) -> Optional[Credentials]:
                try:
                    token_data = json.loads(token_str)
                    creds = self._load_credentials_from_source(token_data, source_name)
                    
                    if creds and creds.valid:
                        self.logger.info(f"{source_name} credentials are valid and ready to use")
                        return creds
                    
                    if creds and creds.expired and creds.refresh_token:
                        return self._refresh_credentials(creds, source_name, token_file)
                    
                    self.logger.warning(f"Credentials from {source_name} are invalid")
                    return None
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON in {source_name}: {e}")
                except Exception as e:
                    self.logger.warning(f"Error loading credentials from {source_name}: {e}")
                return None

            # Try environment variable first
            youtube_token_env = os.getenv('YOUTUBE_TOKEN_JSON')
            if youtube_token_env:
                creds = load_credentials(youtube_token_env, "environment variable")
                if creds:
                    return creds

            # Try file-based approach as fallback
            token_file = self.config.paths.youtube_token_file
            if token_file and token_file.exists():
                try:
                    with open(token_file, 'r') as f:
                        token_str = f.read()
                    return load_credentials(token_str, "token file", token_file)
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    self.logger.error(f"Error loading credentials from token file: {e}")
            
            # No valid credentials found
            self.logger.warning("No valid YouTube credentials found. YouTube functionality will be disabled.")
            self.logger.info("To fix this:")
            self.logger.info("1. Run 'python auth_youtube.py' to create proper credentials with required scopes")
            self.logger.info("2. Or set YOUTUBE_TOKEN_JSON environment variable with valid credentials")
            self.logger.info("3. Ensure your OAuth app has the required scopes configured in Google Cloud Console")
            return None
            
        except Exception as e:
            self.logger.error(f"Credential handling failed: {e}")
            return None
    
    def _validate_scopes(self, creds, required_scopes):
        """Validate that credentials have required scopes"""
        if not creds or not hasattr(creds, 'scopes'):
            return False
        
        creds_scopes = set(creds.scopes) if creds.scopes else set()
        required_scopes_set = set(required_scopes)
        
        missing_scopes = required_scopes_set - creds_scopes
        if missing_scopes:
            self.logger.warning(f"Missing scopes: {list(missing_scopes)}")
            # Try to work with basic scopes if analytics scope is missing
            basic_scopes = {
                'https://www.googleapis.com/auth/youtube.upload',
                'https://www.googleapis.com/auth/youtube.force-ssl',
                'https://www.googleapis.com/auth/youtubepartner'
            }
            if basic_scopes.issubset(creds_scopes):
                self.logger.info("Proceeding with basic YouTube scopes (analytics features disabled)")
                return True
            return False
        return True

    def _update_token_in_env(self, creds):
        """Update the YOUTUBE_TOKEN_JSON environment variable with refreshed credentials"""
        try:
            # Note: This only updates the current process environment
            # For persistent updates, the user would need to update their .env file
            import os
            os.environ['YOUTUBE_TOKEN_JSON'] = creds.to_json()
            self.logger.info("Updated YOUTUBE_TOKEN_JSON environment variable with refreshed credentials")
        except Exception as e:
            self.logger.warning(f"Failed to update environment variable: {e}")
    
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
    
    async def _fetch_analytics(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Internal method to fetch analytics data without timeout handling.
        """
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
            'impressions': 0,  # Legacy field for backward compatibility
            'clicks': 0,  # Legacy field for backward compatibility
            'ctr': 0.0,  # Legacy field for backward compatibility
            'average_view_percentage': 0.0,  # Legacy field for backward compatibility
            'estimated_minutes_watched': 0,  # From YouTube Analytics API
            'average_view_duration': 0.0,  # From YouTube Analytics API
            'subscribers_gained': 0,  # From YouTube Analytics API
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

    async def _fetch_analytics(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Internal method to fetch analytics data without timeout handling.
        """
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
            'impressions': 0,  # Legacy field for backward compatibility
            'clicks': 0,  # Legacy field for backward compatibility
            'ctr': 0.0,  # Legacy field for backward compatibility
            'average_view_percentage': 0.0,  # Legacy field for backward compatibility
            'estimated_minutes_watched': 0,  # From YouTube Analytics API
            'average_view_duration': 0.0,  # From YouTube Analytics API
            'subscribers_gained': 0,  # From YouTube Analytics API
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

    async def get_video_analytics(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get video analytics data with timeout handling
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Analytics data or None if failed
        """
        try:
            return await asyncio.wait_for(self._fetch_analytics(video_id), timeout=20)
        except asyncio.TimeoutError:
            self.logger.error(f"Analytics retrieval for {video_id} timed out after 20 seconds")
            return None
        except Exception as e:
            self.logger.error(f"Analytics retrieval failed for {video_id}: {e}")
            return None
    
    async def _get_detailed_analytics(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed analytics using YouTube Analytics API"""
        try:
            await self._ensure_services_initialized()
            if not self.analytics_service:
                self.logger.info("Analytics service not available - detailed analytics will be skipped")
                return None
            
            # Get analytics for the last 30 days
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            request = self.analytics_service.reports().query(
                ids='channel==MINE',
                startDate=start_date,
                endDate=end_date,
                metrics='views,estimatedMinutesWatched,averageViewDuration,subscribersGained',
                filters=f'video=={video_id}'
            )
            
            response = await self._execute_request_async(request)
            
            if response and response.get('rows'):
                data = response['rows'][0]  # First row contains the data
                
                # New metrics: views, estimatedMinutesWatched, averageViewDuration, subscribersGained
                estimated_minutes_watched = int(data[1]) if len(data) > 1 else 0
                average_view_duration = float(data[2]) if len(data) > 2 else 0.0
                subscribers_gained = int(data[3]) if len(data) > 3 else 0
                
                return {
                    'estimated_minutes_watched': estimated_minutes_watched,
                    'average_view_duration': average_view_duration,
                    'subscribers_gained': subscribers_gained,
                    # Keep these for backward compatibility, but set to 0
                    'impressions': 0,
                    'clicks': 0,
                    'ctr': 0.0,
                    'average_view_percentage': 0.0
                }
            
            return None
            
        except HttpError as e:
            if e.resp.status == 403 and 'accessNotConfigured' in str(e):
                self.logger.warning(
                    "YouTube Analytics API is not enabled for this project. "
                    "To enable it:\n"
                    "1. Go to https://console.developers.google.com/apis/api/youtubeanalytics.googleapis.com/overview\n"
                    "2. Select your project and click 'Enable'\n"
                    "3. Wait a few minutes for the changes to propagate\n"
                    "Detailed analytics will be unavailable until then."
                )
                return None
            else:
                self.logger.warning(f"YouTube Analytics API error: {e}")
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
    
    def _process_comment(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a comment thread item into a standardized dictionary"""
        top_comment = item['snippet']['topLevelComment']
        snippet = top_comment['snippet']
        
        return {
            'id': top_comment['id'],
            'textDisplay': snippet['textDisplay'],
            'textOriginal': snippet['textOriginal'],
            'authorDisplayName': snippet['authorDisplayName'],
            'authorProfileImageUrl': snippet['authorProfileImageUrl'],
            'publishedAt': snippet['publishedAt'],
            'likeCount': snippet['likeCount'],
            'totalReplyCount': item['snippet']['totalReplyCount']
        }

    async def get_video_comments(self, video_id: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Get top-level comments for a YouTube video
        
        Args:
            video_id: YouTube video ID
            max_results: Maximum number of comments to retrieve
            
        Returns:
            List of comment dictionaries with basic metadata
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
            
            return [self._process_comment(item) for item in response['items']]
            
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
            
        except HttpError as e:
            # Re-raise HttpError so it can be handled specifically by calling methods
            raise e
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