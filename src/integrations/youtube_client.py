"""
Enhanced YouTube Client with A/B Testing and Analytics Integration
Handles video uploads, thumbnail management, comment interactions, and performance tracking.
"""

import logging
import asyncio
import functools
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
    # from google_auth_oauthlib.flow import InstalledAppFlow

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
        self._services_init_lock = asyncio.Lock()

        if not GOOGLE_API_AVAILABLE:
            self.logger.warning(
                "Google API libraries not available - YouTube features will be limited"
            )

    def _initialize_services(self):
        """Initialize YouTube and Analytics API services"""
        try:
            credentials = self._get_credentials()
            if credentials:
                self.youtube_service = build("youtube", "v3", credentials=credentials)
                self.analytics_service = build(
                    "youtubeAnalytics", "v2", credentials=credentials
                )
                self._services_initialized = True
                self.logger.info("YouTube API services initialized successfully")
            else:
                self.logger.warning(
                    "Failed to obtain YouTube API credentials, YouTube features will be disabled."
                )

        except Exception as e:
            self.logger.error(f"YouTube API initialization failed: {e}")

    async def _ensure_services_initialized(self):
        """Ensure services are initialized before use"""
        if self._services_initialized or not GOOGLE_API_AVAILABLE:
            return

        async with self._services_init_lock:
            if not self._services_initialized:
                await asyncio.to_thread(self._initialize_services)

    def _get_credentials(self):
        """Get YouTube API credentials from environment variable or file"""
        try:
            # First try YOUTUBE_TOKEN_JSON environment variable
            youtube_token_env = os.getenv("YOUTUBE_TOKEN_JSON")
            if youtube_token_env:
                try:
                    token_data = json.loads(youtube_token_env)
                    creds = Credentials.from_authorized_user_info(
                        token_data, self.scopes
                    )
                    self.logger.info(
                        "Loaded credentials from YOUTUBE_TOKEN_JSON environment variable"
                    )

                    # If credentials are valid, return them immediately
                    if creds and creds.valid:
                        self.logger.info("Credentials are valid and ready to use")
                        return creds

                    # If credentials exist but are expired, try to refresh
                    if creds and creds.expired and creds.refresh_token:
                        try:
                            creds.refresh(Request())
                            self.logger.info(
                                "Successfully refreshed credentials from environment variable"
                            )
                            # Update the environment variable with refreshed token
                            self._update_token_in_env(creds)
                            return creds
                        except Exception as refresh_error:
                            self.logger.warning(
                                f"Failed to refresh credentials from env var: {refresh_error}"
                            )
                            # Fall through to try file-based approach

                except json.JSONDecodeError as e:
                    self.logger.error(
                        f"Invalid JSON in YOUTUBE_TOKEN_JSON environment variable: {e}"
                    )
                    return None
                except Exception as e:
                    self.logger.warning(
                        f"Error loading credentials from environment variable: {e}"
                    )
                    # Fall through to try file-based approach

            # Try file-based approach as fallback
            token_file = self.config.paths.youtube_token_file
            if token_file and token_file.exists():
                try:
                    with open(token_file, "r") as f:
                        token_data = json.load(f)
                    creds = Credentials.from_authorized_user_info(
                        token_data, self.scopes
                    )
                    self.logger.info(
                        f"Loaded credentials from token file: {token_file}"
                    )

                    # If credentials are valid, return them immediately
                    if creds and creds.valid:
                        self.logger.info(
                            "File-based credentials are valid and ready to use"
                        )
                        return creds

                    # If credentials exist but are expired, try to refresh
                    if creds and creds.expired and creds.refresh_token:
                        try:
                            creds.refresh(Request())
                            self.logger.info(
                                "Successfully refreshed credentials from token file"
                            )
                            # Update the token file with refreshed credentials
                            with open(token_file, "w") as f:
                                f.write(creds.to_json())
                            self.logger.info(f"Updated token file: {token_file}")
                            return creds
                        except Exception as refresh_error:
                            self.logger.error(
                                f"Failed to refresh credentials from file: {refresh_error}"
                            )
                            return None
                    else:
                        self.logger.error(
                            "File-based credentials are invalid and cannot be refreshed"
                        )
                        return None

                except (json.JSONDecodeError, FileNotFoundError) as e:
                    self.logger.error(f"Error loading credentials from token file: {e}")
                    return None

            self.logger.warning(
                "No valid YouTube credentials found. YouTube functionality will be disabled."
            )
            self.logger.info("To fix this:")
            self.logger.info(
                "1. Set YOUTUBE_TOKEN_JSON environment variable with valid credentials, OR"
            )
            self.logger.info(
                "2. Run 'python auth_youtube.py' to create/refresh youtube_token.json file"
            )
            return None

        except Exception as e:
            self.logger.error(f"Credential handling failed: {e}")
            return None

    def _update_token_in_env(self, creds):
        """Update in-process YOUTUBE_TOKEN_JSON (not persisted to disk)."""
        try:
            # Note: This only updates the current process environment
            # For persistent updates, the user would need to update their .env file
            import os

            os.environ["YOUTUBE_TOKEN_JSON"] = creds.to_json()
            self.logger.info(
                "Updated YOUTUBE_TOKEN_JSON for this process only; persist manually for future sessions"
            )
        except Exception as e:
            self.logger.warning(f"Failed to update environment variable: {e}")

    async def upload_video(
        self,
        video_path: str,
        metadata: Dict[str, Any],
        thumbnail_path: Optional[str] = None,
    ) -> Dict[str, Any]:
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
                return {"success": False, "error": "YouTube service not available"}

            self.logger.info(f"Uploading video: {video_path}")

            # Prepare video metadata
            video_body = {
                "snippet": {
                    "title": metadata.get("title", "Untitled Video"),
                    "description": metadata.get("description", ""),
                    "tags": metadata.get("tags", []),
                    "categoryId": metadata.get("category_id", "22"),  # People & Blogs
                    "defaultLanguage": "en",
                    "defaultAudioLanguage": "en",
                },
                "status": {
                    "privacyStatus": metadata.get("privacy_status", "public"),
                    "madeForKids": False,
                    "selfDeclaredMadeForKids": False,
                },
            }

            # Upload video
            loop = asyncio.get_running_loop()
            media_partial = functools.partial(
                MediaFileUpload,
                video_path,
                chunksize=-1,
                resumable=True,
                mimetype="video/*",
            )
            media = await loop.run_in_executor(None, media_partial)

            # Execute upload
            insert_request = self.youtube_service.videos().insert(
                part=",".join(video_body.keys()), body=video_body, media_body=media
            )

            response = await self._execute_request_async(insert_request)

            if not response:
                return {"success": False, "error": "Upload failed"}

            video_id = response["id"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"

            # Upload thumbnail if provided
            if thumbnail_path and Path(thumbnail_path).exists():
                await self.update_video_thumbnail(video_id, thumbnail_path)

            self.logger.info(f"Video uploaded successfully: {video_id}")

            return {
                "success": True,
                "video_id": video_id,
                "video_url": video_url,
                "upload_time": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Video upload failed: {e}")
            return {"success": False, "error": str(e)}

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
            loop = asyncio.get_running_loop()
            media_partial = functools.partial(
                MediaFileUpload, thumbnail_path, mimetype="image/jpeg"
            )
            media = await loop.run_in_executor(None, media_partial)

            request = self.youtube_service.thumbnails().set(
                videoId=video_id, media_body=media
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
                part="snippet,statistics,status", id=video_id
            )

            response = await self._execute_request_async(request)

            if response and response.get("items"):
                video_info = response["items"][0]
                return {
                    "id": video_info["id"],
                    "title": video_info["snippet"]["title"],
                    "description": video_info["snippet"]["description"],
                    "publishedAt": video_info["snippet"]["publishedAt"],
                    "tags": video_info["snippet"].get("tags", []),
                    "statistics": video_info.get("statistics", {}),
                    "status": video_info.get("status", {}),
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

            # Always get baseline statistics from the videos endpoint.
            video_info = await self.get_video_info(video_id)
            if not video_info:
                return None

            stats = video_info.get("statistics", {})

            analytics = {
                "video_id": video_id,
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "dislikes": int(stats.get("dislikeCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
                "shares": 0,
                "impressions": 0,
                "clicks": 0,
                "ctr": 0.0,
                "average_view_duration": 0.0,
                "average_view_percentage": 0.0,
                "retrieved_at": datetime.now().isoformat(),
            }

            if self.analytics_service:
                detailed_analytics = await self._get_detailed_analytics(video_id)
                if detailed_analytics:
                    analytics.update(detailed_analytics)
                else:
                    self.logger.info(
                        "Falling back to videos endpoint stats only for %s (analytics API returned no usable rows)",
                        video_id,
                    )
            else:
                self.logger.info(
                    "YouTube Analytics service unavailable; using videos endpoint stats only for %s",
                    video_id,
                )

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
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

            request = self.analytics_service.reports().query(
                ids="channel==MINE",
                startDate=start_date,
                endDate=end_date,
                metrics="views,likes,comments,impressions,impressionsCtr,averageViewDuration,averageViewPercentage",
                filters=f"video=={video_id}",
            )

            response = await self._execute_request_async(request)

            if not response:
                return None

            parsed = self._parse_analytics_report_response(response)
            if not parsed:
                return None

            ctr = parsed.get("ctr", 0.0)
            impressions = parsed.get("impressions", 0)
            clicks = int(round(impressions * ctr)) if impressions > 0 and ctr > 0 else 0

            return {
                "impressions": impressions,
                "clicks": clicks,
                "ctr": ctr,
                "average_view_duration": parsed.get("average_view_duration", 0.0),
                "average_view_percentage": parsed.get("average_view_percentage", 0.0),
                "likes": parsed.get("likes", 0),
                "comments": parsed.get("comments", 0),
                "views": parsed.get("views", 0),
            }

        except Exception as e:
            self.logger.warning(f"Detailed analytics failed: {e}")
            return None

    def _parse_analytics_report_response(
        self, response: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Parse YouTube Analytics API report rows robustly using column headers."""
        rows = response.get("rows") or []
        if not rows:
            return None

        column_headers = response.get("columnHeaders") or []
        metric_columns: Dict[str, int] = {}
        for index, column in enumerate(column_headers):
            if isinstance(column, dict) and column.get("columnType") == "METRIC":
                metric_columns[column.get("name", "")] = index

        if not metric_columns:
            metric_columns = {
                "views": 0,
                "likes": 1,
                "comments": 2,
                "impressions": 3,
                "impressionsCtr": 4,
                "averageViewDuration": 5,
                "averageViewPercentage": 6,
            }

        def _int_metric(row: List[Any], metric_name: str) -> int:
            idx = metric_columns.get(metric_name)
            if idx is None or idx >= len(row):
                return 0
            try:
                return int(float(row[idx]))
            except (TypeError, ValueError):
                return 0

        def _float_metric(row: List[Any], metric_name: str) -> float:
            idx = metric_columns.get(metric_name)
            if idx is None or idx >= len(row):
                return 0.0
            try:
                return float(row[idx])
            except (TypeError, ValueError):
                return 0.0

        total_impressions = 0
        total_views = 0
        total_likes = 0
        total_comments = 0
        total_view_duration_seconds = 0.0
        weighted_view_percentage = 0.0
        weighted_ctr = 0.0
        ctr_weight = 0

        for row in rows:
            impressions = _int_metric(row, "impressions")
            views = _int_metric(row, "views")
            likes = _int_metric(row, "likes")
            comments = _int_metric(row, "comments")
            avg_view_duration = _float_metric(row, "averageViewDuration")
            avg_view_pct = _float_metric(row, "averageViewPercentage")
            ctr_value = _float_metric(row, "impressionsCtr")

            total_impressions += impressions
            total_views += views
            total_likes += likes
            total_comments += comments

            if views > 0:
                total_view_duration_seconds += avg_view_duration * views
                weighted_view_percentage += avg_view_pct * views

            if impressions > 0:
                weighted_ctr += ctr_value * impressions
                ctr_weight += impressions

        average_view_duration = (
            total_view_duration_seconds / total_views if total_views > 0 else 0.0
        )
        average_view_percentage = (
            weighted_view_percentage / total_views if total_views > 0 else 0.0
        )
        raw_ctr = weighted_ctr / ctr_weight if ctr_weight > 0 else 0.0

        # impressionsCtr is typically returned as a percentage (e.g., 5.2), normalize to 0..1.
        ctr = raw_ctr / 100.0 if raw_ctr > 1.0 else raw_ctr
        ctr = max(0.0, min(1.0, ctr))

        return {
            "views": total_views,
            "likes": total_likes,
            "comments": total_comments,
            "impressions": total_impressions,
            "ctr": ctr,
            "average_view_duration": average_view_duration,
            "average_view_percentage": average_view_percentage,
        }

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
                part="id", mine=True
            )

            channels_response = await self._execute_request_async(channels_request)

            if not channels_response or not channels_response.get("items"):
                return []

            channel_id = channels_response["items"][0]["id"]

            # Calculate date filter
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat() + "Z"

            # Search for recent videos
            search_request = self.youtube_service.search().list(
                part="snippet",
                channelId=channel_id,
                type="video",
                order="date",
                publishedAfter=cutoff_date,
                maxResults=50,
            )

            search_response = await self._execute_request_async(search_request)

            if not search_response or not search_response.get("items"):
                return []

            videos = []
            for item in search_response["items"]:
                video_info = {
                    "id": item["id"]["videoId"],
                    "title": item["snippet"]["title"],
                    "description": item["snippet"]["description"],
                    "publishedAt": item["snippet"]["publishedAt"],
                    "thumbnails": item["snippet"]["thumbnails"],
                }
                videos.append(video_info)

            return videos

        except Exception as e:
            self.logger.error(f"Failed to get recent videos: {e}")
            return []

    async def get_video_comments(
        self, video_id: str, max_results: int = 50
    ) -> List[Dict[str, Any]]:
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
                part="snippet",
                videoId=video_id,
                maxResults=max_results,
                order="relevance",
            )

            response = await self._execute_request_async(request)

            if not response or not response.get("items"):
                return []

            comments = []
            for item in response["items"]:
                comment_data = item["snippet"]["topLevelComment"]["snippet"]

                comment = {
                    "id": item["snippet"]["topLevelComment"]["id"],
                    "textDisplay": comment_data["textDisplay"],
                    "textOriginal": comment_data["textOriginal"],
                    "authorDisplayName": comment_data["authorDisplayName"],
                    "authorProfileImageUrl": comment_data["authorProfileImageUrl"],
                    "publishedAt": comment_data["publishedAt"],
                    "likeCount": comment_data["likeCount"],
                    "totalReplyCount": item["snippet"]["totalReplyCount"],
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
                part="snippet",
                body={"snippet": {"parentId": comment_id, "textOriginal": reply_text}},
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

            self.logger.warning(
                "heart_comment is not implemented via YouTube API for comment_id=%s",
                comment_id,
            )
            return False

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

            self.logger.warning(
                "pin_comment is not implemented via YouTube API for comment_id=%s",
                comment_id,
            )
            return False

        except Exception as e:
            self.logger.error(f"Failed to pin comment {comment_id}: {e}")
            return False

    async def _execute_request_async(self, request):
        """Execute API request asynchronously"""
        try:
            # Asynchronous execution of Google API requests
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, request.execute)

        except Exception as e:
            self.logger.error(f"Google API request failed: {e}")
            return None

    def get_client_status(self) -> Dict[str, Any]:
        """Get the current status of the YouTube client"""
        return {
            "services_initialized": self._services_initialized,
            "youtube_service": bool(self.youtube_service),
            "analytics_service": bool(self.analytics_service),
        }
