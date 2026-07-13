"""
Reddit API integration for content fetching and filtering.
Handles Reddit authentication, post fetching, and content validation.
"""

import logging
import re
from typing import List, Optional, Dict, Any, Pattern
from dataclasses import dataclass
from datetime import datetime

import asyncpraw
import asyncprawcore
from asyncpraw.models import Submission

from src.config.settings import get_config


@dataclass
class RedditPost:
    """Structured representation of a Reddit post with quality metrics"""

    id: str
    title: str
    url: str
    subreddit: str
    author: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: datetime
    is_video: bool
    selftext: str = ""
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration: Optional[float] = None
    nsfw: bool = False
    spoiler: bool = False
    # Quality metrics for source filtering
    width: int = 0
    height: int = 0
    fps: float = 0
    # Reddit-specific URLs
    reddit_url: Optional[str] = None  # Full Reddit submission URL for API access

    @classmethod
    def from_submission(cls, submission: Submission) -> "RedditPost":
        """Create RedditPost from praw Submission with quality metrics"""
        # Determine if this is a video post
        is_video = False
        video_url = None
        duration = None
        width = 0
        height = 0
        fps = 0

        if hasattr(submission, "is_video") and submission.is_video:
            is_video = True
            if hasattr(submission, "media") and submission.media:
                media = submission.media
                if isinstance(media, dict):
                    reddit_video = media.get("reddit_video", {}) or {}
                else:
                    reddit_video = getattr(media, "reddit_video", {}) or {}

                if not isinstance(reddit_video, dict):
                    reddit_video = {
                        "fallback_url": getattr(reddit_video, "fallback_url", None),
                        "hls_url": getattr(reddit_video, "hls_url", None),
                        "duration": getattr(reddit_video, "duration", None),
                        "width": getattr(reddit_video, "width", 0),
                        "height": getattr(reddit_video, "height", 0),
                        "fps": getattr(reddit_video, "fps", 0),
                        "scanned_size": getattr(reddit_video, "scanned_size", None),
                    }

                video_url = reddit_video.get("fallback_url") or reddit_video.get(
                    "hls_url"
                )
                duration = reddit_video.get("duration")
                width = reddit_video.get("width", 0)
                height = reddit_video.get("height", 0)
                fps = reddit_video.get("fps", 0) or 0
        elif submission.url and any(
            submission.url.lower().endswith(ext)
            for ext in [".mp4", ".webm", ".mov", ".avi"]
        ):
            is_video = True
            video_url = submission.url
        elif "youtube.com" in submission.url or "youtu.be" in submission.url:
            is_video = True
            video_url = submission.url
        elif "v.redd.it" in submission.url:
            is_video = True
            video_url = submission.url

        # Generate the proper Reddit submission URL
        reddit_url = f"https://www.reddit.com/r/{submission.subreddit.display_name}/comments/{submission.id}/"

        return cls(
            id=submission.id,
            title=submission.title,
            url=submission.url,
            subreddit=submission.subreddit.display_name,
            author=str(submission.author) if submission.author else "[deleted]",
            score=submission.score,
            upvote_ratio=submission.upvote_ratio,
            num_comments=submission.num_comments,
            created_utc=datetime.fromtimestamp(submission.created_utc),
            is_video=is_video,
            selftext=getattr(submission, "selftext", "") or "",
            video_url=video_url,
            thumbnail_url=submission.thumbnail
            if hasattr(submission, "thumbnail")
            else None,
            duration=duration,
            nsfw=submission.over_18,
            spoiler=submission.spoiler,
            width=width,
            height=height,
            fps=fps,
            reddit_url=reddit_url,
        )


class RedditClient:
    """Async Reddit API client for fetching and managing posts"""

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.reddit: Optional[asyncpraw.Reddit] = None

        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry"""
        await self._initialize_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def close(self):
        """Close the Reddit client"""
        if self.reddit:
            await self.reddit.close()
            self.reddit = None
            self._initialized = False

    async def _initialize_client(self):
        """Initialize the async Reddit client"""
        if self._initialized:
            return

        if not all(
            [self.config.api.reddit_client_id, self.config.api.reddit_client_secret]
        ):
            self.logger.error(
                "Reddit credentials not configured. Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in your .env file"
            )
            self.logger.error(
                "Run 'python test_reddit_connection.py' for setup instructions"
            )
            return

        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.config.api.reddit_client_id,
                client_secret=self.config.api.reddit_client_secret,
                user_agent=self.config.api.reddit_user_agent,
            )

            # Test the connection
            try:
                user = await self.reddit.user.me()
                if user:
                    self.logger.info(
                        f"Reddit client initialized, authenticated as: {user}"
                    )
                else:
                    self.logger.info("Reddit client initialized (read-only mode)")
            except asyncprawcore.ResponseException:
                self.logger.info("Reddit client initialized (read-only mode)")

            self._initialized = True

        except asyncprawcore.ResponseException as e:
            if e.response.status == 401:
                self.logger.error("Reddit authentication failed: Invalid credentials")
                self.logger.error(
                    "Please check your REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in the .env file"
                )
                self.logger.error(
                    "Run 'python test_reddit_connection.py' to verify your Reddit API setup"
                )
            else:
                self.logger.error(
                    f"Reddit API error during initialization: HTTP {e.response.status} - {e}"
                )
            self.reddit = None
        except Exception as e:
            self.logger.error(f"Failed to initialize Reddit client: {e}")
            self.reddit = None

    def is_connected(self) -> bool:
        """Check if Reddit client is properly initialized"""
        return self.reddit is not None and self._initialized

    async def fetch_posts_from_subreddit(
        self,
        subreddit_name: str,
        sort_method: str = "hot",
        time_filter: str = "day",
        limit: int = 10,
    ) -> List[RedditPost]:
        """
        Fetch posts from a specific subreddit

        Args:
            subreddit_name: Name of the subreddit
            sort_method: 'hot', 'new', 'top', 'rising'
            time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'
            limit: Maximum number of posts to fetch
        """
        if not self.is_connected():
            await self._initialize_client()
            if not self.is_connected():
                self.logger.error("Reddit client not connected")
                return []

        try:
            subreddit = await self.reddit.subreddit(subreddit_name)
            posts = []

            # Get submissions based on sort method
            if sort_method == "hot":
                submissions = subreddit.hot(limit=limit)
            elif sort_method == "new":
                submissions = subreddit.new(limit=limit)
            elif sort_method == "top":
                submissions = subreddit.top(time_filter=time_filter, limit=limit)
            elif sort_method == "rising":
                submissions = subreddit.rising(limit=limit)
            else:
                self.logger.warning(f"Unknown sort method: {sort_method}, using 'hot'")
                submissions = subreddit.hot(limit=limit)

            async for submission in submissions:
                try:
                    post = RedditPost.from_submission(submission)
                    posts.append(post)
                except Exception as e:
                    self.logger.warning(
                        f"Error processing submission {submission.id}: {e}"
                    )
                    continue

            self.logger.info(f"Fetched {len(posts)} posts from r/{subreddit_name}")
            return posts

        except asyncprawcore.NotFound:
            self.logger.error(f"Subreddit r/{subreddit_name} not found")
            return []
        except asyncprawcore.Forbidden:
            self.logger.error(f"Access forbidden to r/{subreddit_name}")
            return []
        except asyncprawcore.ResponseException as e:
            if e.response.status == 401:
                self.logger.error(
                    f"Authentication failed for r/{subreddit_name}: Reddit API credentials invalid or expired"
                )
                self.logger.error(
                    "Please check your REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in the .env file"
                )
                self.logger.error(
                    "Run 'python test_reddit_connection.py' to verify your Reddit API setup"
                )
            else:
                self.logger.error(
                    f"Reddit API error for r/{subreddit_name}: HTTP {e.response.status} - {e}"
                )
            return []
        except Exception as e:
            self.logger.error(f"Error fetching posts from r/{subreddit_name}: {e}")
            return []

    async def get_post_by_url(self, url: str) -> Optional[RedditPost]:
        """Get a specific post by Reddit URL"""
        if not self.is_connected():
            await self._initialize_client()
            if not self.is_connected():
                return None

        try:
            # Handle different URL formats
            if "reddit.com" in url and "/comments/" in url:
                # This is a proper Reddit submission URL
                submission = await self.reddit.submission(url=url)
            elif url.startswith("https://v.redd.it/"):
                # This is a direct video URL - we cannot fetch submission data from this
                self.logger.error(f"Cannot fetch submission data from video URL: {url}")
                return None
            else:
                # Try as submission URL anyway
                submission = await self.reddit.submission(url=url)

            await submission.load()  # Ensure submission data is loaded
            return RedditPost.from_submission(submission)
        except asyncprawcore.exceptions.NotFound:
            self.logger.error(f"Reddit submission not found: {url}")
            return None
        except asyncprawcore.ResponseException as e:
            if e.response.status == 401:
                self.logger.error(
                    f"Authentication failed for URL {url}: Reddit API credentials invalid or expired"
                )
                self.logger.error(
                    "Please check your REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in the .env file"
                )
                self.logger.error(
                    "Run 'python test_reddit_connection.py' to verify your Reddit API setup"
                )
            elif e.response.status == 404:
                self.logger.error(f"Reddit post not found: {url}")
            else:
                self.logger.error(
                    f"Reddit API error for URL {url}: HTTP {e.response.status} - {e}"
                )
            return None
        except Exception as e:
            self.logger.error(f"Error fetching post from URL {url}: {e}")
            return None
