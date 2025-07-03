"""
Reddit API integration for content fetching and filtering.
Handles Reddit authentication, post fetching, and content validation.
"""

import re
import logging
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

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
    def from_submission(cls, submission: Submission) -> 'RedditPost':
        """Create RedditPost from praw Submission with quality metrics"""
        # Determine if this is a video post
        is_video = False
        video_url = None
        duration = None
        width = 0
        height = 0
        fps = 0
        
        if hasattr(submission, 'is_video') and submission.is_video:
            is_video = True
            if hasattr(submission, 'media') and submission.media:
                reddit_video = submission.media.get('reddit_video', {})
                video_url = reddit_video.get('fallback_url') or reddit_video.get('hls_url')
                duration = reddit_video.get('duration')
                width = reddit_video.get('width', 0)
                height = reddit_video.get('height', 0)
                # Try to extract FPS if available
                if hasattr(submission.media, 'reddit_video'):
                    fps = reddit_video.get('fps', 0)
                    # Estimate FPS from duration and frame count if available
                    if not fps and reddit_video.get('duration') and reddit_video.get('scanned_size'):
                        try:
                            fps = reddit_video['scanned_size'] / reddit_video['duration']
                        except (ZeroDivisionError, TypeError):
                            fps = 0
        elif submission.url and any(submission.url.lower().endswith(ext) for ext in ['.mp4', '.webm', '.mov', '.avi']):
            is_video = True
            video_url = submission.url
        elif 'youtube.com' in submission.url or 'youtu.be' in submission.url:
            is_video = True
            video_url = submission.url
        elif 'v.redd.it' in submission.url:
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
            video_url=video_url,
            thumbnail_url=submission.thumbnail if hasattr(submission, 'thumbnail') else None,
            duration=duration,
            nsfw=submission.over_18,
            spoiler=submission.spoiler,
            width=width,
            height=height,
            fps=fps,
            reddit_url=reddit_url
        )


class ContentFilter:
    """Content filtering and validation for Reddit posts"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def contains_forbidden_words(self, text: Optional[str]) -> bool:
        """Check if text contains forbidden words"""
        if not text:
            return False
        
        text_lower = text.lower()
        pattern = r'\b(' + '|'.join(re.escape(word) for word in self.config.content.forbidden_words) + r')\b'
        return re.search(pattern, text_lower) is not None
    
    def contains_unsuitable_content(self, text: Optional[str]) -> bool:
        """Check if text contains unsuitable content types"""
        if not text:
            return False
        
        text_lower = text.lower()
        return any(content_type in text_lower for content_type in self.config.content.unsuitable_content_types)
    
    def is_suitable_for_monetization(self, post: RedditPost) -> Dict[str, Any]:
        """
        Comprehensive content suitability check for monetization, including quality.
        
        Returns:
            Dict with 'is_suitable', 'reason', and 'confidence' keys
        """
        reasons = []
        
        # Check NSFW/spoiler flags
        if post.nsfw:
            reasons.append("NSFW content")
        
        if post.spoiler:
            reasons.append("Spoiler content")
        
        # Check title for forbidden content
        if self.contains_forbidden_words(post.title):
            reasons.append("Forbidden words in title")
        
        if self.contains_unsuitable_content(post.title):
            reasons.append("Unsuitable content type in title")
        
        # Check subreddit name
        if self.contains_forbidden_words(post.subreddit):
            reasons.append("Unsuitable subreddit name")
        
        # Quality checks
        if post.score < 10:
            reasons.append("Low score (potential low quality)")
        
        if post.upvote_ratio < 0.7:
            reasons.append("Low upvote ratio (controversial content)")
        
        # Video-specific checks
        if post.is_video:
            if post.duration:
                if post.duration < 5:
                    reasons.append("Video too short")
                elif post.duration > 300:  # 5 minutes
                    reasons.append("Video too long")
            
            # Source Quality Checks
            if post.width > 0 and post.height > 0:
                min_resolution = min(post.width, post.height)
                if min_resolution < 720:  # Minimum acceptable resolution for Shorts (720p vertical)
                    reasons.append(f"Low source resolution ({post.width}x{post.height})")
            
            if post.fps > 0 and post.fps < 25:  # Minimum acceptable FPS
                reasons.append(f"Low source FPS ({post.fps})")
        
        is_suitable = len(reasons) == 0
        confidence = 95 if is_suitable else max(60, 100 - len(reasons) * 15)
        
        return {
            'is_suitable': is_suitable,
            'reason': '; '.join(reasons) if reasons else 'Content appears suitable',
            'confidence': confidence
        }
    
    def filter_posts(self, posts: List[RedditPost]) -> List[RedditPost]:
        """Filter posts for suitability"""
        suitable_posts = []
        
        for post in posts:
            suitability = self.is_suitable_for_monetization(post)
            
            if suitability['is_suitable']:
                suitable_posts.append(post)
                self.logger.debug(f"Post accepted: {post.title[:50]}...")
            else:
                self.logger.debug(f"Post rejected: {post.title[:50]}... - {suitability['reason']}")
        
        return suitable_posts


class RedditClient:
    """Async Reddit API client for fetching and managing posts"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.reddit: Optional[asyncpraw.Reddit] = None
        self.content_filter = ContentFilter()
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
            
        if not all([self.config.api.reddit_client_id, self.config.api.reddit_client_secret]):
            self.logger.error("Reddit credentials not configured")
            return
        
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.config.api.reddit_client_id,
                client_secret=self.config.api.reddit_client_secret,
                user_agent=self.config.api.reddit_user_agent
            )
            
            # Test the connection
            try:
                user = await self.reddit.user.me()
                if user:
                    self.logger.info(f"Reddit client initialized, authenticated as: {user}")
                else:
                    self.logger.info("Reddit client initialized (read-only mode)")
            except asyncprawcore.ResponseException:
                self.logger.info("Reddit client initialized (read-only mode)")
            
            self._initialized = True
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Reddit client: {e}")
            self.reddit = None
    
    def is_connected(self) -> bool:
        """Check if Reddit client is properly initialized"""
        return self.reddit is not None and self._initialized
    
    async def fetch_posts_from_subreddit(self,
                                        subreddit_name: str,
                                        sort_method: str = 'hot',
                                        time_filter: str = 'day',
                                        limit: int = 10) -> List[RedditPost]:
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
            if sort_method == 'hot':
                submissions = subreddit.hot(limit=limit)
            elif sort_method == 'new':
                submissions = subreddit.new(limit=limit)
            elif sort_method == 'top':
                submissions = subreddit.top(time_filter=time_filter, limit=limit)
            elif sort_method == 'rising':
                submissions = subreddit.rising(limit=limit)
            else:
                self.logger.warning(f"Unknown sort method: {sort_method}, using 'hot'")
                submissions = subreddit.hot(limit=limit)
            
            async for submission in submissions:
                try:
                    post = RedditPost.from_submission(submission)
                    posts.append(post)
                except Exception as e:
                    self.logger.warning(f"Error processing submission {submission.id}: {e}")
                    continue
            
            self.logger.info(f"Fetched {len(posts)} posts from r/{subreddit_name}")
            return posts
            
        except asyncprawcore.NotFound:
            self.logger.error(f"Subreddit r/{subreddit_name} not found")
            return []
        except asyncprawcore.Forbidden:
            self.logger.error(f"Access forbidden to r/{subreddit_name}")
            return []
        except Exception as e:
            self.logger.error(f"Error fetching posts from r/{subreddit_name}: {e}")
            return []
    
    async def fetch_posts_from_multiple_subreddits(self,
                                                  subreddit_names: Optional[List[str]] = None,
                                                  posts_per_subreddit: int = 5) -> List[RedditPost]:
        """
        Fetch posts from multiple subreddits
        
        Args:
            subreddit_names: List of subreddit names (uses curated list if None)
            posts_per_subreddit: Number of posts to fetch from each subreddit
        """
        if subreddit_names is None:
            subreddit_names = self.config.content.curated_subreddits
        
        all_posts = []
        
        for subreddit_name in subreddit_names:
            posts = await self.fetch_posts_from_subreddit(
                subreddit_name,
                limit=posts_per_subreddit
            )
            all_posts.extend(posts)
        
        self.logger.info(f"Fetched {len(all_posts)} total posts from {len(subreddit_names)} subreddits")
        return all_posts
    
    async def get_filtered_video_posts(self,
                                      subreddit_names: Optional[List[str]] = None,
                                      max_posts: Optional[int] = None) -> List[RedditPost]:
        """
        Get filtered video posts suitable for processing
        
        Args:
            subreddit_names: List of subreddit names to search
            max_posts: Maximum number of posts to return
        
        Returns:
            List of suitable video posts
        """
        if max_posts is None:
            max_posts = self.config.content.max_reddit_posts_to_fetch
        
        # Fetch posts from multiple subreddits
        all_posts = await self.fetch_posts_from_multiple_subreddits(
            subreddit_names,
            posts_per_subreddit=max(5, max_posts // len(subreddit_names or self.config.content.curated_subreddits))
        )
        
        # Filter for video posts only
        video_posts = [post for post in all_posts if post.is_video and post.video_url]
        self.logger.info(f"Found {len(video_posts)} video posts out of {len(all_posts)} total posts")
        
        # Apply content filtering
        suitable_posts = self.content_filter.filter_posts(video_posts)
        self.logger.info(f"Filtered to {len(suitable_posts)} suitable posts")
        
        # Sort by score and return top posts
        suitable_posts.sort(key=lambda p: p.score, reverse=True)
        result = suitable_posts[:max_posts]
        
        self.logger.info(f"Returning top {len(result)} posts for processing")
        return result
    
    async def get_post_by_url(self, url: str) -> Optional[RedditPost]:
        """Get a specific post by Reddit URL"""
        if not self.is_connected():
            await self._initialize_client()
            if not self.is_connected():
                return None
        
        try:
            # Handle different URL formats
            if 'reddit.com' in url and '/comments/' in url:
                # This is a proper Reddit submission URL
                submission = await self.reddit.submission(url=url)
            elif url.startswith('https://v.redd.it/'):
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
        except Exception as e:
            self.logger.error(f"Error fetching post from URL {url}: {e}")
            return None
    
    async def get_post_by_url_async(self, url: str) -> Optional[RedditPost]:
        """Get a specific post by Reddit URL using async context manager"""
        async with self:
            return await self.get_post_by_url(url)
    
    async def search_posts(self,
                          query: str,
                          subreddit_names: Optional[List[str]] = None,
                          sort: str = 'relevance',
                          time_filter: str = 'week',
                          limit: int = 10) -> List[RedditPost]:
        """
        Search for posts matching a query
        
        Args:
            query: Search query
            subreddit_names: List of subreddits to search (all if None)
            sort: 'relevance', 'hot', 'top', 'new', 'comments'
            time_filter: Time filter for search
            limit: Maximum results to return
        """
        if not self.is_connected():
            await self._initialize_client()
            if not self.is_connected():
                return []
        
        try:
            if subreddit_names:
                # Search in specific subreddits
                subreddit_str = '+'.join(subreddit_names)
                subreddit = await self.reddit.subreddit(subreddit_str)
            else:
                # Search all of Reddit
                subreddit = await self.reddit.subreddit('all')
            
            submissions = subreddit.search(
                query=query,
                sort=sort,
                time_filter=time_filter,
                limit=limit
            )
            
            posts = []
            async for submission in submissions:
                try:
                    post = RedditPost.from_submission(submission)
                    posts.append(post)
                except Exception as e:
                    self.logger.warning(f"Error processing search result {submission.id}: {e}")
                    continue
            
            self.logger.info(f"Found {len(posts)} posts matching query: {query}")
            return posts
            
        except Exception as e:
            self.logger.error(f"Error searching posts: {e}")
            return []
    
async def create_reddit_client() -> RedditClient:
    """Factory function to create an async Reddit client"""
    client = RedditClient()
    await client._initialize_client()
    return client